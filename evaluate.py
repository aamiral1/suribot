import os
import re
import csv
import time
from datetime import datetime

import boto3
import openpyxl
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

load_dotenv()

from config import get_retrieval_alpha, get_chunking_strategy
from hybrid_retriever import hybrid_retrieve
import bm25_encoder as bm25_enc

# ── Config ────────────────────────────────────────────────────────────────────

OVERLAP_THRESHOLD = 0.3
TOP_K = 8

EVAL_DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "evaluation", "rag_eval_set.xlsx")
RESULTS_DIR       = os.path.join(os.path.dirname(__file__), "..", "evaluation")

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "and", "or", "of",
    "to", "in", "for", "on", "with", "at", "by", "from",
}

# Copied from app.py ChatbotAPI.post — must stay in sync if the prompt changes
SYSTEM_PROMPT = """
        You are the Customer Support and Sales Assistant for Suri Marketing - a social media marketing agency. You sound like a warm, switched-on member of the team — friendly, casual, confident. Never robotic, never salesy.

        ## Scope
        - You only help with questions about Suri Marketing: our services, packages, pricing, process, how we work with clients.
        - If someone asks for something off-topic (essays, translation, code, general advice), politely redirect: "I'm just here for Suri Marketing stuff — happy to tell you what we do if that's useful."
        - Never mention being an AI, a model, or having a "knowledge base." You're just the Suri assistant.

        ## Primary Objective

        Your main goal is to naturally guide conversations toward booking a discovery call.

        - Do NOT push aggressively
        - Do NOT jump to booking immediately
        - First: answer → build understanding → create interest
        - Then: suggest a discovery call as the next step when appropriate

        ## How to answer
        - Keep answers short and sweet like a human sales rep would do.
        - Keep responses to a maximum of ~3 sentences unless more detail is clearly needed.
        - Every factual claim (prices, packages, services, policies) must come from info given to you in this conversation. If you don't have it, say so naturally: "Don't have that one to hand — best to check with someone on the team."
        - Don't invent facts. But don't paste things verbatim either — rephrase into how someone would actually say it out loud.
        - Lead with what the person cares about, then the details.
        - Do not ask multiple questions in one reply.
        - Only ask a question if it clearly moves the conversation forward.
        - If the user is just asking for information, prioritise answering over qualifying.
        - Never guess or assume details about Suri Marketing's services, pricing, or results.
        - Only use information provided in the conversation or system context.
        - If the user shows repeated interest or asks multiple detailed questions, move more directly toward suggesting a discovery call.
        - If the user hesitates (e.g. "not sure", "seems expensive"), acknowledge it briefly and respond calmly without pressure.

        ## Voice
        - Use contractions. "We're," "you're," "don't," "can't." Always.
        - Short sentences. Fragments are fine.
        - Skip marketing-speak: no "end-to-end solutions," "leverage," "holistic," etc.
        - Don't end every reply with a question. Only ask one when it genuinely moves the conversation forward.
        - Bullets only for 3+ parallel items. Default to prose.
        - If someone's rude, stay polite and steer back: "Here to help with Suri Marketing — let me know if there's something I can sort for you."

        ## Lead Qualification & Conversion Behaviour
        - If the user shows interest, ask simple questions to understand:
        - Their business
        - Their goals
        - Their current situation

        - Use this to:
        - Personalise responses
        - Make the service feel relevant

        - When the user is:
        - Curious → educate
        - Engaged → qualify
        - Interested → guide to booking

        ---

        ### When to Suggest a Discovery Call

        Suggest a call when:
        - The user asks about pricing or services in detail
        - The user asks "will this work for me?"
        - The user shows clear interest
        - The conversation reaches a natural pause

        ---

        ### How to Suggest It

        Keep it low-pressure and natural:

        Examples:
        - "Might be easier to walk through this on a quick call — want me to set that up?"
        - "We could go through this properly on a quick discovery call if you want"
        - "Happy to show you exactly how this would work for your business on a call"

        - You are representing a real company, so responses should feel grounded, confident, and aligned with a real service offering — not hypothetical

        ## Examples of good style

        User: What services do you offer?
        You: We do social media content across Instagram, TikTok, and Facebook — strategy, filming guidance, editing, captions, scheduling, the lot. Two packages depending on how much content you want each month (10 or 14 videos), both run on a 4-month minimum. Want the pricing?

        User: How much does it cost?
        You: £995/month for 10 videos, £1195/month for 14. Both cover everything — strategy, editing, captions, posting. Pay upfront and it drops £100/month either way.

        User: What's the difference between Package A and B?
        You: Honestly just volume. A is 10 videos every 4 weeks, B is 14. Same strategy, same platforms, same everything else. B makes sense if you want to post most days; A's plenty if you're building steadily.

        User: Why is the minimum 4 months?
        You: Social takes a bit to get going. First month or two is mostly building content and letting the algorithm work out who to show you to. Four months gives us room to actually move the needle — most people see things click around month two or three.

        User: Do you do Google Ads?
        You: Nah, we don't do paid ads — we're organic-only across Instagram, TikTok, and Facebook. If content on those is what you're after though, happy to tell you more.

        User: Can you write me a caption for my post about coffee?
        You: Caption-writing's part of what we do for clients, not really a one-off thing. Want to hear how the content side works?

        User: What's your refund policy?
        You: Don't have that one to hand — probably easier to run through it on a quick call if you want.
        """

JUDGE_SYSTEM_PROMPT = """You are a strict answer quality evaluator.

- "Correct": The generated answer is relevant to the question AND covers the majority of key points from the gold answer with no significant contradictions.
- "Partially Correct": The generated answer is relevant to the question AND covers at least one key point from the gold answer, but is missing important information.
- "Incorrect": The generated answer fails to address the question, contains no key points from the gold answer, contradicts it, or is off-topic.

Reply with ONLY one of: Correct, Partially Correct, Incorrect"""

# ── Overlap helpers ────────────────────────────────────────────────────────────

def normalise(text: str) -> set:
    text = text.lower()
    text = re.sub(r'[^\w\s£]', ' ', text)
    return {w for w in text.split() if w not in STOPWORDS}


def chunk_overlap(chunk_text: str, span_terms_set: set) -> float:
    if not span_terms_set:
        return 0.0
    return len(normalise(chunk_text) & span_terms_set) / len(span_terms_set)


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(chunks: list, span_terms_set: set) -> tuple:
    relevance = [chunk_overlap(c["text"], span_terms_set) >= OVERLAP_THRESHOLD for c in chunks]

    precision = sum(relevance) / len(chunks) if chunks else 0.0

    all_retrieved_terms = set()
    for c in chunks:
        all_retrieved_terms |= normalise(c["text"])
    recall = len(all_retrieved_terms & span_terms_set) / len(span_terms_set) if span_terms_set else 0.0

    mrr = 0.0
    for rank, rel in enumerate(relevance, 1):
        if rel:
            mrr = 1.0 / rank
            break

    accuracy = 1 if mrr > 0 else 0

    return precision, recall, mrr, accuracy


# ── LLM judge ─────────────────────────────────────────────────────────────────

def judge_answer(client: OpenAI, question: str, gold_answer: str, generated_answer: str) -> str:
    resp = client.chat.completions.create(
        model="gpt-5.4",
        temperature=0,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Question: {question}\n\n"
                f"Gold Answer: {gold_answer}\n\n"
                f"Generated Answer: {generated_answer}"
            )},
        ],
    )
    label = resp.choices[0].message.content.strip()
    return label if label in {"Correct", "Partially Correct", "Incorrect"} else "Incorrect"


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    alpha              = get_retrieval_alpha()
    retrieval_strategy = os.getenv("RETRIEVAL_STRATEGY", "hybrid").strip().lower()
    chunking_strategy  = get_chunking_strategy().value

    print(f"\n[eval] Starting evaluation")
    print(f"[eval] Retrieval: {retrieval_strategy} (alpha={alpha})")
    print(f"[eval] Chunking:  {chunking_strategy}")

    # Init clients
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    pc            = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name    = os.getenv("PINECONE_INDEX_NAME")
    pinecone_index = pc.Index(index_name)

    s3 = boto3.resource(
        service_name="s3",
        region_name="eu-north-1",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )
    bucket_name = os.getenv("AWS_S3_BUCKET")
    bm25_enc.load_from_s3(s3, bucket_name)

    # Load gold standard
    wb   = openpyxl.load_workbook(EVAL_DATASET_PATH)
    ws   = wb.active
    rows = [r for r in ws.iter_rows(min_row=2, values_only=True) if r[0]]

    print(f"[eval] Loaded {len(rows)} questions from dataset\n")

    # Prepare CSV
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(RESULTS_DIR, f"results_{retrieval_strategy}_{chunking_strategy}_{timestamp}.csv")

    fieldnames = [
        "question_id", "question", "retrieval_strategy", "chunking_strategy",
        "precision", "recall", "mrr", "accuracy",
        "response_time_seconds", "response_character_count",
        "generated_answer", "gold_answer", "llm_judge_label",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, row in enumerate(rows, 1):
            question, answer_span, gold_answer = row[0], row[1] or "", row[2] or ""

            print(f"[Q{i:02d}] {question}")

            terms = normalise(answer_span)

            # ── Timed: retrieve + generate ────────────────────────────────────
            start  = time.time()

            chunks = hybrid_retrieve(
                query=question,
                client=client,
                pinecone_index=pinecone_index,
                top_k=TOP_K,
                alpha=alpha,
            )

            context_block = "\n\n---\n\n".join(c["text"] for c in chunks)

            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            if context_block:
                messages.append({
                    "role": "system",
                    "content": (
                        "Relevant information for the user's current question "
                        "(use this as your source of truth, but rephrase naturally — "
                        "do not paste it verbatim):\n\n"
                        f"{context_block}"
                    ),
                })
            messages.append({"role": "user", "content": question})

            response          = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
            elapsed           = time.time() - start
            generated_answer  = response.choices[0].message.content.strip()
            # ─────────────────────────────────────────────────────────────────

            precision, recall, mrr, accuracy = compute_metrics(chunks, terms)
            label = judge_answer(client, question, gold_answer, generated_answer)

            writer.writerow({
                "question_id":             i,
                "question":                question,
                "retrieval_strategy":      retrieval_strategy,
                "chunking_strategy":       chunking_strategy,
                "precision":               round(precision, 4),
                "recall":                  round(recall, 4),
                "mrr":                     round(mrr, 4),
                "accuracy":                accuracy,
                "response_time_seconds":   round(elapsed, 3),
                "response_character_count": len(generated_answer),
                "generated_answer":        generated_answer,
                "gold_answer":             gold_answer,
                "llm_judge_label":         label,
            })
            f.flush()

            print(f"       precision={precision:.3f}  recall={recall:.3f}  mrr={mrr:.3f}  accuracy={accuracy}  time={elapsed:.2f}s  label={label}")
            print(f"       Generated : {generated_answer}")
            print(f"       Gold      : {gold_answer}\n")

    print(f"[eval] Done. Results saved to:\n       {output_path}\n")


if __name__ == "__main__":
    main()
