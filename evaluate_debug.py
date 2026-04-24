"""
Temporary debug script — runs the evaluation pipeline on 2 hardcoded questions
and prints every intermediate step verbosely.
Delete or ignore this file after verifying evaluate.py is working correctly.
"""

import os
import re
import time

import boto3
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

load_dotenv()

from config import get_retrieval_alpha, get_chunking_strategy
from hybrid_retriever import hybrid_retrieve
import bm25_encoder as bm25_enc
from evaluate import (
    OVERLAP_THRESHOLD, TOP_K, SYSTEM_PROMPT, JUDGE_SYSTEM_PROMPT,
    STOPWORDS, normalise, chunk_overlap, compute_metrics, judge_answer,
)

# ── Hardcoded test questions (first 2 rows from rag_eval_set.xlsx) ─────────────

TEST_CASES = [
    {
        "question":    "How much do your social media packages cost?",
        "answer_span": "Package A £995/month £895/month if paid upfront Package B £1195/month £1095/month if paid upfront",
        "gold_answer": "Our main social media packages start at £995/month, or £895/month if paid upfront, and the higher package is £1195/month, or £1095/month upfront. It gives you a clear done-for-you setup depending on how much content volume you want.",
    },
    {
        "question":    "What is included in Package A?",
        "answer_span": "10 Videos Month Strategy content planning content ideas hooks scripts content creation editing content scheduling captions hashtags profile optimisation complete social media management organic content only",
        "gold_answer": "Package A gives you a full organic social media setup, including strategy, content planning, ideas, hooks and scripts, content creation, editing, scheduling, captions, hashtags, profile optimisation, and complete social media management. It's built to take the whole content side off your plate.",
    },
]


def divider(title=""):
    print(f"\n{'─' * 60}  {title}")


def main():
    alpha             = get_retrieval_alpha()
    retrieval_strategy = os.getenv("RETRIEVAL_STRATEGY", "hybrid")
    chunking_strategy  = get_chunking_strategy().value

    divider("CONFIG")
    print(f"  Retrieval strategy : {retrieval_strategy} (alpha={alpha})")
    print(f"  Chunking strategy  : {chunking_strategy}")
    print(f"  Overlap threshold  : {OVERLAP_THRESHOLD}")
    print(f"  Top-K              : {TOP_K}")

    # Init clients
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    pinecone_index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

    s3 = boto3.resource(
        service_name="s3",
        region_name="eu-north-1",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )
    bm25_enc.load_from_s3(s3, os.getenv("AWS_S3_BUCKET"))

    for idx, tc in enumerate(TEST_CASES, 1):
        question    = tc["question"]
        answer_span = tc["answer_span"]
        gold_answer = tc["gold_answer"]

        print(f"\n{'=' * 70}")
        print(f"  QUESTION {idx}: {question}")
        print(f"{'=' * 70}")

        # ── Answer span ───────────────────────────────────────────────────────
        divider("ANSWER SPAN")
        print(f"  Raw        : {answer_span!r}")
        terms = normalise(answer_span)
        print(f"  Normalised terms ({len(terms)}): {sorted(terms)}")

        # ── Retrieval ─────────────────────────────────────────────────────────
        divider("RETRIEVAL")
        start  = time.time()
        chunks = hybrid_retrieve(
            query=question,
            client=client,
            pinecone_index=pinecone_index,
            top_k=TOP_K,
            alpha=alpha,
        )
        retrieval_time = time.time() - start
        print(f"  Retrieved {len(chunks)} chunks in {retrieval_time:.3f}s")

        for i, c in enumerate(chunks, 1):
            chunk_terms   = normalise(c["text"])
            overlap       = chunk_overlap(c["text"], terms)
            is_relevant   = overlap >= OVERLAP_THRESHOLD
            print(f"\n  [Chunk {i}] score={c['score']:.4f}  overlap={overlap:.3f}  relevant={is_relevant}")
            print(f"    text    : {c['text']}")

        # ── Metrics ───────────────────────────────────────────────────────────
        divider("METRICS")
        precision, recall, mrr, accuracy = compute_metrics(chunks, terms)
        print(f"  Precision@{TOP_K} : {precision:.4f}")
        print(f"  Recall@{TOP_K}    : {recall:.4f}")
        print(f"  MRR         : {mrr:.4f}")
        print(f"  Accuracy    : {accuracy}")

        # ── Response generation ───────────────────────────────────────────────
        divider("RESPONSE GENERATION")
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

        print(f"  Messages array has {len(messages)} message(s)")
        print(f"  Context block length: {len(context_block)} chars")

        gen_start         = time.time()
        response          = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
        gen_time          = time.time() - gen_start
        generated_answer  = response.choices[0].message.content.strip()
        total_time        = retrieval_time + gen_time

        print(f"\n  Generated answer ({len(generated_answer)} chars, {gen_time:.3f}s):")
        print(f"  {generated_answer}")
        print(f"\n  Gold answer:")
        print(f"  {gold_answer}")
        print(f"\n  Total response time: {total_time:.3f}s")

        # ── LLM judge ─────────────────────────────────────────────────────────
        divider("LLM JUDGE")
        raw_judge_response = client.chat.completions.create(
            model="gpt-4o-mini",
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
        raw_label = raw_judge_response.choices[0].message.content.strip()
        final_label = raw_label if raw_label in {"Correct", "Partially Correct", "Incorrect"} else "Incorrect"
        print(f"  Judge raw response : {raw_label!r}")
        print(f"  Final label        : {final_label}")
        print(f"  Char count         : {len(generated_answer)}")

    print(f"\n{'=' * 70}")
    print("  DEBUG RUN COMPLETE")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
