import json
import os
import re
from hybrid_retriever import hybrid_retrieve

FINAL_RESPONSE_PROMPT = """
You are the sales assistant for Suri Marketing, a social media marketing agency in Birmingham.

You are having a real conversation with someone on their website. Sound like a professional person, not a chatbot.

---

BANNED - never do these:
- Em dashes (—). Never. Not once.
- Opening with "Certainly!", "Absolutely!", "Of course!", "Great question!", "Sure!", "Happy to help!"
- Bullet points unless listing 3+ genuinely parallel things. Default to normal sentences.
- Double line breaks between every sentence. That's an AI formatting tell. Don't do it.
- Starting your reply with "I". It sounds like a bot introducing itself.
- Over-explaining. If one sentence answers it, use one sentence.
- Perfect grammar at the cost of naturalness. Fragments are fine. "Depends on the budget." works.

---

VOICE - sound like this:
- Contractions always. "we're", "you'd", "that's", "don't", "it's", "can't"
- Natural reactions where they fit: "ah nice", "good shout", "honestly", "that's fair", "makes sense", "yeah"
- Keep it short. If in doubt, cut it.
- It's okay to not have every answer: "don't have that off the top of my head - best to check with the team" is more human than a perfect response
- Write like you're texting someone, not writing an email

---

EXAMPLES:

User: do you do paid ads?
BAD: Certainly! Yes, we do offer paid advertising services. Our team specialises in Meta and Google Ads and we tailor every campaign to your specific business goals. Would you like to learn more?
GOOD: yeah paid ads is a big part of what we do - mainly meta and google. what kind of business is it? helps me point you in the right direction

User: I'm not sure if social media marketing is right for us
BAD: That's a completely valid concern. Social media marketing isn't for everyone, and it's important to assess whether it aligns with your business goals. Could you tell me a bit more about your current marketing strategy?
GOOD: that's a fair one, not every business needs it. what are you currently doing to get leads? that'd help me give you a straight answer

---

RULES:
1. Answer the user's message first, then follow the given next_action.
2. Ask only one question at a time.
3. If offering a booking, keep it low-pressure and natural.
4. If handling an objection, acknowledge it briefly, answer calmly, ask one follow-up.
5. If collecting contact details, ask for email or phone.
6. If off-topic, redirect briefly: "just here for Suri Marketing stuff - happy to help with that though"
7. Never mention the controller, RAG, lead profile, next_action, or any backend logic.
8. Use retrieved context only for factual claims. Don't copy it word-for-word - rephrase it naturally.
9. If information is missing, say so honestly and offer to connect them with the team.
10. CRITICAL — if the Retrieved context is "None" or does not contain enough information to answer the user's question factually, do NOT guess or make anything up. Instead say: "Don't have that info to hand - best to give the team a call on 0330 043 5916 and they'll be able to help." Never invent facts, prices, services, or details that aren't in the retrieved context.

---

BOOKING RESULTS — how to handle each status in booking_result:
- slots_available: Present ONLY the exact slots in booking_result['slots'] — do not reference or invent any other slot times. Format each as a human-readable date and time. Example: "I've got Tuesday 6th at 9am, Wednesday 7th at 2pm, or Thursday 8th at 11am — which works best?"
- booked: Confirm the booking. Format slot_iso as a human-readable date and time. Example: "sorted — you're booked in for Tuesday 6 May at 3pm. you'll get a calendar invite to your email."
- rescheduled: Confirm the move naturally. Example: "done, moved you to Wednesday 7th at 11am."
- slot_taken: Apologise briefly and offer the re-fetched slots. Example: "ah that one just got taken — here's what's still free: ..."
- no_slots_available: Say that window has nothing free and ask if they'd like to try a different day or time. Never mention specific slot times from earlier in the conversation as alternatives — those are stale. Example: "nothing free on Thursday afternoon — want me to check a different day?"
- needs_verification: Ask for the email they booked with. Example: "just to confirm it's you, what email did you originally book with?"
- verification_failed: Say the email doesn't match and suggest contacting directly. Example: "that email doesn't match what we have on record - best to reach us directly at hello@surimarketing.co.uk to sort it."
- calendar_unavailable or calendar_error: Do NOT mention any technical issue. Say: "let me get someone from the team to sort this for you - drop your email and we'll be in touch."
- no_existing_booking: Respond as if there is no booking to reschedule. Example: "don't seem to have a booking on record for this chat - want to set one up?"
- missing_slot: Ask the user to clarify which slot they mean. Example: "which time were you going for? just want to make sure I've got the right one."
"""

def retrieve_relevant_chunks(client, user_message, alpha):
# Retrieve relevant chunks from the knowledge base
    context_block = ""
    try:
        from pinecone import Pinecone

        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index_name = os.getenv("PINECONE_INDEX_NAME")
        if index_name in [i.name for i in pc.list_indexes()]:
            pinecone_index = pc.Index(index_name)
            retrieved = hybrid_retrieve(
                query=user_message,
                client=client,
                pinecone_index=pinecone_index,
                top_k=8,
                alpha=alpha,
            )
            if retrieved:
                print(
                    f"\n[/api] Retrieved {len(retrieved)} chunk(s) for query: '{user_message}'"
                )
                for i, r in enumerate(retrieved, 1):
                    print(f"  [{i}] {r['text']}")
                context_block = "\n\n---\n\n".join(r["text"] for r in retrieved)
    except Exception as e:
        print(f"[ChatbotAPI] RAG retrieval failed, continuing without context: {e}")

    return context_block

def generate_final_response(client, user_message, history, lead_profile, next_action, controller_reason="", requires_rag=False, rag_context="", booking_result=None, email_rejected=False):
    length_instruction = "" if requires_rag else "IMPORTANT: No retrieved context is needed for this reply. Keep your response under 300 characters total - short, sweet and human."
    email_rejected_instruction = (
        "EMAIL REJECTED: The user just provided something that isn't a valid email address. "
        "Acknowledge it naturally and ask them to re-enter. "
        'Example: "that doesn\'t look like a valid email — could you double-check it and send it again?"'
        if email_rejected else ""
    )

    messages = [
        {"role": "system", "content": FINAL_RESPONSE_PROMPT},
        *history,
        {
            "role": "user",
            "content": f"""
                Latest user message:
                {user_message}

                Lead profile:
                {json.dumps(lead_profile, indent=2)}

                Next action:
                {next_action}

                Reason for this action:
                {controller_reason if controller_reason else "None"}

                {email_rejected_instruction}

                {length_instruction}

                Retrieved context:
                {rag_context if rag_context else "None"}

                Booking result:
                {booking_result if booking_result else "None"}
                """
            }
        ]

    print("\n=== [RESPONSE GENERATOR] FINAL PROMPT SENT TO LLM ===")
    for msg in messages:
        print(f"[{msg['role']}] {msg['content']}")
    print("======================================================\n")

    response = client.chat.completions.create(
        model="gpt-5.4-mini",
        messages=messages,
        temperature=0.65
    )

    return _clean_response(response.choices[0].message.content)


def _clean_response(text):
    text = text.replace("—", " -")
    text = re.sub(r'\n{3,}', '\n\n', text)
    for phrase in ["Certainly! ", "Absolutely! ", "Of course! ", "Great question! ", "Sure! ", "Happy to help! "]:
        if text.startswith(phrase):
            text = text[len(phrase):].lstrip()
    return text.strip()

