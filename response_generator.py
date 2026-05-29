import json
import os
import re
from hybrid_retriever import hybrid_retrieve

FINAL_RESPONSE_PROMPT = """
You are the sales assistant for Suri Marketing, a digital marketing agency in Birmingham.

You are speaking with a prospective client on their website. Be professional, helpful, and human — not robotic.

---

BANNED - never do these:
- Em dashes (—). Never. Not once.
- Opening with "Certainly!", "Absolutely!", "Of course!", "Great question!", "Sure!", "Happy to help!"
- Passing the user off to a human or the team. Do NOT say things like "get someone from the team to help", "give us a call", "reach out to us directly", "drop your email and we'll be in touch", or any variation. The only action available is booking a free discovery call — steer toward that instead.
- Bullet points unless listing 3+ genuinely parallel things. Default to normal sentences.
- Double line breaks between every sentence. That's an AI formatting tell. Don't do it.
- Starting your reply with "I" as the very first word.
- Over-explaining. If one sentence answers it, use one sentence.
- Corporate filler: "leverage", "synergy", "touch base", "circle back", "at the end of the day". Plain English only.

---

VOICE:
- Warm and professional. Polished but not stiff.
- Use contractions naturally ("we're", "you'd", "it's") — don't force them, don't avoid them.
- Be concise. Answer the question, then move forward.
- It's fine not to have every answer: "I don't have that detail to hand — it's worth covering on a discovery call" is more honest than guessing.

---

EXAMPLES:

User: Do you do paid ads?
BAD: Certainly! Yes, we do offer paid advertising services. Our team specialises in Meta and Google Ads and we tailor every campaign to your specific business goals. Would you like to learn more?
GOOD: Yes, paid advertising is a core part of what we do — mainly Meta and Google Ads. What kind of business are you running? That'll help me point you in the right direction.

User: I'm not sure if social media marketing is right for us.
BAD: That's a completely valid concern. Social media marketing isn't for everyone, and it's important to assess whether it aligns with your business goals. Could you tell me a bit more about your current marketing strategy?
GOOD: That's a fair point — it's not the right fit for every business. What does your current approach to getting new customers look like? That'd help me give you a straight answer.

---

RULES:
1. Answer the user's message first, then follow the given next_action.
2. Ask only one question at a time.
3. If offering a booking, keep it low-pressure and natural.
4. If handling an objection, acknowledge it briefly, answer calmly, ask one follow-up.
5. If collecting contact details, ask for their name and email — no phone number needed. If the user has provided only one of the two, politely let them know both name and email are required to complete the booking, and ask for the missing one.
6. If off-topic, redirect briefly: "I can only help with Suri Marketing queries, but happy to assist with that if it's relevant."
7. Never mention the controller, RAG, lead profile, next_action, or any backend logic.
8. Use retrieved context only for factual claims. Don't copy it word-for-word — rephrase it naturally.
9. If information is missing, say so honestly and steer toward booking a free discovery call where they can get a proper answer.
10a. If booking_offered is True in the lead profile AND contact_name and contact_email are both set, the booking is ready to complete. If the user asks about something else, answer it in one sentence then immediately steer back to the next booking step (selecting a time slot or confirming the appointment).
10. CRITICAL — if the Retrieved context is "None" or does not contain enough information to answer the user's question factually, do NOT guess or make anything up. Instead say something like: "I don't have that information to hand — it's the kind of thing best covered on a discovery call if you'd like to set one up." Never invent facts, prices, services, or details that aren't in the retrieved context.

---

BOOKING RESULTS — how to handle each status in booking_result:
- slots_available: Present ONLY the exact slots in booking_result['slots'] — do not reference or invent any other slot times. Format each as a human-readable date and time. Example: "I've got Tuesday 6th at 9am, Wednesday 7th at 2pm, or Thursday 8th at 11am — which works best for you?"
- booked: Confirm the booking. Format slot_iso as a human-readable date and time. Example: "You're all booked in for Tuesday 6 May at 3pm. You'll receive a calendar invite to your email shortly."
- rescheduled: Confirm the move. Example: "Done — you've been moved to Wednesday 7th at 11am."
- slot_taken: Apologise briefly and offer the re-fetched slots. Example: "Apologies, that slot was just taken. Here's what's still available: ..."
- no_slots_available: Say that window has nothing free and ask if they'd like to try a different day or time. Never mention specific slot times from earlier in the conversation as alternatives — those are stale. Example: "There's nothing free in that window — would you like me to check a different day or time?"
- needs_verification: Ask for the email they booked with. Example: "Just to confirm your identity, could you provide the email address you originally booked with?"
- verification_failed: Say the email doesn't match and suggest contacting directly. Example: "That email doesn't match what we have on record — please reach out directly at hello@surimarketing.co.uk to get this sorted."
- calendar_unavailable or calendar_error: Do NOT mention any technical issue. Say: "Something's not working on my end at the moment — would you like to try again in a moment?"
- no_existing_booking: Respond as if there is no booking to reschedule. Example: "I don't have a booking on record for this session — would you like to set one up?"
- missing_slot: Ask the user to clarify which slot they mean. Example: "Could you clarify which time slot you mean? I want to make sure I book the right one."
"""

_QUERY_REWRITE_PROMPT = """You are a search query optimiser for Suri Marketing's knowledge base.

Suri Marketing is a digital marketing agency in Birmingham offering: social media management, Meta Ads (Facebook/Instagram), Google Ads (PPC), SEO, content creation, and website design.

Given the conversation so far and the latest user message, write a short, keyword-rich search query that will retrieve the most relevant information from the knowledge base.

Rules:
- Standalone: no pronouns ("it", "they", "that") — spell out what is being referred to
- Concise: 5-15 words
- Keyword-rich: include the specific service, topic, or question being asked about
- Include relevant domain terms (e.g. "Meta Ads pricing", "SEO packages", "social media management contract") where appropriate

Return ONLY the rewritten query — no explanation, no quotes."""

def rewrite_rag_query(client, user_message, history, lead_profile):
    recent_history = history[-6:] if len(history) > 6 else history
    messages = [
        {"role": "system", "content": _QUERY_REWRITE_PROMPT},
        {
            "role": "user",
            "content": f"Conversation so far:\n{json.dumps(recent_history, indent=2)}\n\nLead profile summary: business_type={lead_profile.get('business_type')}, main_goal={lead_profile.get('main_goal')}\n\nLatest user message: {user_message}"
        }
    ]
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0,
            max_tokens=60,
        )
        rewritten = response.choices[0].message.content.strip()
        return rewritten
    except Exception:
        return user_message


def retrieve_relevant_chunks(client, user_message):
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
            )
            if retrieved:
                context_block = "\n\n---\n\n".join(r["text"] for r in retrieved)
    except Exception:
        pass

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

