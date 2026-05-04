import json
import re
from datetime import datetime
from zoneinfo import ZoneInfo

_EMAIL_RE = re.compile(r'^[^@\s]+@[^@\s]+\.[^@\s]+$')

class SalesController:
    # Actions that the Sales Controller is allowed to choose from
    VALID_ACTIONS = {
        "ANSWER_WITH_RAG",
        "ANSWER_FROM_CONTEXT",
        "ASK_CLARIFYING_QUESTION",
        "ASK_SITUATION_QUESTION",
        "ASK_PROBLEM_QUESTION",
        "EXPLAIN_IMPLICATION",
        "MATCH_SOLUTION",
        "HANDLE_OBJECTION",
        "HANDLE_COMPARISON",
        "HANDLE_REJECTION",
        "OFFER_AND_COLLECT",
        "OFFER_AVAILABLE_SLOTS",
        "BOOK_APPOINTMENT",
        "CONFIRM_BOOKING",
        "RESCHEDULE_APPOINTMENT",
        "REDIRECT_OFF_TOPIC",
        "END_CONVERSATION_POLITELY",
        "CONTINUE_HELPFULLY",
        "ACKNOWLEDGE_AND_HOLD",
        "FOLLOW_UP_PREVIOUS_QUESTION",
    }

    LEAD_PROFILE_SCHEMA = {
        "business_type": None,
        "location": None,
        "current_marketing": None,
        "main_goal": None,
        "main_problem": None,
        "pain_confirmed": False,
        "solution_matched": False,
        "objection": None,
        "booking_offered": False,
        "booking_completed": False,
        "contact_name": None,
        "contact_email": None,
        "contact_phone": None,
        "requested_day": None,
        "requested_time_from": None,
        "requested_time_to": None,
    }

    RAG_ACTIONS = {
        "ANSWER_WITH_RAG",
        "MATCH_SOLUTION",
        "HANDLE_OBJECTION",
        "HANDLE_COMPARISON",
    }

    BOOKING_ACTIONS = {
        "OFFER_AVAILABLE_SLOTS",
        "BOOK_APPOINTMENT",
        "CONFIRM_BOOKING",
        "RESCHEDULE_APPOINTMENT",
    }

    CONTROLLER_PROMPT = """
        You are the internal controller for Suri Marketing's website chatbot.

        Your job is to choose the next best assistant action.
        You do NOT write the final customer-facing reply.

        The chatbot goal is:
        1. Answer questions accurately.
        2. Qualify leads.
        3. Handle objections.
        4. Guide suitable users toward booking a discovery call.

        Use a simplified SPIN sales strategy:
        - Situation: understand the user's business/context.
        - Problem: identify their current marketing problem or goal.
        - Implication: explain why the problem matters.
        - Need-payoff: connect Suri's service to the desired result.
        - Booking: offer a call when appropriate.

        Valid actions (with meanings):
        - ANSWER_WITH_RAG: Use retrieved company knowledge to answer factual questions about services, pricing, process, or policies.
        - ANSWER_FROM_CONTEXT: Answer using only conversation history or the lead profile without external knowledge.
        - ASK_CLARIFYING_QUESTION: Ask a follow-up when the user’s message is unclear or ambiguous.
        - ASK_SITUATION_QUESTION: Ask about the user’s business, industry, location, or current setup.
        - ASK_PROBLEM_QUESTION: Ask about the user’s goal, challenge, or what they want to improve.
        - EXPLAIN_IMPLICATION: Explain why the user’s problem matters or what they may lose by not addressing it.
        - MATCH_SOLUTION: Connect Suri’s services directly to the user’s problem or goal.
        - HANDLE_OBJECTION: Respond to concerns about price, trust, contracts, refunds, or results.
        - HANDLE_COMPARISON: Explain how Suri differs from alternatives like freelancers or other agencies.
        - HANDLE_REJECTION: Respond calmly to hesitation or rejection such as “no” or “not now”.
        - OFFER_AND_COLLECT: Suggest a discovery call AND ask for the user's name and email in the same response. Do not first ask if they want to book — go straight to collecting their details naturally. Use this whenever it is appropriate to offer a booking.
        - OFFER_AVAILABLE_SLOTS: Use the booking tool to fetch and present available time slots. Use this after contact details are collected, or when the user asks about available times.
        - BOOK_APPOINTMENT: Use the booking tool to create the calendar event. Only when contact details and a specific time slot are both known.
        - CONFIRM_BOOKING: Confirm that a booking has already been made in this session. Only chosen after BOOK_APPOINTMENT succeeded in a previous turn.
        - RESCHEDULE_APPOINTMENT: Use the booking tool to move an existing booking to a new time. Only when the user has selected a new slot AND has provided an email for identity verification.
        - REDIRECT_OFF_TOPIC: Politely steer the conversation back if the user asks something unrelated.
        - END_CONVERSATION_POLITELY: Close the conversation in a friendly and respectful way.
        - CONTINUE_HELPFULLY: Continue the conversation naturally without changing direction.
        - ACKNOWLEDGE_AND_HOLD: Acknowledge simple replies like “ok” or “thanks” without progressing the flow.
        - FOLLOW_UP_PREVIOUS_QUESTION: Continue from the last question when the user gives a short or vague reply.

        Available tools:
        - CALL_RAG: Retrieves relevant company information from the knowledge base based on the user’s query.
        - BOOK_APPOINTMENT: Returns a list of available future booking slots in the format {"date", "time"}.

        Lead profile fields you can update:
        - business_type
        - location
        - current_marketing
        - main_goal
        - main_problem
        - pain_confirmed
        - solution_matched
        - objection
        - booking_offered
        - booking_completed
        - contact_name
        - contact_email
        - contact_phone

        Rules:
        1. If the user asks about pricing, packages, services, process, contracts, refunds, guarantees, results, or comparisons, set requires_rag = true.
        2. When it is appropriate to offer a booking, choose OFFER_AND_COLLECT — ask for name and email in the same response. Do not ask "would you like to book?" first.
        3. If the user provides contact details after OFFER_AND_COLLECT:
            - Update contact_name, contact_email, and/or contact_phone in lead_profile_updates.
            - Only choose OFFER_AVAILABLE_SLOTS or BOOK_APPOINTMENT once BOTH contact_name AND contact_email are known (either from this message or already in the lead profile). If either is still missing, stay on OFFER_AND_COLLECT and ask for the missing field.
        4. If the user directly asks for a call, or signals that they want a call - Directly OFFER_AND_COLLECT.
        4. If the user directly asks to book at a specific time (e.g. "book me for Tuesday 3pm") and contact details are known, choose BOOK_APPOINTMENT directly — no need for OFFER_AND_COLLECT first.
        5. Do not choose BOOK_APPOINTMENT unless ALL THREE are known: (a) contact_name, (b) contact_email, AND (c) selected_slot.
        6. If the user raises a concern, choose HANDLE_OBJECTION.
        7. If the user compares Suri with alternatives, choose HANDLE_COMPARISON.
        8. If the user rejects or hesitates after OFFER_AND_COLLECT or at any point, choose HANDLE_REJECTION.
        9. If the user asks unrelated things, choose REDIRECT_OFF_TOPIC.
        10. If unclear and not sales-related, choose ASK_CLARIFYING_QUESTION.
        11. Do not ask endless questions.
        12. If business_type and either main_goal or main_problem are known, the user is qualified enough to offer a booking.
        13. Only update lead profile fields when clearly supported by the conversation.
        14. Do not invent values.
        15. Return only valid JSON.
        16. When the user selects a specific slot, set selected_slot_day to the day name exactly as it appears in the Upcoming days table (e.g. "thursday") and selected_slot_time to HH:MM in 24-hour format (e.g. "09:00", "13:00"). Do NOT produce ISO datetimes.
        17. BOOK_APPOINTMENT and RESCHEDULE_APPOINTMENT MUST populate both selected_slot_day AND selected_slot_time. If either is unknown, do not choose those actions.
        18. When the user expresses a day preference (e.g. "Thursday", "next Monday"), set requested_day to that day name in lowercase (e.g. "thursday", "next monday"). When they express a time-of-day preference, set requested_time_from to HH:MM (e.g. "12:00") and requested_time_to to HH:MM or null to mean end of business (17:00).
        19. When resolving a day name to use in requested_day or selected_slot_day, copy it exactly from the "Upcoming days" table — do NOT do arithmetic. For nearest occurrence use the first match; for "next X" use the second match in the table.
        20. When the user refines a time within an already-stated day (e.g. "after 12?", "morning only"), update requested_time_from to the stated HH:MM. Do not change requested_day if it is already set.
        21. If the user says they want to reschedule without naming a new time, choose OFFER_AVAILABLE_SLOTS to show options first.
        22. CONFIRM_BOOKING is only chosen after BOOK_APPOINTMENT has already succeeded in the current session (booking_completed is True in the lead profile).
        23. requested_day, requested_time_from, requested_time_to, selected_slot_day, selected_slot_time, and provided_email all default to null when not applicable.
        24. When the user states an email address in the context of rescheduling, populate provided_email with that email.
        25. When choosing RESCHEDULE_APPOINTMENT, always re-extract selected_slot_day and selected_slot_time from the most recent slot selection in conversation history, even if the user's current message is just providing their email.

        Return JSON exactly like this:
        {
        "next_action": "...",
        "requires_rag": true,
        "requires_booking_tool": false,
        "confidence": 0.0,
        "lead_profile_updates": {
            "business_type": null,
            "location": null,
            "current_marketing": null,
            "main_goal": null,
            "main_problem": null,
            "pain_confirmed": null,
            "solution_matched": null,
            "objection": null,
            "booking_offered": null,
            "booking_completed": null,
            "contact_name": null,
            "contact_email": null,
            "contact_phone": null
        },
        "reason": "brief internal reason",
        "requested_day": null,
        "requested_time_from": null,
        "requested_time_to": null,
        "selected_slot_day": null,
        "selected_slot_time": null,
        "provided_email": null
        }
    """

        
    def __init__(self, client, model="gpt-4o-mini"):
        self.client = client
        self.model = model

    @classmethod
    def create_new_profile(cls):
        return cls.LEAD_PROFILE_SCHEMA.copy()
    
    # enforce lead profile schema
    def normalise_profile(self, profile):
        if not isinstance(profile, dict):
            profile = {}

        # only allow schema keys and fill missing defaults
        return {
            key: profile.get(key, default)
            for key, default in self.LEAD_PROFILE_SCHEMA.items()
        }
    
    # register lead profile updates
    def merge_profile(self, old_profile, updates):
        profile = self.normalise_profile(old_profile)

        if not isinstance(updates, dict):
            return profile

        for key, value in updates.items():
            if key not in self.LEAD_PROFILE_SCHEMA:
                continue

            # do not overwrite known values with null
            if value is None:
                continue

            # reject malformed emails
            if key == "contact_email" and not _EMAIL_RE.match(str(value)):
                continue

            profile[key] = value

        return self.normalise_profile(profile)
    
    def run_controller(self, user_message, history, lead_profile, last_actions, sales_turn=0, current_datetime=None, current_date_label=None, day_map=None):
        lead_profile = self.normalise_profile(lead_profile)

        if current_datetime is None:
            now = datetime.now(ZoneInfo("Europe/London"))
            current_datetime = now.isoformat()
            current_date_label = now.strftime("%A %-d %B %Y")

        messages = [
            {
                "role": "system",
                "content": self.CONTROLLER_PROMPT
            },
            {
                "role": "user",
                "content": f"""
                    Current datetime (Europe/London): {current_datetime}
                    Today is: {current_date_label}

                    Upcoming days — use these exact dates when resolving day names, do not calculate:
{day_map}

                    Conversation history:
                    {json.dumps(history, indent=2)}

                    Lead profile:
                    {json.dumps(lead_profile, indent=2)}

                    Last assistant actions:
                    {json.dumps(last_actions, indent=2)}

                    Sales turn count: {sales_turn}

                    Latest user message:
                    {user_message}
                """
            }
        ]
        print(f"\n=== [CONTROLLER] SALES TURN: {sales_turn} ===")
        print(messages[1]["content"])
        print("==============================================\n")

        response = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=0, response_format={"type":"json_object"}
        )

        output = json.loads(response.choices[0].message.content)

        print("\n=== [CONTROLLER] LLM OUTPUT ===")
        print(json.dumps(output, indent=2))
        print("================================\n")

        return output

    def apply_guardrails(self, controller_output, lead_profile, sales_turn, threshold=3):
        # print("\n=== [GUARDRAILS] INPUT ===")
        # print(json.dumps(controller_output, indent=2))
        # print("==========================\n")
        # --- enforce output schema ---
        if not isinstance(controller_output, dict):
            controller_output = {}

        if not isinstance(controller_output.get("next_action"), str):
            controller_output["next_action"] = "ASK_CLARIFYING_QUESTION"

        if not isinstance(controller_output.get("requires_rag"), bool):
            controller_output["requires_rag"] = False

        if not isinstance(controller_output.get("requires_booking_tool"), bool):
            controller_output["requires_booking_tool"] = False

        raw_confidence = controller_output.get("confidence")
        if not isinstance(raw_confidence, (int, float)) or not (0.0 <= raw_confidence <= 1.0):
            controller_output["confidence"] = 1.0

        if not isinstance(controller_output.get("lead_profile_updates"), dict):
            controller_output["lead_profile_updates"] = {}

        if not isinstance(controller_output.get("reason"), str):
            controller_output["reason"] = ""

        for field in ("requested_day", "requested_time_from", "requested_time_to",
                      "selected_slot_day", "selected_slot_time", "provided_email"):
            if not isinstance(controller_output.get(field), (str, type(None))):
                controller_output[field] = None
        # --- end schema enforcement ---

        # enforce valid action — fallback if LLM returns something unexpected
        if controller_output.get("next_action") not in self.VALID_ACTIONS:
            controller_output["next_action"] = "ASK_CLARIFYING_QUESTION"
            controller_output["reason"] = f"Invalid action overridden by guardrail"

        # sales turn threshold — force booking offer regardless of controller decision
        if sales_turn >= threshold and not lead_profile.get("booking_offered"):
            controller_output["next_action"] = "OFFER_AND_COLLECT"
            controller_output["reason"] = f"Sales turn threshold ({threshold}) reached — overriding to offer booking"
            controller_output["lead_profile_updates"]["booking_offered"] = True
            controller_output["requires_rag"] = False
            controller_output["requires_booking_tool"] = False
            return controller_output

        # # low confidence fallback
        # confidence = controller_output.get("confidence", 1.0)
        # if confidence < 0.7:
        #     if self._is_leadish(lead_profile):
        #         controller_output["next_action"] = "OFFER_BOOKING"
        #         controller_output["reason"] = "Low confidence + qualified lead — overriding to offer booking"
        #     else:
        #         controller_output["next_action"] = "ASK_CLARIFYING_QUESTION"
        #         controller_output["reason"] = "Low confidence + unqualified lead — asking clarifying question"

        # block slot/booking actions if contact details are incomplete
        updates = controller_output.get("lead_profile_updates", {})
        has_name = lead_profile.get("contact_name") or updates.get("contact_name")

        attempted_email = updates.get("contact_email")
        if lead_profile.get("contact_email"):
            has_email = True
            email_rejected = False
        elif attempted_email and _EMAIL_RE.match(str(attempted_email)):
            has_email = True
            email_rejected = False
        elif attempted_email:
            has_email = False
            email_rejected = True
        else:
            has_email = False
            email_rejected = False

        controller_output["email_rejected"] = email_rejected

        if controller_output["next_action"] in {"OFFER_AVAILABLE_SLOTS", "BOOK_APPOINTMENT"} and not (has_name and has_email):
            controller_output["next_action"] = "OFFER_AND_COLLECT"
            controller_output["reason"] = "Missing name or email — collecting both before proceeding to booking"
            controller_output["requires_booking_tool"] = False
            controller_output["requires_rag"] = False
            controller_output["lead_profile_updates"]["booking_offered"] = True
            return controller_output

        # ensure booking actions have a resolvable slot
        if controller_output["next_action"] in {"BOOK_APPOINTMENT", "RESCHEDULE_APPOINTMENT"}:
            if not (controller_output.get("selected_slot_day") and controller_output.get("selected_slot_time")):
                controller_output["next_action"] = "OFFER_AVAILABLE_SLOTS"
                controller_output["reason"] = "Missing selected_slot_day or selected_slot_time — showing available slots"

        controller_output["requires_rag"] = (
            controller_output["next_action"] in self.RAG_ACTIONS
            or bool(controller_output.get("requires_rag"))
        )

        controller_output["requires_booking_tool"] = (
            controller_output["next_action"] in self.BOOKING_ACTIONS
            or bool(controller_output.get("requires_booking_tool"))
        )

        # auto-flip profile flags the controller might forget
        action = controller_output["next_action"]
        if action == "OFFER_AND_COLLECT":
            controller_output["lead_profile_updates"]["booking_offered"] = True
        if action == "CONFIRM_BOOKING":
            controller_output["lead_profile_updates"]["booking_completed"] = True

        print("\n=== [GUARDRAILS] OUTPUT ===")
        print(json.dumps(controller_output, indent=2))
        print("===========================\n")

        return controller_output


    def _is_leadish(self, profile):
        profile = self.normalise_profile(profile)

        return any([
            profile.get("business_type"),
            profile.get("main_goal"),
            profile.get("main_problem"),
            profile.get("current_marketing"),
        ])