import json
import re

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
        "OFFER_BOOKING",
        "COLLECT_CONTACT",
        "OFFER_AVAILABLE_SLOTS",
        "BOOK_APPOINTMENT",
        "CONFIRM_BOOKING",
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
        "contact_email": None,
        "contact_phone": None,
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
        - OFFER_BOOKING: Suggest a discovery call as the next step.
        - COLLECT_CONTACT: Ask for missing contact details such as email or phone number.
        - OFFER_AVAILABLE_SLOTS: Provide booking options such as times or a booking link.
        - BOOK_APPOINTMENT: Use the booking tool when enough information is available to create a booking.
        - CONFIRM_BOOKING: Confirm that a booking has been made or details have been shared.
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
        - contact_email
        - contact_phone

        Rules:
        1. If the user asks about pricing, packages, services, process, contracts, refunds, guarantees, results, or comparisons, set requires_rag = true.
        2. If the user explicitly asks to speak with the team, book a call, arrange a meeting, or have someone contact them:
            - choose COLLECT_CONTACT if email/phone is missing.
            - choose OFFER_AVAILABLE_SLOTS if contact details are already known but no time/slot has been chosen.
            - choose BOOK_APPOINTMENT only if contact details and a preferred time/slot are both available.
        3. If the user provides contact details:
            - update contact_email and/or contact_phone.
            - choose OFFER_AVAILABLE_SLOTS if no preferred time/slot is known.
            - choose BOOK_APPOINTMENT only if a preferred time/slot is also known.
        4. Do not choose BOOK_APPOINTMENT unless the system has enough information to create a booking.
        Minimum booking information:
            - contact_email or contact_phone
            - preferred time/slot or selected booking option
        5. If the user raises a concern, choose HANDLE_OBJECTION.
        6. If the user compares Suri with alternatives, choose HANDLE_COMPARISON.
        7. If the user rejects or hesitates, choose HANDLE_REJECTION.
        8. If the user asks unrelated things, choose REDIRECT_OFF_TOPIC.
        9. If the user seems interested but the system is unsure, choose OFFER_BOOKING.
        10. If unclear and not sales-related, choose ASK_CLARIFYING_QUESTION.
        11. Do not ask endless questions.
        12. If business_type and either main_goal or main_problem are known, the user is qualified enough to offer booking.
        13. Only update lead profile fields when clearly supported by the conversation.
        14. Do not invent values.
        15. Return only valid JSON.

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
            "contact_email": null,
            "contact_phone": null
        },
        "reason": "brief internal reason"
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

            profile[key] = value

        return self.normalise_profile(profile)
    
    def run_controller(self, user_message, history, lead_profile, last_actions, sales_turn=0):
        lead_profile = self.normalise_profile(lead_profile)

        messages = [
            {
                "role": "system",
                "content": self.CONTROLLER_PROMPT
            },
            {
                "role": "user",
                "content": f"""
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
        # --- end schema enforcement ---

        # enforce valid action — fallback if LLM returns something unexpected
        if controller_output.get("next_action") not in self.VALID_ACTIONS:
            controller_output["next_action"] = "ASK_CLARIFYING_QUESTION"
            controller_output["reason"] = f"Invalid action overridden by guardrail"

        # sales turn threshold — force booking offer regardless of controller decision
        if sales_turn >= threshold and not lead_profile.get("booking_offered"):
            controller_output["next_action"] = "OFFER_BOOKING"
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
        if action == "OFFER_BOOKING":
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