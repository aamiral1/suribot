import pytest
from sales_controller import SalesController

SCHEMA_KEYS = set(SalesController.LEAD_PROFILE_SCHEMA.keys())


@pytest.fixture
def sc():
    return SalesController(client=None)


def _valid_output(action="ANSWER_WITH_RAG"):
    return {
        "next_action": action,
        "requires_rag": False,
        "requires_booking_tool": False,
        "confidence": 0.9,
        "lead_profile_updates": {},
        "reason": "test",
    }


# ---------------------------------------------------------------------------
# Phase 1 — create_new_profile + normalise_profile
# ---------------------------------------------------------------------------

class TestCreateNewProfile:
    def test_returns_all_schema_keys(self):
        profile = SalesController.create_new_profile()
        assert set(profile.keys()) == SCHEMA_KEYS

    def test_boolean_defaults_are_false(self):
        profile = SalesController.create_new_profile()
        assert profile["pain_confirmed"] is False
        assert profile["solution_matched"] is False
        assert profile["booking_offered"] is False
        assert profile["booking_completed"] is False

    def test_string_defaults_are_none(self):
        profile = SalesController.create_new_profile()
        assert profile["business_type"] is None
        assert profile["contact_email"] is None
        assert profile["contact_phone"] is None

    def test_returns_a_copy_not_the_schema(self):
        profile = SalesController.create_new_profile()
        profile["business_type"] = "e-commerce"
        assert SalesController.LEAD_PROFILE_SCHEMA["business_type"] is None


class TestNormaliseProfile:
    def test_non_dict_returns_defaults(self, sc):
        result = sc.normalise_profile("not a dict")
        assert set(result.keys()) == SCHEMA_KEYS
        assert result["business_type"] is None

    def test_none_input_returns_defaults(self, sc):
        result = sc.normalise_profile(None)
        assert set(result.keys()) == SCHEMA_KEYS

    def test_missing_keys_filled_with_defaults(self, sc):
        result = sc.normalise_profile({"business_type": "retail"})
        assert result["business_type"] == "retail"
        assert result["contact_email"] is None

    def test_unknown_keys_stripped(self, sc):
        result = sc.normalise_profile({"business_type": "retail", "invented_field": "oops"})
        assert "invented_field" not in result
        assert set(result.keys()) == SCHEMA_KEYS

    def test_known_keys_preserved(self, sc):
        profile = {"business_type": "SaaS", "contact_email": "test@example.com", "booking_offered": True}
        result = sc.normalise_profile(profile)
        assert result["business_type"] == "SaaS"
        assert result["contact_email"] == "test@example.com"
        assert result["booking_offered"] is True


# ---------------------------------------------------------------------------
# Phase 2 — merge_profile
# ---------------------------------------------------------------------------

class TestMergeProfile:
    def test_non_dict_updates_returns_profile_unchanged(self, sc):
        old = SalesController.create_new_profile()
        old["business_type"] = "retail"
        result = sc.merge_profile(old, None)
        assert result["business_type"] == "retail"

    def test_valid_update_overwrites_none(self, sc):
        old = SalesController.create_new_profile()
        result = sc.merge_profile(old, {"business_type": "hospitality"})
        assert result["business_type"] == "hospitality"

    def test_none_in_update_does_not_overwrite_known_value(self, sc):
        old = SalesController.create_new_profile()
        old["contact_email"] = "user@example.com"
        result = sc.merge_profile(old, {"contact_email": None})
        assert result["contact_email"] == "user@example.com"

    def test_unknown_keys_in_update_ignored(self, sc):
        old = SalesController.create_new_profile()
        result = sc.merge_profile(old, {"nonexistent_key": "value"})
        assert "nonexistent_key" not in result
        assert set(result.keys()) == SCHEMA_KEYS

    def test_result_contains_only_schema_keys(self, sc):
        old = {"business_type": "tech", "extra": "noise"}
        result = sc.merge_profile(old, {"main_goal": "leads"})
        assert set(result.keys()) == SCHEMA_KEYS

    def test_boolean_update_applied_correctly(self, sc):
        old = SalesController.create_new_profile()
        result = sc.merge_profile(old, {"booking_offered": True})
        assert result["booking_offered"] is True


# ---------------------------------------------------------------------------
# Phase 3 — apply_guardrails
# ---------------------------------------------------------------------------

class TestApplyGuardrailsSchemaEnforcement:
    def test_non_dict_input_gets_defaults(self, sc):
        result = sc.apply_guardrails("bad input", {}, sales_turn=0)
        assert result["next_action"] == "ASK_CLARIFYING_QUESTION"
        assert result["requires_rag"] is False

    def test_next_action_not_string_falls_back(self, sc):
        out = _valid_output()
        out["next_action"] = 123
        result = sc.apply_guardrails(out, {}, sales_turn=0)
        assert result["next_action"] == "ASK_CLARIFYING_QUESTION"

    def test_requires_rag_not_bool_defaults_false(self, sc):
        out = _valid_output("CONTINUE_HELPFULLY")
        out["requires_rag"] = "yes"
        result = sc.apply_guardrails(out, {}, sales_turn=0)
        assert result["requires_rag"] is False

    def test_requires_booking_tool_not_bool_defaults_false(self, sc):
        out = _valid_output("CONTINUE_HELPFULLY")
        out["requires_booking_tool"] = 1
        result = sc.apply_guardrails(out, {}, sales_turn=0)
        assert result["requires_booking_tool"] is False

    def test_lead_profile_updates_not_dict_defaults_empty(self, sc):
        out = _valid_output("CONTINUE_HELPFULLY")
        out["lead_profile_updates"] = "oops"
        result = sc.apply_guardrails(out, {}, sales_turn=0)
        assert result["lead_profile_updates"] == {}

    def test_reason_not_string_defaults_empty(self, sc):
        out = _valid_output("CONTINUE_HELPFULLY")
        out["reason"] = None
        result = sc.apply_guardrails(out, {}, sales_turn=0)
        assert result["reason"] == ""


class TestApplyGuardrailsActionValidation:
    def test_invalid_action_overridden(self, sc):
        out = _valid_output("TOTALLY_MADE_UP_ACTION")
        result = sc.apply_guardrails(out, {}, sales_turn=0)
        assert result["next_action"] == "ASK_CLARIFYING_QUESTION"

    def test_valid_action_preserved(self, sc):
        out = _valid_output("HANDLE_OBJECTION")
        result = sc.apply_guardrails(out, {}, sales_turn=0)
        assert result["next_action"] == "HANDLE_OBJECTION"


class TestApplyGuardrailsSalesTurn:
    def test_threshold_reached_forces_offer_booking(self, sc):
        out = _valid_output("ASK_SITUATION_QUESTION")
        lead = SalesController.create_new_profile()
        result = sc.apply_guardrails(out, lead, sales_turn=3, threshold=3)
        assert result["next_action"] == "OFFER_BOOKING"

    def test_threshold_reached_but_booking_already_offered_no_override(self, sc):
        out = _valid_output("ASK_SITUATION_QUESTION")
        lead = SalesController.create_new_profile()
        lead["booking_offered"] = True
        result = sc.apply_guardrails(out, lead, sales_turn=3, threshold=3)
        assert result["next_action"] == "ASK_SITUATION_QUESTION"

    def test_below_threshold_no_override(self, sc):
        out = _valid_output("ASK_SITUATION_QUESTION")
        lead = SalesController.create_new_profile()
        result = sc.apply_guardrails(out, lead, sales_turn=2, threshold=3)
        assert result["next_action"] == "ASK_SITUATION_QUESTION"


class TestApplyGuardrailsAutoFlags:
    def test_rag_action_sets_requires_rag_true(self, sc):
        out = _valid_output("ANSWER_WITH_RAG")
        out["requires_rag"] = False
        result = sc.apply_guardrails(out, {}, sales_turn=0)
        assert result["requires_rag"] is True

    def test_booking_action_sets_requires_booking_tool_true(self, sc):
        out = _valid_output("OFFER_AVAILABLE_SLOTS")
        out["requires_booking_tool"] = False
        result = sc.apply_guardrails(out, {}, sales_turn=0)
        assert result["requires_booking_tool"] is True

    def test_non_rag_non_booking_action_flags_remain_false(self, sc):
        out = _valid_output("CONTINUE_HELPFULLY")
        result = sc.apply_guardrails(out, {}, sales_turn=0)
        assert result["requires_rag"] is False
        assert result["requires_booking_tool"] is False

    def test_offer_booking_auto_flips_booking_offered(self, sc):
        out = _valid_output("OFFER_BOOKING")
        result = sc.apply_guardrails(out, {}, sales_turn=0)
        assert result["lead_profile_updates"].get("booking_offered") is True

    def test_confirm_booking_auto_flips_booking_completed(self, sc):
        out = _valid_output("CONFIRM_BOOKING")
        result = sc.apply_guardrails(out, {}, sales_turn=0)
        assert result["lead_profile_updates"].get("booking_completed") is True

    def test_other_action_no_auto_flip(self, sc):
        out = _valid_output("CONTINUE_HELPFULLY")
        result = sc.apply_guardrails(out, {}, sales_turn=0)
        assert "booking_offered" not in result["lead_profile_updates"]
        assert "booking_completed" not in result["lead_profile_updates"]


# ---------------------------------------------------------------------------
# Phase 4 — _is_leadish
# ---------------------------------------------------------------------------

class TestIsLeadish:
    def test_empty_profile_returns_false(self, sc):
        assert sc._is_leadish({}) is False

    def test_business_type_set_returns_true(self, sc):
        assert sc._is_leadish({"business_type": "retail"}) is True

    def test_main_goal_set_returns_true(self, sc):
        assert sc._is_leadish({"main_goal": "more leads"}) is True

    def test_main_problem_set_returns_true(self, sc):
        assert sc._is_leadish({"main_problem": "no visibility"}) is True

    def test_current_marketing_set_returns_true(self, sc):
        assert sc._is_leadish({"current_marketing": "word of mouth"}) is True

    def test_all_four_none_returns_false(self, sc):
        profile = SalesController.create_new_profile()
        assert sc._is_leadish(profile) is False

    def test_unrelated_field_only_returns_false(self, sc):
        assert sc._is_leadish({"contact_email": "x@x.com"}) is False
