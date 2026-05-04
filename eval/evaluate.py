"""
eval/evaluate.py — Sales Controller evaluation harness.

Run from project root (inside venv):
    python eval/evaluate.py

Metrics:
    1. Action Selection Accuracy  — exact match on next_action
    2. Lead Profile Update Accuracy — substring match (strings), exact (bools)
    3. Slot Intent Accuracy — exact match on slot/email intent fields
"""

import json
import os
import sys
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from openai import OpenAI

from sales_controller import SalesController

load_dotenv()

GOLD_PATH = os.path.join(os.path.dirname(__file__), "gold_standard.json")

SLOT_FIELDS = ("requested_day", "requested_time_from", "requested_time_to",
               "selected_slot_day", "selected_slot_time", "provided_email")


def build_day_map() -> tuple[str, dict]:
    """Return (day_map_str, day_lookup_dict) for the next 14 days."""
    today = date.today()
    day_lookup: dict[str, list[date]] = {}
    lines = []

    for offset in range(1, 15):
        d = today + timedelta(days=offset)
        name = d.strftime("%A").lower()
        day_lookup.setdefault(name, []).append(d)
        lines.append(f"  - {name}: {d.strftime('%-d %B %Y')}")

    return "\n".join(lines), day_lookup


def evaluate_action(case: dict, actual: dict) -> tuple[bool, str]:
    expected = case["expected"]["next_action"]
    got = actual.get("next_action", "")
    if isinstance(expected, list):
        passed = got in expected
    else:
        passed = got == expected
    detail = "" if passed else f"  ACTION FAIL: expected {expected!r}, got {got!r}"
    return passed, detail


def evaluate_profile(case: dict, actual: dict) -> tuple[bool, int, int, list[str]]:
    expected_updates = case["expected"].get("lead_profile_updates") or {}
    actual_updates = actual.get("lead_profile_updates") or {}

    checked = 0
    passed = 0
    failures = []

    for key, exp_val in expected_updates.items():
        if exp_val is None:
            continue
        checked += 1
        got_val = actual_updates.get(key)

        if isinstance(exp_val, bool):
            ok = got_val == exp_val
        else:
            exp_str = str(exp_val).lower()
            got_str = str(got_val).lower() if got_val is not None else ""
            ok = exp_str in got_str or got_str in exp_str

        if ok:
            passed += 1
        else:
            failures.append(f"  PROFILE FAIL: {key} — expected {exp_val!r}, got {got_val!r}")

    all_passed = passed == checked
    return all_passed, passed, checked, failures


def evaluate_slots(case: dict, actual: dict) -> tuple[bool | None, int, int, list[str]]:
    expected = case["expected"]
    checked = 0
    passed = 0
    failures = []

    for field in SLOT_FIELDS:
        exp_val = expected.get(field)
        if exp_val is None:
            continue
        checked += 1
        got_val = actual.get(field)
        ok = str(got_val).lower() == str(exp_val).lower()
        if ok:
            passed += 1
        else:
            failures.append(f"  SLOT FAIL: {field} — expected {exp_val!r}, got {got_val!r}")

    if checked == 0:
        return None, 0, 0, []
    return passed == checked, passed, checked, failures


def run_eval():
    with open(GOLD_PATH) as f:
        cases = json.load(f)

    client = OpenAI()
    controller = SalesController(client, model="gpt-4o-mini")

    day_map_str, _ = build_day_map()

    action_pass = 0
    profile_pass_fields = 0
    profile_total_fields = 0
    slot_pass_fields = 0
    slot_total_fields = 0

    COL_ID = 8
    COL_SCENARIO = 44
    header = f"{'ID':<{COL_ID}} {'Scenario':<{COL_SCENARIO}} {'Action':^6}  {'Profile':^7}  {'Slot':^4}"
    sep = "─" * len(header)
    print(f"\n{header}")
    print(sep)

    for case in cases:
        raw = controller.run_controller(
            user_message=case["user_message"],
            history=case["history"],
            lead_profile=case["lead_profile"],
            last_actions=case.get("last_actions", []),
            sales_turn=case.get("sales_turn", 0),
            day_map=day_map_str,
        )
        actual = controller.apply_guardrails(
            controller_output=raw,
            lead_profile=case["lead_profile"],
            sales_turn=case.get("sales_turn", 0),
        )

        a_ok, a_detail = evaluate_action(case, actual)
        p_ok, p_pass, p_total, p_fails = evaluate_profile(case, actual)
        s_ok, s_pass, s_total, s_fails = evaluate_slots(case, actual)

        if a_ok:
            action_pass += 1
        profile_pass_fields += p_pass
        profile_total_fields += p_total
        slot_pass_fields += s_pass
        slot_total_fields += s_total

        action_sym = "✓" if a_ok else "✗"
        profile_sym = "✓" if p_ok else ("✗" if p_total > 0 else "-")
        slot_sym = "✓" if s_ok is True else ("✗" if s_ok is False else "-")

        scenario = case["scenario"][:COL_SCENARIO - 1]
        print(f"{case['id']:<{COL_ID}} {scenario:<{COL_SCENARIO}} {action_sym:^6}  {profile_sym:^7}  {slot_sym:^4}")

        all_failures = ([a_detail] if a_detail else []) + p_fails + s_fails
        for line in all_failures:
            print(line)

    print(sep)

    n = len(cases)
    action_pct = 100 * action_pass / n if n else 0
    profile_pct = 100 * profile_pass_fields / profile_total_fields if profile_total_fields else 0
    slot_pct = 100 * slot_pass_fields / slot_total_fields if slot_total_fields else 0

    print(f"\nAction Selection Accuracy:    {action_pass}/{n}  ({action_pct:.1f}%)")
    print(f"Lead Profile Update Accuracy: {profile_pass_fields}/{profile_total_fields}  ({profile_pct:.1f}%)   [fields checked: {profile_total_fields}]")
    print(f"Slot Intent Accuracy:         {slot_pass_fields}/{slot_total_fields}  ({slot_pct:.1f}%)   [fields checked: {slot_total_fields}]")
    print()


if __name__ == "__main__":
    run_eval()
