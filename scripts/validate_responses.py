"""
validate_responses.py

Validates structuring and judging model responses according to the checklists in docs/prompt_checklist.md.

Usage (CLI):
    python scripts/validate_responses.py --type structuring --response response.json --prompt prompt.txt
    python scripts/validate_responses.py --type judging --response response.json --prompt prompt.txt

Can also be imported as a library:
    from scripts.validate_responses import validate_structuring_response, validate_judging_response
"""

import json
import argparse
from typing import Dict, Any, List, Tuple

# --- Structuring Checklist Items ---
STRUCTURING_CHECKLIST = [
    # SECTION 1
    {
        "id": "problem_relevant_structuring",
        "desc": "Does the structured output match what the question asks for (e.g., equation solving vs. sequence pattern vs. geometric proof)?"
    },
    {
        "id": "assumptions_derived_from_question",
        "desc": "Are initial conditions and domain restrictions explicitly included if stated or clearly implied by the question?"
    },
    # SECTION 2
    {
        "id": "steps_logically_ordered",
        "desc": 'Are the "steps" in a clear, step-by-step sequence that mirrors the actual logical flow?',
        "red_flag": "Steps are jumbled, missing transitions, or overly compressed."
    },
    {
        "id": "intermediate_results_reflect_calculations",
        "desc": 'Are "intermediate_results" used to reflect actual outputs (e.g., derived values, transformed expressions)?'
    },
    {
        "id": "final_answer_matches_reasoning",
        "desc": 'Is "final_answer" accurate and consistent with the preceding steps?'
    },
    # SECTION 3
    {
        "id": "assumptions_field_captures_constraints",
        "desc": 'Does the "assumptions" field capture inferred constraints?'
    },
    {
        "id": "no_invented_justifications",
        "desc": "The structuring should not add assumptions or results that aren’t actually stated in the original solution."
    },
    # SECTION 4
    {
        "id": "all_required_fields_present",
        "desc": 'Are all required fields present: "assumptions" (str), "steps" (list), "intermediate_results" (list), "final_answer" (str), "format_notes" (str)?'
    },
    {
        "id": "format_notes_captured",
        "desc": 'If the original solution emphasizes the final answer (e.g., boxed, bold, LaTeX), is this noted under "format_notes"?'
    },
    # SECTION 5
    {
        "id": "steps_faithfully_mirror_reasoning",
        "desc": "Does each step faithfully mirror the intent and logic of the original model or expert response?"
    },
    {
        "id": "avoids_over_summarization",
        "desc": "Structuring shouldn't compress multi-line reasoning into single steps when doing so obscures logical transitions."
    },
    # SECTION 6
    {
        "id": "structuring_makes_sense_in_context",
        "desc": "Does the structuring make sense in context of the original question?"
    },
    {
        "id": "final_answer_and_reasoning_match_question",
        "desc": "Does the final answer and reasoning correspond to what the question actually asked for?"
    },
]

STRUCTURING_REQUIRED_FIELDS = [
    ("assumptions", str),
    ("steps", list),
    ("intermediate_results", list),
    ("final_answer", str),
    ("format_notes", str),
]

STRUCTURING_RED_FLAGS = [
    # Problem Type, Mistake
    ("Pattern sequence", "Skips pattern explanation in steps"),
    ("Equation solving", "No mention of equation or variable in assumptions"),
    ("Proofs or derivations", "No use of logical transitions (e.g., 'By definition', 'Using theorem')"),
    ("Geometry", "No mention of given constraints or diagram assumptions"),
    ("Steps", "Steps are jumbled, missing transitions, or overly compressed"),
]

# --- Judging Checklist Items ---
JUDGING_CHECKLIST = [
    # SECTION 1
    {
        "id": "objective_match",
        "desc": "Does the model answer the type of question asked (e.g., numeric prediction, proof, simplification)?"
    },
    {
        "id": "data_usage",
        "desc": "Did the model use all relevant information from the question (e.g., initial sequence terms, given constraints)?"
    },
    {
        "id": "final_answer_relevance",
        "desc": "Is the model's final answer responsive to what the question asks for?"
    },
    # SECTION 2
    {
        "id": "appropriate_method_choice",
        "desc": "Given the question format (e.g., number sequence, geometry, algebra), did the model choose a natural or efficient strategy?",
        "red_flag": "Uses high-degree polynomial fitting for a 5-term integer sequence."
    },
    {
        "id": "assumption_justifiability",
        "desc": "Are the model’s assumptions defensible based on the question alone?"
    },
    # SECTION 3
    {
        "id": "correct_identification_of_whats_asked",
        "desc": "Does the model clearly grasp what needs to be solved?"
    },
    {
        "id": "constraints_incorporated",
        "desc": "Did the model respect or explicitly handle any constraints in the question (e.g., integer domain, non-negative inputs)?"
    },
    # SECTION 4
    {
        "id": "different_approach_vs_ideal",
        "desc": "Did the model use a different approach than the ideal? If yes, was it mathematically valid and appropriate given the question?"
    },
    {
        "id": "method_generalizes",
        "desc": "Does the model’s method generalize well if extended to similar questions?"
    },
    # SECTION 5
    {
        "id": "rubric_mapping",
        "desc": "Would this be judged the same way if the question were different? If not, the question must be a deciding factor."
    },
]

# --- Validation Functions ---

def _check_required_fields(response: dict, required_fields: List[Tuple[str, type]]) -> List[str]:
    missing = []
    for field, typ in required_fields:
        if field not in response:
            missing.append(field)
        else:
            if not isinstance(response[field], typ):
                missing.append(field)
    return missing

def _flag_structuring_red_flags(response: dict, prompt: str) -> List[str]:
    flags = []
    # Heuristic checks for red flags
    if "steps" in response and isinstance(response["steps"], list):
        steps = response["steps"]
        if len(steps) > 0 and any(len(step.strip()) == 0 for step in steps):
            flags.append("Empty step(s) in 'steps'.")
        if len(steps) > 0 and any("pattern" in prompt.lower() and "pattern" not in " ".join(steps).lower() for _ in steps):
            flags.append("Pattern sequence: Skips pattern explanation in steps.")
        if len(steps) > 0 and any("definition" in step.lower() or "theorem" in step.lower() for step in steps) is False and "proof" in prompt.lower():
            flags.append("Proofs or derivations: No use of logical transitions (e.g., 'By definition', 'Using theorem').")
        if len(steps) > 0 and (steps != sorted(steps)):
            # This is a naive check; real logical order is hard to check programmatically
            pass
    if "assumptions" in response and isinstance(response["assumptions"], str):
        if ("equation" in prompt.lower() or "variable" in prompt.lower()) and ("equation" not in response["assumptions"].lower() and "variable" not in response["assumptions"].lower()):
            flags.append("Equation solving: No mention of equation or variable in assumptions.")
        if ("geometry" in prompt.lower() or "diagram" in prompt.lower()) and ("constraint" not in response["assumptions"].lower() and "diagram" not in response["assumptions"].lower()):
            flags.append("Geometry: No mention of given constraints or diagram assumptions.")
    return flags

def validate_structuring_response(response: dict, prompt: str) -> dict:
    """
    Validate a structuring model response against the checklist.
    Returns a dict with pass/fail for each checklist item, and flags for red flags or missing fields.
    """
    results = {}
    # 1. Required fields
    missing_fields = _check_required_fields(response, STRUCTURING_REQUIRED_FIELDS)
    results["missing_fields"] = missing_fields

    # 2. Checklist items
    for item in STRUCTURING_CHECKLIST:
        check_id = item["id"]
        desc = item["desc"]
        # Heuristic: For required fields, check presence
        if check_id == "all_required_fields_present":
            results[check_id] = len(missing_fields) == 0
        elif check_id == "steps_logically_ordered":
            # Can't fully check logical order, but can check if steps is a non-empty list
            results[check_id] = isinstance(response.get("steps"), list) and len(response["steps"]) > 0
        elif check_id == "intermediate_results_reflect_calculations":
            results[check_id] = isinstance(response.get("intermediate_results"), list)
        elif check_id == "final_answer_matches_reasoning":
            results[check_id] = isinstance(response.get("final_answer"), str) and len(response["final_answer"].strip()) > 0
        elif check_id == "assumptions_field_captures_constraints":
            results[check_id] = "assumptions" in response
        elif check_id == "format_notes_captured":
            # If format_notes is non-empty when prompt suggests emphasis
            notes = response.get("format_notes", "")
            results[check_id] = isinstance(notes, str)
        else:
            # For other items, mark as True (pass) for now; manual review needed for full fidelity
            results[check_id] = True

    # 3. Red flags
    red_flags = _flag_structuring_red_flags(response, prompt)
    results["red_flags"] = red_flags

    return results

def _flag_judging_red_flags(response: dict, prompt: str) -> List[str]:
    flags = []
    # Example: If method is polynomial fitting for a short sequence
    if "method" in response and "polynomial" in str(response["method"]).lower() and "sequence" in prompt.lower():
        if "5-term" in prompt or "five term" in prompt:
            flags.append("Uses high-degree polynomial fitting for a 5-term integer sequence.")
    return flags

def validate_judging_response(response: dict, prompt: str) -> dict:
    """
    Validate a judging model response against the checklist.
    Returns a dict with pass/fail for each checklist item, and flags for red flags or missing fields.
    """
    results = {}
    # 1. Checklist items
    for item in JUDGING_CHECKLIST:
        check_id = item["id"]
        desc = item["desc"]
        # Heuristic: For some items, check presence of relevant fields
        if check_id == "objective_match":
            results[check_id] = True  # Needs manual review
        elif check_id == "data_usage":
            results[check_id] = True  # Needs manual review
        elif check_id == "final_answer_relevance":
            results[check_id] = "final_answer" in response
        elif check_id == "appropriate_method_choice":
            results[check_id] = "method" in response
        elif check_id == "assumption_justifiability":
            results[check_id] = "assumptions" in response
        elif check_id == "correct_identification_of_whats_asked":
            results[check_id] = True  # Needs manual review
        elif check_id == "constraints_incorporated":
            results[check_id] = True  # Needs manual review
        elif check_id == "different_approach_vs_ideal":
            results[check_id] = True  # Needs manual review
        elif check_id == "method_generalizes":
            results[check_id] = True  # Needs manual review
        elif check_id == "rubric_mapping":
            results[check_id] = True  # Needs manual review
        else:
            results[check_id] = True

    # 2. Red flags
    red_flags = _flag_judging_red_flags(response, prompt)
    results["red_flags"] = red_flags

    # 3. Missing fields (for judging, not strictly required, but useful)
    results["missing_fields"] = [k for k in ["final_answer", "method", "assumptions"] if k not in response]

    return results

# --- CLI Interface ---

def main():
    parser = argparse.ArgumentParser(description="Validate structuring or judging model responses.")
    parser.add_argument("--type", choices=["structuring", "judging"], required=True, help="Type of response to validate.")
    parser.add_argument("--response", required=True, help="Path to response JSON file.")
    parser.add_argument("--prompt", required=True, help="Path to prompt text file.")
    args = parser.parse_args()

    with open(args.response, "r", encoding="utf-8") as f:
        response = json.load(f)
    with open(args.prompt, "r", encoding="utf-8") as f:
        prompt = f.read()

    if args.type == "structuring":
        result = validate_structuring_response(response, prompt)
    else:
        result = validate_judging_response(response, prompt)

    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()