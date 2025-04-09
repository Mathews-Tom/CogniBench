# CogniBench - Postprocessing Module
# Version: 0.2 (Phase 4 - Consistency Checks & Review Flagging)

from typing import Any, Dict, List, Optional, Tuple

# Constants
MIN_JUSTIFICATION_LENGTH = 10  # Minimum characters for a non-trivial justification


def normalize_answer(answer: Optional[str]) -> Optional[str]:
    """
    Performs basic normalization on an answer string for comparison.
    (Lowercase, strip whitespace).
    """
    if answer is None:
        return None
    return answer.lower().strip()


def verify_final_answer(
    extracted_answer: Optional[str], correct_answer: Optional[str]
) -> Tuple[Optional[bool], str]:
    """
    Compares the extracted final answer with the correct answer after normalization.

    Args:
        extracted_answer: The answer string extracted from the model response.
        correct_answer: The ground-truth answer string.

    Returns:
        A tuple containing:
        - bool or None: True if match, False if mismatch, None if verification couldn't be done.
        - str: A message describing the verification outcome.
    """
    norm_extracted = normalize_answer(extracted_answer)
    norm_correct = normalize_answer(correct_answer)

    if norm_extracted is None:
        return None, "Verification skipped: Extracted answer was None."
    if norm_correct is None:
        # This is debatable - should it be a mismatch or skipped?
        # Let's treat it as skipped because we can't compare against nothing.
        return None, "Verification skipped: Correct answer was None."

    match = norm_extracted == norm_correct
    message = f"Verification: {'Match' if match else 'Mismatch'} ('{norm_extracted}' vs '{norm_correct}')."
    return match, message


def aggregate_scores(
    parsed_evaluation_content: Dict[str, Dict[str, str]],
) -> Dict[str, Any]:
    """
    Aggregates the parsed rubric scores, performs basic consistency checks,
    and determines if human review is needed.

    Args:
        parsed_evaluation_content: The dictionary containing the parsed rubric scores
                                    (e.g., {"Problem Understanding": {"score": "Yes", ...}, ...}).
                                    Assumes keys like 'score' and 'justification' exist and are valid strings.

    Returns:
        A dictionary containing:
        - 'aggregated_score': "Pass", "Fail", or "Partial".
        - 'needs_human_review': bool.
        - 'review_reasons': List[str].
    """
    results = {
        "aggregated_score": None,
        "needs_human_review": False,
        "review_reasons": [],
    }
    scores_found = []
    trivial_justifications = []

    if not parsed_evaluation_content:
        results["review_reasons"].append(
            "Aggregation failed: No evaluation content provided."
        )
        results["needs_human_review"] = True
        # Cannot determine score if no content
        return results

    for criterion, details in parsed_evaluation_content.items():
        # We assume the parser validated the basic structure (dict with score/justification)
        score = details.get("score", "").strip().lower()
        justification = details.get("justification", "").strip()

        if not score:
            results["review_reasons"].append(
                f"Missing or empty score for criterion '{criterion}'."
            )
            results["needs_human_review"] = True
            # Treat missing score as needing review, potentially implies failure
            scores_found.append(
                "fail"
            )  # Or a distinct 'error' state? Let's use fail for now.
            continue

        scores_found.append(score)

        # Consistency Check: Trivial Justification for Non-"Yes" scores
        if (
            score in ["no", "partial"]
            and len(justification) <= MIN_JUSTIFICATION_LENGTH
        ):
            trivial_justifications.append(criterion)
            results["needs_human_review"] = True

    if not scores_found:
        results["review_reasons"].append(
            "Aggregation failed: No valid scores found in evaluation content."
        )
        results["needs_human_review"] = True
        return results

    if trivial_justifications:
        results["review_reasons"].append(
            f"Potential trivial justification for 'No'/'Partial' score in criteria: {', '.join(trivial_justifications)}."
        )

    # Determine Aggregated Score
    if "no" in scores_found:
        results["aggregated_score"] = "Fail"
    elif "partial" in scores_found:
        results["aggregated_score"] = "Partial"
        # Optionally flag partial scores for review automatically
        # if "Partial score requires review" not in results["review_reasons"]:
        #     results["review_reasons"].append("Partial score requires review.")
        #     results["needs_human_review"] = True
    elif all(s == "yes" for s in scores_found):
        results["aggregated_score"] = "Pass"
    else:
        # Should not happen if scores are only yes/no/partial, but as fallback:
        results["aggregated_score"] = "Fail"  # Or Needs Review? Fail is safer.
        results["review_reasons"].append(
            f"Inconclusive aggregation state with scores: {scores_found}."
        )
        results["needs_human_review"] = True

    return results


def perform_postprocessing(
    parsed_judge_response: Dict[str, Any],
    extracted_final_answer: Optional[str],
    correct_final_answer: Optional[str],
) -> Dict[str, Any]:
    """
    Orchestrates the post-processing steps: answer verification, score aggregation,
    consistency checks, and human review flagging.

    Args:
        parsed_judge_response: The dictionary returned by parse_judge_response.
                                May contain an 'error' key or an 'evaluation' key.
        extracted_final_answer: The final answer extracted from the model response.
        correct_final_answer: The ground-truth final answer.

    Returns:
        A dictionary containing the consolidated post-processing results:
        - 'final_answer_verified': Optional[bool] (True, False, or None if skipped)
        - 'verification_message': str
        - 'aggregated_score': Optional[str] ("Pass", "Fail", "Partial")
        - 'needs_human_review': bool
        - 'review_reasons': List[str]
        - 'parsing_error': Optional[str] (Content of 'error' key from parser)
    """
    postprocessing_results = {
        "final_answer_verified": None,
        "verification_message": "Verification not performed.",
        "aggregated_score": None,
        "needs_human_review": False,
        "review_reasons": [],
        "parsing_error": None,
    }

    # 1. Check for Parsing Errors
    if "error" in parsed_judge_response:
        postprocessing_results["parsing_error"] = parsed_judge_response["error"]
        postprocessing_results["needs_human_review"] = True
        postprocessing_results["review_reasons"].append(
            f"LLM response parsing failed: {parsed_judge_response['error']}"
        )
        # Cannot aggregate scores if parsing failed
        # Still perform answer verification if possible
        verified, message = verify_final_answer(
            extracted_final_answer, correct_final_answer
        )
        postprocessing_results["final_answer_verified"] = verified
        postprocessing_results["verification_message"] = message
        return postprocessing_results  # Stop here if parsing failed

    # 2. Perform Final Answer Verification
    verified, message = verify_final_answer(
        extracted_final_answer, correct_final_answer
    )
    postprocessing_results["final_answer_verified"] = verified
    postprocessing_results["verification_message"] = message
    # Flag for review if verification failed? Optional rule.
    # if verified is False:
    #     postprocessing_results["needs_human_review"] = True
    #     postprocessing_results["review_reasons"].append("Final answer verification failed.")

    # 3. Perform Score Aggregation and Consistency Checks
    evaluation_content = parsed_judge_response.get("evaluation")
    if isinstance(evaluation_content, dict):
        aggregation_results = aggregate_scores(evaluation_content)
        postprocessing_results["aggregated_score"] = aggregation_results[
            "aggregated_score"
        ]
        # Combine review flags and reasons
        postprocessing_results["needs_human_review"] = (
            postprocessing_results["needs_human_review"]
            or aggregation_results["needs_human_review"]
        )
        postprocessing_results["review_reasons"].extend(
            aggregation_results["review_reasons"]
        )
    else:
        # This case should ideally be caught by the parser, but handle defensively
        postprocessing_results["needs_human_review"] = True
        postprocessing_results["review_reasons"].append(
            "Postprocessing error: 'evaluation' content missing or invalid after successful parsing."
        )
        postprocessing_results["aggregated_score"] = (
            "Fail"  # Assign Fail if aggregation cannot run
        )

    # Ensure unique reasons if duplicates were added
    postprocessing_results["review_reasons"] = sorted(
        list(set(postprocessing_results["review_reasons"]))
    )

    return postprocessing_results


# --- Example Usage ---
if __name__ == "__main__":
    import json

    # Make sure response_parser is importable, adjust path if needed
    try:
        from response_parser import parse_judge_response
    except ImportError:
        print(
            "Warning: Could not import response_parser. Run this script from the 'core' directory or adjust PYTHONPATH."
        )

        # Define a dummy parser if needed for basic testing without the real parser
        def parse_judge_response(resp, **kwargs):
            return resp if isinstance(resp, dict) else {"error": "Dummy parser error"}

    print("--- Test Case 1: Parsing Error ---")
    bad_parse_input = {"error": "Invalid JSON format: ..."}
    results1 = perform_postprocessing(bad_parse_input, "x=5", "x=5")
    print(json.dumps(results1, indent=2))
    assert results1["needs_human_review"] is True
    assert results1["parsing_error"] is not None
    assert results1["aggregated_score"] is None
    assert results1["final_answer_verified"] is True  # Verification still runs

    print("\n--- Test Case 2: All Pass, Answer Match ---")
    good_parse_input_pass = {
        "evaluation": {
            "Problem Understanding": {
                "score": "Yes",
                "justification": "Clearly understood the requirements.",
            },
            "Rigor and Completeness": {
                "score": "YES",
                "justification": "Covered all necessary steps.",
            },
        }
    }
    results2 = perform_postprocessing(good_parse_input_pass, "10", "10")
    print(json.dumps(results2, indent=2))
    assert results2["needs_human_review"] is False
    assert results2["parsing_error"] is None
    assert results2["aggregated_score"] == "Pass"
    assert results2["final_answer_verified"] is True

    print("\n--- Test Case 3: One Fail, Answer Mismatch ---")
    good_parse_input_fail = {
        "evaluation": {
            "Problem Understanding": {
                "score": "Yes",
                "justification": "Clearly understood the requirements.",
            },
            "Results/Formulae": {
                "score": "No",
                "justification": "Calculation error led to wrong result.",
            },
        }
    }
    results3 = perform_postprocessing(good_parse_input_fail, "10", "12")
    print(json.dumps(results3, indent=2))
    assert (
        results3["needs_human_review"] is False
    )  # Failing score doesn't auto-flag unless justification is trivial
    assert results3["parsing_error"] is None
    assert results3["aggregated_score"] == "Fail"
    assert results3["final_answer_verified"] is False

    print("\n--- Test Case 4: Partial Score, Trivial Justification ---")
    good_parse_input_partial_trivial = {
        "evaluation": {
            "Problem Understanding": {
                "score": "Yes",
                "justification": "Clearly understood the requirements.",
            },
            "Assumptions": {
                "score": "Partial",
                "justification": "Missed one",
            },  # Trivial justification
        }
    }
    results4 = perform_postprocessing(good_parse_input_partial_trivial, "A", "A")
    print(json.dumps(results4, indent=2))
    assert results4["needs_human_review"] is True
    assert "Potential trivial justification" in results4["review_reasons"][0]
    assert results4["parsing_error"] is None
    assert results4["aggregated_score"] == "Partial"
    assert results4["final_answer_verified"] is True

    print("\n--- Test Case 5: Fail Score, Trivial Justification ---")
    good_parse_input_fail_trivial = {
        "evaluation": {
            "Problem Understanding": {
                "score": "No",
                "justification": "Wrong",
            }  # Trivial justification
        }
    }
    results5 = perform_postprocessing(good_parse_input_fail_trivial, "A", "B")
    print(json.dumps(results5, indent=2))
    assert results5["needs_human_review"] is True
    assert "Potential trivial justification" in results5["review_reasons"][0]
    assert results5["parsing_error"] is None
    assert results5["aggregated_score"] == "Fail"
    assert results5["final_answer_verified"] is False

    print("\n--- Test Case 6: Missing Evaluation Content (Defensive) ---")
    missing_eval_content = {
        "some_other_key": "value"
    }  # Parser shouldn't allow this, but test defensively
    results6 = perform_postprocessing(missing_eval_content, "A", "B")
    print(json.dumps(results6, indent=2))
    assert results6["needs_human_review"] is True
    assert "evaluation' content missing" in results6["review_reasons"][0]
    assert results6["aggregated_score"] == "Fail"  # Default to Fail

    print("\n--- Test Case 7: Verification Skipped (No Correct Answer) ---")
    results7 = perform_postprocessing(good_parse_input_pass, "10", None)
    print(json.dumps(results7, indent=2))
    assert results7["final_answer_verified"] is None
    assert "Correct answer was None" in results7["verification_message"]
    assert results7["aggregated_score"] == "Pass"  # Aggregation unaffected
    assert results7["needs_human_review"] is False

    print("\n--- All Tests Passed (Implicitly) ---")
