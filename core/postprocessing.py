# -*- coding: utf-8 -*-
"""
CogniBench Postprocessing Module.

This module handles the processing of evaluation results after the judge LLM
response has been parsed. It includes functions for:
- Normalizing answer strings for comparison.
- Verifying the extracted final answer against the correct answer.
- Aggregating rubric scores into an overall pass/fail/partial status.
- Performing consistency checks (e.g., trivial justifications).
- Flagging evaluations that require human review.

Version: 0.3 (Phase 5 - Code Quality Enhancements & SymPy Verification)
"""

import json
import logging
import traceback
from typing import Any, Dict, List, Literal, Optional, Tuple

# Setup logger for this module *before* potential logging during imports
logger = logging.getLogger('backend')


# Attempt to import sympy for mathematical comparison
try:
    import sympy

    # Import specific sympy components needed
    from sympy import (
        Basic,
        N,
        SympifyError,  # N for numerical eval
        simplify,
        sympify,
    )
    from sympy.parsing.latex import parse_latex

    SYMPY_AVAILABLE = True
    logger.info("SymPy library found. Mathematical answer verification enabled.")
except ImportError:
    SYMPY_AVAILABLE = False
    # Define dummy types if sympy is not available to avoid runtime errors on type hints
    Basic = type(None)  # Use NoneType as a placeholder
    logger.warning(
        "SymPy library not found. Mathematical answer verification will fall back "
        "to basic string comparison. Install sympy (`pip install sympy`) for enhanced checking."
    )


def safe_sympy_parse(expr_str: str) -> Optional[Basic]:
    """
    Safely parse a mathematical expression string using SymPy, handling empty or malformed inputs gracefully.

    Args:
        expr_str (str): The mathematical expression string to parse.

    Returns:
        Optional[Basic]: Parsed SymPy expression if successful, None otherwise.
    """
    if not expr_str.strip():
        logger.warning("SymPy parsing failed: Empty expression string.")
        return None
    try:
        expr = sympify(expr_str.replace("^", "**"))
        return expr
    except (SympifyError, TypeError, SyntaxError):
        try:
            expr = parse_latex(expr_str)
            return expr
        except Exception as e:
            logger.warning(f"SymPy parsing failed: {str(e)}")
            return None


# Type alias for aggregated score outcomes
AggregatedScore = Optional[Literal["Pass", "Fail", "Partial"]]


# --- Constants ---
# Minimum character length for a justification to be considered non-trivial.
# Used to flag potentially weak justifications for 'No' or 'Partial' scores.
MIN_JUSTIFICATION_LENGTH: int = 10


def normalize_answer(answer: Optional[str]) -> Optional[str]:
    """
    Performs basic normalization on an answer string for comparison.

    Normalization involves converting the string to lowercase and stripping
    leading/trailing whitespace.

    Args:
        answer: The answer string to normalize, or None.

    Returns:
        The normalized string, or None if the input was None.
    """
    if answer is None:
        return None
    return answer.lower().strip()


def verify_final_answer(
    extracted_answer: Optional[str], correct_answer: Optional[str]
) -> Tuple[Optional[bool], str]:
    """Compares the extracted final answer with the correct (ideal) answer.

    Attempts mathematical equivalence checking using SymPy if available,
    otherwise falls back to normalized string comparison.

    Steps:
    1.  Normalizes both answers using `normalize_answer` (lowercase, stripped).
    2.  Handles cases where either normalized answer is None (returns skipped).
    3.  If SymPy is available:
        a. Attempts to parse both normalized answers using `sympify` (standard)
            and `parse_latex` (LaTeX fallback).
        b. If both parse successfully, compares the expressions using `expr1.equals(expr2)`.
            This checks for symbolic equivalence (e.g., x+y == y+x).
        c. If symbolic comparison fails or errors, logs a warning and proceeds to fallback.
    4.  If SymPy is not available OR SymPy parsing/comparison failed:
        a. Performs direct string comparison of the normalized answers.
    5.  Formats a message indicating the comparison method and result.

    Args:
        extracted_answer: The final answer string extracted from the model's
            response, or None.
        correct_answer: The ground-truth or ideal answer string, or None.

    Returns:
        A tuple containing:
        - Verification result (Optional[bool]): True if match, False if mismatch,
            None if skipped.
        - Verification message (str): Describes outcome and method used.
    """
    norm_extracted = normalize_answer(extracted_answer)
    norm_correct = normalize_answer(correct_answer)

    # Handle cases where one or both answers are None after normalization
    if norm_extracted is None:
        logger.debug("Skipping verification: Extracted answer is None.")
        return None, "Verification skipped: Extracted answer was None."
    if norm_correct is None:
        # Treat comparison against None as skipped, as we can't determine correctness.
        logger.debug("Skipping verification: Correct answer is None.")
        return None, "Verification skipped: Correct answer was None."

    # --- Attempt SymPy Comparison (if available and applicable) ---
    match: Optional[bool] = None
    comparison_method = "String"  # Default method

    if SYMPY_AVAILABLE:
        expr1 = safe_sympy_parse(norm_extracted)
        expr2 = safe_sympy_parse(norm_correct)

        if expr1 is None or expr2 is None:
            logger.warning(
                "Failed to parse one or both answers with SymPy. Falling back to string comparison."
            )
            match = None
            comparison_method = "String"

        # If both parsed successfully, compare them
        if expr1 is not None and expr2 is not None:
            comparison_method = "SymPy"
            # Use simplify and equals for robust comparison
            try:
                # Alternative: simplify(expr1 - expr2) == 0
                # Using equals() handles symbolic equality better
                match = expr1.equals(expr2)
                logger.debug(
                    "SymPy comparison: %s vs %s -> Match: %s", expr1, expr2, match
                )
            except Exception as e:
                logger.warning(
                    "SymPy comparison failed: %s. Falling back to string comparison.",
                    e,
                    exc_info=True,
                )
                # Fallback to string comparison if SymPy comparison itself errors out
                match = None
                comparison_method = "String"  # Revert method

        match = None  # Ensure match is None if SymPy fails

    # --- Fallback to String Comparison ---
    if match is None:  # If SymPy wasn't used, failed, or comparison errored
        comparison_method = "String"
        match = norm_extracted == norm_correct
        logger.debug(
            "String comparison: '%s' vs '%s' -> Match: %s",
            norm_extracted,
            norm_correct,
            match,
        )

    # --- Format Message ---
    # Ensure match is boolean before formatting message
    if match is None:
        # This case should ideally not be reached if inputs are not None,
        # but handle defensively.
        logger.error("Verification ended with match=None unexpectedly.")
        match = False  # Default to mismatch if something went wrong
        message = "Verification Error: Comparison resulted in None."
    else:
        message = (
            f"Verification ({comparison_method}): {'Match' if match else 'Mismatch'} "
            f"(Extracted: '{norm_extracted}', Correct: '{norm_correct}')."
        )

    return match, message


def aggregate_scores(
    parsed_evaluation_content: Dict[str, Dict[str, str]],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Aggregates parsed rubric scores and flags potential issues for review based on configurable rules.

    This function takes the structured evaluation content (parsed from the judge
    LLM's response) and aggregation settings from the configuration file, performing the following:
    1.  Iterates through each criterion and its score/justification.
    2.  Checks for missing scores, flagging for review if found.
    3.  Checks if justifications for "No" or "Partial" scores are potentially
        trivial (based on `MIN_JUSTIFICATION_LENGTH`), flagging for review.
    4.  Determines an overall `aggregated_score` ("Pass", "Fail", "Partial")
        based on the individual criterion scores:
        - "Fail" if any criterion score is "No".
        - "Partial" if any criterion score is "Partial" (and none are "No").
        - "Pass" if all criterion scores are "Yes".
        - Defaults to "Fail" in unexpected scenarios.
    5.  Collects reasons for why human review might be needed.

    Args:
        parsed_evaluation_content: A dictionary where keys are criterion names
            (str) and values are dictionaries containing at least 'score' (str)
            and 'justification' (str). Example:
            `{"Problem Understanding": {"score": "Yes", "justification": "..."}}`
            Assumes the input structure has been partially validated by the parser,
            but performs defensive checks.

    Returns:
        A dictionary containing the aggregation results:
        - 'aggregated_score' (AggregatedScore): "Pass", "Fail", "Partial", or None
            if aggregation could not be performed.
        - 'needs_human_review' (bool): True if any checks indicate potential issues.
        - 'review_reasons' (List[str]): A list of messages explaining why review
            might be needed.
    """
    # Initialize results structure
    aggregation_rules = config.get("aggregation_settings", {}).get(
        "aggregation_rules", {}
    )
    consistency_checks = config.get("aggregation_settings", {}).get(
        "consistency_checks", {}
    )

    results: Dict[str, Any] = {
        "aggregated_score": None,
        "needs_human_review": False,
        "review_reasons": [],
    }
    scores_found: List[str] = []  # Stores normalized scores ('yes', 'no', 'partial')
    trivial_justification_criteria: List[
        str
    ] = []  # Stores criteria with trivial justifications

    if not parsed_evaluation_content:
        msg = "Aggregation failed: Parsed evaluation content is empty."
        logger.warning(msg)
        results["review_reasons"].append(msg)
        results["needs_human_review"] = True
        # Cannot determine an aggregated score if there's no content.
        return results

    for criterion, details in parsed_evaluation_content.items():
        # Assume parser ensures 'details' is a dict, but check keys defensively
        score_raw: str = details.get("score", "")
        justification: str = details.get("justification", "")

        # Normalize score for internal logic
        score_norm: str = score_raw.strip().lower()

        # --- Validation within Aggregation ---
        if not score_norm:
            msg = f"Missing or empty score found for criterion '{criterion}' during aggregation."
            logger.warning(msg)
            results["review_reasons"].append(msg)
            results["needs_human_review"] = True
            # Treat missing score as implicitly 'fail' for aggregation purposes
            scores_found.append("fail")
            continue  # Move to the next criterion

        # Store the normalized score for final aggregation
        scores_found.append(score_norm)

        # Consistency Check: Trivial Justification for Non-"Yes" scores
        # --- Consistency Check: Trivial Justification ---
        # Flag if a 'No' or 'Partial' score has a very short justification.
        if consistency_checks.get("enable_trivial_justification_check", True):
            trivial_length_threshold = consistency_checks.get(
                "trivial_justification_length_threshold", MIN_JUSTIFICATION_LENGTH
            )
            if (
                score_norm in ["no", "partial"]
                and len(justification.strip()) <= trivial_length_threshold
            ):
                logger.warning(
                    "Potential trivial justification found for criterion '%s' (Score: %s, Length: %d)",
                    criterion,
                    score_raw,
                    len(justification.strip()),
                )
                trivial_justification_criteria.append(criterion)
                results["needs_human_review"] = True

    if not scores_found:
        msg = "Aggregation failed: No valid scores were processed from the evaluation content."
        logger.error(
            msg
        )  # This indicates a more severe issue if loop ran but found nothing
        results["review_reasons"].append(msg)
        results["needs_human_review"] = True
        return results

    # Add reason if trivial justifications were found
    if trivial_justification_criteria:
        criteria_str = ", ".join(f"'{c}'" for c in trivial_justification_criteria)
        results["review_reasons"].append(
            f"Potential trivial justification for 'No'/'Partial' score in criteria: {criteria_str}."
        )

    # --- Determine Aggregated Score ---
    # Logic: Fail if any 'no', Partial if any 'partial' (and no 'no'), Pass if all 'yes'.
    final_score: AggregatedScore
    if aggregation_rules.get("fail_if_any_no", True) and "no" in scores_found:
        final_score = "Fail"
    elif (
        aggregation_rules.get("partial_if_any_partial", True)
        and "partial" in scores_found
    ):
        final_score = "Partial"
    elif aggregation_rules.get("pass_if_all_yes", True) and all(
        s == "yes" for s in scores_found
    ):
        final_score = "Pass"
    else:
        msg = f"Inconclusive aggregation state. Scores found: {scores_found}. Defaulting to Fail."
        logger.error(msg)
        results["review_reasons"].append(msg)
        results["needs_human_review"] = True
        final_score = "Fail"

    results["aggregated_score"] = final_score
    logger.debug(
        "Score aggregation complete using configurable rules. Result: %s", final_score
    )

    return results


def perform_postprocessing(
    parsed_judge_response: Dict[str, Any],
    extracted_final_answer: Optional[str],
    correct_final_answer: Optional[str],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Orchestrates all post-processing steps for a single evaluation.

    This function coordinates the analysis of the parsed judge response and
    the extracted final answer. It performs:
    1.  Checks for parsing errors reported by `parse_judge_response`.
    2.  Verifies the extracted final answer against the correct answer using
        `verify_final_answer`.
    3.  Aggregates scores and performs consistency checks using `aggregate_scores`
        (only if parsing was successful).
    4.  Consolidates results, including the final verification status, aggregated
        score, human review flag, and reasons for review.

    Args:
        parsed_judge_response: The dictionary returned by `parse_judge_response`.
            This dictionary might contain an 'error' key if parsing failed, or
            an 'evaluation' key with the structured rubric scores if successful.
        extracted_final_answer: The final answer string extracted from the model
            response during preprocessing, or None.
        correct_final_answer: The ground-truth or ideal final answer string, or None.

    Returns:
        A dictionary summarizing the post-processing results, containing keys:
        - 'final_answer_verified' (Optional[bool]): Result of `verify_final_answer`.
        - 'verification_message' (str): Message from `verify_final_answer`.
        - 'aggregated_score' (AggregatedScore): Result from `aggregate_scores`, or
            None if parsing failed.
        - 'needs_human_review' (bool): Overall flag indicating if review is needed
            due to parsing errors, aggregation flags, or potentially verification failures.
        - 'review_reasons' (List[str]): Consolidated list of reasons for review.
        - 'parsing_error' (Optional[str]): The error message from the parser, if any.
    """
    # Initialize the results dictionary
    postprocessing_results: Dict[str, Any] = {
        "final_answer_verified": None,
        "verification_message": "Verification not performed.",  # Default message
        "aggregated_score": None,
        "needs_human_review": False,
        "review_reasons": [],
        "parsing_error": None,  # Store parser error if present
    }
    logger.debug("Starting postprocessing orchestration...")

    # --- 1. Check for Parsing Errors ---
    parsing_error: Optional[str] = parsed_judge_response.get("error")
    if parsing_error:
        logger.warning(
            "Postprocessing step 1: Parsing error detected: %s", parsing_error
        )
        postprocessing_results["parsing_error"] = parsing_error
        postprocessing_results["needs_human_review"] = True
        postprocessing_results["review_reasons"].append(
            f"LLM response parsing failed: {parsing_error}"
        )
        # Score aggregation is impossible if parsing failed.
        # Still attempt final answer verification below, as it's independent.
        # Cannot aggregate scores if parsing failed
    # --- 2. Perform Final Answer Verification ---
    # This runs regardless of parsing success, as it compares extracted vs ideal.
    logger.debug("Postprocessing step 2: Verifying final answer...")
    verified, message = verify_final_answer(
        extracted_final_answer, correct_final_answer
    )
    postprocessing_results["final_answer_verified"] = verified
    postprocessing_results["verification_message"] = message

    # Optional Rule: Flag for review if verification failed. Uncomment if desired.
    # if verified is False:
    #     logger.warning("Flagging for review due to final answer mismatch.")
    #     postprocessing_results["needs_human_review"] = True
    #     if "Final answer verification failed." not in postprocessing_results["review_reasons"]:
    #          postprocessing_results["review_reasons"].append("Final answer verification failed.")

    # If parsing failed earlier, we cannot proceed to score aggregation. Return now.
    if parsing_error:
        logger.debug("Returning early due to parsing error.")
        # Ensure reasons are unique before returning
        postprocessing_results["review_reasons"] = sorted(
            list(set(postprocessing_results["review_reasons"]))
        )
        return postprocessing_results

    # --- 3. Perform Score Aggregation and Consistency Checks ---
    # This only runs if parsing was successful (i.e., no parsing_error).
    logger.debug("Postprocessing step 3: Aggregating scores...")
    # Flag for review if verification failed? Optional rule.
    # if verified is False:
    #     postprocessing_results["needs_human_review"] = True
    #     postprocessing_results["review_reasons"].append("Final answer verification failed.")

    # We checked for parsing error already, so 'evaluation' should exist if no error.
    # Add defensive check just in case.
    evaluation_content: Optional[Dict[str, Dict[str, str]]] = parsed_judge_response.get(
        "evaluation"
    )

    if isinstance(evaluation_content, dict):
        aggregation_results: Dict[str, Any] = aggregate_scores(
            evaluation_content, config
        )
        postprocessing_results["aggregated_score"] = aggregation_results.get(
            "aggregated_score"
        )

        # Combine the 'needs_human_review' flag and reasons from aggregation
        if aggregation_results.get("needs_human_review"):
            postprocessing_results["needs_human_review"] = True
            postprocessing_results["review_reasons"].extend(
                aggregation_results.get("review_reasons", [])
            )
    else:
        # This case should ideally be caught by the parser, but handle defensively
        # This case indicates an issue if reached after a successful parse.
        msg = "Postprocessing error: 'evaluation' content missing or invalid despite successful parsing."
        logger.error(msg)
        postprocessing_results["needs_human_review"] = True
        postprocessing_results["review_reasons"].append(msg)
        # Assign Fail score if aggregation cannot run due to unexpected missing content
        postprocessing_results["aggregated_score"] = "Fail"

    # --- Final Consolidation ---
    # Ensure unique review reasons and sort them for consistent output
    if postprocessing_results["review_reasons"]:
        postprocessing_results["review_reasons"] = sorted(
            list(set(postprocessing_results["review_reasons"]))
        )
    logger.debug("Postprocessing orchestration finished.")

    return postprocessing_results


# --- Example Usage & Basic Tests ---
if __name__ == "__main__":
    import json

    # Setup basic logging for testing if run directly
    logging.basicConfig(level=logging.DEBUG)
    logger.info("Running postprocessing module tests...")

    # Define dummy parser for isolated testing if response_parser isn't available
    # (though it should be if running from the correct location)
    try:
        from .response_parser import parse_judge_response
    except ImportError:
        logger.warning(
            "Could not import response_parser. Using dummy for tests. "
            "Run script from project root or ensure PYTHONPATH is set."
        )

        # Define a dummy parser if needed for basic testing without the real parser
        def parse_judge_response(resp: Any, **kwargs: Any) -> Dict[str, Any]:
            """Dummy parser for testing postprocessing logic."""
            if isinstance(resp, dict):
                # Simulate successful parse if input is dict
                return resp
            # Simulate a parsing error if input isn't a dict
            return {"error": f"Dummy parser error: Input type {type(resp).__name__}"}

    # --- Define Test Data ---
    good_parse_input_pass = {
        "evaluation": {
            "Problem Understanding": {
                "score": "Yes",
                "justification": "Clearly understood.",
            },
            "Rigor and Completeness": {
                "score": "YES",
                "justification": "Covered all steps.",
            },
        }
    }
    good_parse_input_fail = {
        "evaluation": {
            "Problem Understanding": {
                "score": "Yes",
                "justification": "Clearly understood.",
            },
            "Results Formulae": {"score": "No", "justification": "Calculation error."},
        }
    }
    good_parse_input_partial_trivial = {
        "evaluation": {
            "Problem Understanding": {
                "score": "Yes",
                "justification": "Clearly understood.",
            },
            "Assumptions": {
                "score": "Partial",
                "justification": "Missed one",
            },  # Trivial
        }
    }
    good_parse_input_fail_trivial = {
        "evaluation": {
            "Problem Understanding": {
                "score": "No",
                "justification": "Wrong",
            }  # Trivial
        }
    }
    missing_eval_content = {
        "some_other_key": "value"
    }  # Parser should catch this ideally

    # --- Run Basic Postprocessing Tests ---
    logger.info("\n--- Test Case 1: Parsing Error ---")
    bad_parse_input = {"error": "Invalid JSON format: ..."}
    results1 = perform_postprocessing(bad_parse_input, "x=5", "x=5")
    logger.info("Input: %s", bad_parse_input)
    logger.info("Result:\n%s", json.dumps(results1, indent=2))
    assert results1["needs_human_review"] is True
    assert results1["parsing_error"] is not None
    assert results1["aggregated_score"] is None
    assert (
        results1["final_answer_verified"] is True
    )  # Verification still runs even if parse fails

    logger.info("\n--- Test Case 2: All Pass, Answer Match (String) ---")
    results2 = perform_postprocessing(good_parse_input_pass, "apple", "apple")
    logger.info("Input: %s, Answers: ('apple', 'apple')", good_parse_input_pass)
    logger.info("Result:\n%s", json.dumps(results2, indent=2))
    assert results2["needs_human_review"] is False
    assert results2["parsing_error"] is None
    assert results2["aggregated_score"] == "Pass"
    assert results2["final_answer_verified"] is True
    assert "String" in results2["verification_message"]  # Ensure string method used

    logger.info("\n--- Test Case 3: One Fail, Answer Mismatch (String) ---")
    results3 = perform_postprocessing(good_parse_input_fail, "apple", "orange")
    logger.info("Input: %s, Answers: ('apple', 'orange')", good_parse_input_fail)
    logger.info("Result:\n%s", json.dumps(results3, indent=2))
    assert results3["needs_human_review"] is False
    assert results3["parsing_error"] is None
    assert results3["aggregated_score"] == "Fail"
    assert results3["final_answer_verified"] is False
    assert "String" in results3["verification_message"]

    logger.info("\n--- Test Case 4: Partial Score, Trivial Justification ---")
    results4 = perform_postprocessing(good_parse_input_partial_trivial, "A", "A")
    logger.info("Input: %s, Answers: ('A', 'A')", good_parse_input_partial_trivial)
    logger.info("Result:\n%s", json.dumps(results4, indent=2))
    assert results4["needs_human_review"] is True
    assert "Potential trivial justification" in results4["review_reasons"][0]
    assert results4["parsing_error"] is None
    assert results4["aggregated_score"] == "Partial"
    assert results4["final_answer_verified"] is True

    logger.info("\n--- Test Case 5: Fail Score, Trivial Justification ---")
    results5 = perform_postprocessing(good_parse_input_fail_trivial, "A", "B")
    logger.info("Input: %s, Answers: ('A', 'B')", good_parse_input_fail_trivial)
    logger.info("Result:\n%s", json.dumps(results5, indent=2))
    assert results5["needs_human_review"] is True
    assert "Potential trivial justification" in results5["review_reasons"][0]
    assert results5["parsing_error"] is None
    assert results5["aggregated_score"] == "Fail"
    assert results5["final_answer_verified"] is False

    logger.info("\n--- Test Case 6: Missing Evaluation Content (Defensive) ---")
    results6 = perform_postprocessing(missing_eval_content, "A", "B")
    logger.info("Input: %s, Answers: ('A', 'B')", missing_eval_content)
    logger.info("Result:\n%s", json.dumps(results6, indent=2))
    assert results6["needs_human_review"] is True
    assert "evaluation' content missing" in results6["review_reasons"][0]
    assert results6["aggregated_score"] == "Fail"  # Default to Fail

    logger.info("\n--- Test Case 7: Verification Skipped (No Correct Answer) ---")
    results7 = perform_postprocessing(good_parse_input_pass, "10", None)
    logger.info("Input: %s, Answers: ('10', None)", good_parse_input_pass)
    logger.info("Result:\n%s", json.dumps(results7, indent=2))
    assert results7["final_answer_verified"] is None
    assert "Correct answer was None" in results7["verification_message"]
    assert results7["aggregated_score"] == "Pass"  # Aggregation unaffected
    assert results7["needs_human_review"] is False

    # --- SymPy Verification Tests ---
    logger.info("\n--- Test Case 8: SymPy Verification (Match - Symbolic) ---")
    if SYMPY_AVAILABLE:
        results8 = perform_postprocessing(good_parse_input_pass, "x + y", "y + x")
        logger.info("Input: %s, Answers: ('x + y', 'y + x')", good_parse_input_pass)
        logger.info("Result:\n%s", json.dumps(results8, indent=2))
        assert results8["final_answer_verified"] is True
        assert "SymPy" in results8["verification_message"]
    else:
        logger.warning("Skipping SymPy test 8 as SymPy is not available.")

    logger.info("\n--- Test Case 9: SymPy Verification (Match - LaTeX/Numeric) ---")
    if SYMPY_AVAILABLE:
        results9 = perform_postprocessing(good_parse_input_pass, "\\frac{1}{2}", "0.5")
        logger.info(
            "Input: %s, Answers: ('\\frac{1}{2}', '0.5')", good_parse_input_pass
        )
        logger.info("Result:\n%s", json.dumps(results9, indent=2))
        assert results9["final_answer_verified"] is True
        assert "SymPy" in results9["verification_message"]
    else:
        logger.warning("Skipping SymPy test 9 as SymPy is not available.")

    logger.info("\n--- Test Case 10: SymPy Verification (Match - Different Form) ---")
    if SYMPY_AVAILABLE:
        results10 = perform_postprocessing(good_parse_input_pass, "2*x", "x*2")
        logger.info("Input: %s, Answers: ('2*x', 'x*2')", good_parse_input_pass)
        logger.info("Result:\n%s", json.dumps(results10, indent=2))
        assert results10["final_answer_verified"] is True
        assert "SymPy" in results10["verification_message"]
    else:
        logger.warning("Skipping SymPy test 10 as SymPy is not available.")

    logger.info("\n--- Test Case 11: SymPy Verification (Mismatch) ---")
    if SYMPY_AVAILABLE:
        results11 = perform_postprocessing(good_parse_input_pass, "x + y", "x - y")
        logger.info("Input: %s, Answers: ('x + y', 'x - y')", good_parse_input_pass)
        logger.info("Result:\n%s", json.dumps(results11, indent=2))
        assert results11["final_answer_verified"] is False
        assert "SymPy" in results11["verification_message"]
    else:
        logger.warning("Skipping SymPy test 11 as SymPy is not available.")

    logger.info("\n--- Test Case 12: SymPy Fallback (Parse Error) ---")
    # This should fallback to string comparison because 'invalid[syntax' cannot be parsed
    results12 = perform_postprocessing(good_parse_input_pass, "invalid[syntax", "x")
    logger.info("Input: %s, Answers: ('invalid[syntax', 'x')", good_parse_input_pass)
    logger.info("Result:\n%s", json.dumps(results12, indent=2))
    assert results12["final_answer_verified"] is False  # String comparison fails
    assert "String" in results12["verification_message"]  # Should indicate fallback

    logger.info("\n--- All Postprocessing Tests Passed (Implicitly) ---")
    logger.info("Result:\n%s", json.dumps(results12, indent=2))
    assert results12["final_answer_verified"] is False  # String comparison fails
    assert "String" in results12["verification_message"]  # Should indicate fallback

    logger.info("\n--- All Postprocessing Tests Passed (Implicitly) ---")
