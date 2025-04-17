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
from typing import Any, Dict, List, Literal, Optional, Tuple

# Setup logger for this module *before* potential logging during imports
logger = logging.getLogger("backend")


# Attempt to import sympy for mathematical comparison
try:
    # Import specific sympy components needed
    from sympy import Basic, SympifyError, sympify
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
    Safely parse a mathematical expression string using SymPy.

    Handles empty strings, standard Python math syntax (requires replacing '^' with '**'),
    and attempts to parse LaTeX format as a fallback.

    Args:
        expr_str: The mathematical expression string to parse.

    Returns:
        Parsed SymPy expression if successful, None otherwise.
    """
    if not expr_str or not expr_str.strip():
        logger.debug("SymPy parsing skipped: Empty expression string provided.")
        return None
    try:
        # Replace standard exponentiation caret '^' with Python's '**' before parsing
        # This allows parsing expressions like 'x^2' correctly.
        parsed_expr = sympify(expr_str.replace("^", "**"))
        logger.debug("Successfully parsed '%s' using sympify.", expr_str)
        return parsed_expr
    except (SympifyError, TypeError, SyntaxError) as e_std:
        logger.debug(
            "Standard SymPy parsing failed for '%s' (%s). Attempting LaTeX parse.",
            expr_str,
            e_std,
        )
        try:
            # Fallback: Attempt to parse the string as LaTeX
            parsed_expr = parse_latex(expr_str)
            logger.debug("Successfully parsed '%s' using parse_latex.", expr_str)
            return parsed_expr
        except Exception as e_latex:
            # Catches errors from parse_latex (e.g., LaTeXParsingError, TypeError)
            logger.warning(
                "SymPy parsing failed for '%s' using both standard and LaTeX methods. Error: %s",
                expr_str,
                e_latex,
            )
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


def _attempt_sympy_comparison(
    norm_extracted: str, norm_correct: str
) -> Tuple[Optional[bool], str]:
    """
    Attempts to compare two normalized answer strings using SymPy for mathematical equivalence.

    Args:
        norm_extracted: Normalized extracted answer string.
        norm_correct: Normalized correct answer string.

    Returns:
        A tuple containing:
        - Comparison result (Optional[bool]): True if equivalent, False if not,
            None if SymPy comparison could not be performed (e.g., parsing failed).
        - Comparison method (str): "SymPy" if comparison was successful, "String" if fallback needed.
    """
    comparison_method = "SymPy"
    expr1 = safe_sympy_parse(norm_extracted)
    expr2 = safe_sympy_parse(norm_correct)

    if expr1 is None or expr2 is None:
        logger.warning(
            "SymPy parsing failed for one or both answers ('%s', '%s'). Cannot perform SymPy comparison.",
            norm_extracted,
            norm_correct,
        )
        return None, "String"  # Indicate fallback needed

    try:
        # Use expr.equals(other) for robust symbolic comparison.
        # It checks for mathematical equivalence, handling different forms
        # (e.g., x+y vs y+x, 2*x vs x*2).
        # Using simplify(expr1 - expr2) == 0 is an alternative but might fail for
        # more complex symbolic cases or raise errors during simplification.
        match = expr1.equals(expr2)
        logger.debug(
            "SymPy comparison successful: '%s' vs '%s' -> Match: %s",
            norm_extracted,
            norm_correct,
            match,
        )
        return match, comparison_method
    except Exception as e:
        logger.warning(
            "SymPy .equals() comparison failed between '%s' and '%s': %s. Falling back to string comparison.",
            norm_extracted,
            norm_correct,
            e,
            exc_info=True,  # Include traceback for detailed debugging
        )
        return None, "String"  # Indicate fallback needed


def verify_final_answer(
    extracted_answer: Optional[str], correct_answer: Optional[str]
) -> Tuple[Optional[bool], str]:
    """
    Compares the extracted final answer with the correct (ideal) answer.

    Attempts mathematical equivalence checking using SymPy if available and applicable.
    If SymPy is unavailable, parsing fails, or comparison errors occur, it falls
    back to normalized string comparison.

    Args:
        extracted_answer: The final answer string extracted from the model's response.
        correct_answer: The ground-truth or ideal answer string.

    Returns:
        A tuple containing:
        - Verification result (Optional[bool]): True if match, False if mismatch,
            None if verification was skipped (e.g., missing input).
        - Verification message (str): Describes the outcome and the comparison method used.
    """
    norm_extracted = normalize_answer(extracted_answer)
    norm_correct = normalize_answer(correct_answer)

    # Handle cases where one or both answers are None after normalization
    if norm_extracted is None:
        logger.debug("Skipping verification: Extracted answer is None or empty.")
        return None, "Verification skipped: Extracted answer was None or empty."
    if norm_correct is None:
        logger.debug("Skipping verification: Correct answer is None or empty.")
        return None, "Verification skipped: Correct answer was None or empty."

    match: Optional[bool] = None
    comparison_method = "String"  # Default

    # --- Attempt SymPy Comparison ---
    if SYMPY_AVAILABLE:
        sympy_match, comparison_method = _attempt_sympy_comparison(
            norm_extracted, norm_correct
        )
        # If sympy_match is not None, SymPy comparison was successful (True or False)
        if sympy_match is not None:
            match = sympy_match
        # If sympy_match is None, comparison failed or wasn't possible, proceed to string fallback

    # --- Fallback to String Comparison ---
    if match is None:  # Only if SymPy was skipped, failed, or errored
        comparison_method = "String"  # Ensure method is set to String
        match = norm_extracted == norm_correct
        logger.debug(
            "String comparison: '%s' vs '%s' -> Match: %s",
            norm_extracted,
            norm_correct,
            match,
        )

    # --- Format Result Message ---
    # At this point, 'match' should be either True or False.
    if match is None:
        # Defensive check: This state should not be reachable if inputs were valid.
        logger.error(
            "Internal verification error: 'match' remained None after comparison attempts for '%s' vs '%s'. Defaulting to False.",
            norm_extracted,
            norm_correct,
        )
        match = False  # Default to mismatch if something unexpected happened
        message = f"Verification Error ({comparison_method}): Comparison resulted in None. (Extracted: '{norm_extracted}', Correct: '{norm_correct}')."
    else:
        message = (
            f"Verification ({comparison_method}): {'Match' if match else 'Mismatch'} "
            f"(Extracted: '{norm_extracted}', Correct: '{norm_correct}')."
        )

    return match, message


def _check_trivial_justification(
    criterion: str,
    score_norm: str,
    justification: str,
    consistency_checks: Dict[str, Any],
) -> Optional[str]:
    """
    Checks if a justification for a 'No' or 'Partial' score is potentially trivial (too short).

    Args:
        criterion: The name of the criterion being checked.
        score_norm: The normalized score ('yes', 'no', 'partial').
        justification: The justification text provided.
        consistency_checks: Configuration dictionary for consistency checks.

    Returns:
        The criterion name if its justification is flagged as potentially trivial, otherwise None.
    """
    # Use dictionary access for consistency_checks dict
    if consistency_checks.get(
        "enable_trivial_justification_check", True
    ):  # Default True if key missing
        trivial_length_threshold = consistency_checks.get(
            "trivial_justification_length_threshold",
            MIN_JUSTIFICATION_LENGTH,  # Default constant if key missing
        )
        if (
            score_norm in ["no", "partial"]
            and len(justification.strip()) <= trivial_length_threshold
        ):
            logger.warning(
                "Potential trivial justification found for criterion '%s' (Score: %s, Length: %d)",
                criterion,
                score_norm.capitalize(),  # Show original case-like score
                len(justification.strip()),
            )
            return criterion
    return None


def _determine_aggregated_score(
    scores_found: List[str], aggregation_rules: Dict[str, Any]
) -> Tuple[AggregatedScore, Optional[str]]:
    """
    Determines the overall aggregated score based on a list of normalized criterion scores.

    Args:
        scores_found: List of normalized scores ('yes', 'no', 'partial').
        aggregation_rules: Configuration dictionary for aggregation rules.

    Returns:
        A tuple containing:
        - The final aggregated score ("Pass", "Fail", "Partial").
        - An optional error message if the aggregation state was inconclusive.
    """
    error_msg: Optional[str] = None
    final_score: AggregatedScore

    # Logic: Fail if any 'no', Partial if any 'partial' (and no 'no'), Pass if all 'yes'.
    # Use dictionary access for aggregation_rules dict
    if (
        aggregation_rules.get("fail_if_any_no", True) and "no" in scores_found
    ):  # Default True if key missing
        final_score = "Fail"
    elif (
        aggregation_rules.get("partial_if_any_partial", True)
        and "partial" in scores_found
    ):  # Default True if key missing
        final_score = "Partial"
    elif aggregation_rules.get(
        "pass_if_all_yes", True
    ) and all(  # Default True if key missing
        s == "yes" for s in scores_found
    ):  # Assuming scores_found only contains 'yes', 'no', 'partial' after normalization
        final_score = "Pass"
    else:
        # This state might occur if only 'yes' scores are present but pass_if_all_yes is False,
        # or if the list is empty (though handled earlier), or contains unexpected values.
        error_msg = f"Inconclusive aggregation state. Scores found: {scores_found}. Defaulting to Fail."
        logger.error(error_msg)
        final_score = "Fail"  # Default to Fail in ambiguous cases

    return final_score, error_msg


def aggregate_scores(
    parsed_evaluation_content: Dict[str, Dict[str, str]],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Aggregates parsed rubric scores and flags potential issues for review.

    Iterates through evaluation criteria, checks for missing scores, performs
    consistency checks (like trivial justifications), determines an overall
    aggregated score based on configuration rules, and collects reasons for
    human review.

    Args:
        parsed_evaluation_content: Dictionary of criteria evaluations, typically
            from `response_parser`. Expected format:
            `{"Criterion Name": {"score": "Yes/No/Partial", "justification": "..."}}`
        config: Application configuration dictionary, expected to contain
            `aggregation_settings`.

    Returns:
        A dictionary containing aggregation results:
        - 'aggregated_score': "Pass", "Fail", or "Partial".
        - 'needs_human_review': Boolean flag.
        - 'review_reasons': List of strings explaining review flags.
    """
    # Use dictionary access for config dict
    aggregation_settings = config.get(
        "aggregation_settings", {}
    )  # Default to empty dict
    aggregation_rules = aggregation_settings.get(
        "aggregation_rules", {}
    )  # Default to empty dict
    consistency_checks = aggregation_settings.get(
        "consistency_checks", {}
    )  # Default to empty dict

    # Defaults are now handled by .get(), no need for separate checks/assignments

    results: Dict[str, Any] = {
        "aggregated_score": "Fail",  # Default to Fail
        "needs_human_review": False,
        "review_reasons": [],
    }
    scores_found: List[str] = []
    trivial_justification_criteria: List[str] = []

    if not parsed_evaluation_content:
        msg = "Aggregation failed: Parsed evaluation content is empty."
        logger.warning(msg)
        results["review_reasons"].append(msg)
        results["needs_human_review"] = True
        return results  # Return early, default score is Fail

    # --- Iterate and Validate Each Criterion ---
    for criterion, details in parsed_evaluation_content.items():
        score_raw: Optional[str] = details.get("score")  # Use Optional for safety
        justification: str = details.get("justification", "")

        # Normalize score, handle missing score
        if score_raw is None or not str(score_raw).strip():
            msg = f"Missing or empty score found for criterion '{criterion}' during aggregation."
            logger.warning(msg)
            results["review_reasons"].append(msg)
            results["needs_human_review"] = True
            scores_found.append("fail")  # Treat missing score as 'fail'
            continue
        else:
            score_norm = str(score_raw).strip().lower()
            # Basic validation against expected values (yes/no/partial)
            if score_norm not in ["yes", "no", "partial"]:
                msg = f"Unexpected score value '{score_raw}' found for criterion '{criterion}'. Treating as 'fail'."
                logger.warning(msg)
                results["review_reasons"].append(msg)
                results["needs_human_review"] = True
                scores_found.append("fail")  # Treat unexpected score as 'fail'
                continue

            scores_found.append(score_norm)

            # Check for trivial justification (pass consistency_checks dict)
            trivial_criterion = _check_trivial_justification(
                criterion, score_norm, justification, consistency_checks
            )
            if trivial_criterion:
                trivial_justification_criteria.append(trivial_criterion)
                results["needs_human_review"] = True

    # --- Post-Iteration Checks ---
    if not scores_found:
        # This case occurs if the loop ran but no valid scores were appended (e.g., all were missing/invalid)
        msg = "Aggregation failed: No valid scores were processed from the evaluation content."
        logger.error(msg)
        results["review_reasons"].append(msg)
        results["needs_human_review"] = True
        return results  # Return early, default score is Fail

    # Add reason if trivial justifications were found
    if trivial_justification_criteria:
        criteria_str = ", ".join(
            f"'{c}'" for c in sorted(list(set(trivial_justification_criteria)))
        )
        results["review_reasons"].append(
            f"Potential trivial justification for 'No'/'Partial' score in criteria: {criteria_str}."
        )

    # --- Determine Final Aggregated Score ---
    final_score, error_msg = _determine_aggregated_score(
        scores_found, aggregation_rules
    )
    results["aggregated_score"] = final_score
    if error_msg:
        results["review_reasons"].append(error_msg)
        results["needs_human_review"] = (
            True  # Flag review if aggregation was inconclusive
        )

    logger.debug(
        "Score aggregation complete. Result: %s. Needs Review: %s. Reasons: %s",
        results["aggregated_score"],
        results["needs_human_review"],
        results["review_reasons"],
    )

    return results


def _extract_final_answer_from_structured(
    structured_model_response_obj: Optional[Dict[str, Any]],
) -> Optional[str]:
    """
    Safely extracts the 'final_answer' string from a structured model response object.

    Handles cases where the input is not a dict, the 'response' key is missing
    or not a dict, or the 'final_answer' key is missing or its value is None.

    Args:
        structured_model_response_obj: The dictionary representing the structured
            output from the structuring LLM (e.g., {"response": {"final_answer": ...}}).

    Returns:
        The extracted final answer as a string, or None if it cannot be found or extracted.
    """
    if not isinstance(structured_model_response_obj, dict):
        logger.warning(
            "Cannot extract final answer: Structured response object was not provided or not a dictionary."
        )
        return None

    response_content = structured_model_response_obj.get("response")
    if not isinstance(response_content, dict):
        # Handle case where structuring might have failed and 'response' contains an error string
        # or is otherwise not the expected dictionary.
        logger.warning(
            "Cannot extract final answer: Structured response object's 'response' field is not a dictionary (found type: %s).",
            type(response_content).__name__,
        )
        return None

    final_answer_value = response_content.get("final_answer")
    if final_answer_value is None:
        logger.warning(
            "Cannot extract final answer: Structured response dictionary is missing 'final_answer' key or its value is None."
        )
        return None

    # Ensure the extracted value is returned as a string
    return str(final_answer_value)


def perform_postprocessing(
    parsed_judge_response: Dict[str, Any],
    structured_model_response_obj: Optional[Dict[str, Any]],
    correct_final_answer: Optional[str],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Orchestrates all post-processing steps for a single evaluation result.

    Coordinates the analysis of the parsed judge response and the structured
    model response. It performs:
    1. Checks for parsing errors from the judge response parser.
    2. Extracts the final answer from the structured model response.
    3. Verifies the extracted final answer against the correct answer (using SymPy or string comparison).
    4. Aggregates rubric scores and performs consistency checks (if parsing succeeded).
    5. Consolidates results, including verification status, aggregated score,
        human review flags, and reasons for review.

    Args:
        parsed_judge_response: The dictionary returned by `response_parser.parse_judge_response`.
            May contain 'error' or 'evaluation' keys.
        structured_model_response_obj: The dictionary representing the structured
            output from the structuring LLM (containing the model's final answer).
        correct_final_answer: The ground-truth or ideal final answer string.
        config: The application configuration dictionary.

    Returns:
        A dictionary summarizing the post-processing results:
        - 'final_answer_verified' (Optional[bool]): Result of answer verification.
        - 'verification_message' (str): Message describing verification outcome.
        - 'aggregated_score' (AggregatedScore): Overall score ("Pass", "Fail", "Partial").
        - 'needs_human_review' (bool): Flag indicating need for human review.
        - 'review_reasons' (List[str]): List of reasons for the review flag.
        - 'parsing_error' (Optional[str]): Error message from the parser, if any.
    """
    postprocessing_results: Dict[str, Any] = {
        "final_answer_verified": None,
        "verification_message": "Verification not performed.",
        "aggregated_score": None,  # Will be set later, default None if parsing fails
        "needs_human_review": False,
        "review_reasons": [],
        "parsing_error": None,
    }
    logger.debug("Starting postprocessing orchestration...")

    # --- 1. Check for Judge Response Parsing Errors ---
    parsing_error = parsed_judge_response.get("error")
    if parsing_error:
        logger.warning(
            "Postprocessing: Judge response parsing error detected: %s", parsing_error
        )
        postprocessing_results["parsing_error"] = parsing_error
        postprocessing_results["needs_human_review"] = True
        postprocessing_results["review_reasons"].append(
            f"LLM response parsing failed: {parsing_error}"
        )
        # Aggregation cannot proceed, but verification can still be attempted.

    # --- 2. Extract and Verify Final Answer ---
    # This runs regardless of parsing success, as it uses the structured response.
    logger.debug("Postprocessing: Extracting and verifying final answer...")
    extracted_final_answer = _extract_final_answer_from_structured(
        structured_model_response_obj
    )
    verified, message = verify_final_answer(
        extracted_final_answer, correct_final_answer
    )
    postprocessing_results["final_answer_verified"] = verified
    postprocessing_results["verification_message"] = message

    # Optional Rule: Flag for review if verification failed.
    # Consider making this configurable via `config` if needed.
    # if verified is False:
    #     logger.info("Flagging for review due to final answer mismatch.")
    #     postprocessing_results["needs_human_review"] = True
    #     postprocessing_results["review_reasons"].append("Final answer verification failed.")

    # --- 3. Perform Score Aggregation (only if parsing succeeded) ---
    if not parsing_error:
        logger.debug("Postprocessing: Aggregating scores...")
        evaluation_content = parsed_judge_response.get("evaluation")

        if isinstance(evaluation_content, dict):
            aggregation_results = aggregate_scores(evaluation_content, config)
            postprocessing_results["aggregated_score"] = aggregation_results.get(
                "aggregated_score"
            )

            # Combine review flags and reasons from aggregation
            if aggregation_results.get("needs_human_review"):
                postprocessing_results["needs_human_review"] = True
                postprocessing_results["review_reasons"].extend(
                    aggregation_results.get("review_reasons", [])
                )
        else:
            # This indicates an unexpected state: parsing succeeded according to the
            # `parsed_judge_response` structure (no 'error' key), but the 'evaluation'
            # key is missing or invalid. This suggests a potential issue in the parser logic
            # or an unexpected response format that bypassed the parser's checks.
            msg = "Postprocessing error: 'evaluation' content missing or invalid in successfully parsed judge response."
            logger.error(msg)
            postprocessing_results["needs_human_review"] = True
            postprocessing_results["review_reasons"].append(msg)
            # Assign a default score if aggregation cannot run due to this unexpected state.
            postprocessing_results["aggregated_score"] = "Fail"
    else:
        # If parsing failed, aggregation is skipped. Set aggregated_score explicitly to None.
        postprocessing_results["aggregated_score"] = None
        logger.debug("Postprocessing: Skipping score aggregation due to parsing error.")

    # --- 4. Final Consolidation ---
    # Ensure unique review reasons and sort them for consistent output
    if postprocessing_results["review_reasons"]:
        postprocessing_results["review_reasons"] = sorted(
            list(set(postprocessing_results["review_reasons"]))
        )

    logger.info(
        "Postprocessing finished. Score: %s, Verified: %s, Review Needed: %s",
        postprocessing_results["aggregated_score"],
        postprocessing_results["final_answer_verified"],
        postprocessing_results["needs_human_review"],
    )
    logger.debug("Review Reasons: %s", postprocessing_results["review_reasons"])

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
    # Dummy config for testing aggregation rules
    test_config = {
        "aggregation_settings": {
            "aggregation_rules": {
                "fail_if_any_no": True,
                "partial_if_any_partial": True,
                "pass_if_all_yes": True,
            },
            "consistency_checks": {
                "enable_trivial_justification_check": True,
                "trivial_justification_length_threshold": 10,
            },
        }
    }

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

    # Structured response examples
    structured_resp_ok = {"response": {"final_answer": "x=5"}}
    structured_resp_missing_key = {"response": {"answer": "x=5"}}
    structured_resp_not_dict = {"response": "Error during structuring"}
    structured_resp_none = None

    # --- Run Basic Postprocessing Tests ---
    logger.info("\n--- Test Case 1: Parsing Error ---")
    bad_parse_input = {"error": "Invalid JSON format: ..."}
    results1 = perform_postprocessing(
        bad_parse_input, structured_resp_ok, "x=5", test_config
    )
    logger.info("Input (Parsed Judge): %s", bad_parse_input)
    logger.info("Input (Structured Model): %s", structured_resp_ok)
    logger.info("Result:\n%s", json.dumps(results1, indent=2))
    assert results1["needs_human_review"] is True
    assert results1["parsing_error"] is not None
    assert results1["aggregated_score"] is None  # Aggregation skipped
    assert results1["final_answer_verified"] is True  # Verification still runs

    logger.info("\n--- Test Case 2: All Pass, Answer Match (String) ---")
    results2 = perform_postprocessing(
        good_parse_input_pass,
        {"response": {"final_answer": "apple"}},
        "apple",
        test_config,
    )
    logger.info("Input (Parsed Judge): %s", good_parse_input_pass)
    logger.info("Input (Structured Model): %s", {"response": {"final_answer": "apple"}})
    logger.info("Result:\n%s", json.dumps(results2, indent=2))
    assert results2["needs_human_review"] is False
    assert results2["parsing_error"] is None
    assert results2["aggregated_score"] == "Pass"
    assert results2["final_answer_verified"] is True
    assert "String" in results2["verification_message"]

    logger.info("\n--- Test Case 3: One Fail, Answer Mismatch (String) ---")
    results3 = perform_postprocessing(
        good_parse_input_fail,
        {"response": {"final_answer": "apple"}},
        "orange",
        test_config,
    )
    logger.info("Input (Parsed Judge): %s", good_parse_input_fail)
    logger.info("Input (Structured Model): %s", {"response": {"final_answer": "apple"}})
    logger.info("Result:\n%s", json.dumps(results3, indent=2))
    assert (
        results3["needs_human_review"] is False
    )  # Default: mismatch doesn't trigger review
    assert results3["parsing_error"] is None
    assert results3["aggregated_score"] == "Fail"
    assert results3["final_answer_verified"] is False
    assert "String" in results3["verification_message"]

    logger.info("\n--- Test Case 4: Partial Score, Trivial Justification ---")
    results4 = perform_postprocessing(
        good_parse_input_partial_trivial,
        {"response": {"final_answer": "A"}},
        "A",
        test_config,
    )
    logger.info("Input (Parsed Judge): %s", good_parse_input_partial_trivial)
    logger.info("Input (Structured Model): %s", {"response": {"final_answer": "A"}})
    logger.info("Result:\n%s", json.dumps(results4, indent=2))
    assert results4["needs_human_review"] is True
    assert "Potential trivial justification" in results4["review_reasons"][0]
    assert results4["parsing_error"] is None
    assert results4["aggregated_score"] == "Partial"
    assert results4["final_answer_verified"] is True

    logger.info("\n--- Test Case 5: Fail Score, Trivial Justification ---")
    results5 = perform_postprocessing(
        good_parse_input_fail_trivial,
        {"response": {"final_answer": "A"}},
        "B",
        test_config,
    )
    logger.info("Input (Parsed Judge): %s", good_parse_input_fail_trivial)
    logger.info("Input (Structured Model): %s", {"response": {"final_answer": "A"}})
    logger.info("Result:\n%s", json.dumps(results5, indent=2))
    assert results5["needs_human_review"] is True
    assert "Potential trivial justification" in results5["review_reasons"][0]
    assert results5["parsing_error"] is None
    assert results5["aggregated_score"] == "Fail"
    assert results5["final_answer_verified"] is False

    logger.info("\n--- Test Case 6: Missing Evaluation Content (Defensive) ---")
    # Simulate parser returning successfully but without 'evaluation' key
    parsed_judge_missing_eval = {"some_other_key": "value"}
    results6 = perform_postprocessing(
        parsed_judge_missing_eval, {"response": {"final_answer": "A"}}, "B", test_config
    )
    logger.info("Input (Parsed Judge): %s", parsed_judge_missing_eval)
    logger.info("Input (Structured Model): %s", {"response": {"final_answer": "A"}})
    logger.info("Result:\n%s", json.dumps(results6, indent=2))
    assert results6["needs_human_review"] is True
    assert "evaluation' content missing" in results6["review_reasons"][0]
    assert results6["aggregated_score"] == "Fail"  # Default to Fail

    logger.info("\n--- Test Case 7: Verification Skipped (No Correct Answer) ---")
    results7 = perform_postprocessing(
        good_parse_input_pass, {"response": {"final_answer": "10"}}, None, test_config
    )
    logger.info("Input (Parsed Judge): %s", good_parse_input_pass)
    logger.info("Input (Structured Model): %s", {"response": {"final_answer": "10"}})
    logger.info("Result:\n%s", json.dumps(results7, indent=2))
    assert results7["final_answer_verified"] is None
    assert "Correct answer was None" in results7["verification_message"]
    assert results7["aggregated_score"] == "Pass"
    assert results7["needs_human_review"] is False

    # --- SymPy Verification Tests ---
    logger.info("\n--- Test Case 8: SymPy Verification (Match - Symbolic) ---")
    if SYMPY_AVAILABLE:
        results8 = perform_postprocessing(
            good_parse_input_pass,
            {"response": {"final_answer": "x + y"}},
            "y + x",
            test_config,
        )
        logger.info("Input (Parsed Judge): %s", good_parse_input_pass)
        logger.info(
            "Input (Structured Model): %s", {"response": {"final_answer": "x + y"}}
        )
        logger.info("Result:\n%s", json.dumps(results8, indent=2))
        assert results8["final_answer_verified"] is True
        assert "SymPy" in results8["verification_message"]
    else:
        logger.warning("Skipping SymPy test 8 as SymPy is not available.")

    logger.info("\n--- Test Case 9: SymPy Verification (Match - LaTeX/Numeric) ---")
    if SYMPY_AVAILABLE:
        results9 = perform_postprocessing(
            good_parse_input_pass,
            {"response": {"final_answer": "\\frac{1}{2}"}},
            "0.5",
            test_config,
        )
        logger.info("Input (Parsed Judge): %s", good_parse_input_pass)
        logger.info(
            "Input (Structured Model): %s",
            {"response": {"final_answer": "\\frac{1}{2}"}},
        )
        logger.info("Result:\n%s", json.dumps(results9, indent=2))
        assert results9["final_answer_verified"] is True
        assert "SymPy" in results9["verification_message"]
    else:
        logger.warning("Skipping SymPy test 9 as SymPy is not available.")

    logger.info("\n--- Test Case 10: SymPy Verification (Match - Different Form) ---")
    if SYMPY_AVAILABLE:
        results10 = perform_postprocessing(
            good_parse_input_pass,
            {"response": {"final_answer": "2*x"}},
            "x*2",
            test_config,
        )
        logger.info("Input (Parsed Judge): %s", good_parse_input_pass)
        logger.info(
            "Input (Structured Model): %s", {"response": {"final_answer": "2*x"}}
        )
        logger.info("Result:\n%s", json.dumps(results10, indent=2))
        assert results10["final_answer_verified"] is True
        assert "SymPy" in results10["verification_message"]
    else:
        logger.warning("Skipping SymPy test 10 as SymPy is not available.")

    logger.info("\n--- Test Case 11: SymPy Verification (Mismatch) ---")
    if SYMPY_AVAILABLE:
        results11 = perform_postprocessing(
            good_parse_input_pass,
            {"response": {"final_answer": "x + y"}},
            "x - y",
            test_config,
        )
        logger.info("Input (Parsed Judge): %s", good_parse_input_pass)
        logger.info(
            "Input (Structured Model): %s", {"response": {"final_answer": "x + y"}}
        )
        logger.info("Result:\n%s", json.dumps(results11, indent=2))
        assert results11["final_answer_verified"] is False
        assert "SymPy" in results11["verification_message"]
    else:
        logger.warning("Skipping SymPy test 11 as SymPy is not available.")

    logger.info("\n--- Test Case 12: SymPy Fallback (Parse Error) ---")
    # This should fallback to string comparison because 'invalid[syntax' cannot be parsed
    results12 = perform_postprocessing(
        good_parse_input_pass,
        {"response": {"final_answer": "invalid[syntax"}},
        "x",
        test_config,
    )
    logger.info("Input (Parsed Judge): %s", good_parse_input_pass)
    logger.info(
        "Input (Structured Model): %s", {"response": {"final_answer": "invalid[syntax"}}
    )
    logger.info("Result:\n%s", json.dumps(results12, indent=2))
    assert results12["final_answer_verified"] is False  # String comparison fails
    assert "String" in results12["verification_message"]  # Should indicate fallback

    logger.info("\n--- Test Case 13: Verification with Missing Structured Answer ---")
    results13 = perform_postprocessing(
        good_parse_input_pass, structured_resp_missing_key, "x=5", test_config
    )
    logger.info("Input (Parsed Judge): %s", good_parse_input_pass)
    logger.info("Input (Structured Model): %s", structured_resp_missing_key)
    logger.info("Result:\n%s", json.dumps(results13, indent=2))
    assert results13["final_answer_verified"] is None  # Verification skipped
    assert "Extracted answer was None" in results13["verification_message"]
    assert results13["aggregated_score"] == "Pass"  # Aggregation unaffected

    logger.info(
        "\n--- Test Case 14: Verification with Non-Dict Structured Response ---"
    )
    results14 = perform_postprocessing(
        good_parse_input_pass, structured_resp_not_dict, "x=5", test_config
    )
    logger.info("Input (Parsed Judge): %s", good_parse_input_pass)
    logger.info("Input (Structured Model): %s", structured_resp_not_dict)
    logger.info("Result:\n%s", json.dumps(results14, indent=2))
    assert results14["final_answer_verified"] is None  # Verification skipped
    assert "Extracted answer was None" in results14["verification_message"]

    logger.info("\n--- All Postprocessing Tests Passed (Implicitly) ---")
