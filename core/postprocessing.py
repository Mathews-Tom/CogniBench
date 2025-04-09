# CogniBench - Postprocessing Module
# Version: 0.1 (Phase 2 - Initial Answer Verification)

from typing import Optional, Dict, Any, List

def normalize_answer(answer: Optional[str]) -> Optional[str]:
    """
    Performs basic normalization on an answer string for comparison.
    (Lowercase, strip whitespace).
    More sophisticated normalization (e.g., for mathematical expressions)
    can be added later.
    """
    if answer is None:
        return None
    return answer.lower().strip()

def verify_final_answer(extracted_answer: Optional[str], correct_answer: Optional[str]) -> bool:
    """
    Compares the extracted final answer with the correct answer after normalization.

    Args:
        extracted_answer: The answer string extracted from the model response.
        correct_answer: The ground-truth answer string.

    Returns:
        True if the normalized answers match, False otherwise.
    """
    norm_extracted = normalize_answer(extracted_answer)
    norm_correct = normalize_answer(correct_answer)

    if norm_extracted is None:
        print("Verification Info: Extracted answer was None.")
        return False # Cannot verify if nothing was extracted

    if norm_correct is None:
        print("Verification Info: Correct answer was None.")
        # This might indicate a data issue, but technically they don't match
        return False

    print(f"Comparing Normalized Answers: '{norm_extracted}' vs '{norm_correct}'")
    return norm_extracted == norm_correct

# Example usage (for testing):
# if __name__ == "__main__":
#     print(f"Match 1: {verify_final_answer('x = 5', ' x = 5 ')}")
#     print(f"Match 2: {verify_final_answer('pi^2 - 4', 'PI^2 - 4')}")
#     print(f"Mismatch 1: {verify_final_answer('x=5', 'x=6')}")
#     print(f"Mismatch 2: {verify_final_answer(None, 'x=5')}")
#     print(f"Mismatch 3: {verify_final_answer('x=5', None)}")
#

def aggregate_scores(parsed_rubric_scores: Optional[Dict[str, Any]]) -> Optional[str]:
    """
    Aggregates the parsed rubric scores into an overall assessment.
    Initial simple rule: Pass if all scores are 'Yes', otherwise Fail.

    Args:
        parsed_rubric_scores: The dictionary containing the parsed scores
                              (e.g., {"Problem Understanding": {"score": "Yes", ...}, ...}).

    Returns:
        "Pass" or "Fail" string, or None if input is invalid or empty.
    """
    if not parsed_rubric_scores or not isinstance(parsed_rubric_scores, dict):
        print("Aggregation Info: Invalid or empty rubric scores provided.")
        return None

    all_yes = True
    found_scores = False
    for criterion, details in parsed_rubric_scores.items():
        if isinstance(details, dict) and "score" in details:
            found_scores = True
            score = str(details["score"]).strip().lower()
            if score != "yes":
                all_yes = False
                # We can break early if we find a non-'Yes' score
                # However, iterating through all ensures we process valid structures
                # break # Optional optimization
        else:
            # Handle unexpected structure within a criterion
            print(f"Aggregation Warning: Invalid structure for criterion '{criterion}'. Skipping.")
            all_yes = False # Treat malformed entries as non-passing

    if not found_scores:
        print("Aggregation Info: No valid scores found in the rubric data.")
        return None # Or perhaps "Fail" depending on desired handling

    return "Pass" if all_yes else "Fail"

# Example usage (for testing):
# if __name__ == "__main__":
#     # ... (previous examples) ...
#     print("\n--- Score Aggregation ---")
#     scores_pass = {
#         "Problem Understanding": {"score": "Yes", "justification": "..."},
#         "Results/Formulae": {"score": "Yes", "justification": "..."}
#     }
#     scores_fail_1 = {
#         "Problem Understanding": {"score": "Yes", "justification": "..."},
#         "Results/Formulae": {"score": "No", "justification": "..."}
#     }
#     scores_fail_2 = {
#         "Problem Understanding": {"score": "YES", "justification": "..."}, # Case-insensitivity handled
#         "Results/Formulae": {"score": "no", "justification": "..."}
#     }
#     scores_malformed = {
#         "Problem Understanding": {"score": "Yes"}, # Missing justification, but score is present
#         "Results/Formulae": "Fail" # Invalid structure
#     }
#     scores_empty = {}
#
#     print(f"Scores Pass: {aggregate_scores(scores_pass)}")
#     print(f"Scores Fail 1: {aggregate_scores(scores_fail_1)}")
#     print(f"Scores Fail 2: {aggregate_scores(scores_fail_2)}")
#     print(f"Scores Malformed: {aggregate_scores(scores_malformed)}")
#     print(f"Scores Empty: {aggregate_scores(scores_empty)}")
#     print(f"Scores None: {aggregate_scores(None)}")