# CogniBench/tests/core/test_postprocessing.py


from CogniBench.core.postprocessing import perform_postprocessing

# --- Test Data Fixtures ---


# Simulates output from a successful parse_judge_response call
def mock_parsed_response(evaluation_content: dict) -> dict:
    return {"evaluation": evaluation_content}


# Simulates output from a failed parse_judge_response call
def mock_parse_error(error_message: str = "Simulated parsing error") -> dict:
    return {"error": error_message}


# --- Test Cases for perform_postprocessing ---


def test_postprocessing_parsing_error():
    """Test that a parsing error flags for human review."""
    parsed_input = mock_parse_error("Invalid JSON")
    result = perform_postprocessing(parsed_input, "x=1", "x=1")

    assert result["needs_human_review"] is True
    assert "LLM response parsing failed" in result["review_reasons"][0]
    assert result["parsing_error"] == "Invalid JSON"
    assert result["aggregated_score"] is None  # Aggregation shouldn't run
    assert result["final_answer_verified"] is True  # Verification should still run


def test_postprocessing_all_pass_no_review():
    """Test a clean run with all 'Yes' scores and matching answer."""
    evaluation = {
        "Problem Understanding": {
            "score": "Yes",
            "justification": "Understood the problem well.",
        },
        "Logical Implications": {
            "score": "YES",
            "justification": "Steps were logical and correct.",
        },
    }
    parsed_input = mock_parsed_response(evaluation)
    result = perform_postprocessing(parsed_input, "y=2", "y=2")

    assert result["needs_human_review"] is False
    assert not result["review_reasons"]
    assert result["parsing_error"] is None
    assert result["aggregated_score"] == "Pass"
    assert result["final_answer_verified"] is True


def test_postprocessing_fail_score_no_review_needed():
    """Test that a 'No' score with sufficient justification doesn't flag for review."""
    evaluation = {
        "Problem Understanding": {
            "score": "Yes",
            "justification": "Understood the problem well.",
        },
        "Results Formulae": {
            "score": "No",
            "justification": "The final calculation had an error in step 3.",
        },  # Non-trivial
    }
    parsed_input = mock_parsed_response(evaluation)
    result = perform_postprocessing(parsed_input, "100", "101")  # Answer mismatch

    assert (
        result["needs_human_review"] is False
    )  # Default behavior: Fail score alone doesn't trigger review
    assert not result["review_reasons"]
    assert result["parsing_error"] is None
    assert result["aggregated_score"] == "Fail"
    assert result["final_answer_verified"] is False


def test_postprocessing_fail_score_trivial_justification():
    """Test that a 'No' score with trivial justification flags for review."""
    evaluation = {
        "Problem Understanding": {
            "score": "Yes",
            "justification": "Understood the problem well.",
        },
        "Results Formulae": {"score": "No", "justification": "Wrong."},  # Trivial
    }
    parsed_input = mock_parsed_response(evaluation)
    result = perform_postprocessing(parsed_input, "100", "101")

    assert result["needs_human_review"] is True
    assert "Potential trivial justification" in result["review_reasons"][0]
    assert "Results Formulae" in result["review_reasons"][0]
    assert result["parsing_error"] is None
    assert result["aggregated_score"] == "Fail"
    assert result["final_answer_verified"] is False


def test_postprocessing_partial_score_trivial_justification():
    """Test that a 'Partial' score with trivial justification flags for review."""
    evaluation = {
        "Problem Understanding": {
            "score": "Yes",
            "justification": "Understood the core idea.",
        },
        "Rigor and Completeness": {
            "score": "Partial",
            "justification": "Short",
        },  # Trivial
    }
    parsed_input = mock_parsed_response(evaluation)
    result = perform_postprocessing(parsed_input, "A", "A")

    assert result["needs_human_review"] is True
    assert "Potential trivial justification" in result["review_reasons"][0]
    assert "Rigor and Completeness" in result["review_reasons"][0]
    assert result["parsing_error"] is None
    assert result["aggregated_score"] == "Partial"
    assert result["final_answer_verified"] is True


def test_postprocessing_partial_score_sufficient_justification():
    """Test that a 'Partial' score with sufficient justification doesn't flag for review."""
    evaluation = {
        "Problem Understanding": {
            "score": "Yes",
            "justification": "Understood the core idea.",
        },
        "Rigor and Completeness": {
            "score": "Partial",
            "justification": "Missed addressing the edge case mentioned in the prompt.",
        },  # Non-trivial
    }
    parsed_input = mock_parsed_response(evaluation)
    result = perform_postprocessing(parsed_input, "A", "A")

    assert (
        result["needs_human_review"] is False
    )  # Partial score alone doesn't trigger review by default
    assert not result["review_reasons"]
    assert result["parsing_error"] is None
    assert result["aggregated_score"] == "Partial"
    assert result["final_answer_verified"] is True


def test_postprocessing_missing_evaluation_content():
    """Test defensive handling if parser somehow returned success without 'evaluation'."""
    parsed_input = {"some_other_key": "value"}  # Should not happen with current parser
    result = perform_postprocessing(parsed_input, "A", "B")

    assert result["needs_human_review"] is True
    assert "evaluation' content missing" in result["review_reasons"][0]
    assert result["parsing_error"] is None
    assert result["aggregated_score"] == "Fail"  # Defaults to Fail
    assert result["final_answer_verified"] is False


def test_postprocessing_verification_skipped_no_review():
    """Test that skipped verification (None answer) doesn't trigger review."""
    evaluation = {
        "Problem Understanding": {
            "score": "Yes",
            "justification": "Understood the problem well.",
        }
    }
    parsed_input = mock_parsed_response(evaluation)
    result = perform_postprocessing(
        parsed_input, "Valid Extract", None
    )  # Correct answer is None

    assert result["needs_human_review"] is False
    assert not result["review_reasons"]
    assert result["parsing_error"] is None
    assert result["aggregated_score"] == "Pass"
    assert result["final_answer_verified"] is None  # Verification skipped
    assert "Correct answer was None" in result["verification_message"]


def test_postprocessing_empty_evaluation_content():
    """Test handling of an empty evaluation dictionary."""
    evaluation = {}
    parsed_input = mock_parsed_response(evaluation)
    result = perform_postprocessing(parsed_input, "A", "A")

    assert result["needs_human_review"] is True
    assert (
        "Aggregation failed: No evaluation content provided."
        in result["review_reasons"]
    )
    assert result["parsing_error"] is None
    assert result["aggregated_score"] is None  # Cannot aggregate
    assert result["final_answer_verified"] is True


def test_postprocessing_missing_score_in_criterion():
    """Test handling if a criterion dict is missing the 'score' key (should be caught by parser ideally)."""
    evaluation = {
        "Problem Understanding": {"justification": "Seems ok?"}  # Missing score
    }
    parsed_input = mock_parsed_response(evaluation)
    result = perform_postprocessing(parsed_input, "A", "A")

    assert result["needs_human_review"] is True
    assert (
        "Missing or empty score for criterion 'Problem Understanding'."
        in result["review_reasons"]
    )
    assert result["parsing_error"] is None
    # Aggregated score becomes Fail because a criterion had an issue
    assert result["aggregated_score"] == "Fail"
    assert result["final_answer_verified"] is True
