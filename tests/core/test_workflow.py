import unittest
from unittest.mock import patch

from CogniBench.core.workflow import run_evaluation_workflow


class TestWorkflow(unittest.TestCase):
    @patch("CogniBench.core.workflow.OpenAIClient")
    @patch("CogniBench.core.workflow.load_prompt_template")
    @patch("CogniBench.core.workflow.save_evaluation_result")
    def test_run_evaluation_workflow_success(
        self, mock_save_result, mock_load_template, mock_llm_client
    ):
        mock_load_template.return_value = "Prompt: {prompt_content}\nResponse: {model_response_text}\nIdeal: {ideal_response_text}"
        mock_llm_client.return_value.invoke.return_value = {
            "raw_content": "Evaluation: Good"
        }
        mock_save_result.return_value = {"status": "success"}

        config = {
            "llm_client": {"provider": "openai", "model": "gpt-4o"},
            "evaluation_settings": {
                "judge_model": "gpt-4o",
                "prompt_template": "prompts/full_l1_v0.2.txt",
                "expected_criteria": ["accuracy"],
                "allowed_scores": ["Good", "Bad"],
            },
        }

        result = run_evaluation_workflow(
            prompt="What is 2+2?",
            response="The answer is 4.",
            ideal_response="4",
            config=config,
            task_id="test_task",
            model_id="test_model",
        )

        self.assertEqual(result["status"], "success")
        mock_save_result.assert_called_once()

    @patch("CogniBench.core.workflow.load_prompt_template")
    def test_run_evaluation_workflow_template_failure(self, mock_load_template):
        mock_load_template.return_value = None

        config = {
            "llm_client": {"provider": "openai", "model": "gpt-4o"},
            "evaluation_settings": {
                "judge_model": "gpt-4o",
                "prompt_template": "invalid/path.txt",
                "expected_criteria": ["accuracy"],
                "allowed_scores": ["Good", "Bad"],
            },
        }

        result = run_evaluation_workflow(
            prompt="What is 2+2?",
            response="The answer is 4.",
            ideal_response="4",
            config=config,
        )

        self.assertEqual(result["status"], "error")
        self.assertIn("Failed to load prompt template", result["message"])


if __name__ == "__main__":
    unittest.main()
