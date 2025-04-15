import functools
import logging
from pathlib import Path

logger = logging.getLogger("backend")


@functools.lru_cache(maxsize=32)
def load_prompt_template(template_path_str: str) -> str:
    template_path = Path(template_path_str)
    if not template_path.is_file():
        project_root = Path(__file__).resolve().parent.parent
        resolved_path = project_root / template_path_str
        if resolved_path.is_file():
            template_path = resolved_path
        else:
            logger.error(
                f"Prompt template file not found at '{template_path_str}' or '{resolved_path}'"
            )
            raise FileNotFoundError(
                f"Prompt template file not found at '{template_path_str}' or '{resolved_path}'"
            )

    with template_path.open("r", encoding="utf-8") as f:
        return f.read()


# Existing templates remain unchanged below this line
INITIAL_JUDGE_PROMPT_TEMPLATE = """
You are an expert mathematician and rigorous evaluator assessing an AI model's response to an advanced mathematics problem.
Your task is to evaluate the provided 'MODEL RESPONSE' based ONLY on the following two criteria from our evaluation rubric:

1.  **Problem Understanding:**
    *   **Yes:** The model correctly interprets the question, identifies all constraints, and sets up the problem appropriately based on the 'PROMPT'.
    *   **No:** The model misinterprets the question, misses constraints, or sets up the problem incorrectly.

2.  **Results Formulae:**
    *   **Yes:** The final answer derived by the model is mathematically equivalent to the 'CORRECT ANSWER'. Minor formatting differences are acceptable if mathematically equivalent.
    *   **No:** The final answer derived by the model is mathematically incorrect or inequivalent to the 'CORRECT ANSWER'.

You will be given the following inputs:
- **PROMPT:** The original mathematics problem.
- **MODEL RESPONSE:** The AI model's full response, including reasoning steps and its final answer.
- **IDEAL RESPONSE:** An expert-written correct solution methodology (for context, but focus your evaluation on the criteria above).
- **CORRECT ANSWER:** The ground-truth final answer.

**Instructions:**
Provide your evaluation in JSON format ONLY.
"""

FULL_L1_JUDGE_PROMPT_TEMPLATE = """
You are an expert mathematician and rigorous evaluator assessing an AI model's response to an advanced mathematics problem.
Your task is to evaluate the provided 'MODEL RESPONSE' based on ALL FIVE criteria from the L1 evaluation rubric:

1. Problem Understanding
2. Assumptions
3. Logical Implications
4. Results Formulae
5. Rigor and Completeness

Provide your evaluation in JSON format ONLY.
"""
