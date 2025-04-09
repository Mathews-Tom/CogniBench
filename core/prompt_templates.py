# CogniBench - Prompt Templates
# Version: 0.1 (Phase 1 - Initial Criteria)

# This template focuses on evaluating 'Problem Understanding' and 'Results/Formulae' only.

INITIAL_JUDGE_PROMPT_TEMPLATE = """
You are an expert mathematician and rigorous evaluator assessing an AI model's response to an advanced mathematics problem.
Your task is to evaluate the provided 'MODEL RESPONSE' based ONLY on the following two criteria from our evaluation rubric:

1.  **Problem Understanding:**
    *   **Yes:** The model correctly interprets the question, identifies all constraints, and sets up the problem appropriately based on the 'PROMPT'.
    *   **No:** The model misinterprets the question, misses constraints, or sets up the problem incorrectly.

2.  **Results/Formulae:**
    *   **Yes:** The final answer derived by the model is mathematically equivalent to the 'CORRECT ANSWER'. Minor formatting differences are acceptable if mathematically equivalent.
    *   **No:** The final answer derived by the model is mathematically incorrect or inequivalent to the 'CORRECT ANSWER'.

You will be given the following inputs:
- **PROMPT:** The original mathematics problem.
- **MODEL RESPONSE:** The AI model's full response, including reasoning steps and its final answer.
- **IDEAL RESPONSE:** An expert-written correct solution methodology (for context, but focus your evaluation on the criteria above).
- **CORRECT ANSWER:** The ground-truth final answer.

**Instructions:**

1.  Carefully read the PROMPT and the MODEL RESPONSE.
2.  Compare the MODEL RESPONSE's interpretation and setup against the PROMPT to evaluate **Problem Understanding**.
3.  Identify the final answer within the MODEL RESPONSE. Compare this final answer against the provided **CORRECT ANSWER** to evaluate **Results/Formulae**.
4.  Provide your evaluation in the following JSON format ONLY. Do not include any text outside the JSON structure.

```json
{{
  "evaluation": {{
    "Problem Understanding": {{
      "score": "Yes" or "No",
      "justification": "Your detailed reasoning for the score, citing specific parts of the MODEL RESPONSE and PROMPT."
    }},
    "Results/Formulae": {{
      "score": "Yes" or "No",
      "justification": "Your detailed reasoning for the score, comparing the model's final answer to the CORRECT ANSWER."
    }}
  }}
}}
```

**Input Data:**

**PROMPT:**
{prompt_content}

**MODEL RESPONSE:**
{model_response_text}

**IDEAL RESPONSE:**
{ideal_response_text}

**CORRECT ANSWER:**
{correct_answer}

**Evaluation Output (JSON only):**
"""

# Example usage (for testing):
# filled_prompt = INITIAL_JUDGE_PROMPT_TEMPLATE.format(
#     prompt_content="Example prompt",
#     model_response_text="Example model response",
#     ideal_response_text="Example ideal response",
#     correct_answer="Example correct answer"
# )
# print(filled_prompt)
