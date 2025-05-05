# Prompt Checklist

## ✅ Math Solution Structuring Checklist (Question-Aware)

### 📘 SECTION 1: Alignment with the Problem Statement

* [ ] **Problem-Relevant Structuring**: Does the structured output match what the question asks for (e.g., equation solving vs. sequence pattern vs. geometric proof)?
* [ ] **Assumptions Derived from the Question**: Are initial conditions and domain restrictions explicitly included if stated or clearly implied by the question?

---

### ⚙️ SECTION 2: Structuring Accuracy

* [ ] **Steps Are Logically Ordered**: Are the `"steps"` in a clear, step-by-step sequence that mirrors the actual logical flow?

  * 🔸 Red flag: Steps are jumbled, missing transitions, or overly compressed.
* [ ] **Intermediate Results Reflect Real Calculations**: Are `"intermediate_results"` used to reflect actual outputs (e.g., derived values, transformed expressions)?
* [ ] **Final Answer Matches Reasoning**: Is `"final_answer"` accurate and consistent with the preceding steps?

---

### 🧠 SECTION 3: Reasoning and Assumption Fidelity

* [ ] **Assumptions Field Captures Inferred Constraints**:

  * e.g., If the model used "n is an integer" or "x > 0", is this explicitly recorded?
* [ ] **No Invented Justifications**: The structuring should not add assumptions or results that aren’t actually stated in the original solution.

---

### 🧾 SECTION 4: Format Conformance

* [ ] **All Required Fields Present**:

  * `"assumptions"`: String (empty if none)
  * `"steps"`: Array of ordered strings
  * `"intermediate_results"`: Array (empty if none)
  * `"final_answer"`: String
  * `"format_notes"`: String (can be empty)
* [ ] **Format Notes Captured**: If the original solution emphasizes the final answer (e.g., boxed, bold, LaTeX), is this noted under `"format_notes"`?

---

### 🔍 SECTION 5: Semantic Consistency

* [ ] **Each Step Faithfully Mirrors Original Reasoning**: Does the structured output retain the **intent and logic** of the original model or expert response?
* [ ] **Avoids Over-Summarization**: Structuring shouldn't compress multi-line reasoning into single steps when doing so obscures logical transitions.

---

### 🚦 SECTION 6: Use of the Question

* [ ] **Does the structuring make sense *in context of the original question*?**

  * Would the structure still look valid if you saw only the JSON and the original prompt?
* [ ] **Does the final answer and reasoning correspond to what the question actually asked for?**

---

### 🧠 Example Red Flags (Structuring Errors)

| Problem Type          | Common Structuring Mistake                                             |
| --------------------- | ---------------------------------------------------------------------- |
| Pattern sequence      | Skips pattern explanation in steps                                     |
| Equation solving      | No mention of equation or variable in assumptions                      |
| Proofs or derivations | No use of logical transitions (e.g., "By definition", "Using theorem") |
| Geometry              | No mention of given constraints or diagram assumptions                 |

---

## ✅ Math Model Judging Checklist (with Question Awareness)

### 📘 SECTION 1: Basic Alignment with the Question

* [ ] **Objective Match**: Does the model answer the *type* of question asked (e.g., numeric prediction, proof, simplification)?
* [ ] **Data Usage**: Did the model use *all relevant information* from the question (e.g., initial sequence terms, given constraints)?
* [ ] **Final Answer Relevance**: Is the model's final answer responsive to what the question asks for?

---

### ⚙️ SECTION 2: Method Appropriateness

* [ ] **Appropriate Method Choice**: Given the question format (e.g., number sequence, geometry, algebra), did the model choose a natural or efficient strategy?

  * 🔸 Red flag: Uses high-degree polynomial fitting for a 5-term integer sequence.
* [ ] **Assumption Justifiability**: Are the model’s assumptions defensible based on the question alone?

  * e.g., Assuming a recurrence? That’s valid. Assuming a polynomial fit? Needs justification.

---

### 🧠 SECTION 3: Problem Understanding

* [ ] **Correct Identification of What’s Asked**: Does the model clearly grasp *what needs to be solved*?

  * e.g., If the question is “simplify,” is the model trying to “solve” instead?
* [ ] **Constraints Incorporated**: Did the model respect or explicitly handle any constraints in the question (e.g., integer domain, non-negative inputs)?

---

### 🔍 SECTION 4: Comparative Evaluation (vs. Ideal)

* [ ] **Did the model use a different approach than the ideal?**

  * If yes:

    * [ ] Was the model’s method mathematically valid?
    * [ ] Was it equally or more appropriate *given the question*?
* [ ] **Does the model’s method generalize well if extended to similar questions?**

---

### 🧾 SECTION 5: Rubric Mapping

When scoring rubric items (like “problem understanding” or “assumptions”), ask:

* ✅ Would this be judged the same way **if the question were different**?
* 🔍 If **no**, then you **must consider the question** to be a deciding factor.
