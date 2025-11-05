"""
LLM-as-a-Judge signature for scoring proposed prompts/instructions.

This module provides a DSPy signature that evaluates proposed instructions
based on both dataset-specific improvements and general prompt quality.
"""

import dspy
from typing import Any


class ProposalJudgeSignature(dspy.Signature):
    """You are an expert evaluator of instruction quality for AI systems. Your task is to rigorously and DISCERNINGLY assess proposed instructions. Use the FULL scoring range (0-50) to create clear differentiation between proposals.

**Your Role:**
Evaluate how well a proposed instruction improves upon a current instruction. Be STRICT and DISCRIMINATIVE—most proposals should score in the 20-40 range. Only truly exceptional proposals merit 45-50. Penalize generic, obvious, or safe solutions that don't add real value.

**Evaluation Criteria:**

1. **Dataset Alignment Score (0-50 points):**
   - How directly does the proposal address the SPECIFIC failure patterns shown in the examples?
   - Does it incorporate domain-specific information from the feedback?
   - Does it provide concrete, actionable guidance that would prevent the observed errors?
   - Is it tailored to the actual failure modes rather than being generic?
   - **IMPORTANT**: Does it address edge cases or specific failure modes that require innovative thinking?

   Award EXCELLENT scores (45-50) ONLY when: Proposal directly addresses root causes with SPECIFIC strategies, includes domain-specific insights from feedback, demonstrates deep understanding of failure patterns, and goes beyond generic advice.
   Award GOOD scores (35-44) when: Proposal addresses failures well but is somewhat predictable or lacks innovation.
   Award MEDIUM scores (20-34) when: Proposal addresses some failures but misses key patterns, is generic, or only provides obvious improvements.
   Award LOW scores (0-19) when: Proposal ignores failures, is too vague, misunderstands the problem, or is completely generic.

   **PENALIZE**: Generic advice like "be more careful", "read the problem", or "double-check your work" unless backed by SPECIFIC guidance.

2. **General Quality Score (0-50 points):**
   - **Clarity**: Is the instruction unambiguous and easy to understand?
   - **Specificity**: Does it provide concrete, actionable guidance rather than vague advice?
   - **Completeness**: Does it cover the key aspects of the task comprehensively?
   - **Best Practices**: Does it follow prompt engineering best practices (clear role, specific steps, output format, constraints)?
   - **Innovation**: Does it introduce novel approaches, structures, or insights beyond standard templates?

   Award EXCELLENT scores (45-50) ONLY when: Instruction is crystal clear, provides SPECIFIC steps/examples, comprehensive, well-structured, AND introduces innovative elements that would genuinely improve outcomes.
   Award GOOD scores (35-44) when: Instruction is clear and well-structured but somewhat conventional.
   Award MEDIUM scores (20-34) when: Instruction is understandable but generic, lacks specificity, or misses key elements.
   Award LOW scores (0-19) when: Instruction is vague, ambiguous, incomplete, poorly structured, or unhelpful.

**Scoring Guidelines (CRITICAL):**
- Use the FULL range: Don't cluster scores. Create clear differentiation.
- Be STRICT: Most proposals should score 25-40. Only truly exceptional ones get 45-50.
- Penalize GENERIC solutions: If the proposal could apply to any problem, penalize it (-5 to -10 points).
- Reward INNOVATION: If the proposal introduces novel approaches or addresses edge cases, reward it (+3 to +5 points).
- Be consistent within reason: Similar quality proposals should receive similar scores, but don't be afraid to differentiate.
- Be honest: Don't inflate scores; accurate evaluation helps improve the system.
- Calculate total_score as the exact sum: dataset_alignment_score + general_quality_score.

**Comparison with Other Proposals (if provided):**
If other_proposals_summary is provided, consider:
- Is this proposal more innovative than others?
- Does it address different aspects or use different approaches?
- Reward uniqueness: Proposals that take genuinely different approaches get +2 to +5 bonus points (reflect in dataset_alignment_score).
- Penalize redundancy: If this proposal is too similar to others, it's less valuable.

**Reasoning Requirements:**
Your reasoning must include:
1. Specific examples of how the proposal does/doesn't address the failures
2. Clear justification for each score component (why this score, not higher/lower)
3. What makes this proposal innovative or generic (be specific)
4. Concrete suggestions for improvement (if score < 85)
5. Comparison with other proposals if provided
"""

    # Inputs
    current_instruction: str = dspy.InputField(
        desc="The current instruction that is being improved"
    )

    proposed_instruction: str = dspy.InputField(
        desc="The newly proposed instruction to evaluate"
    )

    reflective_dataset_summary: str = dspy.InputField(
        desc=(
            "Summary of failed examples with feedback, formatted as: "
            "Example N: [inputs] -> [outputs] | Feedback: [what went wrong]"
        )
    )

    other_proposals_summary: str = dspy.InputField(
        desc=(
            "Optional: Summary of other proposals being evaluated in the same batch. "
            "Use this to assess uniqueness and innovation. Format: Brief descriptions of "
            "other proposals' key approaches. Leave empty string if not available."
        )
    )

    # Outputs
    dataset_alignment_score: float = dspy.OutputField(
        desc=(
            "Score (0-50) for how well the proposed instruction addresses "
            "the specific failures and patterns in the reflective dataset"
        )
    )

    general_quality_score: float = dspy.OutputField(
        desc=(
            "Score (0-50) for general prompt quality: clarity, specificity, "
            "actionability, completeness, and adherence to best practices"
        )
    )

    total_score: float = dspy.OutputField(
        desc="Total score (0-100) = dataset_alignment_score + general_quality_score"
    )

    reasoning: str = dspy.OutputField(
        desc=(
            "Detailed reasoning explaining the scores, including: "
            "(1) What specific failures the proposal addresses, "
            "(2) What prompt quality aspects are strong/weak, "
            "(3) Suggestions for further improvement"
        )
    )


def format_reflective_dataset_for_judge(
    reflective_dataset: dict[str, list[dict[str, Any]]],
    max_examples: int = 5
) -> str:
    """
    Format the reflective dataset into a concise summary for the judge.

    Args:
        reflective_dataset: Dictionary mapping component names to lists of failed examples
        max_examples: Maximum number of examples to include per component

    Returns:
        Formatted string summary of the reflective dataset
    """
    summary_parts = []

    for component_name, examples in reflective_dataset.items():
        summary_parts.append(f"\n## Component: {component_name}\n")

        for idx, example in enumerate(examples[:max_examples], 1):
            # Extract key information
            inputs = example.get("Inputs", {})
            outputs = example.get("Generated_Outputs", "N/A")
            feedback = example.get("Feedback", "No feedback provided")

            # Format inputs concisely
            if isinstance(inputs, dict):
                input_str = ", ".join(f"{k}={v}" for k, v in inputs.items())
            else:
                input_str = str(inputs)

            # Format outputs concisely
            if isinstance(outputs, dict):
                output_str = ", ".join(f"{k}={v}" for k, v in outputs.items())
            else:
                output_str = str(outputs)[:200]  # Truncate long outputs

            summary_parts.append(
                f"Example {idx}:\n"
                f"  Inputs: {input_str}\n"
                f"  Outputs: {output_str}\n"
                f"  Feedback: {feedback}\n"
            )

        if len(examples) > max_examples:
            summary_parts.append(f"  ... and {len(examples) - max_examples} more examples\n")

    return "\n".join(summary_parts)


class ProposalJudge:
    """
    Wrapper class for evaluating proposed instructions using LLM-as-a-judge.
    """

    def __init__(self, judge_lm: dspy.LM):
        """
        Initialize the proposal judge.

        Args:
            judge_lm: The language model to use for judging
        """
        self.judge_lm = judge_lm
        self.judge_module = dspy.ChainOfThought(ProposalJudgeSignature)

    def evaluate(
        self,
        current_instruction: str,
        proposed_instruction: str,
        reflective_dataset: dict[str, list[dict[str, Any]]],
        max_examples: int = 5,
        other_proposals: list[str] = None
    ) -> dict[str, Any]:
        """
        Evaluate a proposed instruction.

        Args:
            current_instruction: The current instruction being improved
            proposed_instruction: The newly proposed instruction
            reflective_dataset: Dictionary of failed examples with feedback
            max_examples: Maximum examples to include in evaluation
            other_proposals: Optional list of other proposals in the batch for comparison

        Returns:
            Dictionary containing:
                - total_score: float (0-100)
                - dataset_alignment_score: float (0-50)
                - general_quality_score: float (0-50)
                - reasoning: str
        """
        # Format the reflective dataset
        dataset_summary = format_reflective_dataset_for_judge(
            reflective_dataset, max_examples
        )

        # Format other proposals for comparison if provided
        other_proposals_summary = ""
        if other_proposals:
            other_proposals_summary = "\n".join([
                f"Proposal {i+1}: {prop[:300]}..." if len(prop) > 300 else f"Proposal {i+1}: {prop}"
                for i, prop in enumerate(other_proposals)
            ])

        # Evaluate using the judge LM
        with dspy.context(lm=self.judge_lm):
            result = self.judge_module(
                current_instruction=current_instruction,
                proposed_instruction=proposed_instruction,
                reflective_dataset_summary=dataset_summary,
                other_proposals_summary=other_proposals_summary
            )

        return {
            "total_score": float(result.total_score),
            "dataset_alignment_score": float(result.dataset_alignment_score),
            "general_quality_score": float(result.general_quality_score),
            "reasoning": result.reasoning,
            "proposed_instruction": proposed_instruction
        }
