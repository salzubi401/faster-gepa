"""
LLM-as-a-Judge signature for scoring proposed prompts/instructions.

This module provides a DSPy signature that evaluates proposed instructions
based on both dataset-specific improvements and general prompt quality.
"""

import dspy
from typing import Any


class ProposalJudgeSignature(dspy.Signature):
    """You are an expert evaluator of instruction quality for AI systems. Your task is to rigorously assess proposed instructions based on two critical dimensions.

**Your Role:**
Evaluate how well a proposed instruction improves upon a current instruction, considering both its ability to address specific observed failures and its general quality as a prompt.

**Evaluation Criteria:**

1. **Dataset Alignment Score (0-50 points):**
   - How directly does the proposal address the specific failure patterns shown in the examples?
   - Does it incorporate domain-specific information from the feedback?
   - Does it provide concrete guidance that would prevent the observed errors?
   - Is it tailored to the actual failure modes rather than being generic?

   Award HIGH scores (40-50) when: Proposal directly addresses root causes, includes specific strategies from feedback, demonstrates clear understanding of failure patterns.
   Award MEDIUM scores (20-39) when: Proposal addresses some failures but misses key patterns, or is somewhat generic.
   Award LOW scores (0-19) when: Proposal ignores the failures, is too vague to help, or misunderstands the problem.

2. **General Quality Score (0-50 points):**
   - **Clarity**: Is the instruction unambiguous and easy to understand?
   - **Specificity**: Does it provide concrete, actionable guidance rather than vague advice?
   - **Completeness**: Does it cover the key aspects of the task?
   - **Best Practices**: Does it follow prompt engineering best practices (clear role, specific steps, output format, constraints)?

   Award HIGH scores (40-50) when: Instruction is crystal clear, provides specific steps/examples, comprehensive, well-structured.
   Award MEDIUM scores (20-39) when: Instruction is understandable but could be more specific or complete.
   Award LOW scores (0-19) when: Instruction is vague, ambiguous, incomplete, or poorly structured.

**Scoring Guidelines:**
- Be consistent: Similar quality proposals should receive similar scores.
- Be honest: Don't inflate scores; accurate evaluation helps improve the system.
- Be thorough: Examine the proposal carefully against each criterion.
- Calculate total_score as the exact sum of dataset_alignment_score + general_quality_score.

**Reasoning Requirements:**
Your reasoning must include:
1. Specific examples of how the proposal does/doesn't address the failures
2. Clear justification for each score component
3. Concrete suggestions for improvement (if score < 90)
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
        max_examples: int = 5
    ) -> dict[str, Any]:
        """
        Evaluate a proposed instruction.

        Args:
            current_instruction: The current instruction being improved
            proposed_instruction: The newly proposed instruction
            reflective_dataset: Dictionary of failed examples with feedback
            max_examples: Maximum examples to include in evaluation

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

        # Evaluate using the judge LM
        with dspy.context(lm=self.judge_lm):
            result = self.judge_module(
                current_instruction=current_instruction,
                proposed_instruction=proposed_instruction,
                reflective_dataset_summary=dataset_summary
            )

        return {
            "total_score": float(result.total_score),
            "dataset_alignment_score": float(result.dataset_alignment_score),
            "general_quality_score": float(result.general_quality_score),
            "reasoning": result.reasoning,
            "proposed_instruction": proposed_instruction
        }
