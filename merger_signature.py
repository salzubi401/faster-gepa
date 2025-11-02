"""
Prompt merger signature for synthesizing multiple high-quality proposals.

This module provides a DSPy signature that takes multiple top-scoring proposed
instructions and merges them into a single optimized instruction by extracting
the best elements from each.
"""

import dspy
from typing import Any


class ProposalMergerSignature(dspy.Signature):
    """You are an expert at synthesizing multiple high-quality ideas into a single superior solution. Your task is to merge the best proposals into one optimized instruction that surpasses any individual proposal.

**Your Role:**
Analyze multiple top-scoring instruction proposals and create a merged instruction that combines their strengths while addressing all observed failure patterns.

**Merging Process - Follow These Steps:**

1. **Analyze Each Proposal:**
   - Identify the unique strengths of each proposal
   - Note which specific failures each addresses
   - Recognize innovative strategies or domain knowledge each contains
   - Understand the judge's reasoning for their scores

2. **Find Synergies:**
   - Identify complementary ideas that can be combined
   - Look for proposals that address different aspects of the problem
   - Find reinforcing strategies that work well together

3. **Resolve Conflicts:**
   - When proposals contradict, choose the approach that:
     * Better addresses the failure patterns
     * Is more specific and actionable
     * Has clearer reasoning from the judge
   - Don't just pick the highest-scoring proposal—consider all perspectives

4. **Eliminate Redundancy:**
   - Combine similar suggestions into single, stronger statements
   - Avoid repetitive guidance
   - Keep the instruction concise while comprehensive

5. **Synthesize:**
   - Create a coherent, well-structured instruction
   - Ensure logical flow and clarity
   - Maintain specific, actionable guidance throughout
   - Verify all key failure patterns are addressed

**Quality Criteria for Merged Result:**
- ✓ Incorporates the best insights from all proposals
- ✓ More comprehensive than any single proposal
- ✓ Addresses all failure patterns from the dataset
- ✓ Clear, specific, and immediately actionable
- ✓ Well-structured with logical organization
- ✓ No contradictions or redundancy

**Output Requirements:**
- **merged_instruction**: The final synthesized instruction (ready to use)
- **rationale**: Explain your merging decisions—what you took from each proposal, how you resolved conflicts, and what synergies you discovered
- **improvements_over_current**: Concrete examples of how the merged instruction fixes the specific failures observed in the dataset

**Important:**
The merged instruction should be genuinely better than any individual proposal. Don't just concatenate them—synthesize new value by combining their strengths intelligently.
"""

    # Inputs
    current_instruction: str = dspy.InputField(
        desc="The current instruction that is being improved"
    )

    top_proposals: str = dspy.InputField(
        desc=(
            "The top-n scoring proposed instructions, formatted as:\n"
            "Proposal 1 (Score: X.X):\n[instruction text]\nReasoning: [judge's reasoning]\n\n"
            "Proposal 2 (Score: Y.Y):\n[instruction text]\nReasoning: [judge's reasoning]\n"
            "etc."
        )
    )

    reflective_dataset_summary: str = dspy.InputField(
        desc=(
            "Summary of failed examples that the new instruction should address. "
            "Use this context to ensure the merged instruction addresses key failure patterns."
        )
    )

    # Outputs
    merged_instruction: str = dspy.OutputField(
        desc=(
            "The final merged instruction that combines the best elements of all proposals. "
            "Should be clear, specific, actionable, and address the failures in the dataset."
        )
    )

    rationale: str = dspy.OutputField(
        desc=(
            "Detailed explanation of the merging process, including:\n"
            "1. What elements were taken from each proposal and why\n"
            "2. How conflicts or contradictions were resolved\n"
            "3. What new insights emerged from combining the proposals\n"
            "4. Why this merged instruction is better than any single proposal"
        )
    )

    improvements_over_current: str = dspy.OutputField(
        desc=(
            "Specific improvements the merged instruction makes over the current instruction, "
            "with concrete examples of how it addresses the failures in the reflective dataset"
        )
    )


def format_proposals_for_merger(
    proposals_with_scores: list[dict[str, Any]]
) -> str:
    """
    Format scored proposals for the merger signature.

    Args:
        proposals_with_scores: List of dicts containing:
            - proposed_instruction: str
            - total_score: float
            - reasoning: str
            - (optionally) dataset_alignment_score, general_quality_score

    Returns:
        Formatted string of proposals ready for merging
    """
    formatted_parts = []

    for idx, proposal in enumerate(proposals_with_scores, 1):
        score = proposal.get("total_score", 0)
        instruction = proposal.get("proposed_instruction", "")
        reasoning = proposal.get("reasoning", "No reasoning provided")

        # Optionally include breakdown scores
        breakdown = ""
        if "dataset_alignment_score" in proposal and "general_quality_score" in proposal:
            breakdown = (
                f"\n  - Dataset Alignment: {proposal['dataset_alignment_score']:.1f}/50"
                f"\n  - General Quality: {proposal['general_quality_score']:.1f}/50"
            )

        formatted_parts.append(
            f"### Proposal {idx} (Total Score: {score:.1f}/100){breakdown}\n"
            f"**Instruction:**\n{instruction}\n\n"
            f"**Judge's Reasoning:**\n{reasoning}\n"
        )

    return "\n".join(formatted_parts)


class ProposalMerger:
    """
    Wrapper class for merging multiple top-scoring proposals into a single optimized instruction.
    """

    def __init__(self, merger_lm: dspy.LM):
        """
        Initialize the proposal merger.

        Args:
            merger_lm: The language model to use for merging proposals
        """
        self.merger_lm = merger_lm
        self.merger_module = dspy.ChainOfThought(ProposalMergerSignature)

    def merge(
        self,
        current_instruction: str,
        top_proposals: list[dict[str, Any]],
        reflective_dataset: dict[str, list[dict[str, Any]]],
        max_examples: int = 5
    ) -> dict[str, str]:
        """
        Merge top-scoring proposals into a single optimized instruction.

        Args:
            current_instruction: The current instruction being improved
            top_proposals: List of top-n scoring proposals with scores and reasoning
            reflective_dataset: Dictionary of failed examples with feedback
            max_examples: Maximum examples to include in context

        Returns:
            Dictionary containing:
                - merged_instruction: str
                - rationale: str
                - improvements_over_current: str
        """
        # Import the formatting function from judge_signature
        from judge_signature import format_reflective_dataset_for_judge

        # Format proposals and dataset
        proposals_text = format_proposals_for_merger(top_proposals)
        dataset_summary = format_reflective_dataset_for_judge(
            reflective_dataset, max_examples
        )

        # Merge using the merger LM
        with dspy.context(lm=self.merger_lm):
            result = self.merger_module(
                current_instruction=current_instruction,
                top_proposals=proposals_text,
                reflective_dataset_summary=dataset_summary
            )

        return {
            "merged_instruction": result.merged_instruction,
            "rationale": result.rationale,
            "improvements_over_current": result.improvements_over_current
        }
