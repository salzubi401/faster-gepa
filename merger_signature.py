"""
Prompt merger signature for synthesizing multiple high-quality proposals.

This module provides a DSPy signature that takes multiple top-scoring proposed
instructions and merges them into a single optimized instruction by extracting
the best elements from each.
"""

import dspy
from typing import Any


class ProposalMergerSignature(dspy.Signature):
    """You are an expert at synthesizing multiple high-quality ideas into a single SUPERIOR solution. Your task is CRITICAL: create a merged instruction that DEMONSTRABLY SURPASSES any individual proposal by creating genuine synergies.

**CRITICAL REQUIREMENT:**
The merged instruction MUST be demonstrably superior to any individual proposal. It should:
- Address MORE failure patterns than any single proposal
- Combine complementary strengths in ways that create new value
- Be more comprehensive, specific, and actionable than any single proposal
- NOT just concatenate or average the proposals—CREATE SYNTHESIS

**Your Role:**
Analyze multiple top-scoring instruction proposals and create a merged instruction that intelligently combines their strengths while addressing ALL observed failure patterns.

**Merging Process - Follow These Steps STRICTLY:**

1. **Deep Analysis of Each Proposal:**
   - Identify the UNIQUE, DISTINCTIVE strengths of each proposal (what makes it special?)
   - Map which SPECIFIC failures each addresses (be precise)
   - Recognize innovative strategies, domain knowledge, or novel approaches each contains
   - Understand the judge's reasoning—why did each score well?
   - Identify what each proposal uniquely contributes that others don't

2. **Find GENUINE Synergies (Not Just Combinations):**
   - Look for proposals that address DIFFERENT aspects or failure modes—these can be powerfully combined
   - Identify complementary ideas that REINFORCE each other (e.g., one provides structure, another provides domain details)
   - Find proposals that take DIFFERENT approaches to the same problem—combining these creates more robust solutions
   - Create NEW insights by combining perspectives that individual proposals couldn't achieve alone
   - **CRITICAL**: If proposals are too similar, you must synthesize them into something MORE comprehensive than either

3. **Resolve Conflicts Intelligently:**
   - When proposals contradict, choose the approach that:
     * Better addresses the failure patterns (check the dataset)
     * Is more specific and actionable
     * Has clearer reasoning from the judge
     * Can be combined with elements from other proposals
   - Consider if the conflict reveals a deeper issue—address both perspectives if possible
   - Don't default to the highest-scoring proposal—synthesize the best elements

4. **Eliminate Redundancy and Create Density:**
   - Combine similar suggestions into SINGLE, STRONGER, MORE COMPREHENSIVE statements
   - Avoid repetitive guidance—each sentence should add unique value
   - Keep the instruction concise BUT ensure it's MORE complete than any individual proposal
   - If multiple proposals say similar things, synthesize them into a better version

5. **Synthesize to Create Superiority:**
   - Create a coherent, well-structured instruction with logical flow
   - Ensure it addresses ALL failure patterns comprehensively (more than any single proposal)
   - Maintain specific, actionable guidance throughout
   - Verify the merged instruction would perform better than any individual proposal
   - The final instruction should feel like a unified, expert-crafted solution, not a patchwork

**Quality Criteria for Merged Result (MUST MEET ALL):**
- ✓ Incorporates the BEST insights from ALL proposals (not just some)
- ✓ MORE comprehensive than ANY single proposal (addresses more failure patterns)
- ✓ Addresses ALL failure patterns from the dataset (check completeness)
- ✓ Clear, specific, and immediately actionable (better than any individual)
- ✓ Well-structured with logical organization (professional quality)
- ✓ No contradictions or redundancy (seamless synthesis)
- ✓ Creates NEW value through synergies (not just combination)
- ✓ Would score HIGHER than any individual proposal if judged

**Output Requirements:**
- **merged_instruction**: The final synthesized instruction that is demonstrably superior to any individual proposal
- **rationale**: Detailed explanation including:
  1. What unique elements were taken from each proposal and WHY
  2. How conflicts were resolved and why those choices were made
  3. What NEW insights or synergies emerged from combining proposals
  4. SPECIFIC reasons why this merged instruction is BETTER than any single proposal
  5. How the merged instruction addresses MORE failure patterns than any individual
- **improvements_over_current**: Concrete examples showing:
  1. How the merged instruction fixes SPECIFIC failures from the dataset
  2. What improvements it makes over the current instruction
  3. Why it's better than any individual proposal

**Success Criteria:**
Your merged instruction succeeds if it:
- Addresses more failure patterns than any single proposal
- Combines complementary strengths in ways that create new value
- Is more comprehensive and actionable than any individual proposal
- Would score higher than any individual proposal if evaluated
- Feels like a unified, expert solution rather than a combination

**Remember:**
The goal is SYNTHESIS, not concatenation. The merged instruction should be GENUINELY SUPERIOR to any individual proposal. If you can't create something better, you haven't done your job properly.
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
            "MUST be demonstrably superior to any individual proposal: more comprehensive, "
            "addresses more failure patterns, creates synergies, and is more actionable. "
            "Should be clear, specific, and immediately usable."
        )
    )

    rationale: str = dspy.OutputField(
        desc=(
            "Detailed explanation of the merging process, including:\n"
            "1. What UNIQUE elements were taken from each proposal and WHY\n"
            "2. How conflicts or contradictions were resolved and why those choices were made\n"
            "3. What NEW insights or synergies emerged from combining proposals\n"
            "4. SPECIFIC reasons why this merged instruction is BETTER than any single proposal\n"
            "5. How the merged instruction addresses MORE failure patterns than any individual"
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
