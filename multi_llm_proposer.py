"""
Multi-LLM Proposal Function for GEPA Optimization.

This module implements a custom proposal function that:
1. Uses GEPA's InstructionProposalSignature to generate K proposals in parallel
   with different LLMs/settings
2. Scores each proposal using an LLM-as-a-judge
3. Selects top-n proposals based on scores
4. Merges the best proposals into a single optimized instruction

Usage:
    proposer = MultiLLMProposalFn(
        proposal_lms=[lm1, lm2, lm3],
        judge_lm=judge_lm,
        merger_lm=merger_lm,
        top_n=3,
        reflection_prompt_template=None,  # Optional: customize GEPA prompt
        verbose=True
    )

    # Use with GEPA adapter
    adapter.propose_new_texts = proposer
"""

import dspy
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

from gepa.strategies.instruction_proposal import InstructionProposalSignature
from judge_signature import ProposalJudge
from merger_signature import ProposalMerger


class MultiLLMProposalFn:
    """
    Custom proposal function that uses multiple LLMs to generate, score, and merge proposals.

    Workflow:
    1. Generates K proposals in parallel using GEPA's InstructionProposalSignature
       with different LLM configurations (models and/or temperatures)
    2. Scores each proposal with LLM-as-a-judge on dataset alignment and quality
    3. Selects top-n highest-scoring proposals
    4. Merges top proposals into a single optimized instruction

    Implements the ProposalFn protocol expected by GEPA:
        def __call__(
            candidate: dict[str, str],
            reflective_dataset: dict[str, list[dict[str, Any]]],
            components_to_update: list[str],
        ) -> dict[str, str]
    """

    def __init__(
        self,
        proposal_lms: list[dspy.LM],
        judge_lm: dspy.LM,
        merger_lm: dspy.LM,
        top_n: int = 3,
        reflection_prompt_template: Optional[str] = None,
        verbose: bool = True
    ):
        """
        Initialize the multi-LLM proposal function.

        Args:
            proposal_lms: List of K language models to use for generating proposals
            judge_lm: Language model to use as judge for scoring proposals
            merger_lm: Language model to use for merging top proposals
            top_n: Number of top-scoring proposals to merge
            reflection_prompt_template: Optional custom prompt template for GEPA's
                                       InstructionProposalSignature. If None, uses
                                       GEPA's default template. Must include placeholders:
                                       <curr_instructions> and <inputs_outputs_feedback>
            verbose: Whether to print progress information
        """
        self.proposal_lms = proposal_lms
        self.judge_lm = judge_lm
        self.merger_lm = merger_lm
        self.top_n = min(top_n, len(proposal_lms))  # Can't select more than we generate
        self.verbose = verbose

        # Initialize judge and merger
        self.judge = ProposalJudge(judge_lm)
        self.merger = ProposalMerger(merger_lm)

        # Store custom prompt template for GEPA proposals
        self.reflection_prompt_template = reflection_prompt_template

    def _create_lm_wrapper(self, dspy_lm: dspy.LM):
        """
        Create a wrapper that makes dspy.LM compatible with GEPA's LanguageModel protocol.

        GEPA expects: def __call__(self, prompt: str) -> str
        """
        def lm_callable(prompt: str) -> str:
            # Use dspy.LM's __call__ method with the prompt
            with dspy.context(lm=dspy_lm):
                response = dspy_lm(prompt)
                # Handle different response formats
                if isinstance(response, str):
                    return response
                elif hasattr(response, 'text'):
                    return response.text
                elif hasattr(response, 'content'):
                    return response.content
                elif isinstance(response, list) and len(response) > 0:
                    return str(response[0])
                else:
                    return str(response)
        return lm_callable

    def _generate_proposal(
        self,
        lm: dspy.LM,
        current_instruction: str,
        reflective_dataset: list[dict[str, Any]],
        proposal_idx: int
    ) -> str:
        """
        Generate a single proposal using GEPA's InstructionProposalSignature.

        This follows the pattern from reflective_mutation.py:71-78
        """
        try:
            # Create GEPA-compatible LM wrapper
            lm_wrapper = self._create_lm_wrapper(lm)

            # Use GEPA's InstructionProposalSignature.run() method
            result = InstructionProposalSignature.run(
                lm=lm_wrapper,
                input_dict={
                    "current_instruction_doc": current_instruction,
                    "dataset_with_feedback": reflective_dataset,
                    "prompt_template": self.reflection_prompt_template,
                },
            )

            proposal = result["new_instruction"]

            if self.verbose:
                print(f"  [Proposal {proposal_idx + 1}] Generated with {lm.model}")

            return proposal

        except Exception as e:
            if self.verbose:
                print(f"  [Proposal {proposal_idx + 1}] Error with {lm.model}: {e}")
                import traceback
                print(traceback.format_exc())
            # Return current instruction as fallback
            return current_instruction

    def generate_proposals_parallel(
        self,
        current_instruction: str,
        reflective_dataset: list[dict[str, Any]]
    ) -> list[str]:
        """Generate K proposals in parallel using different LLMs."""
        if self.verbose:
            print(f"\nGenerating {len(self.proposal_lms)} proposals in parallel...")

        # Use ThreadPoolExecutor for parallel generation
        with ThreadPoolExecutor(max_workers=len(self.proposal_lms)) as executor:
            futures = []
            for idx, lm in enumerate(self.proposal_lms):
                future = executor.submit(
                    self._generate_proposal,
                    lm,
                    current_instruction,
                    reflective_dataset,
                    idx
                )
                futures.append(future)

            # Collect results
            proposals = [future.result() for future in futures]

        return proposals

    def score_proposals(
        self,
        current_instruction: str,
        proposals: list[str],
        reflective_dataset: dict[str, list[dict[str, Any]]]
    ) -> list[dict[str, Any]]:
        """Score all proposals using LLM-as-a-judge."""
        if self.verbose:
            print(f"\nScoring {len(proposals)} proposals with judge LLM...")

        scored_proposals = []
        for idx, proposal in enumerate(proposals):
            try:
                score_result = self.judge.evaluate(
                    current_instruction=current_instruction,
                    proposed_instruction=proposal,
                    reflective_dataset=reflective_dataset,
                    max_examples=5
                )
                scored_proposals.append(score_result)

                if self.verbose:
                    print(f"  [Proposal {idx + 1}] Score: {score_result['total_score']:.1f}/100 "
                          f"(Dataset: {score_result['dataset_alignment_score']:.1f}, "
                          f"Quality: {score_result['general_quality_score']:.1f})")

            except Exception as e:
                if self.verbose:
                    print(f"  [Proposal {idx + 1}] Scoring error: {e}")
                # Add with low score if scoring fails
                scored_proposals.append({
                    "proposed_instruction": proposal,
                    "total_score": 0.0,
                    "dataset_alignment_score": 0.0,
                    "general_quality_score": 0.0,
                    "reasoning": f"Scoring failed: {e}"
                })

        return scored_proposals

    def select_top_proposals(
        self,
        scored_proposals: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Select top-n proposals based on scores."""
        # Sort by total score (descending)
        sorted_proposals = sorted(
            scored_proposals,
            key=lambda x: x["total_score"],
            reverse=True
        )

        # Select top-n
        top_proposals = sorted_proposals[:self.top_n]

        if self.verbose:
            print(f"\nSelected top {len(top_proposals)} proposals for merging:")
            for idx, prop in enumerate(top_proposals, 1):
                print(f"  {idx}. Score: {prop['total_score']:.1f}/100")

        return top_proposals

    def merge_proposals(
        self,
        current_instruction: str,
        top_proposals: list[dict[str, Any]],
        reflective_dataset: dict[str, list[dict[str, Any]]]
    ) -> str:
        """Merge top proposals into a single optimized instruction."""
        if self.verbose:
            print(f"\nMerging top {len(top_proposals)} proposals...")

        try:
            merge_result = self.merger.merge(
                current_instruction=current_instruction,
                top_proposals=top_proposals,
                reflective_dataset=reflective_dataset,
                max_examples=5
            )

            merged_instruction = merge_result["merged_instruction"]

            if self.verbose:
                print(f"  Merged instruction created ({len(merged_instruction)} chars)")
                print(f"  Rationale: {merge_result['rationale'][:200]}...")

            return merged_instruction

        except Exception as e:
            if self.verbose:
                print(f"  Merging error: {e}")
            # Fallback to the highest-scoring proposal
            return top_proposals[0]["proposed_instruction"]

    def __call__(
        self,
        candidate: dict[str, str],
        reflective_dataset: dict[str, list[dict[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        """
        Main proposal function implementing the ProposalFn protocol.

        Args:
            candidate: Dictionary mapping component names to current instructions
            reflective_dataset: Dictionary mapping component names to lists of failed examples
            components_to_update: List of component names to generate proposals for

        Returns:
            Dictionary mapping component names to new proposed instructions
        """
        proposed_texts = {}

        for component_name in components_to_update:
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"Processing component: {component_name}")
                print(f"{'='*60}")

            current_instruction = candidate.get(component_name, "")

            # Get reflective dataset for this component (list of examples)
            component_examples = reflective_dataset.get(component_name, [])
            component_dataset = {component_name: component_examples}

            # Step 1: Generate K proposals in parallel
            proposals = self.generate_proposals_parallel(
                current_instruction, component_examples
            )

            # Step 2: Score all proposals with LLM-as-judge
            scored_proposals = self.score_proposals(
                current_instruction, proposals, component_dataset
            )

            # Step 3: Select top-n proposals
            top_proposals = self.select_top_proposals(scored_proposals)

            # Step 4: Merge top proposals
            merged_instruction = self.merge_proposals(
                current_instruction, top_proposals, component_dataset
            )

            proposed_texts[component_name] = merged_instruction

            if self.verbose:
                print(f"\n[Final] New instruction for {component_name}:")
                print(f"  {merged_instruction[:200]}...")

        return proposed_texts
