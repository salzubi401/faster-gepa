"""
Multi-Modal Multi-LLM Proposal Function for GEPA Optimization.

This module implements a multimodal instruction proposer that:
1. Handles multimodal inputs (like dspy.Image) during GEPA optimization
2. Uses multiple LLMs to generate K proposals in parallel
3. Scores each proposal using an LLM-as-a-judge
4. Selects top-n proposals based on scores
5. Merges the best proposals into a single optimized instruction

Usage:
    proposer = MultiModalInstructionProposer(
        proposal_lms=[lm1, lm2, lm3],
        judge_lm=judge_lm,
        merger_lm=merger_lm,
        top_n=3,
        reflection_prompt_template=None,  # Optional: customize GEPA prompt
        verbose=True
    )

    # Use with GEPA adapter
    optimizer = dspy.GEPA(
        instruction_proposer=proposer,
        ...
    )
"""

import dspy
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

from gepa.strategies.instruction_proposal import InstructionProposalSignature
from judge_signature import ProposalJudge
from merger_signature import ProposalMerger


# Try to import ReflectiveExample and ProposalFn from GEPA
try:
    from gepa.strategies.instruction_proposal import ProposalFn, ReflectiveExample
except ImportError:
    # Fallback if imports fail - define minimal types
    from typing import Protocol
    from typing import Any as ReflectiveExample
    
    class ProposalFn(Protocol):
        def __call__(
            self,
            candidate: dict[str, str],
            reflective_dataset: dict[str, list[Any]],
            components_to_update: list[str],
        ) -> dict[str, str]:
            ...


def _convert_reflective_example_to_dict(example: Any) -> dict[str, Any]:
    """
    Convert a ReflectiveExample object to a dict format expected by judge/merger.
    
    Handles both ReflectiveExample objects and dicts for backward compatibility.
    """
    if isinstance(example, dict):
        return example
    
    # Helper to get attribute with multiple possible names
    def get_attr(obj: Any, *names: str, default: Any = None) -> Any:
        for name in names:
            value = getattr(obj, name, None)
            if value is not None:
                return value
            # Also check __dict__ directly
            if hasattr(obj, '__dict__') and name in obj.__dict__:
                return obj.__dict__[name]
        return default
    
    # Extract common attributes with fallback name variations
    result = {}
    
    inputs = get_attr(example, 'Inputs', 'inputs')
    if inputs is not None:
        result['Inputs'] = inputs
    
    outputs = get_attr(example, 'Generated_Outputs', 'generated_outputs', 'Outputs', 'outputs')
    if outputs is not None:
        result['Generated_Outputs'] = outputs
    
    feedback = get_attr(example, 'Feedback', 'feedback')
    if feedback is not None:
        result['Feedback'] = feedback
    
    # If no attributes found, try __dict__ or string representation
    if not result:
        if hasattr(example, '__dict__'):
            result = example.__dict__.copy()
        else:
            result = {'raw': str(example)}
    
    return result


class MultiModalInstructionProposer(ProposalFn):
    """
    GEPA-compatible multimodal instruction proposer with multi-LLM proposal and reranking.

    This class handles multimodal inputs (like dspy.Image) during GEPA optimization by:
    1. Using GEPA's InstructionProposalSignature to generate K proposals in parallel
       with different LLMs/settings (which handles multimodal inputs)
    2. Scoring each proposal using an LLM-as-a-judge
    3. Selecting top-n proposals based on scores
    4. Merging the best proposals into a single optimized instruction

    Workflow:
    1. Generates K proposals in parallel using GEPA's InstructionProposalSignature
       with different LLM configurations (models and/or temperatures)
    2. Scores each proposal with LLM-as-a-judge on dataset alignment and quality
    3. Selects top-n highest-scoring proposals
    4. Merges top proposals into a single optimized instruction

    Implements the ProposalFn protocol expected by GEPA:
        def __call__(
            candidate: dict[str, str],
            reflective_dataset: dict[str, list[ReflectiveExample]],
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
        Initialize the multi-modal multi-LLM proposal function.

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
        reflective_dataset: list[Any],  # Can be ReflectiveExample or dict
        proposal_idx: int
    ) -> str:
        """
        Generate a single proposal using GEPA's InstructionProposalSignature.

        This handles multimodal inputs (like dspy.Image) through GEPA's
        InstructionProposalSignature which supports multimodal data.
        """
        try:
            # Create GEPA-compatible LM wrapper
            lm_wrapper = self._create_lm_wrapper(lm)

            # Use GEPA's InstructionProposalSignature.run() method
            # This handles multimodal inputs (ReflectiveExample objects with dspy.Image)
            result = InstructionProposalSignature.run(
                lm=lm_wrapper,
                input_dict={
                    "current_instruction_doc": current_instruction,
                    "dataset_with_feedback": reflective_dataset,  # Can be ReflectiveExample objects
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
        reflective_dataset: list[Any]  # Can be ReflectiveExample or dict
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
        reflective_dataset: dict[str, list[Any]]  # Can be ReflectiveExample or dict
    ) -> list[dict[str, Any]]:
        """Score all proposals using LLM-as-a-judge."""
        if self.verbose:
            print(f"\nScoring {len(proposals)} proposals with judge LLM...")

        # Convert ReflectiveExample objects to dict format for judge
        converted_dataset = {}
        for component_name, examples in reflective_dataset.items():
            converted_dataset[component_name] = [
                _convert_reflective_example_to_dict(ex) for ex in examples
            ]

        scored_proposals = []
        for idx, proposal in enumerate(proposals):
            try:
                score_result = self.judge.evaluate(
                    current_instruction=current_instruction,
                    proposed_instruction=proposal,
                    reflective_dataset=converted_dataset,
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
        reflective_dataset: dict[str, list[Any]]  # Can be ReflectiveExample or dict
    ) -> str:
        """Merge top proposals into a single optimized instruction."""
        if self.verbose:
            print(f"\nMerging top {len(top_proposals)} proposals...")

        # Convert ReflectiveExample objects to dict format for merger
        converted_dataset = {}
        for component_name, examples in reflective_dataset.items():
            converted_dataset[component_name] = [
                _convert_reflective_example_to_dict(ex) for ex in examples
            ]

        try:
            merge_result = self.merger.merge(
                current_instruction=current_instruction,
                top_proposals=top_proposals,
                reflective_dataset=converted_dataset,
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
        reflective_dataset: dict[str, list[Any]],  # Can be ReflectiveExample or dict
        components_to_update: list[str],
    ) -> dict[str, str]:
        """
        GEPA-compatible proposal function implementing the ProposalFn protocol.

        Args:
            candidate: Current component name -> instruction mapping
            reflective_dataset: Component name -> list of reflective examples
                                (can be ReflectiveExample objects or dicts)
            components_to_update: List of component names to update

        Returns:
            dict: Component name -> new instruction mapping
        """
        proposed_texts = {}
        
        self.last_proposals = {}
        self.last_judgments = {}
        self.last_top_proposals = {}
        self.last_merged = {}

        for component_name in components_to_update:
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"Processing component: {component_name}")
                print(f"{'='*60}")

            current_instruction = candidate.get(component_name, "")

            # Get reflective dataset for this component (list of examples)
            # These can be ReflectiveExample objects (with multimodal support)
            component_examples = reflective_dataset.get(component_name, [])
            component_dataset = {component_name: component_examples}

            # Step 1: Generate K proposals in parallel
            # GEPA's InstructionProposalSignature handles multimodal inputs
            proposals = self.generate_proposals_parallel(
                current_instruction, component_examples
            )
            
            # Store proposals with model info for logging
            self.last_proposals[component_name] = [
                {
                    'proposal': prop,
                    'model': str(self.proposal_lms[idx].model) if idx < len(self.proposal_lms) else 'unknown',
                    'index': idx
                }
                for idx, prop in enumerate(proposals)
            ]

            # Step 2: Score all proposals with LLM-as-judge
            scored_proposals = self.score_proposals(
                current_instruction, proposals, component_dataset
            )
            
            # Store judgments for logging
            self.last_judgments[component_name] = scored_proposals

            # Step 3: Select top-n proposals
            top_proposals = self.select_top_proposals(scored_proposals)
            
            # Store top proposals for logging
            self.last_top_proposals[component_name] = top_proposals

            # Step 4: Merge top proposals
            merged_instruction = self.merge_proposals(
                current_instruction, top_proposals, component_dataset
            )
            
            # Store merged instruction for logging
            self.last_merged[component_name] = merged_instruction

            proposed_texts[component_name] = merged_instruction

            if self.verbose:
                print(f"\n[Final] New instruction for {component_name}:")
                print(f"  {merged_instruction[:200]}...")

        return proposed_texts
