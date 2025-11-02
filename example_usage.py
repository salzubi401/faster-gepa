"""
Example usage of MultiLLMProposalFn with GEPA.

This script demonstrates how to:
1. Configure multiple LLMs with different models and temperature settings
2. Set up judge and merger LLMs
3. Create a MultiLLMProposalFn instance
4. Use it with GEPA optimizer

NOTE: This uses GEPA's original InstructionProposalSignature and formatting
for full compatibility with the GEPA optimizer.
"""

import dspy
from multi_llm_proposer import MultiLLMProposalFn


# ==============================================================================
# Example 1: Basic Setup with Multiple Models
# ==============================================================================

def example_basic_setup():
    """
    Basic example showing how to set up MultiLLMProposalFn with different models.
    """
    print("\n" + "="*80)
    print("Example 1: Basic Setup with Multiple Models")
    print("="*80 + "\n")

    # Configure different LLMs for proposal generation
    # Mix of different models AND different temperature settings
    proposal_lms = [
        # GPT-4 with low temperature (more focused)
        dspy.LM("openai/gpt-4", temperature=0.3),

        # GPT-4 with high temperature (more creative)
        dspy.LM("openai/gpt-4", temperature=0.9),

        # Claude Sonnet (different model provider)
        dspy.LM("anthropic/claude-3-5-sonnet-20241022", temperature=0.5),

        # GPT-3.5 with medium temperature (faster, budget option)
        dspy.LM("openai/gpt-3.5-turbo", temperature=0.6),
    ]

    # Configure judge LLM (typically use a strong, consistent model)
    judge_lm = dspy.LM("openai/gpt-4", temperature=0.2)

    # Configure merger LLM (benefits from reasoning capability)
    merger_lm = dspy.LM("anthropic/claude-3-5-sonnet-20241022", temperature=0.4)

    # Create the multi-LLM proposal function
    proposer = MultiLLMProposalFn(
        proposal_lms=proposal_lms,
        judge_lm=judge_lm,
        merger_lm=merger_lm,
        top_n=3,  # Merge top 3 proposals
        verbose=True
    )

    print("✓ MultiLLMProposalFn configured with:")
    print(f"  - {len(proposal_lms)} proposal LLMs")
    print(f"  - Judge LLM: {judge_lm.model}")
    print(f"  - Merger LLM: {merger_lm.model}")
    print(f"  - Top-n for merging: 3")

    return proposer


# ==============================================================================
# Example 2: Integration with GEPA Optimizer
# ==============================================================================

def example_gepa_integration():
    """
    Example showing how to use MultiLLMProposalFn with GEPA optimizer.

    NOTE: This uses the actual GEPA library's optimizer, not DSPy's GEPA.
    The proposer works with GEPA's adapter-based architecture.
    """
    print("\n" + "="*80)
    print("Example 2: Integration with GEPA Optimizer")
    print("="*80 + "\n")

    # Set up the multi-LLM proposer
    proposer = example_basic_setup()

    print("\n✓ MultiLLMProposalFn is ready for GEPA integration")
    print("\nTo use with GEPA:")
    print("  1. Set up your GEPA adapter (DspyAdapter or custom)")
    print("  2. Pass the proposer via adapter.propose_new_texts = proposer")
    print("  3. Or use ReflectiveMutationProposer with custom proposer")
    print("\nExample:")
    print("  from gepa.core.adapter import GEPAAdapter")
    print("  ")
    print("  # Your adapter")
    print("  adapter = YourAdapter(...)")
    print("  ")
    print("  # Inject custom proposer")
    print("  adapter.propose_new_texts = proposer")
    print("  ")
    print("  # Run GEPA optimization")
    print("  from gepa import gepa")
    print("  results = gepa.optimize(adapter, ...)")

    return proposer


# ==============================================================================
# Example 3: Custom Model Configuration Strategies
# ==============================================================================

def example_model_strategies():
    """
    Examples of different strategies for configuring proposal LMs.
    """
    print("\n" + "="*80)
    print("Example 3: Different Model Configuration Strategies")
    print("="*80 + "\n")

    # Strategy 1: Same model, temperature variation
    print("Strategy 1: Same model with temperature diversity")
    strategy1_lms = [
        dspy.LM("openai/gpt-4", temperature=0.2),
        dspy.LM("openai/gpt-4", temperature=0.5),
        dspy.LM("openai/gpt-4", temperature=0.8),
    ]
    print(f"  Created {len(strategy1_lms)} LMs with temperatures: 0.2, 0.5, 0.8\n")

    # Strategy 2: Different model providers
    print("Strategy 2: Different model providers for diversity")
    strategy2_lms = [
        dspy.LM("openai/gpt-4"),
        dspy.LM("anthropic/claude-3-5-sonnet-20241022"),
        dspy.LM("google/gemini-pro"),
    ]
    print(f"  Created {len(strategy2_lms)} LMs from OpenAI, Anthropic, Google\n")

    # Strategy 3: Mix of models and temperatures
    print("Strategy 3: Mix of both model and temperature diversity")
    strategy3_lms = [
        dspy.LM("openai/gpt-4", temperature=0.3),
        dspy.LM("openai/gpt-4", temperature=0.9),
        dspy.LM("anthropic/claude-3-5-sonnet-20241022", temperature=0.5),
        dspy.LM("anthropic/claude-3-5-sonnet-20241022", temperature=0.8),
        dspy.LM("openai/gpt-3.5-turbo", temperature=0.6),
    ]
    print(f"  Created {len(strategy3_lms)} LMs with mixed providers and temps\n")

    # Strategy 4: Budget-conscious (using cheaper models)
    print("Strategy 4: Budget-conscious with cheaper models")
    strategy4_lms = [
        dspy.LM("openai/gpt-3.5-turbo", temperature=0.3),
        dspy.LM("openai/gpt-3.5-turbo", temperature=0.7),
        dspy.LM("anthropic/claude-3-haiku-20240307", temperature=0.5),
    ]
    print(f"  Created {len(strategy4_lms)} LMs using GPT-3.5 and Claude Haiku\n")

    return {
        "strategy1": strategy1_lms,
        "strategy2": strategy2_lms,
        "strategy3": strategy3_lms,
        "strategy4": strategy4_lms,
    }


# ==============================================================================
# Example 4: Using with AIME Dataset (from your project)
# ==============================================================================

def example_aime_integration():
    """
    Example showing how to use MultiLLMProposalFn with AIME dataset.
    This integrates with your existing AIME project.
    """
    print("\n" + "="*80)
    print("Example 4: Integration with AIME Dataset")
    print("="*80 + "\n")

    # Set up proposal LMs optimized for math reasoning
    proposal_lms = [
        # GPT-4 is strong at math reasoning
        dspy.LM("openai/gpt-4", temperature=0.2),
        dspy.LM("openai/gpt-4", temperature=0.6),

        # Claude Sonnet also good at reasoning
        dspy.LM("anthropic/claude-3-5-sonnet-20241022", temperature=0.3),
        dspy.LM("anthropic/claude-3-5-sonnet-20241022", temperature=0.7),

        # O1-mini for additional reasoning capability
        dspy.LM("openai/o1-mini", temperature=1.0),
    ]

    # Use GPT-4 for judging (consistent, reliable)
    judge_lm = dspy.LM("openai/gpt-4", temperature=0.1)

    # Use Claude Sonnet for merging (excellent at synthesis)
    merger_lm = dspy.LM("anthropic/claude-3-5-sonnet-20241022", temperature=0.3)

    proposer = MultiLLMProposalFn(
        proposal_lms=proposal_lms,
        judge_lm=judge_lm,
        merger_lm=merger_lm,
        top_n=3,
        verbose=True
    )

    print("✓ MultiLLMProposalFn configured for AIME dataset:")
    print(f"  - {len(proposal_lms)} proposal LMs (GPT-4, Claude, O1)")
    print(f"  - Judge: GPT-4 (temp=0.1)")
    print(f"  - Merger: Claude Sonnet (temp=0.3)")
    print("\nThis configuration is optimized for mathematical reasoning tasks.")
    print("\nTo use with your AIME dataset:")
    print("  1. Load your AIME examples (from aime_dataset.py)")
    print("  2. Define your math QA program")
    print("  3. Create GEPA optimizer with this proposer")
    print("  4. Run optimization on train/val splits")

    return proposer


# ==============================================================================
# Example 5: Testing the Proposer Directly
# ==============================================================================

def example_direct_testing():
    """
    Example showing how to test the proposer directly without GEPA.
    Useful for debugging and understanding the proposal process.
    """
    print("\n" + "="*80)
    print("Example 5: Testing Proposer Directly")
    print("="*80 + "\n")

    # Simple setup with fewer models for faster testing
    proposer = MultiLLMProposalFn(
        proposal_lms=[
            dspy.LM("openai/gpt-3.5-turbo", temperature=0.4),
            dspy.LM("openai/gpt-3.5-turbo", temperature=0.8),
        ],
        judge_lm=dspy.LM("openai/gpt-4", temperature=0.2),
        merger_lm=dspy.LM("openai/gpt-4", temperature=0.3),
        top_n=2,
        verbose=True
    )

    # Create mock data to test the proposer
    candidate = {
        "generate_answer": "Answer the question directly and concisely."
    }

    reflective_dataset = {
        "generate_answer": [
            {
                "Inputs": {"question": "What is 2+2?"},
                "Generated_Outputs": {"answer": "The answer involves addition"},
                "Feedback": "Too vague. Should provide the specific numerical answer."
            },
            {
                "Inputs": {"question": "What is the capital of France?"},
                "Generated_Outputs": {"answer": "It's a European city"},
                "Feedback": "Should directly state 'Paris', not describe it."
            }
        ]
    }

    components_to_update = ["generate_answer"]

    print("Testing proposer with mock data...\n")
    print(f"Current instruction: {candidate['generate_answer']}")
    print(f"Number of failed examples: {len(reflective_dataset['generate_answer'])}")

    # Call the proposer
    result = proposer(
        candidate=candidate,
        reflective_dataset=reflective_dataset,
        components_to_update=components_to_update
    )

    print("\n" + "="*80)
    print("Result:")
    print("="*80)
    print(f"\nNew instruction: {result['generate_answer']}")

    return result


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("MultiLLMProposalFn Examples")
    print("="*80)

    # Run examples (uncomment the ones you want to try)

    # Example 1: Basic setup
    proposer = example_basic_setup()

    # Example 2: GEPA integration
    # optimizer = example_gepa_integration()

    # Example 3: Model strategies
    # strategies = example_model_strategies()

    # Example 4: AIME integration
    # aime_proposer = example_aime_integration()

    # Example 5: Direct testing (requires API keys)
    # result = example_direct_testing()

    print("\n" + "="*80)
    print("Examples completed!")
    print("="*80 + "\n")
    print("Next steps:")
    print("1. Configure your API keys for the LLMs you want to use")
    print("2. Uncomment and run the examples above")
    print("3. Integrate with your own DSPy program and dataset")
    print("4. Run GEPA optimization with the multi-LLM proposer")
