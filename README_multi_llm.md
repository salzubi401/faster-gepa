# Multi-LLM Proposal Function for DSPy GEPA

A custom proposal function for DSPy's GEPA optimizer that uses **multiple LLMs** to generate, score, and merge prompt proposals for improved optimization results.

## Overview

This implementation extends DSPy GEPA's proposal mechanism with a sophisticated multi-stage process:

1. **Parallel Proposal Generation**: K different LLMs independently generate new prompt proposals
2. **LLM-as-a-Judge Scoring**: Each proposal is evaluated on both dataset-specific improvements and general quality
3. **Top-N Selection**: Proposals are ranked by score and the best ones are selected
4. **Intelligent Merging**: A merger LLM synthesizes the top proposals into a single optimized instruction

## Key Features

✅ **Model Diversity**: Use different LLM providers (OpenAI, Anthropic, Google) for diverse proposals
✅ **Temperature Variation**: Generate proposals with different creativity levels
✅ **Dual-Criteria Judging**: Evaluate on both dataset alignment and general prompt quality
✅ **Configurable Top-N**: Flexibly control how many proposals to merge
✅ **Drop-in Replacement**: Works seamlessly with existing GEPA workflows
✅ **Verbose Mode**: Detailed logging for debugging and analysis

## Installation

Ensure you have DSPy installed:

```bash
pip install dspy-ai
```

No additional dependencies required beyond DSPy.

## Quick Start

```python
import dspy
from multi_llm_proposer import MultiLLMProposalFn

# Configure proposal LLMs (mix of models and temperatures)
proposal_lms = [
    dspy.LM("openai/gpt-4", temperature=0.3),
    dspy.LM("openai/gpt-4", temperature=0.9),
    dspy.LM("anthropic/claude-3-5-sonnet-20241022", temperature=0.5),
]

# Configure judge and merger LLMs
judge_lm = dspy.LM("openai/gpt-4", temperature=0.2)
merger_lm = dspy.LM("anthropic/claude-3-5-sonnet-20241022", temperature=0.4)

# Create the multi-LLM proposer
proposer = MultiLLMProposalFn(
    proposal_lms=proposal_lms,
    judge_lm=judge_lm,
    merger_lm=merger_lm,
    top_n=3,
    verbose=True
)

# Use with GEPA
from dspy.teleprompt import GEPA

optimizer = GEPA(
    metric=your_metric_function,
    instruction_proposer=proposer,  # Use custom proposer
    reflection_lm=dspy.LM("openai/gpt-4"),
    max_full_evals=10
)

# Run optimization
optimized_program = optimizer.compile(
    student=your_program,
    trainset=train_examples,
    valset=val_examples
)
```

## Architecture

### 1. Proposal Generation

The `MultiLLMProposalFn` generates K proposals in parallel using different LLMs:

```python
def generate_proposals_parallel(self, current_instruction, reflective_dataset, component_name):
    # Uses ThreadPoolExecutor for parallel LLM calls
    # Each LLM independently proposes a new instruction
    # Returns list of K proposals
```

**Benefits**:
- **Diversity**: Different models bring different perspectives
- **Creativity Range**: Temperature variation balances conservative vs. creative ideas
- **Efficiency**: Parallel execution reduces total time

### 2. LLM-as-a-Judge Scoring

Each proposal is scored by a judge LLM on two dimensions:

```python
class ProposalJudgeSignature(dspy.Signature):
    dataset_alignment_score: float  # 0-50: How well it addresses failures
    general_quality_score: float    # 0-50: Clarity, specificity, actionability
    total_score: float              # 0-100: Sum of both scores
    reasoning: str                  # Detailed explanation
```

**Scoring Criteria**:
- **Dataset Alignment (0-50)**:
  - Addresses specific failures in reflective dataset
  - Fixes identified patterns and issues
  - Relevant to the task at hand

- **General Quality (0-50)**:
  - Clarity and specificity
  - Actionability
  - Completeness
  - Best practices adherence

### 3. Top-N Selection

Proposals are ranked by total score and the top N are selected for merging:

```python
def select_top_proposals(self, scored_proposals):
    sorted_proposals = sorted(scored_proposals, key=lambda x: x["total_score"], reverse=True)
    return sorted_proposals[:self.top_n]
```

### 4. Proposal Merging

A merger LLM combines the best elements of top proposals:

```python
class ProposalMergerSignature(dspy.Signature):
    merged_instruction: str          # Final synthesized instruction
    rationale: str                   # Explanation of merge decisions
    improvements_over_current: str   # Specific improvements made
```

**Merging Process**:
1. Analyzes strengths of each top proposal
2. Identifies complementary ideas
3. Eliminates redundancy and contradictions
4. Synthesizes a coherent final instruction

## Configuration Strategies

### Strategy 1: Temperature Diversity (Same Model)

Use the same model with different temperatures for controlled variation:

```python
proposal_lms = [
    dspy.LM("openai/gpt-4", temperature=0.2),  # Conservative
    dspy.LM("openai/gpt-4", temperature=0.5),  # Balanced
    dspy.LM("openai/gpt-4", temperature=0.8),  # Creative
]
```

**Best for**: When you want consistent style but varied creativity

### Strategy 2: Model Diversity

Use different model providers for maximum diversity:

```python
proposal_lms = [
    dspy.LM("openai/gpt-4"),
    dspy.LM("anthropic/claude-3-5-sonnet-20241022"),
    dspy.LM("google/gemini-pro"),
]
```

**Best for**: When you want different reasoning approaches and perspectives

### Strategy 3: Mixed Strategy (Recommended)

Combine both model and temperature diversity:

```python
proposal_lms = [
    dspy.LM("openai/gpt-4", temperature=0.3),
    dspy.LM("openai/gpt-4", temperature=0.9),
    dspy.LM("anthropic/claude-3-5-sonnet-20241022", temperature=0.5),
    dspy.LM("anthropic/claude-3-5-sonnet-20241022", temperature=0.8),
]
```

**Best for**: Maximum diversity and best results (recommended for production)

### Strategy 4: Budget-Conscious

Use cheaper models for cost-effective optimization:

```python
proposal_lms = [
    dspy.LM("openai/gpt-3.5-turbo", temperature=0.3),
    dspy.LM("openai/gpt-3.5-turbo", temperature=0.7),
    dspy.LM("anthropic/claude-3-haiku-20240307", temperature=0.5),
]
```

**Best for**: Development, testing, or budget-constrained scenarios

## API Reference

### `MultiLLMProposalFn`

Main class implementing the multi-LLM proposal function.

#### Constructor Parameters

- **`proposal_lms`** (`list[dspy.LM]`): List of K language models for generating proposals
- **`judge_lm`** (`dspy.LM`): Language model to use as judge for scoring
- **`merger_lm`** (`dspy.LM`): Language model to use for merging top proposals
- **`top_n`** (`int`, default=3): Number of top-scoring proposals to merge
- **`base_proposer_signature`** (`type`, optional): Custom DSPy signature for proposal generation
- **`verbose`** (`bool`, default=True): Whether to print progress information

#### Methods

##### `__call__(candidate, reflective_dataset, components_to_update) -> dict[str, str]`

Main method implementing the `ProposalFn` protocol.

**Parameters**:
- `candidate`: Dictionary mapping component names to current instructions
- `reflective_dataset`: Dictionary mapping component names to lists of failed examples
- `components_to_update`: List of component names to generate proposals for

**Returns**: Dictionary mapping component names to new proposed instructions

##### `generate_proposals_parallel(current_instruction, reflective_dataset, component_name) -> list[str]`

Generate K proposals in parallel using different LLMs.

##### `score_proposals(current_instruction, proposals, reflective_dataset) -> list[dict]`

Score all proposals using LLM-as-a-judge.

##### `select_top_proposals(scored_proposals) -> list[dict]`

Select top-n proposals based on scores.

##### `merge_proposals(current_instruction, top_proposals, reflective_dataset) -> str`

Merge top proposals into a single optimized instruction.

### `ProposalJudge`

Wrapper class for evaluating proposed instructions.

#### Constructor Parameters

- **`judge_lm`** (`dspy.LM`): The language model to use for judging

#### Methods

##### `evaluate(current_instruction, proposed_instruction, reflective_dataset, max_examples=5) -> dict`

Evaluate a proposed instruction.

**Returns**:
```python
{
    "total_score": float,
    "dataset_alignment_score": float,
    "general_quality_score": float,
    "reasoning": str,
    "proposed_instruction": str
}
```

### `ProposalMerger`

Wrapper class for merging multiple top-scoring proposals.

#### Constructor Parameters

- **`merger_lm`** (`dspy.LM`): The language model to use for merging

#### Methods

##### `merge(current_instruction, top_proposals, reflective_dataset, max_examples=5) -> dict`

Merge top-scoring proposals into a single optimized instruction.

**Returns**:
```python
{
    "merged_instruction": str,
    "rationale": str,
    "improvements_over_current": str
}
```

## Examples

See `example_usage.py` for comprehensive examples including:

1. **Basic Setup**: Simple configuration with multiple models
2. **GEPA Integration**: Full integration with GEPA optimizer
3. **Model Strategies**: Different approaches to configuring proposal LMs
4. **AIME Integration**: Example for mathematical reasoning tasks
5. **Direct Testing**: Testing the proposer without GEPA for debugging

## Performance Considerations

### Computational Cost

The multi-LLM approach makes K+2 LLM calls per proposal iteration:
- K parallel calls for proposal generation
- 1 call for judging each proposal (batched in sequence)
- 1 call for merging top proposals

**Optimization tips**:
- Use cheaper models for proposal generation
- Reserve expensive models for judging and merging
- Reduce K for faster iterations
- Use parallel execution (already implemented)

### Cost Optimization

**Budget-friendly configuration**:
```python
# Use GPT-3.5 for proposals, GPT-4 for judge/merger
proposal_lms = [
    dspy.LM("openai/gpt-3.5-turbo", temperature=0.3),
    dspy.LM("openai/gpt-3.5-turbo", temperature=0.7),
]
judge_lm = dspy.LM("openai/gpt-4", temperature=0.2)
merger_lm = dspy.LM("openai/gpt-4", temperature=0.3)
```

**High-performance configuration**:
```python
# Use top models throughout
proposal_lms = [
    dspy.LM("openai/gpt-4", temperature=0.3),
    dspy.LM("openai/gpt-4", temperature=0.9),
    dspy.LM("anthropic/claude-3-5-sonnet-20241022", temperature=0.5),
    dspy.LM("openai/o1-mini", temperature=1.0),
]
judge_lm = dspy.LM("openai/gpt-4", temperature=0.1)
merger_lm = dspy.LM("anthropic/claude-3-5-sonnet-20241022", temperature=0.3)
```

## Debugging

Enable verbose mode to see detailed progress:

```python
proposer = MultiLLMProposalFn(
    proposal_lms=proposal_lms,
    judge_lm=judge_lm,
    merger_lm=merger_lm,
    verbose=True  # Shows all steps
)
```

**Output includes**:
- Proposal generation progress for each LLM
- Scores for each proposal (total, dataset alignment, quality)
- Top-N selection results
- Merge process and final instruction

## Troubleshooting

### Issue: Proposals are too similar

**Solution**: Increase temperature variation or use more diverse models

```python
# Before: Too similar
proposal_lms = [
    dspy.LM("openai/gpt-4", temperature=0.3),
    dspy.LM("openai/gpt-4", temperature=0.4),
]

# After: More diverse
proposal_lms = [
    dspy.LM("openai/gpt-4", temperature=0.2),
    dspy.LM("openai/gpt-4", temperature=0.9),
    dspy.LM("anthropic/claude-3-5-sonnet-20241022", temperature=0.6),
]
```

### Issue: Judge scores are inconsistent

**Solution**: Use a more stable model with lower temperature

```python
# Use GPT-4 with very low temperature for consistent judging
judge_lm = dspy.LM("openai/gpt-4", temperature=0.1)
```

### Issue: Merged instruction loses quality

**Solution**: Increase top_n or use a better merger model

```python
# Increase top_n to include more perspectives
proposer = MultiLLMProposalFn(
    proposal_lms=proposal_lms,
    judge_lm=judge_lm,
    merger_lm=dspy.LM("openai/gpt-4", temperature=0.3),  # Better merger
    top_n=4  # Include more proposals in merge
)
```

### Issue: Too expensive

**Solution**: Use cheaper models or reduce K

```python
# Reduce number of proposals
proposal_lms = [
    dspy.LM("openai/gpt-3.5-turbo", temperature=0.4),
    dspy.LM("openai/gpt-3.5-turbo", temperature=0.8),
]
```

## Advanced Usage

### Custom Proposal Signature

You can provide a custom signature for proposal generation:

```python
class MyCustomProposalSignature(dspy.Signature):
    """Custom proposal signature for specific task"""
    current_instruction: str = dspy.InputField()
    examples_with_feedback: str = dspy.InputField()
    domain_knowledge: str = dspy.InputField()  # Additional context
    new_instruction: str = dspy.OutputField()

proposer = MultiLLMProposalFn(
    proposal_lms=proposal_lms,
    judge_lm=judge_lm,
    merger_lm=merger_lm,
    base_proposer_signature=MyCustomProposalSignature
)
```

### Integration with Custom Metrics

The proposer works with any GEPA-compatible metric:

```python
def custom_metric_with_feedback(example, pred, trace=None):
    # Your custom scoring logic
    score = evaluate_prediction(example, pred)

    # Provide detailed feedback for failures
    if score < threshold:
        feedback = analyze_failure(example, pred)
    else:
        feedback = "Success"

    return score, feedback

optimizer = GEPA(
    metric=custom_metric_with_feedback,
    instruction_proposer=proposer,
    ...
)
```

## Best Practices

1. **Start Small**: Begin with 2-3 proposal LMs and increase if needed
2. **Use Diverse Models**: Mix different providers for best results
3. **Temperature Range**: Include both conservative (0.2-0.4) and creative (0.7-0.9) temperatures
4. **Stable Judge**: Use low temperature (0.1-0.2) for consistent judging
5. **Strong Merger**: Use your best model for merging (Claude Sonnet or GPT-4)
6. **Monitor Costs**: Track API usage, especially with many proposal LMs
7. **Test Directly**: Use Example 5 in `example_usage.py` to test proposer behavior
8. **Iterate on top_n**: Experiment with different values (2-4 typically works well)

## License

This implementation is designed to work with DSPy and follows its licensing.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Citation

If you use this in your research, please cite the DSPy GEPA paper:

```bibtex
@article{opsahl2024gepa,
  title={GEPA: Improving Instruction-Following Agents via Self-Improvement with Reflective Feedback},
  author={Opsahl-Ong, Beryl and Karamcheti, Siddharth and Liang, Percy},
  journal={arXiv preprint arXiv:2501.00118},
  year={2024}
}
```
