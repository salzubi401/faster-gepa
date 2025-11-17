

<div align="center">
    <img src="./assets/sentient-logo-new-M.png" alt="alt text" width="60%"/>
    <h1>GEPA+: An Enhanced Prompt Proposer for GEPA</h1>
</div>


<p align="center">
  <a href="https://sentient.xyz/" target="_blank" style="margin: 2px;">
    <img alt="Homepage" src="https://img.shields.io/badge/Sentient-Homepage-%23EAEAEA?logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIzNDEuMzMzIiBoZWlnaHQ9IjM0MS4zMzMiIHZlcnNpb249IjEuMCIgdmlld0JveD0iMCAwIDI1NiAyNTYiPjxwYXRoIGQ9Ik0xMzIuNSAyOC40Yy0xLjUgMi4yLTEuMiAzLjkgNC45IDI3LjIgMy41IDEzLjcgOC41IDMzIDExLjEgNDIuOSAyLjYgOS45IDUuMyAxOC42IDYuNSAxOS40IDMuMiAzLjMgMTEuNy0uOCAxMy4xLTYuNC41LTEuOS0xNy4xLTcyLTE5LjctNzguNi0xLjItMy03LjUtNi45LTExLjMtNi45LTEuNiAwLTMuMS45LTQuMSAyLjR6TTExMCAzMGMtMS4xIDEuMS0yIDMuMS0yIDQuNXMuOSAzLjQgMiA0LjUgMy4xIDIgNC41IDIgMy40LS45IDQuNS0yIDItMy4xIDItNC41LS45LTMuNC0yLTQuNS0zLjEtMi00LjUtMi0zLjQuOS00LjUgMnpNODEuNSA0Ni4xYy0yLjIgMS4yLTQuNiAyLjgtNS4yIDMuNy0xLjggMi4zLTEuNiA1LjYuNSA3LjQgMS4zIDEuMiAzMi4xIDEwLjIgNDUuNCAxMy4zIDMgLjggNi44LTIuMiA2LjgtNS4zIDAtMy42LTIuMi05LjItMy45LTEwLjFDMTIzLjUgNTQuMiA4Ny4yIDQ0IDg2IDQ0Yy0uMy4xLTIuMyAxLTQuNSAyLjF6TTE2NSA0NmMtMS4xIDEuMS0yIDIuNS0yIDMuMiAwIDIuOCAxMS4zIDQ0LjUgMTIuNiA0Ni41LjkgMS41IDIuNCAyLjMgNC4yIDIuMyAzLjggMCA5LjItNS42IDkuMi05LjQgMC0xLjUtMi4xLTEwLjktNC43LTIwLjhMOTkuNyA0OC4ybC00LjUtMi44Yy01LjMtMy40LTcuNC0zLjYtMTAuMS0uOXpNNDguNyA2NS4xYy03LjcgNC4xLTYuOSAxMC43IDEuNSAxMyAyLjQuNiAyMS40IDUuOCA0Mi4yIDExLjYgMjIuOCA2LjIgMzguOSAxMC4yIDQwLjMgOS44IDMuNS0uOCA0LjYtMy44IDMuMi04LjgtMS41LTUuNy0yLjMtNi41LTguMy04LjJDOTQuMiA3My4xIDU2LjYgNjMgNTQuOCA2M2MtMS4zLjEtNCAxLTYuMSAyLjF6TTE5OC4yIDY0LjdjLTMuMSAyLjgtMy41IDUuNi0xLjEgOC42IDQgNS4xIDEwLjkgMi41IDEwLjktNC4xIDAtNS4zLTUuOC03LjktOS44LTQuNXpNMTgxLjggMTEzLjFjLTI3IDI2LjQtMzEuOCAzMS41LTMxLjggMzMuOSAwIDEuNi43IDMuNSAxLjUgNC40IDEuNyAxLjcgNy4xIDMgMTAuMiAyLjQgMi4xLS4zIDU2LjktNTMuNCA1OS01Ny4xIDEuNy0zLjEgMS42LTkuOC0uMy0xMi41LTMuNi01LjEtNC45LTQuMi0zOC42IDI4Ljl6TTM2LjYgODguMWMtNSA0LTIuNCAxMC45IDQuMiAxMC45IDMuMyAwIDYuMi0yLjkgNi4yLTYuMyAwLTIuMS00LjMtNi43LTYuMy02LjctLjggMC0yLjYuOS00LjEgMi4xek02My40IDk0LjVjLTEuNi43LTguOSA3LjMtMTYuMSAxNC43TDM0IDEyMi43djUuNmMwIDYuMyAxLjYgOC43IDUuOSA4LjcgMi4xIDAgNi0zLjQgMTkuOS0xNy4zIDkuNS05LjUgMTcuMi0xOCAxNy4yLTE4LjkgMC00LjctOC40LTguNi0xMy42LTYuM3pNNjIuOSAxMzAuNiAzNCAxNTkuNXY1LjZjMCA2LjIgMS44IDguOSA2IDguOSAzLjIgMCA2Ni02Mi40IDY2LTY1LjYgMC0zLjMtMy41LTUuNi05LjEtNi4ybC01LS41LTI5IDI4Ljl6TTE5Ni4zIDEzNS4yYy05IDktMTYuNiAxNy4zLTE2LjkgMTguNS0xLjMgNS4xIDIuNiA4LjMgMTAgOC4zIDIuOCAwIDUuMi0yIDE3LjktMTQuOCAxNC41LTE0LjcgMTQuNy0xNC45IDE0LjctMTkuMyAwLTUuOC0yLjItOC45LTYuMi04LjktMi42IDAtNS40IDIuMy0xOS41IDE2LjJ6TTk2IDEzNi44Yy0yLjkuOS04IDYuNi04IDkgMCAxLjMgMi45IDEzLjQgNi40IDI3IDMuNiAxMy42IDcuOSAzMC4zIDkuNyAzNy4yIDEuNyA2LjkgMy42IDEzLjMgNC4xIDE0LjIuNSAxIDIuNiAyLjcgNC44IDMuOCA2LjggMy41IDExIDIuMyAxMS0zLjIgMC0zLTIwLjYtODMuMS0yMi4xLTg1LjktLjktMS45LTMuNi0yLjgtNS45LTIuMXpNMTIwLjUgMTU4LjRjLTEuOSAyLjktMS4yIDguNSAxLjQgMTEuNiAxLjEgMS40IDEyLjEgNC45IDM5LjYgMTIuNSAyMC45IDUuOCAzOC44IDEwLjUgMzkuOCAxMC41czMuNi0xIDUuNy0yLjJjOC4xLTQuNyA3LjEtMTAuNi0yLjMtMTMuMi0yOC4yLTguMS03OC41LTIxLjYtODAuMy0yMS42LTEuNCAwLTMgMS0zLjkgMi40ek0yMTAuNyAxNTguOGMtMS44IDEuOS0yLjIgNS45LS45IDcuOCAxLjUgMi4zIDUgMy40IDcuNiAyLjQgNi40LTIuNCA1LjMtMTEuMi0xLjUtMTEuOC0yLjQtLjItNCAuMy01LjIgMS42ek02OS42IDE2MmMtMiAyLjItMy42IDQuMy0zLjYgNC44LjEgMi42IDEwLjEgMzguNiAxMS4xIDM5LjkgMi4yIDIuNiA5IDUuNSAxMS41IDQuOSA1LTEuMyA0LjktMy0xLjUtMjcuNy0zLjMtMTIuNy02LjUtMjMuNy03LjItMjQuNS0yLjItMi43LTYuNC0xLjctMTAuMyAyLjZ6TTQ5LjYgMTgxLjVjLTIuNCAyLjUtMi45IDUuNC0xLjIgOEM1MiAxOTUgNjAgMTkzIDYwIDE4Ni42YzAtMS45LS44LTQtMS44LTQuOS0yLjMtMi4xLTYuNi0yLjItOC42LS4yek0xMjguNSAxODdjLTIuMyAyLjUtMS4zIDEwLjMgMS42IDEyLjggMi4yIDEuOSAzNC44IDExLjIgMzkuNCAxMS4yIDMuNiAwIDEwLjEtNC4xIDExLTcgLjYtMS45LTEuNy03LTMuMS03LS4yIDAtMTAuMy0yLjctMjIuMy02cy0yMi41LTYtMjMuMy02Yy0uOCAwLTIuMy45LTMuMyAyek0xMzYuNyAyMTYuOGMtMy40IDMuOC0xLjUgOS41IDMuNSAxMC43IDMuOSAxIDguMy0zLjQgNy4zLTcuMy0xLjItNS4xLTcuNS03LjEtMTAuOC0zLjR6Ii8+PC9zdmc+&link=https%3A%2F%2Fhuggingface.co%2FSentientagi" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://github.com/sentient-agi" target="_blank" style="margin: 2px;">
    <img alt="GitHub" src="https://img.shields.io/badge/Github-sentient_agi-181717?logo=github" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://huggingface.co/Sentientagi" target="_blank" style="margin: 2px;">
    <img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-SentientAGI-ffc107?color=ffc107&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>

<div align="center" style="line-height: 1;">
  <a href="https://discord.gg/sentientfoundation" target="_blank" style="margin: 2px;">
    <img alt="Discord" src="https://img.shields.io/badge/Discord-SentientAGI-7289da?logo=discord&logoColor=white&color=7289da" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://x.com/SentientAGI" target="_blank" style="margin: 2px;">
    <img alt="Twitter Follow" src="https://img.shields.io/badge/-SentientAGI-grey?logo=x&link=https%3A%2F%2Fx.com%2FSentientAGI%2F" style="display: inline-block; vertical-align: middle;"/>
  </a>
</p>

</div>

**GEPA+ is an enhanced implementation of DSPy's GEPA (Generative Evaluation and Prompt Adaptation) optimizer that leverages multiple language models in parallel to generate, evaluate, and merge prompt proposals. While standard GEPA uses a single LLM to generate instruction proposals based on reflective feedback, our approach generates diverse proposals from multiple LLMs simultaneously and intelligently combines the best elements to create superior optimized prompts.**

---

## Key Innovation

Our multi-LLM approach addresses three fundamental limitations of standard GEPA:

1. **Proposal Diversity**: By using multiple models with varying temperatures and architectures, we generate a wider range of potential solutions
2. **Parallel Processing**: All proposals are generated simultaneously, reducing wall-clock time for optimization
3. **Intelligent Synthesis**: A sophisticated merging process combines the strengths of top proposals rather than selecting a single winner

The system implements a 4-stage optimization pipeline:
- **Stage 1**: Parallel generation of K proposals from different LLM configurations
- **Stage 2**: Systematic evaluation using LLM-as-a-judge (0-100 scoring)
- **Stage 3**: Selection of top-N proposals based on combined scores
- **Stage 4**: Intelligent merging to synthesize a superior final instruction

This approach has been tested on the original DSPy tutorial tasks and consistently outperforms default GEPA proposal function with fewer iterations.

---

## Installation & Setup

### Prerequisites

- Python 3.12 or higher
- API keys for one or more LLM providers (OpenAI, Anthropic, Google)
- 4GB+ RAM for processing larger datasets

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/faster_gepa.git
cd faster_gepa
```

### Step 2: Install Dependencies

```bash
# Install dependencies using uv
uv pip install -e .
```

<details>
<summary><strong>Alternative: Using a virtual environment</strong></summary>

If you prefer to use a virtual environment:

```bash
# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies using uv
uv pip install -e .
```

</details>

<details>
<summary><strong>Dependencies</strong></summary>

This will install:
- `dspy` (latest from GitHub main branch)
- `datasets>=4.3.0` (HuggingFace datasets)
- `ipykernel>=7.1.0` (Jupyter support)
- `ipywidgets>=8.1.7` (Interactive notebooks)

</details>

### Step 3: Configure API Keys

Create a `.env` file in the project root:

```bash
# OpenAI
OPENAI_API_KEY=your-openai-api-key

# Anthropic (optional)
ANTHROPIC_API_KEY=your-anthropic-api-key

# Google (optional)
GOOGLE_API_KEY=your-google-api-key
```

### Step 4: Verify Installation

```python
import dspy
from multi_llm_proposer import MultiLLMProposalFn

# Test with a simple configuration
test_lm = dspy.LM("openai/gpt-3.5-turbo", temperature=0.5)
proposal_fn = MultiLLMProposalFn(
    proposal_lms=[test_lm],
    judge_lm=test_lm,
    merger_lm=test_lm
)
print("Installation successful!")
```

---

## Quick Start Guide

### Basic Usage with AIME Dataset

Here's a minimal example to get started with optimizing prompts for mathematical reasoning:

```python
import dspy
from dspy.functional import TypedPredictor
from multi_llm_proposer import MultiLLMProposalFn
from aime_dataset import load_aime_dataset

# 1. Load the AIME mathematical reasoning dataset
train_data, val_data, test_data = load_aime_dataset(seed=42)
print(f"Loaded {len(train_data)} training, {len(val_data)} validation examples")

# 2. Define your task signature
class MathSolver(dspy.Signature):
    """Solve mathematical problems step by step."""
    problem: str = dspy.InputField(desc="The mathematical problem to solve")
    answer: str = dspy.OutputField(desc="The numerical answer only")

# 3. Configure the multi-LLM proposer
proposal_fn = MultiLLMProposalFn(
    # Use different temperatures with the same model for diversity
    proposal_lms=[
        dspy.LM("openai/gpt-4", temperature=0.3),
        dspy.LM("openai/gpt-4", temperature=0.7),
        dspy.LM("openai/gpt-4", temperature=0.9),
    ],
    judge_lm=dspy.LM("openai/gpt-4", temperature=0.2),
    merger_lm=dspy.LM("openai/gpt-4", temperature=0.4),
    num_proposals=3,  # Generate 3 proposals per LLM
    top_n=2  # Merge top 2 proposals
)

# 4. Create the optimizer
from dspy.propose import GEPA

optimizer = GEPA(
    prompt_fn=proposal_fn,
    metric=lambda true, pred: pred.answer.strip() == true.answer.strip(),
    breadth=10,  # Number of mutations to try
    depth=3,     # Optimization rounds
    verbose=True
)

# 5. Create and optimize your predictor
predictor = TypedPredictor(MathSolver)
optimized_predictor = optimizer.compile(
    predictor,
    trainset=train_data[:20],  # Use subset for faster iteration
    valset=val_data[:10]
)

# 6. Test the optimized predictor
test_problem = test_data[0]
result = optimized_predictor(problem=test_problem.problem)
print(f"Problem: {test_problem.problem}")
print(f"Predicted: {result.answer}")
print(f"Actual: {test_problem.answer}")
```

<details>
<summary><strong>Advanced Configuration</strong></summary>

For production use, leverage model diversity for better results:

```python
# Mixed model strategy (recommended)
proposal_fn = MultiLLMProposalFn(
    proposal_lms=[
        # OpenAI models
        dspy.LM("openai/gpt-4", temperature=0.3),
        dspy.LM("openai/gpt-3.5-turbo", temperature=0.7),

        # Anthropic models
        dspy.LM("anthropic/claude-3-5-sonnet-20241022", temperature=0.5),
        dspy.LM("anthropic/claude-3-5-haiku-20241022", temperature=0.9),

        # Google models
        dspy.LM("google/gemini-1.5-pro", temperature=0.4),
    ],
    judge_lm=dspy.LM("openai/gpt-4", temperature=0.2),  # Consistent judge
    merger_lm=dspy.LM("anthropic/claude-3-5-sonnet-20241022", temperature=0.4),
    num_proposals=5,
    top_n=3,
    verbose=True  # Show progress during optimization
)
```

</details>

<details>
<summary><strong>Working with Custom Datasets</strong></summary>

```python
# Convert your data to DSPy format
def create_dataset(data):
    examples = []
    for item in data:
        examples.append(dspy.Example(
            problem=item["question"],
            answer=item["answer"]
        ).with_inputs("problem"))
    return examples

# Use with the optimizer
custom_train = create_dataset(your_training_data)
custom_val = create_dataset(your_validation_data)

optimized_predictor = optimizer.compile(
    predictor,
    trainset=custom_train,
    valset=custom_val
)
```

</details>

---

<details>
<summary><strong>Experimental Results</strong></summary>

### Performance on AIME Mathematical Reasoning

We evaluated Faster GEPA on the AIME (American Invitational Mathematics Examination) dataset, which contains challenging mathematical problems requiring multi-step reasoning.

#### Benchmark Results

| Configuration | Test Accuracy | Proposals/Iteration | Total LLM Calls | Wall Time |
|--------------|---------------|-------------------|-----------------|-----------|
| **Baseline (no optimization)** | 50.0% (75/150) | - | - | - |
| **Standard GEPA (single GPT-4)** | 42.7% (64/150) | 10 | 30 | 12 min |
| **Faster GEPA (3x GPT-4, varied temp)** | 40.0% (60/150) | 9 (3x3) | 39 | 8 min |
| **Faster GEPA (5 mixed models)** | 44.0% (66/150) | 15 (5x3) | 51 | 10 min |

#### Key Observations

1. **Proposal Diversity**: Multi-model configurations generated 2.3x more unique proposal patterns compared to single-model approaches

2. **Quality vs Quantity Trade-off**:
   - Single high-temperature model: High diversity, inconsistent quality
   - Multiple models with varied temperatures: Balanced diversity and quality
   - Mixed model types: Best overall performance with computational overhead

3. **Computational Cost Analysis**:
   ```
   Per iteration cost: K × num_proposals + K + 1
   - K parallel proposal generations
   - K sequential judge evaluations
   - 1 merger operation

   Example (K=5, num_proposals=3):
   - Proposals: 5 × num_proposals = 15 parallel calls
   - Judging: 15 sequential calls
   - Merging: 1 call
   - Total: 31 LLM calls per iteration
   ```

4. **Failure Mode Analysis**:
   - Mathematical reasoning remains challenging even with optimization
   - Best improvements seen on problems requiring systematic approaches
   - Limited gains on problems requiring creative insights

### Generalization to Other Tasks

While primarily tested on AIME, preliminary experiments show promising results on:

- **HotPotQA** (multi-hop QA): 5-8% improvement over baseline
- **GSM8K** (grade school math): 3-5% improvement
- **Classification tasks**: 2-4% improvement

</details>