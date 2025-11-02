"""
DSPy signatures that replicate GEPA's InstructionProposalSignature.

This module provides DSPy-based signatures that match GEPA's original
instruction proposal prompt template and formatting logic, ensuring full
compatibility with GEPA's reflective mutation approach.
"""

import re
import dspy
from typing import Any


def format_reflective_dataset_to_markdown(samples: list[dict[str, Any]]) -> str:
    """
    Format reflective dataset samples to markdown, matching GEPA's format_samples logic.

    This replicates the exact formatting from gepa/strategies/instruction_proposal.py
    to ensure consistency with GEPA's prompt structure.

    Args:
        samples: List of dicts containing Inputs, Generated_Outputs, Feedback, etc.

    Returns:
        Markdown-formatted string with hierarchical headers
    """
    def render_value(value, level=3):
        """
        Recursively render a value as markdown with appropriate header levels.

        Args:
            value: The value to render (dict, list, or scalar)
            level: Current markdown header depth (###, ####, etc.)
        """
        if isinstance(value, dict):
            s = ""
            for k, v in value.items():
                s += f"{'#' * level} {k}\n"
                s += render_value(v, min(level + 1, 6))
            if not value:
                s += "\n"
            return s
        elif isinstance(value, (list, tuple)):
            s = ""
            for i, item in enumerate(value):
                s += f"{'#' * level} Item {i + 1}\n"
                s += render_value(item, min(level + 1, 6))
            if not value:
                s += "\n"
            return s
        else:
            return f"{str(value).strip()}\n\n"

    def convert_sample_to_markdown(sample: dict, examplenum: int) -> str:
        """Convert a single sample to markdown with Example header."""
        s = f"# Example {examplenum}\n"
        for key, val in sample.items():
            s += f"## {key}\n"
            s += render_value(val, level=3)
        return s

    return "\n\n".join(
        convert_sample_to_markdown(sample, i + 1)
        for i, sample in enumerate(samples)
    )


def extract_instruction_from_backticks(lm_output: str) -> str:
    """
    Extract instruction text from LLM output, handling ``` code blocks.

    This replicates GEPA's output_extractor logic from InstructionProposalSignature.

    Args:
        lm_output: Raw LLM output string

    Returns:
        Extracted instruction text (without backticks)
    """
    # Find the first and last backtick positions (if any)
    start = lm_output.find("```") + 3
    end = lm_output.rfind("```")

    # Handle if the first and last backticks are the same or overlap
    if start >= end:
        # Handle incomplete blocks
        stripped = lm_output.strip()
        if stripped.startswith("```"):
            # Remove opening ``` and optional language specifier
            match = re.match(r"^```\S*\n?", lm_output)
            if match:
                return lm_output[match.end():].strip()
        elif stripped.endswith("```"):
            # Remove closing ```
            return stripped[:-3].strip()
        return stripped

    # Skip optional language specifier
    content = lm_output[start:end]
    match = re.match(r"^\S*\n", content)
    if match:
        content = content[match.end():]

    return content.strip()


class GEPAInstructionProposal(dspy.Signature):
    """
    DSPy signature replicating GEPA's InstructionProposalSignature.

    This signature matches GEPA's default prompt template and expects the same
    input/output format for seamless integration with GEPA-style optimization.

    Prompt Template (from GEPA):
    ----------------------------
    I provided an assistant with the following instructions to perform a task for me:
    ```
    {current_instruction_doc}
    ```

    The following are examples of different task inputs provided to the assistant
    along with the assistant's response for each of them, and some feedback on how
    the assistant's response could be better:
    ```
    {dataset_with_feedback}
    ```

    Your task is to write a new instruction for the assistant.

    Read the inputs carefully and identify the input format and infer detailed task
    description about the task I wish to solve with the assistant.

    Read all the assistant responses and the corresponding feedback. Identify all
    niche and domain specific factual information about the task and include it in
    the instruction, as a lot of it may not be available to the assistant in the
    future. The assistant may have utilized a generalizable strategy to solve the
    task, if so, include that in the instruction as well.

    Provide the new instructions within ``` blocks.
    """

    current_instruction_doc: str = dspy.InputField(
        desc="The current instruction that needs improvement"
    )

    dataset_with_feedback: str = dspy.InputField(
        desc=(
            "Examples formatted as markdown with inputs, outputs, and feedback. "
            "Use format_reflective_dataset_to_markdown() to create this."
        )
    )

    new_instruction: str = dspy.OutputField(
        desc=(
            "An improved instruction addressing the failures in the examples. "
            "Should be provided within ``` code blocks in the LLM response."
        )
    )


class GEPAInstructionProposalModule(dspy.Module):
    """
    DSPy module wrapper for GEPA-style instruction proposal.

    This module handles the complete proposal workflow:
    1. Formats the reflective dataset to markdown
    2. Calls the LLM with the GEPA prompt template
    3. Extracts the new instruction from backticks

    Usage:
        proposer = GEPAInstructionProposalModule()
        with dspy.context(lm=my_lm):
            result = proposer(
                current_instruction="Answer questions directly",
                reflective_dataset=[{
                    "Inputs": {"question": "What is 2+2?"},
                    "Generated_Outputs": {"answer": "Around 4"},
                    "Feedback": "Be precise, say exactly '4'"
                }]
            )
            new_instruction = result.new_instruction
    """

    def __init__(self, use_cot: bool = True):
        """
        Initialize the proposal module.

        Args:
            use_cot: Whether to use Chain of Thought (recommended for better proposals)
        """
        super().__init__()
        if use_cot:
            self.propose = dspy.ChainOfThought(GEPAInstructionProposal)
        else:
            self.propose = dspy.Predict(GEPAInstructionProposal)

    def forward(
        self,
        current_instruction: str,
        reflective_dataset: list[dict[str, Any]]
    ) -> dspy.Prediction:
        """
        Generate a new instruction based on the current one and failed examples.

        Args:
            current_instruction: The current instruction text
            reflective_dataset: List of dicts with Inputs, Generated_Outputs, Feedback

        Returns:
            dspy.Prediction with new_instruction field (already extracted from backticks)
        """
        # Format the reflective dataset to markdown (GEPA style)
        dataset_markdown = format_reflective_dataset_to_markdown(reflective_dataset)

        # Generate proposal
        result = self.propose(
            current_instruction_doc=current_instruction,
            dataset_with_feedback=dataset_markdown
        )

        # Extract instruction from backticks (GEPA style)
        # Note: DSPy may already do some extraction, but we ensure GEPA compatibility
        raw_instruction = result.new_instruction
        cleaned_instruction = extract_instruction_from_backticks(raw_instruction)

        # Return with cleaned instruction
        result.new_instruction = cleaned_instruction
        return result


# Alternative: Custom prompt template support (like GEPA)
def create_custom_gepa_proposal_signature(prompt_template: str) -> type:
    """
    Create a custom GEPA proposal signature with a user-defined prompt template.

    This allows you to customize the prompt while maintaining GEPA compatibility.
    The prompt_template must include placeholders:
    - <curr_instructions> - will be replaced with current instruction
    - <inputs_outputs_feedback> - will be replaced with formatted examples

    Args:
        prompt_template: Custom prompt template string with placeholders

    Returns:
        A new DSPy Signature class with the custom prompt

    Example:
        template = '''Current instruction:
        <curr_instructions>

        Failed examples:
        <inputs_outputs_feedback>

        Write a better instruction in ``` blocks.'''

        CustomProposal = create_custom_gepa_proposal_signature(template)
        proposer = dspy.ChainOfThought(CustomProposal)
    """
    # Validate template has required placeholders
    required_placeholders = ["<curr_instructions>", "<inputs_outputs_feedback>"]
    missing = [p for p in required_placeholders if p not in prompt_template]
    if missing:
        raise ValueError(
            f"Prompt template missing required placeholders: {', '.join(missing)}"
        )

    # Create dynamic signature class with custom docstring (becomes the prompt)
    class CustomGEPAProposal(dspy.Signature):
        __doc__ = prompt_template.replace(
            "<curr_instructions>", "{current_instruction_doc}"
        ).replace(
            "<inputs_outputs_feedback>", "{dataset_with_feedback}"
        )

        current_instruction_doc: str = dspy.InputField()
        dataset_with_feedback: str = dspy.InputField()
        new_instruction: str = dspy.OutputField()

    return CustomGEPAProposal
