"""
Prompt loading utilities with strong typing.
"""

from enum import Enum
from importlib.resources import files


class PromptType(Enum):
    """Supported prompt types."""

    ROLEPLAY = "roleplay"
    CODING = "coding"
    GENERAL = "general"


def load_prompt(prompt_type: PromptType) -> str:
    """
    Load a prompt template from the agent.config.prompts package.

    Args:
        prompt_type: Type of prompt to load

    Returns:
        The prompt template content as a string

    Raises:
        FileNotFoundError: If the prompt file doesn't exist
    """
    config_files = files("agent.data.configs.prompts")
    prompt_file = config_files / f"{prompt_type.value}.txt"
    return prompt_file.read_text(encoding="utf-8")
