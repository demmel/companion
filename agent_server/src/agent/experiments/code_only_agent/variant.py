"""Variant interface for code-only agent benchmarks."""

from agent.llm.models import SupportedModel
from agent.llm.router import LLM


class LLMCodeAgentVariant:
    """A variant of the code-only agent using a specific LLM model."""

    def __init__(self, llm: LLM, model: SupportedModel):
        self.llm = llm
        self.model = model

    def name(self) -> str:
        """Return the variant name (model identifier)."""
        return str(self.model.value)
