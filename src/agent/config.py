"""
Agent configuration system for pluggable prompts and tools
"""

from typing import List, Dict, Any, Optional, Protocol, Type, Iterator
from dataclasses import dataclass

from agent.tools import BaseTool


@dataclass
class AgentConfig:
    """Configuration for an agent instance"""

    name: str
    description: str
    prompt_template: str
    tools: List[BaseTool]
    default_state: Optional[Dict[str, Any]] = None
    max_iterations: int = 3  # Maximum tool execution iterations per user input

    def build_prompt(
        self,
        tools_description: str,
        current_state: Optional[Dict[str, Any]] = None,
        iteration_info: Optional[tuple] = None,
    ) -> str:
        """Build the system prompt with tools and state"""
        state = current_state or self.default_state or {}

        # Build iteration info string
        iteration_str = ""
        if iteration_info:
            current_iteration, max_iterations = iteration_info
            iteration_str = f"(Iteration {current_iteration}/{max_iterations})"

        # Replace placeholders in template
        return self.prompt_template.format(
            tools_description=tools_description,
            state_info=self._build_state_info(state),
            iteration_info=iteration_str,
        )

    def _build_state_info(self, state: Dict[str, Any]) -> str:
        """Build state information string"""
        # This can be customized per config type
        return ""

    def format_response(self, response: Any, state: Dict[str, Any]) -> str:
        """Format the response into a string"""
        # Default implementation, can be overridden
        return str(response).strip()

    def get_summarization_system_prompt(self) -> str:
        """Get the system prompt for summarization that includes current state context"""
        # Build a summarization system prompt that includes relevant state
        base_prompt = "You are a conversation summarizer. Your task is to create concise, accurate summaries that preserve essential context."

        # This can be overridden by specific configs to include state-specific context
        return base_prompt

    def get_summarization_prompt(self) -> str:
        """Get the user prompt for summarization request"""

        # Default summarization prompt
        return """Please provide a summary of the conversation so far, 
including key points and decisions made.  Include all important context from 
the conversation history needed to continue the discussion."""


def get_all_configs() -> Dict[str, AgentConfig]:
    """Get all available agent configurations"""

    from .configs.roleplay import RoleplayConfig
    from .configs.coding import CodingConfig
    from .configs.general import GeneralConfig

    return {
        "roleplay": RoleplayConfig(),
        "coding": CodingConfig(),
        "general": GeneralConfig(),
    }


def get_config(config_name: str) -> AgentConfig:
    """Get an agent configuration by name"""
    configs = get_all_configs()
    if config_name not in configs:
        raise ValueError(
            f"Unknown config: {config_name}. Available: {list(configs.keys())}"
        )
    return configs[config_name]


def list_configs() -> Dict[str, str]:
    """List all available configurations with descriptions"""
    return {name: config.description for name, config in get_all_configs().items()}
