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
    summarization_prompt: Optional[str] = None
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
        iteration_text = ""
        if iteration_info:
            current, total = iteration_info
            iteration_text = f"\n--- TURN {current}/{total} ---"
            if current == total:
                iteration_text += "\nThis is your FINAL turn. You must provide a complete response with dialogue only (no tools available)."
            else:
                iteration_text += f"\nREMINDER: Keep responses to 1 sentence max. User needs to respond after each exchange."

        # Replace placeholders in template
        return self.prompt_template.format(
            tools_description=tools_description,
            state_info=self._build_state_info(state),
            iteration_info=iteration_text,
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

    def get_summarization_prompt(self, message_count: int) -> str:
        """Get the user prompt for summarization request"""
        if self.summarization_prompt:
            return self.summarization_prompt.format(message_count=message_count)

        # Default summarization prompt
        return f"""Please provide a concise summary of the {message_count} messages above, focusing on:
1. Key information and context
2. Important events and developments  
3. Relevant details to remember for continuing the conversation

Provide a structured summary that captures the essential information."""


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
