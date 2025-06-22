from agent.config import AgentConfig


class GeneralConfig(AgentConfig):
    """Configuration for general-purpose agents"""

    def __init__(self):
        super().__init__(
            name="general",
            description="General-purpose AI assistant",
            prompt_template="""You are a helpful AI assistant that can assist with a wide variety of tasks.

Available tools:
{tools_description}

TOOL USAGE:
Use the appropriate tools to help accomplish the user's requests effectively.

Be helpful, accurate, and efficient in your responses.
""",
            tools=[],  # TODO: Implement general tools
        )
