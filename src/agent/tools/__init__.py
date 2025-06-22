from abc import ABC, abstractmethod
from typing import Dict, Any, Callable, List, Type, Optional
from pydantic import BaseModel


class ToolInput(BaseModel):
    """Base class for tool inputs"""

    pass


class BaseTool(ABC):
    """Base class for all agent tools"""

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name"""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description"""
        pass

    @property
    @abstractmethod
    def input_schema(self) -> Type[ToolInput]:
        """Input schema for the tool"""
        pass

    @abstractmethod
    def run(self, agent, input_data: ToolInput) -> str:
        """Execute the tool with the given input"""
        pass

    def get_schema_description(self) -> str:
        """Get a formatted description of the tool and its parameters"""
        schema = self.input_schema.model_json_schema()
        properties = schema.get("properties", {})
        required = schema.get("required", [])

        param_descriptions = []
        for param_name, param_info in properties.items():
            param_type = param_info.get("type", "string")
            param_desc = param_info.get("description", "")
            is_required = "required" if param_name in required else "optional"
            param_descriptions.append(
                f"  - {param_name} ({param_type}) ({is_required}): {param_desc}"
            )

        if param_descriptions:
            params_text = "\n" + "\n".join(param_descriptions)
        else:
            params_text = ""

        return f"- {self.name}: {self.description}{params_text}"


class ToolRegistry:
    """Registry for agent tools"""

    def __init__(self, agent=None, tools: List[BaseTool] = []):
        self.agent = agent  # Reference to the agent for state access
        self.tools: Dict[str, "BaseTool"] = {}

        if tools:
            self._register_tools(tools)

    def _register_tools(self, tools: List[BaseTool]):
        """Register tool instances"""
        for tool in tools:
            self.tools[tool.name] = tool

    def register_tool(self, tool: BaseTool):
        """Register a single tool instance"""
        self.tools[tool.name] = tool

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a specific tool"""
        return self.tools.get(name)

    def list_tools(self) -> List[str]:
        """List all available tools"""
        return list(self.tools.keys())

    def get_tools_description(self) -> str:
        """Get formatted description of all tools with schemas"""
        descriptions = []
        for tool in self.tools.values():
            descriptions.append(tool.get_schema_description())

        return "\n\n".join(descriptions)

    def execute(self, tool_name: str, input_data: Dict[str, Any]) -> str:
        """Execute a tool with validated input"""
        tool = self.get_tool(tool_name)
        if not tool:
            return f"Tool '{tool_name}' not found"

        try:
            # Validate input against schema
            validated_input = tool.input_schema(**input_data)

            # Execute tool with agent and validated data
            result = tool.run(self.agent, validated_input)
            return str(result)
        except Exception as e:
            return f"Error executing tool '{tool_name}': {str(e)}"
