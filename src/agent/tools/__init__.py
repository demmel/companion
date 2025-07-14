from abc import ABC, abstractmethod
from typing import Dict, Any, Callable, List, Type, Optional, Union, Literal
from pydantic import BaseModel
import functools

from agent.types import ToolResult


class ToolNotFoundError(Exception):
    """Raised when a requested tool doesn't exist"""

    pass


class ToolExecutionError(Exception):
    """Raised when a tool exists but fails during execution"""

    pass


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
    def run(
        self,
        agent,
        input_data: ToolInput,
        tool_id: str,
        progress_callback: Callable[[Any], None],
    ) -> ToolResult:
        """
        Execute the tool with the given input.

        Args:
            agent: The agent instance
            input_data: Validated input data
            tool_id: Unique identifier for this tool execution
            progress_callback: Function to call with progress updates (JSON-serializable data)

        Returns:
            ToolResult (tagged union of success/error)

        Example:
            progress_callback({"stage": "loading", "progress": 0.1})
            progress_callback({"stage": "processing", "progress": 0.5})
            return ToolSuccessResult(type="success", content="Operation completed")
            # or
            return ToolErrorResult(type="error", error="Something went wrong")
        """
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
            enum_values = param_info.get("enum", None)
            max_length = param_info.get("maxLength", None)
            min_length = param_info.get("minLength", None)
            max_items = param_info.get("maxItems", None)
            min_items = param_info.get("minItems", None)

            param_description = (
                f"  - {param_name} ({param_type}) ({is_required}): {param_desc}"
            )
            if enum_values:
                enum_str = ", ".join(
                    f'"{value}"' if isinstance(value, str) else str(value)
                    for value in enum_values
                )
                param_description += f" (allowed values: {enum_str})"
            if max_length is not None:
                param_description += f" (max length: {max_length})"
            if min_length is not None:
                param_description += f" (min length: {min_length})"
            if max_items is not None:
                param_description += f" (max items: {max_items})"
            if min_items is not None:
                param_description += f" (min items: {min_items})"

            param_descriptions.append(param_description)

        if param_descriptions:
            params_text = "\n" + "\n".join(param_descriptions)
        else:
            params_text = ""

        return f"- {self.name}: {self.description}{params_text}"


# Legacy decorator removed - updating tools directly to new callback interface


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

    def has_tool(self, name: str) -> bool:
        """Check if a tool exists"""
        return name in self.tools

    def list_tools(self) -> List[str]:
        """List all available tools"""
        return list(self.tools.keys())

    def get_tools_description(self) -> str:
        """Get formatted description of all tools with schemas"""
        descriptions = []
        for tool in self.tools.values():
            descriptions.append(tool.get_schema_description())

        tool_call_instructions = """
You may call tools using the following syntax:

🚨 CRITICAL: Use EXACT syntax with underscore and call ID in parentheses:

TOOL_CALL: tool_name (call_1)
{
"parameter": "value"
}

REQUIRED FORMAT COMPONENTS:
- "TOOL_CALL:" with underscore (NOT "TOOL CALL:")  
- Tool name after the colon
- Call ID in parentheses like (call_1), (call_2), etc.
- JSON parameters on following lines
- NO markdown formatting, NO asterisks, NO emphasis

For multiple tools:
TOOL_CALL: tool_name_1 (call_1)
{
"parameter": "value"
}
TOOL_CALL: tool_name_2 (call_2)
{
"parameter": "value"
}

WRONG: "TOOL CALL: tool_name" (missing underscore and call ID)
WRONG: "**TOOL_CALL**: tool_name" (has markdown asterisks)
RIGHT: "TOOL_CALL: tool_name (call_1)" (plain text with underscore and call ID)
"""

        tool_descriptions = "\n\n".join(descriptions)
        return tool_descriptions + "\n\n" + tool_call_instructions

    def execute(
        self,
        tool_name: str,
        tool_id: str,
        input_data: Dict[str, Any],
        progress_callback: Callable[[Any], None],
    ) -> ToolResult:
        """Execute a tool with validated input and progress callback"""
        tool = self.get_tool(tool_name)
        if not tool:
            # This shouldn't happen if has_tool() was checked first
            raise RuntimeError(f"Tool '{tool_name}' not found - check has_tool() first")

        try:
            # Validate input against schema
            validated_input = tool.input_schema(**input_data)

            # Execute tool with agent and validated data
            return tool.run(self.agent, validated_input, tool_id, progress_callback)
        except Exception as e:
            raise ToolExecutionError(f"Error executing tool '{tool_name}': {str(e)}")
