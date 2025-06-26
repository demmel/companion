"""
Test configurations for agent testing.
These are minimal, focused configs without production complexity.
"""

from agent.config import AgentConfig
from agent.tools import BaseTool, ToolInput
from pydantic import Field


class MockToolInput(ToolInput):
    """Mock tool input for testing"""
    message: str = Field(description="Test message")


class MockTool(BaseTool):
    """Mock tool that just returns a predictable result"""
    
    @property
    def name(self) -> str:
        return "mock_tool"
    
    @property
    def description(self) -> str:
        return "Mock tool for testing"
    
    @property
    def input_schema(self):
        return MockToolInput
    
    def run(self, agent, input_data):
        return f"Mock result: {input_data.message}"


class StatefulMockTool(BaseTool):
    """Mock tool that modifies agent state"""
    
    @property
    def name(self) -> str:
        return "stateful_mock_tool"
    
    @property
    def description(self) -> str:
        return "Mock tool that modifies state"
    
    @property
    def input_schema(self):
        return MockToolInput
    
    def run(self, agent, input_data):
        agent.set_state("last_tool_message", input_data.message)
        return f"State updated: {input_data.message}"


class FailingMockTool(BaseTool):
    """Mock tool that always fails for error testing"""
    
    @property
    def name(self) -> str:
        return "failing_tool"
    
    @property
    def description(self) -> str:
        return "Tool that always fails"
    
    @property
    def input_schema(self):
        return MockToolInput
    
    def run(self, agent, input_data):
        raise RuntimeError(f"Tool failed with: {input_data.message}")


# Test configurations

def create_empty_config():
    """Config with no tools for basic testing"""
    return AgentConfig(
        name="empty",
        description="Empty config for basic testing",
        prompt_template="You are a test assistant.\n{tools_description}\n{state_info}\n{iteration_info}",
        tools=[],
        default_state={"test_mode": True}
    )


def create_simple_config():
    """Config with one simple tool"""
    return AgentConfig(
        name="simple",
        description="Simple config with one tool",
        prompt_template="You are a test assistant with tools.\n{tools_description}\n{state_info}\n{iteration_info}",
        tools=[MockTool()],
        default_state={"test_mode": True}
    )


def create_multi_tool_config():
    """Config with multiple tools for complex testing"""
    return AgentConfig(
        name="multi_tool",
        description="Config with multiple tools",
        prompt_template="You are a test assistant with multiple tools.\n{tools_description}\n{state_info}\n{iteration_info}",
        tools=[MockTool(), StatefulMockTool(), FailingMockTool()],
        default_state={"test_mode": True}
    )


def create_stateful_config():
    """Config focused on state management testing"""
    return AgentConfig(
        name="stateful",
        description="Config for state testing",
        prompt_template="You are a stateful test assistant.\n{tools_description}\n{state_info}\n{iteration_info}",
        tools=[StatefulMockTool()],
        default_state={
            "test_mode": True,
            "initial_value": "test",
            "counter": 0
        }
    )


def create_iteration_test_config():
    """Config with specific iteration limit for testing"""
    return AgentConfig(
        name="iteration_test",
        description="Config for iteration testing",
        prompt_template="You are a test assistant.\n{tools_description}\n{state_info}\n{iteration_info}",
        tools=[MockTool()],
        default_state={"test_mode": True},
        max_iterations=2  # Low limit for testing
    )