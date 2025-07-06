#!/usr/bin/env python3
"""
Test script for the new callback-based tool interface
"""

from agent.tools import (
    BaseTool,
    ToolInput,
    ToolResult,
)
from typing import Type, Callable, Any
from pydantic import Field

from agent.types import TextToolContent, ToolCallError, ToolCallSuccess


# Simple test tool for interface testing
class TestToolInput(ToolInput):
    message: str = Field(description="Test message")
    should_error: bool = Field(default=False, description="Whether to return an error")


class TestTool(BaseTool):
    """Simple tool for testing the callback interface"""

    @property
    def name(self) -> str:
        return "test_tool"

    @property
    def description(self) -> str:
        return "A simple tool for testing"

    @property
    def input_schema(self) -> Type[ToolInput]:
        return TestToolInput

    def run(
        self,
        agent,
        input_data: TestToolInput,
        tool_id: str,
        progress_callback: Callable[[Any], None],
    ) -> ToolResult:
        # Test progress callbacks
        progress_callback({"stage": "starting", "progress": 0.0})
        progress_callback({"stage": "processing", "progress": 0.5})
        progress_callback({"stage": "finishing", "progress": 1.0})

        if input_data.should_error:
            return ToolCallError(error="Test error message")
        else:
            return ToolCallSuccess(
                content=TextToolContent(text=f"Processed: {input_data.message}")
            )


def test_callback_tool_interface():
    """Test that the new callback-based tool interface works correctly"""

    # Create a test tool instance
    tool = TestTool()

    # Create mock agent
    class MockAgent:
        def __init__(self):
            self.state = {}

        def get_state(self, key):
            return self.state.get(key)

        def set_state(self, key, value):
            self.state[key] = value

    agent = MockAgent()

    # Create input data for success case
    input_data = TestToolInput(message="Hello World", should_error=False)

    # Test the callback interface
    progress_calls = []

    def progress_callback(data):
        progress_calls.append(data)
        print(f"Got progress callback: {data}")

    result = tool.run(agent, input_data, "test_tool_id", progress_callback)

    print(f"Debug - progress_calls: {progress_calls}")
    print(f"Debug - result: {result}")

    # Assertions
    assert (
        len(progress_calls) == 3
    ), f"Expected 3 progress calls, got {len(progress_calls)}"
    assert (
        progress_calls[0]["stage"] == "starting"
    ), f"First progress should be 'starting', got {progress_calls[0]}"
    assert (
        progress_calls[1]["stage"] == "processing"
    ), f"Second progress should be 'processing', got {progress_calls[1]}"
    assert (
        progress_calls[2]["stage"] == "finishing"
    ), f"Third progress should be 'finishing', got {progress_calls[2]}"

    assert result is not None, "Tool should return a result"
    assert isinstance(
        result, ToolCallSuccess
    ), f"Result should be ToolCallSuccess, got {type(result)}"
    assert (
        result.type == "success"
    ), f"Result type should be 'success', got {result.type}"
    assert (
        result.content.type == "text"
    ), f"Content type should be 'text', got {result.content.type}"
    assert (
        result.content.text == "Processed: Hello World"
    ), f"Expected specific content, got: {result.content}"

    print("✅ Success case passed!")


def test_error_case():
    """Test error handling in callback interface"""

    tool = TestTool()

    class MockAgent:
        def __init__(self):
            self.state = {}

        def get_state(self, key):
            return self.state.get(key)

        def set_state(self, key, value):
            self.state[key] = value

    agent = MockAgent()
    input_data = TestToolInput(message="Error test", should_error=True)

    progress_calls = []

    def progress_callback(data):
        progress_calls.append(data)

    result = tool.run(agent, input_data, "test_tool_id", progress_callback)

    # Assertions
    assert (
        len(progress_calls) == 3
    ), f"Expected 3 progress calls even on error, got {len(progress_calls)}"
    assert isinstance(
        result, ToolCallError
    ), f"Result should be ToolCallError, got {type(result)}"
    assert result.type == "error", f"Result type should be 'error', got {result.type}"
    assert (
        result.error == "Test error message"
    ), f"Expected specific error, got: {result.error}"

    print("✅ Error case passed!")


if __name__ == "__main__":
    test_callback_tool_interface()
    test_error_case()
    print("🎉 All callback-based tool tests completed successfully!")
