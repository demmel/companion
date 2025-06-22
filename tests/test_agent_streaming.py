"""
Tests for agent streaming with tool execution loops
"""

import pytest
from unittest.mock import Mock

from agent.core import Agent
from agent.agent_events import (
    AgentTextEvent,
    ToolStartedEvent,
    ToolProgressEvent,
    ToolFinishedEvent,
    AgentErrorEvent,
)


class TestAgentStreaming:
    """Test agent streaming with tool execution loops"""

    def test_tool_only_then_dialogue_flow(self):
        """Test that agent correctly handles tool-only response followed by dialogue"""
        # Create agent with mocked LLM
        agent = Agent(model="test-model", verbose=False)

        # Mock the LLM client to return specific responses
        mock_llm = Mock()
        agent.llm = mock_llm

        # First LLM call: tool-only response
        first_response = [
            {"message": {"content": "TOOL_CALL: assume_character (call_1)\n"}},
            {
                "message": {
                    "content": '{"character_name": "Jim", "personality": "Gruff but caring"}'
                }
            },
        ]

        # Second LLM call: dialogue response (after tool execution)
        second_response = [
            {"message": {"content": '"Hey there, '}},
            {"message": {"content": "what do you "}},
            {"message": {"content": 'want?"'}},
        ]

        # Mock LLM to return different responses on each call
        mock_llm.chat.side_effect = [iter(first_response), iter(second_response)]

        # Mock tool execution
        agent.tools.execute = Mock(return_value="Character Jim created successfully")

        # Collect all events from streaming
        events = list(agent.chat_stream("Hi, Jim"))

        # Verify exact event sequence
        expected_sequence = [
            ToolStartedEvent,  # Tool execution started
            ToolFinishedEvent,  # Tool execution completed
            AgentTextEvent,  # Start of dialogue text
            AgentTextEvent,  # Continued dialogue text
            AgentTextEvent,  # End of dialogue text
        ]

        actual_sequence = [type(event) for event in events]
        assert actual_sequence == expected_sequence

        # Verify tool started details
        tool_started = events[0]
        assert tool_started.tool_name == "assume_character"
        assert tool_started.tool_id == "call_1"
        assert tool_started.parameters == {
            "character_name": "Jim",
            "personality": "Gruff but caring",
        }

        # Verify tool finished
        tool_finished = events[1]
        assert tool_finished.tool_id == "call_1"
        assert "Character Jim created" in tool_finished.result

        # Verify dialogue text
        text_events = events[2:5]
        full_text = "".join(e.content for e in text_events)
        assert full_text == '"Hey there, what do you want?"'

        # Verify LLM was called exactly twice
        assert mock_llm.chat.call_count == 2

        # Verify conversation history has correct structure
        assert len(agent.conversation_history) == 4
        assert agent.conversation_history[0].content == "Hi, Jim"
        assert "TOOL_CALL:" in agent.conversation_history[1].content
        assert "Tool results:" in agent.conversation_history[2].content
        assert "Hey there" in agent.conversation_history[3].content

    def test_no_tools_simple_response(self):
        """Test that simple responses without tools work correctly"""
        agent = Agent(model="test-model", verbose=False)

        mock_llm = Mock()
        agent.llm = mock_llm

        simple_response = [
            {"message": {"content": "Hello! "}},
            {"message": {"content": "How are you?"}},
        ]

        mock_llm.chat.return_value = iter(simple_response)

        events = list(agent.chat_stream("Hello"))

        # Should be exactly 2 text events in sequence
        expected_sequence = [AgentTextEvent, AgentTextEvent]
        actual_sequence = [type(event) for event in events]
        assert actual_sequence == expected_sequence

        # Verify content
        assert events[0].content == "Hello! "
        assert events[1].content == "How are you?"

        # Should only call LLM once
        assert mock_llm.chat.call_count == 1

    @pytest.mark.parametrize("max_iterations", [3, 5])
    def test_iteration_limit_enforced(self, max_iterations):
        """Test that agent respects the configured iteration limit"""
        agent = Agent(model="test-model", verbose=False)

        # Mock the config's max_iterations for this test
        agent.config.max_iterations = max_iterations

        mock_llm = Mock()
        agent.llm = mock_llm

        # Tool response for each iteration (complete tool call)
        tool_response = [
            {"message": {"content": "TOOL_CALL: set_mood (call_1)\n"}},
            {"message": {"content": '{"mood": "happy"}'}},
        ]

        final_response = [
            {"message": {"content": "Hello there!"}},
        ]

        # Provide tool responses for all but final iteration, then text response
        # This tests that agent naturally stops at max_iterations
        responses = []
        for i in range(max_iterations - 1):
            responses.append(iter(tool_response))
        responses.append(iter(final_response))
        mock_llm.chat.side_effect = responses

        agent.tools.execute = Mock(return_value="Mood set")

        events = list(agent.chat_stream("Hello"))

        # Should call LLM exactly max_iterations times
        assert mock_llm.chat.call_count == max_iterations

    def test_no_tools_on_final_iteration(self):
        """Test that tools are not available on the final iteration"""
        agent = Agent(model="test-model", verbose=False)

        # Set a known iteration count for this test
        test_max_iterations = 3
        agent.config.max_iterations = test_max_iterations

        mock_llm = Mock()
        agent.llm = mock_llm

        # Tool responses for non-final iterations
        tool_response = [
            {"message": {"content": "TOOL_CALL: set_mood (call_1)\n"}},
            {"message": {"content": '{"mood": "happy"}'}},
        ]

        # Final response: just text (no tools should be available)
        final_response = [
            {"message": {"content": '"Hello there!"'}},
        ]

        # Create responses: tools for first iterations, then final response
        responses = []
        for i in range(test_max_iterations - 1):
            responses.append(iter(tool_response))
        responses.append(iter(final_response))
        mock_llm.chat.side_effect = responses

        agent.tools.execute = Mock(return_value="Mood set")

        events = list(agent.chat_stream("Hello"))

        # Should call LLM exactly test_max_iterations times
        assert mock_llm.chat.call_count == test_max_iterations

        # Final call should not have tools in system prompt
        final_call_args = mock_llm.chat.call_args_list[test_max_iterations - 1][
            0
        ]  # Get positional args (final call)
        final_messages = final_call_args[0]  # First positional arg is the messages list
        final_system_prompt = final_messages[0].content
        assert (
            "Available roleplay tools:" not in final_system_prompt
            or "Available roleplay tools:\n\n" in final_system_prompt
        )

    def test_iteration_info_in_prompt(self):
        """Test that iteration information is included in the prompt"""
        agent = Agent(model="test-model", verbose=False)

        mock_llm = Mock()
        agent.llm = mock_llm

        # Simple text response
        text_response = [{"message": {"content": "Hello!"}}]
        mock_llm.chat.return_value = iter(text_response)

        list(agent.chat_stream("Hello"))

        # Check that system prompt contains iteration info
        call_args = mock_llm.chat.call_args[0]  # Get positional args
        messages = call_args[0]  # First positional arg is the messages list
        system_prompt = messages[0].content
        # Check that system prompt contains iteration info (format: TURN X/Y)
        import re

        turn_match = re.search(r"TURN (\d+)/(\d+)", system_prompt)
        assert (
            turn_match is not None
        ), f"Expected TURN X/Y format in system prompt, got: {system_prompt[:200]}..."
        assert turn_match.group(1) == "1"  # First turn
