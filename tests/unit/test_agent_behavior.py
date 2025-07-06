"""
Behavior-focused integration tests that test complete user workflows.
These tests focus on what the agent does, not how it does it.
Tests core agent behavior independent of specific configurations.
"""

from unittest.mock import Mock

from agent.core import Agent
from agent.agent_events import (
    AgentTextEvent,
    ToolStartedEvent,
    ToolFinishedEvent,
    AgentErrorEvent,
)
from test_configs import (
    create_empty_config,
    create_simple_config,
    create_multi_tool_config,
    create_stateful_config,
    create_iteration_test_config,
)


class TestAgentBehavior:
    """Test complete agent workflows and behaviors"""

    def mock_llm_stream(self, agent, chunks):
        """Helper to mock LLM streaming responses"""

        def stream_generator():
            for chunk in chunks:
                yield {"message": {"content": chunk}}

        agent.llm.chat.return_value = stream_generator()

    def collect_stream_events(self, agent, user_input):
        """Helper to collect all events from a stream"""
        events = []
        for event in agent.chat_stream(user_input):
            events.append(event)
        return events

    def get_text_content(self, events):
        """Extract all text content from events"""
        text_parts = []
        for event in events:
            if isinstance(event, AgentTextEvent):
                text_parts.append(event.content)
        return "".join(text_parts)

    def get_tool_calls(self, events):
        """Extract tool call information from events"""
        tool_calls = []
        for event in events:
            if isinstance(event, ToolStartedEvent):
                tool_calls.append(
                    {"name": event.tool_name, "parameters": event.parameters}
                )
        return tool_calls

    def get_error_events(self, events):
        """Extract error events"""
        return [event for event in events if isinstance(event, AgentErrorEvent)]

    def test_text_response_streaming(self):
        """Test that agent delivers text responses via streaming events"""
        config = create_empty_config()
        agent = Agent(config=config, model="test-model", verbose=False)
        agent.llm = Mock()

        expected_response = "Hello, how can I help you today?"
        self.mock_llm_stream(agent, [expected_response])

        events = self.collect_stream_events(agent, "Hello")
        text = self.get_text_content(events)

        assert text == expected_response

    def test_chunked_text_streaming(self):
        """Test that text is delivered in separate streaming chunks"""
        config = create_empty_config()
        agent = Agent(config=config, model="test-model", verbose=False)
        agent.llm = Mock()

        chunks = ["Hello ", "there, ", "how ", "are ", "you?"]
        self.mock_llm_stream(agent, chunks)

        events = self.collect_stream_events(agent, "Hi")
        text_events = [e for e in events if isinstance(e, AgentTextEvent)]

        # Should receive each chunk as separate event
        assert len(text_events) == len(chunks)
        for i, event in enumerate(text_events):
            assert event.content == chunks[i]

        # Combined content should be complete
        full_text = self.get_text_content(events)
        assert full_text == "Hello there, how are you?"

    def test_tool_call_detection_and_execution(self):
        """Test that agent detects and executes tool calls"""
        config = create_simple_config()
        agent = Agent(config=config, model="test-model", verbose=False)
        agent.llm = Mock()

        self.mock_llm_stream(
            agent,
            [
                "Working on it. ",
                "TOOL_CALL: mock_tool (call_1)\n",
                '{"message": "test input"}',
            ],
        )

        events = self.collect_stream_events(agent, "Do something")
        tool_calls = self.get_tool_calls(events)
        text = self.get_text_content(events)

        # Should execute the tool call
        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == "mock_tool"
        assert tool_calls[0]["parameters"] == {"message": "test input"}

        # Should deliver text content
        assert text == "Working on it. "

    def test_multiple_tool_execution(self):
        """Test that multiple tools can be executed in sequence"""
        config = create_multi_tool_config()
        agent = Agent(config=config, model="test-model", verbose=False)
        agent.llm = Mock()

        self.mock_llm_stream(
            agent,
            [
                "TOOL_CALL: mock_tool (call_1)\n",
                '{"message": "first"} ',
                "TOOL_CALL: stateful_mock_tool (call_2)\n",
                '{"message": "second"}',
            ],
        )

        events = self.collect_stream_events(agent, "Use multiple tools")
        tool_calls = self.get_tool_calls(events)

        # Should execute both tools in order
        assert len(tool_calls) == 2
        assert tool_calls[0]["name"] == "mock_tool"
        assert tool_calls[0]["parameters"] == {"message": "first"}
        assert tool_calls[1]["name"] == "stateful_mock_tool"
        assert tool_calls[1]["parameters"] == {"message": "second"}

    def test_agent_state_api(self):
        """Test that agent state can be get and set via public API"""
        config = create_empty_config()
        agent = Agent(config=config, model="test-model", verbose=False)

        # Test setting and getting specific key
        agent.set_state("test_key", "test_value")
        assert agent.get_state("test_key") == "test_value"

        # Test getting full state
        full_state = agent.get_state()
        assert isinstance(full_state, dict)
        assert full_state["test_key"] == "test_value"

        # Test getting non-existent key returns None
        assert agent.get_state("nonexistent") is None

    def test_invalid_tool_error_handling(self):
        """Test that invalid tool calls generate appropriate error events"""
        config = create_simple_config()
        agent = Agent(config=config, model="test-model", verbose=False)
        agent.llm = Mock()

        self.mock_llm_stream(
            agent,
            [
                "Trying something. ",
                "TOOL_CALL: nonexistent_tool (call_1)\n",
                '{"param": "value"}',
            ],
        )

        events = self.collect_stream_events(agent, "Do something")
        error_events = self.get_error_events(events)
        tool_started_events = [e for e in events if isinstance(e, ToolStartedEvent)]
        text = self.get_text_content(events)

        # Should generate error event for invalid tool
        assert len(error_events) == 1
        assert "nonexistent_tool" in error_events[0].message
        assert "not found" in error_events[0].message

        # Should NOT emit ToolStartedEvent for nonexistent tool
        assert len(tool_started_events) == 0

        # Should still deliver text content
        assert text == "Trying something. "

    def test_malformed_tool_call_error_handling(self):
        """Test that malformed tool calls are handled gracefully"""
        config = create_simple_config()
        agent = Agent(config=config, model="test-model", verbose=False)
        agent.llm = Mock()

        self.mock_llm_stream(
            agent,
            [
                "Working. ",
                "TOOL_CALL: mock_tool (call_1)\n",
                '{"message": "test", invalid',  # Malformed JSON
            ],
        )

        events = self.collect_stream_events(agent, "Test")
        error_events = self.get_error_events(events)

        # Should generate error for malformed JSON
        assert len(error_events) > 0

        # Should still deliver text content
        assert self.get_text_content(events) == "Working. "

    def test_iteration_limit_enforcement(self):
        """Test that agent respects configured iteration limits"""
        config = create_iteration_test_config()
        agent = Agent(config=config, model="test-model", verbose=False)
        agent.llm = Mock()

        call_count = 0

        def mock_responses(*args, **kwargs):
            nonlocal call_count
            call_count += 1

            if call_count >= agent.config.max_iterations:
                # Final iteration - no tools
                return iter([{"message": {"content": f"Final response {call_count}"}}])
            else:
                # Return tool call to trigger next iteration
                return iter(
                    [
                        {"message": {"content": f"Iteration {call_count} "}},
                        {"message": {"content": "TOOL_CALL: mock_tool (call_1)\n"}},
                        {
                            "message": {
                                "content": f'{{"message": "iteration {call_count}"}}'
                            }
                        },
                    ]
                )

        agent.llm.chat.side_effect = mock_responses

        events = self.collect_stream_events(agent, "Keep going")
        tool_calls = self.get_tool_calls(events)

        # Should stop at max iterations
        assert call_count == agent.config.max_iterations
        assert (
            len(tool_calls) == agent.config.max_iterations - 1
        )  # Final iteration has no tools

    def test_tool_state_modification(self):
        """Test that tools can modify agent state"""
        config = create_stateful_config()
        agent = Agent(config=config, model="test-model", verbose=False)
        agent.llm = Mock()

        # Check initial state
        assert agent.get_state("last_tool_message") is None
        assert agent.get_state("initial_value") == "test"

        self.mock_llm_stream(
            agent,
            [
                "TOOL_CALL: stateful_mock_tool (call_1)\n",
                '{"message": "state_change_test"}',
            ],
        )

        self.collect_stream_events(agent, "Change state")

        # State should be modified by tool
        assert agent.get_state("last_tool_message") == "state_change_test"
        assert agent.get_state("initial_value") == "test"  # Should be unchanged

    def test_context_info_api(self):
        """Test that context info API returns meaningful data"""
        config = create_empty_config()
        agent = Agent(config=config, model="test-model", verbose=False)
        agent.llm = Mock()

        # Fresh agent should have minimal context
        context_info = agent.get_context_info()
        assert context_info.conversation_messages == 0
        assert context_info.message_count >= 1  # At least system prompt
        assert context_info.context_limit > 0
        assert context_info.usage_percentage >= 0
        assert context_info.estimated_tokens > 0  # System prompt has tokens

        # After conversation, context should grow
        self.mock_llm_stream(agent, ["Response"])
        self.collect_stream_events(agent, "Message")

        updated_context = agent.get_context_info()
        assert (
            updated_context.conversation_messages > context_info.conversation_messages
        )
        assert updated_context.message_count > context_info.message_count
        assert updated_context.estimated_tokens > context_info.estimated_tokens

    def test_conversation_reset_api(self):
        """Test that conversation reset API works"""
        config = create_empty_config()
        agent = Agent(config=config, model="test-model", verbose=False)
        agent.llm = Mock()

        # Have a conversation to create some context
        self.mock_llm_stream(agent, ["Response"])
        self.collect_stream_events(agent, "Message")

        # Context should show some usage
        context_before = agent.get_context_info()
        assert context_before.conversation_messages > 0

        # Reset conversation
        agent.reset_conversation()

        # Context should be cleared
        context_after = agent.get_context_info()
        assert context_after.conversation_messages == 0

    def test_configuration_integration(self):
        """Test that agent properly integrates with its configuration"""
        config = create_multi_tool_config()
        agent = Agent(config=config, model="test-model", verbose=False)

        # Should use provided configuration
        assert agent.config is config
        assert agent.config.name == "multi_tool"

        # Should initialize tools from config
        assert agent.tools is not None
        assert len(agent.tools.tools) == 3  # multi_tool_config has 3 tools

        # Should have the expected tools available
        tool_names = [tool.name for tool in agent.tools.tools.values()]
        assert "mock_tool" in tool_names
        assert "stateful_mock_tool" in tool_names
        assert "failing_tool" in tool_names

    def test_default_state_initialization(self):
        """Test that agent initializes with configuration's default state"""
        config = create_stateful_config()
        agent = Agent(config=config, model="test-model", verbose=False)

        # Should have default state from config
        assert agent.get_state("test_mode") is True
        assert agent.get_state("initial_value") == "test"
        assert agent.get_state("counter") == 0

    def test_tool_execution_failure_handling(self):
        """Test that tool execution failures are handled gracefully"""
        config = create_multi_tool_config()
        agent = Agent(config=config, model="test-model", verbose=False)
        agent.llm = Mock()

        self.mock_llm_stream(
            agent,
            [
                "Trying to fail. ",
                "TOOL_CALL: failing_tool (call_1)\n",
                '{"message": "cause failure"}',
            ],
        )

        events = self.collect_stream_events(agent, "Make it fail")
        error_events = self.get_error_events(events)
        text = self.get_text_content(events)

        # Should emit ToolStartedEvent (tool exists)
        tool_started_events = [e for e in events if isinstance(e, ToolStartedEvent)]
        tool_finished_events = [e for e in events if isinstance(e, ToolFinishedEvent)]

        assert len(tool_started_events) == 1
        assert tool_started_events[0].tool_name == "failing_tool"

        # Should handle tool failure with error result type
        assert len(tool_finished_events) == 1
        assert tool_finished_events[0].result.type == "error"
        assert "cause failure" in tool_finished_events[0].result.error

        # Should still deliver text content
        assert text == "Trying to fail. "

    def test_state_persistence_across_interactions(self):
        """Test that state persists across multiple interactions"""
        config = create_stateful_config()
        agent = Agent(config=config, model="test-model", verbose=False)
        agent.llm = Mock()

        # Set some state
        agent.set_state("persistent_key", "persistent_value")

        # First interaction
        self.mock_llm_stream(agent, ["First"])
        self.collect_stream_events(agent, "First")
        assert agent.get_state("persistent_key") == "persistent_value"

        # Second interaction
        self.mock_llm_stream(agent, ["Second"])
        self.collect_stream_events(agent, "Second")
        assert agent.get_state("persistent_key") == "persistent_value"

    def test_event_types_are_correct(self):
        """Test that agent emits correct event types"""
        config = create_simple_config()
        agent = Agent(config=config, model="test-model", verbose=False)
        agent.llm = Mock()

        self.mock_llm_stream(
            agent,
            [
                "Text before tool. ",
                "TOOL_CALL: mock_tool (call_1)\n",
                '{"message": "test"} ',
                "Text after tool.",
            ],
        )

        events = self.collect_stream_events(agent, "Test all event types")

        # Should have text events
        text_events = [e for e in events if isinstance(e, AgentTextEvent)]
        assert len(text_events) > 0

        # Should have tool events
        tool_started_events = [e for e in events if isinstance(e, ToolStartedEvent)]
        tool_finished_events = [e for e in events if isinstance(e, ToolFinishedEvent)]
        assert len(tool_started_events) > 0
        assert len(tool_finished_events) > 0

        # Tool events should match
        assert len(tool_started_events) == len(tool_finished_events)
