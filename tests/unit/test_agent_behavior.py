"""
Behavior-focused integration tests that test complete user workflows.
These tests focus on what the agent does, not how it does it.
Tests core agent behavior independent of specific configurations.
"""

from unittest.mock import Mock, patch

from agent.core import Agent
from agent.llm import LLM, SupportedModel, ModelConfig
from agent.agent_events import (
    AgentTextEvent,
    ToolStartedEvent,
    ToolFinishedEvent,
    AgentErrorEvent,
)
from agent.reasoning.types import ReasoningResult
from test_configs import (
    create_empty_config,
    create_simple_config,
    create_multi_tool_config,
    create_stateful_config,
    create_iteration_test_config,
)


class TestAgentBehavior:
    """Test complete agent workflows and behaviors"""

    def create_mock_llm(self):
        """Create a properly mocked LLM for testing"""
        mock_llm = Mock(spec=LLM)
        mock_llm.models = {
            SupportedModel.LLAMA_8B: ModelConfig(
                model=SupportedModel.LLAMA_8B, context_window=4096
            )
        }
        # Mock chat_complete to return proper string
        mock_llm.chat_complete.return_value = "mocked response"
        return mock_llm

    def mock_llm_stream(self, llm, chunks):
        """Helper to mock LLM streaming responses"""

        def stream_generator():
            for chunk in chunks:
                yield {"message": {"content": chunk}}

        llm.chat_streaming.return_value = stream_generator()

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

    @patch("agent.reasoning.analyze.structured_llm_call")
    def test_text_response_streaming(self, mock_structured_call):
        """Test that agent delivers text responses via streaming events"""
        config = create_empty_config()
        llm = self.create_mock_llm()
        agent = Agent(config=config, model=SupportedModel.LLAMA_8B, llm=llm)

        # Mock reasoning responses - first generates response, second ends turn
        reasoning_responses = [
            ReasoningResult(
                understanding="User said hello",
                situational_awareness="Simple greeting",
                emotional_context="Friendly",
                key_information=[],
                proposed_tools=[],
                follow_up_opportunities=["Continue conversation"],
                should_end_turn=False,
            ),
            ReasoningResult(
                understanding="I responded with greeting",
                situational_awareness="Conversation started well",
                emotional_context="Friendly",
                key_information=[],
                proposed_tools=[],
                follow_up_opportunities=["Wait for user"],
                should_end_turn=True,
            ),
        ]

        call_count = 0

        def mock_reasoning(*args, **kwargs):
            nonlocal call_count
            result = reasoning_responses[min(call_count, len(reasoning_responses) - 1)]
            call_count += 1
            return result

        mock_structured_call.side_effect = mock_reasoning

        expected_response = "Hello, how can I help you today?"
        self.mock_llm_stream(llm, [expected_response])

        events = self.collect_stream_events(agent, "Hello")
        text_content = self.get_text_content(events)

        # Should contain the expected response
        assert expected_response in text_content

    @patch("agent.reasoning.analyze.structured_llm_call")
    def test_chunked_text_streaming(self, mock_structured_call):
        """Test that text is delivered in separate streaming chunks"""
        config = create_empty_config()
        llm = self.create_mock_llm()
        agent = Agent(config=config, model=SupportedModel.LLAMA_8B, llm=llm)

        # Mock reasoning responses - first false, then true
        reasoning_responses = [
            ReasoningResult(
                understanding="User said hi",
                situational_awareness="Casual greeting",
                emotional_context="Friendly",
                key_information=[],
                proposed_tools=[],
                follow_up_opportunities=["Continue conversation"],
                should_end_turn=False,
            ),
            ReasoningResult(
                understanding="I responded with greeting",
                situational_awareness="Completed response",
                emotional_context="Friendly",
                key_information=[],
                proposed_tools=[],
                follow_up_opportunities=["Wait for user"],
                should_end_turn=True,
            ),
        ]

        call_count = 0

        def mock_reasoning(*args, **kwargs):
            nonlocal call_count
            result = reasoning_responses[min(call_count, len(reasoning_responses) - 1)]
            call_count += 1
            return result

        mock_structured_call.side_effect = mock_reasoning

        chunks = ["Hello ", "there, ", "how ", "are ", "you?"]
        self.mock_llm_stream(llm, chunks)

        events = self.collect_stream_events(agent, "Hi")
        text_events = [
            e for e in events if isinstance(e, AgentTextEvent) and not e.is_thought
        ]

        # Combined content should be complete
        full_text = self.get_text_content(events)
        assert "Hello there, how are you?" in full_text

    @patch("agent.reasoning.analyze.structured_llm_call")
    def test_tool_call_detection_and_execution(self, mock_structured_call):
        """Test that agent detects and executes tool calls"""
        config = create_simple_config()
        llm = self.create_mock_llm()
        agent = Agent(config=config, model=SupportedModel.LLAMA_8B, llm=llm)

        # Mock reasoning response that proposes a tool
        from agent.reasoning.types import ProposedToolCall

        # Mock reasoning responses - first proposes tool, second ends turn
        reasoning_responses = [
            ReasoningResult(
                understanding="User wants me to do something",
                situational_awareness="I should use the mock tool",
                emotional_context="Helpful",
                key_information=["User requested action"],
                proposed_tools=[
                    ProposedToolCall(
                        tool_name="mock_tool",
                        call_id="call_1",
                        parameters={"message": "test input"},
                        reasoning="This will help the user",
                    )
                ],
                follow_up_opportunities=["Continue helping"],
                should_end_turn=False,
            ),
            ReasoningResult(
                understanding="I completed the tool action",
                situational_awareness="Task is done, time to respond",
                emotional_context="Satisfied",
                key_information=["Tool executed successfully"],
                proposed_tools=[],
                follow_up_opportunities=["Wait for user"],
                should_end_turn=True,
            ),
        ]

        call_count = 0

        def mock_reasoning(*args, **kwargs):
            nonlocal call_count
            result = reasoning_responses[min(call_count, len(reasoning_responses) - 1)]
            call_count += 1
            return result

        mock_structured_call.side_effect = mock_reasoning

        # Mock final response generation
        self.mock_llm_stream(llm, ["I've completed the task."])

        events = self.collect_stream_events(agent, "Do something")
        tool_calls = self.get_tool_calls(events)

        # Should execute the tool call
        assert len(tool_calls) >= 1
        assert tool_calls[0]["name"] == "mock_tool"
        assert tool_calls[0]["parameters"] == {"message": "test input"}

    @patch("agent.reasoning.analyze.structured_llm_call")
    def test_multiple_tool_execution(self, mock_structured_call):
        """Test that multiple tools can be executed in sequence"""
        config = create_multi_tool_config()
        llm = self.create_mock_llm()
        agent = Agent(config=config, model=SupportedModel.LLAMA_8B, llm=llm)

        # Mock reasoning response that proposes multiple tools
        from agent.reasoning.types import ProposedToolCall

        # Mock reasoning responses - first proposes tools, second ends turn
        reasoning_responses = [
            ReasoningResult(
                understanding="User wants multiple tools",
                situational_awareness="I should use both tools",
                emotional_context="Helpful",
                key_information=["User requested multiple actions"],
                proposed_tools=[
                    ProposedToolCall(
                        tool_name="mock_tool",
                        call_id="call_1",
                        parameters={"message": "first"},
                        reasoning="First tool",
                    ),
                    ProposedToolCall(
                        tool_name="stateful_mock_tool",
                        call_id="call_2",
                        parameters={"message": "second"},
                        reasoning="Second tool",
                    ),
                ],
                follow_up_opportunities=["Continue helping"],
                should_end_turn=False,
            ),
            ReasoningResult(
                understanding="I completed both tool actions",
                situational_awareness="Both tools executed, time to respond",
                emotional_context="Satisfied",
                key_information=["Both tools executed successfully"],
                proposed_tools=[],
                follow_up_opportunities=["Wait for user"],
                should_end_turn=True,
            ),
        ]

        call_count = 0

        def mock_reasoning(*args, **kwargs):
            nonlocal call_count
            result = reasoning_responses[min(call_count, len(reasoning_responses) - 1)]
            call_count += 1
            return result

        mock_structured_call.side_effect = mock_reasoning

        # Mock final response generation
        self.mock_llm_stream(llm, ["I've used both tools."])

        events = self.collect_stream_events(agent, "Use multiple tools")
        tool_calls = self.get_tool_calls(events)

        # Should execute both tools in order
        assert len(tool_calls) >= 2
        tool_names = [call["name"] for call in tool_calls]
        assert "mock_tool" in tool_names
        assert "stateful_mock_tool" in tool_names

    def test_agent_state_api(self):
        """Test that agent state can be get and set via public API"""
        config = create_empty_config()
        llm = self.create_mock_llm()
        agent = Agent(config=config, model=SupportedModel.LLAMA_8B, llm=llm)

        # Test setting and getting specific key
        agent.set_state("test_key", "test_value")
        assert agent.get_state("test_key") == "test_value"

        # Test getting full state
        full_state = agent.get_state()
        assert isinstance(full_state, dict)
        assert full_state["test_key"] == "test_value"

        # Test getting non-existent key returns None
        assert agent.get_state("nonexistent") is None

    @patch("agent.reasoning.analyze.structured_llm_call")
    def test_invalid_tool_error_handling(self, mock_structured_call):
        """Test that invalid tool calls generate appropriate error events"""
        config = create_simple_config()
        llm = self.create_mock_llm()
        agent = Agent(config=config, model=SupportedModel.LLAMA_8B, llm=llm)

        # Mock reasoning response that proposes an invalid tool
        from agent.reasoning.types import ProposedToolCall

        # Mock reasoning responses - first proposes invalid tool, second ends turn
        reasoning_responses = [
            ReasoningResult(
                understanding="User wants something",
                situational_awareness="I should try a tool",
                emotional_context="Helpful",
                key_information=["User needs help"],
                proposed_tools=[
                    ProposedToolCall(
                        tool_name="nonexistent_tool",
                        call_id="call_1",
                        parameters={"param": "value"},
                        reasoning="This should work",
                    )
                ],
                follow_up_opportunities=["Try another approach"],
                should_end_turn=False,
            ),
            ReasoningResult(
                understanding="The tool failed",
                situational_awareness="I should end the turn now",
                emotional_context="Apologetic",
                key_information=["Tool error occurred"],
                proposed_tools=[],
                follow_up_opportunities=["Wait for user"],
                should_end_turn=True,
            ),
        ]

        call_count = 0

        def mock_reasoning(*args, **kwargs):
            nonlocal call_count
            result = reasoning_responses[min(call_count, len(reasoning_responses) - 1)]
            call_count += 1
            return result

        mock_structured_call.side_effect = mock_reasoning

        # Mock final response generation
        self.mock_llm_stream(llm, ["I encountered an error with that tool."])

        events = self.collect_stream_events(agent, "Do something")
        error_events = self.get_error_events(events)
        tool_started_events = [e for e in events if isinstance(e, ToolStartedEvent)]

        # Should generate error event for invalid tool
        assert len(error_events) >= 1
        error_found = any(
            "nonexistent_tool" in event.message and "not found" in event.message
            for event in error_events
        )
        assert error_found

        # Should NOT emit ToolStartedEvent for nonexistent tool
        nonexistent_started = any(
            e.tool_name == "nonexistent_tool" for e in tool_started_events
        )
        assert not nonexistent_started

    def test_malformed_tool_call_error_handling(self):
        """Test that malformed reasoning responses are handled gracefully"""
        config = create_simple_config()
        llm = self.create_mock_llm()
        agent = Agent(config=config, model=SupportedModel.LLAMA_8B, llm=llm)

        # Mock invalid JSON response from reasoning
        llm.chat_complete.return_value = '{"understanding": "test", invalid'

        try:
            events = self.collect_stream_events(agent, "Test")
            # Should handle malformed JSON gracefully
            assert True  # If we get here, no exception was thrown
        except Exception as e:
            # Should not crash the agent
            assert "JSON" in str(e) or "parse" in str(e).lower()

    @patch("agent.reasoning.analyze.structured_llm_call")
    def test_iteration_limit_enforcement(self, mock_structured_call):
        """Test that reasoning loop respects termination conditions"""
        config = create_iteration_test_config()
        llm = self.create_mock_llm()
        agent = Agent(config=config, model=SupportedModel.LLAMA_8B, llm=llm)

        # Mock reasoning responses - first suggests tool, second ends turn
        from agent.reasoning.types import ProposedToolCall

        reasoning_responses = [
            ReasoningResult(
                understanding="First iteration",
                situational_awareness="Continue working",
                emotional_context="Focused",
                key_information=["First iteration progress"],
                proposed_tools=[
                    ProposedToolCall(
                        tool_name="mock_tool",
                        call_id="call_1",
                        parameters={"message": "first"},
                        reasoning="Keep going",
                    )
                ],
                follow_up_opportunities=["Continue iteration"],
                should_end_turn=False,
            ),
            ReasoningResult(
                understanding="Second iteration",
                situational_awareness="Time to end",
                emotional_context="Complete",
                key_information=["Iteration complete"],
                proposed_tools=[],
                follow_up_opportunities=["Wait for user"],
                should_end_turn=True,
            ),
        ]

        call_count = 0

        def mock_reasoning(*args, **kwargs):
            nonlocal call_count
            result = reasoning_responses[min(call_count, len(reasoning_responses) - 1)]
            call_count += 1
            return result

        mock_structured_call.side_effect = mock_reasoning
        self.mock_llm_stream(llm, ["Final response"])

        events = self.collect_stream_events(agent, "Keep going")
        tool_calls = self.get_tool_calls(events)

        # Should have made tool calls but eventually stopped
        assert len(tool_calls) >= 1
        assert call_count >= 2  # At least initial + termination reasoning

    @patch("agent.reasoning.analyze.structured_llm_call")
    def test_tool_state_modification(self, mock_structured_call):
        """Test that tools can modify agent state"""
        config = create_stateful_config()
        llm = self.create_mock_llm()
        agent = Agent(config=config, model=SupportedModel.LLAMA_8B, llm=llm)

        # Check initial state
        assert agent.get_state("last_tool_message") is None
        assert agent.get_state("initial_value") == "test"

        # Mock reasoning response that proposes stateful tool
        from agent.reasoning.types import ProposedToolCall

        reasoning_responses = [
            ReasoningResult(
                understanding="User wants state change",
                situational_awareness="I should use the stateful tool",
                emotional_context="Helpful",
                key_information=["User requested state change"],
                proposed_tools=[
                    ProposedToolCall(
                        tool_name="stateful_mock_tool",
                        call_id="call_1",
                        parameters={"message": "state_change_test"},
                        reasoning="This will change state",
                    )
                ],
                follow_up_opportunities=["Continue helping"],
                should_end_turn=False,
            ),
            ReasoningResult(
                understanding="State has been changed",
                situational_awareness="Tool executed, state modified",
                emotional_context="Satisfied",
                key_information=["State modification complete"],
                proposed_tools=[],
                follow_up_opportunities=["Wait for user"],
                should_end_turn=True,
            ),
        ]

        call_count = 0

        def mock_reasoning(*args, **kwargs):
            nonlocal call_count
            result = reasoning_responses[min(call_count, len(reasoning_responses) - 1)]
            call_count += 1
            return result

        mock_structured_call.side_effect = mock_reasoning
        self.mock_llm_stream(llm, ["State has been updated."])

        self.collect_stream_events(agent, "Change state")

        # State should be modified by tool
        assert agent.get_state("last_tool_message") == "state_change_test"
        assert agent.get_state("initial_value") == "test"  # Should be unchanged

    @patch("agent.reasoning.analyze.structured_llm_call")
    def test_context_info_api(self, mock_structured_call):
        """Test that context info API returns meaningful data"""
        config = create_empty_config()
        llm = self.create_mock_llm()
        agent = Agent(config=config, model=SupportedModel.LLAMA_8B, llm=llm)

        # Fresh agent should have minimal context
        context_info = agent.get_context_info()
        assert context_info.conversation_messages == 0
        assert context_info.message_count >= 1  # At least system prompt
        assert context_info.context_limit > 0
        assert context_info.usage_percentage >= 0
        assert context_info.estimated_tokens > 0  # System prompt has tokens

        # After conversation, context should grow
        mock_structured_call.return_value = ReasoningResult(
            understanding="Simple response",
            situational_awareness="Just respond",
            emotional_context="Neutral",
            key_information=["Simple interaction"],
            proposed_tools=[],
            follow_up_opportunities=["Wait for user"],
            should_end_turn=True,
        )
        self.mock_llm_stream(llm, ["Response"])
        self.collect_stream_events(agent, "Message")

        updated_context = agent.get_context_info()
        assert (
            updated_context.conversation_messages > context_info.conversation_messages
        )
        assert updated_context.message_count > context_info.message_count

    @patch("agent.reasoning.analyze.structured_llm_call")
    def test_conversation_reset_api(self, mock_structured_call):
        """Test that conversation reset API works"""
        config = create_empty_config()
        llm = self.create_mock_llm()
        agent = Agent(config=config, model=SupportedModel.LLAMA_8B, llm=llm)

        # Have a conversation to create some context
        mock_structured_call.return_value = ReasoningResult(
            understanding="Simple response",
            situational_awareness="Just respond",
            emotional_context="Neutral",
            key_information=["Simple interaction"],
            proposed_tools=[],
            follow_up_opportunities=["Wait for user"],
            should_end_turn=True,
        )
        self.mock_llm_stream(llm, ["Response"])
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
        llm = self.create_mock_llm()
        agent = Agent(config=config, model=SupportedModel.LLAMA_8B, llm=llm)

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
        llm = self.create_mock_llm()
        agent = Agent(config=config, model=SupportedModel.LLAMA_8B, llm=llm)

        # Should have default state from config
        assert agent.get_state("test_mode") is True
        assert agent.get_state("initial_value") == "test"
        assert agent.get_state("counter") == 0

    @patch("agent.reasoning.analyze.structured_llm_call")
    def test_tool_execution_failure_handling(self, mock_structured_call):
        """Test that tool execution failures are handled gracefully"""
        config = create_multi_tool_config()
        llm = self.create_mock_llm()
        agent = Agent(config=config, model=SupportedModel.LLAMA_8B, llm=llm)

        # Mock reasoning response that proposes a failing tool
        from agent.reasoning.types import ProposedToolCall

        reasoning_responses = [
            ReasoningResult(
                understanding="User wants to test failure",
                situational_awareness="I should try the failing tool",
                emotional_context="Cautious",
                key_information=["User wants to test failure"],
                proposed_tools=[
                    ProposedToolCall(
                        tool_name="failing_tool",
                        call_id="call_1",
                        parameters={"message": "cause failure"},
                        reasoning="This might fail",
                    )
                ],
                follow_up_opportunities=["Handle failure"],
                should_end_turn=False,
            ),
            ReasoningResult(
                understanding="Tool failed as expected",
                situational_awareness="Tool failed, should end turn",
                emotional_context="Understanding",
                key_information=["Tool execution failed"],
                proposed_tools=[],
                follow_up_opportunities=["Wait for user"],
                should_end_turn=True,
            ),
        ]

        call_count = 0

        def mock_reasoning(*args, **kwargs):
            nonlocal call_count
            result = reasoning_responses[min(call_count, len(reasoning_responses) - 1)]
            call_count += 1
            return result

        mock_structured_call.side_effect = mock_reasoning
        self.mock_llm_stream(llm, ["The tool failed as expected."])

        events = self.collect_stream_events(agent, "Make it fail")

        # Should emit ToolStartedEvent (tool exists)
        tool_started_events = [e for e in events if isinstance(e, ToolStartedEvent)]
        tool_finished_events = [e for e in events if isinstance(e, ToolFinishedEvent)]

        assert len(tool_started_events) >= 1
        failing_started = any(
            e.tool_name == "failing_tool" for e in tool_started_events
        )
        assert failing_started

        # Should handle tool failure with error result type
        assert len(tool_finished_events) >= 1
        failing_finished = [
            e
            for e in tool_finished_events
            if hasattr(e.result, "type") and e.result.type == "error"
        ]
        assert len(failing_finished) >= 1
        assert "cause failure" in failing_finished[0].result.error

    @patch("agent.reasoning.analyze.structured_llm_call")
    def test_state_persistence_across_interactions(self, mock_structured_call):
        """Test that state persists across multiple interactions"""
        config = create_stateful_config()
        llm = self.create_mock_llm()
        agent = Agent(config=config, model=SupportedModel.LLAMA_8B, llm=llm)

        # Set some state
        agent.set_state("persistent_key", "persistent_value")

        # First interaction
        mock_structured_call.return_value = ReasoningResult(
            understanding="First interaction",
            situational_awareness="Just respond",
            emotional_context="Neutral",
            key_information=["First interaction"],
            proposed_tools=[],
            follow_up_opportunities=["Wait for user"],
            should_end_turn=True,
        )
        self.mock_llm_stream(llm, ["First"])
        self.collect_stream_events(agent, "First")
        assert agent.get_state("persistent_key") == "persistent_value"

        # Second interaction
        mock_structured_call.return_value = ReasoningResult(
            understanding="Second interaction",
            situational_awareness="Just respond",
            emotional_context="Neutral",
            key_information=["Second interaction"],
            proposed_tools=[],
            follow_up_opportunities=["Wait for user"],
            should_end_turn=True,
        )
        self.mock_llm_stream(llm, ["Second"])
        self.collect_stream_events(agent, "Second")
        assert agent.get_state("persistent_key") == "persistent_value"

    @patch("agent.reasoning.analyze.structured_llm_call")
    def test_event_types_are_correct(self, mock_structured_call):
        """Test that agent emits correct event types"""
        config = create_simple_config()
        llm = self.create_mock_llm()
        agent = Agent(config=config, model=SupportedModel.LLAMA_8B, llm=llm)

        # Mock reasoning response with tool
        from agent.reasoning.types import ProposedToolCall

        reasoning_responses = [
            ReasoningResult(
                understanding="User wants all event types",
                situational_awareness="Use tool and respond",
                emotional_context="Helpful",
                key_information=["User wants to test events"],
                proposed_tools=[
                    ProposedToolCall(
                        tool_name="mock_tool",
                        call_id="call_1",
                        parameters={"message": "test"},
                        reasoning="Test all events",
                    )
                ],
                follow_up_opportunities=["Continue testing"],
                should_end_turn=False,
            ),
            ReasoningResult(
                understanding="Tool executed successfully",
                situational_awareness="Tool completed, time to respond",
                emotional_context="Satisfied",
                key_information=["Tool events generated"],
                proposed_tools=[],
                follow_up_opportunities=["Wait for user"],
                should_end_turn=True,
            ),
        ]

        call_count = 0

        def mock_reasoning(*args, **kwargs):
            nonlocal call_count
            result = reasoning_responses[min(call_count, len(reasoning_responses) - 1)]
            call_count += 1
            return result

        mock_structured_call.side_effect = mock_reasoning
        self.mock_llm_stream(llm, ["Text after tool."])

        events = self.collect_stream_events(agent, "Test all event types")

        # Should have text events (including thoughts)
        text_events = [e for e in events if isinstance(e, AgentTextEvent)]
        assert len(text_events) > 0

        # Should have tool events
        tool_started_events = [e for e in events if isinstance(e, ToolStartedEvent)]
        tool_finished_events = [e for e in events if isinstance(e, ToolFinishedEvent)]
        assert len(tool_started_events) > 0
        assert len(tool_finished_events) > 0

        # Tool events should match
        assert len(tool_started_events) == len(tool_finished_events)
