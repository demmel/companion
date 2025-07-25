"""
Tests for auto-summarization functionality
"""

import pytest
from unittest.mock import patch
from typing import List


from agent.core import Agent
from agent.types import (
    SummarizationContent,
    SystemMessage,
    TextContent,
    TextToolContent,
    ToolCallSuccess,
    UserMessage,
    AgentMessage,
)
from agent.agent_events import (
    SummarizationStartedEvent,
    SummarizationFinishedEvent,
)
from agent.llm import Message as LLMMessage, LLMClient
from test_configs import create_empty_config


class MockLLMClient(LLMClient):
    """Mock LLM client for testing"""

    def __init__(self, mock_summary_response="Mock summary of conversation"):
        self.context_window = 4000
        self.mock_summary_response = mock_summary_response

    def chat_complete(self, model, messages: List[LLMMessage], **kwargs) -> str:
        """Mock chat completion"""
        return self.mock_summary_response

    def chat(self, messages: List[LLMMessage]):
        """Mock streaming chat - not used in summarization"""
        yield {"message": {"content": "mock response"}}


@pytest.fixture
def mock_agent():
    """Create agent with mocked LLM"""
    from unittest.mock import Mock
    from agent.llm import SupportedModel
    
    config = create_empty_config()
    mock_llm = Mock()
    mock_llm.models = {
        SupportedModel.LLAMA_8B: Mock(context_window=4096)
    }
    agent = Agent(config=config, model=SupportedModel.LLAMA_8B, llm=mock_llm)
    agent.llm = MockLLMClient()
    return agent


@pytest.fixture
def agent_with_long_history(mock_agent):
    """Agent with conversation history that exceeds summarization threshold"""
    # Create 20 messages (user/agent alternating) - enough to trigger summarization
    for i in range(10):
        mock_agent.conversation_history.append(
            UserMessage(content=[TextContent(text=f"User message {i+1}")])
        )
        mock_agent.conversation_history.append(
            AgentMessage(
                content=[TextContent(text=f"Agent response {i+1}")], tool_calls=[]
            )
        )

    # Mock context info to trigger summarization
    from agent.core import ContextInfo
    mock_context_info = ContextInfo(
        message_count=25,
        conversation_messages=20,  # 20 conversation messages (above threshold)
        estimated_tokens=3200,  # Above 75% threshold
        context_limit=4096,
        usage_percentage=80.0,
        approaching_limit=True,
    )
    with patch.object(mock_agent, "get_context_info") as mock_context:
        mock_context.return_value = mock_context_info
        yield mock_agent


class TestAutoSummarization:
    """Test auto-summarization functionality"""

    @patch("agent.reasoning.analyze.structured_llm_call")
    def test_summarization_not_triggered_when_under_threshold(self, mock_structured_call, mock_agent):
        """Test that summarization doesn't trigger when context usage is low"""
        # Add a few messages to conversation history
        mock_agent.conversation_history.append(
            UserMessage(content=[TextContent(text="Hello")])
        )

        # Mock reasoning response
        from agent.reasoning.types import ReasoningResult
        mock_structured_call.return_value = ReasoningResult(
            understanding="User said test message",
            situational_awareness="Simple test interaction",
            emotional_context="Neutral",
            key_information=[],
            proposed_tools=[],
            follow_up_opportunities=[],
            should_end_turn=True,
        )

        # Mock context info to NOT trigger summarization
        from agent.core import ContextInfo
        mock_context_info = ContextInfo(
            message_count=2,
            conversation_messages=1,
            estimated_tokens=100,
            context_limit=4096,
            usage_percentage=50.0,
            approaching_limit=False,
        )
        with patch.object(mock_agent, "get_context_info") as mock_context:
            mock_context.return_value = mock_context_info

            events = list(mock_agent.chat_stream("Test message"))

            # Should not contain summarization events
            event_types = [type(event).__name__ for event in events]
            assert "SummarizationStartedEvent" not in event_types
            assert "SummarizationFinishedEvent" not in event_types

    @patch("agent.reasoning.analyze.structured_llm_call")
    def test_summarization_not_triggered_with_few_messages(self, mock_structured_call, mock_agent):
        """Test that summarization doesn't trigger when there are too few messages"""
        # Add only 3 messages (less than keep_recent default of 10)
        for i in range(3):
            mock_agent.conversation_history.append(
                UserMessage(content=[TextContent(text=f"Message {i}")])
            )

        # Mock reasoning response
        from agent.reasoning.types import ReasoningResult
        mock_structured_call.return_value = ReasoningResult(
            understanding="User said test message",
            situational_awareness="Simple test interaction",
            emotional_context="Neutral",
            key_information=[],
            proposed_tools=[],
            follow_up_opportunities=[],
            should_end_turn=True,
        )

        # Mock context to trigger summarization
        from agent.core import ContextInfo
        mock_context_info = ContextInfo(
            message_count=5,
            conversation_messages=3,
            estimated_tokens=3200,  # Above threshold
            context_limit=4096,
            usage_percentage=80.0,
            approaching_limit=True,
        )
        with patch.object(mock_agent, "get_context_info") as mock_context:
            mock_context.return_value = mock_context_info

            events = list(mock_agent.chat_stream("Test message"))

            # Should not contain summarization events (too few messages)
            event_types = [type(event).__name__ for event in events]
            assert "SummarizationStartedEvent" not in event_types
            assert "SummarizationFinishedEvent" not in event_types

    @patch("agent.reasoning.analyze.structured_llm_call")
    def test_summarization_triggered_correctly(self, mock_structured_call, agent_with_long_history):
        """Test that summarization triggers with proper conditions"""
        agent = agent_with_long_history

        # Mock reasoning response
        from agent.reasoning.types import ReasoningResult
        mock_structured_call.return_value = ReasoningResult(
            understanding="User sent new message",
            situational_awareness="Continue conversation",
            emotional_context="Neutral",
            key_information=[],
            proposed_tools=[],
            follow_up_opportunities=[],
            should_end_turn=True,
        )

        events = list(agent.chat_stream("New message"))

        # Should contain summarization events
        event_types = [type(event).__name__ for event in events]
        assert "SummarizationStartedEvent" in event_types
        assert "SummarizationFinishedEvent" in event_types

        # Check event order
        started_idx = next(
            i for i, e in enumerate(events) if isinstance(e, SummarizationStartedEvent)
        )
        finished_idx = next(
            i for i, e in enumerate(events) if isinstance(e, SummarizationFinishedEvent)
        )
        assert started_idx < finished_idx

    @patch("agent.reasoning.analyze.structured_llm_call")
    def test_summarization_started_event_content(self, mock_structured_call, agent_with_long_history):
        """Test the content of SummarizationStartedEvent"""
        agent = agent_with_long_history

        # Mock reasoning response
        from agent.reasoning.types import ReasoningResult
        mock_structured_call.return_value = ReasoningResult(
            understanding="User sent new message",
            situational_awareness="Continue conversation",
            emotional_context="Neutral",
            key_information=[],
            proposed_tools=[],
            follow_up_opportunities=[],
            should_end_turn=True,
        )

        events = list(agent.chat_stream("New message"))
        started_event = next(
            e for e in events if isinstance(e, SummarizationStartedEvent)
        )

        # Should have correct message counts (20 from fixture + 1 user message = 21 total, minus 10 recent)
        assert started_event.messages_to_summarize == 10  # 20 from fixture - 10 keep_recent
        assert started_event.recent_messages_kept == 10
        assert started_event.context_usage_before == 80.0

    @patch("agent.reasoning.analyze.structured_llm_call")
    def test_summarization_finished_event_content(self, mock_structured_call, agent_with_long_history):
        """Test the content of SummarizationFinishedEvent"""
        agent = agent_with_long_history

        # Mock reasoning response
        from agent.reasoning.types import ReasoningResult
        mock_structured_call.return_value = ReasoningResult(
            understanding="User sent new message",
            situational_awareness="Continue conversation",
            emotional_context="Neutral",
            key_information=[],
            proposed_tools=[],
            follow_up_opportunities=[],
            should_end_turn=True,
        )

        events = list(agent.chat_stream("New message"))
        finished_event = next(
            e for e in events if isinstance(e, SummarizationFinishedEvent)
        )

        # Should contain summary and counts
        assert "Mock summary" in finished_event.summary
        assert finished_event.messages_summarized == 10
        assert finished_event.messages_after > 0  # Should have some messages left
        assert 0 <= finished_event.context_usage_after <= 100

    @patch("agent.reasoning.analyze.structured_llm_call")
    def test_dual_history_behavior(self, mock_structured_call, agent_with_long_history):
        """Test that summarization affects LLM history but preserves user history"""
        agent = agent_with_long_history

        # Mock reasoning response
        from agent.reasoning.types import ReasoningResult
        mock_structured_call.return_value = ReasoningResult(
            understanding="User sent new message",
            situational_awareness="Continue conversation",
            emotional_context="Neutral",
            key_information=[],
            proposed_tools=[],
            follow_up_opportunities=[],
            should_end_turn=True,
        )

        original_user_count = len(agent.conversation_history.get_full_history())
        original_llm_count = len(agent.conversation_history.get_summarized_history())

        list(agent.chat_stream("New message"))

        # User history should be mostly preserved (with notification added)
        assert len(agent.conversation_history.get_full_history()) >= original_user_count

        # LLM history should be condensed  
        assert len(agent.conversation_history.get_summarized_history()) < original_llm_count

        # LLM history should start with summary
        first_message = agent.conversation_history.get_summarized_history()[0]
        assert isinstance(first_message, SystemMessage)
        assert isinstance(first_message.content[0], SummarizationContent)
        assert first_message.content[0].type == "summarization"
        assert "Summarized" in first_message.content[0].title
        assert first_message.content[0].messages_summarized > 0

    @patch("agent.reasoning.analyze.structured_llm_call")
    def test_summarization_uses_config_prompts(self, mock_structured_call, agent_with_long_history):
        """Test that summarization uses config-specific prompts"""
        agent = agent_with_long_history

        # Mock reasoning response
        from agent.reasoning.types import ReasoningResult
        mock_structured_call.return_value = ReasoningResult(
            understanding="User sent new message",
            situational_awareness="Continue conversation",
            emotional_context="Neutral",
            key_information=[],
            proposed_tools=[],
            follow_up_opportunities=[],
            should_end_turn=True,
        )

        with (
            patch.object(
                agent.config, "get_summarization_system_prompt"
            ) as mock_sys_prompt,
            patch.object(agent.config, "get_summarization_prompt") as mock_user_prompt,
        ):

            mock_sys_prompt.return_value = "Custom system prompt"
            mock_user_prompt.return_value = "Custom user prompt"

            list(agent.chat_stream("New message"))

            # Should have called config methods
            mock_sys_prompt.assert_called_once()
            mock_user_prompt.assert_called_once()

    @patch("agent.reasoning.analyze.structured_llm_call")
    def test_summarization_includes_state_context(self, mock_structured_call, agent_with_long_history):
        """Test that summarization includes current agent state"""
        agent = agent_with_long_history

        # Mock reasoning response
        from agent.reasoning.types import ReasoningResult
        mock_structured_call.return_value = ReasoningResult(
            understanding="User sent new message",
            situational_awareness="Continue conversation",
            emotional_context="Neutral",
            key_information=[],
            proposed_tools=[],
            follow_up_opportunities=[],
            should_end_turn=True,
        )

        # Set some simple test state
        agent.state = {"test_key": "test_value", "another_key": "another_value"}

        # Mock the _build_state_info method to return something we can test
        with (
            patch.object(agent.config, "_build_state_info") as mock_state_info,
            patch.object(agent.llm, "chat_complete") as mock_complete,
        ):

            mock_state_info.return_value = "Test state: 2 keys"
            mock_complete.return_value = "Mock summary"

            list(agent.chat_stream("New message"))

            # Verify that _build_state_info was called with the agent state
            mock_state_info.assert_called_with(agent.state)
            assert (
                mock_state_info.call_count >= 1
            )  # Called at least once during summarization

            # Check that state context was included in system prompt
            # call_args is (model, messages_list, **kwargs)
            call_args = mock_complete.call_args[0]
            model = call_args[0]  # First arg is model
            messages_list = call_args[1]  # Second arg is messages list
            system_message = messages_list[0]  # First message should be system

            # Should include the mocked state context
            assert "CURRENT STATE CONTEXT" in system_message.content
            assert "Test state: 2 keys" in system_message.content

    @patch("agent.reasoning.analyze.structured_llm_call")
    def test_user_notification_insertion(self, mock_structured_call, agent_with_long_history):
        """Test that users see summarization notification in their history"""
        agent = agent_with_long_history

        # Mock reasoning response
        from agent.reasoning.types import ReasoningResult
        mock_structured_call.return_value = ReasoningResult(
            understanding="User sent new message",
            situational_awareness="Continue conversation",
            emotional_context="Neutral",
            key_information=[],
            proposed_tools=[],
            follow_up_opportunities=[],
            should_end_turn=True,
        )

        original_count = len(agent.conversation_history.get_full_history())

        list(agent.chat_stream("New message"))

        # Should have system messages with structured summarization content
        from agent.types import SystemMessage, SummarizationContent

        system_messages = [
            msg
            for msg in agent.conversation_history.get_full_history()
            if isinstance(msg, SystemMessage)
            and isinstance(msg.content[0], SummarizationContent)
        ]
        assert len(system_messages) == 1

        # Check the structured content
        notification = system_messages[0]
        assert isinstance(notification.content[0], SummarizationContent)
        assert notification.content[0].type == "summarization"
        assert "Summarized" in notification.content[0].title
        assert notification.content[0].messages_summarized > 0

        # Notification should be in correct chronological position
        notification_index = agent.conversation_history.get_full_history().index(notification)
        assert 0 <= notification_index < len(agent.conversation_history.get_full_history())

    @patch("agent.reasoning.analyze.structured_llm_call")
    def test_summarization_uses_single_message_approach(self, mock_structured_call, agent_with_long_history):
        """Test that summarization uses single message concatenation approach"""
        agent = agent_with_long_history

        # Mock reasoning response
        from agent.reasoning.types import ReasoningResult
        mock_structured_call.return_value = ReasoningResult(
            understanding="User sent new message",
            situational_awareness="Continue conversation",
            emotional_context="Neutral",
            key_information=[],
            proposed_tools=[],
            follow_up_opportunities=[],
            should_end_turn=True,
        )

        with patch.object(agent.llm, "chat_complete") as mock_complete:
            mock_complete.return_value = "Mock summary"

            list(agent.chat_stream("New message"))

            # Check that the new single-message approach was used
            # call_args is (model, messages_list, **kwargs)
            call_args = mock_complete.call_args[0]
            model = call_args[0]  # First arg is model
            messages_list = call_args[1]  # Second arg is messages list

            # Should have exactly 2 messages: system prompt + user request with conversation text
            assert len(messages_list) == 2
            assert messages_list[0].role == "system"
            assert messages_list[1].role == "user"

            # User message should contain concatenated conversation text
            user_message = messages_list[1].content
            assert "Please provide a summary" in user_message
            assert "USER:" in user_message  # Should have conversation text
            assert "ASSISTANT:" in user_message


class TestSummarizationEdgeCases:
    """Test edge cases and error conditions"""

    @patch("agent.reasoning.analyze.structured_llm_call")
    def test_empty_conversation_summarization(self, mock_structured_call, mock_agent):
        """Test behavior with empty conversation"""
        # Mock reasoning response
        from agent.reasoning.types import ReasoningResult
        mock_structured_call.return_value = ReasoningResult(
            understanding="User sent first message",
            situational_awareness="Starting conversation",
            emotional_context="Neutral",
            key_information=[],
            proposed_tools=[],
            follow_up_opportunities=[],
            should_end_turn=True,
        )

        with patch.object(mock_agent, "get_context_info") as mock_context:
            mock_context.return_value.approaching_limit = True
            mock_context.return_value.usage_percentage = 80.0

            events = list(mock_agent.chat_stream("First message"))

            # Should not trigger summarization with only one message
            event_types = [type(event).__name__ for event in events]
            assert "SummarizationStartedEvent" not in event_types

    @patch("agent.reasoning.analyze.structured_llm_call")
    def test_summarization_with_tool_calls(self, mock_structured_call, mock_agent):
        """Test summarization works with messages containing tool calls"""
        from agent.types import ToolCallFinished

        # Mock reasoning response
        from agent.reasoning.types import ReasoningResult
        mock_structured_call.return_value = ReasoningResult(
            understanding="User sent new message",
            situational_awareness="Continue conversation with tool history",
            emotional_context="Neutral",
            key_information=[],
            proposed_tools=[],
            follow_up_opportunities=[],
            should_end_turn=True,
        )

        # Add messages with tool calls (need enough to trigger summarization)
        tool_call = ToolCallFinished(
            tool_name="test_tool",
            tool_id="call_1",
            parameters={"param": "value"},
            result=ToolCallSuccess(content=TextToolContent(text="Success")),
        )

        # Add more messages to trigger summarization (need > 10 messages)
        for i in range(12):
            mock_agent.conversation_history.append(
                UserMessage(content=[TextContent(text=f"User {i}")])
            )

            mock_agent.conversation_history.append(
                AgentMessage(
                    content=[TextContent(text=f"Agent {i}")], tool_calls=[tool_call]
                )
            )

        with patch.object(mock_agent, "get_context_info") as mock_context:
            # Create a mock context info object with proper attributes
            from agent.core import ContextInfo
            mock_context_info = ContextInfo(
                message_count=25,
                conversation_messages=24,  # 24 conversation messages (above threshold)
                estimated_tokens=3200,  # Above 75% threshold
                context_limit=4096,
                usage_percentage=80.0,
                approaching_limit=True,
            )
            mock_context.return_value = mock_context_info

            events = list(mock_agent.chat_stream("New message"))

            # Should handle tool calls gracefully
            assert any(isinstance(e, SummarizationStartedEvent) for e in events)
            assert any(isinstance(e, SummarizationFinishedEvent) for e in events)

    @patch("agent.reasoning.analyze.structured_llm_call")
    def test_summarization_llm_error_handling(self, mock_structured_call, agent_with_long_history):
        """Test handling of LLM errors during summarization"""
        agent = agent_with_long_history

        # Mock reasoning response
        from agent.reasoning.types import ReasoningResult
        mock_structured_call.return_value = ReasoningResult(
            understanding="User sent new message",
            situational_awareness="Continue conversation",
            emotional_context="Neutral",
            key_information=[],
            proposed_tools=[],
            follow_up_opportunities=[],
            should_end_turn=True,
        )

        with patch.object(agent.llm, "chat_complete") as mock_complete:
            mock_complete.side_effect = Exception("LLM Error")

            # Should raise the exception (for now - could be enhanced to handle gracefully)
            with pytest.raises(Exception, match="LLM Error"):
                list(agent.chat_stream("New message"))


class TestSummarizationConfiguration:
    """Test configuration-specific summarization behavior"""

    def test_test_config_summarization_context(self):
        """Test that test config provides basic summarization context"""
        config = create_empty_config()

        # Test system prompt is reasonable
        system_prompt = config.get_summarization_system_prompt()
        assert "summariz" in system_prompt.lower()

        # Test user prompt mentions message count
        user_prompt = config.get_summarization_prompt()
        assert "conversation" in user_prompt

    def test_config_state_info_in_summarization(self, mock_agent):
        """Test that config state info method is callable and works"""
        # Set simple test state
        mock_agent.state = {"test_key": "test_value", "another_key": "another_value"}

        state_info = mock_agent.config._build_state_info(mock_agent.state)

        # Base config returns empty string, which is expected behavior
        assert isinstance(state_info, str)  # Should return a string

        # Test that the method is callable and doesn't crash
        empty_state_info = mock_agent.config._build_state_info({})
        assert isinstance(empty_state_info, str)


if __name__ == "__main__":
    pytest.main([__file__])
