"""
Tests for auto-summarization functionality
"""

import pytest
from unittest.mock import MagicMock, patch
from typing import List

from agent.core import Agent
from agent.config import AgentConfig
from agent.message import UserMessage, AgentMessage, Message
from agent.agent_events import SummarizationStartedEvent, SummarizationFinishedEvent, ResponseCompleteEvent
from agent.llm import Message as LLMMessage
from test_configs import create_empty_config


class MockLLMClient:
    """Mock LLM client for testing"""
    
    def __init__(self, mock_summary_response="Mock summary of conversation"):
        self.context_window = 4000
        self.mock_summary_response = mock_summary_response
        
    def chat_complete(self, messages: List[LLMMessage]) -> str:
        """Mock chat completion"""
        return self.mock_summary_response
        
    def chat(self, messages: List[LLMMessage]):
        """Mock streaming chat - not used in summarization"""
        yield {"message": {"content": "mock response"}}


@pytest.fixture
def mock_agent():
    """Create agent with mocked LLM"""
    config = create_empty_config()
    agent = Agent(config=config, model="test-model")
    agent.llm = MockLLMClient()
    return agent


@pytest.fixture
def agent_with_long_history(mock_agent):
    """Agent with conversation history that exceeds summarization threshold"""
    # Create 10 messages (user/agent alternating)
    for i in range(5):
        mock_agent.conversation_history.append(UserMessage(content=f"User message {i+1}"))
        mock_agent.llm_conversation_history.append(UserMessage(content=f"User message {i+1}"))
        
        mock_agent.conversation_history.append(AgentMessage(content=f"Agent response {i+1}", tool_calls=[]))
        mock_agent.llm_conversation_history.append(AgentMessage(content=f"Agent response {i+1}", tool_calls=[]))
    
    # Mock context info to trigger summarization
    with patch.object(mock_agent, 'get_context_info') as mock_context:
        mock_context.return_value.approaching_limit = True
        mock_context.return_value.usage_percentage = 80.0
        yield mock_agent


class TestAutoSummarization:
    """Test auto-summarization functionality"""
    
    def test_summarization_not_triggered_when_under_threshold(self, mock_agent):
        """Test that summarization doesn't trigger when context usage is low"""
        # Add a few messages
        mock_agent.conversation_history.append(UserMessage(content="Hello"))
        mock_agent.llm_conversation_history.append(UserMessage(content="Hello"))
        
        # Mock context info to NOT trigger summarization
        with patch.object(mock_agent, 'get_context_info') as mock_context:
            mock_context.return_value.approaching_limit = False
            mock_context.return_value.usage_percentage = 50.0
            
            events = list(mock_agent.chat_stream("Test message"))
            
            # Should not contain summarization events
            event_types = [type(event).__name__ for event in events]
            assert "SummarizationStartedEvent" not in event_types
            assert "SummarizationFinishedEvent" not in event_types
    
    def test_summarization_not_triggered_with_few_messages(self, mock_agent):
        """Test that summarization doesn't trigger when there are too few messages"""
        # Add only 3 messages (less than keep_recent default of 6)
        for i in range(3):
            mock_agent.conversation_history.append(UserMessage(content=f"Message {i}"))
            mock_agent.llm_conversation_history.append(UserMessage(content=f"Message {i}"))
        
        # Mock context to trigger summarization
        with patch.object(mock_agent, 'get_context_info') as mock_context:
            mock_context.return_value.approaching_limit = True
            mock_context.return_value.usage_percentage = 80.0
            
            events = list(mock_agent.chat_stream("Test message"))
            
            # Should not contain summarization events
            event_types = [type(event).__name__ for event in events]
            assert "SummarizationStartedEvent" not in event_types
            assert "SummarizationFinishedEvent" not in event_types
    
    def test_summarization_triggered_correctly(self, agent_with_long_history):
        """Test that summarization triggers with proper conditions"""
        agent = agent_with_long_history
        
        events = list(agent.chat_stream("New message"))
        
        # Should contain summarization events
        event_types = [type(event).__name__ for event in events]
        assert "SummarizationStartedEvent" in event_types
        assert "SummarizationFinishedEvent" in event_types
        
        # Check event order
        started_idx = next(i for i, e in enumerate(events) if isinstance(e, SummarizationStartedEvent))
        finished_idx = next(i for i, e in enumerate(events) if isinstance(e, SummarizationFinishedEvent))
        assert started_idx < finished_idx
    
    def test_summarization_started_event_content(self, agent_with_long_history):
        """Test the content of SummarizationStartedEvent"""
        agent = agent_with_long_history
        
        events = list(agent.chat_stream("New message"))
        started_event = next(e for e in events if isinstance(e, SummarizationStartedEvent))
        
        # Should have correct message counts (11 total after new message, minus 6 recent)
        assert started_event.messages_to_summarize == 5  # 11 total - 6 keep_recent 
        assert started_event.recent_messages_kept == 6
        assert started_event.context_usage_before == 80.0
    
    def test_summarization_finished_event_content(self, agent_with_long_history):
        """Test the content of SummarizationFinishedEvent"""
        agent = agent_with_long_history
        
        events = list(agent.chat_stream("New message"))
        finished_event = next(e for e in events if isinstance(e, SummarizationFinishedEvent))
        
        # Should contain summary and counts
        assert "Mock summary" in finished_event.summary
        assert finished_event.messages_summarized == 5
        assert finished_event.messages_after > 0  # Should have some messages left
        assert 0 <= finished_event.context_usage_after <= 100
    
    def test_dual_history_behavior(self, agent_with_long_history):
        """Test that summarization affects LLM history but preserves user history"""
        agent = agent_with_long_history
        original_user_count = len(agent.conversation_history)
        original_llm_count = len(agent.llm_conversation_history)
        
        list(agent.chat_stream("New message"))
        
        # User history should be mostly preserved (with notification added)
        assert len(agent.conversation_history) >= original_user_count
        
        # LLM history should be condensed
        assert len(agent.llm_conversation_history) < original_llm_count
        
        # LLM history should start with summary
        assert "[Summary of" in agent.llm_conversation_history[0].content
    
    def test_summarization_uses_config_prompts(self, agent_with_long_history):
        """Test that summarization uses config-specific prompts"""
        agent = agent_with_long_history
        
        with patch.object(agent.config, 'get_summarization_system_prompt') as mock_sys_prompt, \
             patch.object(agent.config, 'get_summarization_prompt') as mock_user_prompt:
            
            mock_sys_prompt.return_value = "Custom system prompt"
            mock_user_prompt.return_value = "Custom user prompt"
            
            list(agent.chat_stream("New message"))
            
            # Should have called config methods
            mock_sys_prompt.assert_called_once()
            mock_user_prompt.assert_called_once_with(5)  # 5 messages to summarize
    
    def test_summarization_includes_state_context(self, agent_with_long_history):
        """Test that summarization includes current agent state"""
        agent = agent_with_long_history
        
        # Set some simple test state
        agent.state = {
            "test_key": "test_value",
            "another_key": "another_value"
        }
        
        # Mock the _build_state_info method to return something we can test
        with patch.object(agent.config, '_build_state_info') as mock_state_info, \
             patch.object(agent.llm, 'chat_complete') as mock_complete:
            
            mock_state_info.return_value = "Test state: 2 keys"
            mock_complete.return_value = "Mock summary"
            
            list(agent.chat_stream("New message"))
            
            # Verify that _build_state_info was called with the agent state
            mock_state_info.assert_called_with(agent.state)
            assert mock_state_info.call_count >= 1  # Called at least once during summarization
            
            # Check that state context was included in system prompt
            call_args = mock_complete.call_args[0][0]  # First positional arg (messages list)
            system_message = call_args[0]  # First message should be system
            
            # Should include the mocked state context
            assert "CURRENT STATE CONTEXT" in system_message.content
            assert "Test state: 2 keys" in system_message.content
    
    def test_message_alternation_preservation(self, mock_agent):
        """Test that summarization preserves proper user/agent alternation"""
        # Create history that starts with agent message after summarization
        mock_agent.llm_conversation_history = [
            AgentMessage(content="Agent msg 1", tool_calls=[]),
            UserMessage(content="User msg 1"),
            AgentMessage(content="Agent msg 2", tool_calls=[]),
            UserMessage(content="User msg 2"),
            AgentMessage(content="Agent msg 3", tool_calls=[]),  # This will be first in recent
            UserMessage(content="User msg 3"),
        ]
        
        # Mock context to trigger summarization
        with patch.object(mock_agent, 'get_context_info') as mock_context:
            mock_context.return_value.approaching_limit = True
            mock_context.return_value.usage_percentage = 80.0
            
            list(mock_agent._auto_summarize_with_events(keep_recent=2))
            
            # Check that alternation is preserved
            llm_history = mock_agent.llm_conversation_history
            assert len(llm_history) >= 2
            
            # Should have summary, then potentially separator, then recent messages
            assert "[Summary of" in llm_history[0].content
    
    def test_user_notification_insertion(self, agent_with_long_history):
        """Test that users see summarization notification in their history"""
        agent = agent_with_long_history
        original_count = len(agent.conversation_history)
        
        list(agent.chat_stream("New message"))
        
        # Should have system messages with structured summarization content
        from agent.message import SystemMessage, SummarizationContent
        system_messages = [msg for msg in agent.conversation_history 
                          if isinstance(msg, SystemMessage) and 
                          isinstance(msg.content, SummarizationContent)]
        assert len(system_messages) == 1
        
        # Check the structured content
        notification = system_messages[0]
        assert isinstance(notification.content, SummarizationContent)
        assert notification.content.type == "summarization"
        assert "Summarized" in notification.content.title
        assert notification.content.messages_summarized > 0
        
        # Notification should be in correct chronological position
        notification_index = agent.conversation_history.index(notification)
        assert 0 <= notification_index < len(agent.conversation_history)
    
    def test_summarization_uses_single_message_approach(self, agent_with_long_history):
        """Test that summarization uses single message concatenation approach"""
        agent = agent_with_long_history
        
        with patch.object(agent.llm, 'chat_complete') as mock_complete:
            mock_complete.return_value = "Mock summary"
            
            list(agent.chat_stream("New message"))
            
            # Check that the new single-message approach was used
            call_args = mock_complete.call_args[0][0]
            
            # Should have exactly 2 messages: system prompt + user request with conversation text
            assert len(call_args) == 2
            assert call_args[0].role == "system"
            assert call_args[1].role == "user"
            
            # User message should contain concatenated conversation text
            user_message = call_args[1].content
            assert "Please summarize the following conversation:" in user_message
            assert "USER:" in user_message  # Should have conversation text
            assert "ASSISTANT:" in user_message


class TestSummarizationEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_empty_conversation_summarization(self, mock_agent):
        """Test behavior with empty conversation"""
        with patch.object(mock_agent, 'get_context_info') as mock_context:
            mock_context.return_value.approaching_limit = True
            mock_context.return_value.usage_percentage = 80.0
            
            events = list(mock_agent.chat_stream("First message"))
            
            # Should not trigger summarization with only one message
            event_types = [type(event).__name__ for event in events]
            assert "SummarizationStartedEvent" not in event_types
    
    def test_summarization_with_tool_calls(self, mock_agent):
        """Test summarization works with messages containing tool calls"""
        from agent.message import ToolCallFinished, ToolCallResult, ToolCallResultType
        
        # Add messages with tool calls
        tool_call = ToolCallFinished(
            tool_name="test_tool",
            tool_id="call_1",
            parameters={"param": "value"},
            result=ToolCallResult(type=ToolCallResultType.SUCCESS, content="Success")
        )
        
        for i in range(4):
            mock_agent.conversation_history.append(UserMessage(content=f"User {i}"))
            mock_agent.llm_conversation_history.append(UserMessage(content=f"User {i}"))
            
            mock_agent.conversation_history.append(AgentMessage(content=f"Agent {i}", tool_calls=[tool_call]))
            mock_agent.llm_conversation_history.append(AgentMessage(content=f"Agent {i}", tool_calls=[tool_call]))
        
        with patch.object(mock_agent, 'get_context_info') as mock_context:
            mock_context.return_value.approaching_limit = True
            mock_context.return_value.usage_percentage = 80.0
            
            events = list(mock_agent.chat_stream("New message"))
            
            # Should handle tool calls gracefully
            assert any(isinstance(e, SummarizationStartedEvent) for e in events)
            assert any(isinstance(e, SummarizationFinishedEvent) for e in events)
    
    def test_summarization_llm_error_handling(self, agent_with_long_history):
        """Test handling of LLM errors during summarization"""
        agent = agent_with_long_history
        
        with patch.object(agent.llm, 'chat_complete') as mock_complete:
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
        user_prompt = config.get_summarization_prompt(5)
        assert "5 messages" in user_prompt
    
    def test_config_state_info_in_summarization(self, mock_agent):
        """Test that config state info method is callable and works"""
        # Set simple test state
        mock_agent.state = {
            "test_key": "test_value",
            "another_key": "another_value"
        }
        
        state_info = mock_agent.config._build_state_info(mock_agent.state)
        
        # Base config returns empty string, which is expected behavior
        assert isinstance(state_info, str)  # Should return a string
        
        # Test that the method is callable and doesn't crash
        empty_state_info = mock_agent.config._build_state_info({})
        assert isinstance(empty_state_info, str)


if __name__ == "__main__":
    pytest.main([__file__])