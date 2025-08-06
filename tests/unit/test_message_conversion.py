"""
Tests for message conversion between structured format and LLM format
"""

from agent.core import message_to_llm_messages
from agent.types import (
    TextContent,
    ThoughtContent,
    TextToolContent,
    ToolCallError,
    ToolCallSuccess,
    UserMessage,
    AgentMessage,
    ToolCallFinished,
)
class TestMessageToLLMConversion:
    """Test conversion of structured messages to LLM format"""

    def create_mock_reasoning(self, text: str) -> str:
        """Create a mock reasoning result for tests"""
        return f"Understanding: {text}"

    def test_user_message_conversion(self):
        """Test that user messages convert correctly"""
        user_msg = UserMessage(content=[TextContent(text="Hello, how are you?")])

        llm_messages = list(message_to_llm_messages(user_msg))

        assert len(llm_messages) == 1
        assert llm_messages[0].role == "user"
        assert llm_messages[0].content == "Hello, how are you?"

    def test_agent_message_no_tools(self):
        """Test agent message without tool calls"""
        agent_msg = AgentMessage(
            content=[TextContent(text="I'm doing well, thank you!")], tool_calls=[]
        )

        llm_messages = list(message_to_llm_messages(agent_msg))

        assert len(llm_messages) == 1
        assert llm_messages[0].role == "assistant"
        assert llm_messages[0].content == "I'm doing well, thank you!"

    def test_agent_message_with_tool_calls(self):
        """Test agent message with tool calls generates proper LLM format"""
        tool_call = ToolCallFinished(
            tool_name="set_mood",
            tool_id="call_123",
            parameters={"mood": "happy", "intensity": 8},
            result=ToolCallSuccess(
                content=TextToolContent(text="Mood set to happy"),
                llm_feedback="Success",
            ),
        )

        agent_msg = AgentMessage(
            content=[TextContent(text="Let me set your mood to happy.")],
            tool_calls=[tool_call],
        )

        llm_messages = list(message_to_llm_messages(agent_msg))

        # Should generate 2 messages: agent message + tool results
        assert len(llm_messages) == 2

        # First message: agent with tool call syntax
        agent_llm_msg = llm_messages[0]
        assert agent_llm_msg.role == "assistant"
        assert "Let me set your mood to happy." in agent_llm_msg.content
        assert "TOOL_CALL: set_mood (call_123)" in agent_llm_msg.content
        assert '"mood": "happy"' in agent_llm_msg.content
        assert '"intensity": 8' in agent_llm_msg.content

        # Second message: tool results as user message
        results_llm_msg = llm_messages[1]
        assert results_llm_msg.role == "user"
        assert "TOOL_RESULT: set_mood (call_123)" in results_llm_msg.content
        assert "Success" in results_llm_msg.content

    def test_agent_message_with_multiple_tool_calls(self):
        """Test agent message with multiple tool calls"""
        tool_call_1 = ToolCallFinished(
            tool_name="set_mood",
            tool_id="call_1",
            parameters={"mood": "happy"},
            result=ToolCallSuccess(
                content=TextToolContent(text="Mood set to happy"),
                llm_feedback="Success",
            ),
        )

        tool_call_2 = ToolCallFinished(
            tool_name="remember_detail",
            tool_id="call_2",
            parameters={"detail": "User likes coffee"},
            result=ToolCallError(error="Memory storage failed"),
        )

        agent_msg = AgentMessage(
            content=[TextContent(text="I'll set your mood and remember that.")],
            tool_calls=[tool_call_1, tool_call_2],
        )

        llm_messages = list(message_to_llm_messages(agent_msg))

        assert len(llm_messages) == 2

        # Agent message should include both tool calls
        agent_llm_msg = llm_messages[0]
        assert "TOOL_CALL: set_mood (call_1)" in agent_llm_msg.content
        assert "TOOL_CALL: remember_detail (call_2)" in agent_llm_msg.content

        # Results message should include both results
        results_llm_msg = llm_messages[1]
        assert "TOOL_RESULT: set_mood (call_1)" in results_llm_msg.content
        assert "TOOL_RESULT: remember_detail (call_2)" in results_llm_msg.content
        assert "Success" in results_llm_msg.content
        assert "Memory storage failed" in results_llm_msg.content

    def test_agent_message_no_text_only_tools(self):
        """Test agent message with only tool calls and no dialogue text"""
        tool_call = ToolCallFinished(
            tool_name="assume_character",
            tool_id="call_abc",
            parameters={"name": "Alice", "personality": "cheerful"},
            result=ToolCallSuccess(
                content=TextToolContent(text="Character Alice created"),
                llm_feedback="Success",
            ),
        )

        agent_msg = AgentMessage(content=[], tool_calls=[tool_call])  # No dialogue text

        llm_messages = list(message_to_llm_messages(agent_msg))

        assert len(llm_messages) == 2

        # Agent message should have tool call but minimal content
        agent_llm_msg = llm_messages[0]
        assert "TOOL_CALL: assume_character (call_abc)" in agent_llm_msg.content

        # Results message should have tool result
        results_llm_msg = llm_messages[1]
        assert "Success" in results_llm_msg.content

    def test_agent_message_with_thoughts_excluded(self):
        """Test that thoughts are excluded by default"""
        agent_msg = AgentMessage(
            content=[
                ThoughtContent(
                    text="Let me think about this...",
                    reasoning=self.create_mock_reasoning("thinking"),
                ),
                TextContent(text="Here's my response"),
            ],
            tool_calls=[],
        )

        llm_messages = list(message_to_llm_messages(agent_msg))

        assert len(llm_messages) == 1
        assert llm_messages[0].role == "assistant"
        assert llm_messages[0].content == "Here's my response"
        assert "Let me think about this..." not in llm_messages[0].content

    def test_agent_message_with_thoughts_included(self):
        """Test that thoughts are included when requested"""
        agent_msg = AgentMessage(
            content=[
                ThoughtContent(
                    text="Let me think about this...",
                    reasoning=self.create_mock_reasoning("thinking"),
                ),
                TextContent(text="Here's my response"),
            ],
            tool_calls=[],
        )

        llm_messages = list(message_to_llm_messages(agent_msg, include_thoughts=True))

        assert len(llm_messages) == 1
        assert llm_messages[0].role == "assistant"
        expected_content = "<think>\nLet me think about this...\n</think>\nHere's my response"
        assert llm_messages[0].content == expected_content

    def test_agent_message_only_thoughts(self):
        """Test message with only thought content"""
        agent_msg = AgentMessage(
            content=[
                ThoughtContent(
                    text="Just thinking...",
                    reasoning=self.create_mock_reasoning("thoughts"),
                )
            ],
            tool_calls=[],
        )

        # Excluded by default
        llm_messages = list(message_to_llm_messages(agent_msg))
        assert len(llm_messages) == 1
        assert llm_messages[0].content == ""

        # Included when requested
        llm_messages = list(message_to_llm_messages(agent_msg, include_thoughts=True))
        assert len(llm_messages) == 1
        assert llm_messages[0].content == "<think>\nJust thinking...\n</think>"

    def test_agent_message_with_tool_running_state(self):
        """Test that only finished tool calls generate results"""
        from agent.types import ToolCallRunning

        running_tool = ToolCallRunning(
            tool_name="slow_tool",
            tool_id="call_running",
            parameters={"task": "processing"},
        )

        finished_tool = ToolCallFinished(
            tool_name="fast_tool",
            tool_id="call_done",
            parameters={"task": "complete"},
            result=ToolCallSuccess(
                content=TextToolContent(text="Task completed"), llm_feedback="Success"
            ),
        )

        agent_msg = AgentMessage(
            content=[TextContent(text="Working on your tasks...")],
            tool_calls=[running_tool, finished_tool],
        )

        llm_messages = list(message_to_llm_messages(agent_msg))

        assert len(llm_messages) == 2

        # Agent message should include both tool calls
        agent_llm_msg = llm_messages[0]
        assert "TOOL_CALL: slow_tool (call_running)" in agent_llm_msg.content
        assert "TOOL_CALL: fast_tool (call_done)" in agent_llm_msg.content

        # Results should only include finished tool
        results_llm_msg = llm_messages[1]
        assert "TOOL_RESULT: fast_tool (call_done)" in results_llm_msg.content
        assert "Success" in results_llm_msg.content
        assert "call_running" not in results_llm_msg.content
