"""
Tests for roleplay presenter whitespace and newline compression
Focus on actual output behavior, not internal state
"""

import pytest
from unittest.mock import Mock
from io import StringIO

from agent.presenters.roleplay import RoleplayPresenter
from agent.agent_events import AgentTextEvent, ToolStartedEvent, ToolFinishedEvent


class TestRoleplayPresenterWhitespace:
    """Test roleplay presenter output formatting and newline compression"""

    def setup_method(self):
        """Set up test environment with output capture"""
        self.mock_agent = Mock()
        self.mock_agent.get_state.return_value = {
            "current_character_id": "char1",
            "characters": {
                "char1": {
                    "name": "Luna",
                    "mood": "neutral",
                    "mood_intensity": "moderate",
                }
            },
        }

        self.presenter = RoleplayPresenter(self.mock_agent)
        # Replace console with StringIO to capture output
        self.output_buffer = StringIO()

        # Mock console.print to write to our buffer
        def mock_print(*args, **kwargs):
            content = args[0] if args else ""
            end = kwargs.get("end", "\n")
            # Strip rich markup for testing (simplified)
            import re

            clean_content = re.sub(r"\[/?[^\]]*\]", "", str(content))
            self.output_buffer.write(clean_content + end)

        self.presenter.console.print = mock_print

    def get_output(self):
        """Get the captured output"""
        return self.output_buffer.getvalue()

    def clear_output(self):
        """Clear the output buffer"""
        self.output_buffer = StringIO()

        def mock_print(*args, **kwargs):
            content = args[0] if args else ""
            end = kwargs.get("end", "\n")
            import re

            clean_content = re.sub(r"\[/?[^\]]*\]", "", str(content))
            self.output_buffer.write(clean_content + end)

        self.presenter.console.print = mock_print

    def simulate_stream(self, events):
        """Simulate a stream of events"""
        for event in events:
            if isinstance(event, AgentTextEvent):
                self.presenter._handle_formatted_text(event.content)
            elif isinstance(event, ToolStartedEvent):
                self.presenter._handle_tool_started(event)
            elif isinstance(event, ToolFinishedEvent):
                self.presenter._handle_tool_finished(event)

    def test_excessive_newlines_compressed_single_chunk(self):
        """Test that excessive newlines in a single chunk are compressed to max 2"""
        events = [AgentTextEvent(content="Hello\n\n\n\n\n\nWorld")]

        self.simulate_stream(events)
        output = self.get_output()

        # Should be compressed to max 2 consecutive newlines
        assert "Hello\n\nWorld" in output
        assert "\n\n\n" not in output

    def test_excessive_newlines_across_chunks(self):
        """Test newline compression across multiple streaming chunks"""
        events = [
            AgentTextEvent(content="Hello\n"),
            AgentTextEvent(content="\n"),
            AgentTextEvent(content="\n"),
            AgentTextEvent(content="\n"),
            AgentTextEvent(content="\n"),
            AgentTextEvent(content="World"),
        ]

        self.simulate_stream(events)
        output = self.get_output()

        # Should be compressed to max 2 consecutive newlines
        assert "Hello\n\nWorld" in output
        assert "\n\n\n" not in output

    def test_tool_surrounded_by_double_newlines_no_prior_newlines(self):
        """Test tool output is surrounded by double newlines when no prior newlines"""
        events = [
            AgentTextEvent(content="Some text"),
            ToolStartedEvent(
                tool_name="character_action",
                tool_id="test1",
                parameters={"action": "waves"},
            ),
            ToolFinishedEvent(tool_id="test1", result="success"),
            AgentTextEvent(content="More text"),
        ]

        # Set up the active tool
        self.presenter.active_tools["test1"] = Mock()
        self.presenter.active_tools["test1"].name = "character_action"
        self.presenter.active_tools["test1"].parameters = {"action": "waves"}

        self.simulate_stream(events)
        output = self.get_output()

        # Tool output should be surrounded by double newlines
        assert "Some text\n\n*waves*\n\nMore text" in output

    def test_tool_no_extra_spacing_with_existing_newlines(self):
        """Test tool doesn't add extra spacing when there are already newlines"""
        events = [
            AgentTextEvent(content="Some text\n\n"),
            ToolStartedEvent(
                tool_name="character_action",
                tool_id="test1",
                parameters={"action": "waves"},
            ),
            ToolFinishedEvent(tool_id="test1", result="success"),
            AgentTextEvent(content="More text"),
        ]

        self.presenter.active_tools["test1"] = Mock()
        self.presenter.active_tools["test1"].name = "character_action"
        self.presenter.active_tools["test1"].parameters = {"action": "waves"}

        self.simulate_stream(events)
        output = self.get_output()

        # Should have proper tool spacing but not excessive
        assert "Some text\n\n*waves*\n\nMore text" in output
        assert "\n\n\n" not in output

    def test_multiple_tools_proper_spacing(self):
        """Test multiple tools in sequence have proper spacing"""
        events = [
            AgentTextEvent(content="Text"),
            ToolStartedEvent(
                tool_name="character_action",
                tool_id="test1",
                parameters={"action": "waves"},
            ),
            ToolFinishedEvent(tool_id="test1", result="success"),
            ToolStartedEvent(
                tool_name="internal_thought",
                tool_id="test2",
                parameters={"thought": "thinking"},
            ),
            ToolFinishedEvent(tool_id="test2", result="success"),
            AgentTextEvent(content="More text"),
        ]

        self.presenter.active_tools["test1"] = Mock()
        self.presenter.active_tools["test1"].name = "character_action"
        self.presenter.active_tools["test1"].parameters = {"action": "waves"}

        self.presenter.active_tools["test2"] = Mock()
        self.presenter.active_tools["test2"].name = "internal_thought"
        self.presenter.active_tools["test2"].parameters = {"thought": "thinking"}

        self.simulate_stream(events)
        output = self.get_output()

        # Each tool should be properly spaced
        lines = output.split("\n")
        # Should not have more than 2 consecutive empty lines anywhere
        consecutive_empty = 0
        max_consecutive = 0
        for line in lines:
            if line.strip() == "":
                consecutive_empty += 1
                max_consecutive = max(max_consecutive, consecutive_empty)
            else:
                consecutive_empty = 0

        assert max_consecutive <= 2

    def test_newlines_mixed_with_formatted_text(self):
        """Test newline compression with quotes and actions"""
        events = [AgentTextEvent(content='"Hello"\n\n\n*waves*\n\n\n\n"World"')]

        self.simulate_stream(events)
        output = self.get_output()

        # Should compress newlines but preserve formatting
        assert "\n\n\n\n" not in output
        assert '"Hello"' in output
        assert "*waves*" in output
        assert '"World"' in output

    def test_hidden_tools_no_output_spacing(self):
        """Test that hidden tools don't affect spacing"""
        events = [
            AgentTextEvent(content="Some text"),
            ToolStartedEvent(
                tool_name="remember_detail",
                tool_id="test1",
                parameters={"detail": "info"},
            ),
            ToolFinishedEvent(tool_id="test1", result="success"),
            AgentTextEvent(content="More text"),
        ]

        self.presenter.active_tools["test1"] = Mock()
        self.presenter.active_tools["test1"].name = "remember_detail"  # Hidden tool
        self.presenter.active_tools["test1"].parameters = {"detail": "info"}

        self.simulate_stream(events)
        output = self.get_output()

        # Hidden tools should not add any spacing or output
        clean_output = output.replace("\n", "")
        assert clean_output == "Some textMore text"

    def test_scene_setting_proper_spacing(self):
        """Test scene setting tool has proper spacing"""
        events = [
            AgentTextEvent(content="Looking around"),
            ToolStartedEvent(
                tool_name="scene_setting",
                tool_id="test1",
                parameters={"location": "garden", "atmosphere": "peaceful"},
            ),
            ToolFinishedEvent(tool_id="test1", result="success"),
            AgentTextEvent(content="I continue"),
        ]

        self.presenter.active_tools["test1"] = Mock()
        self.presenter.active_tools["test1"].name = "scene_setting"
        self.presenter.active_tools["test1"].parameters = {
            "location": "garden",
            "atmosphere": "peaceful",
        }

        self.simulate_stream(events)
        output = self.get_output()

        # Scene setting should be surrounded by double newlines
        assert "Looking around\n\n📍 garden - peaceful\n\nI continue" in output

    def test_mood_transition_spacing(self):
        """Test mood transition has proper spacing"""
        events = [
            AgentTextEvent(content="I feel different"),
            ToolStartedEvent(
                tool_name="set_mood", tool_id="test1", parameters={"mood": "happy"}
            ),
            ToolFinishedEvent(tool_id="test1", result="success"),
            AgentTextEvent(content="Much better"),
        ]

        self.presenter.active_tools["test1"] = Mock()
        self.presenter.active_tools["test1"].name = "set_mood"
        self.presenter.active_tools["test1"].parameters = {"mood": "happy"}

        self.simulate_stream(events)
        output = self.get_output()

        # Mood transition should be surrounded by double newlines
        assert "I feel different\n\n" in output
        assert "→" in output  # Mood transition emoji
        assert "\n\nMuch better" in output
