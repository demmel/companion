"""
Tests for the presenter system
"""

import pytest
from unittest.mock import Mock, patch
from io import StringIO

from agent.core import Agent
from agent.config import get_config
from agent.presenters import get_presenter_for_config
from agent.presenters.roleplay import RoleplayPresenter
from agent.presenters.generic import GenericPresenter


class TestPresenterSystem:
    """Test the presenter system"""

    def test_get_presenter_for_config(self):
        """Test that correct presenters are returned for different configs"""
        config = get_config("roleplay")
        agent = Agent(config=config, model="test-model")

        # Test roleplay presenter
        presenter = get_presenter_for_config("roleplay", agent)
        assert isinstance(presenter, RoleplayPresenter)
        assert presenter.agent is agent

        # Test generic presenter fallback
        presenter = get_presenter_for_config("coding", agent)
        assert isinstance(presenter, GenericPresenter)
        assert presenter.agent is agent

    def test_roleplay_presenter_initialization(self):
        """Test roleplay presenter initialization"""
        config = get_config("roleplay")
        agent = Agent(config=config, model="test-model")
        presenter = RoleplayPresenter(agent)

        assert presenter.agent is agent
        assert presenter.last_speaking_character is None
        assert "remember_detail" in presenter.hidden_tools
        assert "correct_detail" in presenter.hidden_tools
        assert "happy" in presenter.mood_emojis
        assert "angry" in presenter.mood_colors

    def test_generic_presenter_initialization(self):
        """Test generic presenter initialization"""
        config = get_config("general")
        agent = Agent(config=config, model="test-model")
        presenter = GenericPresenter(agent)

        assert presenter.agent is agent
        assert presenter.active_tools == {}

    @patch("rich.console.Console.print")
    def test_roleplay_presenter_character_header(self, mock_print):
        """Test that roleplay presenter shows character headers correctly"""
        config = get_config("roleplay")
        agent = Agent(config=config, model="test-model")
        presenter = RoleplayPresenter(agent)

        # Mock agent state with character
        agent.set_state("current_character_id", "char_123")
        agent.set_state(
            "characters",
            {
                "char_123": {
                    "name": "Alice",
                    "mood": "happy",
                    "mood_intensity": "high",
                    "personality": "Cheerful helper",
                }
            },
        )

        # Mock LLM response
        mock_llm = Mock()
        agent.llm = mock_llm
        text_response = [{"message": {"content": "Hello there!"}}]
        mock_llm.chat.return_value = iter(text_response)

        # Process stream
        presenter.process_stream("Hi Alice")

        # Verify character header was shown
        # Should have multiple print calls including character header
        assert mock_print.call_count >= 2

        # Look for character header in the calls
        header_found = False
        for call in mock_print.call_args_list:
            args, kwargs = call
            if args and "Alice" in str(args[0]) and "🎭" in str(args[0]):
                header_found = True
                break

        assert header_found, "Character header should be displayed"

    @patch("rich.console.Console.print")
    def test_roleplay_presenter_hidden_tools(self, mock_print):
        """Test that hidden tools don't show in roleplay presenter"""
        config = get_config("roleplay")
        agent = Agent(config=config, model="test-model")
        presenter = RoleplayPresenter(agent)

        # Mock LLM response with hidden tool
        mock_llm = Mock()
        agent.llm = mock_llm
        tool_response = [
            {"message": {"content": "TOOL_CALL: remember_detail (call_1)\n"}},
            {"message": {"content": '{"detail": "User likes coffee"}'}},
        ]
        mock_llm.chat.return_value = iter(tool_response)

        # Mock tool execution
        agent.tools.execute = Mock(return_value="Detail remembered")

        # Process stream
        presenter.process_stream("I like coffee")

        # Verify that remember_detail tool execution is hidden
        # Should not see any tool-related output
        for call in mock_print.call_args_list:
            args, kwargs = call
            if args:
                content = str(args[0])
                assert "remember_detail" not in content
                assert "🔧" not in content  # Tool start indicator
                assert (
                    "✓" not in content or "completed" not in content
                )  # Tool completion
