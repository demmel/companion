"""
Roleplay presenter with character-aware presentation
"""

from dataclasses import dataclass
from typing import Dict, Any
from rich.console import Console
from agent.agent_events import (
    AgentTextEvent,
    ToolStartedEvent,
    ToolFinishedEvent,
    AgentErrorEvent,
)
from . import BasePresenter


@dataclass
class ActiveTool:
    """Information about an active tool execution"""

    name: str
    parameters: Dict[str, Any]


class RoleplayPresenter(BasePresenter):
    """Presenter optimized for roleplay interactions"""

    def __init__(self, agent):
        super().__init__(agent)
        self.console = Console()
        self.last_speaking_character = None
        self.hidden_tools = {"remember_detail", "correct_detail"}
        self.active_tools: Dict[str, ActiveTool] = {}  # Track tool_id -> ActiveTool

        # Text formatting state machine
        self.formatting_state = {
            "in_quotes": False,
            "in_asterisks": False,
            "in_parens": False,
        }

        # Newline compression state
        self.consecutive_newlines = 0
        self.max_consecutive_newlines = 2

        # Mood emoji mapping
        self.mood_emojis = {
            "happy": "😊",
            "excited": "🤩",
            "playful": "😈",
            "flirtatious": "😘",
            "sad": "😢",
            "angry": "😠",
            "frustrated": "😤",
            "annoyed": "🙄",
            "nervous": "😰",
            "shy": "😊",
            "confident": "😎",
            "mysterious": "😏",
            "seductive": "😍",
            "mischievous": "😋",
            "gentle": "🥰",
            "fierce": "🔥",
            "neutral": "😐",
            "curious": "🤔",
            "surprised": "😯",
            "worried": "😟",
        }

        # Mood colors for styling
        self.mood_colors = {
            "happy": "bright_yellow",
            "excited": "bright_magenta",
            "playful": "bright_cyan",
            "flirtatious": "bright_magenta",
            "seductive": "magenta",
            "sad": "blue",
            "angry": "bright_red",
            "frustrated": "red",
            "annoyed": "yellow",
            "mysterious": "magenta",
            "confident": "bright_blue",
            "gentle": "green",
            "fierce": "bright_red",
            "neutral": "white",
            "mischievous": "bright_cyan",
        }

    def _print(self, *args, **kwargs):
        """Central print method that handles newline compression and tracking"""
        content = args[0] if args else ""
        end = kwargs.get("end", "\n")

        # Handle empty print() calls - these are just adding newlines
        if not args or (len(args) == 1 and content == ""):
            self.consecutive_newlines += 1
            if self.consecutive_newlines <= self.max_consecutive_newlines:
                self.console.print(*args, **kwargs)
            return

        # Handle single newline characters (char-by-char processing)
        if content == "\n" and end == "":
            self.consecutive_newlines += 1
            if self.consecutive_newlines <= self.max_consecutive_newlines:
                self.console.print(*args, **kwargs)
            return

        # Handle regular content with potential trailing newlines
        content_str = str(content)

        # Check if content has actual text (not just newlines)
        if content_str.rstrip("\n"):
            # Content has non-newline characters, print normally and reset counter
            self.consecutive_newlines = 0
            self.console.print(*args, **kwargs)

            # Count trailing newlines in content + end newline
            trailing_newlines = len(content_str) - len(content_str.rstrip("\n"))
            if end == "\n":
                trailing_newlines += 1
            self.consecutive_newlines = trailing_newlines
        else:
            # Content is only newlines
            total_newlines = len(content_str)
            if end == "\n":
                total_newlines += 1

            # Check if we should print based on consecutive limit
            if (
                self.consecutive_newlines + total_newlines
                <= self.max_consecutive_newlines
            ):
                self.console.print(*args, **kwargs)
                self.consecutive_newlines += total_newlines
            else:
                # Skip printing, but update counter to limit
                self.consecutive_newlines = self.max_consecutive_newlines

    def _ensure_tool_spacing(self):
        """Ensure proper spacing before tool output (double newlines)"""
        if self.consecutive_newlines == 0:
            self._print()
            self._print()
        elif self.consecutive_newlines == 1:
            self._print()

    def process_stream(self, user_input: str) -> None:
        """Process events with roleplay-specific presentation"""
        response_started = False

        for event in self.agent.chat_stream(user_input):
            if isinstance(event, AgentTextEvent):
                # Show character header if this is start of response
                if not response_started:
                    self._show_character_header()
                    response_started = True

                self._handle_formatted_text(event.content)

            elif isinstance(event, ToolStartedEvent):
                self._handle_tool_started(event)

            elif isinstance(event, ToolFinishedEvent):
                self._handle_tool_finished(event)

            elif isinstance(event, AgentErrorEvent):
                self._print(f"[red]❌ Error: {event.message}[/red]")

        self._print()  # Final newline

    def _handle_formatted_text(self, content: str):
        """Handle text with character-by-character formatting state machine"""
        for char in content:
            # Handle formatting characters specially
            if char == '"':
                # Quote characters are always printed in quote style
                self._print(char, end="", style="bright_green")
                self.formatting_state["in_quotes"] = not self.formatting_state[
                    "in_quotes"
                ]
            elif char == "*":
                # Only treat as action marker if not inside quotes
                if self.formatting_state["in_quotes"]:
                    # Inside quotes - just print as regular text
                    self._print(char, end="", style="bright_green")
                else:
                    # Outside quotes - treat as action marker
                    self._print(char, end="", style="italic bright_blue")
                    self.formatting_state["in_asterisks"] = not self.formatting_state[
                        "in_asterisks"
                    ]
            elif char == "(":
                if not self.formatting_state["in_parens"]:
                    # Opening paren starts thought style
                    self._print(char, end="", style="dim yellow")
                    self.formatting_state["in_parens"] = True
                else:
                    # Regular paren inside thoughts
                    self._print(char, end="", style="dim yellow")
            elif char == ")":
                if self.formatting_state["in_parens"]:
                    # Closing paren ends thought style
                    self._print(char, end="", style="dim yellow")
                    self.formatting_state["in_parens"] = False
                else:
                    # Regular paren outside thoughts
                    self._print(char, end="", style=None)
            else:
                # Regular characters use current formatting state
                if self.formatting_state["in_quotes"]:
                    style = "bright_green"
                elif self.formatting_state["in_asterisks"]:
                    style = "italic bright_blue"
                elif self.formatting_state["in_parens"]:
                    style = "dim yellow"
                else:
                    style = None  # Default console color

                self._print(char, end="", style=style)

    def _show_character_header(self):
        """Show character header if character has changed or first response"""
        state = self.agent.get_state()
        current_char_id = state.get("current_character_id")

        if not current_char_id:
            # No character active, show generic header
            self._print(f"\n[bold green]🤖 Agent[/bold green]: ", end="")
            return

        characters = state.get("characters", {})
        current_char = characters.get(current_char_id)

        if not current_char:
            self._print(f"\n[bold green]🤖 Agent[/bold green]: ", end="")
            return

        # Only show header if character changed
        if self.last_speaking_character != current_char_id:
            char_name = current_char["name"]
            mood = current_char.get("mood", "neutral")
            intensity = current_char.get("mood_intensity", "medium")

            mood_emoji = self.mood_emojis.get(mood, "😐")
            mood_color = self.mood_colors.get(mood, "white")

            header = f"\n🎭 **{char_name}** {mood_emoji} *({mood} - {intensity})*"
            self._print(header, style=mood_color)

            # Add scene context if available
            scene = state.get("global_scene")
            if scene:
                scene_text = f"📍 *{scene['location']}*"
                if scene.get("atmosphere"):
                    scene_text += f" - {scene['atmosphere']}"
                self._print(scene_text, style="dim")

            self._print()  # Blank line before dialogue
            self.last_speaking_character = current_char_id

    def _handle_tool_started(self, event):
        """Handle tool started events"""
        # Track tool info for later use in finished event
        self.active_tools[event.tool_id] = ActiveTool(
            name=event.tool_name, parameters=event.parameters
        )

        if event.tool_name in self.hidden_tools:
            return  # Don't show anything for hidden tools

        # For visible tools, we might show something depending on the tool
        # For now, most tools are silent until completion

    def _handle_tool_finished(self, event):
        """Handle tool finished events with roleplay-specific presentation"""
        # Get tool info from when it started
        active_tool = self.active_tools.get(event.tool_id)
        if not active_tool:
            return  # Tool not tracked

        tool_name = active_tool.name
        tool_params = active_tool.parameters

        if tool_name in self.hidden_tools:
            return  # Don't show anything for hidden tools

        if tool_name == "assume_character":
            # Don't show - character header will handle the transition
            return

        elif tool_name == "set_mood":
            self._show_mood_transition(event, tool_params)

        elif tool_name == "character_action":
            self._show_character_action(event, tool_params)

        elif tool_name == "internal_thought":
            self._show_internal_thought(event, tool_params)

        elif tool_name == "scene_setting":
            self._show_scene_setting(event, tool_params)

        else:
            # For other tools, show generic completion
            self._ensure_tool_spacing()
            self._print(f"[dim green]✓ {tool_name} completed[/dim green]")
            self._print()

        # Clean up tracked tool
        del self.active_tools[event.tool_id]

    def _show_mood_transition(self, event, tool_params):
        """Show mood transition with emoji and styling"""
        # Get current character for mood context
        state = self.agent.get_state()
        current_char_id = state.get("current_character_id")

        if current_char_id:
            characters = state.get("characters", {})
            current_char = characters.get(current_char_id)
            if current_char:
                old_mood = current_char.get("previous_mood", "neutral")
                new_mood = tool_params.get("mood", "neutral")

                old_emoji = self.mood_emojis.get(old_mood, "😐")
                new_emoji = self.mood_emojis.get(new_mood, "😐")
                mood_color = self.mood_colors.get(new_mood, "white")

                # Ensure proper spacing before tool output
                self._ensure_tool_spacing()

                # Show transition
                transition = f"{old_emoji} → {new_emoji}"
                self._print(f"[{mood_color}]{transition}[/{mood_color}]", end=" ")

                # Show flavor text if provided in parameters
                flavor_text = tool_params.get("flavor_text", "")
                if flavor_text.strip():
                    self._print(f"[dim {mood_color}]{flavor_text}[/dim {mood_color}]")
                else:
                    self._print()
                # Add trailing newline for proper spacing
                self._print()

    def _show_character_action(self, event, tool_params):
        """Show character action with special styling"""
        # Display the action parameter, not the tool result
        action_text = tool_params.get("action", "")
        if action_text.strip():
            # Ensure proper spacing before tool output
            self._ensure_tool_spacing()

            self._print(f"[italic bright_blue]*{action_text}*[/italic bright_blue]")
            self._print()  # Add trailing newline

    def _show_internal_thought(self, event, tool_params):
        """Show internal thoughts with special styling"""
        thought_text = tool_params.get("thought", "")
        if thought_text.strip():
            # Ensure proper spacing before tool output
            self._ensure_tool_spacing()

            self._print(f"[dim yellow]💭 {thought_text}[/dim yellow]")
            self._print()  # Add trailing newline

    def _show_scene_setting(self, event, tool_params):
        """Show scene setting changes"""
        location = tool_params.get("location", "")
        atmosphere = tool_params.get("atmosphere", "")
        time = tool_params.get("time", "")

        scene_parts = []
        if location:
            scene_parts.append(location)
        if atmosphere:
            scene_parts.append(atmosphere)
        if time:
            scene_parts.append(f"({time})")

        if scene_parts:
            # Ensure proper spacing before tool output
            self._ensure_tool_spacing()

            scene_text = " - ".join(scene_parts)
            self._print(f"[dim magenta]📍 {scene_text}[/dim magenta]")
            self._print()  # Add trailing newline
