"""
Streaming event system for agent responses
Provides events for text chunks and tool calls with multi-tool support
"""

from typing import Iterator, Dict, Any
from enum import Enum
from dataclasses import dataclass
import json
import re


class StreamEventType(Enum):
    """Types of streaming events"""

    TEXT = "text"
    TOOL_CALL = "tool_call"
    INVALID_TOOL_CALL = "invalid_tool_call"


@dataclass
class TextEvent:
    """A chunk of response text"""

    delta: str
    type: StreamEventType = StreamEventType.TEXT


@dataclass
class ToolCallEvent:
    """Complete tool call detected"""

    id: str
    tool_name: str
    parameters: Dict[str, Any]
    type: StreamEventType = StreamEventType.TOOL_CALL


@dataclass
class InvalidToolCallEvent:
    """Invalid or malformed tool call detected"""

    id: str
    tool_name: str
    error: str
    raw_content: str
    type: StreamEventType = StreamEventType.INVALID_TOOL_CALL


# Union type for streaming events
StreamEvent = TextEvent | ToolCallEvent | InvalidToolCallEvent


class ParsingState(Enum):
    """States for the streaming parser state machine"""

    NORMAL_TEXT = "normal_text"
    TOOL_PARSING = "tool_parsing"
    BETWEEN_TOOLS = "between_tools"


class StreamingParser:
    """
    Parses streaming tokens with prefix-based buffering for optimal UX
    Format: TOOL_CALL: tool_name (call_id)
    """

    def __init__(self, debug: bool = False):
        # Persistent state across chunks
        self.state = ParsingState.NORMAL_TEXT
        self.prefix_buffer = ""  # For partial prefix detection
        self.tool_buffer = ""  # For accumulating tool call content
        self.current_tool_name = ""
        self.current_tool_id = ""
        self.debug = debug

        # Extensible prefix patterns with their handlers
        self.prefixes = {
            "TOOL_CALL:": self._handle_tool_call_complete,
        }

    def parse_chunk(self, chunk: str) -> Iterator[StreamEvent]:
        """Parse a chunk of streaming text and yield appropriate events"""
        # Per-chunk text accumulator
        text_accumulator = ""

        for char in chunk:
            # Process character and get any events
            events = list(self._process_char(char))

            for event in events:
                if isinstance(event, TextEvent):
                    # Accumulate text within this chunk
                    text_accumulator += event.delta
                else:
                    # Non-text event - emit any accumulated text first
                    if text_accumulator:
                        yield TextEvent(delta=text_accumulator)
                        text_accumulator = ""
                    # Then emit the non-text event
                    yield event

        # Emit any remaining text from this chunk
        if text_accumulator:
            yield TextEvent(delta=text_accumulator)

    def _process_char(self, char: str) -> Iterator[StreamEvent]:
        """Process a single character based on current state"""
        if self.state == ParsingState.NORMAL_TEXT:
            yield from self._process_normal_char(char)
        elif self.state == ParsingState.TOOL_PARSING:
            yield from self._process_tool_char(char)
        elif self.state == ParsingState.BETWEEN_TOOLS:
            yield from self._process_between_tools_char(char)

    def _process_normal_char(self, char: str) -> Iterator[StreamEvent]:
        """Process character in normal text mode with prefix detection"""
        potential_buffer = self.prefix_buffer + char

        # Check if potential buffer matches any known prefix
        matching_prefix = self._find_matching_prefix(potential_buffer)

        if matching_prefix:
            self.prefix_buffer = potential_buffer

            # Check if we have a complete match
            if self.prefix_buffer == matching_prefix:
                if self.debug:
                    print(f"[DEBUG] Complete prefix detected: {matching_prefix}")

                # Call the appropriate handler
                handler = self.prefixes[matching_prefix]
                yield from handler()

                # Reset prefix buffer
                self.prefix_buffer = ""
            return
        else:
            # Adding this char breaks all prefixes
            # Emit buffer as text if it exists
            if self.prefix_buffer:
                yield TextEvent(delta=self.prefix_buffer)

            # Check if current char starts any new prefix
            if self._char_starts_prefix(char):
                self.prefix_buffer = char
            else:
                self.prefix_buffer = ""
                yield TextEvent(delta=char)

    def _find_matching_prefix(self, buffer: str) -> str:
        """Find if buffer is a prefix of any known pattern"""
        for prefix in self.prefixes.keys():
            if prefix.startswith(buffer):
                return prefix
        return ""

    def _char_starts_prefix(self, char: str) -> bool:
        """Check if character could start any known prefix"""
        for prefix in self.prefixes.keys():
            if prefix.startswith(char):
                return True
        return False

    def _handle_tool_call_complete(self) -> Iterator[StreamEvent]:
        """Handle complete TOOL_CALL: prefix"""
        self.tool_buffer = ""
        self.state = ParsingState.TOOL_PARSING
        return
        yield  # Make this a generator

    def _process_tool_char(self, char: str) -> Iterator[StreamEvent]:
        """Process character in tool parsing mode"""
        self.tool_buffer += char

        # Try to parse tool name and ID if we don't have them yet
        if not self.current_tool_name:
            # Look for pattern: tool_name (call_id)
            match = re.match(r"\s*(\w+)\s*\(([^)]+)\)", self.tool_buffer)
            if match:
                self.current_tool_name = match.group(1)
                self.current_tool_id = match.group(2)
                # Remove the matched part, leaving potential JSON
                self.tool_buffer = self.tool_buffer[match.end() :]

        # If we have tool name/ID, try to parse JSON
        if self.current_tool_name:
            yield from self._try_parse_json()

    def _try_parse_json(self) -> Iterator[StreamEvent]:
        """Try to parse complete JSON from tool buffer"""
        # Find JSON boundaries
        brace_start = self.tool_buffer.find("{")
        if brace_start == -1:
            return  # No opening brace yet

        # Count braces to find complete JSON
        brace_count = 0
        json_end = -1

        for i in range(brace_start, len(self.tool_buffer)):
            if self.tool_buffer[i] == "{":
                brace_count += 1
            elif self.tool_buffer[i] == "}":
                brace_count -= 1
                if brace_count == 0:
                    json_end = i + 1
                    break

        if json_end != -1:
            # Found complete JSON
            json_str = self.tool_buffer[brace_start:json_end]

            try:
                parameters = json.loads(json_str)

                # Emit successful tool call
                yield ToolCallEvent(
                    id=self.current_tool_id,
                    tool_name=self.current_tool_name,
                    parameters=parameters,
                )

                # Move to between tools state
                self._reset_tool_state()
                self.state = ParsingState.BETWEEN_TOOLS

                # Process any remaining content after the JSON
                remaining = self.tool_buffer[json_end:]
                self.tool_buffer = ""
                for remaining_char in remaining:
                    yield from self._process_char(remaining_char)

            except json.JSONDecodeError as e:
                # Invalid JSON
                yield InvalidToolCallEvent(
                    id=self.current_tool_id,
                    tool_name=self.current_tool_name,
                    error=f"Invalid JSON: {str(e)}",
                    raw_content=self.tool_buffer,
                )
                self._reset_tool_state()
                self.state = ParsingState.NORMAL_TEXT

    def _process_between_tools_char(self, char: str) -> Iterator[StreamEvent]:
        """Process character between tool calls"""
        potential_buffer = self.prefix_buffer + char

        # Check if potential buffer matches any known prefix
        matching_prefix = self._find_matching_prefix(potential_buffer)

        if matching_prefix:
            self.prefix_buffer = potential_buffer

            # Check if we have a complete match
            if self.prefix_buffer == matching_prefix:
                if self.debug:
                    print(
                        f"[DEBUG] Complete prefix detected in between_tools: {matching_prefix}"
                    )

                # Call the appropriate handler
                handler = self.prefixes[matching_prefix]
                yield from handler()

                # Reset prefix buffer
                self.prefix_buffer = ""
        else:
            # Not a valid prefix - ignore content between tools
            # Reset and check if current char starts new prefix
            if self._char_starts_prefix(char):
                self.prefix_buffer = char
            else:
                self.prefix_buffer = ""

        # Always return empty iterator if no events to emit
        return
        yield  # This line never executes but makes this a generator

    def _reset_tool_state(self):
        """Reset tool parsing state"""
        self.current_tool_name = ""
        self.current_tool_id = ""
        self.tool_buffer = ""

    def finalize(self) -> Iterator[StreamEvent]:
        """Finalize parsing and emit any remaining content"""
        if self.debug:
            print(
                f"[DEBUG] Finalize called - state: {self.state}, prefix_buffer: '{self.prefix_buffer}', tool_buffer: '{self.tool_buffer}', tool_name: '{self.current_tool_name}'"
            )

        if self.state == ParsingState.NORMAL_TEXT and self.prefix_buffer:
            if self.debug:
                print(
                    f"[DEBUG] Emitting remaining prefix as text: '{self.prefix_buffer}'"
                )
            # Emit any remaining prefix buffer as text
            yield TextEvent(delta=self.prefix_buffer)
        elif self.state == ParsingState.TOOL_PARSING:
            # Incomplete tool call at end of stream
            if self.current_tool_name:
                if self.debug:
                    print(
                        f"[DEBUG] Incomplete tool call with valid name: {self.current_tool_name}"
                    )
                yield InvalidToolCallEvent(
                    id=self.current_tool_id,
                    tool_name=self.current_tool_name,
                    error="Incomplete tool call at end of stream",
                    raw_content=self.tool_buffer,
                )
            else:
                if self.debug:
                    print(f"[DEBUG] Incomplete tool header, no valid name found")
                # Incomplete tool header
                yield InvalidToolCallEvent(
                    id="unknown",
                    tool_name="unknown",
                    error="Incomplete tool header at end of stream",
                    raw_content=self.tool_buffer,
                )
        elif self.state == ParsingState.BETWEEN_TOOLS and self.prefix_buffer:
            if self.debug:
                print(
                    f"[DEBUG] Emitting remaining prefix from between_tools: '{self.prefix_buffer}'"
                )
            # Emit any remaining prefix as text
            yield TextEvent(delta=self.prefix_buffer)
