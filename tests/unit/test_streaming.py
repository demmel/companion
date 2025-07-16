"""
Tests for streaming parser with comprehensive edge case coverage
"""

from agent.streaming import (
    StreamingParser,
    TextEvent,
    ToolCallEvent,
    InvalidToolCallEvent,
)


class TestStreamingParser:
    """Test cases for the streaming parser state machine"""

    def test_simple_text_only(self):
        """Simple text should emit single text event"""
        parser = StreamingParser()
        events = list(parser.parse_chunk("Hello world"))

        assert len(events) == 1
        assert isinstance(events[0], TextEvent)
        assert events[0].delta == "Hello world"

    def test_simple_tool_call(self):
        """Simple tool call should emit single tool call event"""
        parser = StreamingParser()
        events = list(
            parser.parse_chunk('TOOL_CALL: search (call_1)\n{"query": "test"}')
        )

        assert len(events) == 1
        assert isinstance(events[0], ToolCallEvent)
        assert events[0].id == "call_1"
        assert events[0].tool_name == "search"
        assert events[0].parameters == {"query": "test"}

    def test_text_then_tool_call(self):
        """Text followed by tool call should emit text then tool"""
        parser = StreamingParser()
        events = list(
            parser.parse_chunk(
                'Let me search for that. TOOL_CALL: search (call_1)\n{"query": "test"}'
            )
        )

        assert len(events) == 2
        assert isinstance(events[0], TextEvent)
        assert events[0].delta == "Let me search for that. "
        assert isinstance(events[1], ToolCallEvent)
        assert events[1].tool_name == "search"

    def test_tool_call_then_text_ignored(self):
        """Text after tool call should be ignored"""
        parser = StreamingParser()
        events = list(
            parser.parse_chunk(
                'TOOL_CALL: search (call_1)\n{"query": "test"} and here is more text'
            )
        )

        assert len(events) == 1
        assert isinstance(events[0], ToolCallEvent)
        assert events[0].tool_name == "search"
        # Text after tool call should be ignored

    def test_multiple_tool_calls(self):
        """Multiple tool calls should emit multiple events"""
        parser = StreamingParser()
        events = list(
            parser.parse_chunk(
                'TOOL_CALL: search (call_1)\n{"query": "test"} TOOL_CALL: filter (call_2)\n{"type": "recent"}'
            )
        )

        assert len(events) == 2
        assert isinstance(events[0], ToolCallEvent)
        assert events[0].id == "call_1"
        assert events[0].tool_name == "search"
        assert isinstance(events[1], ToolCallEvent)
        assert events[1].id == "call_2"
        assert events[1].tool_name == "filter"

    def test_invalid_tool_call_bad_json(self):
        """Invalid JSON should emit invalid tool call event"""
        parser = StreamingParser()
        events = list(
            parser.parse_chunk('TOOL_CALL: search (call_1)\n{"query": invalid}')
        )

        assert len(events) == 1
        assert isinstance(events[0], InvalidToolCallEvent)
        assert events[0].id == "call_1"
        assert events[0].tool_name == "search"
        assert "Invalid JSON" in events[0].error

    def test_false_prefix_tools(self):
        """TOOLS should emit as text, not trigger tool parsing"""
        parser = StreamingParser()
        events = list(parser.parse_chunk("Check out these TOOLS for development"))

        assert len(events) == 1
        assert isinstance(events[0], TextEvent)
        assert events[0].delta == "Check out these TOOLS for development"

    def test_false_prefix_tool_call_typo(self):
        """TOOL_CAL should emit as text, not trigger tool parsing"""
        parser = StreamingParser()
        events = list(parser.parse_chunk("I made a TOOL_CAL typo here"))

        assert len(events) == 1
        assert isinstance(events[0], TextEvent)
        assert events[0].delta == "I made a TOOL_CAL typo here"

    def test_partial_prefix_across_chunks(self):
        """Partial TOOL_CALL prefix should work across chunks"""
        parser = StreamingParser()

        # First chunk with partial prefix
        events1 = list(parser.parse_chunk("Hello TOO"))
        assert len(events1) == 1
        assert isinstance(events1[0], TextEvent)
        assert events1[0].delta == "Hello "

        # Second chunk completes the prefix
        events2 = list(parser.parse_chunk("L_CALL: search (call_1)\n"))
        assert len(events2) == 0  # No events yet, still building tool call

        # Third chunk completes the tool call
        events3 = list(parser.parse_chunk('{"query": "test"}'))
        assert len(events3) == 1
        assert isinstance(events3[0], ToolCallEvent)
        assert events3[0].tool_name == "search"

    def test_tool_call_json_across_chunks(self):
        """Tool call JSON spanning chunks should work"""
        parser = StreamingParser()

        # First chunk with tool header
        events1 = list(parser.parse_chunk("TOOL_CALL: search (call_1)\n"))
        assert len(events1) == 0

        # Second chunk with partial JSON
        events2 = list(parser.parse_chunk('{"query":'))
        assert len(events2) == 0

        # Third chunk completes JSON
        events3 = list(parser.parse_chunk(' "test"}'))
        assert len(events3) == 1
        assert isinstance(events3[0], ToolCallEvent)
        assert events3[0].parameters == {"query": "test"}

    def test_text_batching_within_chunk(self):
        """Consecutive text chars should be batched into single event"""
        parser = StreamingParser()
        events = list(parser.parse_chunk("Hello world this is a long sentence"))

        assert len(events) == 1
        assert isinstance(events[0], TextEvent)
        assert events[0].delta == "Hello world this is a long sentence"

    def test_text_not_batched_across_chunks(self):
        """Text should not be batched across chunks"""
        parser = StreamingParser()

        events1 = list(parser.parse_chunk("Hello "))
        assert len(events1) == 1
        assert isinstance(events1[0], TextEvent)
        assert events1[0].delta == "Hello "

        events2 = list(parser.parse_chunk("world"))
        assert len(events2) == 1
        assert isinstance(events2[0], TextEvent)
        assert events2[0].delta == "world"

    def test_mixed_text_and_tools_in_chunk(self):
        """Mixed content should emit correctly batched events"""
        parser = StreamingParser()
        events = list(
            parser.parse_chunk(
                'Hello TOOL_CALL: search (call_1)\n{"query": "test"} ignored'
            )
        )

        assert len(events) == 2
        assert isinstance(events[0], TextEvent)
        assert events[0].delta == "Hello "
        assert isinstance(events[1], ToolCallEvent)
        assert events[1].tool_name == "search"

    def test_finalize_with_remaining_text(self):
        """Finalize should emit any remaining text"""
        parser = StreamingParser()
        list(parser.parse_chunk("Hello TOO"))  # Leaves "TOO" in prefix buffer

        events = list(parser.finalize())
        assert len(events) == 1
        assert isinstance(events[0], TextEvent)
        assert events[0].delta == "TOO"

    def test_finalize_with_incomplete_tool(self):
        """Finalize should emit invalid tool call for incomplete tool"""
        parser = StreamingParser()
        list(parser.parse_chunk('TOOL_CALL: search (call_1)\n{"incomplete'))

        events = list(parser.finalize())
        assert len(events) == 1
        assert isinstance(events[0], InvalidToolCallEvent)
        assert events[0].tool_name == "search"
        assert "Incomplete tool call" in events[0].error

    def test_empty_chunks(self):
        """Empty chunks should not cause issues"""
        parser = StreamingParser()
        events1 = list(parser.parse_chunk(""))
        events2 = list(parser.parse_chunk("Hello"))
        events3 = list(parser.parse_chunk(""))

        assert len(events1) == 0
        assert len(events2) == 1
        assert isinstance(events2[0], TextEvent)
        assert events2[0].delta == "Hello"
        assert len(events3) == 0

    def test_complex_realistic_scenario(self):
        """Complex realistic scenario with multiple features"""
        parser = StreamingParser()

        # Simulate realistic streaming chunks
        chunks = [
            "I'll help you search for that information. ",
            "TOOL_CALL: search_files (call_1)\n",
            '{"pattern": "*.py", "directory": "./src"} ',
            "TOOL_CALL: analyze_code (call_2)\n",
            '{"file_path": "main.py"} ',
            "That should give us what we need.",
        ]

        all_events = []
        for chunk in chunks:
            events = list(parser.parse_chunk(chunk))
            all_events.extend(events)

        # Should have: text, tool_call, tool_call (last text ignored)
        assert len(all_events) == 3
        assert isinstance(all_events[0], TextEvent)
        assert all_events[0].delta == "I'll help you search for that information. "
        assert isinstance(all_events[1], ToolCallEvent)
        assert all_events[1].tool_name == "search_files"
        assert all_events[1].id == "call_1"
        assert isinstance(all_events[2], ToolCallEvent)
        assert all_events[2].tool_name == "analyze_code"
        assert all_events[2].id == "call_2"

    def test_unknown_sequences_as_text(self):
        """Test that unknown sequences are treated as normal text"""
        parser = StreamingParser()
        events = list(parser.parse_chunk("UNKNOWN_SEQUENCE: something"))

        assert len(events) == 1
        assert isinstance(events[0], TextEvent)
        assert events[0].delta == "UNKNOWN_SEQUENCE: something"

    def test_prefix_system_extensibility(self):
        """Test that new prefixes can be easily added to the system"""
        parser = StreamingParser()

        # Verify the prefix dictionary contains our expected patterns
        assert "TOOL_CALL:" in parser.prefixes

        # Verify handlers are callable
        assert callable(parser.prefixes["TOOL_CALL:"])

        # Verify we can easily add new prefixes (extensibility)
        assert hasattr(parser, "prefixes")
        assert isinstance(parser.prefixes, dict)

    def test_simple_thought_parsing(self):
        """Simple thought section should emit thought events"""
        parser = StreamingParser()
        events = list(parser.parse_chunk("<think>Let me think about this...</think>"))

        assert len(events) == 1
        assert isinstance(events[0], TextEvent)
        assert events[0].delta == "Let me think about this..."
        assert events[0].is_thought is True

    def test_thought_with_regular_text(self):
        """Thought section followed by regular text"""
        parser = StreamingParser()
        events = list(parser.parse_chunk("<think>Thinking...</think>Here's my response"))

        assert len(events) == 2
        assert isinstance(events[0], TextEvent)
        assert events[0].delta == "Thinking..."
        assert events[0].is_thought is True
        assert isinstance(events[1], TextEvent)
        assert events[1].delta == "Here's my response"
        assert events[1].is_thought is False

    def test_thought_with_tool_calls_inside(self):
        """Tool calls inside thoughts should be treated as plain text"""
        parser = StreamingParser()
        events = list(parser.parse_chunk('<think>I should use TOOL_CALL: search {"query": "test"}</think>'))

        assert len(events) == 1
        assert isinstance(events[0], TextEvent)
        assert events[0].delta == 'I should use TOOL_CALL: search {"query": "test"}'
        assert events[0].is_thought is True

    def test_thought_across_chunks(self):
        """Thought content spanning multiple chunks"""
        parser = StreamingParser()
        
        events1 = list(parser.parse_chunk("<think>Let me think"))
        assert len(events1) == 1
        assert events1[0].is_thought is True
        assert events1[0].delta == "Let me think"
        
        events2 = list(parser.parse_chunk(" about this problem"))
        assert len(events2) == 1
        assert events2[0].is_thought is True
        assert events2[0].delta == " about this problem"
        
        events3 = list(parser.parse_chunk("</think>Done thinking"))
        assert len(events3) == 1
        assert events3[0].is_thought is False
        assert events3[0].delta == "Done thinking"

    def test_tool_calls_after_thoughts(self):
        """Tool calls should work normally after thought sections"""
        parser = StreamingParser()
        events = list(parser.parse_chunk('<think>Planning...</think>TOOL_CALL: search (call_1)\n{"query": "test"}'))

        assert len(events) == 2
        assert isinstance(events[0], TextEvent)
        assert events[0].delta == "Planning..."
        assert events[0].is_thought is True
        assert isinstance(events[1], ToolCallEvent)
        assert events[1].tool_name == "search"
