"""
Unit tests for trigger-based streaming events
"""

import pytest
from unittest.mock import Mock, MagicMock
from datetime import datetime

from agent.chain_of_action.reasoning_loop import ActionBasedReasoningLoop
from agent.chain_of_action.trigger_history import TriggerHistory
from agent.chain_of_action.callbacks import ActionCallback
from agent.chain_of_action.action_types import ActionType
from agent.chain_of_action.context import ActionResult
from agent.state import State, create_default_agent_state
from agent.llm import LLM


class MockStreamingCallback(ActionCallback):
    """Test callback that captures emitted events"""
    
    def __init__(self):
        self.events = []
    
    def on_trigger_started(self, trigger_type: str, trigger_content: str, entry_id: str):
        self.events.append({
            'type': 'trigger_started',
            'trigger_type': trigger_type,
            'trigger_content': trigger_content,
            'entry_id': entry_id
        })
    
    def on_trigger_completed(self, entry_id: str, total_actions: int, successful_actions: int):
        self.events.append({
            'type': 'trigger_completed',
            'entry_id': entry_id,
            'total_actions': total_actions,
            'successful_actions': successful_actions
        })
    
    def on_sequence_started(self, sequence_number: int, total_actions: int, reasoning: str):
        self.events.append({
            'type': 'sequence_started',
            'sequence_number': sequence_number,
            'total_actions': total_actions
        })
    
    def on_action_started(self, action_type: ActionType, context: str, sequence_number: int, action_number: int, entry_id: str):
        self.events.append({
            'type': 'action_started',
            'action_type': action_type.value,
            'context': context,
            'entry_id': entry_id
        })
    
    def on_action_progress(self, action_type: ActionType, progress_data, sequence_number: int, action_number: int, entry_id: str):
        self.events.append({
            'type': 'action_progress',
            'action_type': action_type.value,
            'progress_data': progress_data,
            'entry_id': entry_id
        })
    
    def on_action_finished(self, action_type: ActionType, result: ActionResult, sequence_number: int, action_number: int, entry_id: str):
        self.events.append({
            'type': 'action_finished',
            'action_type': action_type.value,
            'result': result,
            'entry_id': entry_id
        })
    
    def on_sequence_finished(self, sequence_number: int, total_results: int, successful_actions: int):
        self.events.append({
            'type': 'sequence_finished',
            'sequence_number': sequence_number
        })
    
    def on_evaluation(self, has_repetition: bool, pattern_detected: str, original_actions: int, corrected_actions: int):
        pass
    
    def on_processing_complete(self, total_sequences: int, total_actions: int):
        self.events.append({
            'type': 'processing_complete',
            'total_sequences': total_sequences,
            'total_actions': total_actions
        })


def test_trigger_streaming_events():
    """Test that proper trigger streaming events are emitted"""
    # Setup
    reasoning_loop = ActionBasedReasoningLoop()
    trigger_history = TriggerHistory()
    callback = MockStreamingCallback()
    state = create_default_agent_state()
    
    # Create a mock LLM that handles multiple calls properly
    mock_llm = Mock(spec=LLM)
    
    # Track calls to prevent infinite loop
    call_count = 0
    
    # Mock different structured LLM calls with side_effect
    def mock_generate_complete(model, prompt, **kwargs):
        nonlocal call_count
        call_count += 1
        
        # Action planner call (contains action planning prompt patterns)
        if "I am planning my next actions" in prompt or "situation_analysis" in prompt:
            if call_count == 1:
                # First call - return an action
                return """{
                    "situation_analysis": "The user is greeting me with a friendly question about my wellbeing",
                    "reasoning": "I should respond warmly and naturally to their greeting",
                    "actions": [
                        {
                            "action": "speak",
                            "context": "Respond warmly to the user's greeting"
                        }
                    ]
                }"""
            else:
                # Subsequent calls - return WAIT action to stop loop
                return """{
                    "situation_analysis": "I have completed my response",
                    "reasoning": "I should wait for the user's next input",
                    "actions": [
                        {
                            "action": "wait",
                            "context": "Waiting for user's next input"
                        }
                    ]
                }"""
        # Action evaluator call (contains evaluation prompt patterns)
        elif "repetitive patterns" in prompt or "has_repetition" in prompt:
            return """{
                "has_repetition": false,
                "pattern_detected": "",
                "corrected_actions": []
            }"""
        else:
            # Default fallback
            return "{}"
    
    mock_llm.generate_complete = Mock(side_effect=mock_generate_complete)
    
    # Mock streaming LLM call for SPEAK action (used by speak action)
    def mock_generate_streaming(model, prompt):
        # Check what type of action this is for
        if "wait" in prompt.lower():
            # WAIT action - just return a simple response
            yield "*waiting patiently*"
        else:
            # SPEAK action - simulate streaming text generation with multiple chunks
            response_chunks = [
                "Hello! ",
                "I'm doing ",
                "well, thank you ",
                "for asking."
            ]
            for chunk in response_chunks:
                yield chunk
    
    mock_llm.generate_streaming = Mock(side_effect=mock_generate_streaming)
    
    # Process user input
    reasoning_loop.process_user_input(
        user_input="Hello, how are you?",
        user_name="TestUser",
        state=state,
        llm=mock_llm,
        model="test-model",
        callback=callback,
        trigger_history=trigger_history
    )
    
    # Verify events were emitted in correct order
    event_types = [event['type'] for event in callback.events]
    
    # Should start with trigger_started
    assert event_types[0] == 'trigger_started'
    assert callback.events[0]['trigger_type'] == 'user_input'
    assert callback.events[0]['trigger_content'] == 'Hello, how are you?'
    
    # Should have entry_id in all relevant events
    entry_id = callback.events[0]['entry_id']
    assert entry_id is not None
    
    # All action events should have same entry_id
    action_events = [e for e in callback.events if e['type'].startswith('action_')]
    for event in action_events:
        assert event['entry_id'] == entry_id
    
    # Should have action_progress events from streaming (this tests actual streaming)
    progress_events = [e for e in callback.events if e['type'] == 'action_progress']
    assert len(progress_events) > 0, "Should have streaming progress events from SPEAK action"
    
    # Should end with processing_complete and trigger_completed
    assert 'processing_complete' in event_types
    assert 'trigger_completed' in event_types
    
    # trigger_completed should have correct entry_id
    trigger_completed = next(e for e in callback.events if e['type'] == 'trigger_completed')
    assert trigger_completed['entry_id'] == entry_id


def test_trigger_history_entry_creation():
    """Test that trigger history entries are created with consistent IDs"""
    reasoning_loop = ActionBasedReasoningLoop()
    trigger_history = TriggerHistory()
    callback = MockStreamingCallback()
    state = create_default_agent_state()
    
    # Create a mock LLM
    mock_llm = Mock(spec=LLM)
    
    # Track calls to prevent infinite loop
    call_count = 0
    
    # Mock different structured LLM calls
    def mock_generate_complete(model, prompt, **kwargs):
        nonlocal call_count
        call_count += 1
        
        if "I am planning my next actions" in prompt or "situation_analysis" in prompt:
            if call_count == 1:
                return """{
                    "situation_analysis": "I received a test input that needs a response",
                    "reasoning": "I should respond to the test input appropriately",
                    "actions": [
                        {
                            "action": "speak",
                            "context": "Test context"
                        }
                    ]
                }"""
            else:
                return """{
                    "situation_analysis": "I have completed my response",
                    "reasoning": "I should wait for the user's next input",
                    "actions": [
                        {
                            "action": "wait",
                            "context": "Waiting for user's next input"
                        }
                    ]
                }"""
        elif "repetitive patterns" in prompt or "has_repetition" in prompt:
            return """{
                "has_repetition": false,
                "pattern_detected": "",
                "corrected_actions": []
            }"""
        else:
            return "{}"
    
    mock_llm.generate_complete = Mock(side_effect=mock_generate_complete)
    
    # Mock streaming for action execution
    def mock_generate_streaming_test(model, prompt):
        if "wait" in prompt.lower():
            yield "*waiting*"
        else:
            yield "Test response"
    
    mock_llm.generate_streaming = Mock(side_effect=mock_generate_streaming_test)
    
    initial_count = len(trigger_history.entries)
    
    # Process user input
    reasoning_loop.process_user_input(
        user_input="Test input",
        user_name="TestUser", 
        state=state,
        llm=mock_llm,
        model="test-model",
        callback=callback,
        trigger_history=trigger_history
    )
    
    # Should have added one entry to trigger history
    assert len(trigger_history.entries) == initial_count + 1
    
    # The entry should have same ID as streaming events
    entry_id_from_streaming = callback.events[0]['entry_id']
    entry_id_from_history = trigger_history.entries[-1].entry_id
    
    assert entry_id_from_streaming == entry_id_from_history


def test_multiple_triggers_have_different_ids():
    """Test that multiple triggers get different entry IDs"""
    reasoning_loop = ActionBasedReasoningLoop()
    trigger_history = TriggerHistory()
    callback1 = MockStreamingCallback()
    callback2 = MockStreamingCallback()
    state = create_default_agent_state()
    
    # Create a mock LLM
    mock_llm = Mock(spec=LLM)
    
    # Track calls to prevent infinite loop  
    call_counts = [0, 0]  # Track for each trigger separately
    
    # Mock different structured LLM calls
    def mock_generate_complete(model, prompt, **kwargs):
        if "I am planning my next actions" in prompt or "situation_analysis" in prompt:
            # Determine which trigger this is for and increment its counter
            trigger_index = 0 if "First input" in prompt else 1 if "Second input" in prompt else 0
            call_counts[trigger_index] += 1
            
            if call_counts[trigger_index] == 1:
                return """{
                    "situation_analysis": "I received input that needs a response",
                    "reasoning": "I should respond appropriately",
                    "actions": [
                        {
                            "action": "speak",
                            "context": "Context"
                        }
                    ]
                }"""
            else:
                return """{
                    "situation_analysis": "I have completed my response",
                    "reasoning": "I should wait for the user's next input",
                    "actions": [
                        {
                            "action": "wait",
                            "context": "Waiting for user's next input"
                        }
                    ]
                }"""
        elif "repetitive patterns" in prompt or "has_repetition" in prompt:
            return """{
                "has_repetition": false,
                "pattern_detected": "",
                "corrected_actions": []
            }"""
        else:
            return "{}"
    
    mock_llm.generate_complete = Mock(side_effect=mock_generate_complete)
    
    # Mock streaming for action execution
    def mock_generate_streaming_multi(model, prompt):
        if "wait" in prompt.lower():
            yield "*waiting*"
        else:
            yield "Response"
    
    mock_llm.generate_streaming = Mock(side_effect=mock_generate_streaming_multi)
    
    # Process first trigger
    reasoning_loop.process_user_input(
        user_input="First input",
        user_name="User1",
        state=state,
        llm=mock_llm,
        model="test-model",
        callback=callback1,
        trigger_history=trigger_history
    )
    
    # Process second trigger
    reasoning_loop.process_user_input(
        user_input="Second input", 
        user_name="User2",
        state=state,
        llm=mock_llm,
        model="test-model",
        callback=callback2,
        trigger_history=trigger_history
    )
    
    # Should have different entry IDs
    entry_id_1 = callback1.events[0]['entry_id']
    entry_id_2 = callback2.events[0]['entry_id']
    
    assert entry_id_1 != entry_id_2
    assert len(trigger_history.entries) == 2