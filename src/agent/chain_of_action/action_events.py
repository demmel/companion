"""
Event system for chain_of_action streaming.
"""

from pydantic import BaseModel
from .action_types import ActionType
from .context import ActionResult


class ActionEvent(BaseModel):
    """Base class for all action events"""

    event_type: str


class ActionStartedEvent(ActionEvent):
    """Emitted when an action starts executing"""

    event_type: str = "action_started"
    action_type: ActionType
    context: str
    sequence_number: int
    action_number: int


class ActionFinishedEvent(ActionEvent):
    """Emitted when an action completes"""

    event_type: str = "action_finished"
    action_type: ActionType
    result: ActionResult
    sequence_number: int
    action_number: int


class SequenceStartedEvent(ActionEvent):
    """Emitted when an action sequence starts"""

    event_type: str = "sequence_started"
    sequence_number: int
    total_actions: int
    reasoning: str


class SequenceFinishedEvent(ActionEvent):
    """Emitted when an action sequence completes"""

    event_type: str = "sequence_finished"
    sequence_number: int
    total_results: int
    successful_actions: int


class EvaluationEvent(ActionEvent):
    """Emitted when action evaluation detects repetition"""

    event_type: str = "evaluation"
    has_repetition: bool
    pattern_detected: str
    original_actions: int
    corrected_actions: int


class ProcessingCompleteEvent(ActionEvent):
    """Emitted when all processing is complete"""

    event_type: str = "processing_complete"
    total_sequences: int
    total_actions: int


class SpeakProgressData(BaseModel):
    """Progress data for SPEAK action streaming"""

    text: str
    is_partial: bool


class ThinkProgressData(BaseModel):
    """Progress data for THINK action streaming"""

    text: str
    is_partial: bool
