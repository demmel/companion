"""
Callback system for chain_of_action streaming.
"""

from typing import Protocol, Optional, Any
from .action_types import ActionType
from .context import ActionResult


class ActionCallback(Protocol):
    """Protocol for action callbacks"""

    def on_sequence_started(
        self, sequence_number: int, total_actions: int, reasoning: str
    ) -> None:
        """Called when an action sequence starts"""
        pass

    def on_action_started(
        self,
        action_type: ActionType,
        context: str,
        sequence_number: int,
        action_number: int,
    ) -> None:
        """Called when an action starts executing"""
        pass

    def on_action_progress(
        self,
        action_type: ActionType,
        progress_data: Any,
        sequence_number: int,
        action_number: int,
    ) -> None:
        """Called during action execution for streaming progress"""
        pass

    def on_action_finished(
        self,
        action_type: ActionType,
        result: ActionResult,
        sequence_number: int,
        action_number: int,
    ) -> None:
        """Called when an action completes"""
        pass

    def on_sequence_finished(
        self, sequence_number: int, total_results: int, successful_actions: int
    ) -> None:
        """Called when an action sequence completes"""
        pass

    def on_evaluation(
        self,
        has_repetition: bool,
        pattern_detected: str,
        original_actions: int,
        corrected_actions: int,
    ) -> None:
        """Called when action evaluation detects repetition"""
        pass

    def on_processing_complete(self, total_sequences: int, total_actions: int) -> None:
        """Called when all processing is complete"""
        pass


class NoOpCallback:
    """Default no-op callback for when streaming is not needed"""

    def on_sequence_started(
        self, sequence_number: int, total_actions: int, reasoning: str
    ) -> None:
        pass

    def on_action_started(
        self,
        action_type: ActionType,
        context: str,
        sequence_number: int,
        action_number: int,
    ) -> None:
        pass

    def on_action_progress(
        self,
        action_type: ActionType,
        progress_data: Any,
        sequence_number: int,
        action_number: int,
    ) -> None:
        pass

    def on_action_finished(
        self,
        action_type: ActionType,
        result: ActionResult,
        sequence_number: int,
        action_number: int,
    ) -> None:
        pass

    def on_sequence_finished(
        self, sequence_number: int, total_results: int, successful_actions: int
    ) -> None:
        pass

    def on_evaluation(
        self,
        has_repetition: bool,
        pattern_detected: str,
        original_actions: int,
        corrected_actions: int,
    ) -> None:
        pass

    def on_processing_complete(self, total_sequences: int, total_actions: int) -> None:
        pass
