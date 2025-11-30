"""
Action log for recording and replaying memory graph operations.

Provides checkpoint-based replay functionality for observability and time-travel debugging.
"""

import json
import logging
from pathlib import Path
from typing import List, Tuple, Optional

from pydantic import BaseModel

from agent.chain_of_action.trigger_history import TriggerHistory
from .models import MemoryGraph, ContextGraph
from .actions import MemoryAction, CheckpointAction
from .reducer import apply_action

logger = logging.getLogger(__name__)


class MemoryActionLog(BaseModel):
    """Log of memory actions with checkpoint-based replay functionality."""

    actions: List[MemoryAction] = []

    def add_action(self, action: MemoryAction) -> None:
        """Add an action to the log."""
        self.actions.append(action)
        logger.debug(f"Added action {action.action_type} ({action.id[:8]})")

    def add_checkpoint(self, label: str, description: str) -> CheckpointAction:
        """Add a checkpoint action."""
        checkpoint = CheckpointAction(
            label=label,
            description=description,
        )
        self.add_action(checkpoint)
        return checkpoint

    def get_checkpoints(self) -> List[CheckpointAction]:
        """Get all checkpoint actions in the log."""
        return [
            action for action in self.actions if isinstance(action, CheckpointAction)
        ]

    def get_checkpoint_labels(self) -> List[str]:
        """Get all checkpoint labels in chronological order."""
        return [checkpoint.label for checkpoint in self.get_checkpoints()]

    def find_checkpoint_index(self, label: str) -> Optional[int]:
        """Find the index of a checkpoint action by label."""
        for i, action in enumerate(self.actions):
            if isinstance(action, CheckpointAction) and action.label == label:
                return i
        return None

    def get_actions_between_checkpoints(
        self, start_label: str, end_label: str
    ) -> List[MemoryAction]:
        """Get all actions between two checkpoints (exclusive of checkpoints)."""
        start_idx = self.find_checkpoint_index(start_label)
        end_idx = self.find_checkpoint_index(end_label)

        if start_idx is None:
            raise ValueError(f"Start checkpoint '{start_label}' not found")
        if end_idx is None:
            raise ValueError(f"End checkpoint '{end_label}' not found")
        if start_idx >= end_idx:
            raise ValueError("Start checkpoint must come before end checkpoint")

        return self.actions[start_idx + 1 : end_idx]

    def get_actions_since_checkpoint(self, label: str) -> List[MemoryAction]:
        """Get all actions since a checkpoint (exclusive of the checkpoint)."""
        checkpoint_idx = self.find_checkpoint_index(label)
        if checkpoint_idx is None:
            raise ValueError(f"Checkpoint '{label}' not found")

        return self.actions[checkpoint_idx + 1 :]

    def replay_from_empty(
        self, trigger_history: TriggerHistory
    ) -> Tuple[MemoryGraph, ContextGraph]:
        """Replay all actions from empty state to reconstruct current state."""
        return self.replay_actions(trigger_history, self.actions)

    def replay_to_checkpoint(
        self, trigger_history: TriggerHistory, label: str
    ) -> Tuple[MemoryGraph, ContextGraph]:
        """Replay actions up to and including a specific checkpoint."""
        checkpoint_idx = self.find_checkpoint_index(label)
        if checkpoint_idx is None:
            raise ValueError(f"Checkpoint '{label}' not found")

        actions_to_replay = self.actions[: checkpoint_idx + 1]
        return self.replay_actions(trigger_history, actions_to_replay)

    def replay_actions(
        self, trigger_history: TriggerHistory, actions: List[MemoryAction]
    ) -> Tuple[MemoryGraph, ContextGraph]:
        """Replay a specific list of actions from empty state."""
        graph = MemoryGraph()
        context = ContextGraph(elements=[], edges=[])

        for action in actions:
            apply_action(trigger_history, graph, context, action)

        logger.info(
            f"Replayed {len(actions)} actions - Graph: {len(graph.elements)} memories, "
            f"{len(graph.edges)} edges, Context: {len(context.elements)} elements"
        )

        return graph, context

    def save_to_file(self, filepath: str) -> None:
        """Save the action log to a JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            f.write(self.model_dump_json(indent=2))

        logger.info(f"Saved action log with {len(self.actions)} actions to {filepath}")

    @classmethod
    def load_from_file(cls, filepath: str) -> "MemoryActionLog":
        """Load an action log from a JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)

        action_log = cls.model_validate(data)
        logger.info(
            f"Loaded action log with {len(action_log.actions)} actions from {filepath}"
        )

        return action_log
