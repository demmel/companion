"""
Single-step action processor for DAG memory visualization.

Processes action logs one step at a time to enable step-by-step graph evolution
visualization, tracking changes between each action.
"""

import logging
from typing import List, Set, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from agent.chain_of_action.trigger_history import TriggerHistory, TriggerHistoryEntry
from agent.chain_of_action.trigger import WakeupTrigger
from ..models import MemoryGraph, ContextGraph
from ..actions import MemoryAction, CheckpointAction
from ..action_log import MemoryActionLog
from ..reducer import apply_action

logger = logging.getLogger(__name__)


@dataclass
class GraphState:
    """Snapshot of graph and context at a specific point in time."""

    step_index: int
    action: MemoryAction
    memory_graph: MemoryGraph
    context_graph: ContextGraph
    timestamp: datetime

    # Delta information for highlighting changes
    added_memories: Set[str]
    removed_memories: Set[str]
    modified_memories: Set[str]
    added_edges: Set[str]
    removed_edges: Set[str]
    added_to_context: Set[str]
    removed_from_context: Set[str]


@dataclass
class StepInfo:
    """Information about a specific step for UI display."""

    step_index: int
    action_type: str
    action_id: str
    timestamp: datetime
    description: str
    is_checkpoint: bool
    checkpoint_label: Optional[str] = None


class StepwiseGraphReconstructor:
    """Reconstructs memory graph evolution step-by-step from action logs."""

    def __init__(self):
        self.action_log: Optional[MemoryActionLog] = None
        self.graph_states: List[GraphState] = []
        self.current_step: int = -1

    def load(self, file_path: str) -> None:
        """Load action log from JSON file."""
        action_log = MemoryActionLog.load_from_file(file_path)
        trigger_history = TriggerHistory()
        add_container_actions = [
            action.container_id
            for action in action_log.actions
            if action.action_type == "add_container"
        ]
        for container in add_container_actions:
            trigger_history.entries.append(
                TriggerHistoryEntry(
                    entry_id=container, actions_taken=[], trigger=WakeupTrigger()
                )
            )
        self.action_log = action_log
        self.trigger_history = trigger_history
        self._process_all_steps()
        logger.info(f"Loaded {len(self.graph_states)} graph states from {file_path}")

    def _process_all_steps(self) -> None:
        """Process all actions to generate step-by-step graph states."""
        if not self.action_log:
            return

        memory_graph = MemoryGraph()
        context_graph = ContextGraph(elements=[], edges=[])

        self.graph_states = []

        for i, action in enumerate(self.action_log.actions):
            # Store previous state for delta calculation
            prev_memory_ids = set(memory_graph.elements.keys())
            prev_edge_ids = set(memory_graph.edges.keys())
            prev_context_ids = {elem.memory.id for elem in context_graph.elements}

            # Apply the action
            apply_action(self.trigger_history, memory_graph, context_graph, action)

            # Calculate deltas
            current_memory_ids = set(memory_graph.elements.keys())
            current_edge_ids = set(memory_graph.edges.keys())
            current_context_ids = {elem.memory.id for elem in context_graph.elements}

            added_memories = current_memory_ids - prev_memory_ids
            removed_memories = prev_memory_ids - current_memory_ids
            added_edges = current_edge_ids - prev_edge_ids
            removed_edges = prev_edge_ids - current_edge_ids
            added_to_context = current_context_ids - prev_context_ids
            removed_from_context = prev_context_ids - current_context_ids

            # Find modified memories (confidence updates, etc.)
            modified_memories = set()
            for mem_id in current_memory_ids & prev_memory_ids:
                # This is a simplification - in practice we might want to deep compare
                # For now, assume no in-place modifications beyond confidence updates
                pass

            # Create deep copies for state snapshot
            state = GraphState(
                step_index=i,
                action=action,
                memory_graph=MemoryGraph(
                    elements=dict(memory_graph.elements),
                    containers=dict(memory_graph.containers),
                    edges=dict(memory_graph.edges),
                ),
                context_graph=ContextGraph(
                    elements=list(context_graph.elements),
                    edges=list(context_graph.edges),
                ),
                timestamp=action.timestamp,
                added_memories=added_memories,
                removed_memories=removed_memories,
                modified_memories=modified_memories,
                added_edges=added_edges,
                removed_edges=removed_edges,
                added_to_context=added_to_context,
                removed_from_context=removed_from_context,
            )

            self.graph_states.append(state)

    def get_step_count(self) -> int:
        """Get total number of steps available."""
        return len(self.graph_states)

    def get_current_step(self) -> int:
        """Get current step index."""
        return self.current_step

    def set_step(self, step_index: int) -> Optional[GraphState]:
        """Set current step and return the graph state."""
        if 0 <= step_index < len(self.graph_states):
            self.current_step = step_index
            return self.graph_states[step_index]
        return None

    def next_step(self) -> Optional[GraphState]:
        """Advance to next step."""
        if self.current_step < len(self.graph_states) - 1:
            self.current_step += 1
            return self.graph_states[self.current_step]
        return None

    def prev_step(self) -> Optional[GraphState]:
        """Go back to previous step."""
        if self.current_step > 0:
            self.current_step -= 1
            return self.graph_states[self.current_step]
        return None

    def get_current_state(self) -> Optional[GraphState]:
        """Get current graph state."""
        if 0 <= self.current_step < len(self.graph_states):
            return self.graph_states[self.current_step]
        return None

    def get_checkpoints(self) -> List[Tuple[int, CheckpointAction]]:
        """Get list of checkpoint steps with their actions."""
        checkpoints = []
        for i, state in enumerate(self.graph_states):
            if isinstance(state.action, CheckpointAction):
                checkpoints.append((i, state.action))
        return checkpoints

    def jump_to_checkpoint(self, checkpoint_label: str) -> Optional[GraphState]:
        """Jump to a specific checkpoint by label."""
        for i, state in enumerate(self.graph_states):
            if (
                isinstance(state.action, CheckpointAction)
                and state.action.label == checkpoint_label
            ):
                self.current_step = i
                return state
        return None

    def get_step_info(self, step_index: int) -> Optional[StepInfo]:
        """Get display information for a specific step."""
        if 0 <= step_index < len(self.graph_states):
            state = self.graph_states[step_index]
            action = state.action

            is_checkpoint = isinstance(action, CheckpointAction)
            checkpoint_label = action.label if is_checkpoint else None

            # Generate description based on action type
            description = self._generate_action_description(action)

            return StepInfo(
                step_index=step_index,
                action_type=action.action_type,
                action_id=action.id,
                timestamp=action.timestamp,
                description=description,
                is_checkpoint=is_checkpoint,
                checkpoint_label=checkpoint_label,
            )
        return None

    def _generate_action_description(self, action: MemoryAction) -> str:
        """Generate human-readable description of an action."""
        match action.action_type:
            case "add_memory":
                preview = (
                    action.memory.content[:50] + "..."
                    if len(action.memory.content) > 50
                    else action.memory.content
                )
                return f"Added memory: {preview}"
            case "add_connection":
                return f"Added {action.edge.edge_type} connection"
            case "add_to_context":
                return f"Added memory to context ({action.initial_tokens} tokens)"
            case "add_edge_to_context":
                return f"Added edge to context"
            case "remove_from_context":
                return f"Removed {len(action.memory_ids)} memories from context: {action.reason}"
            case "update_confidence":
                return f"Updated confidence to {action.new_confidence}: {action.reason}"
            case "add_container":
                return f"Added container with {len(action.element_ids)} elements"
            case "apply_token_decay":
                return f"Applied token decay (-{action.decay_amount})"
            case "checkpoint":
                return f"Checkpoint: {action.description}"
            case _:
                return f"Action: {action.action_type}"
