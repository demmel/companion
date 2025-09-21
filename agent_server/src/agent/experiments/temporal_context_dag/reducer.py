"""
Reducer function for applying memory actions to mutable graph state.

The reducer applies concrete actions to existing MemoryGraph and ContextGraph objects,
mutating them directly for efficiency while maintaining replayability through action logs.
"""

import logging
from typing import Union

from .models import MemoryGraph, ContextGraph, ContextElement, ConfidenceLevel
from .actions import (
    MemoryAction,
    AddMemoryAction,
    AddEdgeAction,
    UpdateConfidenceAction,
    AddToContextAction,
    AddEdgeToContextAction,
    RemoveFromContextAction,
    AddContainerAction,
    CheckpointAction,
)

logger = logging.getLogger(__name__)


def apply_action(
    graph: MemoryGraph, context: ContextGraph, action: MemoryAction
) -> None:
    """
    Apply a memory action to the graph and context state.

    Mutates the graph and context objects directly based on the concrete
    results stored in the action.

    Args:
        graph: Memory graph to mutate
        context: Context graph to mutate
        action: Action containing concrete changes to apply
    """
    if isinstance(action, AddMemoryAction):
        _apply_add_memory(graph, action)
    elif isinstance(action, AddEdgeAction):
        _apply_add_connection(graph, action)
    elif isinstance(action, UpdateConfidenceAction):
        _apply_update_confidence(graph, action)
    elif isinstance(action, AddToContextAction):
        _apply_add_to_context(context, action)
    elif isinstance(action, AddEdgeToContextAction):
        _apply_add_edge_to_context(context, action)
    elif isinstance(action, RemoveFromContextAction):
        _apply_remove_from_context(context, action)
    elif isinstance(action, AddContainerAction):
        _apply_add_container(graph, action)
    elif isinstance(action, CheckpointAction):
        _apply_checkpoint(action)
    else:
        logger.warning(f"Unknown action type: {type(action)}")


def _apply_add_memory(graph: MemoryGraph, action: AddMemoryAction) -> None:
    """Add a memory element to the graph."""
    graph.elements[action.memory.id] = action.memory
    logger.debug(f"Added memory {action.memory.id[:8]} to graph")


def _apply_add_connection(graph: MemoryGraph, action: AddEdgeAction) -> None:
    """Add a connection edge to the graph."""
    graph.edges[action.edge.id] = action.edge
    logger.debug(
        f"Added edge {action.edge.edge_type.value} from {action.edge.source_id[:8]} to {action.edge.target_id[:8]}"
    )


def _apply_update_confidence(
    graph: MemoryGraph, action: UpdateConfidenceAction
) -> None:
    """Update the confidence level of a memory."""
    memory = graph.elements.get(action.memory_id)
    if memory:
        old_confidence = memory.confidence_level.value
        memory.confidence_level = ConfidenceLevel(action.new_confidence)
        logger.debug(
            f"Updated confidence of memory {action.memory_id[:8]} from {old_confidence} to {action.new_confidence} ({action.reason})"
        )
    else:
        logger.warning(f"Memory {action.memory_id} not found for confidence update")


def _apply_add_to_context(context: ContextGraph, action: AddToContextAction) -> None:
    """Add a memory element to the working context."""
    context.elements.append(action.context_element)
    logger.debug(
        f"Added memory {action.context_element.memory.id[:8]} to context ({action.context_element.tokens} tokens)"
    )


def _apply_add_edge_to_context(
    context: ContextGraph, action: AddEdgeToContextAction
) -> None:
    """Add an edge to the working context."""
    context.edges.append(action.edge)
    logger.debug(
        f"Added edge {action.edge.edge_type.value} from {action.edge.source_id[:8]} to {action.edge.target_id[:8]} to context"
    )


def _apply_remove_from_context(
    context: ContextGraph, action: RemoveFromContextAction
) -> None:
    """Remove specific memories and edges from the working context."""
    # Remove elements with matching memory IDs
    context.elements = [
        elem for elem in context.elements if elem.memory.id not in action.memory_ids
    ]

    # Remove edges with matching edge IDs
    context.edges = [edge for edge in context.edges if edge.id not in action.edge_ids]

    logger.debug(
        f"Removed {len(action.memory_ids)} memories and {len(action.edge_ids)} edges from context ({action.reason})"
    )


def _apply_add_container(graph: MemoryGraph, action: AddContainerAction) -> None:
    """Add a memory container to the graph."""
    from .memory_formation import create_memory_container
    from agent.chain_of_action.trigger_history import TriggerHistoryEntry
    from agent.chain_of_action.trigger import UserInputTrigger

    # Create a mock trigger for the container
    # In real usage, this would come from the original trigger
    mock_trigger = TriggerHistoryEntry(
        trigger=UserInputTrigger(content="", user_name=""),
        actions_taken=[],
        timestamp=action.trigger_timestamp,
        entry_id=action.container_id,
    )

    container = create_memory_container(
        trigger=mock_trigger, element_ids=action.element_ids
    )
    graph.containers[action.container_id] = container
    logger.debug(
        f"Added container {action.container_id} with {len(action.element_ids)} memories"
    )


def _apply_checkpoint(action: CheckpointAction) -> None:
    """Process a checkpoint action (mainly for logging)."""
    logger.info(f"Checkpoint '{action.label}': {action.description}")
