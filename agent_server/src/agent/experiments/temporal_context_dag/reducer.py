"""
Reducer function for applying memory actions to mutable graph state.

The reducer applies concrete actions to existing MemoryGraph and ContextGraph objects,
mutating them directly for efficiency while maintaining replayability through action logs.
"""

import logging
from typing import assert_never

from .models import ContextElement, MemoryGraph, ContextGraph, ConfidenceLevel
from .actions import (
    ApplyTokenDecayAction,
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
    match action:
        case AddMemoryAction():
            _apply_add_memory(graph, action)
        case AddEdgeAction():
            _apply_add_connection(graph, action)
        case UpdateConfidenceAction():
            _apply_update_confidence(graph, action)
        case AddToContextAction():
            _apply_add_to_context(graph, context, action)
        case AddEdgeToContextAction():
            _apply_add_edge_to_context(graph, context, action)
        case RemoveFromContextAction():
            _apply_remove_from_context(context, action)
        case AddContainerAction():
            _apply_add_container(graph, action)
        case ApplyTokenDecayAction():
            _apply_token_decay(context, action)
        case CheckpointAction():
            _apply_checkpoint(action)
        case _:
            assert_never(action)


def _apply_add_memory(graph: MemoryGraph, action: AddMemoryAction) -> None:
    """Add a memory element to the graph."""
    graph.elements[action.memory.id] = action.memory
    logger.debug(f"Added memory {action.memory.id[:8]} to graph")


def _apply_add_connection(graph: MemoryGraph, action: AddEdgeAction) -> None:
    """Add a connection edge to the graph."""

    from .connection_system import update_confidence_for_correction_edge

    graph.edges[action.edge.id] = action.edge

    update_confidence_for_correction_edge(graph, action.edge)

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


def _apply_add_to_context(
    graph: MemoryGraph, context: ContextGraph, action: AddToContextAction
) -> None:
    """Add a memory element to the working context."""
    # Check if memory already exists in context
    memory_id = action.memory_id
    for existing_elem in context.elements:
        if existing_elem.memory.id == memory_id:
            if action.reinforce_tokens > 0:
                # Reinforce existing memory with additional tokens
                existing_elem.tokens += action.reinforce_tokens
                existing_elem.tokens = min(
                    existing_elem.tokens, 100
                )  # Cap at 100 tokens
                logger.debug(
                    f"Reinforced memory {memory_id[:8]} with {action.reinforce_tokens} tokens "
                    f"(now {existing_elem.tokens} total)"
                )
            else:
                logger.debug(
                    f"Memory {memory_id[:8]} already exists in context, skipping"
                )
            return

    context.elements.append(
        ContextElement(memory=graph.elements[memory_id], tokens=action.initial_tokens)
    )
    logger.debug(
        f"Added memory {context.elements[-1].memory.id[:8]} to context ({context.elements[-1].tokens} tokens)"
    )


def _apply_add_edge_to_context(
    graph: MemoryGraph, context: ContextGraph, action: AddEdgeToContextAction
) -> None:
    """Add an edge to the working context."""
    # Check if edge already exists in context
    edge_id = action.edge_id
    for existing_edge in context.edges:
        if existing_edge.id == edge_id:
            logger.debug(f"Edge {edge_id[:8]} already exists in context, skipping")
            return
    edge = graph.edges[edge_id]

    context.edges.append(edge)
    if action.should_boost_source_tokens:
        # Boost tokens of source memory in context if present
        for elem in context.elements:
            if elem.memory.id == edge.source_id:
                elem.tokens += 1  # Simple boost logic; can be adjusted
                logger.debug(
                    f"Boosted tokens of memory {elem.memory.id[:8]} to {elem.tokens} due to edge addition"
                )
                break
    logger.debug(
        f"Added edge {edge.edge_type.value} from {edge.source_id[:8]} to {edge.target_id[:8]} to context"
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


def _apply_token_decay(context: ContextGraph, action: ApplyTokenDecayAction) -> None:
    """Apply token decay to specific context memories."""
    for elem in context.elements:
        elem.tokens = max(1, elem.tokens - action.decay_amount)


def _apply_checkpoint(action: CheckpointAction) -> None:
    """Process a checkpoint action (mainly for logging)."""
    logger.info(f"Checkpoint '{action.label}': {action.description}")
