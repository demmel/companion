"""
Reducer function for applying memory actions to mutable graph state.

The reducer applies concrete actions to existing MemoryGraph and ContextGraph objects,
mutating them directly for efficiency while maintaining replayability through action logs.
"""

import logging
from typing import assert_never

from agent.chain_of_action.trigger_history import TriggerHistory
from .models import (
    ContextElement,
    MemoryContainer,
    MemoryGraph,
    ContextGraph,
    ConfidenceLevel,
)
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
    trigger_history: TriggerHistory,
    graph: MemoryGraph,
    context: ContextGraph,
    action: MemoryAction,
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
            logger.debug(f"Applying AddMemoryAction for memory {action.memory.id[:8]}")
            _apply_add_memory(graph, action)
        case AddEdgeAction():
            logger.debug(
                f"Applying AddEdgeAction for edge {action.edge.id[:8]} "
                f"({action.edge.source_id[:8]} --{action.edge.edge_type.value}--> {action.edge.target_id[:8]})"
            )
            _apply_add_connection(graph, action)
        case UpdateConfidenceAction():
            logger.debug(
                f"Applying UpdateConfidenceAction for memory {action.memory_id[:8]} "
                f"to {action.new_confidence} ({action.reason})"
            )
            _apply_update_confidence(graph, action)
        case AddToContextAction():
            logger.debug(
                f"Applying AddToContextAction for memory {action.memory_id[:8]} "
                f"({action.initial_tokens} initial tokens, {action.reinforce_tokens} reinforce tokens)"
            )
            _apply_add_to_context(graph, context, action)
        case AddEdgeToContextAction():
            logger.debug(
                f"Applying AddEdgeToContextAction for edge {action.edge_id[:8]} "
                f"(boost source tokens: {action.should_boost_source_tokens})"
            )
            _apply_add_edge_to_context(graph, context, action)
        case RemoveFromContextAction():
            logger.debug(
                f"Applying RemoveFromContextAction removing "
                f"{len(action.memory_ids)} memories and {len(action.edge_ids)} edges ({action.reason})"
            )
            _apply_remove_from_context(context, action)
        case AddContainerAction():
            logger.debug(
                f"Applying AddContainerAction for container {action.container_id} "
                f"with {len(action.element_ids)} memories"
            )
            _apply_add_container(trigger_history, graph, action)
        case ApplyTokenDecayAction():
            logger.debug(
                f"Applying ApplyTokenDecayAction with decay amount {action.decay_amount}"
            )
            _apply_token_decay(context, action)
        case CheckpointAction():
            logger.debug(f"Applying CheckpointAction '{action.label}'")
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

    # Check for duplicate edge signature (source_id, edge_type, target_id)
    edge_signature = (
        action.edge.source_id,
        action.edge.edge_type,
        action.edge.target_id,
    )

    # Check if this exact signature already exists in the graph
    for existing_edge in graph.edges.values():
        if (
            existing_edge.source_id,
            existing_edge.edge_type,
            existing_edge.target_id,
        ) == edge_signature:
            logger.debug(
                f"Skipping duplicate edge: {action.edge.source_id[:8]} --{action.edge.edge_type.value}--> {action.edge.target_id[:8]} (already exists as {existing_edge.id[:8]})"
            )
            return

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
            if action.reinforce_tokens > 0 and existing_elem.tokens <= 100:
                # Only reinforce memories that are at or below 100 tokens
                # For memories already over 100, skip reinforcement entirely
                existing_elem.tokens += action.reinforce_tokens
                existing_elem.tokens = min(
                    existing_elem.tokens, 100
                )  # Cap at 100 tokens only for memories that were <= 100
                logger.debug(
                    f"Reinforced memory {memory_id[:8]} with {action.reinforce_tokens} tokens "
                    f"(now {existing_elem.tokens} total)"
                )
            elif existing_elem.tokens > 100:
                logger.debug(
                    f"Memory {memory_id[:8]} has {existing_elem.tokens} tokens (>100), skipping reinforcement"
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


def _apply_add_container(
    trigger_history: TriggerHistory, graph: MemoryGraph, action: AddContainerAction
) -> None:
    """Add a memory container to the graph."""

    container = MemoryContainer(
        trigger=trigger_history.get_entry_by_id(action.container_id),
        element_ids=action.element_ids,
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
    logger.debug(f"Checkpoint '{action.label}': {action.description}")
