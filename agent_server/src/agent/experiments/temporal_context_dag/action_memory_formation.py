"""
Action-based memory formation that emits actions instead of directly mutating state.
"""

import logging
from typing import List

from agent.chain_of_action.trigger_history import TriggerHistoryEntry
from agent.llm import LLM, SupportedModel
from agent.state import State

from .models import ContextGraph
from .actions import (
    MemoryAction,
    AddMemoryAction,
    AddEdgeAction,
    AddContainerAction,
    AddToContextAction,
    AddEdgeToContextAction,
)
from .memory_formation import extract_memories_from_interaction

logger = logging.getLogger(__name__)


def extract_memories_as_actions(
    trigger: TriggerHistoryEntry,
    state: State,
    context: ContextGraph,
    llm: LLM,
    model: SupportedModel,
) -> List[MemoryAction]:
    """
    Extract memories from an interaction and return them as actions.

    Uses the existing memory extraction logic but returns actions instead of
    directly modifying the graph state.

    Args:
        trigger: The trigger history entry containing trigger and actions
        state: Current agent state
        context: Current context graph with existing memories
        llm: LLM instance for memory extraction
        model: Model to use for extraction decisions

    Returns:
        List of actions to apply for this interaction
    """
    # Use existing extraction logic to get memories and connections
    memories, connections = extract_memories_from_interaction(
        trigger, state, context, llm, model
    )

    actions = []

    # Create actions for adding memories to the graph
    for memory in memories:
        actions.append(AddMemoryAction(memory=memory.memory))

    # Create actions for adding memories to context
    for memory in memories:
        actions.append(AddToContextAction(context_element=memory))

    # Create actions for adding connections to graph
    for connection in connections:
        actions.append(AddEdgeAction(edge=connection))

    # Create actions for adding connections to context
    for connection in connections:
        actions.append(AddEdgeToContextAction(edge=connection))

    # Create action for adding the container
    if memories:
        actions.append(
            AddContainerAction(
                container_id=trigger.entry_id,
                element_ids=[m.memory.id for m in memories],
                trigger_timestamp=trigger.timestamp,
            )
        )

    logger.info(
        f"Generated {len(actions)} actions for interaction {trigger.entry_id}: "
        f"{len(memories)} memories, {len(connections)} connections"
    )

    return actions
