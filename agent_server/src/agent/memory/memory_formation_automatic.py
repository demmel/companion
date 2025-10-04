"""
Automatic memory formation from triggers and actions.

Creates memories directly from structured data without LLM paraphrasing.
"""

import uuid
from typing import List

from agent.chain_of_action.action.action_data import ActionData
from agent.chain_of_action.action.action_types import ActionType
from agent.chain_of_action.prompts import format_action_for_diary
from agent.chain_of_action.trigger import (
    Trigger,
    format_trigger_for_prompt,
)
from agent.chain_of_action.trigger_history import TriggerHistoryEntry
from agent.embedding_service import get_embedding_service
from agent.memory.models import MemoryElement, ConfidenceLevel


def create_trigger_memory(trigger: Trigger, entry_id: str) -> MemoryElement:
    """
    Create memory from trigger - no LLM needed.

    Args:
        trigger: The trigger event
        entry_id: ID of the TriggerHistoryEntry container

    Returns:
        MemoryElement with trigger content
    """
    trigger_text = format_trigger_for_prompt(trigger)

    # Both user input and wakeup triggers are confirmed events
    confidence = ConfidenceLevel.USER_CONFIRMED

    embedding_service = get_embedding_service()

    return MemoryElement(
        id=str(uuid.uuid4()),
        content=trigger_text,
        timestamp=trigger.timestamp,
        confidence_level=confidence,
        sequence_in_container=0,
        container_id=entry_id,
        embedding_vector=embedding_service.encode(trigger_text),
    )


def create_action_memory(
    action: ActionData, sequence: int, entry_id: str
) -> MemoryElement:
    """
    Create memory from action - no LLM needed.

    Args:
        action: The action data
        sequence: Sequence number within container
        entry_id: ID of the TriggerHistoryEntry container

    Returns:
        MemoryElement with action content formatted as diary entry
    """
    # Use existing diary formatting which includes reasoning and result
    content = format_action_for_diary(action)

    embedding_service = get_embedding_service()

    return MemoryElement(
        id=str(uuid.uuid4()),
        content=content,
        timestamp=action.start_timestamp,
        confidence_level=ConfidenceLevel.STRONG_INFERENCE,
        sequence_in_container=sequence + 1,  # +1 because trigger is at sequence 0
        container_id=entry_id,
        embedding_vector=embedding_service.encode(content),
    )


def create_memories_from_trigger_entry(
    trigger_entry: TriggerHistoryEntry,
) -> List[MemoryElement]:
    """
    Create all memories for a trigger entry.

    Args:
        trigger_entry: The complete trigger entry with actions

    Returns:
        List of MemoryElements (trigger + non-WAIT actions)
    """
    memories = []

    # Trigger memory (always sequence 0)
    memories.append(
        create_trigger_memory(trigger_entry.trigger, trigger_entry.entry_id)
    )

    # Action memories (skip WAIT)
    for i, action in enumerate(trigger_entry.actions_taken):
        if action.type == ActionType.WAIT:
            continue
        memories.append(
            create_action_memory(action, sequence=i, entry_id=trigger_entry.entry_id)
        )

    return memories
