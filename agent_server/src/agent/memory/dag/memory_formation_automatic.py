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
from agent.memory.dag.models import MemoryElement, ConfidenceLevel


def create_memories_from_trigger_entry(
    trigger_entry: TriggerHistoryEntry,
) -> List[MemoryElement]:
    """
    Create all memories for a trigger entry.

    Uses batch encoding for embeddings (206x faster than individual encodes).

    Args:
        trigger_entry: The complete trigger entry with actions

    Returns:
        List of MemoryElements (trigger + non-WAIT actions)
    """
    # Collect all content first for batch encoding
    trigger_text = format_trigger_for_prompt(trigger_entry.trigger)
    contents = [trigger_text]

    # Collect action content and metadata
    action_data: List[tuple[ActionData, int, str]] = []
    for i, action in enumerate(trigger_entry.actions_taken):
        if action.type == ActionType.WAIT:
            continue
        content = format_action_for_diary(action)
        contents.append(content)
        action_data.append((action, i, content))

    # Single batch encode call (206x faster than individual encodes)
    embedding_service = get_embedding_service()
    embeddings = embedding_service.encode_batch(contents)

    # Create trigger memory with pre-computed embedding
    memories = [
        MemoryElement(
            id=str(uuid.uuid4()),
            content=trigger_text,
            timestamp=trigger_entry.trigger.timestamp,
            confidence_level=ConfidenceLevel.USER_CONFIRMED,
            sequence_in_container=0,
            container_id=trigger_entry.entry_id,
            embedding_vector=embeddings[0],
        )
    ]

    # Create action memories with pre-computed embeddings
    for idx, (action, sequence, content) in enumerate(action_data):
        memories.append(
            MemoryElement(
                id=str(uuid.uuid4()),
                content=content,
                timestamp=action.start_timestamp,
                confidence_level=ConfidenceLevel.STRONG_INFERENCE,
                sequence_in_container=sequence + 1,
                container_id=trigger_entry.entry_id,
                embedding_vector=embeddings[idx + 1],
            )
        )

    return memories
