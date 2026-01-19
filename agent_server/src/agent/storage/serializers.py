"""
Serializers for converting between Pydantic models and SQLite storage.

Handles compression, polymorphic type handling, and binary encoding.
"""

import zlib
from typing import TypeVar

from pydantic import BaseModel, TypeAdapter

from agent.chain_of_action.trigger import Trigger
from agent.chain_of_action.action.action_data import ActionData
from agent.chain_of_action.action.action_types import ActionType
from agent.storage.models import ActionTable


T = TypeVar("T", bound=BaseModel)


# Type adapters for polymorphic deserialization
_TRIGGER_ADAPTER: TypeAdapter[Trigger] = TypeAdapter(Trigger)
_ACTION_DATA_ADAPTER: TypeAdapter[ActionData] = TypeAdapter(ActionData)


def compress_json(obj: BaseModel) -> bytes:
    """
    Compress Pydantic model to zlib bytes.

    Uses compression level 6 (good balance of speed and compression).
    Typically achieves 60-70% compression on JSON data.
    """
    json_bytes = obj.model_dump_json().encode("utf-8")
    return zlib.compress(json_bytes, level=6)


def decompress_json(data: bytes, model_class: type[T]) -> T:
    """Decompress zlib bytes and validate to Pydantic model."""
    json_bytes = zlib.decompress(data)
    return model_class.model_validate_json(json_bytes)


def compress_string(s: str) -> bytes:
    """Compress a string to zlib bytes."""
    return zlib.compress(s.encode("utf-8"), level=6)


def decompress_string(data: bytes) -> str:
    """Decompress zlib bytes to string."""
    return zlib.decompress(data).decode("utf-8")


class TriggerSerializer:
    """Handles serialization of polymorphic Trigger types."""

    @staticmethod
    def get_type(trigger: Trigger) -> str:
        """Extract trigger type string for indexing."""
        return trigger.type

    @staticmethod
    def to_blob(trigger: Trigger) -> bytes:
        """Compress trigger to zlib bytes."""
        return compress_json(trigger)

    @staticmethod
    def from_blob(trigger_blob: bytes) -> Trigger:
        """Reconstruct typed trigger from compressed blob."""
        json_bytes = zlib.decompress(trigger_blob)
        return _TRIGGER_ADAPTER.validate_json(json_bytes)


class ActionSerializer:
    """
    Handles serialization of polymorphic ActionData types.

    Stores the full action as a compressed JSON blob and uses Pydantic's
    discriminated union (via the 'type' field) for deserialization.
    """

    @staticmethod
    def to_row(
        action: ActionData,
        trigger_entry_id: str,
        sequence: int,
    ) -> ActionTable:
        """
        Convert ActionData to ActionTable row.

        Stores key fields for indexing and the full action as compressed blob.
        """
        # Compress the full action JSON for storage
        action_blob = compress_json(action)

        # Determine result type for filtering
        result_type = action.result.type  # "success" or "failure"

        return ActionTable(
            trigger_entry_id=trigger_entry_id,
            sequence_order=sequence,
            action_type=action.type.value,
            start_timestamp=action.start_timestamp,
            duration_ms=action.duration_ms,
            reasoning=action.reasoning,
            input_blob=action_blob,  # Store full action here
            result_type=result_type,
            result_blob=b"",  # Not used - full action is in input_blob
        )

    @staticmethod
    def from_row(row: ActionTable) -> ActionData:
        """
        Reconstruct ActionData from ActionTable row.

        Uses Pydantic's discriminated union via the type adapter.
        """
        # The full action is stored in input_blob
        json_bytes = zlib.decompress(row.input_blob)
        return _ACTION_DATA_ADAPTER.validate_json(json_bytes)


# Note: Embeddings are stored in ChromaDB, not SQLite.
# ChromaDB handles vector serialization internally.
