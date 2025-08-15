"""
Conversation persistence system with unique IDs and auto-save functionality
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass

from .types import ConversationData, Message
from .state import State, create_default_agent_state
from .chain_of_action.trigger_history import (
    TriggerHistory,
    TriggerHistoryEntry,
    SummaryRecord,
)
from pydantic import BaseModel
from typing import List


class TriggerHistoryData(BaseModel):
    """Serializable trigger history data for persistence"""

    entries: List[TriggerHistoryEntry]
    summaries: List[SummaryRecord]


@dataclass
class ConversationMetadata:
    """Metadata for a saved conversation"""

    conversation_id: str
    created_at: str  # ISO format timestamp
    last_updated: str  # ISO format timestamp
    message_count: int
    title: Optional[str] = None  # Optional user-provided title


class ConversationPersistence:
    """Manages conversation persistence with unique IDs"""

    def __init__(self, conversations_dir: str = "conversations"):
        self.conversations_dir = Path(conversations_dir)
        self.conversations_dir.mkdir(exist_ok=True)

        # Create metadata index file if it doesn't exist
        self.index_file = self.conversations_dir / "index.json"
        if not self.index_file.exists():
            self._save_index({})

    def generate_conversation_id(self) -> str:
        """Generate a unique conversation ID based on timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Add microseconds for uniqueness if multiple conversations start at same second
        microseconds = int(time.time() * 1000000) % 1000000
        return f"conversation_{timestamp}_{microseconds:06d}"

    def save_conversation(
        self,
        conversation_id: str,
        state: State,
        trigger_history: TriggerHistory,
        title: Optional[str] = None,
        save_baseline: bool = True,
    ) -> None:
        """Save a conversation with its state and optional trigger history"""

        self._save_state_and_triggers(conversation_id, state, trigger_history)
        if save_baseline:
            self._save_state_and_triggers("baseline", state, trigger_history)

        # Update metadata index
        self._update_metadata(conversation_id, len(trigger_history.entries), title)

    def _save_state_and_triggers(
        self, prefix: str, state: State, trigger_history: TriggerHistory
    ) -> None:
        """Save the state and trigger history for a conversation"""
        state_file = self.conversations_dir / f"{prefix}_state.json"
        with open(state_file, "w") as f:
            f.write(state.model_dump_json(indent=2))

        trigger_file = self.conversations_dir / f"{prefix}_triggers.json"
        trigger_data = TriggerHistoryData(
            entries=trigger_history.entries, summaries=trigger_history.summaries
        )
        with open(trigger_file, "w") as f:
            f.write(trigger_data.model_dump_json(indent=2))

    def load_conversation(self, conversation_id: str) -> tuple[ConversationData, State]:
        """Load a conversation and its state"""
        # Load conversation data
        conversation_file = self.conversations_dir / f"{conversation_id}.json"
        if not conversation_file.exists():
            raise FileNotFoundError(f"Conversation {conversation_id} not found")

        with open(conversation_file, "r") as f:
            conversation_data = ConversationData.model_validate(json.load(f))

        # Load the agent's state
        state_file = self.conversations_dir / f"{conversation_id}_state.json"
        if state_file.exists():
            with open(state_file, "r") as f:
                state = State.model_validate(json.load(f))
        else:
            # Fallback to default state if no state file exists
            state = create_default_agent_state()

        return conversation_data, state

    def list_conversations(self) -> List[ConversationMetadata]:
        """List all saved conversations with metadata"""
        index = self._load_index()
        conversations = []

        for conv_id, metadata in index.items():
            conversations.append(
                ConversationMetadata(
                    conversation_id=conv_id,
                    created_at=metadata.get("created_at", ""),
                    last_updated=metadata.get("last_updated", ""),
                    message_count=metadata.get("message_count", 0),
                    title=metadata.get("title"),
                )
            )

        # Sort by last_updated (most recent first)
        conversations.sort(key=lambda x: x.last_updated, reverse=True)
        return conversations

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation and its associated files"""
        conversation_file = self.conversations_dir / f"{conversation_id}.json"
        state_file = self.conversations_dir / f"{conversation_id}_state.json"

        deleted = False
        if conversation_file.exists():
            conversation_file.unlink()
            deleted = True

        if state_file.exists():
            state_file.unlink()

        # Remove from index
        index = self._load_index()
        if conversation_id in index:
            del index[conversation_id]
            self._save_index(index)

        return deleted

    def conversation_exists(self, conversation_id: str) -> bool:
        """Check if a conversation exists"""
        conversation_file = self.conversations_dir / f"{conversation_id}.json"
        return conversation_file.exists()

    def _update_metadata(
        self, conversation_id: str, message_count: int, title: Optional[str]
    ) -> None:
        """Update metadata for a conversation"""
        index = self._load_index()
        now = datetime.now().isoformat()

        if conversation_id not in index:
            # New conversation
            index[conversation_id] = {
                "created_at": now,
                "last_updated": now,
                "message_count": message_count,
                "title": title,
            }
        else:
            # Update existing conversation
            index[conversation_id]["last_updated"] = now
            index[conversation_id]["message_count"] = message_count
            if title is not None:
                index[conversation_id]["title"] = title

        self._save_index(index)

    def _load_index(self) -> dict:
        """Load the conversation index"""
        if not self.index_file.exists():
            return {}

        with open(self.index_file, "r") as f:
            return json.load(f)

    def _save_index(self, index: dict) -> None:
        """Save the conversation index"""
        with open(self.index_file, "w") as f:
            json.dump(index, f, indent=2)
