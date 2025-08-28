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


class ConversationPersistence:
    """Manages conversation persistence with unique IDs"""

    def __init__(self, conversations_dir: str = "conversations"):
        self.conversations_dir = Path(conversations_dir)
        self.conversations_dir.mkdir(exist_ok=True)

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
