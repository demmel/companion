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
from pydantic import BaseModel, Field
from typing import List


class AgentData(BaseModel):
    """Serializable agent data for persistence"""

    initial_exchange: TriggerHistoryEntry | None = Field(default=None)
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
        initial_exchange: TriggerHistoryEntry,
        save_baseline: bool = True,
    ) -> None:
        """Save a conversation with its state and optional trigger history"""

        self._save_state_and_triggers(
            conversation_id, state, trigger_history, initial_exchange
        )
        if save_baseline:
            self._save_state_and_triggers(
                "baseline", state, trigger_history, initial_exchange
            )

    def _save_state_and_triggers(
        self,
        prefix: str,
        state: State,
        trigger_history: TriggerHistory,
        initial_exchange: TriggerHistoryEntry,
    ) -> None:
        """Save the state and trigger history for a conversation"""
        state_file = self.conversations_dir / f"{prefix}_state.json"
        with open(state_file, "w") as f:
            f.write(state.model_dump_json(indent=2))

        trigger_file = self.conversations_dir / f"{prefix}_triggers.json"
        trigger_data = AgentData(
            initial_exchange=initial_exchange,
            entries=trigger_history.entries,
            summaries=trigger_history.summaries,
        )
        with open(trigger_file, "w") as f:
            f.write(trigger_data.model_dump_json(indent=2))

    def load_conversation(
        self,
        prefix: str,
    ) -> tuple[TriggerHistory, State | None, TriggerHistoryEntry | None]:
        """Load trigger history and state from conversation files with given prefix"""

        trigger_file = self._trigger_file_name(prefix)
        state_file = self._state_file_name(prefix)

        # Load trigger history
        trigger_history = TriggerHistory()
        initial_exchange = None
        if os.path.exists(trigger_file):
            with open(trigger_file, "r") as f:
                trigger_data = AgentData.model_validate(json.load(f))
                # Populate the trigger history
                initial_exchange = trigger_data.initial_exchange
                trigger_history.entries = trigger_data.entries
                trigger_history.summaries = trigger_data.summaries

        # Load state
        state = None
        if os.path.exists(state_file):
            with open(state_file, "r") as f:
                state = State.model_validate(json.load(f))

        return trigger_history, state, initial_exchange

    def _trigger_file_name(self, prefix: str) -> str:
        """Get the trigger file name for a conversation"""
        return f"{self.conversations_dir}/{prefix}_triggers.json"

    def _state_file_name(self, prefix: str) -> str:
        """Get the state file name for a conversation"""
        return f"{self.conversations_dir}/{prefix}_state.json"
