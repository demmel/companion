"""
Conversation persistence system with unique IDs and auto-save functionality
"""

from dataclasses import dataclass
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import List

from agent.memory.memory import IMemory
from agent.memory.storage import load_memory, save_memory
from agent.timeit import timeit

from .state import State
from .chain_of_action.trigger_history import (
    TriggerHistory,
    TriggerHistoryEntry,
)
from pydantic import BaseModel
from typing import List


class ConversationData(BaseModel):
    """Serializable agent data for persistence"""

    entries: List[TriggerHistoryEntry]


@dataclass
class AgentData:
    """Serializable agent data for persistence"""

    trigger_history: TriggerHistory
    state: State
    memory: IMemory


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
        memory: IMemory,
        save_baseline: bool = True,
    ) -> None:
        """Save a conversation with its state and optional trigger history"""

        self._save_state_and_triggers(
            conversation_id,
            state,
            trigger_history,
            memory,
        )
        if save_baseline:
            self._save_state_and_triggers("baseline", state, trigger_history, memory)

    def _save_state_and_triggers(
        self,
        prefix: str,
        state: State,
        trigger_history: TriggerHistory,
        memory: IMemory,
    ) -> None:
        """Save the state and trigger history for a conversation"""
        state_file = self._state_file_name(prefix)
        with timeit("Saving state to fiel"):
            with open(state_file, "w", encoding="utf-8") as f:
                f.write(state.model_dump_json(indent=2))

        trigger_file = self._trigger_file_name(prefix)
        trigger_data = ConversationData(
            entries=trigger_history.entries,
        )
        with timeit("Saving trigger history to file"):
            with open(trigger_file, "w", encoding="utf-8") as f:
                f.write(trigger_data.model_dump_json(indent=2))

        save_memory(
            self.conversations_dir,
            prefix,
            memory,
        )

    def load_agent_data(self, prefix: str) -> AgentData:
        """Load agent data (state and trigger history) from conversation files with given prefix"""

        trigger_file = self._trigger_file_name(prefix)
        state_file = self._state_file_name(prefix)

        if not os.path.exists(trigger_file):
            raise FileNotFoundError(f"Trigger file not found: {trigger_file}")

        if not os.path.exists(state_file):
            raise FileNotFoundError(f"State file not found: {state_file}")

        # Load trigger history
        trigger_history = TriggerHistory()
        with open(trigger_file, "r", encoding="utf-8") as f:
            trigger_data = ConversationData.model_validate(json.load(f))
            # Populate the trigger history
            trigger_history.entries = trigger_data.entries

        # Load state
        with open(state_file, "r", encoding="utf-8") as f:
            state = State.model_validate(json.load(f))

        memory = load_memory(
            self.conversations_dir,
            prefix,
            trigger_history,
            resave=False,
        )

        return AgentData(
            trigger_history=trigger_history,
            state=state,
            memory=memory,
        )

    def _trigger_file_name(self, prefix: str) -> str:
        """Get the trigger file name for a conversation"""
        return f"{self.conversations_dir}/{prefix}_triggers.json"

    def _state_file_name(self, prefix: str) -> str:
        """Get the state file name for a conversation"""
        return f"{self.conversations_dir}/{prefix}_state.json"
