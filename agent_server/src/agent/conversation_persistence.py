"""
Conversation persistence system with unique IDs and auto-save functionality
"""

from dataclasses import dataclass
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from agent.experiments.temporal_context_dag.dag_memory_manager import DagMemoryManager

from .state import State
from .chain_of_action.trigger_history import (
    TriggerHistory,
    TriggerHistoryEntry,
    SummaryRecord,
)
from pydantic import BaseModel, Field
from typing import List


class ConversationData(BaseModel):
    """Serializable agent data for persistence"""

    initial_exchange: TriggerHistoryEntry | None = Field(default=None)
    entries: List[TriggerHistoryEntry]
    summaries: List[SummaryRecord]


@dataclass
class AgentData:
    """Serializable agent data for persistence"""

    trigger_history: TriggerHistory
    state: State | None
    initial_exchange: TriggerHistoryEntry | None


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
        initial_exchange: Optional[TriggerHistoryEntry],
        save_baseline: bool = True,
        dag_memory_manager: Optional[DagMemoryManager] = None,
    ) -> None:
        """Save a conversation with its state and optional trigger history"""

        self._save_state_and_triggers(
            conversation_id,
            state,
            trigger_history,
            initial_exchange,
            dag_memory_manager,
        )
        if save_baseline:
            self._save_state_and_triggers(
                "baseline", state, trigger_history, initial_exchange, dag_memory_manager
            )

    def _save_state_and_triggers(
        self,
        prefix: str,
        state: State,
        trigger_history: TriggerHistory,
        initial_exchange: Optional[TriggerHistoryEntry],
        dag_memory_manager: Optional[DagMemoryManager] = None,
    ) -> None:
        """Save the state and trigger history for a conversation"""
        state_file = self._state_file_name(prefix)
        with open(state_file, "w") as f:
            f.write(state.model_dump_json(indent=2))

        trigger_file = self._trigger_file_name(prefix)
        trigger_data = ConversationData(
            initial_exchange=initial_exchange,
            entries=trigger_history.entries,
            summaries=trigger_history.summaries,
        )
        with open(trigger_file, "w") as f:
            f.write(trigger_data.model_dump_json(indent=2))

        # Save DAG memory data if present
        if dag_memory_manager:
            dag_file = self._dag_file_name(prefix)
            dag_memory_manager.save_to_file(dag_file)

    def load_agent_data(self, prefix: str) -> AgentData:
        """Load agent data (state and trigger history) from conversation files with given prefix"""
        trigger_history, state, initial_exchange, _ = self.load_conversation(prefix)
        return AgentData(
            trigger_history=trigger_history,
            initial_exchange=initial_exchange,
            state=state,
        )

    def load_conversation(
        self,
        prefix: str,
    ) -> tuple[
        TriggerHistory,
        State | None,
        TriggerHistoryEntry | None,
        DagMemoryManager | None,
    ]:
        """Load trigger history and state from conversation files with given prefix"""

        trigger_file = self._trigger_file_name(prefix)
        state_file = self._state_file_name(prefix)

        # Load trigger history
        trigger_history = TriggerHistory()
        initial_exchange = None
        if os.path.exists(trigger_file):
            with open(trigger_file, "r") as f:
                trigger_data = ConversationData.model_validate(json.load(f))
                # Populate the trigger history
                initial_exchange = trigger_data.initial_exchange
                trigger_history.entries = trigger_data.entries
                trigger_history.summaries = trigger_data.summaries

        # Load state
        state = None
        if os.path.exists(state_file):
            with open(state_file, "r") as f:
                state = State.model_validate(json.load(f))

        dag = None
        if os.path.exists(self._dag_file_name(prefix)):
            dag = DagMemoryManager.load_from_file(self._dag_file_name(prefix))

        return trigger_history, state, initial_exchange, dag

    def _trigger_file_name(self, prefix: str) -> str:
        """Get the trigger file name for a conversation"""
        return f"{self.conversations_dir}/{prefix}_triggers.json"

    def _state_file_name(self, prefix: str) -> str:
        """Get the state file name for a conversation"""
        return f"{self.conversations_dir}/{prefix}_state.json"

    def _dag_file_name(self, prefix: str) -> str:
        """Get the DAG memory file name for a conversation"""
        return f"{self.conversations_dir}/{prefix}_dag.json"
