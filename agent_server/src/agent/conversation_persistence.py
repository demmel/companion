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

from agent.timeit import timeit
from agent.memory import (
    DagMemoryManager,
)
from agent.memory.action_log import MemoryActionLog

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

    entries: List[TriggerHistoryEntry]


@dataclass
class AgentData:
    """Serializable agent data for persistence"""

    trigger_history: TriggerHistory
    state: State
    dag_memory_manager: DagMemoryManager


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
        dag_memory_manager: DagMemoryManager,
        save_baseline: bool = True,
    ) -> None:
        """Save a conversation with its state and optional trigger history"""

        self._save_state_and_triggers(
            conversation_id,
            state,
            trigger_history,
            dag_memory_manager,
        )
        if save_baseline:
            self._save_state_and_triggers(
                "baseline", state, trigger_history, dag_memory_manager
            )

    def _save_state_and_triggers(
        self,
        prefix: str,
        state: State,
        trigger_history: TriggerHistory,
        dag_memory_manager: DagMemoryManager,
    ) -> None:
        """Save the state and trigger history for a conversation"""
        state_file = self._state_file_name(prefix)
        with timeit("Saving state to fiel"):
            with open(state_file, "w") as f:
                f.write(state.model_dump_json(indent=2))

        trigger_file = self._trigger_file_name(prefix)
        trigger_data = ConversationData(
            entries=trigger_history.entries,
        )
        with timeit("Saving trigger history to file"):
            with open(trigger_file, "w") as f:
                f.write(trigger_data.model_dump_json(indent=2))

        # Save DAG memory data if present
        if dag_memory_manager:
            dag_file = self._dag_file_name(prefix)
            with timeit("Saving DAG memory to file"):
                dag_memory_manager.save_to_file(dag_file)
            with timeit("Saving DAG memory action log to file"):
                dag_memory_manager.save_action_log(
                    self._dag_action_log_file_name(prefix)
                )

    def load_agent_data(self, prefix: str) -> AgentData:
        """Load agent data (state and trigger history) from conversation files with given prefix"""

        trigger_file = self._trigger_file_name(prefix)
        state_file = self._state_file_name(prefix)
        dag_file = self._dag_file_name(prefix)
        dag_action_log_file = self._dag_action_log_file_name(prefix)

        if not os.path.exists(trigger_file):
            raise FileNotFoundError(f"Trigger file not found: {trigger_file}")

        if not os.path.exists(state_file):
            raise FileNotFoundError(f"State file not found: {state_file}")

        if not os.path.exists(dag_file):
            raise FileNotFoundError(f"DAG memory file not found: {dag_file}")

        if not os.path.exists(dag_action_log_file):
            raise FileNotFoundError(
                f"DAG action log file not found: {dag_action_log_file}"
            )

        # Load trigger history
        trigger_history = TriggerHistory()
        with open(trigger_file, "r") as f:
            trigger_data = ConversationData.model_validate(json.load(f))
            # Populate the trigger history
            trigger_history.entries = trigger_data.entries

        # Load state
        with open(state_file, "r") as f:
            state = State.model_validate(json.load(f))

        with timeit("Loading DAG memory from file"):
            dag = DagMemoryManager.load_from_file(
                self._dag_file_name(prefix), trigger_history
            )
        # with timeit("Loading DAG memory from action log file"):
        #     dag = DagMemoryManager.load_from_action_log(
        #         self._dag_action_log_file_name(prefix), trigger_history=trigger_history
        #     )
        with timeit("Loading DAG memory action log from file"):
            action_log = MemoryActionLog.load_from_file(
                self._dag_action_log_file_name(prefix)
            )
        dag.action_log = action_log
        with timeit("Replaying DAG memory action log"):
            _, _ = dag.action_log.replay_from_empty(trigger_history)

        # self.save_conversation(prefix, state, trigger_history, dag, save_baseline=False)

        return AgentData(
            trigger_history=trigger_history,
            state=state,
            dag_memory_manager=dag,
        )

    def _trigger_file_name(self, prefix: str) -> str:
        """Get the trigger file name for a conversation"""
        return f"{self.conversations_dir}/{prefix}_triggers.json"

    def _state_file_name(self, prefix: str) -> str:
        """Get the state file name for a conversation"""
        return f"{self.conversations_dir}/{prefix}_state.json"

    def _dag_file_name(self, prefix: str) -> str:
        """Get the DAG memory file name for a conversation"""
        return f"{self.conversations_dir}/{prefix}_dag.json"

    def _dag_action_log_file_name(self, prefix: str) -> str:
        """Get the DAG memory action log file name for a conversation"""
        return f"{self.conversations_dir}/{prefix}_dag_actions.json"
