"""
Conversation persistence system with unique IDs and directory-per-conversation structure.

Architecture:
- Each conversation lives in its own directory: conversations/{conversation_id}/
- baseline/ = the current working conversation (always exists after first message)
- conversation_id/ = timestamped backup of the same data
- Dual-write: every write goes to BOTH baseline and conversation_id simultaneously
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import shutil
import time
from datetime import datetime
from pathlib import Path

from agent.memory.memory import IMemory
from agent.memory.storage import load_memory, save_memory
from agent.storage import (
    ITriggerHistory,
    TriggerHistory,
    MirroredTriggerHistory,
    create_trigger_history,
)
from agent.timeit import timeit

from .state import State


logger = logging.getLogger(__name__)


@dataclass
class ConversationContext:
    """Result of creating or loading a conversation."""

    conversation_id: str
    trigger_history: ITriggerHistory
    persistence: ConversationPersistence
    state: State | None = None
    memory: IMemory | None = None


@dataclass
class AgentData:
    """Loaded agent data for an existing conversation."""

    trigger_history: ITriggerHistory
    state: State
    memory: IMemory


class ConversationPersistence:
    """
    Manages conversation persistence with directory-per-conversation structure.

    Directory layout:
        conversations/
            baseline/           <- Current working conversation
                triggers.db
                chroma/
                state.json
                dag.json
                dag_actions.json
            conversation_20240115_123456_789012/
                triggers.db
                chroma/
                state.json
                dag.json
                dag_actions.json
    """

    BASELINE = "baseline"

    def __init__(self, conversations_dir: str = "conversations"):
        self.conversations_dir = Path(conversations_dir)
        self.conversations_dir.mkdir(exist_ok=True)

    def generate_conversation_id(self) -> str:
        """Generate a unique conversation ID based on timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Add microseconds for uniqueness if multiple conversations start at same second
        microseconds = int(time.time() * 1000000) % 1000000
        return f"conversation_{timestamp}_{microseconds:06d}"

    def get_conversation_dir(self, conversation_id: str) -> Path:
        """Get directory for a conversation, creating it if needed."""
        conv_dir = self.conversations_dir / conversation_id
        conv_dir.mkdir(exist_ok=True)
        return conv_dir

    def _copy_conversation(self, from_id: str, to_id: str) -> None:
        """Copy conversation directory contents."""
        from_dir = self.conversations_dir / from_id
        to_dir = self.conversations_dir / to_id

        if not from_dir.exists():
            logger.warning(f"Source conversation directory does not exist: {from_dir}")
            return

        if to_dir.exists():
            # Remove existing directory to ensure clean copy
            shutil.rmtree(to_dir)

        shutil.copytree(from_dir, to_dir)
        logger.info(f"Copied conversation from {from_id} to {to_id}")

    def _create_trigger_history(self, conversation_id: str) -> TriggerHistory:
        """Create a TriggerHistory for a conversation directory."""
        conv_dir = self.get_conversation_dir(conversation_id)
        return create_trigger_history(conv_dir / "triggers.db")

    def _create_mirrored_history(self, conversation_id: str) -> MirroredTriggerHistory:
        """Create trigger history that writes to both baseline and conversation_id."""
        baseline_history = self._create_trigger_history(self.BASELINE)
        conv_history = self._create_trigger_history(conversation_id)
        return MirroredTriggerHistory(baseline_history, conv_history)

    def new_conversation(self) -> ConversationContext:
        """
        Create a new conversation.

        Writes go to both baseline/ and a new timestamped conversation_id/.
        Returns a ConversationContext with mirrored trigger history.
        """
        # Clear baseline directory to start fresh (retry for Windows file locking)
        baseline_dir = self.conversations_dir / self.BASELINE
        if baseline_dir.exists():
            for attempt in range(10):
                try:
                    shutil.rmtree(baseline_dir)
                    break
                except PermissionError:
                    if attempt == 9:
                        raise
                    time.sleep(0.1 * (attempt + 1))

        conversation_id = self.generate_conversation_id()
        trigger_history = self._create_mirrored_history(conversation_id)

        logger.info(f"Created new conversation: {conversation_id}")

        return ConversationContext(
            conversation_id=conversation_id,
            trigger_history=trigger_history,
            persistence=self,
        )

    def load_conversation(
        self, conversation_id: str, use_individual_formatting: bool
    ) -> ConversationContext:
        """
        Load an existing conversation.

        - Copies source conversation to baseline/ (if not already baseline)
        - Copies source conversation to a new timestamped backup
        - Returns mirrored history that writes to both
        - Loads state and memory from source

        Args:
            conversation_id: ID of conversation to load (can be "baseline" or timestamped)
            use_individual_formatting: Whether to use individual memory formatting

        Returns:
            ConversationContext with mirrored trigger history and loaded state/memory
        """
        source_dir = self.get_conversation_dir(conversation_id)
        state_file = source_dir / "state.json"

        if not state_file.exists():
            raise FileNotFoundError(f"State file not found: {state_file}")

        # Generate new conversation ID for this session's backup
        new_id = self.generate_conversation_id()

        # Copy source to baseline (if loading from a non-baseline conversation)
        if conversation_id != self.BASELINE:
            self._copy_conversation(conversation_id, self.BASELINE)

        # Copy source to new conversation backup
        self._copy_conversation(conversation_id, new_id)

        # Create mirrored history pointing to baseline and new backup
        trigger_history = self._create_mirrored_history(new_id)

        # Load state from source
        with open(state_file, "r", encoding="utf-8") as f:
            state = State.model_validate(json.load(f))

        # Load memory from baseline (which now contains the source data)
        baseline_dir = self.get_conversation_dir(self.BASELINE)
        memory = load_memory(
            baseline_dir,
            "",  # Empty prefix - files are directly in conversation dir
            trigger_history._primary,
            use_individual_formatting,
            resave=False,
        )

        logger.info(f"Loaded conversation {conversation_id} -> {new_id}")

        return ConversationContext(
            conversation_id=new_id,
            trigger_history=trigger_history,
            persistence=self,
            state=state,
            memory=memory,
        )

    def save_conversation(
        self,
        conversation_id: str,
        state: State,
        trigger_history: ITriggerHistory,
        memory: IMemory,
    ) -> None:
        """
        Save conversation state and memory.

        Note: Trigger entries are persisted immediately via MirroredTriggerHistory.
        This method saves state.json and memory files to both baseline and conversation_id.

        Args:
            conversation_id: The conversation ID (for the backup copy)
            state: Current agent state
            trigger_history: Trigger history (used for memory operations)
            memory: Memory system to save
        """
        # Save to both baseline and conversation_id
        for target_id in [self.BASELINE, conversation_id]:
            conv_dir = self.get_conversation_dir(target_id)
            self._save_to_dir(conv_dir, state, memory)

        logger.info(f"Saved conversation state to {conversation_id} and baseline")

    def _save_to_dir(
        self,
        conv_dir: Path,
        state: State,
        memory: IMemory,
    ) -> None:
        """Save state and memory to a conversation directory."""
        # Save state
        state_file = conv_dir / "state.json"
        with timeit(f"Saving state to {state_file}"):
            with open(state_file, "w", encoding="utf-8") as f:
                f.write(state.model_dump_json(indent=2))

        # Save memory (uses empty prefix since files go directly in conv_dir)
        save_memory(conv_dir, "", memory)

    def load_agent_data(
        self, conversation_id: str, use_individual_formatting: bool
    ) -> AgentData:
        """
        Load just state and memory for an existing conversation.

        This is used when trigger history is already connected separately.

        Args:
            conversation_id: ID of conversation to load
            use_individual_formatting: Whether to use individual memory formatting

        Returns:
            AgentData with trigger_history, state, and memory
        """
        conv_dir = self.get_conversation_dir(conversation_id)
        state_file = conv_dir / "state.json"

        if not state_file.exists():
            raise FileNotFoundError(f"State file not found: {state_file}")

        # Load state
        with open(state_file, "r", encoding="utf-8") as f:
            state = State.model_validate(json.load(f))

        # Load memory (needs a trigger history for replay)
        trigger_history = self._create_trigger_history(conversation_id)
        memory = load_memory(
            conv_dir,
            "",  # Empty prefix
            trigger_history,
            use_individual_formatting,
            resave=False,
        )

        return AgentData(
            trigger_history=trigger_history,
            state=state,
            memory=memory,
        )
