"""
MirroredTriggerHistory - writes to both baseline and conversation-specific directories.

Dual-write architecture ensures data is always available in both locations:
- baseline/ = the current working conversation (always exists after first message)
- conversation_id/ = timestamped backup of the same data
"""

from datetime import datetime
from typing import Iterator, Protocol

from agent.chain_of_action.trigger import Trigger
from agent.chain_of_action.action.action_data import ActionData
from agent.chain_of_action.trigger_history_entry import TriggerHistoryEntry
from agent.storage.interface import ITriggerHistory


class MirroredTriggerHistory(ITriggerHistory):
    """
    Writes to both baseline and conversation_id directories.

    This ensures every write is persisted to both locations for:
    - baseline/ = always-available working copy
    - conversation_id/ = timestamped backup

    Read operations delegate to primary (baseline).
    """

    def __init__(self, primary: ITriggerHistory, mirror: ITriggerHistory):
        """
        Initialize mirrored trigger history.

        Args:
            primary: The primary trigger history (baseline) - used for reads
            mirror: The mirror trigger history (conversation_id) - receives writes only
        """
        self._primary = primary
        self._mirror = mirror

    # ===================
    # Write Operations (write to both)
    # ===================

    def add_entry(self, entry: TriggerHistoryEntry) -> None:
        """Add a new trigger history entry to both storages."""
        self._primary.add_entry(entry)
        self._mirror.add_entry(entry)

    def update_entry(self, entry: TriggerHistoryEntry) -> None:
        """Update an existing entry in both storages."""
        self._primary.update_entry(entry)
        self._mirror.update_entry(entry)

    # ===================
    # Read Operations (delegate to primary)
    # ===================

    def get_first_entry(self) -> TriggerHistoryEntry | None:
        """Get the first (oldest) trigger entry from primary storage."""
        return self._primary.get_first_entry()

    def get_last_entry(self) -> TriggerHistoryEntry | None:
        """Get the last (most recent) trigger entry from primary storage."""
        return self._primary.get_last_entry()

    def get_entry_by_id(self, entry_id: str) -> TriggerHistoryEntry:
        """Get a single entry by ID from primary storage. Raises KeyError if not found."""
        return self._primary.get_entry_by_id(entry_id)

    def get_entry_count(self) -> int:
        """Get total number of entries from primary storage."""
        return self._primary.get_entry_count()

    def __len__(self) -> int:
        """Return the total number of entries from primary storage."""
        return len(self._primary)

    def iter_entries(self, reverse: bool, start: int) -> Iterator[TriggerHistoryEntry]:
        """Iterate over entries from primary storage."""
        return self._primary.iter_entries(reverse=reverse, start=start)

    def get_entry_index(self, entry_id: str) -> int:
        """Get the 0-based index position of an entry from primary storage."""
        return self._primary.get_entry_index(entry_id)

    def get_last_entry_by_trigger_type(
        self, trigger_type: str
    ) -> TriggerHistoryEntry | None:
        """Get the last entry of a specific trigger type from primary storage."""
        return self._primary.get_last_entry_by_trigger_type(trigger_type)
