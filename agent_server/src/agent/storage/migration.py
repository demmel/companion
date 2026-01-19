"""
Migration utilities for converting from JSON file-based storage to SQLite + ChromaDB.

Handles both:
1. Directory structure migration (prefix-based to directory-per-conversation)
2. JSON -> SQLite/ChromaDB trigger history migration
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from agent.chain_of_action.trigger_history import TriggerHistory
from agent.conversation_persistence import ConversationData
from agent.storage.trigger_history_sqlite import (
    TriggerHistorySQLite,
    create_trigger_history_sqlite,
)

logger = logging.getLogger(__name__)


@dataclass
class MigrationResult:
    """Result of a migration operation."""

    conversation_id: str
    success: bool
    entries_migrated: int
    error: str | None = None


class TriggerHistoryMigrator:
    """Handles migration of trigger history from JSON to SQLite + ChromaDB."""

    def migrate_json_to_sqlite(
        self,
        json_path: str | Path,
        db_path: str | Path,
    ) -> MigrationResult:
        """
        Convert existing JSON trigger history to SQLite + ChromaDB.

        Args:
            json_path: Path to triggers.json file.
            db_path: Path for new triggers.db file.

        Returns:
            MigrationResult with success status and entry count.
        """
        json_path = Path(json_path)
        db_path = Path(db_path)
        conversation_id = json_path.stem.replace("_triggers", "")

        try:
            # Load JSON data
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Parse using existing Pydantic model
            conversation_data = ConversationData.model_validate(data)

            # Create SQLite + ChromaDB storage
            history = create_trigger_history_sqlite(db_path)

            # Add all entries
            for entry in conversation_data.entries:
                history.add_entry(entry)

            entry_count = history.get_entry_count()
            history.dispose()

            logger.info(
                f"Migrated {entry_count} entries from {json_path.name} to {db_path.name}"
            )

            return MigrationResult(
                conversation_id=conversation_id,
                success=True,
                entries_migrated=entry_count,
            )

        except Exception as e:
            logger.error(f"Migration failed for {json_path}: {e}")
            return MigrationResult(
                conversation_id=conversation_id,
                success=False,
                entries_migrated=0,
                error=str(e),
            )

    def migrate_all(self, conversations_dir: str | Path) -> list[MigrationResult]:
        """
        Migrate all *_triggers.json files in a directory.

        Args:
            conversations_dir: Directory containing conversation files.

        Returns:
            List of MigrationResults for each conversation.
        """
        conversations_dir = Path(conversations_dir)
        results = []

        # Find all old-style trigger files
        for json_path in conversations_dir.glob("*_triggers.json"):
            # Skip if already migrated (db exists)
            prefix = json_path.stem.replace("_triggers", "")
            conv_dir = conversations_dir / prefix

            if conv_dir.is_dir() and (conv_dir / "triggers.db").exists():
                logger.info(f"Skipping {prefix} - already migrated")
                continue

            # Migrate to new directory structure first
            conv_dir = migrate_conversation_directory(conversations_dir, prefix)
            db_path = conv_dir / "triggers.db"
            new_json_path = conv_dir / "triggers.json"

            # Migrate JSON to SQLite
            result = self.migrate_json_to_sqlite(new_json_path, db_path)
            results.append(result)

        return results


def migrate_conversation_directory(
    conversations_dir: Path,
    prefix: str,
) -> Path:
    """
    Migrate old prefix-based files to new directory-per-conversation structure.

    Old structure:
        conversations/
            baseline_state.json
            baseline_triggers.json
            baseline_dag.json
            baseline_dag_actions.json

    New structure:
        conversations/
            baseline/
                state.json
                triggers.db (SQLite)
                chroma/     (ChromaDB embeddings)
                dag.json
                dag_actions.json

    Args:
        conversations_dir: Root conversations directory.
        prefix: Conversation prefix (e.g., "baseline").

    Returns:
        Path to the conversation directory.
    """
    conversations_dir = Path(conversations_dir)
    conv_dir = conversations_dir / prefix

    if conv_dir.exists() and conv_dir.is_dir():
        # Already migrated or new conversation
        return conv_dir

    # Check for old-style files
    old_triggers = conversations_dir / f"{prefix}_triggers.json"
    if not old_triggers.exists():
        # New conversation - create directory
        conv_dir.mkdir(exist_ok=True)
        return conv_dir

    # Migrate: create directory and move files
    logger.info(f"Migrating conversation {prefix} to directory structure")
    conv_dir.mkdir()

    # Map old suffix to new filename
    suffix_map = {
        "_state.json": "state.json",
        "_triggers.json": "triggers.json",
        "_dag.json": "dag.json",
        "_dag_actions.json": "dag_actions.json",
    }

    for old_suffix, new_name in suffix_map.items():
        old_path = conversations_dir / f"{prefix}{old_suffix}"
        if old_path.exists():
            new_path = conv_dir / new_name
            old_path.rename(new_path)
            logger.debug(f"Moved {old_path.name} -> {conv_dir.name}/{new_name}")

    return conv_dir


def ensure_conversation_directory(
    conversations_dir: Path,
    conversation_id: str,
) -> Path:
    """
    Ensure conversation directory exists, migrating if necessary.

    Args:
        conversations_dir: Root conversations directory.
        conversation_id: Conversation ID/prefix.

    Returns:
        Path to the conversation directory.
    """
    return migrate_conversation_directory(conversations_dir, conversation_id)


def load_trigger_history(
    conversations_dir: Path,
    conversation_id: str,
) -> TriggerHistorySQLite:
    """
    Load trigger history, auto-migrating to SQLite + ChromaDB if needed.

    Priority:
    1. Use existing SQLite database if present
    2. Migrate JSON to SQLite if JSON exists
    3. Create new empty database

    Args:
        conversations_dir: Root conversations directory.
        conversation_id: Conversation ID/prefix.

    Returns:
        Loaded TriggerHistorySQLite instance.
    """
    # Ensure directory structure (migrate if needed)
    conv_dir = ensure_conversation_directory(conversations_dir, conversation_id)

    db_path = conv_dir / "triggers.db"
    json_path = conv_dir / "triggers.json"

    if db_path.exists():
        # Use existing SQLite + ChromaDB
        logger.info(f"Loading SQLite trigger history from {db_path}")
        history = create_trigger_history_sqlite(db_path)
        return history

    if json_path.exists():
        # Auto-migrate JSON to SQLite + ChromaDB
        logger.info(f"Auto-migrating {json_path} to SQLite + ChromaDB")
        migrator = TriggerHistoryMigrator()
        result = migrator.migrate_json_to_sqlite(json_path, db_path)

        if not result.success:
            raise RuntimeError(f"Failed to migrate {json_path}: {result.error}")

        history = create_trigger_history_sqlite(db_path)
        return history

    # New conversation
    logger.info(f"Creating new SQLite + ChromaDB trigger history at {db_path}")
    history = create_trigger_history_sqlite(db_path)
    return history


def load_trigger_history_json(
    conversations_dir: Path,
    conversation_id: str,
) -> TriggerHistory:
    """
    Load trigger history from JSON (legacy method for compatibility/benchmarking).
    """
    conv_dir = ensure_conversation_directory(conversations_dir, conversation_id)
    json_path = conv_dir / "triggers.json"

    if not json_path.exists():
        # Check old-style path
        old_path = conversations_dir / f"{conversation_id}_triggers.json"
        if old_path.exists():
            json_path = old_path

    trigger_history = TriggerHistory()

    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        conversation_data = ConversationData.model_validate(data)
        trigger_history.entries = conversation_data.entries

    return trigger_history


def export_sqlite_to_json(
    db_history: TriggerHistorySQLite,
    output_path: str | Path,
) -> int:
    """
    Export SQLite trigger history back to JSON format.

    Args:
        db_history: The SQLite trigger history instance.
        output_path: Path for the output JSON file.

    Returns:
        Number of entries exported.
    """
    output_path = Path(output_path)

    # Get all entries from SQLite
    entries = db_history.get_all_entries()

    # Create ConversationData and export
    conversation_data = ConversationData(entries=entries)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(conversation_data.model_dump_json(indent=2))

    logger.info(f"Exported {len(entries)} entries to {output_path}")
    return len(entries)
