"""
Mass migration script: JSON flat files to directory-per-conversation with SQLite + ChromaDB.

Migrates all existing conversations from old flat-file JSON format to new directory structure.

Usage:
    uv run python -m agent.scripts.migrate_conversations --dry-run
    uv run python -m agent.scripts.migrate_conversations
"""

import argparse
import logging
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

from tqdm import tqdm

from agent.conversation_persistence import AgentData, ConversationPersistence
from agent.memory.dag.storage import (
    _dag_action_log_file_name,
    _dag_file_name,
    save_dag_memory,
)
from agent.storage import create_trigger_history_sqlite

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class MigrationResult:
    """Result of migrating a single conversation."""

    prefix: str
    success: bool
    entries_migrated: int = 0
    error: str | None = None
    archived: bool = False


@dataclass
class MigrationReport:
    """Report of all migration operations."""

    total_prefixes: int = 0
    successful: list[MigrationResult] = field(default_factory=list)
    failed: list[MigrationResult] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    def add_success(self, result: MigrationResult) -> None:
        self.successful.append(result)

    def add_failure(self, result: MigrationResult) -> None:
        self.failed.append(result)

    def add_skipped(self, prefix: str) -> None:
        self.skipped.append(prefix)

    def print_summary(self) -> None:
        print("\n" + "=" * 60)
        print("MIGRATION REPORT")
        print("=" * 60)
        print(f"Total prefixes found: {self.total_prefixes}")
        print(f"Successful migrations: {len(self.successful)}")
        print(f"Failed (archived): {len(self.failed)}")
        print(f"Skipped (already migrated): {len(self.skipped)}")

        if self.successful:
            print("\nSuccessful migrations:")
            total_entries = 0
            for result in self.successful:
                print(f"  - {result.prefix}: {result.entries_migrated} entries")
                total_entries += result.entries_migrated
            print(f"  Total entries migrated: {total_entries}")

        if self.failed:
            print("\nFailed migrations (archived):")
            for result in self.failed:
                error_preview = (
                    result.error[:80] + "..." if result.error and len(result.error) > 80 else result.error
                )
                print(f"  - {result.prefix}: {error_preview}")

        if self.skipped:
            print(f"\nSkipped {len(self.skipped)} already-migrated conversations")

        print("=" * 60)


def discover_conversation_prefixes(conversations_dir: Path) -> list[str]:
    """
    Find all unique conversation prefixes from trigger files.

    Handles two naming patterns:
    1. New format: {prefix}_triggers.json
    2. Old format: {prefix}.json (with matching {prefix}_state.json)

    Args:
        conversations_dir: Root conversations directory.

    Returns:
        List of unique prefixes (e.g., ["baseline", "chloe_20250725_234412_437648"]).
    """
    prefixes: set[str] = set()

    def is_backup_file(filename: str) -> bool:
        if filename.endswith(".bak"):
            return True
        if filename.endswith(".old"):
            return True
        if "_backup_" in filename:
            return True
        if "_rollback_" in filename:
            return True
        return False

    # Pattern 1: *_triggers.json files
    for trigger_file in conversations_dir.glob("*_triggers.json"):
        filename = trigger_file.name
        if is_backup_file(filename):
            continue
        prefix = filename.replace("_triggers.json", "")
        prefixes.add(prefix)

    # Pattern 2: *.json files with matching *_state.json (old format)
    for json_file in conversations_dir.glob("*.json"):
        filename = json_file.name
        if is_backup_file(filename):
            continue
        # Skip if it's a _triggers, _state, _dag, or _dag_actions file
        if "_triggers.json" in filename:
            continue
        if "_state.json" in filename:
            continue
        if "_dag.json" in filename:
            continue
        if "_dag_actions.json" in filename:
            continue

        # This might be an old-format trigger file - check for matching state file
        prefix = filename.replace(".json", "")
        state_file = conversations_dir / f"{prefix}_state.json"
        if state_file.exists():
            prefixes.add(prefix)

    return sorted(prefixes)


def is_already_migrated(conversations_dir: Path, prefix: str) -> bool:
    """Check if a conversation has already been migrated to the new format."""
    conv_dir = conversations_dir / prefix
    return conv_dir.is_dir() and (conv_dir / "triggers.db").exists()


def archive_failed(
    conversations_dir: Path,
    prefix: str,
    error: str,
    dry_run: bool = False,
) -> Path:
    """
    Move failed conversation files to archive directory.

    Args:
        conversations_dir: Root conversations directory.
        prefix: Conversation prefix.
        error: Error message from failed migration.
        dry_run: If True, only log what would happen.

    Returns:
        Path to archive directory.
    """
    archive_dir = conversations_dir / "archive" / prefix

    if dry_run:
        logger.info(f"[DRY RUN] Would archive {prefix} to {archive_dir}")
        return archive_dir

    archive_dir.mkdir(parents=True, exist_ok=True)

    # Find all files with this prefix (both old and new formats)
    patterns_to_move = [
        f"{prefix}_state.json",
        f"{prefix}_triggers.json",
        f"{prefix}_dag.json",
        f"{prefix}_dag_actions.json",
        f"{prefix}.json",  # Old format trigger file
    ]

    moved_count = 0
    for pattern in patterns_to_move:
        old_path = conversations_dir / pattern
        if old_path.exists():
            new_path = archive_dir / pattern
            shutil.move(str(old_path), str(new_path))
            logger.debug(f"Moved {old_path.name} to archive")
            moved_count += 1

    # Write error.txt with failure reason
    error_file = archive_dir / "error.txt"
    error_file.write_text(f"Migration failed for {prefix}\n\nError:\n{error}\n")

    logger.info(f"Archived {moved_count} files for {prefix} to {archive_dir}")
    return archive_dir


def save_new_format(
    conversations_dir: Path,
    prefix: str,
    agent_data: AgentData,
    dry_run: bool = False,
) -> Path:
    """
    Save agent data in new directory-per-conversation format.

    Creates:
        conversations/{prefix}/
            state.json
            triggers.db (SQLite)
            chroma/ (ChromaDB embeddings)
            dag.json
            dag_actions.json

    Args:
        conversations_dir: Root conversations directory.
        prefix: Conversation prefix.
        agent_data: Loaded agent data to save.
        dry_run: If True, only log what would happen.

    Returns:
        Path to the new conversation directory.
    """
    conv_dir = conversations_dir / prefix

    if dry_run:
        logger.info(f"[DRY RUN] Would create {conv_dir} with new format")
        return conv_dir

    conv_dir.mkdir(exist_ok=True)

    # Save state
    state_file = conv_dir / "state.json"
    state_file.write_text(agent_data.state.model_dump_json(indent=2))
    logger.debug(f"Saved state to {state_file}")

    # Save triggers to SQLite + ChromaDB
    db_path = conv_dir / "triggers.db"
    history = create_trigger_history_sqlite(db_path)
    for entry in agent_data.trigger_history.entries:
        history.add_entry(entry)
    entry_count = history.get_entry_count()
    history.dispose()
    logger.debug(f"Saved {entry_count} trigger entries to {db_path}")

    # Save DAG memory
    save_dag_memory(conv_dir, "", agent_data.memory)  # type: ignore[arg-type]
    # Rename from _dag.json to dag.json (no prefix in new format)
    old_dag = conv_dir / "_dag.json"
    new_dag = conv_dir / "dag.json"
    if old_dag.exists():
        old_dag.rename(new_dag)
    old_dag_actions = conv_dir / "_dag_actions.json"
    new_dag_actions = conv_dir / "dag_actions.json"
    if old_dag_actions.exists():
        old_dag_actions.rename(new_dag_actions)

    logger.info(f"Created {conv_dir} with new format ({entry_count} entries)")
    return conv_dir


def migrate_single_conversation(
    conversations_dir: Path,
    prefix: str,
    persistence: ConversationPersistence,
    dry_run: bool = False,
) -> MigrationResult:
    """
    Attempt to migrate a single conversation.

    Args:
        conversations_dir: Root conversations directory.
        prefix: Conversation prefix.
        persistence: ConversationPersistence instance.
        dry_run: If True, only log what would happen.

    Returns:
        MigrationResult with success status.
    """
    try:
        # Check which trigger file format exists
        new_format = conversations_dir / f"{prefix}_triggers.json"
        old_format = conversations_dir / f"{prefix}.json"

        if not new_format.exists() and old_format.exists():
            # Old format: {prefix}.json -> rename to {prefix}_triggers.json temporarily
            # so ConversationPersistence can find it
            import shutil
            shutil.copy(str(old_format), str(new_format))
            try:
                agent_data = persistence.load_agent_data(prefix, use_individual_formatting=True)
            finally:
                # Clean up the temporary file
                if new_format.exists():
                    new_format.unlink()
        else:
            # Try to load agent data using existing persistence
            agent_data = persistence.load_agent_data(prefix, use_individual_formatting=True)

        if dry_run:
            entry_count = len(agent_data.trigger_history.entries)
            return MigrationResult(
                prefix=prefix,
                success=True,
                entries_migrated=entry_count,
            )

        # Save in new format
        save_new_format(conversations_dir, prefix, agent_data, dry_run=False)

        return MigrationResult(
            prefix=prefix,
            success=True,
            entries_migrated=len(agent_data.trigger_history.entries),
        )

    except Exception as e:
        # Truncate error message to avoid spamming logs with long validation errors
        error_str = str(e)
        if len(error_str) > 200:
            error_str = error_str[:200] + "..."
        error_msg = f"{type(e).__name__}: {error_str}"

        if not dry_run:
            archive_failed(conversations_dir, prefix, error_msg, dry_run=False)

        return MigrationResult(
            prefix=prefix,
            success=False,
            error=error_msg,
            archived=not dry_run,
        )


def migrate_all_conversations(
    conversations_dir: Path,
    dry_run: bool = False,
) -> MigrationReport:
    """
    Migrate all conversations from flat-file JSON to directory-per-conversation format.

    Args:
        conversations_dir: Root conversations directory.
        dry_run: If True, only log what would happen.

    Returns:
        MigrationReport with all results.
    """
    report = MigrationReport()

    # Discover all prefixes
    prefixes = discover_conversation_prefixes(conversations_dir)
    report.total_prefixes = len(prefixes)
    logger.info(f"Found {len(prefixes)} conversation prefixes")

    if not prefixes:
        logger.info("No conversations to migrate")
        return report

    # Create persistence instance
    persistence = ConversationPersistence(str(conversations_dir))

    # Suppress noisy logging during migration loop
    logging.getLogger("agent").setLevel(logging.WARNING)

    # Process each prefix with progress bar
    for prefix in tqdm(prefixes, desc="Migrating", unit="conv"):
        # Check if already migrated
        if is_already_migrated(conversations_dir, prefix):
            report.add_skipped(prefix)
            continue

        # Attempt migration
        result = migrate_single_conversation(
            conversations_dir,
            prefix,
            persistence,
            dry_run=dry_run,
        )

        if result.success:
            report.add_success(result)
        else:
            report.add_failure(result)

    return report


def main() -> int:
    """Main entry point for the migration script."""
    parser = argparse.ArgumentParser(
        description="Migrate conversations from JSON to SQLite + ChromaDB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    uv run python -m agent.scripts.migrate_conversations --dry-run
    uv run python -m agent.scripts.migrate_conversations
    uv run python -m agent.scripts.migrate_conversations --conversations-dir /path/to/conversations
        """,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without making changes",
    )
    parser.add_argument(
        "--conversations-dir",
        type=Path,
        default=Path("conversations"),
        help="Path to conversations directory (default: conversations)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.dry_run:
        logger.info("=" * 60)
        logger.info("DRY RUN MODE - No changes will be made")
        logger.info("=" * 60)

    conversations_dir = args.conversations_dir.resolve()

    if not conversations_dir.exists():
        logger.error(f"Conversations directory not found: {conversations_dir}")
        return 1

    logger.info(f"Migrating conversations in: {conversations_dir}")

    # Run migration
    report = migrate_all_conversations(conversations_dir, dry_run=args.dry_run)

    # Print summary
    report.print_summary()

    # Exit 0 - archiving incompatible conversations is expected behavior
    return 0


if __name__ == "__main__":
    sys.exit(main())
