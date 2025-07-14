"""
Progress Reporting - Explicit DI with Strong Typing

Lightweight progress reporting using explicit dependency injection.
Pass progress reporters to constructors, never Optional - always use null reporter.
Strong typing through Protocol interfaces, no hidden dependencies.
"""

from typing import Protocol, Optional
from contextlib import contextmanager
import logging

import rich
import rich.progress

logger = logging.getLogger(__name__)


class ProgressReporter(Protocol):
    """Strongly typed interface for progress reporting

    Clean contract with explicit method signatures for type safety.
    """

    def start_task(self, description: str, total: Optional[float] = None) -> str:
        """Start a new progress task

        Args:
            description: Human-readable task description
            total: Optional total units for completion (None = indeterminate)

        Returns:
            Unique task identifier for subsequent updates
        """
        ...

    def update_task(
        self, task_id: str, progress: float, description: Optional[str] = None
    ) -> None:
        """Update progress on an existing task

        Args:
            task_id: Task identifier from start_task
            progress: Progress value (0.0 to 1.0 for determinate, arbitrary for indeterminate)
            description: Optional updated description
        """
        ...

    def finish_task(self, task_id: str) -> None:
        """Mark a task as completed and clean up

        Args:
            task_id: Task identifier to finish
        """
        ...

    @contextmanager
    def task(self, description: str, total: Optional[float] = None):
        """Convenience method to create a managed task

        Usage:
            with progress.task("Processing", total=100) as task:
                task.update(0.5, "Halfway done")
        """
        task_id = self.start_task(description, total)
        try:
            yield TaskHandle(self, task_id)
        finally:
            self.finish_task(task_id)

    def input(self, prompt: str = "") -> str:
        """Get user input while progress is running

        Args:
            prompt: Text to display as input prompt

        Returns:
            User input string
        """
        ...

    def print(self, *args, **kwargs) -> None:
        """Print safely while progress is running

        This coordinates with the progress display to avoid visual artifacts.
        Use this instead of builtin print() when progress is active.

        Args:
            *args, **kwargs: Same as builtin print()
        """
        ...


class TaskHandle:
    """Task handle for updating progress"""

    def __init__(self, reporter: ProgressReporter, task_id: str):
        self._reporter = reporter
        self._task_id = task_id

    def update(self, progress: float, description: Optional[str] = None):
        """Update this task's progress"""
        self._reporter.update_task(self._task_id, progress, description)


class NullProgressReporter:
    """Null object pattern - no-op implementation

    Allows progress-aware code to work cleanly without conditionals.
    Always use this instead of Optional[ProgressReporter].
    """

    def start_task(self, description: str, total: Optional[float] = None) -> str:
        return "null"

    def update_task(
        self, task_id: str, progress: float, description: Optional[str] = None
    ) -> None:
        pass

    def finish_task(self, task_id: str) -> None:
        pass

    @contextmanager
    def task(self, description: str, total: Optional[float] = None):
        """Null implementation of task context manager"""
        task_id = self.start_task(description, total)
        try:
            yield TaskHandle(self, task_id)
        finally:
            self.finish_task(task_id)

    def input(self, prompt: str = "") -> str:
        """Null implementation of input - uses built-in input"""
        return input(prompt)

    def print(self, *args, **kwargs) -> None:
        """Null implementation of print - uses built-in print"""
        print(*args, **kwargs)


# Example Rich Progress implementation
class RichProgressReporter:
    """Implementation using Rich Progress for CLI applications"""

    def __init__(self, rich_progress: rich.progress.Progress):
        self._progress = rich_progress

    def start_task(self, description: str, total: Optional[float] = None) -> str:
        task_id = self._progress.add_task(description, total=total)
        logger.debug(f"Started task {task_id}: {description} (total={total})")
        return str(task_id)

    def update_task(
        self, task_id: str, progress: float, description: Optional[str] = None
    ) -> None:
        try:
            rich_task_id = int(task_id)

            # Check if task still exists
            if rich_task_id not in self._progress.task_ids:
                logger.warning(f"Task {task_id} no longer exists")
                return

            # Find the task object
            task = None
            for t in self._progress.tasks:
                if t.id == rich_task_id:
                    task = t
                    break

            if task is None:
                logger.warning(f"Could not find task object for ID {task_id}")
                return

            # Update description if provided
            if description:
                self._progress.update(rich_task_id, description=description)

            # Update progress
            if task.total:
                completed = progress * task.total
                self._progress.update(rich_task_id, completed=completed)
                logger.debug(
                    f"Updated task {task_id}: {progress:.2f} ({completed:.1f}/{task.total})"
                )
            else:
                # Indeterminate progress - description already updated above if provided
                logger.debug(f"Updated task {task_id}: {progress:.2f} (indeterminate)")

        except (ValueError, TypeError) as e:
            logger.error(f"Invalid task ID '{task_id}': {e}")
        except Exception as e:
            logger.error(f"Failed to update task {task_id}: {e}")
            logger.error(f"Available task IDs: {self._progress.task_ids}")

    def finish_task(self, task_id: str) -> None:
        try:
            rich_task_id = int(task_id)

            # Check if task still exists
            if rich_task_id not in self._progress.task_ids:
                logger.warning(f"Task {task_id} no longer exists")
                return

            self._progress.remove_task(rich_task_id)
            logger.debug(f"Finished task {task_id}")

        except (ValueError, TypeError) as e:
            logger.error(f"Invalid task ID '{task_id}': {e}")
        except Exception as e:
            logger.error(f"Failed to finish task {task_id}: {e}")
            logger.error(f"Available task IDs: {self._progress.task_ids}")

    @contextmanager
    def task(self, description: str, total: Optional[float] = None):
        """Rich implementation of task context manager"""
        task_id = self.start_task(description, total)
        try:
            yield TaskHandle(self, task_id)
        finally:
            self.finish_task(task_id)

    def input(self, prompt: str = "") -> str:
        """Get user input by temporarily pausing Rich Progress display"""
        # Based on Rich GitHub issue #1535 - temporarily pause the live display
        was_started = (
            self._progress.live.is_started if hasattr(self._progress, "live") else True
        )
        logger.debug(f"Getting user input, progress was_started: {was_started}")

        original_transient = None
        if was_started:
            # Set transient to avoid display issues (recommended by Rich maintainers)
            if hasattr(self._progress, "live"):
                original_transient = self._progress.live.transient
                self._progress.live.transient = True
                logger.debug(f"Set transient mode, original: {original_transient}")

            self._progress.stop()
            logger.debug("Stopped progress display for input")

        try:
            # Use regular input since progress is paused
            logger.debug(f"Prompting user: {prompt[:50]}...")
            result = input(prompt)
            logger.debug(f"User input received: {len(result)} characters")
            return result
        finally:
            if was_started:
                self._progress.start()
                logger.debug("Restarted progress display")
                # Restore original transient setting
                if hasattr(self._progress, "live") and original_transient is not None:
                    self._progress.live.transient = original_transient
                    logger.debug(f"Restored transient mode: {original_transient}")

    def print(self, *args, **kwargs) -> None:
        """Print safely using Rich's console to avoid display artifacts"""
        # Rich Progress has a console that coordinates with the live display
        if hasattr(self._progress, "console"):
            self._progress.console.print(*args, **kwargs)
        else:
            # Fallback to builtin print if no console available
            print(*args, **kwargs)


# Example usage patterns:


# 1. Constructor injection (recommended)
class MyService:
    def __init__(self, progress: ProgressReporter):
        self.progress = progress

    def do_work(self):
        with self.progress.task("Doing work", total=100) as task:
            for i in range(100):
                task.update(i / 100, f"Step {i}")
                # ... work


# 2. CLI setup
def cli_main():
    from rich.progress import Progress

    with Progress() as rich_progress:
        progress_reporter = RichProgressReporter(rich_progress)
        service = MyService(progress_reporter)
        service.do_work()


# 3. No progress needed
def no_progress_main():
    service = MyService(NullProgressReporter())
    service.do_work()
