"""
Experiment runner for executing experiments.

Orchestrates the execution of test cases × variants × runs and saves
all raw data to disk.
"""

import time
import logging
from datetime import datetime
from typing import (
    TypeVar,
    Generic,
    Sequence,
    Callable,
    Optional,
    Protocol,
)

from .base import TestCase
from .data import RunData, RunMetadata
from .storage import ExperimentStorage

logger = logging.getLogger(__name__)


class HasName(Protocol):
    """Protocol for objects that have a name() method."""

    def name(self) -> str: ...


# Type variable for variant interface (bounded to HasName)
TVariant = TypeVar("TVariant", bound=HasName)


class ExperimentRunner(Generic[TVariant]):
    """
    Orchestrates experiment execution.

    Runs variants × test cases × num_runs and saves all raw data to disk.
    Handles retries and error logging.

    Type parameters:
        TVariant: The variant interface type (e.g., StructuredOutputFormat)
    """

    def __init__(
        self,
        storage: ExperimentStorage,
        max_retries: int = 3,
        progress_callback: Optional[Callable[[str], None]] = None,
    ):
        """
        Initialize experiment runner.

        Args:
            storage: Storage instance for saving data
            max_retries: Maximum retry attempts on failure
            progress_callback: Optional callback for progress updates
        """
        self.storage = storage
        self.max_retries = max_retries
        self.progress_callback = progress_callback or self._default_progress

    def _default_progress(self, message: str) -> None:
        """Default progress callback that logs."""
        logger.info(message)

    def run_experiment(
        self,
        variants: Sequence[TVariant],
        test_cases: Sequence[TestCase[TVariant]],
        num_runs: int = 1,
    ) -> str:
        """
        Execute full experiment: variants × test cases × num_runs.

        Each test case executes itself using each variant.
        Supports heterogeneous types - each test case can have different data types.

        Args:
            variants: List of variants to test
            test_cases: List of test cases to run
            num_runs: Number of repetitions per variant/test case pair

        Returns:
            Experiment run timestamp (directory name)
        """
        # Generate timestamp for this experiment run
        run_ts = f"run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

        total_runs = len(variants) * len(test_cases) * num_runs
        self.progress_callback(
            f"Starting experiment: {len(variants)} variants × "
            f"{len(test_cases)} test cases × {num_runs} runs = "
            f"{total_runs} total executions"
        )

        # Execute all combinations
        completed = 0
        for variant in variants:
            variant_name = variant.name()
            self.progress_callback(f"\nTesting variant: {variant_name}")

            for test_case in test_cases:
                self.progress_callback(f"  Test case: {test_case.name()}")

                for run_index in range(num_runs):
                    self.progress_callback(f"    Run {run_index + 1}/{num_runs}")

                    # Execute single run with retries
                    run_data, metadata = self._execute_single_run(
                        variant=variant, test_case=test_case, run_index=run_index
                    )

                    # Save to disk
                    self.storage.save_run(run_data, metadata, run_ts)

                    # Log result
                    status = "✅" if metadata.success else "❌"
                    retry_count = metadata.retry_count()
                    retry_info = f", {retry_count} retries" if retry_count > 0 else ""
                    self.progress_callback(
                        f"      {status} {metadata.duration_seconds:.2f}s{retry_info}"
                    )

                    completed += 1

        self.progress_callback(
            f"\nExperiment complete! Saved to: {self.storage.base_dir / run_ts}"
        )
        return run_ts

    def _execute_single_run(
        self, variant: TVariant, test_case: TestCase[TVariant], run_index: int
    ) -> tuple[RunData, RunMetadata]:
        """
        Execute a single run with retry logic.

        Args:
            variant: Variant to use for execution
            test_case: Test case to run
            run_index: Index of this run

        Returns:
            Tuple of (RunData, RunMetadata)
        """
        start_time = time.time()
        expected_output = test_case.expected_output()
        errors = []

        # Get variant name (try name() method, name attribute, or str())
        variant_name = variant.name()

        # Retry loop
        for attempt in range(self.max_retries + 1):
            try:
                # Execute test case using variant
                output_data = test_case.execute(variant)

                # Success!
                duration = time.time() - start_time

                run_data = RunData(
                    variant_name=variant_name,
                    test_case_name=test_case.name(),
                    run_index=run_index,
                    output_data=output_data,
                    expected_output=expected_output,
                    timestamp=datetime.now(),
                )

                metadata = RunMetadata(
                    duration_seconds=duration,
                    success=True,
                )

                return run_data, metadata

            except Exception as e:
                errors.append(str(e))
                logger.warning(
                    f"Attempt {attempt + 1}/{self.max_retries + 1} failed: {errors[-1]}"
                )

        # All retries failed
        duration = time.time() - start_time
        logger.error(f"All {self.max_retries + 1} attempts failed. Errors: {errors}")

        # Create RunData with None output on failure
        run_data = RunData(
            variant_name=variant_name,
            test_case_name=test_case.name(),
            run_index=run_index,
            output_data=None,
            expected_output=expected_output,
            timestamp=datetime.now(),
        )

        metadata = RunMetadata(
            duration_seconds=duration,
            success=False,
            errors=errors,
        )

        return run_data, metadata
