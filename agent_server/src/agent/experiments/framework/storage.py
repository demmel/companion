"""
Storage layer for experiment framework.

Handles saving and loading experiment data to/from disk with proper
directory structure and JSON serialization.
"""

from pathlib import Path
from typing import Tuple, List
from pydantic import BaseModel

from .data import RunData, RunMetadata


class ExperimentStorage:
    """
    Handles saving and loading experiment data.

    Manages the directory structure:
        run_TIMESTAMP/
            variant_NAME/
                testcase_NAME/
                    run_0/
                        data.json
                        metadata.json
                    run_1/
                        ...

    Uses JSON serialization with Pydantic model_dump()/model_validate().
    Supports heterogeneous types - type information is embedded in the data.
    """

    def __init__(self, base_dir: Path):
        """
        Initialize storage.

        Args:
            base_dir: Base directory for all experiments
        """
        self.base_dir = Path(base_dir)

    def save_run[TOutput: BaseModel, TExpected: BaseModel](
        self,
        run_data: RunData[TOutput, TExpected],
        metadata: RunMetadata,
        run_ts: str,
    ) -> Path:
        """
        Save a single run to disk as JSON.

        Args:
            run_data: The run data to save
            metadata: The run metadata
            run_ts: Timestamp identifier for this experiment run

        Returns:
            Path to the run directory
        """
        # Create directory structure
        run_dir = self.get_run_path(
            run_ts,
            run_data.variant_name,
            run_data.test_case_name,
            run_data.run_index,
        )
        run_dir.mkdir(parents=True, exist_ok=True)

        # Save data using custom model_dump() + json.dumps()
        with open(run_dir / "data.json", "w") as f:
            f.write(run_data.model_dump_json(indent=2))

        # Save metadata using Pydantic's model_dump()
        with open(run_dir / "metadata.json", "w") as f:
            f.write(metadata.model_dump_json(indent=2))

        return run_dir

    def load_run[TOutput: BaseModel, TExpected: BaseModel](
        self, run_ts: str, variant_name: str, test_case_name: str, run_index: int
    ) -> Tuple[RunData[TOutput, TExpected], RunMetadata]:
        """
        Load a single run from disk.

        Uses embedded type metadata to correctly deserialize typed fields.

        Args:
            run_path: Path to the run directory

        Returns:
            Tuple of (RunData, RunMetadata)
        """
        # Load raw JSON
        run_dir = self.get_run_path(run_ts, variant_name, test_case_name, run_index)

        with open(run_dir / "data.json", "r") as f:
            run_data = RunData[TOutput, TExpected].model_validate_json(f.read())

        # Load metadata
        with open(run_dir / "metadata.json", "r") as f:
            metadata = RunMetadata.model_validate_json(f.read())

        return run_data, metadata

    def get_run_path(
        self, run_ts: str, variant_name: str, test_case_name: str, run_index: int
    ) -> Path:
        """
        Get the path to a specific run directory.

        Args:
            run_ts: Timestamp identifier for this experiment run
            variant_name: Name of the variant
            test_case_name: Name of the test case
            run_index: Index of the run

        Returns:
            Path to the run directory
        """
        return (
            self.base_dir
            / run_ts
            / f"variant_{variant_name}"
            / f"testcase_{test_case_name}"
            / f"run_{run_index}"
        )

    def list_experiment_runs(self) -> List[str]:
        """
        List all experiment run timestamps in base directory.

        Returns:
            List of run_TIMESTAMP directory names
        """
        if not self.base_dir.exists():
            return []

        return [
            d.name
            for d in self.base_dir.iterdir()
            if d.is_dir() and d.name.startswith("run_")
        ]

    def list_variants(self, run_ts: str) -> List[str]:
        """List all variant names in an experiment run."""
        run_dir = self.base_dir / run_ts
        if not run_dir.exists():
            return []

        return [
            d.name.replace("variant_", "")
            for d in run_dir.iterdir()
            if d.is_dir() and d.name.startswith("variant_")
        ]

    def list_test_cases(self, run_ts: str, variant_name: str) -> List[str]:
        """List all test case names for a variant in an experiment run."""
        variant_dir = self.base_dir / run_ts / f"variant_{variant_name}"
        if not variant_dir.exists():
            return []

        return [
            d.name.replace("testcase_", "")
            for d in variant_dir.iterdir()
            if d.is_dir() and d.name.startswith("testcase_")
        ]

    def list_runs(
        self, run_ts: str, variant_name: str, test_case_name: str
    ) -> List[int]:
        """List all run indices for a variant/test case combination."""
        testcase_dir = (
            self.base_dir
            / run_ts
            / f"variant_{variant_name}"
            / f"testcase_{test_case_name}"
        )
        if not testcase_dir.exists():
            return []

        return sorted(
            [
                int(d.name.replace("run_", ""))
                for d in testcase_dir.iterdir()
                if d.is_dir() and d.name.startswith("run_")
            ]
        )
