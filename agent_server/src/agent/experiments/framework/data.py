"""
Data models for experiment framework.

Defines the structure of raw data and metadata saved during experiments.
"""

from datetime import datetime
from typing import Optional, Dict
from pydantic import BaseModel


class RunData(BaseModel):
    """
    Raw data from a single experimental run.

    This contains all the input, output, and expected data for one execution
    of a test case using a variant. Supports heterogeneous types - each test
    case can have completely different data types.

    Type information is preserved through metadata fields for correct deserialization.
    """
    model_config = {"arbitrary_types_allowed": True}

    variant_name: str
    test_case_name: str
    run_index: int

    # Data fields (actual types vary by test case)
    output_data: Optional[BaseModel]  # None if execution failed
    expected_output: Optional[BaseModel]

    # Type metadata for deserialization
    output_type_module: Optional[str]  # e.g., "agent.experiments.autonomous_research.extraction"
    output_type_name: Optional[str]    # e.g., "ExtractionResponse"
    expected_type_module: Optional[str]
    expected_type_name: Optional[str]

    timestamp: datetime

    def model_dump(self, **kwargs):
        """Custom serialization to handle BaseModel fields correctly."""
        # Exclude the BaseModel fields from super().model_dump()
        exclude = {"output_data", "expected_output"}
        data = super().model_dump(exclude=exclude, **kwargs)

        # Serialize BaseModel fields manually
        if self.output_data is not None:
            data["output_data"] = self.output_data.model_dump()
        else:
            data["output_data"] = None

        if self.expected_output is not None:
            data["expected_output"] = self.expected_output.model_dump()
        else:
            data["expected_output"] = None

        return data

    @classmethod
    def create(
        cls,
        variant_name: str,
        test_case_name: str,
        run_index: int,
        output_data: Optional[BaseModel],
        expected_output: Optional[BaseModel],
        timestamp: datetime,
    ) -> "RunData":
        """
        Create RunData with automatic type metadata extraction.

        Args:
            variant_name: Name of the variant
            test_case_name: Name of the test case
            run_index: Index of this run
            output_data: Output from execution (None if failed)
            expected_output: Expected output for comparison (None if not available)
            timestamp: When this run occurred

        Returns:
            RunData with type metadata automatically populated
        """
        return cls(
            variant_name=variant_name,
            test_case_name=test_case_name,
            run_index=run_index,
            output_data=output_data,
            expected_output=expected_output,
            output_type_module=type(output_data).__module__ if output_data else None,
            output_type_name=type(output_data).__name__ if output_data else None,
            expected_type_module=type(expected_output).__module__ if expected_output else None,
            expected_type_name=type(expected_output).__name__ if expected_output else None,
            timestamp=timestamp,
        )


class RunMetadata(BaseModel):
    """
    Execution metadata for a single run.

    Contains information about how the run executed (duration, success,
    errors, retries, etc.) but not the actual data.
    """

    duration_seconds: float
    success: bool
    error_message: Optional[str] = None
    retry_count: int = 0


class TestCaseMetrics(BaseModel):
    """Aggregated metrics for a single test case across multiple runs."""

    total_runs: int
    successful_runs: int
    success_rate: float
    mean_duration: float
    mean_retries: float
    metrics: Dict[str, float]  # Dynamic metrics from MetricsCalculator


class VariantResults(BaseModel):
    """Results for a single variant across all test cases."""

    test_cases: Dict[str, TestCaseMetrics]


class ExperimentResults(BaseModel):
    """Complete experiment results across all variants and test cases."""

    run_ts: str
    variants: Dict[str, VariantResults]
