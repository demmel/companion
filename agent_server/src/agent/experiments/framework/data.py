"""
Data models for experiment framework.

Defines the structure of raw data and metadata saved during experiments.
"""

from datetime import datetime
from typing import Generic, Iterable, List, Optional, Dict, Sequence, TypeVar
from pydantic import BaseModel, Field

TOutput = TypeVar("TOutput", bound=BaseModel)
TExpected = TypeVar("TExpected", bound=BaseModel)


class RunData(BaseModel, Generic[TOutput, TExpected]):
    """
    Raw data from a single experimental run.

    This contains all the input, output, and expected data for one execution
    of a test case using a variant. Supports heterogeneous types - each test
    case can have completely different data types.
    """

    variant_name: str
    test_case_name: str
    run_index: int

    # Data fields (actual types vary by test case)
    output_data: Optional[TOutput]  # None if execution failed
    expected_output: Optional[TExpected]

    timestamp: datetime


class RunMetadata(BaseModel):
    """
    Execution metadata for a single run.

    Contains information about how the run executed (duration, success,
    errors, retries, etc.) but not the actual data.
    """

    duration_seconds: float
    success: bool
    errors: List[str] = Field(default_factory=list)

    def retry_count(self) -> int:
        """Get the number of retries attempted."""
        return len(self.errors)


class Metric(BaseModel):
    """A single metric value for a run."""

    mean: float
    stddev: float
    min: float
    max: float
    n: int  # Sample size

    @staticmethod
    def from_values(values: Sequence[float]) -> "Metric":
        """Create Metric from a list of values."""
        import statistics

        return Metric(
            mean=statistics.mean(values),
            stddev=statistics.stdev(values) if len(values) > 1 else 0.0,
            min=min(values),
            max=max(values),
            n=len(values),
        )

    def confidence_interval(self, confidence: float = 0.95) -> tuple[float, float]:
        """
        Calculate confidence interval from summary statistics.

        Args:
            confidence: Confidence level (default 0.95 for 95% CI)

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        from scipy import stats
        import numpy as np

        # Handle cases with no variance or insufficient data
        if self.n < 2 or self.stddev < 1e-10:
            return (self.mean, self.mean)

        # Calculate standard error
        std_err = self.stddev / np.sqrt(self.n)

        try:
            ci = stats.t.interval(confidence, self.n - 1, loc=self.mean, scale=std_err)
            lower = float(ci[0])
            upper = float(ci[1])

            # Check for NaN or infinite results
            if np.isnan(lower) or np.isnan(upper) or np.isinf(lower) or np.isinf(upper):
                return (self.mean, self.mean)

            return (lower, upper)
        except Exception:
            # Fallback to just the mean if CI calculation fails
            return (self.mean, self.mean)


class TestCaseMetrics(BaseModel):
    """Aggregated metrics for a single test case across multiple runs."""

    total_runs: int
    successful_runs: int
    success_rate: float
    duration: Metric
    retries: Metric
    metrics: Dict[
        str, Metric
    ]  # Dynamic per-run metrics from MetricsCalculator (aggregated from all runs, failed runs = 0.0)
    comparative_metrics: Dict[str, float] = (
        {}
    )  # Comparative metrics across variants (not aggregated)


class VariantResults(BaseModel):
    """Results for a single variant across all test cases."""

    test_cases: Dict[str, TestCaseMetrics]


class ExperimentResults(BaseModel):
    """Complete experiment results across all variants and test cases."""

    run_ts: str
    variants: Dict[str, VariantResults]
