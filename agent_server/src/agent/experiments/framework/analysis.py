"""
Experiment analyzer for calculating metrics from saved data.

Loads raw experiment data and calculates metrics on-demand,
allowing recalculation without re-running expensive experiments.
"""

import logging
from typing import Dict, List
import statistics

from .base import MetricsCalculator
from .data import RunMetadata, TestCaseMetrics, VariantResults, ExperimentResults
from .storage import ExperimentStorage

logger = logging.getLogger(__name__)


class ExperimentAnalyzer:
    """
    Analyzes saved experiment data.

    Loads raw data from disk and calculates metrics using provided
    MetricsCalculator. Allows recalculating metrics without re-running
    experiments.

    Works with heterogeneous experiments - each test case may have
    different data types. Type deserialization is handled automatically
    by the storage layer.
    """

    def __init__(self, storage: ExperimentStorage):
        """
        Initialize analyzer.

        Args:
            storage: Storage instance for loading data
        """
        self.storage = storage

    def calculate_metrics(
        self, run_ts: str, calculator: MetricsCalculator
    ) -> ExperimentResults:
        """
        Calculate metrics for all runs in an experiment.

        Works with heterogeneous types - each test case may have different
        input/output types. Storage handles deserialization automatically.

        Args:
            run_ts: Experiment run timestamp
            calculator: MetricsCalculator to use (handles BaseModel polymorphically)

        Returns:
            ExperimentResults with typed structure
        """
        variant_results = {}
        variants = self.storage.list_variants(run_ts)

        for variant_name in variants:
            test_case_metrics = {}
            test_cases = self.storage.list_test_cases(run_ts, variant_name)

            for test_case_name in test_cases:
                # Load all runs for this variant/test case
                run_indices = self.storage.list_runs(
                    run_ts, variant_name, test_case_name
                )

                run_metrics_list = []
                metadata_list = []

                for run_index in run_indices:
                    run_path = (
                        self.storage.base_dir
                        / run_ts
                        / f"variant_{variant_name}"
                        / f"testcase_{test_case_name}"
                        / f"run_{run_index}"
                    )

                    # Load run - type deserialization handled automatically
                    run_data, metadata = self.storage.load_run(run_path)

                    metadata_list.append(metadata)

                    # Calculate metrics if run was successful
                    if metadata.success and run_data.output_data is not None:
                        metrics = calculator.calculate(
                            run_data.output_data, run_data.expected_output
                        )
                        run_metrics_list.append(metrics)

                # Aggregate metrics across runs
                test_case_metrics[test_case_name] = self._aggregate_metrics(
                    run_metrics_list, metadata_list
                )

            variant_results[variant_name] = VariantResults(test_cases=test_case_metrics)

        return ExperimentResults(run_ts=run_ts, variants=variant_results)

    def _aggregate_metrics(
        self, run_metrics_list: List[Dict[str, float]], metadata_list: List[RunMetadata]
    ) -> TestCaseMetrics:
        """
        Aggregate metrics across multiple runs.

        Args:
            run_metrics_list: List of metric dicts from each run
            metadata_list: List of metadata from each run

        Returns:
            Aggregated metrics as TestCaseMetrics
        """
        total_runs = len(metadata_list)
        successful_runs = sum(1 for m in metadata_list if m.success)

        metrics: Dict[str, float] = {}

        # Aggregate each metric across runs
        if run_metrics_list:
            metric_names = run_metrics_list[0].keys()

            for metric_name in metric_names:
                values = [m[metric_name] for m in run_metrics_list]

                metrics[f"{metric_name}_mean"] = statistics.mean(values)
                metrics[f"{metric_name}_std"] = (
                    statistics.stdev(values) if len(values) > 1 else 0.0
                )
                metrics[f"{metric_name}_min"] = min(values)
                metrics[f"{metric_name}_max"] = max(values)

        return TestCaseMetrics(
            total_runs=total_runs,
            successful_runs=successful_runs,
            success_rate=successful_runs / total_runs if total_runs > 0 else 0.0,
            mean_duration=(
                statistics.mean(m.duration_seconds for m in metadata_list)
                if metadata_list
                else 0.0
            ),
            mean_retries=(
                statistics.mean(m.retry_count for m in metadata_list)
                if metadata_list
                else 0.0
            ),
            metrics=metrics,
        )

    def generate_report(self, results: ExperimentResults) -> str:
        """
        Generate human-readable text report.

        Args:
            results: Results from calculate_metrics()

        Returns:
            Formatted report string
        """
        lines = []
        lines.append("=" * 80)
        lines.append("EXPERIMENT RESULTS")
        lines.append("=" * 80)
        lines.append(f"\nExperiment: {results.run_ts}")
        lines.append(f"Variants: {len(results.variants)}")

        # Overall summary
        lines.append("\n" + "-" * 80)
        lines.append("OVERALL SUMMARY")
        lines.append("-" * 80)

        for variant_name, variant_results in results.variants.items():
            lines.append(f"\n{variant_name}:")
            for test_case_name, test_metrics in variant_results.test_cases.items():
                lines.append(f"  {test_case_name}:")
                lines.append(f"    Success rate: {test_metrics.success_rate:.1%}")

                # Show metric means if available
                for key, value in test_metrics.metrics.items():
                    if key.endswith("_mean") and not key.startswith("mean_"):
                        metric_name = key.replace("_mean", "")
                        std = test_metrics.metrics.get(f"{metric_name}_std", 0.0)
                        lines.append(f"    {metric_name}: {value:.3f} ± {std:.3f}")

        lines.append("\n" + "=" * 80)
        return "\n".join(lines)

    def compare_variants(
        self,
        results: ExperimentResults,
        metric_name: str = "f1_mean",
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare variants on a specific metric.

        Args:
            results: Results from calculate_metrics()
            metric_name: Name of metric to compare

        Returns:
            Dict mapping variant_name -> test_case_name -> metric_value
        """
        comparison = {}

        for variant_name, variant_results in results.variants.items():
            comparison[variant_name] = {}
            for test_case_name, test_metrics in variant_results.test_cases.items():
                comparison[variant_name][test_case_name] = test_metrics.metrics.get(
                    metric_name, 0.0
                )

        return comparison
