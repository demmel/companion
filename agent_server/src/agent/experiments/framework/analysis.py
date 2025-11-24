"""
Experiment analyzer for calculating metrics from saved data.

Loads raw experiment data and calculates metrics on-demand,
allowing recalculation without re-running expensive experiments.
"""

import logging
from typing import Dict, List
from anthropic import BaseModel

from .base import MetricsCalculator
from .data import (
    Metric,
    RunMetadata,
    TestCaseMetrics,
    VariantResults,
    ExperimentResults,
)
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
        # Extract type information from calculator's generic parameters
        import typing
        from pydantic import BaseModel

        output_type = BaseModel
        expected_type = BaseModel

        orig_bases = getattr(type(calculator), "__orig_bases__", None)
        if orig_bases:
            args = typing.get_args(orig_bases[0])
            if len(args) >= 2:
                output_type = args[0]
                expected_type = args[1]

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
                metric_names = None  # Track metric names from first successful run

                for run_index in run_indices:
                    # Load run with type information from calculator
                    run_data, metadata = self.storage.load_run_with_types(
                        run_ts,
                        variant_name,
                        test_case_name,
                        run_index,
                        output_type,
                        expected_type,
                    )

                    metadata_list.append(metadata)

                    # Calculate metrics if run was successful
                    if metadata.success and run_data.output_data is not None:
                        metrics = calculator.calculate(
                            run_data.output_data, run_data.expected_output
                        )
                        if metric_names is None:
                            metric_names = set(metrics.keys())
                        run_metrics_list.append(metrics)
                    else:
                        # Failed run - will backfill with 0.0 later
                        run_metrics_list.append(None)

                # Backfill failed runs with 0.0 for all metrics
                if metric_names:
                    for i, metrics in enumerate(run_metrics_list):
                        if metrics is None:
                            run_metrics_list[i] = {name: 0.0 for name in metric_names}
                else:
                    # All runs failed - no metrics to calculate
                    run_metrics_list = []

                # Aggregate metrics across runs
                test_case_metrics[test_case_name] = self._aggregate_metrics(
                    run_metrics_list, metadata_list
                )

            variant_results[variant_name] = VariantResults(test_cases=test_case_metrics)

        # Calculate comparative metrics if supported by calculator
        self._calculate_comparative_metrics(run_ts, calculator, variant_results)

        return ExperimentResults(run_ts=run_ts, variants=variant_results)

    def _calculate_comparative_metrics(
        self,
        run_ts: str,
        calculator: MetricsCalculator,
        variant_results: Dict[str, VariantResults],
    ) -> None:
        """
        Calculate comparative metrics across variants for each test case.

        Modifies variant_results in-place by adding comparative_metrics to each TestCaseMetrics.

        Args:
            run_ts: Experiment run timestamp
            calculator: MetricsCalculator to use
            variant_results: Dict mapping variant_name -> VariantResults (modified in-place)
        """
        # Find all unique test case names
        all_test_cases = set()
        for variant_result in variant_results.values():
            all_test_cases.update(variant_result.test_cases.keys())

        # For each test case, collect outputs from all variants and calculate comparative metrics
        for test_case_name in all_test_cases:
            # Collect all outputs for this test case across all variants
            variant_outputs: Dict[str, List[BaseModel]] = {}

            for variant_name in variant_results.keys():
                # Check if this variant has this test case
                if test_case_name not in variant_results[variant_name].test_cases:
                    continue

                # Load all outputs for this variant/test case
                run_indices = self.storage.list_runs(
                    run_ts, variant_name, test_case_name
                )
                outputs = []

                for run_index in run_indices:
                    run_data, metadata = self.storage.load_run(
                        run_ts, variant_name, test_case_name, run_index
                    )

                    if metadata.success and run_data.output_data is not None:
                        outputs.append(run_data.output_data)

                if outputs:
                    variant_outputs[variant_name] = outputs

            # Calculate comparative metrics for this test case
            if len(variant_outputs) >= 2:  # Need at least 2 variants to compare
                comparative_metrics = calculator.calculate_comparative(
                    test_case_name, variant_outputs
                )
                # comparative_metrics is Dict[metric_name, Dict[variant_name, score]]
                # Example: {"richness": {"json": 0.9, "xml": 0.7}}

                # Store comparative metrics back into each variant's TestCaseMetrics
                for variant_name in variant_outputs.keys():
                    if test_case_name in variant_results[variant_name].test_cases:
                        # Extract this variant's scores across all comparative metrics
                        variant_comparative_scores = {}
                        for metric_name, variant_scores in comparative_metrics.items():
                            if variant_name in variant_scores:
                                variant_comparative_scores[metric_name] = (
                                    variant_scores[variant_name]
                                )

                        # Store in comparative_metrics dict
                        variant_results[variant_name].test_cases[
                            test_case_name
                        ].comparative_metrics = variant_comparative_scores

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

        metrics: Dict[str, Metric] = {}

        # Aggregate each metric across runs
        if run_metrics_list:
            metric_names = run_metrics_list[0].keys()

            for metric_name in metric_names:
                values = [m[metric_name] for m in run_metrics_list]
                metrics[metric_name] = Metric.from_values(values)

        return TestCaseMetrics(
            total_runs=total_runs,
            successful_runs=successful_runs,
            success_rate=successful_runs / total_runs if total_runs > 0 else 0.0,
            duration=Metric.from_values([m.duration_seconds for m in metadata_list]),
            retries=Metric.from_values([m.retry_count() for m in metadata_list]),
            metrics=metrics,
        )

    def generate_report(self, results: ExperimentResults) -> str:
        """
        Generate simple human-readable text report.

        For detailed comparative analysis, use generate_comparison_report() from reports module.

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
                for key, metric in test_metrics.metrics.items():
                    lines.append(f"    {key}: {metric.mean:.3f} ± {metric.stddev:.3f}")

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
                metric = test_metrics.metrics.get(metric_name)
                comparison[variant_name][test_case_name] = (
                    metric.mean if metric else 0.0
                )

        return comparison
