"""
Reusable experiment framework for running and analyzing experiments.

This framework provides generic, type-safe abstractions for:
- Defining variants to test
- Defining test cases with inputs and expected outputs
- Running experiments (variants × test cases × runs)
- Saving all raw data to disk
- Calculating metrics from saved data
- Analyzing and comparing results

Key principle: Save all raw data, compute metrics from saved data.
This allows recalculating metrics without re-running expensive experiments.
"""

from .base import TestCase, MetricsCalculator
from .data import RunData, RunMetadata
from .storage import ExperimentStorage
from .runner import ExperimentRunner
from .analysis import ExperimentAnalyzer

__all__ = [
    "TestCase",
    "MetricsCalculator",
    "RunData",
    "RunMetadata",
    "ExperimentStorage",
    "ExperimentRunner",
    "ExperimentAnalyzer",
]
