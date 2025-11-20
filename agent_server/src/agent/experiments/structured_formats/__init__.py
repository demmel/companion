"""
Structured Output Format Experiment

Tests different output formats (JSON, XML, YAML, etc.) to find which works best
with local LLMs for structured data extraction.

Usage:
    python -m agent.experiments.structured_formats.run_experiment_framework --num-runs 15
"""

from .base_format import StructuredOutputFormat
from .formats import JSONFormat, XMLFormat, YAMLFormat, SExpFormat
from .test_cases import ALL_TEST_CASES

__all__ = [
    "StructuredOutputFormat",
    "JSONFormat",
    "XMLFormat",
    "YAMLFormat",
    "SExpFormat",
    "ALL_TEST_CASES",
]
