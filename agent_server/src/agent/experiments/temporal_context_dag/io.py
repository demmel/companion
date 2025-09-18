"""
Parser for converting baseline_triggers.json data into DAG memory format.
"""

import logging


from .models import MemoryGraph

logger = logging.getLogger(__name__)


def save_dag_to_json(graph: MemoryGraph, output_path: str):
    """Save memory graph to JSON file for persistence."""
    with open(output_path, "w") as f:
        f.write(graph.model_dump_json(indent=2))


def load_dag_from_json(input_path: str) -> MemoryGraph:
    """Load memory graph from JSON file."""
    with open(input_path, "r") as f:
        data = MemoryGraph.model_validate_json(f.read())
    return data
