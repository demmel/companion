#!/usr/bin/env python3
"""
Test script for the DAG Memory Evolution Visualizer.

Creates a simple test action log and runs basic functionality tests.
"""

import sys
import json
import tempfile
from pathlib import Path
from datetime import datetime

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from agent.experiments.temporal_context_dag.models import MemoryGraph, ContextGraph
from agent.experiments.temporal_context_dag.actions import (
    CheckpointAction,
    AddMemoryAction,
    AddToContextAction,
    AddEdgeAction,
)
from agent.experiments.temporal_context_dag.models import MemoryElement, MemoryEdge
from agent.experiments.temporal_context_dag.memory_types import MemoryType
from agent.experiments.temporal_context_dag.edge_types import GraphEdgeType
from agent.experiments.temporal_context_dag.models import ConfidenceLevel
from agent.experiments.temporal_context_dag.action_log import MemoryActionLog
from agent.experiments.temporal_context_dag.visualizer.action_processor import (
    StepwiseGraphReconstructor,
)
from agent.experiments.temporal_context_dag.visualizer.graph_extractor import (
    GraphExtractor,
)


def create_test_action_log():
    """Create a simple test action log with a few memories and connections."""
    action_log = MemoryActionLog()

    # Add checkpoint
    action_log.add_checkpoint("start", "Test session started")

    # Create some test memories
    memory1 = MemoryElement(
        id="mem1",
        content="I am an AI assistant designed to help users with various tasks",
        evidence="System initialization message",
        timestamp=datetime.now(),
        emotional_significance=0.7,
        confidence_level=ConfidenceLevel.USER_CONFIRMED,
        memory_type=MemoryType.IDENTITY,
        embedding_vector=[0.1] * 768,  # Dummy embedding
    )

    memory2 = MemoryElement(
        id="mem2",
        content="User asked about the weather in San Francisco",
        evidence="User query: 'What's the weather like in SF?'",
        timestamp=datetime.now(),
        emotional_significance=0.3,
        confidence_level=ConfidenceLevel.STRONG_INFERENCE,
        memory_type=MemoryType.FACTUAL,
        embedding_vector=[0.2] * 768,
    )

    memory3 = MemoryElement(
        id="mem3",
        content="I committed to always be helpful and honest in my responses",
        evidence="Core system directive",
        timestamp=datetime.now(),
        emotional_significance=0.9,
        confidence_level=ConfidenceLevel.USER_CONFIRMED,
        memory_type=MemoryType.COMMITMENT,
        embedding_vector=[0.3] * 768,
    )

    # Add memory actions
    action_log.add_action(AddMemoryAction(memory=memory1))
    action_log.add_action(AddToContextAction(memory_id="mem1", initial_tokens=60))

    action_log.add_action(AddMemoryAction(memory=memory2))
    action_log.add_action(AddToContextAction(memory_id="mem2", initial_tokens=30))

    action_log.add_checkpoint("mid", "Added initial memories")

    action_log.add_action(AddMemoryAction(memory=memory3))
    action_log.add_action(AddToContextAction(memory_id="mem3", initial_tokens=80))

    # Add a connection
    edge = MemoryEdge(
        source_id="mem1", target_id="mem3", edge_type=GraphEdgeType.EXPLAINS
    )
    action_log.add_action(AddEdgeAction(edge=edge))

    action_log.add_checkpoint("end", "Test session completed")

    return action_log


def test_action_processor():
    """Test the StepwiseGraphReconstructor."""
    print("Testing StepwiseGraphReconstructor...")

    # Create test action log
    action_log = create_test_action_log()

    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write(action_log.model_dump_json(indent=2))
        temp_path = f.name

    try:
        # Test processor
        processor = StepwiseGraphReconstructor()
        processor.load(temp_path)

        step_count = processor.get_step_count()
        print(f"  Loaded {step_count} steps")

        # Test stepping through
        for i in range(min(3, step_count)):
            state = processor.set_step(i)
            if state:
                print(
                    f"  Step {i}: {state.action.action_type} - {len(state.memory_graph.elements)} memories"
                )

        # Test checkpoints
        checkpoints = processor.get_checkpoints()
        print(f"  Found {len(checkpoints)} checkpoints")
        for step_idx, checkpoint in checkpoints:
            print(f"    Step {step_idx}: {checkpoint.label}")

        print("  ✓ StepwiseGraphReconstructor test passed")

    finally:
        # Clean up temp file
        Path(temp_path).unlink()


def test_graph_extractor():
    """Test the GraphExtractor."""
    print("Testing GraphExtractor...")

    # Create test action log
    action_log = create_test_action_log()

    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write(action_log.model_dump_json(indent=2))
        temp_path = f.name

    try:
        # Test processor and extractor together
        processor = StepwiseGraphReconstructor()
        processor.load(temp_path)
        extractor = GraphExtractor()

        # Test extraction on a few steps
        for i in range(min(3, processor.get_step_count())):
            state = processor.set_step(i)
            if state:
                viz_data = extractor.extract_visualization_data(state)
                print(
                    f"  Step {i}: {len(viz_data.nodes)} nodes, {len(viz_data.edges)} edges"
                )
                print(f"    Action: {viz_data.action_description}")

        # Test legend
        legend = extractor.get_legend_data()
        print(f"  Legend has {len(legend['memory_types'])} memory types")
        print(f"  Legend has {len(legend['edge_types'])} edge types")

        print("  ✓ GraphExtractor test passed")

    finally:
        # Clean up temp file
        Path(temp_path).unlink()


def main():
    """Run all tests."""
    print("DAG Memory Evolution Visualizer Tests")
    print("=" * 40)

    try:
        test_action_processor()
        test_graph_extractor()

        print("\n✅ All tests passed!")
        print("\nTo run the web interface:")
        print(
            "  uv run python src/agent/experiments/temporal_context_dag/visualizer/run_visualizer.py"
        )

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
