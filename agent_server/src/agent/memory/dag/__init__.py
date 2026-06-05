"""
DAG-Based Memory System Experiment

A proof of concept implementation of a Directed Acyclic Graph (DAG) based
memory system for agent context management.
"""

from .models import (
    ContextElement,
    ContextGraph,
    MemoryContainer,
    MemoryEdge,
    MemoryGraph,
    GraphEdgeType,
    ConfidenceLevel,
)

from .memory_formation import (
    create_context_element,
    add_memory_container_to_graph,
)

from .connection_system import (
    add_connections_to_graph,
)


from .context_formatting import (
    format_context,
)

# Action-based components
from .actions import (
    MemoryAction,
    AddMemoryAction,
    AddEdgeAction,
    UpdateConfidenceAction,
    AddToContextAction,
    AddEdgeToContextAction,
    RemoveFromContextAction,
    AddContainerAction,
    CheckpointAction,
)

from .action_log import (
    MemoryActionLog,
)

# NOTE: DagMemoryManager (the implementation) is intentionally NOT re-exported here. It
# imports agent.memory.memory (the IMemory interface), so eagerly importing it from this
# package __init__ would make `import agent.memory.dag.<anything>` drag the manager — and
# therefore memory.memory — in. That blocks memory.memory from importing the pure dag data
# containers (e.g. MemoryAction). Import it directly: `from
# agent.memory.dag.dag_memory_manager import DagMemoryManager`.

__all__ = [
    # Models
    "ContextElement",
    "ContextGraph",
    "MemoryContainer",
    "MemoryEdge",
    "MemoryGraph",
    "GraphEdgeType",
    "ConfidenceLevel",
    # Memory Formation
    "create_context_element",
    "add_memory_container_to_graph",
    # Connection System
    "add_connections_to_graph",
    # Context Formatting
    "format_context",
    # Action-based components
    "MemoryAction",
    "AddMemoryAction",
    "AddEdgeAction",
    "UpdateConfidenceAction",
    "AddToContextAction",
    "AddEdgeToContextAction",
    "RemoveFromContextAction",
    "AddContainerAction",
    "CheckpointAction",
    "MemoryActionLog",
]
