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
    extract_memories_from_interaction,
    create_memory_container,
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

# from .dag_memory_manager import (
#     DagMemoryManager,
# )

from .action_dag_memory_manager import (
    DagMemoryManager,
)

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
    "extract_memories_from_interaction",
    "create_memory_container",
    "add_memory_container_to_graph",
    # Connection System
    "add_connections_to_graph",
    # Context Formatting
    "format_context",
    # Memory Managers
    "DagMemoryManager",
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
