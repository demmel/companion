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
    MemoryEdgeType,
    ConfidenceLevel,
)

from .memory_formation import (
    create_context_element,
    extract_memories_from_interaction,
    create_memory_container,
    add_memory_container_to_graph,
)

from .connection_system import (
    build_connection_prompt,
    decide_connections_llm,
    add_connections_to_graph,
    detect_similar_memories,
)


from .io import (
    save_dag_to_json,
    load_dag_from_json,
)


from .context_formatting import (
    format_context,
)

__all__ = [
    # Models
    "ContextElement",
    "ContextGraph",
    "MemoryContainer",
    "MemoryEdge",
    "MemoryGraph",
    "MemoryEdgeType",
    "ConfidenceLevel",
    # Memory Formation
    "create_context_element",
    "extract_memories_from_interaction",
    "create_memory_container",
    "add_memory_container_to_graph",
    # Connection System
    "build_connection_prompt",
    "decide_connections_llm",
    "add_connections_to_graph",
    "detect_similar_memories",
    # I/O
    "save_dag_to_json",
    "load_dag_from_json",
    # Context Formatting
    "format_context",
]
