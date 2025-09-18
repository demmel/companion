"""
LLM-driven system for deciding connections between memory elements.
"""

from enum import Enum
import json
from typing import List
import logging
import time

from agent.chain_of_action.prompts import format_section
from agent.state import (
    State,
    build_agent_state_description,
    build_agent_state_description,
)
from agent.structured_llm import direct_structured_llm_call
from agent.llm import LLM, SupportedModel
from pydantic import Field

from .models import (
    MemoryElement,
    MemoryGraph,
    MemoryEdge,
    MemoryEdgeType,
)

logger = logging.getLogger(__name__)


def build_connection_prompt(
    state: State,
    new_memories: List[MemoryElement],
    context_memories: List[MemoryElement],
) -> str:
    """
    Build prompt for LLM to decide connections between new and existing memories.

    Args:
        new_memories: Memory elements from current interaction
        context_memories: Memory elements currently in context (within token budget)

    Returns:
        Formatted prompt for connection decision
    """
    prompt = f"""I'm {state.name}, {state.role}. I need to decide which of my new memories should connect to my existing memories in context.

{build_agent_state_description(state)}

{format_section("MY EXISTING MEMORIES", "\n".join(f"- [{m.id}] {m.content}" for m in context_memories))}

{format_section("MY NEW MEMORIES", "\n".join(f"- [{m.id}] {m.content}" for m in new_memories))}

## Task:
I will decide which existing memories in context should connect TO my new memories. Connections must respect temporal order: older memories can connect to newer memories, but not the reverse.

I will only create connections that are meaningful and clear. If no clear relationship exists, I will not include any connection.

IMPORTANT: I will only return connections that I actually want to create. I will not return empty connections with blank target_memory_id or edge_type fields.

For each connection I want to make, I will provide:
1. My reasoning for why these memories should be connected
2. Which existing memory (source) connects to which new memory (target) - use exact memory IDs
3. What type of relationship it is

Connection direction: EXISTING_MEMORY → NEW_MEMORY (older to newer)

Remember: I can only connect existing memories to new memories. I won't create connections between new memories (those happen automatically within the interaction).
"""

    return prompt


def decide_connections_llm(
    state: State,
    new_memories: List[MemoryElement],
    context_memories: List[MemoryElement],
    llm: LLM,
    model: SupportedModel,
) -> List[MemoryEdge]:
    """
    Use LLM to decide which connections to create between new and existing memories.

    Args:
        new_memories: Memory elements from current interaction
        context_memories: Memory elements currently in context
        llm: LLM instance for making the call
        model: Model to use for connection decisions

    Returns:
        List of connection decisions from the agent
    """
    if not new_memories or not context_memories:
        return []

    start_time = time.time()
    prompt = build_connection_prompt(state, new_memories, context_memories)

    try:
        # Since direct_structured_llm_call expects a single model, not List[Model],
        # we need a wrapper model for multiple connections
        from pydantic import BaseModel

        class ConnectionType(str, Enum):
            RELATES_TO = "relates_to"
            EXPLAINS = "explains"

        class ConnectionDecision(BaseModel):
            """Agent's decision about connecting two memory elements."""

            reasoning: str = Field(
                description="My reasoning for why these memories should be connected"
            )
            source_memory_id: str = Field(
                description="ID of the new memory to connect from"
            )
            target_memory_id: str = Field(description="ID of the memory to connect to")
            edge_type: ConnectionType = Field(
                ...,
                description="Type of connection",
                json_schema_extra={
                    "enum": [
                        {
                            "value": ConnectionType.RELATES_TO,
                            "description": "General semantic connection between memories that share concepts or themes",
                        },
                        {
                            "value": ConnectionType.EXPLAINS,
                            "description": "One memory explains, provides reasoning for, or gives context to another",
                        },
                    ]
                },
            )
            initial_tokens: int = Field(
                description="Initial token count for this connection", ge=10, le=100
            )

        class ConnectionDecisions(BaseModel):
            connections: List[ConnectionDecision]

        response = direct_structured_llm_call(
            prompt=prompt,
            response_model=ConnectionDecisions,
            model=model,
            llm=llm,
            caller="dag_connection_decision",
            temperature=0.3,
        )

        elapsed = time.time() - start_time
        connections = response.connections if response else []
        logger.info(
            f"  Connection decision completed in {elapsed:.3f}s - found {len(connections)} connections"
        )
        return [
            MemoryEdge(
                source_id=conn.source_memory_id,
                target_id=conn.target_memory_id,
                edge_type=MemoryEdgeType(conn.edge_type.value),
            )
            for conn in connections
        ]
    except Exception as e:
        elapsed = time.time() - start_time
        logger.warning(f"  Connection decision failed in {elapsed:.3f}s: {e}")
        return []


def add_connections_to_graph(
    graph: MemoryGraph, connections: List[MemoryEdge]
) -> MemoryGraph:
    """
    Add agent-decided connections to the memory graph.

    Validates edge constraints before adding.
    """
    added_count = 0
    for i, connection in enumerate(connections):
        logger.info(
            f"  Processing connection {i+1}: {connection.edge_type} from {connection.source_id[:8]} to {connection.target_id[:8]}"
        )

        # Validate that both memories exist in graph
        if (
            connection.source_id not in graph.elements
            or connection.target_id not in graph.elements
        ):
            logger.warning(f"  Skipping connection {i+1}: memory not found in graph")
            continue

        # Validate edge type is agent-controlled
        agent_controlled_types = {MemoryEdgeType.RELATES_TO, MemoryEdgeType.EXPLAINS}
        if connection.edge_type not in agent_controlled_types:
            logger.warning(
                f"  Skipping connection {i+1}: edge type {connection.edge_type} not in agent-controlled set"
            )
            continue

        # Validate temporal constraints
        if not _validate_edge_temporal_constraints(
            graph, connection.source_id, connection.target_id, connection.edge_type
        ):
            logger.warning(
                f"  Skipping connection {i+1}: temporal constraint violation"
            )
            continue

        # Create and add edge
        edge = MemoryEdge(
            source_id=connection.source_id,
            target_id=connection.target_id,
            edge_type=connection.edge_type,
        )
        graph.edges.append(edge)
        added_count += 1
        logger.info(f"  Successfully added connection {i+1}")

    logger.info(f"  Added {added_count} out of {len(connections)} connections to graph")
    return graph


def _validate_edge_temporal_constraints(
    graph: MemoryGraph, source_id: str, target_id: str, edge_type: MemoryEdgeType
) -> bool:
    """
    Validate that edge follows temporal constraints for inter-container connections.

    Inter-container edges must go forward in time only.
    Intra-container edges are always allowed.
    """
    # Find containers for source and target
    source_container = None
    target_container = None

    for container in graph.containers.values():
        if source_id in container.element_ids:
            source_container = container
        if target_id in container.element_ids:
            target_container = container

    # If same container, always allow (intra-container edge)
    if (
        source_container
        and target_container
        and source_container.trigger.entry_id == target_container.trigger.entry_id
    ):
        return True

    # If different containers, must be forward in time
    if source_container and target_container:
        return source_container.trigger.timestamp < target_container.trigger.timestamp

    # If containers not found, allow (shouldn't happen in normal operation)
    return True


def detect_similar_memories(
    graph: MemoryGraph, similarity_threshold: float = 0.8
) -> List[tuple[str, str]]:
    """
    Detect semantically similar memories using embedding similarity.

    Args:
        graph: Memory graph to analyze
        similarity_threshold: Minimum similarity to consider for connection

    Returns:
        List of (memory_id_1, memory_id_2) pairs that are similar
    """
    from agent.memory.embedding_service import get_embedding_service

    embedding_service = get_embedding_service()
    similar_pairs = []

    # Get all memories that don't have embeddings yet
    memories_to_embed = []
    for memory in graph.elements.values():
        if memory.embedding_vector is None:
            memories_to_embed.append(memory)

    # Generate embeddings in batch for efficiency
    if memories_to_embed:
        logger.info(f"  Generating embeddings for {len(memories_to_embed)} memories")
        texts = [memory.content for memory in memories_to_embed]
        embeddings = embedding_service.encode_batch(texts)

        # Update memories with their embeddings
        for memory, embedding in zip(memories_to_embed, embeddings):
            memory.embedding_vector = embedding

    # Find similar pairs by comparing all embeddings
    memories_with_embeddings = [
        m for m in graph.elements.values() if m.embedding_vector is not None
    ]

    for i, memory1 in enumerate(memories_with_embeddings):
        for memory2 in memories_with_embeddings[i + 1 :]:
            assert memory1.embedding_vector is not None, "Embedding 1 should be set"
            assert memory2.embedding_vector is not None, "Embedding 2 should be set"

            similarity = embedding_service.cosine_similarity(
                memory1.embedding_vector, memory2.embedding_vector
            )

            if similarity >= similarity_threshold:
                similar_pairs.append((memory1.id, memory2.id))

    logger.info(
        f"  Found {len(similar_pairs)} similar memory pairs above threshold {similarity_threshold}"
    )
    return similar_pairs
