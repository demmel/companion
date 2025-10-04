"""
Functions for creating memory elements from agent interactions.
"""

import uuid
import logging
from datetime import datetime
from typing import List

from agent.chain_of_action.trigger_history import TriggerHistoryEntry
from agent.memory.actions import (
    AddContainerAction,
    AddEdgeAction,
    AddEdgeToContextAction,
    AddMemoryAction,
    AddToContextAction,
    MemoryAction,
)
from agent.memory.context_formatting import format_element
from agent.memory.edge_types import (
    AgentControlledEdgeType,
    GraphEdgeType,
    get_prompt_edge_type_list,
    get_edge_type_memory_formation_descriptions,
)
from agent.llm import LLM, SupportedModel
from agent.state import State, build_agent_state_description

logger = logging.getLogger(__name__)

from .models import (
    ContextElement,
    ContextGraph,
    ContextElement,
    MemoryContainer,
    MemoryEdge,
    MemoryGraph,
    MemoryElement,
    ConfidenceLevel,
)
from .token_allocation import get_creation_tokens


def create_context_element(
    container_id: str,
    content: str,
    timestamp: datetime,
    initial_tokens: int,
    confidence_level: ConfidenceLevel,
    sequence_in_container: int,
) -> ContextElement:
    """Create a new memory element with given content and metadata."""
    from agent.embedding_service import get_embedding_service

    # Generate embedding for the memory
    embedding_service = get_embedding_service()
    embedding_text = f"{content}"
    embedding_vector = embedding_service.encode(embedding_text)

    return ContextElement(
        memory=MemoryElement(
            id=str(uuid.uuid4()),
            container_id=container_id,
            content=content,
            timestamp=timestamp,
            confidence_level=confidence_level,
            sequence_in_container=sequence_in_container,
            embedding_vector=embedding_vector,
        ),
        tokens=initial_tokens,
    )


def add_memory_container_to_graph(
    graph: MemoryGraph, container: MemoryContainer, memories: List[MemoryElement]
) -> MemoryGraph:
    """
    Add a memory container and its elements to the graph.

    Creates temporal "follows" edges to previous container if one exists.
    """
    # Add all memory elements to graph
    for memory in memories:
        graph.elements[memory.id] = memory

    # No automatic follows edges - let intelligent connection system handle all edges

    # Add container to graph
    graph.containers[container.trigger.entry_id] = container

    return graph


def extract_memories_as_actions(
    trigger: TriggerHistoryEntry,
    state: State,
    context: ContextGraph,
    llm: LLM,
    model: SupportedModel,
    memory_graph: MemoryGraph,
) -> List[MemoryAction]:
    """
    Extract memories from an interaction and return them as actions.

    Uses the existing memory extraction logic but returns actions instead of
    directly modifying the graph state.

    Args:
        trigger: The trigger history entry containing trigger and actions
        state: Current agent state
        context: Current context graph with existing memories
        llm: LLM instance for memory extraction
        model: Model to use for extraction decisions
        memory_graph: Memory graph to access containers for compressed display

    Returns:
        List of actions to apply for this interaction
    """

    from agent.memory.memory_formation_automatic import (
        create_memories_from_trigger_entry,
    )

    memories = [
        ContextElement(memory=m, tokens=get_creation_tokens())
        for m in create_memories_from_trigger_entry(trigger)
    ]
    connections = extract_semantic_connections(
        new_memories=memories,
        context=context,
        trigger_entry=trigger,
        state=state,
        llm=llm,
        model=model,
        memory_graph=memory_graph,
    )

    actions = []

    # Create actions for adding memories to the graph
    for memory in memories:
        actions.append(AddMemoryAction(memory=memory.memory))

    # Create actions for adding memories to context
    for memory in memories:
        actions.append(
            AddToContextAction(memory_id=memory.memory.id, initial_tokens=memory.tokens)
        )

    # Create actions for adding connections to graph
    for connection in connections:
        actions.append(AddEdgeAction(edge=connection))

    # Create actions for adding connections to context
    for connection in connections:
        actions.append(
            AddEdgeToContextAction(
                edge_id=connection.id, should_boost_source_tokens=True
            )
        )

    # Create action for adding the container
    if memories:
        actions.append(
            AddContainerAction(
                container_id=trigger.entry_id,
                element_ids=[m.memory.id for m in memories],
                trigger_timestamp=trigger.timestamp,
            )
        )

    logger.info(
        f"Generated {len(actions)} actions for interaction {trigger.entry_id}: "
        f"{len(memories)} memories, {len(connections)} connections"
    )

    return actions


def extract_semantic_connections(
    new_memories: List[ContextElement],
    context: ContextGraph,
    trigger_entry: TriggerHistoryEntry,
    state: State,
    llm: LLM,
    model: SupportedModel,
    memory_graph: MemoryGraph,
) -> List[MemoryEdge]:
    """
    LLM analyzes semantic relationships between new memories and existing context.
    Returns ONLY edges, no new memories (memories are created automatically beforehand).

    Args:
        new_memories: Auto-created memories from trigger+actions
        context: Current context graph with existing memories
        trigger_entry: The trigger entry (for situational_context)
        state: Current agent state
        llm: LLM instance for connection analysis
        model: Model to use for analysis
        memory_graph: Optional memory graph to access containers for compressed display

    Returns:
        List of MemoryEdge objects representing semantic connections
    """

    if not new_memories or not context.elements:
        return []

    from agent.structured_llm import direct_structured_llm_call
    from pydantic import BaseModel, Field
    from agent.memory.context_formatting import format_context
    from agent.chain_of_action.prompts import format_section

    # Format new memories chronologically for the prompt
    new_memories_text = "\n\n".join(
        [
            format_element(
                element=element,
                forward_edges=[],
                backward_edges=[],
            )
            for element in sorted(
                new_memories, key=lambda m: m.memory.sequence_in_container
            )
        ]
    )

    dag_context_text = (
        format_context(context, memory_graph, use_individual_formatting=True)
        if context.elements
        else "No existing memories in current context."
    )

    prompt = f"""I'm {state.name}, {state.role}. I just had an experience and formed memories of what happened. Now I need to identify how these new memories connect to my existing memories and to each other.

{build_agent_state_description(state)}

{format_section("MY EXISTING MEMORIES IN CONTEXT", dag_context_text)}

{format_section("MY INITIAL ANALYSIS OF THE SITUATION", trigger_entry.situational_context)}

{format_section("WHAT ACTUALLY HAPPENED (NEW MEMORIES)", new_memories_text)}

## My Task:
I will identify semantic relationships between these memories. For each meaningful connection, I'll determine:
- Which existing memory connects to which new memory (or which new memories connect to each other)
- What type of relationship it is ({get_prompt_edge_type_list()})
- Why this connection is meaningful

{get_edge_type_memory_formation_descriptions()}

I will only create connections that are genuinely meaningful and clear.
{f"Since there are no existing memories in context, I will NOT create connections to existing memories." if not context.elements else ""}"""

    class ConnectionFromExisting(BaseModel):
        """Connection from an existing memory to a new memory."""

        reasoning: str = Field(
            description="My reasoning for why this existing memory should connect to this new memory"
        )
        source_memory_id: str = Field(
            description="ID of the existing memory that should connect to this new memory"
        )
        target_memory_id: str = Field(
            description="ID of the new memory being connected to"
        )
        edge_type: AgentControlledEdgeType = Field(
            description="Type of connection from the existing memory to this new memory"
        )

    class IntraConnection(BaseModel):
        """Connection between new memories within this same interaction."""

        reasoning: str = Field(
            description="My reasoning for why these memories should be connected"
        )
        source_memory_id: str = Field(description="ID of the source new memory")
        target_memory_id: str = Field(description="ID of the target new memory")
        edge_type: AgentControlledEdgeType = Field(
            description="Type of connection between these memories"
        )

    class SemanticConnections(BaseModel):
        connections_to_context: List[ConnectionFromExisting] = Field(
            default_factory=list,
            description="Connections from existing context memories to new memories",
        )
        intra_connections: List[IntraConnection] = Field(
            default_factory=list,
            description="Connections between the new memories themselves",
        )

    try:
        response = direct_structured_llm_call(
            prompt=prompt,
            response_model=SemanticConnections,
            model=model,
            llm=llm,
            caller="semantic_connection_extraction",
        )

        edges = []
        edge_signatures = set()

        # Create mapping of new memory IDs (may be truncated in LLM response)
        new_memory_ids = {mem.memory.id: mem for mem in new_memories}
        new_memory_ids_by_prefix = {
            mem.memory.id[:8]: mem.memory.id for mem in new_memories
        }

        # Create set of valid existing memory IDs
        valid_existing_ids = {elem.memory.id for elem in context.elements}
        valid_existing_ids_by_prefix = {mid[:8]: mid for mid in valid_existing_ids}

        # Process connections from existing memories to new memories
        for connection in response.connections_to_context:
            # Resolve source ID (existing memory)
            source_id = connection.source_memory_id
            if source_id not in valid_existing_ids:
                # Try prefix match
                if source_id in valid_existing_ids_by_prefix:
                    source_id = valid_existing_ids_by_prefix[source_id]
                else:
                    logger.warning(
                        f"Invalid existing memory ID '{source_id}' not found in context"
                    )
                    continue

            # Resolve target ID (new memory)
            target_id = connection.target_memory_id
            if target_id not in new_memory_ids:
                # Try prefix match
                if target_id in new_memory_ids_by_prefix:
                    target_id = new_memory_ids_by_prefix[target_id]
                else:
                    logger.warning(
                        f"Invalid new memory ID '{target_id}' not found in new memories"
                    )
                    continue

            # Create edge if not duplicate
            edge_type = GraphEdgeType(connection.edge_type.value)
            edge_signature = (source_id, edge_type, target_id)

            if edge_signature not in edge_signatures:
                edges.append(
                    MemoryEdge(
                        source_id=source_id,
                        target_id=target_id,
                        edge_type=edge_type,
                    )
                )
                edge_signatures.add(edge_signature)

        # Process intra-connections between new memories
        for connection in response.intra_connections:
            # Resolve both IDs (both are new memories)
            source_id = connection.source_memory_id
            if source_id not in new_memory_ids:
                if source_id in new_memory_ids_by_prefix:
                    source_id = new_memory_ids_by_prefix[source_id]
                else:
                    logger.warning(
                        f"Invalid new memory ID '{source_id}' in intra-connection"
                    )
                    continue

            target_id = connection.target_memory_id
            if target_id not in new_memory_ids:
                if target_id in new_memory_ids_by_prefix:
                    target_id = new_memory_ids_by_prefix[target_id]
                else:
                    logger.warning(
                        f"Invalid new memory ID '{target_id}' in intra-connection"
                    )
                    continue

            # Create edge if not duplicate
            edge_type = GraphEdgeType(connection.edge_type.value)
            edge_signature = (source_id, edge_type, target_id)

            if edge_signature not in edge_signatures:
                edges.append(
                    MemoryEdge(
                        source_id=source_id,
                        target_id=target_id,
                        edge_type=edge_type,
                    )
                )
                edge_signatures.add(edge_signature)

        logger.info(
            f"Extracted {len(edges)} semantic connections for interaction {trigger_entry.entry_id}: "
            f"{len(response.connections_to_context)} to context, "
            f"{len(response.intra_connections)} intra-connections"
        )

        return edges

    except Exception as e:
        logger.warning(
            f"Semantic connection extraction failed for interaction {trigger_entry.entry_id}: {e}"
        )
        return []
