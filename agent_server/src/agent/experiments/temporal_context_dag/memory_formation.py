"""
Functions for creating memory elements from agent interactions.
"""

import uuid
import logging
from datetime import datetime
from typing import List
from enum import Enum

from agent.chain_of_action.prompts import format_section, format_single_trigger_entry
from agent.chain_of_action.trigger_history import TriggerHistoryEntry
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
    MemoryEdgeType,
    MemoryElement,
)
from .connection_system import decide_connections_llm
from .context_formatting import format_context


def create_context_element(
    content: str,
    timestamp: datetime,
    emotional_significance: float,
    initial_tokens: int,
) -> ContextElement:
    """Create a new memory element with given content and metadata."""
    return ContextElement(
        memory=MemoryElement(
            id=str(uuid.uuid4()),
            content=content,
            timestamp=timestamp,
            emotional_significance=emotional_significance,
        ),
        tokens=initial_tokens,
    )


def extract_memories_from_interaction(
    trigger: TriggerHistoryEntry,
    state: State,
    context: ContextGraph,
    llm: LLM,
    model: SupportedModel,
) -> tuple[List[ContextElement], List[MemoryEdge]]:
    """
    Use agent/LLM to intelligently extract significant memory elements from an interaction
    along with their connections to existing memories.

    Args:
        trigger: The trigger history entry containing trigger and actions
        state: Current agent state
        context: Current context graph with existing memories
        llm: LLM instance for memory extraction
        model: Model to use for extraction decisions

    Returns:
        Tuple of (memory_elements, connections) where connections are from existing memories to new memories
    """
    from agent.structured_llm import direct_structured_llm_call
    from pydantic import BaseModel, Field

    # Build context of the interaction for the agent to analyze
    interaction_context = format_single_trigger_entry(trigger)

    # Build context memories information for connection formation
    context_memories_info = ""
    if context.elements:
        context_memories_info = "\n".join(
            [f"- [{elem.memory.id}] {elem.memory.content}" for elem in context.elements]
        )
    else:
        context_memories_info = "No existing memories in current context."

    # Build prompt with existing context if available
    prompt = f"""I'm {state.name}, {state.role}. I need to analyze this interaction I just had and extract the most significant memories worth preserving, along with their connections to my existing memories.

    {build_agent_state_description(state)}

{format_section("MY EXISTING MEMORIES IN CONTEXT", context_memories_info)}

{format_section("WHAT JUST HAPPENED", interaction_context)}

## Task:
I will review this interaction and identify the most significant facts, events, insights, or changes that are worth remembering. For each memory, I will also identify which of my existing memories should connect to it.

I will consider:
- Important information about the user or situation
- Significant decisions or changes I made
- Key insights or realizations
- Important emotional or relational moments
- Facts that might be relevant for future interactions
- Vivid details, specific dialogue, and emotional nuances that capture the essence of the moment
- Avoid creating memories for things I already remember (see existing context above)

For each significant memory, I will provide:
1. Rich, detailed content that captures the specific essence, emotions, and vivid details of what happened (preserve the flavor and personality of the interaction)
2. Why this is significant (emotional_significance as a score 0.0-1.0)
3. Which existing memories (if any) should connect TO this new memory, along with:
   - The reasoning for each connection
   - The type of relationship (relates_to, explains, follows, or updates)
   - ONLY use memory IDs from the "MY EXISTING MEMORIES IN CONTEXT" section above

Connection types:
- relates_to: General semantic connection between memories that share concepts or themes
- explains: An existing memory explains, provides reasoning for, or gives context to this new memory
- follows: An existing memory is a direct chronological predecessor or leads into this new memory
- updates: This new memory contains information that supersedes, refines, or builds upon an existing memory

I will only extract memories that are genuinely significant and create connections that are meaningful and clear.
{f"Since there are no existing memories in context, I will NOT create any connections." if not context.elements else ""}"""

    class ConnectionType(str, Enum):
        RELATES_TO = "relates_to"
        EXPLAINS = "explains"
        FOLLOWS = "follows"
        UPDATES = "updates"

    class ConnectionFromExisting(BaseModel):
        """Connection from an existing memory to this new memory."""

        reasoning: str = Field(
            description="My reasoning for why this existing memory should connect to this new memory"
        )
        source_memory_id: str = Field(
            description="ID of the existing memory that should connect to this new memory"
        )
        edge_type: ConnectionType = Field(
            description="Type of connection from the existing memory to this new memory"
        )

    class MemoryExtraction(BaseModel):
        reasoning: str = Field(
            description="My reasoning for why this memory is significant"
        )
        content: str = Field(description="The content of the memory")
        emotional_significance: float = Field(
            description="Emotional significance of the memory", ge=0.0, le=1.0
        )
        connections_from_existing: List[ConnectionFromExisting] = Field(
            default_factory=list,
            description="List of existing memories that should connect to this new memory",
        )

    class MemoryExtractions(BaseModel):
        memories: List[MemoryExtraction] = Field(
            description="List of significant memories extracted"
        )

    try:
        response = direct_structured_llm_call(
            prompt=prompt,
            response_model=MemoryExtractions,
            model=model,
            llm=llm,
            caller="memory_extraction",
        )

        # Convert to MemoryElement objects and extract connections
        memories = []
        connections = []

        for extraction in response.memories:
            memory = create_context_element(
                content=extraction.content,
                timestamp=trigger.timestamp,
                emotional_significance=extraction.emotional_significance,
                initial_tokens=int(extraction.emotional_significance * 100),
            )
            memories.append(memory)

            # Extract connections from existing memories to this new memory
            for connection in extraction.connections_from_existing:
                edge = MemoryEdge(
                    source_id=connection.source_memory_id,
                    target_id=memory.memory.id,
                    edge_type=MemoryEdgeType(connection.edge_type.value),
                )
                connections.append(edge)

        logger.info(
            f"  Extracted {len(memories)} significant memories and {len(connections)} connections from interaction {trigger.entry_id}"
        )
        return memories, connections

    except Exception as e:
        logger.warning(
            f"  Memory extraction failed for interaction {trigger.entry_id}: {e}"
        )
        raise e


def create_memory_container(
    trigger: TriggerHistoryEntry,
    element_ids: List[str],
) -> MemoryContainer:
    """Create a memory container grouping elements from an interaction."""
    return MemoryContainer(
        trigger=trigger,
        element_ids=element_ids,
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


def create_intelligent_connections(
    graph: MemoryGraph,
    context: ContextGraph,
    state: State,
    new_container: MemoryContainer,
    llm: LLM,
    model: SupportedModel,
) -> List[MemoryEdge]:
    """
    Create intelligent connections between new memories and existing context.

    Args:
        graph: Memory graph to update
        new_container: Container with new memories to connect
        llm: LLM instance for connection decisions
        model: Model to use for connection decisions
        context_budget: Token budget for selecting context memories

    Returns:
        Updated memory graph with new connections
    """
    # Get new memories from the container
    new_memories = [graph.elements[elem_id] for elem_id in new_container.element_ids]
    context_memories = context.elements

    if not new_memories or not context_memories:
        logger.info(
            f"  Skipping connections: new_memories={len(new_memories)}, context_memories={len(context_memories)}"
        )
        return []

    logger.info(
        f"  Making LLM call with {len(new_memories)} new memories and {len(context_memories)} context memories"
    )

    # Use LLM to decide connections
    connections = decide_connections_llm(
        state=state,
        new_memories=new_memories,
        context_memories=[cm.memory for cm in context_memories],
        llm=llm,
        model=model,
    )

    logger.info(f"  LLM returned {len(connections)} connections")

    return connections
