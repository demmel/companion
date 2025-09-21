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
from agent.experiments.temporal_context_dag.edge_types import (
    AgentControlledEdgeType,
    GraphEdgeType,
    get_prompt_type_list,
    get_memory_formation_descriptions,
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


def create_context_element(
    content: str,
    evidence: str,
    timestamp: datetime,
    emotional_significance: float,
    initial_tokens: int,
    confidence_level: ConfidenceLevel,
) -> ContextElement:
    """Create a new memory element with given content and metadata."""
    return ContextElement(
        memory=MemoryElement(
            id=str(uuid.uuid4()),
            content=content,
            evidence=evidence,
            timestamp=timestamp,
            emotional_significance=emotional_significance,
            confidence_level=confidence_level,
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

I will NOT create memories for:
- Routine procedural actions like "waiting for a response" or "observing reactions"
- Low-value operational details that don't add meaningful information
- Generic statements about planning future actions
- Repetitive patterns that I've already captured in other memories

For each significant memory, I will provide:
1. A simple identifier (e.g., M1, M2, M3) for referencing within this interaction
2. Content: A rich, descriptive summary that captures what happened - the key insight, event, or fact being remembered
3. Evidence: Exact quotes, specific dialogue, concrete facts, or precise details that prove this memory happened - the raw evidence
4. Why this is significant (emotional_significance as a score 0.0-1.0)
5. Which existing memories (if any) should connect TO this new memory, along with:
   - The reasoning for each connection
   - The type of relationship ({get_prompt_type_list()})
   - ONLY use memory IDs from the "MY EXISTING MEMORIES IN CONTEXT" section above

I can also create connections between the new memories I'm forming by listing intra_connections that ONLY reference the memory IDs I assigned for this interaction (M1, M2, etc.). These intra_connections should NOT reference any existing memory IDs from the context above.

{get_memory_formation_descriptions()}

I will only extract memories that are genuinely significant and create connections that are meaningful and clear.
{f"Since there are no existing memories in context, I will NOT create any connections." if not context.elements else ""}"""

    class ConfidenceLevelType(str, Enum):
        USER_CONFIRMED = "user_confirmed"
        STRONG_INFERENCE = "strong_inference"
        REASONABLE_ASSUMPTION = "reasonable_assumption"
        SPECULATIVE = "speculative"

    class ConnectionFromExisting(BaseModel):
        """Connection from an existing memory to this new memory."""

        reasoning: str = Field(
            description="My reasoning for why this existing memory should connect to this new memory"
        )
        source_memory_id: str = Field(
            description="ID of the existing memory that should connect to this new memory"
        )
        edge_type: AgentControlledEdgeType = Field(
            description="Type of connection from the existing memory to this new memory"
        )

    class MemoryExtraction(BaseModel):
        id: str = Field(
            description="Identifier for this memory (e.g., 'M1', 'M2', 'M3')"
        )
        reasoning: str = Field(
            description="My reasoning for why this memory is significant"
        )
        content: str = Field(
            description="A rich, descriptive summary of what happened - the key insight, event, or fact being remembered"
        )
        evidence: str = Field(
            description="Exact quotes, specific dialogue, concrete facts, or precise details that prove this memory - the raw evidence"
        )
        emotional_significance: float = Field(
            description="Emotional significance of the memory", ge=0.0, le=1.0
        )
        confidence_level: ConfidenceLevelType = Field(
            description="How confident I am in this memory's accuracy",
            json_schema_extra={
                "enum": [
                    {
                        "value": ConfidenceLevelType.USER_CONFIRMED,
                        "description": "Direct user statements/confirmations - highest reliability",
                    },
                    {
                        "value": ConfidenceLevelType.STRONG_INFERENCE,
                        "description": "High-confidence deductions from user input or my own verified actions",
                    },
                    {
                        "value": ConfidenceLevelType.REASONABLE_ASSUMPTION,
                        "description": "Logical assumptions that could be wrong but seem reasonable",
                    },
                    {
                        "value": ConfidenceLevelType.SPECULATIVE,
                        "description": "Uncertain inferences, especially about external world - use with caution",
                    },
                ]
            },
        )
        connections_from_existing: List[ConnectionFromExisting] = Field(
            default_factory=list,
            description="List of existing memories that should connect to this new memory",
        )

    class IntraInteractionConnection(BaseModel):
        """Connection between memories within this same interaction."""

        reasoning: str = Field(
            description="My reasoning for why these memories should be connected"
        )
        source_memory_id: str = Field(
            description="ID of the source memory (from the memories being created)"
        )
        target_memory_id: str = Field(
            description="ID of the target memory (from the memories being created)"
        )
        edge_type: AgentControlledEdgeType = Field(
            description="Type of connection between these memories"
        )

    class MemoryExtractions(BaseModel):
        memories: List[MemoryExtraction] = Field(
            description="List of significant memories extracted"
        )
        intra_connections: List[IntraInteractionConnection] = Field(
            default_factory=list,
            description="Connections between memories within this same interaction",
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

        # Create mapping from agent-assigned IDs to actual memory IDs
        id_mapping = {}

        for extraction in response.memories:
            # Convert LLM confidence level to ConfidenceLevel enum
            confidence_level = ConfidenceLevel(extraction.confidence_level.value)

            memory = create_context_element(
                content=extraction.content,
                evidence=extraction.evidence,
                timestamp=trigger.timestamp,
                emotional_significance=extraction.emotional_significance,
                initial_tokens=int(extraction.emotional_significance * 100),
                confidence_level=confidence_level,
            )
            memories.append(memory)

            # Map agent-assigned ID to actual memory ID
            id_mapping[extraction.id] = memory.memory.id

            # Extract connections from existing memories to this new memory
            for connection in extraction.connections_from_existing:
                edge = MemoryEdge(
                    source_id=connection.source_memory_id,
                    target_id=memory.memory.id,
                    edge_type=GraphEdgeType(connection.edge_type.value),
                )
                connections.append(edge)

        # Extract intra-interaction connections between new memories
        for intra_connection in response.intra_connections:
            if (
                intra_connection.source_memory_id in id_mapping
                and intra_connection.target_memory_id in id_mapping
            ):
                edge = MemoryEdge(
                    source_id=id_mapping[intra_connection.source_memory_id],
                    target_id=id_mapping[intra_connection.target_memory_id],
                    edge_type=GraphEdgeType(intra_connection.edge_type.value),
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
