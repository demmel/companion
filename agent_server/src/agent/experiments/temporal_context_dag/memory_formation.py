"""
Functions for creating memory elements from agent interactions.
"""

import uuid
import logging
from datetime import datetime
from typing import List

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
) -> List[ContextElement]:
    """
    Use agent/LLM to intelligently extract significant memory elements from an interaction.

    Args:
        trigger: The trigger history entry containing trigger and actions
        llm: LLM instance for memory extraction
        model: Model to use for extraction decisions

    Returns:
        List of memory elements that the agent deemed significant
    """
    from agent.structured_llm import direct_structured_llm_call
    from pydantic import BaseModel, Field

    # Build context of the interaction for the agent to analyze
    interaction_context = format_single_trigger_entry(trigger)

    # Build prompt with existing context if available
    prompt = f"""I'm {state.name}, {state.role}. I need to analyze this interaction I just had and extract the most significant memories worth preserving.

    {build_agent_state_description(state)}

{format_section("MY WORKING MEMORY", format_context(context))}

{format_section("WHAT JUST HAPPENED", interaction_context)}

## Task:
I will review this interaction and identify the most significant facts, events, insights, or changes that are worth remembering.

I will consider:
- Important information about the user or situation
- Significant decisions or changes I made
- Key insights or realizations
- Important emotional or relational moments
- Facts that might be relevant for future interactions
- Avoid creating memories for things I already remember (see existing context above)

For each significant memory, provide:
1. The specific content worth remembering (be concise but complete)
2. Why this is significant (emotional_significance as a score 0.0-1.0)
3. How many tokens this memory should initially receive (10-100 based on importance)

I will only extract memories that are genuinely significant."""

    class MemoryExtraction(BaseModel):
        reasoning: str = Field(
            description="My reasoning for why this memory is significant"
        )
        content: str = Field(description="The content of the memory")
        emotional_significance: float = Field(
            description="Emotional significance of the memory", ge=0.0, le=1.0
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

        # Convert to MemoryElement objects
        memories = []
        for extraction in response.memories:
            memory = create_context_element(
                content=extraction.content,
                timestamp=trigger.timestamp,
                emotional_significance=extraction.emotional_significance,
                initial_tokens=int(extraction.emotional_significance * 100),
            )
            memories.append(memory)

        logger.info(
            f"  Extracted {len(memories)} significant memories from interaction {trigger.entry_id}"
        )
        return memories

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
