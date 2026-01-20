"""
LLM-based topic name and keyword generation.

Three approaches:
- Simple: Direct question about shared topic
- Structured: Extract themes first, then name
- Contrastive: Define by what makes cluster different
"""

import logging
from typing import List
from pydantic import BaseModel, Field

from agent.llm import LLM, SupportedModel
from agent.structured_llm import direct_structured_llm_call
from agent.memory.dag.models import MemoryElement, MemoryGraph
from agent.state import State, build_agent_state_description
from agent.chain_of_action.prompts import format_section

from .models import TopicCluster, TopicNamingApproach, TopicNamingResult

logger = logging.getLogger(__name__)


# Pydantic models for LLM responses
class SimpleTopicResponse(BaseModel):
    """Response for simple topic naming approach."""

    topic_name: str = Field(description="2-5 word name for this topic")
    description: str = Field(description="1-2 sentence description of the topic")
    keywords: List[str] = Field(description="3-5 key terms that define this topic")


class StructuredTopicResponse(BaseModel):
    """Response for structured topic naming approach."""

    common_themes: List[str] = Field(
        description="Themes that appear across these memories"
    )
    topic_name: str = Field(description="Concise name capturing the main theme")
    description: str = Field(description="2-3 sentence description")
    keywords: List[str] = Field(description="5-8 defining keywords")
    reasoning: str = Field(description="Why this name captures the essence")


class ContrastiveTopicResponse(BaseModel):
    """Response for contrastive topic naming approach."""

    similarities: List[str] = Field(description="What makes these memories similar")
    distinguishing_features: List[str] = Field(
        description="What distinguishes this from other topics"
    )
    topic_name: str = Field(description="Name that emphasizes uniqueness")
    description: str = Field(
        description="Description focusing on distinguishing features"
    )
    keywords: List[str] = Field(description="Keywords unique to this topic")


def generate_topic_name_simple(
    cluster: TopicCluster,
    memory_graph: MemoryGraph,
    state: State,
    llm: LLM,
    model: SupportedModel,
    sample_size: int = 10,
) -> SimpleTopicResponse:
    """
    Generate topic name using simple approach.

    Prompt: "Here are N related memories. What topic do they share?
             Give a 2-5 word name for this topic."
    """
    # Get sample memories from cluster
    sample_memories = _get_sample_memories(cluster, memory_graph, sample_size)

    # Build memory list
    memory_list = _format_memories_for_prompt(sample_memories)

    state_desc = build_agent_state_description(state)

    prompt = f"""I am {state.name}. I'm analyzing a group of my memories that have been clustered together as semantically similar.

{state_desc}

{format_section("MEMORIES IN THIS CLUSTER", memory_list)}

My task:
1. Identify what topic or theme connects these memories
2. Give a concise 2-5 word name for this topic
3. Provide a brief description
4. List 3-5 key terms that define this topic

Think from my perspective - what aspect of my experience do these memories capture?"""

    try:
        response = direct_structured_llm_call(
            prompt=prompt,
            response_model=SimpleTopicResponse,
            model=model,
            llm=llm,
            caller="topic_naming_simple",
        )
        return response
    except Exception as e:
        logger.warning(f"Simple topic naming failed: {e}")
        return SimpleTopicResponse(
            topic_name="General Topic",
            description=f"Cluster of {len(cluster.memory_ids)} related memories",
            keywords=["memories", "related"],
        )


def generate_topic_name_structured(
    cluster: TopicCluster,
    memory_graph: MemoryGraph,
    state: State,
    llm: LLM,
    model: SupportedModel,
    sample_size: int = 10,
) -> StructuredTopicResponse:
    """
    Generate topic name using structured approach.

    First extract common themes, then synthesize a name.
    """
    sample_memories = _get_sample_memories(cluster, memory_graph, sample_size)
    memory_list = _format_memories_for_prompt(sample_memories)

    state_desc = build_agent_state_description(state)

    prompt = f"""I am {state.name}. I'm performing a structured analysis of a cluster of my memories to identify the underlying topic.

{state_desc}

{format_section("MEMORIES IN THIS CLUSTER", memory_list)}

My analysis process:
1. First, identify the common themes that appear across these memories
2. Then, synthesize a concise topic name that captures the main theme
3. Write a 2-3 sentence description of what this cluster represents
4. List 5-8 defining keywords
5. Explain why this name captures the essence of these memories

Think systematically about what patterns emerge from these memories."""

    try:
        response = direct_structured_llm_call(
            prompt=prompt,
            response_model=StructuredTopicResponse,
            model=model,
            llm=llm,
            caller="topic_naming_structured",
        )
        return response
    except Exception as e:
        logger.warning(f"Structured topic naming failed: {e}")
        return StructuredTopicResponse(
            common_themes=["general"],
            topic_name="General Topic",
            description=f"Cluster of {len(cluster.memory_ids)} related memories",
            keywords=["memories", "related"],
            reasoning="Fallback due to analysis failure",
        )


def generate_topic_name_contrastive(
    cluster: TopicCluster,
    memory_graph: MemoryGraph,
    all_clusters: List[TopicCluster],
    state: State,
    llm: LLM,
    model: SupportedModel,
    sample_size: int = 10,
) -> ContrastiveTopicResponse:
    """
    Generate topic name using contrastive approach.

    Define topic by what makes it different from other clusters.
    Requires knowledge of other clusters for comparison.
    """
    sample_memories = _get_sample_memories(cluster, memory_graph, sample_size)
    memory_list = _format_memories_for_prompt(sample_memories)

    # Get sample from other clusters for contrast
    other_samples = []
    for other in all_clusters:
        if other.id != cluster.id:
            other_mems = _get_sample_memories(other, memory_graph, 3)
            if other_mems:
                other_samples.append(f"- {other_mems[0].content[:100]}...")
    other_context = (
        "\n".join(other_samples[:5]) if other_samples else "No other clusters"
    )

    state_desc = build_agent_state_description(state)

    prompt = f"""I am {state.name}. I'm analyzing what makes a specific cluster of my memories unique compared to others.

{state_desc}

{format_section("MEMORIES IN THIS CLUSTER", memory_list)}

{format_section("SAMPLES FROM OTHER CLUSTERS (for contrast)", other_context)}

My task:
1. Identify what makes these memories similar to each other
2. Identify what distinguishes this cluster from the other clusters
3. Create a name that emphasizes what makes this cluster unique
4. Write a description focusing on the distinguishing features
5. List keywords that are unique to this topic

Focus on what makes this cluster DIFFERENT, not just what it contains."""

    try:
        response = direct_structured_llm_call(
            prompt=prompt,
            response_model=ContrastiveTopicResponse,
            model=model,
            llm=llm,
            caller="topic_naming_contrastive",
        )
        return response
    except Exception as e:
        logger.warning(f"Contrastive topic naming failed: {e}")
        return ContrastiveTopicResponse(
            similarities=["related memories"],
            distinguishing_features=["unique content"],
            topic_name="Distinct Topic",
            description=f"Cluster of {len(cluster.memory_ids)} related memories",
            keywords=["memories", "distinct"],
        )


def generate_topic_names_all_approaches(
    cluster: TopicCluster,
    memory_graph: MemoryGraph,
    all_clusters: List[TopicCluster],
    state: State,
    llm: LLM,
    model: SupportedModel,
) -> TopicNamingResult:
    """
    Generate names using all three approaches for comparison.

    Returns TopicNamingResult with names from each approach.
    """
    logger.info(f"Generating names for cluster {cluster.id} using all approaches")

    simple_resp = generate_topic_name_simple(cluster, memory_graph, state, llm, model)
    structured_resp = generate_topic_name_structured(
        cluster, memory_graph, state, llm, model
    )
    contrastive_resp = generate_topic_name_contrastive(
        cluster, memory_graph, all_clusters, state, llm, model
    )

    names = {
        TopicNamingApproach.SIMPLE: simple_resp.topic_name,
        TopicNamingApproach.STRUCTURED: structured_resp.topic_name,
        TopicNamingApproach.CONTRASTIVE: contrastive_resp.topic_name,
    }

    descriptions = {
        TopicNamingApproach.SIMPLE: simple_resp.description,
        TopicNamingApproach.STRUCTURED: structured_resp.description,
        TopicNamingApproach.CONTRASTIVE: contrastive_resp.description,
    }

    # Determine best approach (structured is generally more thorough)
    best_approach = TopicNamingApproach.STRUCTURED

    evaluation_notes = (
        f"Simple: '{simple_resp.topic_name}' - direct and concise. "
        f"Structured: '{structured_resp.topic_name}' - thorough analysis. "
        f"Contrastive: '{contrastive_resp.topic_name}' - emphasizes uniqueness."
    )

    return TopicNamingResult(
        cluster_id=cluster.id,
        names_by_approach=names,
        descriptions_by_approach=descriptions,
        best_approach=best_approach,
        evaluation_notes=evaluation_notes,
    )


def name_all_clusters(
    clusters: List[TopicCluster],
    memory_graph: MemoryGraph,
    state: State,
    llm: LLM,
    model: SupportedModel,
    approach: TopicNamingApproach = TopicNamingApproach.STRUCTURED,
) -> List[TopicCluster]:
    """
    Apply topic naming to all clusters, updating their name/description/keywords.

    Returns new list of clusters with populated fields.
    """
    logger.info(f"Naming {len(clusters)} clusters using {approach.value} approach")

    named_clusters = []

    for cluster in clusters:
        if approach == TopicNamingApproach.SIMPLE:
            response = generate_topic_name_simple(
                cluster, memory_graph, state, llm, model
            )
            named_cluster = TopicCluster(
                id=cluster.id,
                name=response.topic_name,
                description=response.description,
                memory_ids=cluster.memory_ids,
                centroid=cluster.centroid,
                coherence_score=cluster.coherence_score,
                keywords=response.keywords,
            )
        elif approach == TopicNamingApproach.STRUCTURED:
            response = generate_topic_name_structured(
                cluster, memory_graph, state, llm, model
            )
            named_cluster = TopicCluster(
                id=cluster.id,
                name=response.topic_name,
                description=response.description,
                memory_ids=cluster.memory_ids,
                centroid=cluster.centroid,
                coherence_score=cluster.coherence_score,
                keywords=response.keywords,
            )
        elif approach == TopicNamingApproach.CONTRASTIVE:
            response = generate_topic_name_contrastive(
                cluster, memory_graph, clusters, state, llm, model
            )
            named_cluster = TopicCluster(
                id=cluster.id,
                name=response.topic_name,
                description=response.description,
                memory_ids=cluster.memory_ids,
                centroid=cluster.centroid,
                coherence_score=cluster.coherence_score,
                keywords=response.keywords,
            )
        else:
            raise ValueError(f"Unknown approach: {approach}")

        named_clusters.append(named_cluster)
        logger.info(f"Named cluster: '{named_cluster.name}'")

    return named_clusters


def _get_sample_memories(
    cluster: TopicCluster, memory_graph: MemoryGraph, sample_size: int
) -> List[MemoryElement]:
    """Get sample of memories from cluster for prompting."""
    memories = []
    for mem_id in cluster.memory_ids[:sample_size]:
        if mem_id in memory_graph.elements:
            memories.append(memory_graph.elements[mem_id])
    return memories


def _format_memories_for_prompt(memories: List[MemoryElement]) -> str:
    """Format memories as numbered list for LLM prompt."""
    lines = []
    for i, mem in enumerate(memories, 1):
        timestamp = mem.timestamp.strftime("%Y-%m-%d %H:%M")
        content = mem.content[:200] + "..." if len(mem.content) > 200 else mem.content
        lines.append(f"{i}. [{timestamp}] {content}")
    return "\n".join(lines)
