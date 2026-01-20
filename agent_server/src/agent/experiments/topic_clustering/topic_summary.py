"""
LLM-based cluster summarization for topic clusters.
"""

import logging
from typing import List
from pydantic import BaseModel, Field

from agent.llm import LLM, SupportedModel
from agent.structured_llm import direct_structured_llm_call
from agent.memory.dag.models import MemoryGraph, MemoryElement
from agent.state import State, build_agent_state_description
from agent.chain_of_action.prompts import format_section

from .models import TopicCluster

logger = logging.getLogger(__name__)


class TopicSummaryResponse(BaseModel):
    """LLM response for topic summarization."""

    summary: str = Field(description="Comprehensive 3-5 sentence summary of the topic")
    key_events: List[str] = Field(description="Key events or facts mentioned")
    themes: List[str] = Field(description="Major themes present")
    searchable_terms: List[str] = Field(
        description="Terms that would help find this cluster"
    )


class SummaryQualityEvaluation(BaseModel):
    """Evaluation of summary quality."""

    completeness_score: float = Field(
        ge=0.0, le=1.0, description="Coverage of key content"
    )
    accuracy_score: float = Field(ge=0.0, le=1.0, description="Factual accuracy")
    coherence_score: float = Field(
        ge=0.0, le=1.0, description="Logical flow and clarity"
    )
    searchability_score: float = Field(
        ge=0.0, le=1.0, description="Likelihood queries would match"
    )
    overall_score: float = Field(ge=0.0, le=1.0, description="Weighted average")
    issues: List[str] = Field(default_factory=list, description="Identified issues")


def generate_topic_summary(
    cluster: TopicCluster,
    memory_graph: MemoryGraph,
    state: State,
    llm: LLM,
    model: SupportedModel,
) -> TopicSummaryResponse:
    """
    Generate comprehensive summary for a topic cluster.
    """
    # Get all memories in cluster (or sample if too many)
    sample_size = min(len(cluster.memory_ids), 15)
    memories = []
    for mem_id in cluster.memory_ids[:sample_size]:
        if mem_id in memory_graph.elements:
            memories.append(memory_graph.elements[mem_id])

    memory_list = _format_memories_for_summary(memories)

    state_desc = build_agent_state_description(state)

    prompt = f"""I am {state.name}. I need to create a comprehensive summary of a topic cluster from my memories.

{state_desc}

Topic Name: {cluster.name}
Topic Description: {cluster.description}
Number of memories: {len(cluster.memory_ids)}

{format_section("SAMPLE MEMORIES FROM THIS CLUSTER", memory_list)}

My task:
1. Write a comprehensive 3-5 sentence summary that captures what this topic cluster is about
2. List the key events or facts mentioned in these memories
3. Identify the major themes present
4. List terms that would help someone search for and find this cluster

Write the summary in FIRST PERSON from my perspective. Be specific about what happened or was discussed."""

    try:
        response = direct_structured_llm_call(
            prompt=prompt,
            response_model=TopicSummaryResponse,
            model=model,
            llm=llm,
            caller="topic_summary",
        )
        return response
    except Exception as e:
        logger.warning(f"Topic summary generation failed: {e}")
        return TopicSummaryResponse(
            summary=f"This cluster contains {len(cluster.memory_ids)} memories about {cluster.name}.",
            key_events=[],
            themes=[cluster.name] if cluster.name else [],
            searchable_terms=cluster.keywords if cluster.keywords else [],
        )


def evaluate_summary_quality(
    summary: TopicSummaryResponse,
    cluster: TopicCluster,
    memory_graph: MemoryGraph,
    state: State,
    llm: LLM,
    model: SupportedModel,
) -> SummaryQualityEvaluation:
    """
    Evaluate quality of generated summary using LLM-as-judge.

    Checks:
    - Does summary capture what the cluster is about?
    - Does summary mention key memories?
    - Would searching the summary find the right cluster?
    - Are there hallucinations?
    """
    # Get memories for verification
    sample_size = min(len(cluster.memory_ids), 10)
    memories = []
    for mem_id in cluster.memory_ids[:sample_size]:
        if mem_id in memory_graph.elements:
            memories.append(memory_graph.elements[mem_id])

    memory_list = _format_memories_for_summary(memories)

    prompt = f"""I need to evaluate the quality of a summary written about a cluster of memories.

CLUSTER NAME: {cluster.name}

SUMMARY TO EVALUATE:
{summary.summary}

KEY EVENTS LISTED:
{chr(10).join('- ' + e for e in summary.key_events)}

ACTUAL MEMORIES IN CLUSTER:
{memory_list}

Evaluate the summary on these criteria (score 0.0 to 1.0):

1. COMPLETENESS (0.0-1.0): Does the summary capture the main content and themes of the memories?
2. ACCURACY (0.0-1.0): Is everything in the summary actually supported by the memories? (Look for hallucinations)
3. COHERENCE (0.0-1.0): Is the summary logically structured and easy to understand?
4. SEARCHABILITY (0.0-1.0): Would someone searching for topics in these memories find this summary?

Also identify any specific issues (hallucinations, missing key content, etc.)."""

    try:
        response = direct_structured_llm_call(
            prompt=prompt,
            response_model=SummaryQualityEvaluation,
            model=model,
            llm=llm,
            caller="summary_quality_evaluation",
        )

        # Calculate overall score as weighted average
        response.overall_score = (
            response.completeness_score * 0.3
            + response.accuracy_score * 0.3
            + response.coherence_score * 0.2
            + response.searchability_score * 0.2
        )

        return response
    except Exception as e:
        logger.warning(f"Summary quality evaluation failed: {e}")
        return SummaryQualityEvaluation(
            completeness_score=0.5,
            accuracy_score=0.5,
            coherence_score=0.5,
            searchability_score=0.5,
            overall_score=0.5,
            issues=["Evaluation failed - scores are placeholders"],
        )


def summarize_all_clusters(
    clusters: List[TopicCluster],
    memory_graph: MemoryGraph,
    state: State,
    llm: LLM,
    model: SupportedModel,
) -> List[TopicSummaryResponse]:
    """
    Generate summaries for all clusters.
    """
    logger.info(f"Generating summaries for {len(clusters)} clusters")

    summaries = []
    for cluster in clusters:
        logger.info(f"Summarizing cluster '{cluster.name}'")
        summary = generate_topic_summary(cluster, memory_graph, state, llm, model)
        summaries.append(summary)

    return summaries


def _format_memories_for_summary(memories: List[MemoryElement]) -> str:
    """Format memories for summary generation prompt."""
    lines = []
    for i, mem in enumerate(memories, 1):
        timestamp = mem.timestamp.strftime("%Y-%m-%d %H:%M")
        content = mem.content[:250] + "..." if len(mem.content) > 250 else mem.content
        lines.append(f"{i}. [{timestamp}] {content}")
    return "\n".join(lines)
