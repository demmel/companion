"""Retrieval Quality Evaluation.

This experiment measures the QUALITY of retrieved memories, not just routing accuracy.
It answers: "Given the right strategy, are we retrieving relevant memories?"

Key metrics:
- Semantic similarity between query and retrieved memories
- Coverage: Are all relevant facts covered?
- Precision: Are retrieved memories actually relevant?
- Answer extractability: Can the query be answered from retrieved context?

Usage:
    uv run python -m agent.experiments.unified_retrieval.retrieval_quality --conversation <id>
"""

import argparse
import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from agent.embedding_service import get_embedding_service, EmbeddingService
from agent.llm import create_llm, SupportedModel
from agent.structured_llm import direct_structured_llm_call

from pydantic import BaseModel, Field

from .build_indices import load_indices, CACHE_DIR, load_conversation_memories
from .models import QueryType, Memory
from .query_classifier import RuleBasedQueryClassifier
from .unified_retriever import UnifiedRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Pydantic Models for LLM Evaluation
# =============================================================================


class RelevanceJudgment(BaseModel):
    """LLM judgment of memory relevance to query."""

    is_relevant: bool = Field(description="Whether this memory is relevant to answering the query")
    relevance_score: float = Field(description="Relevance score from 0.0 to 1.0", ge=0.0, le=1.0)
    reasoning: str = Field(description="Brief explanation of relevance judgment")


class AnswerabilityJudgment(BaseModel):
    """LLM judgment of whether query can be answered from context."""

    can_answer: bool = Field(description="Whether the query can be answered from the provided context")
    confidence: float = Field(description="Confidence in answerability from 0.0 to 1.0", ge=0.0, le=1.0)
    missing_info: str = Field(description="What information is missing, if any")
    answer_sketch: str = Field(description="Brief sketch of what the answer would be")


# =============================================================================
# Quality Metrics
# =============================================================================


@dataclass
class RetrievalQualityResult:
    """Quality metrics for a single query."""

    query_id: str
    query_text: str
    query_type: str
    strategy_used: str

    # Embedding-based metrics
    avg_similarity: float  # Average cosine similarity of retrieved memories to query
    max_similarity: float  # Max similarity (best match)
    min_similarity: float  # Min similarity (worst match)

    # LLM-judged metrics (optional, expensive)
    llm_relevance_score: float | None = None  # Average LLM-judged relevance
    llm_answerability: float | None = None  # Can query be answered?

    # Counts
    num_memories_retrieved: int = 0
    num_facts_retrieved: int = 0
    latency_ms: float = 0.0


@dataclass
class QualityExperimentResult:
    """Complete quality experiment results."""

    conversation_id: str
    num_memories_total: int
    num_queries: int
    results: list[RetrievalQualityResult]
    summary: str


# =============================================================================
# Quality Evaluation Functions
# =============================================================================


def compute_embedding_similarity(
    query: str,
    memories: list[Memory],
    embedding_service: EmbeddingService,
) -> tuple[float, float, float]:
    """Compute embedding-based similarity metrics.

    Returns: (avg_similarity, max_similarity, min_similarity)
    """
    if not memories:
        return 0.0, 0.0, 0.0

    query_emb = np.array(embedding_service.encode(query))

    similarities: list[float] = []
    for memory in memories:
        if memory.embedding_vector:
            mem_emb = np.array(memory.embedding_vector)
        else:
            mem_emb = np.array(embedding_service.encode(memory.content))

        # Cosine similarity
        sim = float(np.dot(query_emb, mem_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(mem_emb) + 1e-8))
        similarities.append(sim)

    return float(np.mean(similarities)), float(np.max(similarities)), float(np.min(similarities))


def judge_answerability_llm(
    query: str,
    context: str,
    llm,
    model: SupportedModel,
) -> AnswerabilityJudgment:
    """Use LLM to judge if query can be answered from context."""
    prompt = f"""Given the following memory context and query, determine if the query can be answered.

QUERY: {query}

MEMORY CONTEXT:
{context}

Evaluate:
1. Can this query be fully answered from the provided context?
2. What information, if any, is missing?
3. Provide a brief sketch of what the answer would be."""

    return direct_structured_llm_call(
        prompt=prompt,
        response_model=AnswerabilityJudgment,
        model=model,
        llm=llm,
        caller="retrieval_quality",
        temperature=0.1,
    )


# =============================================================================
# Test Queries with Expected Content
# =============================================================================


QUALITY_TEST_QUERIES = [
    {
        "id": "qual_001",
        "query": "What is the user currently wearing?",
        "query_type": "current_state",
        "expected_keywords": ["robe", "dress", "negligee", "appearance"],
    },
    {
        "id": "qual_002",
        "query": "What has the user worn today?",
        "query_type": "history",
        "expected_keywords": ["crimson", "negligee", "robe", "cashmere"],
    },
    {
        "id": "qual_003",
        "query": "How is David feeling?",
        "query_type": "current_state",
        "expected_keywords": ["mood", "feeling", "warm", "longing"],
    },
    {
        "id": "qual_004",
        "query": "What activities have happened recently?",
        "query_type": "history",
        "expected_keywords": ["cuddling", "kissing", "flirting"],
    },
    {
        "id": "qual_005",
        "query": "Describe the setting and atmosphere",
        "query_type": "entity_overview",
        "expected_keywords": ["couch", "living room", "rain", "storm"],
    },
    {
        "id": "qual_006",
        "query": "What happened at Elliot Bay?",
        "query_type": "temporal",
        "expected_keywords": ["storm", "rain", "bench", "kiss"],
    },
    {
        "id": "qual_007",
        "query": "What does David want?",
        "query_type": "current_state",
        "expected_keywords": ["desire", "want", "lap"],
    },
    {
        "id": "qual_008",
        "query": "How has the user's mood changed?",
        "query_type": "history",
        "expected_keywords": ["mood", "feeling", "flirty", "tender", "passionate"],
    },
]


def evaluate_keyword_coverage(
    memories: list[Memory],
    expected_keywords: list[str],
) -> float:
    """Compute what fraction of expected keywords appear in retrieved memories."""
    if not expected_keywords:
        return 1.0

    combined_text = " ".join(m.content.lower() for m in memories)

    found = sum(1 for kw in expected_keywords if kw.lower() in combined_text)
    return found / len(expected_keywords)


# =============================================================================
# Main Evaluation
# =============================================================================


def run_quality_evaluation(
    conversation_id: str,
    use_llm_judgment: bool = False,
    max_memories: int | None = None,
) -> QualityExperimentResult:
    """Run retrieval quality evaluation."""
    print("\n" + "=" * 70)
    print("RETRIEVAL QUALITY EVALUATION")
    print("=" * 70)

    # Load indices
    cache_dir = CACHE_DIR / conversation_id
    embedding_service = get_embedding_service()

    if cache_dir.exists():
        print(f"Loading cached indices from {cache_dir}")
        kg, memory_index, episode_index, topic_clusters = load_indices(
            cache_dir, embedding_service
        )
    else:
        raise FileNotFoundError(f"No cached indices found at {cache_dir}. Run build_indices first.")

    # Create retriever
    retriever = UnifiedRetriever(
        kg=kg,
        memory_index=memory_index,
        episode_index=episode_index,
        topic_clusters=topic_clusters,
        embedding_service=embedding_service,
        classifier=RuleBasedQueryClassifier(),
    )

    # Optionally set up LLM for judgments
    llm = None
    model = None
    if use_llm_judgment:
        llm = create_llm()
        model = SupportedModel.MISTRAL_SMALL_3_2_Q4

    print(f"\nMemory index contains {len(memory_index.memories)} memories")
    print(f"Running {len(QUALITY_TEST_QUERIES)} quality test queries")
    if use_llm_judgment:
        print("Using LLM for answerability judgments (slower)")

    results: list[RetrievalQualityResult] = []

    for query_data in QUALITY_TEST_QUERIES:
        print(f"\n  Testing: {query_data['query'][:50]}...")

        start_time = time.time()
        context = retriever.retrieve(query_data["query"])
        latency_ms = (time.time() - start_time) * 1000

        # Compute embedding similarity
        avg_sim, max_sim, min_sim = compute_embedding_similarity(
            query_data["query"],
            context.memories,
            embedding_service,
        )

        # Compute keyword coverage
        keyword_coverage = evaluate_keyword_coverage(
            context.memories,
            query_data.get("expected_keywords", []),
        )

        # LLM judgment (optional)
        llm_answerability = None
        if use_llm_judgment and llm and context.context_text:
            try:
                judgment = judge_answerability_llm(
                    query_data["query"],
                    context.context_text,
                    llm,
                    model,
                )
                llm_answerability = judgment.confidence if judgment.can_answer else 0.0
            except Exception as e:
                logger.warning(f"LLM judgment failed: {e}")

        result = RetrievalQualityResult(
            query_id=query_data["id"],
            query_text=query_data["query"],
            query_type=query_data["query_type"],
            strategy_used=context.strategy_used,
            avg_similarity=avg_sim,
            max_similarity=max_sim,
            min_similarity=min_sim,
            llm_relevance_score=keyword_coverage,  # Use keyword coverage as proxy
            llm_answerability=llm_answerability,
            num_memories_retrieved=len(context.memories),
            num_facts_retrieved=len(context.facts),
            latency_ms=latency_ms,
        )
        results.append(result)

        print(f"    Strategy: {context.strategy_used}")
        print(f"    Retrieved: {len(context.memories)} memories, {len(context.facts)} facts")
        print(f"    Avg similarity: {avg_sim:.3f}, Keyword coverage: {keyword_coverage:.1%}")

    # Generate summary
    summary_lines = [
        "",
        "=" * 70,
        "RETRIEVAL QUALITY SUMMARY",
        "=" * 70,
        "",
        f"{'Query Type':<20} {'Avg Sim':>10} {'Max Sim':>10} {'Coverage':>10} {'Memories':>10}",
        "-" * 70,
    ]

    # Group by query type
    by_type: dict[str, list[RetrievalQualityResult]] = {}
    for r in results:
        by_type.setdefault(r.query_type, []).append(r)

    for qtype, type_results in sorted(by_type.items()):
        avg_sim = np.mean([r.avg_similarity for r in type_results])
        max_sim = np.mean([r.max_similarity for r in type_results])
        coverage = np.mean([r.llm_relevance_score or 0 for r in type_results])
        memories = np.mean([r.num_memories_retrieved for r in type_results])

        summary_lines.append(
            f"{qtype:<20} {avg_sim:>10.3f} {max_sim:>10.3f} {coverage:>9.1%} {memories:>10.1f}"
        )

    # Overall metrics
    overall_avg_sim = np.mean([r.avg_similarity for r in results])
    overall_coverage = np.mean([r.llm_relevance_score or 0 for r in results])

    summary_lines.extend([
        "-" * 70,
        f"{'OVERALL':<20} {overall_avg_sim:>10.3f} {'-':>10} {overall_coverage:>9.1%} {'-':>10}",
        "",
        "=" * 70,
        "KEY FINDINGS",
        "=" * 70,
        "",
    ])

    if overall_avg_sim > 0.5:
        summary_lines.append(f"  Good semantic relevance: avg similarity = {overall_avg_sim:.3f}")
    else:
        summary_lines.append(f"  Low semantic relevance: avg similarity = {overall_avg_sim:.3f}")

    if overall_coverage > 0.7:
        summary_lines.append(f"  Good keyword coverage: {overall_coverage:.1%}")
    else:
        summary_lines.append(f"  Missing expected content: only {overall_coverage:.1%} coverage")

    summary = "\n".join(summary_lines)
    print(summary)

    return QualityExperimentResult(
        conversation_id=conversation_id,
        num_memories_total=len(memory_index.memories),
        num_queries=len(results),
        results=results,
        summary=summary,
    )


def save_results(result: QualityExperimentResult, output_file: Path) -> None:
    """Save results to JSON."""
    data = {
        "conversation_id": result.conversation_id,
        "num_memories_total": result.num_memories_total,
        "num_queries": result.num_queries,
        "results": [asdict(r) for r in result.results],
        "summary": result.summary,
    }

    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to {output_file}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval quality"
    )
    parser.add_argument(
        "--conversation",
        type=str,
        required=True,
        help="Conversation ID",
    )
    parser.add_argument(
        "--llm-judgment",
        action="store_true",
        help="Use LLM for answerability judgments (slower)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file",
    )

    args = parser.parse_args()

    result = run_quality_evaluation(
        conversation_id=args.conversation,
        use_llm_judgment=args.llm_judgment,
    )

    if args.output:
        save_results(result, Path(args.output))
    else:
        output_file = CACHE_DIR / f"{args.conversation}_quality.json"
        save_results(result, output_file)


if __name__ == "__main__":
    main()
