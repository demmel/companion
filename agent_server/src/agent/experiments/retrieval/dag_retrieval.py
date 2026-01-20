"""Proper DAG-based retrieval experiment.

This experiment tests retrieval against actual MemoryElements from the DAG,
not compressed_summary from trigger entries.

Key differences from previous experiments:
- Uses actual MemoryElement.content (granular atomic events)
- Uses container_id as event boundary for ground truth
- Generates paraphrased queries via LLM (not copying exact text)
"""

import hashlib
import json
import logging
import random
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from pydantic import BaseModel, Field

from agent.embedding_service import EmbeddingService, get_embedding_service
from agent.llm import SupportedModel, create_llm
from agent.llm.router import LLM
from agent.memory.dag.models import MemoryElement, MemoryEdge, ConfidenceLevel
from agent.structured_llm import direct_structured_llm_call

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CONVERSATIONS_DIR = Path("conversations")
DAG_FILE = "archive/conversation_20251024_083630_306692/conversation_20251024_083630_306692_dag.json"
CACHE_DIR = Path("src/agent/experiments/retrieval/cache")


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class DagMemory:
    """A memory element from the DAG."""
    id: str
    content: str
    container_id: str
    timestamp: str
    confidence_level: str
    embedding: list[float] | None = None


@dataclass
class TestQuery:
    """A test query with ground truth."""
    query_text: str
    query_type: str  # "event", "state"
    ground_truth_memory_ids: set[str]
    container_id: str | None = None
    description: str = ""


# =============================================================================
# Data Loading
# =============================================================================


def load_dag_memories(dag_path: Path) -> tuple[list[DagMemory], dict[str, list[str]]]:
    """Load actual MemoryElements from DAG JSON file.

    Returns:
        memories: List of DagMemory objects
        edges: Dict mapping source_id -> list of target_ids
    """
    with open(dag_path, encoding='utf-8') as f:
        data = json.load(f)

    memories: list[DagMemory] = []
    for elem_data in data['memory']['elements'].values():
        memories.append(DagMemory(
            id=elem_data['id'],
            content=elem_data['content'],
            container_id=elem_data['container_id'],
            timestamp=elem_data['timestamp'],
            confidence_level=elem_data['confidence_level'],
            embedding=elem_data.get('embedding_vector'),
        ))

    # Build edge adjacency
    edges: dict[str, list[str]] = defaultdict(list)
    for edge_data in data['memory']['edges'].values():
        edges[edge_data['source_id']].append(edge_data['target_id'])

    return memories, dict(edges)


def group_by_container(memories: list[DagMemory]) -> dict[str, list[DagMemory]]:
    """Group memories by container_id."""
    by_container: dict[str, list[DagMemory]] = defaultdict(list)
    for m in memories:
        by_container[m.container_id].append(m)
    return dict(by_container)


# =============================================================================
# Query Generation
# =============================================================================


class QueryGenerationResponse(BaseModel):
    """Response model for query generation."""
    question: str = Field(description="A natural question to retrieve these memories")


QUERY_GENERATION_PROMPT = """Given these memories from a conversation, generate a natural question someone might ask to recall this event. The question should NOT use exact words from the memories - paraphrase naturally.

Memories:
{memories}

Generate ONE natural question that would retrieve these memories."""


def generate_paraphrased_query(
    container_memories: list[DagMemory],
    llm: LLM,
    model: SupportedModel,
) -> str:
    """Generate a paraphrased query for a set of memories using LLM."""
    # Format memories for prompt
    memory_text = "\n".join(
        f"- {m.content[:200]}" for m in container_memories[:5]  # Limit to avoid token overflow
    )

    prompt = QUERY_GENERATION_PROMPT.format(memories=memory_text)

    response = direct_structured_llm_call(
        prompt=prompt,
        response_model=QueryGenerationResponse,
        model=model,
        llm=llm,
        caller="generate_query",
    )

    return response.question.strip().strip('"')


def generate_event_queries(
    memories: list[DagMemory],
    llm: LLM,
    model: SupportedModel,
    n_queries: int = 100,
    cache_key: str | None = None,
) -> list[TestQuery]:
    """Generate paraphrased event queries using LLM."""
    # Check cache first
    if cache_key:
        cache_file = CACHE_DIR / f"{cache_key}_event_queries.json"
        if cache_file.exists():
            with open(cache_file, encoding='utf-8') as f:
                data = json.load(f)
            queries = [
                TestQuery(
                    query_text=q['query_text'],
                    query_type=q['query_type'],
                    ground_truth_memory_ids=set(q['ground_truth_memory_ids']),
                    container_id=q.get('container_id'),
                    description=q.get('description', ''),
                )
                for q in data
            ]
            print(f"  Loaded {len(queries)} queries from cache")
            return queries

    by_container = group_by_container(memories)

    # Filter to containers with 3+ memories (meaningful events)
    candidates = [
        cid for cid, mems in by_container.items()
        if len(mems) >= 3
    ]

    print(f"  Found {len(candidates)} containers with 3+ memories")

    # Sample containers
    sample_size = min(n_queries, len(candidates))
    sampled = random.sample(candidates, sample_size)

    queries: list[TestQuery] = []
    for i, container_id in enumerate(sampled):
        container_memories = by_container[container_id]

        try:
            query_text = generate_paraphrased_query(container_memories, llm, model)

            queries.append(TestQuery(
                query_text=query_text,
                query_type="event",
                ground_truth_memory_ids={m.id for m in container_memories},
                container_id=container_id,
                description=f"Event query for container {container_id[:8]}",
            ))

            if (i + 1) % 10 == 0:
                print(f"  Generated {i + 1}/{sample_size} queries")

        except Exception as e:
            logger.warning(f"Failed to generate query for container {container_id}: {e}")

    # Cache queries
    if cache_key:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file = CACHE_DIR / f"{cache_key}_event_queries.json"
        data = [
            {
                'query_text': q.query_text,
                'query_type': q.query_type,
                'ground_truth_memory_ids': list(q.ground_truth_memory_ids),
                'container_id': q.container_id,
                'description': q.description,
            }
            for q in queries
        ]
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        print(f"  Cached {len(queries)} queries to {cache_file}")

    return queries


# =============================================================================
# Retrieval Strategies
# =============================================================================


def retrieve_naive_similarity(
    query: str,
    memories: list[DagMemory],
    embedding_service: EmbeddingService,
    top_k: int = 20,
) -> list[str]:
    """Baseline: pure embedding similarity retrieval."""
    query_emb = np.array(embedding_service.encode(query))

    scores: list[tuple[DagMemory, float]] = []
    for m in memories:
        if m.embedding:
            mem_emb = np.array(m.embedding)
            score = float(np.dot(query_emb, mem_emb))
            scores.append((m, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return [m.id for m, _ in scores[:top_k]]


def retrieve_with_edge_expansion(
    query: str,
    memories: list[DagMemory],
    edges: dict[str, list[str]],
    embedding_service: EmbeddingService,
    top_k: int = 20,
    expansion_hops: int = 1,
) -> list[str]:
    """Similarity + edge expansion: find similar memories, then follow edges."""
    # First get top candidates by similarity
    initial_k = max(5, top_k // 2)
    initial_results = retrieve_naive_similarity(query, memories, embedding_service, initial_k)

    # Expand via edges
    expanded = set(initial_results)
    frontier = set(initial_results)

    for _ in range(expansion_hops):
        new_frontier: set[str] = set()
        for mem_id in frontier:
            if mem_id in edges:
                for target_id in edges[mem_id]:
                    if target_id not in expanded:
                        expanded.add(target_id)
                        new_frontier.add(target_id)
        frontier = new_frontier

    # Re-rank by similarity
    mem_by_id = {m.id: m for m in memories}
    query_emb = np.array(embedding_service.encode(query))

    scored: list[tuple[str, float]] = []
    for mem_id in expanded:
        if mem_id in mem_by_id and mem_by_id[mem_id].embedding:
            mem_emb = np.array(mem_by_id[mem_id].embedding)
            score = float(np.dot(query_emb, mem_emb))
            scored.append((mem_id, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [mid for mid, _ in scored[:top_k]]


def retrieve_container_aware(
    query: str,
    memories: list[DagMemory],
    embedding_service: EmbeddingService,
    top_k: int = 20,
) -> list[str]:
    """Find similar memories, then include all from same container."""
    # Get top similar memories
    initial_k = 5
    initial_results = retrieve_naive_similarity(query, memories, embedding_service, initial_k)

    # Get their containers
    mem_by_id = {m.id: m for m in memories}
    containers_hit = {mem_by_id[mid].container_id for mid in initial_results if mid in mem_by_id}

    # Get all memories from those containers
    by_container = group_by_container(memories)
    expanded: list[str] = []
    for cid in containers_hit:
        if cid in by_container:
            expanded.extend(m.id for m in by_container[cid])

    # Dedupe while preserving order
    seen: set[str] = set()
    result: list[str] = []
    for mid in expanded:
        if mid not in seen:
            seen.add(mid)
            result.append(mid)

    return result[:top_k]


# =============================================================================
# Metrics
# =============================================================================


def compute_recall_at_k(retrieved: list[str], ground_truth: set[str], k: int) -> float:
    """Compute Recall@k: fraction of ground truth in top-k."""
    if not ground_truth:
        return 0.0
    retrieved_set = set(retrieved[:k])
    return len(retrieved_set & ground_truth) / len(ground_truth)


def compute_precision_at_k(retrieved: list[str], ground_truth: set[str], k: int) -> float:
    """Compute Precision@k: fraction of top-k that are ground truth."""
    if k == 0:
        return 0.0
    retrieved_set = set(retrieved[:k])
    return len(retrieved_set & ground_truth) / k


def compute_stats(values: list[float]) -> dict[str, float]:
    """Compute mean, std, and 95% CI for a list of values."""
    if not values:
        return {"mean": 0.0, "std": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n": 0}

    mean = float(np.mean(values))
    std = float(np.std(values))
    n = len(values)
    se = std / np.sqrt(n) if n > 0 else 0.0
    ci_low = mean - 1.96 * se
    ci_high = mean + 1.96 * se

    return {"mean": mean, "std": std, "ci_low": ci_low, "ci_high": ci_high, "n": n}


def format_stats(stats: dict[str, float]) -> str:
    """Format statistics for display."""
    return f"{stats['mean']:.3f} +/- {stats['std']:.3f} (95% CI: [{stats['ci_low']:.3f}, {stats['ci_high']:.3f}])"


# =============================================================================
# Main Experiment
# =============================================================================


def run_experiment(n_queries: int = 100) -> None:
    """Run the DAG-based retrieval experiment."""
    print("\n" + "=" * 70)
    print("DAG-BASED RETRIEVAL EXPERIMENT")
    print("=" * 70)
    print("Using actual MemoryElements from DAG (not compressed_summary)")
    print("Event boundary = container_id (one trigger = one event)")

    # Load data
    dag_path = CONVERSATIONS_DIR / DAG_FILE
    if not dag_path.exists():
        print(f"Error: DAG file not found: {dag_path}")
        return

    print(f"\nLoading DAG from {dag_path}...")
    memories, edges = load_dag_memories(dag_path)
    print(f"  Loaded {len(memories)} memories")
    print(f"  Loaded {len(edges)} edge sources")

    by_container = group_by_container(memories)
    print(f"  {len(by_container)} containers")

    # Filter to memories with embeddings
    memories_with_emb = [m for m in memories if m.embedding]
    print(f"  {len(memories_with_emb)} memories have embeddings")

    # Setup
    llm = create_llm()
    model = SupportedModel.MISTRAL_SMALL_3_2_Q4
    embedding_service = get_embedding_service()

    # Compute cache key
    cache_key = hashlib.md5(DAG_FILE.encode()).hexdigest()[:12]
    print(f"\nCache key: {cache_key}")

    # Generate queries
    print("\nGenerating event queries...")
    queries = generate_event_queries(
        memories_with_emb, llm, model,
        n_queries=n_queries,
        cache_key=cache_key,
    )
    print(f"  Generated {len(queries)} event queries")

    # Show sample queries
    print("\nSample queries:")
    for q in queries[:3]:
        print(f"  Q: {q.query_text}")
        print(f"     Ground truth: {len(q.ground_truth_memory_ids)} memories")

    # Run retrieval
    print("\n" + "-" * 70)
    print("RUNNING RETRIEVAL COMPARISONS")
    print("-" * 70)

    strategies = {
        "naive_sim": lambda q: retrieve_naive_similarity(
            q, memories_with_emb, embedding_service, top_k=20
        ),
        "edge_expand": lambda q: retrieve_with_edge_expansion(
            q, memories_with_emb, edges, embedding_service, top_k=20
        ),
        "container_aware": lambda q: retrieve_container_aware(
            q, memories_with_emb, embedding_service, top_k=20
        ),
    }

    results: dict[str, list[dict[str, float]]] = {
        name: [] for name in strategies
    }

    for i, query in enumerate(queries):
        for strat_name, strat_fn in strategies.items():
            retrieved = strat_fn(query.query_text)

            recall_5 = compute_recall_at_k(retrieved, query.ground_truth_memory_ids, 5)
            recall_10 = compute_recall_at_k(retrieved, query.ground_truth_memory_ids, 10)
            recall_20 = compute_recall_at_k(retrieved, query.ground_truth_memory_ids, 20)
            precision_5 = compute_precision_at_k(retrieved, query.ground_truth_memory_ids, 5)

            results[strat_name].append({
                "recall@5": recall_5,
                "recall@10": recall_10,
                "recall@20": recall_20,
                "precision@5": precision_5,
            })

        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{len(queries)} queries")

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS (with 95% confidence intervals)")
    print("=" * 70)

    metrics = ["recall@5", "recall@10", "recall@20", "precision@5"]

    for metric in metrics:
        print(f"\n--- {metric.upper()} ---")
        print(f"{'Strategy':<18} {'Value':<45} {'n'}")
        print("-" * 70)

        for strat_name in strategies:
            values = [r[metric] for r in results[strat_name]]
            stats = compute_stats(values)
            stats_str = format_stats(stats)
            print(f"{strat_name:<18} {stats_str:<45} {stats['n']}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\nDataset: {len(memories_with_emb)} memories, {len(queries)} queries")
    print("\nKey findings:")

    for metric in ["recall@10"]:
        print(f"\n{metric}:")
        strat_stats: dict[str, dict[str, float]] = {}
        for strat_name in strategies:
            values = [r[metric] for r in results[strat_name]]
            strat_stats[strat_name] = compute_stats(values)

        # Find best
        best_strat = max(strat_stats, key=lambda s: strat_stats[s]["mean"])

        for strat_name, stats in strat_stats.items():
            marker = " <-- BEST" if strat_name == best_strat else ""
            print(f"  {strat_name}: {stats['mean']:.3f} +/- {stats['std']:.3f}{marker}")


if __name__ == "__main__":
    run_experiment(n_queries=50)
