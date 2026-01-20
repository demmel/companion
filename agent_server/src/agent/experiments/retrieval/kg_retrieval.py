"""Fair KG-based retrieval experiment.

This experiment compares:
- naive_sim: embed query, find similar memories
- fact_sim: embed query, find similar facts
- kg_aware: resolve entity/attribute from query, lookup in KG

All start from raw query text - no hardcoding.
"""

import hashlib
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from agent.embedding_service import EmbeddingService, get_embedding_service
from agent.llm import SupportedModel, create_llm

from .attribute_retrieval import (
    MemorySample,
    TypedFact,
    load_memories,
    extract_all_facts,
    retrieve_naive_similarity,
    retrieve_fact_similarity,
)
from .knowledge_graph import KnowledgeGraph, retrieve_kg_aware

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CONVERSATIONS_DIR = Path("conversations")
DATA_FILE = (
    "conversation_20251024_083630_306692_triggers.json"  # 1402 entries with summaries
)
CACHE_DIR = Path("src/agent/experiments/retrieval/cache")


# =============================================================================
# Caching Functions
# =============================================================================


def compute_cache_key(memories: list[MemorySample]) -> str:
    """Compute a hash-based cache key from memory IDs."""
    memory_ids = sorted(m.memory_id for m in memories)
    content = "|".join(memory_ids)
    return hashlib.md5(content.encode()).hexdigest()[:12]


def save_facts_to_json(facts: list[TypedFact], cache_file: Path) -> None:
    """Save extracted facts to JSON cache."""
    data = [asdict(f) for f in facts]
    with open(cache_file, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Cached {len(facts)} facts to {cache_file}")


def load_facts_from_json(cache_file: Path) -> list[TypedFact]:
    """Load facts from JSON cache."""
    with open(cache_file) as f:
        data = json.load(f)
    return [TypedFact(**item) for item in data]


def extract_or_load_facts(
    memories: list[MemorySample],
    llm: object,
    model: SupportedModel,
    cache_key: str,
) -> list[TypedFact]:
    """Load facts from cache if available, otherwise extract and cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{cache_key}_facts.json"

    if cache_file.exists():
        facts = load_facts_from_json(cache_file)
        print(f"  Loaded {len(facts)} facts from cache")
        return facts

    print(f"  No cache found, extracting facts...")
    facts = extract_all_facts(memories, llm, model)
    save_facts_to_json(facts, cache_file)
    return facts


# =============================================================================
# Statistical Functions
# =============================================================================


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


def format_stats(stats: dict[str, float], metric_name: str) -> str:
    """Format statistics for display."""
    return f"{stats['mean']:.3f} ± {stats['std']:.3f} (95% CI: [{stats['ci_low']:.3f}, {stats['ci_high']:.3f}])"


@dataclass
class TestQuery:
    """A test query with ground truth."""

    query_text: str
    query_type: str  # "entity_overview", "specific_attribute", "episodic"
    ground_truth_memory_ids: set[str]
    description: str


def populate_kg(
    facts: list[TypedFact],
    embedding_service: EmbeddingService,
) -> KnowledgeGraph:
    """Populate a knowledge graph from extracted facts."""
    kg = KnowledgeGraph(embedding_service)

    print("\nPopulating knowledge graph...")
    for fact in facts:
        kg.add_fact(
            raw_entity=fact.entity,
            raw_attribute=fact.attribute,
            value=fact.value,
            source_memory_id=fact.source_memory_id,
            timestamp=fact.timestamp,
        )

    # Print stats
    print(f"  Entities: {len(kg.entity_resolver.entities)}")
    for entity_id, entity in kg.entity_resolver.entities.items():
        print(f"    {entity.canonical_name}: {len(entity.aliases)} aliases")

    print(f"  Attributes in schema: {len(kg.attribute_normalizer.schema)}")
    print(f"  Facts: {len(kg.facts)}")

    return kg


def generate_test_queries(
    kg: KnowledgeGraph,
    memories: list[MemorySample],
) -> list[TestQuery]:
    """Generate test queries with proper ground truth from KG."""
    queries: list[TestQuery] = []

    print("\nGenerating test queries from KG...")

    # Entity overview queries
    for entity_id, entity in kg.entity_resolver.entities.items():
        current_state = kg.get_current_state(entity_id)
        if len(current_state) >= 2:
            ground_truth = set(f.source_memory_id for f in current_state)
            queries.append(
                TestQuery(
                    query_text=f"What do I know about {entity.canonical_name}?",
                    query_type="entity_overview",
                    ground_truth_memory_ids=ground_truth,
                    description=f"entity_overview for '{entity.canonical_name}' ({len(ground_truth)} memories)",
                )
            )
            print(
                f"  entity_overview: '{entity.canonical_name}' -> {len(ground_truth)} memories"
            )

    # Specific attribute queries
    for entity_id, entity in kg.entity_resolver.entities.items():
        entity_facts = kg.get_entity_facts(entity_id)

        # Group by attribute
        by_attr: dict[str, list] = defaultdict(list)
        for fact in entity_facts:
            by_attr[fact.attribute_id].append(fact)

        for attr_id, attr_facts in by_attr.items():
            if len(attr_facts) >= 2:  # Need multiple values to test
                attr_type = kg.attribute_normalizer.get_attribute_type(attr_id)
                if attr_type == "replacement":
                    # Ground truth = most recent
                    most_recent = max(attr_facts, key=lambda f: f.timestamp)
                    ground_truth = {most_recent.source_memory_id}
                else:
                    # Ground truth = all
                    ground_truth = set(f.source_memory_id for f in attr_facts)

                # Get canonical attribute name
                attr_name = attr_id.replace("attr_", "").replace("_", " ")
                if attr_id in kg.attribute_normalizer.schema:
                    attr_name = kg.attribute_normalizer.schema[attr_id].canonical_name

                queries.append(
                    TestQuery(
                        query_text=f"What is {entity.canonical_name}'s {attr_name}?",
                        query_type="specific_attribute",
                        ground_truth_memory_ids=ground_truth,
                        description=f"specific_attribute: {entity.canonical_name}.{attr_name}",
                    )
                )
                print(f"  specific_attribute: '{entity.canonical_name}.{attr_name}'")

    # Episodic queries - generate for all memories to get enough samples
    episodic_count = 0
    for memory in memories:
        # Use a snippet from the middle of the content for variety
        start_idx = min(20, len(memory.content) // 4)
        snippet = memory.content[start_idx : start_idx + 80]
        queries.append(
            TestQuery(
                query_text=f"Tell me about the conversation where: {snippet}...",
                query_type="episodic",
                ground_truth_memory_ids={memory.memory_id},
                description=f"episodic for memory {memory.timestamp}",
            )
        )
        episodic_count += 1

    print(f"  episodic: {episodic_count} queries")

    return queries


def compute_metrics(
    retrieved: list[str],
    ground_truth: set[str],
    query_type: str,
) -> dict[str, float]:
    """Compute appropriate metrics based on query type."""
    retrieved_set = set(retrieved)

    if query_type == "entity_overview":
        # Multi-answer: Recall, Precision, F1
        if not ground_truth:
            return {"recall": 0.0, "precision": 0.0, "f1": 0.0}

        true_positives = len(retrieved_set & ground_truth)
        recall = true_positives / len(ground_truth) if ground_truth else 0.0
        precision = true_positives / len(retrieved) if retrieved else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return {"recall": recall, "precision": precision, "f1": f1}

    else:
        # Single-answer: MRR
        mrr = 0.0
        for i, mem_id in enumerate(retrieved):
            if mem_id in ground_truth:
                mrr = 1.0 / (i + 1)
                break

        return {"mrr": mrr}


def run_experiment(max_memories: int = 100) -> None:
    """Run the fair KG-based retrieval experiment.

    Args:
        max_memories: Maximum number of memories to load (default 100)
    """
    print("\n" + "=" * 70)
    print("FAIR KG-BASED RETRIEVAL EXPERIMENT (SCALED)")
    print("=" * 70)
    print("All strategies start from raw query text - no hardcoding")
    print(f"Target: {max_memories} memories, 50+ queries per type")

    llm = create_llm()
    model = SupportedModel.MISTRAL_SMALL_3_2_Q4
    embedding_service = get_embedding_service()

    # Load memories
    data_path = CONVERSATIONS_DIR / DATA_FILE
    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}")
        return

    memories = load_memories(data_path, max_samples=max_memories)
    print(f"\nLoaded {len(memories)} memories")

    # Extract or load cached facts
    cache_key = compute_cache_key(memories)
    print(f"\nCache key: {cache_key}")
    all_facts = extract_or_load_facts(memories, llm, model, cache_key)
    print(f"Total facts: {len(all_facts)}")

    # Populate knowledge graph
    kg = populate_kg(all_facts, embedding_service)

    # Generate queries
    queries = generate_test_queries(kg, memories)
    print(f"\nTotal queries: {len(queries)}")

    by_type = defaultdict(list)
    for q in queries:
        by_type[q.query_type].append(q)

    for qt, qs in by_type.items():
        print(f"  {qt}: {len(qs)}")

    # Run retrieval
    print("\n" + "-" * 70)
    print("RUNNING RETRIEVAL COMPARISONS")
    print("-" * 70)

    results: dict[str, dict[str, list[dict[str, float]]]] = {
        qt: {"naive_sim": [], "fact_sim": [], "kg_aware": []}
        for qt in ["entity_overview", "specific_attribute", "episodic"]
    }

    for query in queries:
        # Naive similarity
        retrieved = retrieve_naive_similarity(
            query.query_text, memories, embedding_service
        )
        metrics = compute_metrics(
            retrieved, query.ground_truth_memory_ids, query.query_type
        )
        results[query.query_type]["naive_sim"].append(metrics)

        # Fact similarity
        retrieved = retrieve_fact_similarity(
            query.query_text, all_facts, embedding_service
        )
        metrics = compute_metrics(
            retrieved, query.ground_truth_memory_ids, query.query_type
        )
        results[query.query_type]["fact_sim"].append(metrics)

        # KG-aware (fair - uses embedding similarity to resolve entity/attribute)
        retrieved = retrieve_kg_aware(query.query_text, kg)
        metrics = compute_metrics(
            retrieved, query.ground_truth_memory_ids, query.query_type
        )
        results[query.query_type]["kg_aware"].append(metrics)

    # Print results with statistical analysis
    print("\n" + "=" * 70)
    print("RESULTS BY QUERY TYPE (with 95% confidence intervals)")
    print("=" * 70)

    for query_type in ["entity_overview", "specific_attribute", "episodic"]:
        type_results = results[query_type]
        print(f"\n--- {query_type.upper()} ---")

        if query_type == "entity_overview":
            print(f"(Ground truth = set of memories with current state)")
            print(f"\n{'Strategy':<12} {'F1':<40} {'n'}")
            print("-" * 60)

            for strategy_name in ["naive_sim", "fact_sim", "kg_aware"]:
                evals = type_results[strategy_name]
                if not evals:
                    continue

                f1_values = [e["f1"] for e in evals]
                stats = compute_stats(f1_values)
                stats_str = format_stats(stats, "F1")
                print(f"{strategy_name:<12} {stats_str:<40} {stats['n']}")

        else:
            print(f"(Ground truth = specific memory/memories)")
            print(f"\n{'Strategy':<12} {'MRR':<40} {'n'}")
            print("-" * 60)

            for strategy_name in ["naive_sim", "fact_sim", "kg_aware"]:
                evals = type_results[strategy_name]
                if not evals:
                    continue

                mrr_values = [e["mrr"] for e in evals]
                stats = compute_stats(mrr_values)
                stats_str = format_stats(stats, "MRR")
                print(f"{strategy_name:<12} {stats_str:<40} {stats['n']}")

    # Summary with statistical analysis
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    for query_type in ["entity_overview", "specific_attribute", "episodic"]:
        type_results = results[query_type]
        if not type_results["naive_sim"]:
            continue

        print(f"\n{query_type}:")

        if query_type == "entity_overview":
            metric_key = "f1"
            metric_name = "F1"
        else:
            metric_key = "mrr"
            metric_name = "MRR"

        # Find best strategy
        best_strat = None
        best_mean = -1.0
        strat_stats: dict[str, dict[str, float]] = {}

        for strat in ["naive_sim", "fact_sim", "kg_aware"]:
            values = [e[metric_key] for e in type_results[strat]]
            stats = compute_stats(values)
            strat_stats[strat] = stats
            if stats["mean"] > best_mean:
                best_mean = stats["mean"]
                best_strat = strat

        for strat in ["naive_sim", "fact_sim", "kg_aware"]:
            stats = strat_stats[strat]
            marker = " <-- BEST" if strat == best_strat else ""
            print(
                f"  {strat}: {metric_name}={stats['mean']:.3f} ± {stats['std']:.3f} (n={stats['n']}){marker}"
            )

        # Check if CIs overlap (basic significance check)
        if best_strat:
            best_stats = strat_stats[best_strat]
            for strat, stats in strat_stats.items():
                if strat != best_strat:
                    # CIs don't overlap if one's high is below other's low
                    if stats["ci_high"] < best_stats["ci_low"]:
                        print(
                            f"  -> {best_strat} significantly better than {strat} (CIs don't overlap)"
                        )

    print("\n" + "=" * 70)
    print(f"Dataset: {len(memories)} memories, {len(all_facts)} facts")
    print("=" * 70)


if __name__ == "__main__":
    run_experiment()
