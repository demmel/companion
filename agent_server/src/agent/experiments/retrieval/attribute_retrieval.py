"""Attribute-aware retrieval experiment.

Key insight: Facts have different semantics:
- Replacement: mood, location, appearance (most recent wins)
- Additive: preferences, relationships, experiences (accumulates)

For "What do I know about David?", correct retrieval = set of memories containing:
- Most recent value for each replacement attribute
- All values for additive attributes
"""

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import groupby
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from agent.embedding_service import EmbeddingService, get_embedding_service
from agent.llm import LLM, SupportedModel, create_llm
from agent.structured_llm import direct_structured_llm_call

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CONVERSATIONS_DIR = Path("conversations")
DATA_FILE = "conversation_20251024_083630_306692_triggers.json"


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class MemorySample:
    """A memory sample from the conversation."""

    memory_id: str
    content: str
    timestamp: int  # Index for ordering


@dataclass
class TypedFact:
    """A fact extracted from a memory with attribute type."""

    entity: str  # "David", "user", "work"
    attribute: str  # "current_mood", "food_preference"
    attribute_type: str  # "replacement" or "additive"
    value: str  # "happy", "pizza"
    source_memory_id: str
    timestamp: int  # Memory index for ordering


@dataclass
class TestQuery:
    """A test query with ground truth."""

    query_text: str
    query_type: str  # "entity_overview", "specific_attribute", "episodic"
    ground_truth_memory_ids: set[str]
    description: str


@dataclass
class RetrievalResult:
    """Result of a retrieval for a query."""

    query: TestQuery
    retrieved_memory_ids: list[str]
    # Metrics
    recall: float = 0.0
    precision: float = 0.0
    f1: float = 0.0
    mrr: float = 0.0  # For single-answer queries


# =============================================================================
# Pydantic Models for LLM Extraction
# =============================================================================


class ExtractedFact(BaseModel):
    """A single fact extracted by the LLM."""

    entity: str = Field(
        description="Who/what is this about (e.g., 'David', 'user', 'work')"
    )
    attribute: str = Field(
        description="What property (e.g., 'current_mood', 'food_preference', 'appearance')"
    )
    attribute_type: str = Field(
        description="'replacement' if new values override old, 'additive' if values accumulate"
    )
    value: str = Field(description="The actual information")


class FactExtractionResponse(BaseModel):
    """LLM response for fact extraction."""

    facts: list[ExtractedFact] = Field(
        description="List of facts extracted from the memory"
    )


FACT_EXTRACTION_PROMPT = """Extract structured facts from this memory.

For each fact, identify:
- entity: who/what is this about (use consistent names like "David", "user", "companion")
- attribute: what property (use consistent naming like "current_mood", "appearance", "food_preference")
- attribute_type:
  - "replacement" if new values override old (mood, location, appearance, status, what someone is wearing, current activity)
  - "additive" if values accumulate (preferences, likes, dislikes, relationships, experiences, knowledge, things learned)
- value: the actual information

Examples:
- "David is happy" -> entity="David", attribute="current_mood", attribute_type="replacement", value="happy"
- "David likes pizza" -> entity="David", attribute="food_preference", attribute_type="additive", value="pizza"
- "Companion is wearing red dress" -> entity="companion", attribute="appearance", attribute_type="replacement", value="red dress"

MEMORY:
{content}

Extract all facts from this memory."""


# =============================================================================
# Data Loading
# =============================================================================


def load_memories(filepath: Path, max_samples: int = 15) -> list[MemorySample]:
    """Load memory samples from triggers file."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    memories: list[MemorySample] = []

    for i, entry_data in enumerate(data["entries"]):
        if "compressed_summary" in entry_data and entry_data["compressed_summary"]:
            content = entry_data["compressed_summary"]
            if len(content) > 100:
                memories.append(
                    MemorySample(
                        memory_id=entry_data.get("entry_id", f"memory_{i}"),
                        content=content,
                        timestamp=i,
                    )
                )

        if len(memories) >= max_samples:
            break

    logger.info(f"Loaded {len(memories)} memories")
    return memories


# =============================================================================
# Fact Extraction
# =============================================================================


def extract_typed_facts(
    memory: MemorySample,
    llm: LLM,
    model: SupportedModel,
) -> list[TypedFact]:
    """Extract typed facts from a memory."""
    prompt = FACT_EXTRACTION_PROMPT.format(content=memory.content)

    try:
        response = direct_structured_llm_call(
            prompt=prompt,
            response_model=FactExtractionResponse,
            model=model,
            llm=llm,
            caller="extract_typed_facts",
        )

        facts: list[TypedFact] = []
        for f in response.facts:
            # Normalize entity and attribute names
            entity = f.entity.lower().strip()
            attribute = f.attribute.lower().strip().replace(" ", "_")
            attr_type = f.attribute_type.lower().strip()

            # Validate attribute_type
            if attr_type not in ["replacement", "additive"]:
                attr_type = "replacement"  # Default to replacement if unclear

            facts.append(
                TypedFact(
                    entity=entity,
                    attribute=attribute,
                    attribute_type=attr_type,
                    value=f.value,
                    source_memory_id=memory.memory_id,
                    timestamp=memory.timestamp,
                )
            )

        return facts

    except Exception as e:
        logger.error(f"Fact extraction failed for {memory.memory_id}: {e}")
        return []


def extract_all_facts(
    memories: list[MemorySample],
    llm: LLM,
    model: SupportedModel,
) -> list[TypedFact]:
    """Extract typed facts from all memories."""
    all_facts: list[TypedFact] = []

    print("\nExtracting typed facts from memories...")
    for memory in memories:
        facts = extract_typed_facts(memory, llm, model)
        all_facts.extend(facts)
        print(f"  Memory {memory.timestamp}: {len(facts)} facts")
        for f in facts[:3]:  # Show first 3
            print(
                f"    - {f.entity}.{f.attribute} ({f.attribute_type}) = {f.value[:30]}..."
            )

    return all_facts


# =============================================================================
# Ground Truth Computation
# =============================================================================


def group_facts_by_entity(facts: list[TypedFact]) -> dict[str, list[TypedFact]]:
    """Group facts by entity."""
    by_entity: dict[str, list[TypedFact]] = defaultdict(list)
    for fact in facts:
        by_entity[fact.entity].append(fact)
    return dict(by_entity)


def get_ground_truth_memories(entity: str, entity_facts: list[TypedFact]) -> set[str]:
    """Get the set of memory IDs needed to answer 'What do I know about {entity}?'"""
    needed: set[str] = set()

    # Group by attribute
    sorted_facts = sorted(entity_facts, key=lambda f: f.attribute)
    for attribute, attr_facts_iter in groupby(sorted_facts, key=lambda f: f.attribute):
        attr_facts = list(attr_facts_iter)

        if attr_facts[0].attribute_type == "replacement":
            # Only need most recent
            most_recent = max(attr_facts, key=lambda f: f.timestamp)
            needed.add(most_recent.source_memory_id)
        else:  # additive
            # Need all of them
            for fact in attr_facts:
                needed.add(fact.source_memory_id)

    return needed


def get_specific_attribute_ground_truth(
    entity: str,
    attribute: str,
    entity_facts: list[TypedFact],
) -> set[str]:
    """Get ground truth for a specific attribute query."""
    attr_facts = [f for f in entity_facts if f.attribute == attribute]
    if not attr_facts:
        return set()

    if attr_facts[0].attribute_type == "replacement":
        most_recent = max(attr_facts, key=lambda f: f.timestamp)
        return {most_recent.source_memory_id}
    else:
        return {f.source_memory_id for f in attr_facts}


# =============================================================================
# Query Generation
# =============================================================================


def generate_test_queries(
    facts: list[TypedFact],
    memories: list[MemorySample],
) -> list[TestQuery]:
    """Generate test queries with proper ground truth."""
    queries: list[TestQuery] = []

    by_entity = group_facts_by_entity(facts)

    print("\nGenerating test queries...")

    # Entity overview queries
    for entity, entity_facts in by_entity.items():
        if len(entity_facts) >= 3:  # Only entities with enough facts
            ground_truth = get_ground_truth_memories(entity, entity_facts)
            if len(ground_truth) >= 2:  # Only if multiple memories needed
                queries.append(
                    TestQuery(
                        query_text=f"What do I know about {entity}?",
                        query_type="entity_overview",
                        ground_truth_memory_ids=ground_truth,
                        description=f"entity_overview for '{entity}' (needs {len(ground_truth)} memories)",
                    )
                )
                print(f"  entity_overview: '{entity}' -> {len(ground_truth)} memories")

    # Specific attribute queries (for replacement attributes with updates)
    for entity, entity_facts in by_entity.items():
        # Group by attribute
        sorted_facts = sorted(entity_facts, key=lambda f: f.attribute)
        for attribute, attr_facts_iter in groupby(
            sorted_facts, key=lambda f: f.attribute
        ):
            attr_facts = list(attr_facts_iter)

            # Only replacement attributes with multiple values
            if attr_facts[0].attribute_type == "replacement" and len(attr_facts) >= 2:
                ground_truth = get_specific_attribute_ground_truth(
                    entity, attribute, entity_facts
                )
                if ground_truth:
                    queries.append(
                        TestQuery(
                            query_text=f"What is {entity}'s current {attribute.replace('_', ' ')}?",
                            query_type="specific_attribute",
                            ground_truth_memory_ids=ground_truth,
                            description=f"specific_attribute: {entity}.{attribute}",
                        )
                    )
                    print(f"  specific_attribute: '{entity}.{attribute}'")

    # Episodic queries - one per memory
    for memory in memories[:5]:  # Limit to first 5
        queries.append(
            TestQuery(
                query_text=f"Tell me about the conversation where: {memory.content[:80]}...",
                query_type="episodic",
                ground_truth_memory_ids={memory.memory_id},
                description=f"episodic for memory {memory.timestamp}",
            )
        )
        print(f"  episodic: memory {memory.timestamp}")

    return queries


# =============================================================================
# Retrieval Strategies
# =============================================================================


def compute_similarity(
    query_embedding: np.ndarray,
    doc_embeddings: list[np.ndarray],
) -> list[float]:
    """Compute cosine similarity."""
    return [float(np.dot(query_embedding, doc)) for doc in doc_embeddings]


def retrieve_naive_similarity(
    query: str,
    memories: list[MemorySample],
    embedding_service: EmbeddingService,
    top_k: int = 5,
) -> list[str]:
    """Baseline: pure similarity retrieval."""
    query_emb = np.array(embedding_service.encode(query))
    memory_embs = [np.array(embedding_service.encode(m.content)) for m in memories]
    scores = compute_similarity(query_emb, memory_embs)

    ranked = sorted(zip(memories, scores), key=lambda x: x[1], reverse=True)
    return [m.memory_id for m, _ in ranked[:top_k]]


def retrieve_naive_recency(
    query: str,
    memories: list[MemorySample],
    embedding_service: EmbeddingService,
    top_k: int = 5,
    recency_weight: float = 0.5,
) -> list[str]:
    """Similarity + recency weighting."""
    query_emb = np.array(embedding_service.encode(query))
    memory_embs = [np.array(embedding_service.encode(m.content)) for m in memories]
    sim_scores = compute_similarity(query_emb, memory_embs)

    n = len(memories)
    recency_scores = [m.timestamp / (n - 1) if n > 1 else 1.0 for m in memories]

    combined = [
        (1 - recency_weight) * sim + recency_weight * rec
        for sim, rec in zip(sim_scores, recency_scores)
    ]

    ranked = sorted(zip(memories, combined), key=lambda x: x[1], reverse=True)
    return [m.memory_id for m, _ in ranked[:top_k]]


def retrieve_fact_similarity(
    query: str,
    facts: list[TypedFact],
    embedding_service: EmbeddingService,
    top_k: int = 5,
) -> list[str]:
    """Match query against extracted facts, return source memories."""
    query_emb = np.array(embedding_service.encode(query))

    # Embed each fact value
    fact_texts = [f"{f.entity} {f.attribute}: {f.value}" for f in facts]
    fact_embs = [np.array(embedding_service.encode(t)) for t in fact_texts]

    scores = compute_similarity(query_emb, fact_embs)

    # Get best score per memory
    memory_scores: dict[str, float] = {}
    for fact, score in zip(facts, scores):
        if fact.source_memory_id not in memory_scores:
            memory_scores[fact.source_memory_id] = score
        else:
            memory_scores[fact.source_memory_id] = max(
                memory_scores[fact.source_memory_id], score
            )

    ranked = sorted(memory_scores.items(), key=lambda x: x[1], reverse=True)
    return [mem_id for mem_id, _ in ranked[:top_k]]


def retrieve_attribute_aware(
    query: str,
    entity: str,
    facts: list[TypedFact],
    top_k: int = 5,
) -> list[str]:
    """Use typed facts to get correct memory set for an entity."""
    entity_facts = [f for f in facts if f.entity == entity]
    if not entity_facts:
        return []

    ground_truth = get_ground_truth_memories(entity, entity_facts)
    return list(ground_truth)[:top_k]


def retrieve_attribute_aware_specific(
    entity: str,
    attribute: str,
    facts: list[TypedFact],
) -> list[str]:
    """Use typed facts to get correct memory for a specific attribute."""
    attr_facts = [f for f in facts if f.entity == entity and f.attribute == attribute]
    if not attr_facts:
        return []

    if attr_facts[0].attribute_type == "replacement":
        # Return most recent
        most_recent = max(attr_facts, key=lambda f: f.timestamp)
        return [most_recent.source_memory_id]
    else:
        # Return all
        return [f.source_memory_id for f in attr_facts]


def retrieve_attribute_aware_episodic(
    query: str,
    facts: list[TypedFact],
    embedding_service: EmbeddingService,
    top_k: int = 5,
) -> list[str]:
    """Use fact matching for episodic queries - find facts that match the query."""
    query_emb = np.array(embedding_service.encode(query))

    # Match against fact values
    fact_texts = [f.value for f in facts]
    fact_embs = [np.array(embedding_service.encode(t)) for t in fact_texts]

    scores = compute_similarity(query_emb, fact_embs)

    # Return source memories for best matching facts
    scored_facts = sorted(zip(facts, scores), key=lambda x: x[1], reverse=True)

    seen_memories: set[str] = set()
    result: list[str] = []
    for fact, _ in scored_facts:
        if fact.source_memory_id not in seen_memories:
            seen_memories.add(fact.source_memory_id)
            result.append(fact.source_memory_id)
            if len(result) >= top_k:
                break

    return result


# =============================================================================
# Metrics
# =============================================================================


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


# =============================================================================
# Experiment Runner
# =============================================================================


def run_experiment() -> None:
    """Run the attribute-aware retrieval experiment."""
    print("\n" + "=" * 70)
    print("ATTRIBUTE-AWARE RETRIEVAL EXPERIMENT")
    print("=" * 70)
    print("Key insight: replacement vs additive attributes need different handling")

    llm = create_llm()
    model = SupportedModel.MISTRAL_SMALL_3_2_Q4
    embedding_service = get_embedding_service()

    # Load memories
    data_path = CONVERSATIONS_DIR / DATA_FILE
    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}")
        return

    memories = load_memories(data_path, max_samples=10)
    print(f"\nLoaded {len(memories)} memories")

    # Extract typed facts
    all_facts = extract_all_facts(memories, llm, model)
    print(f"\nTotal facts extracted: {len(all_facts)}")

    # Show fact type distribution
    replacement_count = sum(1 for f in all_facts if f.attribute_type == "replacement")
    additive_count = sum(1 for f in all_facts if f.attribute_type == "additive")
    print(f"  Replacement: {replacement_count}")
    print(f"  Additive: {additive_count}")

    # Generate queries
    queries = generate_test_queries(all_facts, memories)
    print(f"\nTotal queries: {len(queries)}")

    by_type = defaultdict(list)
    for q in queries:
        by_type[q.query_type].append(q)

    for qt, qs in by_type.items():
        print(f"  {qt}: {len(qs)}")

    # Run retrieval for each strategy
    print("\n" + "-" * 70)
    print("RUNNING RETRIEVAL COMPARISONS")
    print("-" * 70)

    strategies = {
        "naive_sim": lambda q: retrieve_naive_similarity(
            q.query_text, memories, embedding_service
        ),
        "naive_rec": lambda q: retrieve_naive_recency(
            q.query_text, memories, embedding_service
        ),
        "fact_sim": lambda q: retrieve_fact_similarity(
            q.query_text, all_facts, embedding_service
        ),
    }

    # For attribute_aware, we need the entity/attribute from the query
    def extract_entity_from_query(query: TestQuery) -> str:
        if query.query_type == "entity_overview":
            # "What do I know about {entity}?"
            return (
                query.query_text.replace("What do I know about ", "")
                .replace("?", "")
                .strip()
            )
        return ""

    def extract_entity_attribute_from_query(query: TestQuery) -> tuple[str, str]:
        # "What is {entity}'s current {attribute}?"
        # description format: "specific_attribute: {entity}.{attribute}"
        if "specific_attribute:" in query.description:
            parts = query.description.replace("specific_attribute: ", "").split(".")
            if len(parts) == 2:
                return parts[0], parts[1]
        return "", ""

    results: dict[str, dict[str, list[dict[str, float]]]] = {
        qt: {s: [] for s in list(strategies.keys()) + ["attr_aware"]}
        for qt in ["entity_overview", "specific_attribute", "episodic"]
    }

    for query in queries:
        for strategy_name, strategy_fn in strategies.items():
            retrieved = strategy_fn(query)
            metrics = compute_metrics(
                retrieved, query.ground_truth_memory_ids, query.query_type
            )
            results[query.query_type][strategy_name].append(metrics)

        # Attribute-aware for all query types
        if query.query_type == "entity_overview":
            entity = extract_entity_from_query(query)
            retrieved = retrieve_attribute_aware(query.query_text, entity, all_facts)
            metrics = compute_metrics(
                retrieved, query.ground_truth_memory_ids, query.query_type
            )
            results[query.query_type]["attr_aware"].append(metrics)

        elif query.query_type == "specific_attribute":
            entity, attribute = extract_entity_attribute_from_query(query)
            retrieved = retrieve_attribute_aware_specific(entity, attribute, all_facts)
            metrics = compute_metrics(
                retrieved, query.ground_truth_memory_ids, query.query_type
            )
            results[query.query_type]["attr_aware"].append(metrics)

        elif query.query_type == "episodic":
            retrieved = retrieve_attribute_aware_episodic(
                query.query_text, all_facts, embedding_service
            )
            metrics = compute_metrics(
                retrieved, query.ground_truth_memory_ids, query.query_type
            )
            results[query.query_type]["attr_aware"].append(metrics)

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS BY QUERY TYPE")
    print("=" * 70)

    for query_type in ["entity_overview", "specific_attribute", "episodic"]:
        type_results = results[query_type]
        print(f"\n--- {query_type.upper()} ---")

        if query_type == "entity_overview":
            print(f"(Ground truth = set of memories with current state)")
            print(
                f"\n{'Strategy':<12} {'Recall':<10} {'Precision':<12} {'F1':<10} {'Count'}"
            )
            print("-" * 55)

            for strategy_name in ["naive_sim", "naive_rec", "fact_sim", "attr_aware"]:
                evals = type_results[strategy_name]
                if not evals:
                    continue

                avg_recall = sum(e["recall"] for e in evals) / len(evals)
                avg_precision = sum(e["precision"] for e in evals) / len(evals)
                avg_f1 = sum(e["f1"] for e in evals) / len(evals)

                print(
                    f"{strategy_name:<12} {avg_recall:<10.3f} {avg_precision:<12.3f} {avg_f1:<10.3f} {len(evals)}"
                )

        else:
            print(f"(Ground truth = specific memory)")
            print(f"\n{'Strategy':<12} {'MRR':<10} {'Count'}")
            print("-" * 30)

            for strategy_name in ["naive_sim", "naive_rec", "fact_sim", "attr_aware"]:
                evals = type_results[strategy_name]
                if not evals:
                    continue

                avg_mrr = sum(e["mrr"] for e in evals) / len(evals)
                print(f"{strategy_name:<12} {avg_mrr:<10.3f} {len(evals)}")

    # Summary
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    for query_type in ["entity_overview", "specific_attribute", "episodic"]:
        type_results = results[query_type]
        print(f"\n{query_type}:")

        if query_type == "entity_overview":
            if type_results["attr_aware"]:
                attr_aware_f1 = sum(e["f1"] for e in type_results["attr_aware"]) / len(
                    type_results["attr_aware"]
                )
                naive_sim_f1 = sum(e["f1"] for e in type_results["naive_sim"]) / len(
                    type_results["naive_sim"]
                )
                print(f"  Attribute-aware F1: {attr_aware_f1:.3f}")
                print(f"  Naive similarity F1: {naive_sim_f1:.3f}")
                if attr_aware_f1 > naive_sim_f1:
                    print(
                        f"  -> Attribute-aware wins by {(attr_aware_f1 - naive_sim_f1) / naive_sim_f1 * 100:.1f}%"
                    )
        else:
            if type_results["attr_aware"]:
                attr_aware_mrr = sum(
                    e["mrr"] for e in type_results["attr_aware"]
                ) / len(type_results["attr_aware"])
                naive_sim_mrr = sum(e["mrr"] for e in type_results["naive_sim"]) / len(
                    type_results["naive_sim"]
                )
                best_naive = max(
                    sum(e["mrr"] for e in type_results["naive_sim"])
                    / len(type_results["naive_sim"]),
                    sum(e["mrr"] for e in type_results["naive_rec"])
                    / len(type_results["naive_rec"]),
                )
                print(f"  Attribute-aware MRR: {attr_aware_mrr:.3f}")
                print(f"  Best naive MRR: {best_naive:.3f}")
                if attr_aware_mrr > best_naive:
                    print(f"  -> Attribute-aware wins")
                else:
                    print(f"  -> Naive approach wins")


if __name__ == "__main__":
    run_experiment()
