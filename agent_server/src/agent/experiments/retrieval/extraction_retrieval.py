"""Test whether extraction improves retrieval - PROPERLY DESIGNED.

Key fix: Ground truth must be correct for each query type.
- current_state: correct answer = MOST RECENT memory mentioning that entity
- episodic: correct answer = specific source memory
"""

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from agent.embedding_service import EmbeddingService, get_embedding_service
from agent.llm import LLM, SupportedModel, create_llm
from agent.structured_llm import direct_structured_llm_call

from agent.experiments.memory_extraction.extraction import extract_facts
from agent.experiments.memory_extraction.models import ExtractionResult, MemorySample

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CONVERSATIONS_DIR = Path("conversations")
DATA_FILE = "conversation_20251024_083630_306692_triggers.json"


@dataclass
class TypedQuery:
    """A test query with proper ground truth."""

    query_text: str
    query_type: str  # "current_state" or "episodic"
    correct_memory_id: str  # The ACTUAL correct answer
    all_relevant_memory_ids: list[
        str
    ]  # All memories that could match (for current_state)
    description: str


class EntityExtractionResponse(BaseModel):
    """LLM response for entity extraction."""

    people: list[str] = Field(
        description="Names of people mentioned (e.g., Sarah, Mike, mom, boss)"
    )
    topics: list[str] = Field(
        description="Key topics discussed (e.g., work, health, relationship)"
    )


class EpisodicQueryResponse(BaseModel):
    """LLM response for generating episodic query."""

    query: str = Field(
        description="A query asking about a specific moment or event in this memory"
    )


ENTITY_EXTRACTION_PROMPT = """Extract key entities from this memory.

MEMORY:
{content}

List:
1. People mentioned (by name or relation like "mom", "boss")
2. Key topics discussed (work, health, hobbies, etc.)

Be consistent with naming - use the same name/term across memories."""


EPISODIC_QUERY_PROMPT = """Generate a query asking about a SPECIFIC moment or event in this memory.

MEMORY:
{content}

The query should:
- Ask about something specific that happened (not general facts)
- Use phrases like "remember when", "what happened when", "tell me about the time"
- Be answerable ONLY by this specific memory

Example: "Remember when you were stressed about the presentation?"
"""


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
                        source_type="compressed_summary",
                        timestamp=str(i),  # Use index as timestamp for ordering
                    )
                )

        if len(memories) >= max_samples:
            break

    logger.info(f"Loaded {len(memories)} memories")
    return memories


def extract_entities(
    memory: MemorySample,
    llm: LLM,
    model: SupportedModel,
) -> list[str]:
    """Extract entities/topics from a memory."""
    prompt = ENTITY_EXTRACTION_PROMPT.format(content=memory.content)

    try:
        response = direct_structured_llm_call(
            prompt=prompt,
            response_model=EntityExtractionResponse,
            model=model,
            llm=llm,
            caller="extract_entities",
        )
        # Normalize to lowercase for matching
        entities = [e.lower().strip() for e in response.people + response.topics]
        return list(set(entities))  # Deduplicate
    except Exception as e:
        logger.error(f"Entity extraction failed: {e}")
        return []


def build_entity_memory_map(
    memories: list[MemorySample],
    llm: LLM,
    model: SupportedModel,
) -> dict[str, list[int]]:
    """Build map of entity -> list of memory indices (ordered by time)."""
    entity_map: dict[str, list[int]] = defaultdict(list)

    print("\nExtracting entities from memories...")
    for i, memory in enumerate(memories):
        entities = extract_entities(memory, llm, model)
        print(f"  Memory {i}: {entities}")
        for entity in entities:
            entity_map[entity].append(i)

    return dict(entity_map)


def find_shared_entities(entity_map: dict[str, list[int]]) -> dict[str, list[int]]:
    """Filter to entities that appear in multiple memories."""
    return {
        entity: indices for entity, indices in entity_map.items() if len(indices) >= 2
    }


def generate_current_state_query(entity: str) -> str:
    """Generate a current_state query for an entity."""
    # Simple template-based generation
    if entity in ["mom", "dad", "brother", "sister", "boss", "partner"]:
        return f"What's going on with my {entity}?"
    elif entity in ["work", "job"]:
        return "What's happening with work?"
    elif entity in ["health", "sleep", "exercise"]:
        return f"How is my {entity} lately?"
    else:
        return f"What do I know about {entity}?"


def generate_episodic_query(
    memory: MemorySample,
    llm: LLM,
    model: SupportedModel,
) -> str:
    """Generate an episodic query for a specific memory."""
    prompt = EPISODIC_QUERY_PROMPT.format(content=memory.content)

    try:
        response = direct_structured_llm_call(
            prompt=prompt,
            response_model=EpisodicQueryResponse,
            model=model,
            llm=llm,
            caller="generate_episodic_query",
        )
        return response.query
    except Exception as e:
        logger.error(f"Episodic query generation failed: {e}")
        return ""


def create_test_queries(
    memories: list[MemorySample],
    shared_entities: dict[str, list[int]],
    llm: LLM,
    model: SupportedModel,
) -> list[TypedQuery]:
    """Create test queries with correct ground truth."""
    queries: list[TypedQuery] = []

    # Current_state queries: for shared entities, correct answer = most recent
    print("\nGenerating current_state queries for shared entities...")
    for entity, memory_indices in shared_entities.items():
        most_recent_idx = max(memory_indices)  # Highest index = most recent
        query_text = generate_current_state_query(entity)

        queries.append(
            TypedQuery(
                query_text=query_text,
                query_type="current_state",
                correct_memory_id=memories[most_recent_idx].memory_id,
                all_relevant_memory_ids=[memories[i].memory_id for i in memory_indices],
                description=f"current_state for '{entity}' (correct: memory {most_recent_idx})",
            )
        )
        print(f"  '{entity}': memories {memory_indices} -> correct = {most_recent_idx}")

    # Episodic queries: for each memory, correct answer = that memory
    print("\nGenerating episodic queries...")
    for i, memory in enumerate(memories):
        query_text = generate_episodic_query(memory, llm, model)
        if query_text:
            queries.append(
                TypedQuery(
                    query_text=query_text,
                    query_type="episodic",
                    correct_memory_id=memory.memory_id,
                    all_relevant_memory_ids=[memory.memory_id],
                    description=f"episodic for memory {i}",
                )
            )
            print(f"  Memory {i}: '{query_text[:50]}...'")

    return queries


def compute_similarity(
    query_embedding: np.ndarray,
    doc_embeddings: list[np.ndarray],
) -> list[float]:
    """Compute cosine similarity."""
    return [float(np.dot(query_embedding, doc)) for doc in doc_embeddings]


def retrieve_raw_similarity(
    query: str,
    memories: list[MemorySample],
    embedding_service: EmbeddingService,
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """Retrieve using raw memories + similarity."""
    query_emb = np.array(embedding_service.encode(query))
    memory_embs = [np.array(embedding_service.encode(m.content)) for m in memories]
    scores = compute_similarity(query_emb, memory_embs)
    ranked = sorted(zip(memories, scores), key=lambda x: x[1], reverse=True)
    return [(m.memory_id, s) for m, s in ranked[:top_k]]


def retrieve_raw_recency(
    query: str,
    memories: list[MemorySample],
    embedding_service: EmbeddingService,
    top_k: int = 5,
    recency_weight: float = 0.5,
) -> list[tuple[str, float]]:
    """Retrieve using raw memories + recency-weighted similarity."""
    query_emb = np.array(embedding_service.encode(query))
    memory_embs = [np.array(embedding_service.encode(m.content)) for m in memories]
    sim_scores = compute_similarity(query_emb, memory_embs)

    n = len(memories)
    recency_scores = [i / (n - 1) if n > 1 else 1.0 for i in range(n)]

    combined = [
        (1 - recency_weight) * sim + recency_weight * rec
        for sim, rec in zip(sim_scores, recency_scores)
    ]

    ranked = sorted(zip(memories, combined), key=lambda x: x[1], reverse=True)
    return [(m.memory_id, s) for m, s in ranked[:top_k]]


def retrieve_extracted_similarity(
    query: str,
    extractions: list[ExtractionResult],
    embedding_service: EmbeddingService,
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """Retrieve using extracted facts + similarity."""
    query_emb = np.array(embedding_service.encode(query))
    memory_scores: list[tuple[str, float]] = []

    for extraction in extractions:
        if not extraction.facts:
            memory_scores.append((extraction.memory_id, 0.0))
            continue

        fact_texts = [f.content for f in extraction.facts]
        fact_embs = [np.array(embedding_service.encode(t)) for t in fact_texts]
        similarities = compute_similarity(query_emb, fact_embs)
        max_sim = max(similarities) if similarities else 0.0
        memory_scores.append((extraction.memory_id, max_sim))

    ranked = sorted(memory_scores, key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


def retrieve_extracted_recency(
    query: str,
    extractions: list[ExtractionResult],
    embedding_service: EmbeddingService,
    top_k: int = 5,
    recency_weight: float = 0.5,
) -> list[tuple[str, float]]:
    """Retrieve using extracted facts + recency-weighted similarity."""
    query_emb = np.array(embedding_service.encode(query))
    n = len(extractions)
    memory_scores: list[tuple[str, float]] = []

    for i, extraction in enumerate(extractions):
        recency = i / (n - 1) if n > 1 else 1.0

        if not extraction.facts:
            memory_scores.append((extraction.memory_id, recency_weight * recency))
            continue

        fact_texts = [f.content for f in extraction.facts]
        fact_embs = [np.array(embedding_service.encode(t)) for t in fact_texts]
        similarities = compute_similarity(query_emb, fact_embs)
        max_sim = max(similarities) if similarities else 0.0

        combined = (1 - recency_weight) * max_sim + recency_weight * recency
        memory_scores.append((extraction.memory_id, combined))

    ranked = sorted(memory_scores, key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


def evaluate_retrieval(
    query: TypedQuery,
    results: list[tuple[str, float]],
) -> dict:
    """Evaluate retrieval results."""
    retrieved_ids = [r[0] for r in results]

    try:
        rank = retrieved_ids.index(query.correct_memory_id) + 1
        rr = 1.0 / rank
    except ValueError:
        rank = -1
        rr = 0.0

    return {
        "query": query.query_text,
        "query_type": query.query_type,
        "correct": query.correct_memory_id,
        "retrieved_top": retrieved_ids[0] if retrieved_ids else None,
        "rank": rank,
        "reciprocal_rank": rr,
        "found_in_top3": query.correct_memory_id in retrieved_ids[:3],
    }


def run_experiment() -> None:
    """Run the properly designed extraction vs retrieval experiment."""
    print("\n" + "=" * 70)
    print("EXTRACTION vs RETRIEVAL EXPERIMENT (Properly Designed)")
    print("=" * 70)
    print("Key fix: current_state correct answer = MOST RECENT memory")

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

    # Extract facts for later use
    print("\nExtracting facts (Approach A)...")
    extractions: list[ExtractionResult] = []
    for memory in memories:
        try:
            result = extract_facts(
                content=memory.content,
                approach="A",
                llm=llm,
                model=model,
                memory_id=memory.memory_id,
            )
            extractions.append(result)
            print(f"  {memory.memory_id[:15]}...: {len(result.facts)} facts")
        except Exception as e:
            logger.error(f"Extraction failed: {e}")

    # Build entity map and find shared entities
    entity_map = build_entity_memory_map(memories, llm, model)
    shared_entities = find_shared_entities(entity_map)
    print(f"\nFound {len(shared_entities)} entities appearing in multiple memories:")
    for entity, indices in shared_entities.items():
        print(f"  '{entity}': memories {indices}")

    if not shared_entities:
        print(
            "WARNING: No shared entities found. Cannot test current_state queries properly."
        )

    # Generate test queries with correct ground truth
    all_queries = create_test_queries(memories, shared_entities, llm, model)

    cs_queries = [q for q in all_queries if q.query_type == "current_state"]
    ep_queries = [q for q in all_queries if q.query_type == "episodic"]
    print(f"\nTotal queries: {len(all_queries)}")
    print(f"  current_state: {len(cs_queries)} (correct = most recent)")
    print(f"  episodic: {len(ep_queries)} (correct = source)")

    # Run retrieval
    print("\n" + "-" * 70)
    print("RUNNING RETRIEVAL COMPARISONS")
    print("-" * 70)

    strategies = {
        "raw_sim": lambda q: retrieve_raw_similarity(q, memories, embedding_service),
        "raw_rec": lambda q: retrieve_raw_recency(q, memories, embedding_service),
        "ext_sim": lambda q: retrieve_extracted_similarity(
            q, extractions, embedding_service
        ),
        "ext_rec": lambda q: retrieve_extracted_recency(
            q, extractions, embedding_service
        ),
    }

    results: dict[str, dict[str, list[dict]]] = {
        "current_state": {s: [] for s in strategies},
        "episodic": {s: [] for s in strategies},
    }

    for query in all_queries:
        for strategy_name, strategy_fn in strategies.items():
            retrieved = strategy_fn(query.query_text)
            eval_result = evaluate_retrieval(query, retrieved)
            results[query.query_type][strategy_name].append(eval_result)

    # Results
    print("\n" + "=" * 70)
    print("RESULTS BY QUERY TYPE")
    print("=" * 70)

    for query_type in ["current_state", "episodic"]:
        print(f"\n--- {query_type.upper()} QUERIES ---")
        if query_type == "current_state":
            print("(Correct answer = MOST RECENT memory mentioning entity)")
        else:
            print("(Correct answer = specific source memory)")

        print(f"\n{'Strategy':<12} {'MRR':<8} {'Top-3 Acc':<10} {'Count':<6}")
        print("-" * 40)

        for strategy_name in strategies:
            evals = results[query_type][strategy_name]
            if not evals:
                continue

            mrr = sum(e["reciprocal_rank"] for e in evals) / len(evals)
            top3_acc = sum(1 for e in evals if e["found_in_top3"]) / len(evals)

            print(f"{strategy_name:<12} {mrr:<8.3f} {top3_acc:<10.1%} {len(evals):<6}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for query_type in ["current_state", "episodic"]:
        if not results[query_type]["raw_sim"]:
            continue

        raw_sim_mrr = sum(
            e["reciprocal_rank"] for e in results[query_type]["raw_sim"]
        ) / len(results[query_type]["raw_sim"])
        raw_rec_mrr = sum(
            e["reciprocal_rank"] for e in results[query_type]["raw_rec"]
        ) / len(results[query_type]["raw_rec"])
        ext_sim_mrr = sum(
            e["reciprocal_rank"] for e in results[query_type]["ext_sim"]
        ) / len(results[query_type]["ext_sim"])
        ext_rec_mrr = sum(
            e["reciprocal_rank"] for e in results[query_type]["ext_rec"]
        ) / len(results[query_type]["ext_rec"])

        print(f"\n{query_type}:")
        print(f"  Raw similarity:       {raw_sim_mrr:.3f}")
        print(
            f"  Raw + recency:        {raw_rec_mrr:.3f}  {'<-- recency helps!' if raw_rec_mrr > raw_sim_mrr else ''}"
        )
        print(f"  Extracted similarity: {ext_sim_mrr:.3f}")
        print(
            f"  Extracted + recency:  {ext_rec_mrr:.3f}  {'<-- recency helps!' if ext_rec_mrr > ext_sim_mrr else ''}"
        )

        best = max(
            [
                ("raw_sim", raw_sim_mrr),
                ("raw_rec", raw_rec_mrr),
                ("ext_sim", ext_sim_mrr),
                ("ext_rec", ext_rec_mrr),
            ],
            key=lambda x: x[1],
        )
        print(f"  Best: {best[0]} ({best[1]:.3f})")


if __name__ == "__main__":
    run_experiment()
