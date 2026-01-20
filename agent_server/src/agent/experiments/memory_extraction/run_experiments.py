"""Main experiment runner for memory extraction experiment."""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from agent.chain_of_action.trigger_history_entry import TriggerHistoryEntry
from agent.embedding_service import get_embedding_service
from agent.llm import LLM, create_llm, SupportedModel
from agent.structured_llm import direct_structured_llm_call

from .extraction import extract_facts, extract_batch
from .evaluation import annotate_extraction, compute_metrics, print_annotation_summary
from .models import ExtractionResult, MemorySample, TestQueryGenerationResponse
from .prompts import GENERATE_TEST_QUERIES_PROMPT
from .retrieval import TestQuery, evaluate_retrieval, print_retrieval_comparison

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Paths
CONVERSATIONS_DIR = Path("conversations")
RESULTS_DIR = Path("src/agent/experiments/memory_extraction/results")
DATA_FILE = "conversation_20251024_083630_306692_triggers.json"


def load_memories_from_triggers(
    filepath: Path, max_samples: int = 30
) -> list[MemorySample]:
    """Load memory samples from a trigger history file."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    memories: list[MemorySample] = []

    for entry_data in data["entries"]:
        entry = TriggerHistoryEntry.model_validate(entry_data)

        # Extract from compressed_summary (primary source)
        if entry.compressed_summary and len(entry.compressed_summary) > 100:
            memories.append(
                MemorySample(
                    memory_id=entry.entry_id,
                    content=entry.compressed_summary,
                    source_type="compressed_summary",
                    timestamp=str(entry.timestamp),
                )
            )

        if len(memories) >= max_samples:
            break

    logger.info(f"Loaded {len(memories)} memory samples from {filepath}")
    return memories


def generate_test_queries_for_memory(
    memory: MemorySample,
    llm: LLM,
    model: SupportedModel,
) -> list[TestQuery]:
    """Generate test queries for a single memory using LLM."""
    prompt = GENERATE_TEST_QUERIES_PROMPT.format(content=memory.content)

    try:
        response = direct_structured_llm_call(
            prompt=prompt,
            response_model=TestQueryGenerationResponse,
            model=model,
            llm=llm,
            caller="generate_test_queries",
        )

        queries = []
        for query_text in response.queries:
            queries.append(
                TestQuery(
                    query_text=query_text,
                    expected_memory_id=memory.memory_id,
                    description=f"Generated from memory {memory.memory_id[:15]}...",
                )
            )
        return queries
    except Exception as e:
        logger.error(f"Failed to generate queries for memory {memory.memory_id}: {e}")
        return []


def create_test_queries(
    memories: list[MemorySample],
    llm: LLM,
    model: SupportedModel,
) -> list[TestQuery]:
    """Create test queries by generating queries from each memory.

    This ensures queries are distributed across different memories,
    with each query mapped to its source memory for ground truth.
    """
    all_queries: list[TestQuery] = []

    print(f"Generating test queries from {len(memories)} memories...")

    for memory in memories:
        queries = generate_test_queries_for_memory(memory, llm, model)
        all_queries.extend(queries)
        if queries:
            print(f"  {memory.memory_id[:20]}...: {len(queries)} queries generated")

    print(f"Total test queries generated: {len(all_queries)}")
    return all_queries


def run_experiment_1_approach_comparison(
    memories: list[MemorySample],
    llm: LLM,
    model: SupportedModel,
) -> dict:
    """
    Experiment 1: Compare extraction approaches WITH quality metrics.

    For each approach, measures:
    - Facts per memory
    - Compression ratio
    - Accuracy rate (% CORRECT)
    - Hallucination rate (% HALLUCINATED)
    - Inference rate (% INFERRED)
    - Omission count
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Approach Comparison (with quality metrics)")
    print("=" * 60)

    approaches = ["A", "B", "C", "D", "E"]
    results = {}

    # Use smaller subset for speed since we're running accuracy on ALL approaches
    sample_memories = memories[:5]
    print(f"Using {len(sample_memories)} memories for comparison")

    for approach in approaches:
        print(f"\n{'='*40}")
        print(f"Running approach {approach}...")
        print("=" * 40)

        approach_extractions: list[ExtractionResult] = []
        approach_annotations = []
        total_facts = 0
        total_compression = 0.0

        for memory in sample_memories:
            try:
                # Extract facts
                result = extract_facts(
                    content=memory.content,
                    approach=approach,
                    llm=llm,
                    model=model,
                    memory_id=memory.memory_id,
                )
                approach_extractions.append(result)
                total_facts += len(result.facts)
                total_compression += result.compression_ratio

                # Annotate for accuracy
                if result.facts:
                    annotation = annotate_extraction(result, llm, model)
                    approach_annotations.append(annotation)
                    print(
                        f"  {memory.memory_id[:20]}...: {len(result.facts)} facts, "
                        f"accuracy={annotation.accuracy_rate:.0%}, "
                        f"hallucination={annotation.hallucination_rate:.0%}"
                    )
                else:
                    print(f"  {memory.memory_id[:20]}...: 0 facts extracted")

            except Exception as e:
                logger.error(f"Failed approach {approach} on {memory.memory_id}: {e}")

        # Compute aggregate metrics
        num_processed = len(approach_extractions)
        avg_facts = total_facts / num_processed if num_processed else 0
        avg_compression = total_compression / num_processed if num_processed else 0

        # Compute accuracy metrics across all annotations
        metrics = compute_metrics(approach_annotations)

        results[approach] = {
            "avg_facts_per_memory": avg_facts,
            "avg_compression_ratio": avg_compression,
            "total_memories_processed": num_processed,
            "accuracy_rate": metrics["accuracy_rate"],
            "hallucination_rate": metrics["hallucination_rate"],
            "inference_rate": metrics["inference_rate"],
            "avg_omissions_per_memory": metrics["avg_omissions_per_memory"],
            "total_facts": metrics["total_facts"],
            "extractions": approach_extractions,
        }

        print(f"\nApproach {approach} Summary:")
        print(f"  Avg facts per memory: {avg_facts:.1f}")
        print(f"  Avg compression ratio: {avg_compression:.2f}")
        print(f"  Accuracy rate: {metrics['accuracy_rate']:.1%}")
        print(f"  Hallucination rate: {metrics['hallucination_rate']:.1%}")
        print(f"  Inference rate: {metrics['inference_rate']:.1%}")
        print(f"  Avg omissions: {metrics['avg_omissions_per_memory']:.1f}")

    # Print comparison table
    print("\n" + "=" * 80)
    print("APPROACH COMPARISON (full metrics)")
    print("=" * 80)
    print(
        f"{'Approach':<10} {'Facts':<8} {'Compress':<10} {'Accuracy':<10} {'Halluc':<10} {'Infer':<10} {'Omit':<8}"
    )
    print("-" * 80)
    for approach in approaches:
        r = results[approach]
        print(
            f"{approach:<10} {r['avg_facts_per_memory']:<8.1f} {r['avg_compression_ratio']:<10.2f} "
            f"{r['accuracy_rate']:<10.1%} {r['hallucination_rate']:<10.1%} "
            f"{r['inference_rate']:<10.1%} {r['avg_omissions_per_memory']:<8.1f}"
        )

    return results


def select_best_approach(results: dict) -> str:
    """
    Select the best approach based on quality criteria.

    Criteria (in order):
    1. Hallucination rate < 5% (hard requirement)
    2. Highest accuracy rate
    3. Lowest omission rate
    4. Reasonable compression (0.3-0.7x preferred)
    """
    # Filter approaches with hallucination rate < 5%
    valid_approaches = [a for a, r in results.items() if r["hallucination_rate"] < 0.05]

    if not valid_approaches:
        # If all approaches have high hallucination, warn and pick lowest
        print("WARNING: All approaches have hallucination rate >= 5%")
        valid_approaches = list(results.keys())
        valid_approaches.sort(key=lambda a: results[a]["hallucination_rate"])
        return valid_approaches[0]

    # Among valid approaches, pick highest accuracy
    best = max(valid_approaches, key=lambda a: results[a]["accuracy_rate"])

    print(f"\nBest approach selection:")
    print(f"  Valid approaches (hallucination < 5%): {valid_approaches}")
    print(f"  Selected: {best} (accuracy={results[best]['accuracy_rate']:.1%})")

    return best


def run_experiment_2_accuracy(
    memories: list[MemorySample],
    best_approach: str,
    llm,
    model: SupportedModel,
) -> dict:
    """
    Experiment 2: Accuracy annotation.

    Annotate facts from best approach as CORRECT/HALLUCINATED/INFERRED.
    """
    print("\n" + "=" * 60)
    print(f"EXPERIMENT 2: Accuracy Annotation (Approach {best_approach})")
    print("=" * 60)

    # Use subset for annotation
    sample_memories = memories[:5]

    extractions: list[ExtractionResult] = []
    for memory in sample_memories:
        try:
            result = extract_facts(
                content=memory.content,
                approach=best_approach,
                llm=llm,
                model=model,
                memory_id=memory.memory_id,
            )
            extractions.append(result)
        except Exception as e:
            logger.error(f"Extraction failed: {e}")

    # Annotate each extraction
    annotation_results = []
    for extraction in extractions:
        print(
            f"\nAnnotating {extraction.memory_id[:20]}... ({len(extraction.facts)} facts)"
        )
        try:
            annotation = annotate_extraction(extraction, llm, model)
            annotation_results.append(annotation)
            print_annotation_summary(annotation)
        except Exception as e:
            logger.error(f"Annotation failed: {e}")

    # Compute aggregate metrics
    metrics = compute_metrics(annotation_results)

    print("\n" + "-" * 60)
    print("AGGREGATE METRICS")
    print("-" * 60)
    print(f"Total facts: {metrics['total_facts']}")
    print(f"Accuracy rate: {metrics['accuracy_rate']:.1%}")
    print(f"Hallucination rate: {metrics['hallucination_rate']:.1%}")
    print(f"Inference rate: {metrics['inference_rate']:.1%}")
    print(f"Avg omissions per memory: {metrics['avg_omissions_per_memory']:.1f}")

    return {
        "approach": best_approach,
        "metrics": metrics,
        "num_memories": len(annotation_results),
    }


def run_experiment_3_retrieval(
    memories: list[MemorySample],
    extractions: list[ExtractionResult],
    llm: LLM,
    model: SupportedModel,
) -> dict:
    """
    Experiment 3: Retrieval impact.

    Compare raw vs extracted vs hybrid search.
    Uses dynamically generated test queries from each memory.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Retrieval Impact")
    print("=" * 60)

    embedding_service = get_embedding_service()

    # Generate test queries dynamically from the memories
    test_queries = create_test_queries(memories, llm, model)

    if not test_queries:
        print("No test queries available")
        return {}

    # Show which memories have queries
    memory_ids_with_queries = set(q.expected_memory_id for q in test_queries)
    print(
        f"\nQueries distributed across {len(memory_ids_with_queries)} different memories"
    )
    print(f"Running {len(test_queries)} test queries...")

    # Show examples
    for query in test_queries[:3]:
        print_retrieval_comparison(
            query=query.query_text,
            memories=memories,
            extractions=extractions,
            expected_memory_id=query.expected_memory_id,
            embedding_service=embedding_service,
            top_k=3,
        )

    # Compute overall metrics
    metrics = evaluate_retrieval(
        test_queries=test_queries,
        memories=memories,
        extractions=extractions,
        embedding_service=embedding_service,
    )

    print("\n" + "-" * 60)
    print("RETRIEVAL METRICS")
    print("-" * 60)
    print(f"Raw MRR: {metrics['raw_mrr']:.3f}")
    print(f"Extracted MRR: {metrics['extracted_mrr']:.3f}")
    print(f"Hybrid MRR: {metrics['hybrid_mrr']:.3f}")
    print(f"Raw wins: {metrics['raw_wins']}")
    print(f"Extracted wins: {metrics['extracted_wins']}")
    print(f"Hybrid wins: {metrics['hybrid_wins']}")

    return metrics


def save_results(results: dict, filename: str) -> None:
    """Save results to JSON file."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    filepath = RESULTS_DIR / filename

    # Convert non-serializable objects
    def make_serializable(obj):
        if isinstance(obj, ExtractionResult):
            return {
                "memory_id": obj.memory_id,
                "num_facts": len(obj.facts),
                "compression_ratio": obj.compression_ratio,
                "summary": obj.summary,
            }
        elif isinstance(obj, list):
            return [make_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        else:
            return obj

    serializable_results = make_serializable(results)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(serializable_results, f, indent=2, default=str)

    print(f"Results saved to {filepath}")


def main():
    """Run all experiments."""
    print("\n" + "=" * 60)
    print("MEMORY EXTRACTION EXPERIMENT")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 60)

    # Initialize
    llm = create_llm()
    model = SupportedModel.MISTRAL_SMALL_3_2_Q4

    # Load data
    data_path = CONVERSATIONS_DIR / DATA_FILE
    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}")
        sys.exit(1)

    memories = load_memories_from_triggers(data_path, max_samples=30)

    if not memories:
        print("Error: No memories loaded")
        sys.exit(1)

    print(f"Loaded {len(memories)} memories")

    # Run experiments
    all_results = {"timestamp": datetime.now().isoformat()}

    # Experiment 1: Approach comparison (now includes accuracy for ALL approaches)
    exp1_results = run_experiment_1_approach_comparison(memories, llm, model)
    all_results["experiment_1"] = {
        approach: {
            "avg_facts_per_memory": r["avg_facts_per_memory"],
            "avg_compression_ratio": r["avg_compression_ratio"],
            "accuracy_rate": r["accuracy_rate"],
            "hallucination_rate": r["hallucination_rate"],
            "inference_rate": r["inference_rate"],
            "avg_omissions_per_memory": r["avg_omissions_per_memory"],
        }
        for approach, r in exp1_results.items()
    }

    # Select best approach using quality criteria (not just fact count)
    best_approach = select_best_approach(exp1_results)

    # Experiment 2: Deep dive on best approach
    exp2_results = run_experiment_2_accuracy(memories, best_approach, llm, model)
    all_results["experiment_2"] = exp2_results

    # Get extractions for retrieval experiment
    print("\nGenerating extractions for retrieval experiment...")
    extractions = extract_batch(memories[:10], best_approach, llm, model)

    # Experiment 3: Retrieval (now with dynamically generated queries)
    exp3_results = run_experiment_3_retrieval(memories[:10], extractions, llm, model)
    all_results["experiment_3"] = exp3_results

    # Save all results
    save_results(
        all_results, f"experiment_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
