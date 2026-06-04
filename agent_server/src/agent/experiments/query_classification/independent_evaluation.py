"""
Independent evaluation of query classifiers.

This evaluation uses a test set created by a DIFFERENT source (Claude) than
the training data (Mistral Small 3.2), testing true generalization.

Key differences from original experiment:
1. Test queries written by different LLM (Claude, not Mistral)
2. Labels assigned with explicit reasoning
3. No overlap with few-shot examples
4. Includes deliberately ambiguous edge cases
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from agent.embedding_service import EmbeddingService
from agent.llm import SupportedModel, create_llm

from .classifiers.embedding_classifier import EmbeddingClassifier
from .classifiers.llm_few_shot import LLMFewShotClassifier
from .classifiers.llm_zero_shot import LLMZeroShotClassifier
from .models import ClassificationResult, QueryType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EXPERIMENT_DIR = Path(__file__).parent
MODELS_DIR = EXPERIMENT_DIR / "output" / "models"
RESULTS_DIR = EXPERIMENT_DIR / "output" / "results"

# Few-shot examples to EXCLUDE from test set (prevent leakage)
FEW_SHOT_EXAMPLES = {
    "What is David wearing?",
    "Where does Sarah work?",
    "What has David worn this week?",
    "Remember when we talked about cooking?",
    "What do you know about Sarah?",
    "Tell me about my dog",
    "What happened yesterday?",
    "How was I feeling in December?",
    "How did the interview go?",
    "Any update on that?",
    "I saw Sarah at the coffee shop",
    "My mom called me today",
    "Hello!",
    "What time is it?",
}


@dataclass
class LabeledQuery:
    """A query with its label and reasoning."""
    query: str
    label: str
    reasoning: str
    is_ambiguous: bool = False
    alternative_labels: list[str] | None = None


# Independent test set - written by Claude, not Mistral
# Each query has explicit reasoning for its label
INDEPENDENT_TEST_SET: list[LabeledQuery] = [
    # ===== CURRENT_STATE (asking for current attribute value) =====
    LabeledQuery(
        query="What color is my car?",
        label="current_state",
        reasoning="Asking for a current attribute (color) of an entity (car)",
    ),
    LabeledQuery(
        query="Who is my dentist?",
        label="current_state",
        reasoning="Asking for current relationship/attribute",
    ),
    LabeledQuery(
        query="What medication am I taking?",
        label="current_state",
        reasoning="Asking for current state of medication",
    ),
    LabeledQuery(
        query="How old is my nephew?",
        label="current_state",
        reasoning="Asking for current attribute (age)",
    ),
    LabeledQuery(
        query="What's my wifi password?",
        label="current_state",
        reasoning="Asking for a stored current value",
    ),

    # ===== HISTORY (past events, changes over time) =====
    LabeledQuery(
        query="What movies have I watched this year?",
        label="history",
        reasoning="Asking about accumulated events over time",
    ),
    LabeledQuery(
        query="When did I last go to the gym?",
        label="history",
        reasoning="Asking about a past event",
    ),
    LabeledQuery(
        query="What restaurants have we discussed?",
        label="history",
        reasoning="Asking about past conversation topics",
    ),
    LabeledQuery(
        query="How has my sleep been lately?",
        label="history",
        reasoning="Asking about patterns/changes over time",
    ),
    LabeledQuery(
        query="What did I tell you about my boss?",
        label="history",
        reasoning="Asking about past conversation content",
    ),

    # ===== ENTITY_OVERVIEW (everything about an entity) =====
    LabeledQuery(
        query="Fill me in on my brother",
        label="entity_overview",
        reasoning="Requesting comprehensive info about entity",
    ),
    LabeledQuery(
        query="What's the deal with Project Alpha?",
        label="entity_overview",
        reasoning="Requesting overview of a topic/project",
    ),
    LabeledQuery(
        query="Remind me about the Johnson account",
        label="entity_overview",
        reasoning="Requesting all known info about entity",
    ),
    LabeledQuery(
        query="Who is Dr. Martinez again?",
        label="entity_overview",
        reasoning="Requesting entity overview/refresh",
    ),
    LabeledQuery(
        query="Give me the rundown on my car situation",
        label="entity_overview",
        reasoning="Requesting comprehensive status/info",
    ),

    # ===== TEMPORAL (time-bounded queries) =====
    LabeledQuery(
        query="What meetings do I have tomorrow?",
        label="temporal",
        reasoning="Explicitly time-bounded (tomorrow)",
    ),
    LabeledQuery(
        query="What was happening in March?",
        label="temporal",
        reasoning="Explicitly time-bounded (March)",
    ),
    LabeledQuery(
        query="Anything important from last weekend?",
        label="temporal",
        reasoning="Explicitly time-bounded (last weekend)",
    ),
    LabeledQuery(
        query="What were we working on in Q3?",
        label="temporal",
        reasoning="Explicitly time-bounded (Q3)",
    ),
    LabeledQuery(
        query="How did Monday go?",
        label="temporal",
        reasoning="Explicitly time-bounded (Monday)",
    ),

    # ===== CONTINUITY (following up on ongoing situations) =====
    LabeledQuery(
        query="Did they ever get back to you?",
        label="continuity",
        reasoning="Following up on unresolved situation",
    ),
    LabeledQuery(
        query="Is that still happening?",
        label="continuity",
        reasoning="Checking status of ongoing situation",
    ),
    LabeledQuery(
        query="What's the latest on that?",
        label="continuity",
        reasoning="Following up on recent topic",
    ),
    LabeledQuery(
        query="Did you sort out the issue?",
        label="continuity",
        reasoning="Following up on problem resolution",
    ),
    LabeledQuery(
        query="Where did we land on that decision?",
        label="continuity",
        reasoning="Following up on pending decision",
    ),

    # ===== PROACTIVE_CONTEXT (statements needing entity context) =====
    LabeledQuery(
        query="I bumped into Karen at the store",
        label="proactive_context",
        reasoning="Statement mentioning entity - need Karen's context",
    ),
    LabeledQuery(
        query="My therapist said something interesting",
        label="proactive_context",
        reasoning="Statement mentioning entity - need therapist context",
    ),
    LabeledQuery(
        query="The project deadline got moved",
        label="proactive_context",
        reasoning="Statement about entity - need project context",
    ),
    LabeledQuery(
        query="I'm thinking about what John said",
        label="proactive_context",
        reasoning="Statement mentioning entity - need John context",
    ),
    LabeledQuery(
        query="Remember that restaurant? I went back",
        label="proactive_context",
        reasoning="Reference to entity needing context retrieval",
    ),

    # ===== NO_RETRIEVAL (no memory lookup needed) =====
    LabeledQuery(
        query="Can you help me write an email?",
        label="no_retrieval",
        reasoning="Task request, no memory needed",
    ),
    LabeledQuery(
        query="What's 15% of 230?",
        label="no_retrieval",
        reasoning="Calculation, no memory needed",
    ),
    LabeledQuery(
        query="How do I make french toast?",
        label="no_retrieval",
        reasoning="General knowledge question",
    ),
    LabeledQuery(
        query="Thanks for your help!",
        label="no_retrieval",
        reasoning="Social pleasantry",
    ),
    LabeledQuery(
        query="Never mind, I figured it out",
        label="no_retrieval",
        reasoning="Conversation closer, no retrieval needed",
    ),

    # ===== AMBIGUOUS EDGE CASES =====
    # These test the boundaries between types

    LabeledQuery(
        query="My sister called yesterday",
        label="proactive_context",
        reasoning="Primary need is sister's context, time is incidental",
        is_ambiguous=True,
        alternative_labels=["temporal"],
    ),
    LabeledQuery(
        query="What's going on with the renovation?",
        label="continuity",
        reasoning="Following up on ongoing situation",
        is_ambiguous=True,
        alternative_labels=["entity_overview"],
    ),
    LabeledQuery(
        query="I finally finished the book",
        label="proactive_context",
        reasoning="Statement that may need book context",
        is_ambiguous=True,
        alternative_labels=["continuity"],
    ),
    LabeledQuery(
        query="How's mom doing?",
        label="current_state",
        reasoning="Asking for current state of a person",
        is_ambiguous=True,
        alternative_labels=["entity_overview", "continuity"],
    ),
    LabeledQuery(
        query="What did we decide about the trip?",
        label="continuity",
        reasoning="Following up on a decision",
        is_ambiguous=True,
        alternative_labels=["history"],
    ),
]


def verify_no_leakage() -> bool:
    """Verify no test queries match few-shot examples."""
    test_queries = {q.query for q in INDEPENDENT_TEST_SET}
    overlap = test_queries & FEW_SHOT_EXAMPLES
    if overlap:
        logger.error(f"Data leakage detected! Overlapping queries: {overlap}")
        return False
    logger.info("No data leakage detected")
    return True


def evaluate_classifier(
    name: str,
    classify_fn: callable,
    test_set: list[LabeledQuery],
) -> dict:
    """Evaluate a classifier on the independent test set."""
    results = []
    correct = 0
    correct_or_alternative = 0

    for item in test_set:
        result = classify_fn(item.query)

        # Handle different return types
        if isinstance(result, ClassificationResult):
            predicted = result.predicted_type.value
            confidence = result.confidence
        else:
            predicted = result.predicted_type.value if hasattr(result.predicted_type, 'value') else str(result.predicted_type)
            confidence = result.confidence

        is_correct = predicted == item.label
        is_alternative = item.alternative_labels and predicted in item.alternative_labels

        if is_correct:
            correct += 1
            correct_or_alternative += 1
        elif is_alternative:
            correct_or_alternative += 1

        results.append({
            "query": item.query,
            "true_label": item.label,
            "predicted": predicted,
            "correct": is_correct,
            "correct_or_alternative": is_correct or is_alternative,
            "confidence": confidence,
            "is_ambiguous": item.is_ambiguous,
            "reasoning": item.reasoning,
        })

    n = len(test_set)
    n_ambiguous = sum(1 for q in test_set if q.is_ambiguous)
    n_clear = n - n_ambiguous

    clear_correct = sum(1 for r, q in zip(results, test_set) if r["correct"] and not q.is_ambiguous)

    return {
        "name": name,
        "accuracy": correct / n,
        "accuracy_with_alternatives": correct_or_alternative / n,
        "clear_cases_accuracy": clear_correct / n_clear if n_clear > 0 else 0,
        "total": n,
        "correct": correct,
        "ambiguous_count": n_ambiguous,
        "predictions": results,
    }


def main() -> None:
    """Run independent evaluation."""
    print("=" * 70)
    print("INDEPENDENT EVALUATION")
    print("=" * 70)
    print(f"\nTest set: {len(INDEPENDENT_TEST_SET)} queries")
    print(f"  - Created by: Claude (different from Mistral training data)")
    print(f"  - Clear cases: {sum(1 for q in INDEPENDENT_TEST_SET if not q.is_ambiguous)}")
    print(f"  - Ambiguous cases: {sum(1 for q in INDEPENDENT_TEST_SET if q.is_ambiguous)}")

    # Verify no data leakage
    if not verify_no_leakage():
        return

    # Load classifiers
    llm = create_llm()
    embedding_service = EmbeddingService()

    results = {}

    # 1. LLM Zero-Shot
    print("\n--- Evaluating: LLM Zero-Shot ---")
    zero_shot = LLMZeroShotClassifier(llm, SupportedModel.MISTRAL_SMALL_3_2_Q4)
    results["llm_zero_shot"] = evaluate_classifier(
        "llm_zero_shot",
        zero_shot.classify,
        INDEPENDENT_TEST_SET,
    )

    # 2. LLM Few-Shot
    print("\n--- Evaluating: LLM Few-Shot ---")
    few_shot = LLMFewShotClassifier(llm, SupportedModel.MISTRAL_SMALL_3_2_Q4)
    results["llm_few_shot"] = evaluate_classifier(
        "llm_few_shot",
        few_shot.classify,
        INDEPENDENT_TEST_SET,
    )

    # 3. Embedding Logistic (if model exists)
    logistic_path = MODELS_DIR / "logistic_classifier.pkl"
    if logistic_path.exists():
        print("\n--- Evaluating: Embedding Logistic ---")
        embedding_clf = EmbeddingClassifier(embedding_service, classifier_type="logistic")
        embedding_clf.load(logistic_path)
        results["embedding_logistic"] = evaluate_classifier(
            "embedding_logistic",
            embedding_clf.classify,
            INDEPENDENT_TEST_SET,
        )

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS (Independent Test Set)")
    print("=" * 70)

    print(f"\n{'Classifier':<25} {'Accuracy':>10} {'w/ Alt':>10} {'Clear Only':>12}")
    print("-" * 60)

    for name, r in sorted(results.items(), key=lambda x: -x[1]["accuracy"]):
        print(
            f"{name:<25} "
            f"{r['accuracy']:>10.1%} "
            f"{r['accuracy_with_alternatives']:>10.1%} "
            f"{r['clear_cases_accuracy']:>12.1%}"
        )

    # Error analysis
    print("\n" + "=" * 70)
    print("ERROR ANALYSIS")
    print("=" * 70)

    best_classifier = max(results.keys(), key=lambda x: results[x]["accuracy"])
    best_results = results[best_classifier]

    print(f"\nErrors from {best_classifier}:")
    for pred in best_results["predictions"]:
        if not pred["correct"]:
            alt_note = ""
            if pred["correct_or_alternative"]:
                alt_note = " (acceptable alternative)"
            print(f"  '{pred['query']}'")
            print(f"    Expected: {pred['true_label']}, Got: {pred['predicted']}{alt_note}")
            print(f"    Reasoning: {pred['reasoning']}")
            print()

    # Save results
    output_path = RESULTS_DIR / "independent_evaluation.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")

    # Compare to original results
    print("\n" + "=" * 70)
    print("COMPARISON TO ORIGINAL (POTENTIALLY BIASED) RESULTS")
    print("=" * 70)

    original_path = RESULTS_DIR / "summary.json"
    if original_path.exists():
        with open(original_path) as f:
            original = json.load(f)

        print(f"\n{'Classifier':<25} {'Original':>10} {'Independent':>12} {'Delta':>10}")
        print("-" * 60)

        for name in results:
            if name in original:
                orig_acc = original[name]["accuracy"]
                new_acc = results[name]["accuracy"]
                delta = new_acc - orig_acc
                print(
                    f"{name:<25} "
                    f"{orig_acc:>10.1%} "
                    f"{new_acc:>12.1%} "
                    f"{delta:>+10.1%}"
                )


if __name__ == "__main__":
    main()
