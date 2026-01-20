# Retrieval Experiments - Plan

## Background

The memory extraction experiment revealed that extraction works reasonably well, but the real challenge is retrieval. Before testing whether extraction helps retrieval, we need to design retrieval mechanisms that work well for companion agents.

## Companion Agent Retrieval Needs

A companion agent needs to handle diverse retrieval scenarios:

| Query Type | Example | Retrieval Logic |
|------------|---------|-----------------|
| **Fact** | "What's my dog's name?" | Find relevant fact; if updated, latest wins |
| **State** | "What am I wearing?" | Most recent state update for this attribute |
| **Episodic** | "Remember when I was stressed about X?" | Specific moment with emotional context |
| **Relationship** | "Who is Sarah?" | Structured knowledge about a person |
| **Pattern** | "When do I usually feel tired?" | Aggregation across memories |
| **Proactive** | (no explicit query) | Surface relevant memories based on context |

---

## Experiment 1: Query Type Classification

**Goal**: Determine if we can reliably classify incoming queries by type.

**Hypothesis**: Different query types need different retrieval strategies. If we can classify queries, we can route them appropriately.

**Method**:
1. Create a dataset of 50+ example queries covering all types
2. Test LLM classification accuracy
3. Measure inter-annotator agreement (human baseline)

**Metrics**:
- Classification accuracy
- Confusion matrix between types
- Edge cases and ambiguous queries

---

## Experiment 2: Temporal Retrieval

**Goal**: Handle state/temporal queries correctly.

**Problem**: For "What is she wearing?", the correct answer is the *most recent* appearance change, not the first mention.

**Method**:
1. Create test dataset with temporal sequences (e.g., multiple appearance changes)
2. Test different retrieval strategies:
   - A: Naive embedding similarity (current approach)
   - B: Recency-weighted similarity
   - C: State tracking (overwrite old state with new)
   - D: Temporal index + similarity

**Metrics**:
- Accuracy on temporal queries
- False positives (returning outdated info)

---

## Experiment 3: Episodic Retrieval

**Goal**: Retrieve specific moments and emotional memories.

**Problem**: Episodic queries like "Remember when we talked about my job interview?" need to find a specific moment, not just related facts.

**Method**:
1. Create test dataset of episodic queries with ground truth memories
2. Test different retrieval strategies:
   - A: Embedding similarity on full memory
   - B: Embedding similarity on extracted summary
   - C: Event-based indexing
   - D: Emotional context matching

**Metrics**:
- MRR for correct episode retrieval
- Qualitative: Does the retrieved memory capture the right moment?

---

## Experiment 4: Fact Retrieval with Updates

**Goal**: Handle facts that change over time.

**Problem**: "What city does she live in?" may have multiple answers across memories if the user moved.

**Method**:
1. Create test dataset with fact evolution (e.g., moved cities, changed jobs)
2. Test different strategies:
   - A: Return most recent mention
   - B: Return all mentions with timestamps
   - C: Maintain fact store with versioning
   - D: LLM synthesis from multiple memories

**Metrics**:
- Accuracy of returned fact (most current)
- Ability to retrieve fact history when asked

---

## Experiment 5: Proactive Retrieval

**Goal**: Surface relevant memories without explicit query.

**Problem**: During conversation, the agent should recall relevant context without the user asking.

**Method**:
1. Given conversation context, what memories should surface?
2. Test different strategies:
   - A: Similarity to recent conversation
   - B: Entity matching (people/topics mentioned)
   - C: Emotional state matching
   - D: Routine/pattern matching

**Metrics**:
- Human evaluation: Is the surfaced memory relevant?
- Does it enhance conversation without being intrusive?

---

## Implementation Order

1. **Experiment 1 (Classification)**: Foundation for routing
2. **Experiment 2 (Temporal)**: Critical for companion correctness
3. **Experiment 4 (Facts with updates)**: Most common use case
4. **Experiment 3 (Episodic)**: Important for emotional connection
5. **Experiment 5 (Proactive)**: Advanced feature

---

## Test Data Strategy

Rather than using LLM-as-judge (unreliable), we need:

1. **Synthetic data with known ground truth**:
   - Generate conversation sequences with controlled state changes
   - Each query has a definitively correct answer

2. **Human annotation**:
   - Small set of real memories with human-labeled answers
   - Multiple annotators for agreement measurement

3. **Functional tests**:
   - Specific scenarios with expected behavior
   - E.g., "After 3 appearance changes, query should return memory #3"

---

## Files to Create

- `query_classifier.py` - Classify queries by type
- `temporal_retrieval.py` - Handle state/temporal queries
- `episodic_retrieval.py` - Handle moment-based queries
- `fact_retrieval.py` - Handle fact queries with updates
- `proactive_retrieval.py` - Context-based memory surfacing
- `test_data/` - Synthetic and annotated test datasets
- `evaluation.py` - Evaluation framework
- `run_experiments.py` - Orchestrate all experiments

---

## Success Criteria

1. **Classification accuracy > 90%** for query type routing
2. **Temporal queries return correct (most recent) state > 95%**
3. **Fact queries return current value > 90%**
4. **Episodic queries return correct memory in top-3 > 80%**
5. **Proactive retrieval judged helpful > 70% by human eval**

---

## Next Steps

1. Create synthetic test data with controlled temporal sequences
2. Implement query classifier
3. Run Experiment 2 (temporal) first - this addresses the main flaw found in extraction experiments
