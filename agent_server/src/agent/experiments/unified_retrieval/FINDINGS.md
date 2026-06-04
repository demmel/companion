# Unified Retrieval Experiment - Findings

## Executive Summary

This experiment evaluates retrieval strategies using **proper IR metrics** (precision, recall, F1, MRR) against **ground truth datasets**.

Previous evaluation approaches were flawed:
- **Circular evaluation**: Test queries and classifier patterns written to match each other
- **Answerability metric**: Meaningless - "Can we answer the query?" doesn't tell us if we retrieved the right content
- **No ground truth**: Without knowing what SHOULD be retrieved, we can't measure accuracy

## Proper Evaluation Framework

### The Core Question

Evaluation must answer: **"Did we retrieve the right content?"**

NOT:
- "Is the response better with context?" (requires subjective judgment)
- "Can we answer the query?" (circular if we wrote both query and expected answer)
- "What's the classification accuracy?" (irrelevant if retrieval result is the same)

### Ground Truth Dataset

Test queries with known expected results:

```json
{
  "query": "What is Sarah wearing?",
  "query_type": "current_state",
  "expected_memory_ids": ["entry_456"],
  "expected_entity": "sarah",
  "expected_attribute": "clothing",
  "confidence": 0.9,
  "needs_review": false
}
```

For each query, we know exactly which memories/facts SHOULD be retrieved.

### IR Metrics

| Metric | Description |
|--------|-------------|
| **Precision** | What % of retrieved items are in the expected set |
| **Recall** | What % of expected items were retrieved |
| **F1** | Harmonic mean of precision and recall |
| **MRR** | Mean Reciprocal Rank (1/rank of first correct answer) |

### Running Evaluation

```bash
# Generate ground truth dataset (semi-automated)
uv run python -m agent.experiments.unified_retrieval.generate_test_queries \
    --conversation <id> \
    --output test_queries_groundtruth.json

# Review and correct the suggested ground truth
# Set needs_review=false for verified queries

# Run evaluation with proper metrics
uv run python -m agent.experiments.unified_retrieval.evaluate \
    --dataset test_queries_groundtruth.json \
    --conversation <id>
```

## Strategies Compared

| Strategy | Description |
|----------|-------------|
| **similarity_only** | Pure embedding similarity search (baseline) |
| **kg_only** | Route everything through knowledge graph |
| **unified** | Full pipeline with query classification and routing |

## What Went Wrong Previously

### Circular Evaluation (Initial)

The initial experiment measured "classification accuracy" by:
1. Writing test queries with expected types (e.g., `"What is X wearing?" → current_state`)
2. Writing classifier patterns (e.g., `"what is" → current_state`)
3. Measuring if they match

This is meaningless - of course they match, both sides were written together.

### Answerability Metric (Second Attempt)

The "real evaluation" used LLM-as-judge to rate "answerability":
- 75% of queries were "answerable" with simple similarity search
- But this doesn't mean we retrieved the RIGHT content
- Memories often contain paraphrases of user input (self-matching)

### Missing Temporal Isolation

Previous experiments didn't isolate retrieval temporally:
- When evaluating turn N, we retrieved from ALL memories
- This included the current turn's own memory (circular)

## Files

| File | Purpose |
|------|---------|
| `generate_test_queries.py` | Semi-automated ground truth generation |
| `evaluate.py` | Proper IR evaluation with precision/recall |
| `test_queries_groundtruth.json` | Ground truth dataset (after review) |
| `unified_retriever.py` | Main retrieval pipeline |
| `query_classifier.py` | Query type classification |
| `models.py` | Data structures |
| `build_indices.py` | Index construction |

## Deleted Files

The following flawed implementations were removed:
- `real_evaluation.py` - Used meaningless "answerability" metric
- `temporal_retrieval/retrieval_value_experiment.py` - Flawed methodology
- `temporal_retrieval/results_baseline.json` - Results from flawed experiment

## Ground Truth Generation Process

### Semi-Automated Pipeline

1. **Extract candidate queries** from conversation history:
   - Find user messages with entity references ("Sarah", "the interview")
   - Find messages with time references ("yesterday", "last week")
   - Find messages asking about state ("what is X wearing?")

2. **Auto-suggest expected results** using heuristics:
   - For entity queries: search memories mentioning that entity
   - For time queries: find memories in that time range
   - For state queries: find most recent fact about attribute
   - Score by similarity and recency

3. **Output for human review**:
   - Each query marked with `needs_review: true`
   - Reviewer verifies/corrects suggestions
   - Sets `needs_review: false` for confirmed queries

4. **Run evaluation** on verified queries only

### Query Type Distribution

The generator aims for balanced coverage:
- `current_state`: Queries about current entity attributes
- `history`: Queries about past events/changes
- `entity_overview`: General questions about an entity
- `temporal`: Time-based queries ("yesterday", "last week")
- `continuity`: Follow-up questions on ongoing topics
- `proactive_context`: Statements mentioning known entities

## Lessons Learned

1. **Write the evaluation first** before building the system
2. **Use real data** not synthetic queries
3. **Measure end-to-end** not intermediate metrics
4. **Ground truth is essential** - without it, metrics are meaningless
5. **Be skeptical** of impressive numbers
6. **Temporal isolation** - don't retrieve from the current turn's memory

## Scale Results (Still Valid)

The scale experiment used proper methodology:

| Memories | P95 Latency | QPS |
|----------|-------------|-----|
| 1,000 | 26ms | 43 |
| 5,000 | 87ms | 12 |
| 10,000 | 157ms | 6.5 |

Similarity search scales well to 10K+ memories.

## Next Steps

1. Generate ground truth for a real conversation
2. Have human review verify the suggestions
3. Run proper evaluation with the verified dataset
4. Compare strategies and identify per-query-type winners
5. Update retrieval pipeline based on objective results
