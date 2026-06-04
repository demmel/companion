# Query Type Classification Experiment

## Goal

Build a fast, accurate query type classifier that routes queries to the optimal retrieval strategy. The classifier must handle both user-facing queries and agent proactive retrieval needs, while minimizing LLM calls for production use.

---

# Part 2: Query Extraction Redesign

## Problem Statement

The original experiment (below) solved the **wrong problem**:
- **Original**: Takes a standalone query → classifies its type
- **Needed**: Takes a user message + context → extracts queries with their types

## Why This Matters

In the real system, the input is not a clean query - it's a user message like:
> "I saw Sarah at the coffee shop, she looked tired"

The system needs to:
1. **Detect references** that need context (Sarah, tiredness implication)
2. **Generate queries** to fetch relevant context
3. **Assign types** to route each query correctly

Query type is **part of extraction**, not a separate classification step.

## Correct Flow

```
User Message: "I saw Sarah at the coffee shop, she looked tired"
Context: [Recent conversation about Sarah's job stress]
                           │
                           ▼
              ┌─────────────────────────┐
              │   Query Extraction      │
              │   (with type tagging)   │
              └─────────────────────────┘
                           │
                           ▼
Output: [
  {query: "Sarah", type: "entity_overview", reasoning: "Need Sarah's background"},
  {query: "Sarah current state", type: "current_state", reasoning: "For tiredness context"},
  {query: "Sarah job stress", type: "continuity", reasoning: "Reference to prior topic"}
]
```

## New Implementation

### Files Added

| File | Purpose |
|------|---------|
| `extractor.py` | LLM-based query extraction from message+context |
| `create_extraction_dataset.py` | Hand-crafted extraction examples |
| `evaluate_extraction.py` | Extraction quality evaluation |
| `models.py` | Added `ExtractedQuery`, `ExtractionResult`, etc. |

### Models

```python
class ExtractedQuery(BaseModel):
    query_text: str      # The query to run
    query_type: QueryType # current_state, history, entity_overview, etc.
    reference: str       # What triggered this (entity name, topic, etc.)
    reasoning: str       # Why this query would help

class ExtractionResult(BaseModel):
    queries: list[ExtractedQuery]
    context_summary: str  # Why these queries matter
```

### Extractor Usage

```python
from agent.experiments.query_classification.extractor import (
    QueryExtractor,
    ExtractorConfig,
)
from agent.llm import create_llm

llm = create_llm()
extractor = QueryExtractor(llm)

result = extractor.extract(
    message="I saw Sarah at the coffee shop, she looked tired",
    context=["Earlier we discussed Sarah's new job"],
)

for query in result.queries:
    print(f"{query.query_text} ({query.query_type.value})")
```

### Evaluation Metrics

The extraction evaluation measures:
- **Reference Detection Recall**: Did we find all important references?
- **Query Type Accuracy**: Is the type correct for retrieval routing?
- **No-Retrieval Detection**: Did we correctly identify no-retrieval cases?

### Running

```bash
# Test extractor with sample inputs
uv run python -m agent.experiments.query_classification.extractor

# Create extraction dataset
uv run python -m agent.experiments.query_classification.create_extraction_dataset

# Evaluate extraction quality
uv run python -m agent.experiments.query_classification.evaluate_extraction
```

## Relationship to Original Experiment

The original standalone classification experiment is still useful for:
- Understanding query type patterns
- Benchmarking LLM vs embedding classifiers
- Training data for the embedding approach

But the **extraction approach** is what the actual system needs.

---

# Part 1: Original Standalone Classification (Preserved)

## How This Fits

**Read first:** `../memory_architecture/ARCHITECTURE.md` - describes the overall memory system architecture.

This experiment builds the **Query Classification** component. The classifier determines which retrieval strategy to use for each query. Other parallel experiments are building:
- Unified Retrieval (`../unified_retrieval/`) - the full pipeline
- Temporal Retrieval (`../temporal_retrieval/`) - time-based query handling

## Context

This is a companion agent designed for decades-long conversations. Prior experiments showed:
- KG-aware retrieval is ~10x better for state queries
- Similarity retrieval is ~18x better for episodic queries
- **Wrong strategy = terrible results** - classification accuracy is critical
- Retrieval runs every turn (always-on proactive), so speed matters

## Query Types

| Type | Description | Examples | Retrieval Strategy |
|------|-------------|----------|-------------------|
| `current_state` | Current value of an attribute | "What is David wearing?", "Where does Sarah work?" | KG: most recent value |
| `history` | Past events or changes over time | "What has David worn?", "Remember when we talked about X?" | Similarity search on memories |
| `entity_overview` | Everything known about an entity | "What do I know about Sarah?", "Tell me about my dog" | KG: all facts (replacement + additive) |
| `temporal` | Time-bounded queries | "What happened yesterday?", "This morning...", "When I was stressed" | Episode index + time filtering |
| `continuity` | Following up on ongoing situations | "How did the interview go?", "Any update on that?" | Recent memories + topic matching |
| `proactive_context` | Agent needs context for user's message | User mentions "Sarah" → fetch Sarah context | Reference detection + KG/similarity |
| `no_retrieval` | No retrieval needed | "Hello!", "What time is it?", "Thanks" | Skip |

### Important Distinction: Current State vs History

This distinction is critical:
- "What is David's mood?" → `current_state` → return most recent mood
- "How has David's mood been?" → `history` → return mood changes over time
- "What was David's mood yesterday?" → `temporal` → time-bounded history

## Experiment Design

### Phase 1: Dataset Creation

Create 250+ queries with human-verified labels.

**Sources:**
1. **Real conversations**: Extract from existing conversation logs
2. **Generated**: Create queries programmatically from memory content
3. **Hand-crafted**: Write edge cases and ambiguous examples

**Label each query:**
```json
{
  "query": "What is Sarah wearing today?",
  "type": "current_state",
  "entities": ["Sarah"],
  "attributes": ["appearance", "clothing"],
  "time_reference": "today",
  "is_proactive": false,
  "reasoning": "Asking for current state of appearance attribute"
}
```

**Target distribution:**
- current_state: 40
- history: 40
- entity_overview: 30
- temporal: 40
- continuity: 30
- proactive_context: 40
- no_retrieval: 30

### Phase 2: Classifier Approaches

#### Approach A: LLM Zero-Shot
Use LLM to classify with type definitions in prompt. Baseline for accuracy.

#### Approach B: LLM Few-Shot
Add 2 examples per type. Should improve over zero-shot.

#### Approach C: Embedding Classifier
1. Embed all queries using same embedding model as memories
2. Train classifier (logistic regression, small MLP, or gradient boosting)
3. Cross-validate

**Why this matters:** If embedding classifier achieves 90%+ accuracy, we can avoid LLM calls for classification entirely.

#### Approach D: Two-Stage Hybrid
1. Fast embedding classifier for high-confidence cases
2. LLM fallback for low-confidence cases

This balances speed and accuracy.

### Phase 3: Evaluation

**Metrics:**
- Overall accuracy
- Per-class precision, recall, F1
- Confusion matrix (which types get confused?)
- Latency (ms per classification)
- Cost (LLM tokens used)

**Targets:**
- 90%+ overall accuracy
- 85%+ per-class F1
- <50ms for embedding classifier
- <500ms for LLM classifier

### Phase 4: Error Analysis

For misclassified queries:
1. Is ground truth ambiguous? (Some queries could be multiple types)
2. Are there patterns in errors?
3. Can prompt/features be improved?

### Phase 5: Proactive Context Classification

Test separately: Given user input, what references need context?

Input: "I saw Sarah at the coffee shop, she looked tired"
Output:
- Entity "Sarah" → fetch entity_overview
- Implicit reference to "coffee shop" → maybe fetch if it's a known place

This is different from classifying user *questions* - it's about detecting what background context would help the agent respond well.

## Deliverables

1. `dataset/` - Labeled query datasets (train/test split)
2. `classifiers/` - All classifier implementations
3. `evaluate.py` - Evaluation pipeline
4. `results/` - Metrics, confusion matrices, error analysis
5. `FINDINGS.md` - Results and recommendations
6. Production-ready classifier module

## Files to Reference

- `../retrieval/FINDINGS.md` - Prior classification attempts, query type analysis
- `../retrieval/query_generation.py` - Reference detection logic
- `../retrieval/knowledge_graph.py` - Entity resolver (useful for entity extraction)

## Success Criteria

- [ ] 250+ queries labeled with reasoning
- [ ] LLM and embedding classifiers implemented
- [ ] 90%+ overall accuracy achieved
- [ ] Embedding classifier tested (for LLM-free production path)
- [ ] Proactive context detection tested
- [ ] Clear recommendation for production use

## Running

```bash
# Create dataset
uv run python -m agent.experiments.query_classification.create_dataset

# Train embedding classifier
uv run python -m agent.experiments.query_classification.train

# Evaluate all approaches
uv run python -m agent.experiments.query_classification.evaluate

# Error analysis
uv run python -m agent.experiments.query_classification.analyze
```

## Notes

- Use pydantic models for data structures
- Annotate all function signatures with types
- Run tests before claiming success
- This experiment blocks unified_retrieval - that experiment depends on this classifier
