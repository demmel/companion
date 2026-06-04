# Unified Retrieval Architecture Experiment

## Goal

Build an end-to-end retrieval pipeline that combines proven strategies (KG for state, similarity for episodic, episodes for temporal) with intelligent routing. The system must scale to decades of conversation and run every turn with minimal latency.

## How This Fits

**Read first:** `../memory_architecture/ARCHITECTURE.md` - describes the overall memory system architecture.

This experiment builds the **Unified Retrieval Pipeline** that integrates all components. Other parallel experiments are building:
- Query Classification (`../query_classification/`) - classifier optimization
- Temporal Retrieval (`../temporal_retrieval/`) - time-based query handling

This experiment includes its own baseline classifier, so it can proceed independently.

## Context

From prior experiments:
- KG-aware: ~10x better for state queries (F1=0.707 vs 0.077)
- Similarity: ~18x better for episodic queries (MRR=0.500 vs 0.028)
- Episode detection: 95%+ boundary quality with hybrid LLM+rules
- Topic clustering: Cross-action KNN finds semantic topics

**Scale requirement:** Decades of conversation = potentially millions of memories. Need efficient indexing from the start.

**Always-on:** Retrieval runs every turn to provide proactive context.

## Architecture

```
Every Turn
    │
    ▼
┌─────────────────────────┐
│   Reference Detection   │  ← Extract entities, topics, time refs from user input
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│   Query Classification  │  ← LLM zero-shot baseline (swappable)
└─────────────────────────┘
    │
    ├─── current_state ────────► KG Lookup
    │                            └─ Entity → Attribute → Most recent value
    │
    ├─── history ──────────────► Similarity Search
    │                            └─ Query embedding → Top-K memories
    │
    ├─── entity_overview ──────► KG Aggregation
    │                            └─ All facts for entity (replacement + additive logic)
    │
    ├─── temporal ─────────────► Episode Index
    │                            └─ Time filter → Episode summaries
    │
    ├─── continuity ───────────► Topic + Recency
    │                            └─ Recent memories in same topic cluster
    │
    ├─── proactive_context ────► Multi-Strategy
    │                            └─ For each detected reference, fetch appropriate context
    │
    └─── no_retrieval ─────────► Skip
```

## Experiment Design

### Phase 1: Component Integration

Gather and adapt existing implementations:

| Component | Source | Adaptation Needed |
|-----------|--------|-------------------|
| Reference detection | `retrieval/query_generation.py` | Extract to reusable module |
| Query classifier | Build in this experiment | LLM zero-shot baseline (see below) |
| KG infrastructure | `retrieval/knowledge_graph.py` | Add incremental updates |
| Similarity search | `retrieval/kg_retrieval.py` | Optimize for scale |
| Episode index | `episode_summaries/detection.py` | Add time-based lookup |
| Topic clusters | `topic_clustering/clustering.py` | Add query-to-cluster matching |

#### Baseline Query Classifier

This experiment includes its own LLM-based zero-shot classifier:

```python
class LLMQueryClassifier:
    """Zero-shot LLM classifier for query type routing."""

    QUERY_TYPES = [
        "current_state",   # "What is X wearing?" → KG most recent
        "history",         # "What has X worn?" → similarity search
        "entity_overview", # "What do I know about X?" → KG all facts
        "temporal",        # "What happened yesterday?" → episode index
        "continuity",      # "How did the interview go?" → recent + topic
        "proactive_context", # User mentions entity → fetch context
        "no_retrieval",    # "Hello!", "Thanks" → skip
    ]

    def classify(self, query: str, context: list[str]) -> str:
        """Classify query type using LLM."""
        # Zero-shot prompt with type definitions
        ...
```

This baseline enables the unified retrieval pipeline to function independently. If a separate query_classification experiment produces a better classifier, it can be swapped in later via the `QueryClassifier` protocol interface.

### Phase 2: Scale Exploration

Test different indexing approaches for decades-scale:

#### Approach A: Flat + Filtering
- Store all memories in vector DB
- Filter by time/entity/topic at query time
- Simple but may not scale

#### Approach B: Hierarchical Summarization
- Memories → Episodes → Periods → Years
- Query matches at appropriate level
- Drill down as needed

#### Approach C: KG Primary
- Extract all facts to knowledge graph
- Query KG for state, fall back to memories for narrative
- KG stays bounded even as memories grow

#### Approach D: Hybrid
- KG for facts/state (bounded growth)
- Episode summaries for narrative/events (compressed)
- Raw memories only for recent window

**Evaluation:** Test each with simulated decades of data (extrapolate from current conversation patterns).

### Phase 3: Pipeline Implementation

```python
@dataclass
class RetrievalContext:
    """Context returned to the agent for response generation."""
    query_type: str
    strategy_used: str

    # What was retrieved
    facts: list[Fact]           # From KG
    memories: list[Memory]       # From similarity search
    episodes: list[EpisodeSummary]  # From episode index

    # Formatted for LLM consumption
    context_text: str

    # Metadata
    latency_ms: float
    num_candidates_searched: int


class UnifiedRetriever:
    def __init__(
        self,
        classifier: QueryClassifier,
        kg: KnowledgeGraph,
        memory_index: MemoryIndex,
        episode_index: EpisodeIndex,
        topic_clusters: TopicClusters,
    ):
        ...

    def retrieve(
        self,
        user_input: str,
        conversation_context: list[str],
    ) -> RetrievalContext:
        """Run retrieval for a single turn."""
        # 1. Detect references
        refs = self.detect_references(user_input, conversation_context)

        # 2. Classify
        query_type = self.classifier.classify(user_input, refs)

        # 3. Route and retrieve
        return self._route_and_retrieve(query_type, refs, user_input)
```

### Phase 4: Evaluation

**Test Set:**
100+ queries spanning all types, with ground truth for expected retrieval.

**Metrics:**
| Metric | Description |
|--------|-------------|
| Precision | Retrieved context is relevant |
| Recall | Relevant context was retrieved |
| MRR | For single-answer queries |
| F1 | For multi-answer queries |
| Latency | End-to-end time |
| Context quality | LLM-judged usefulness for response |

**Baselines:**
1. Naive: Similarity search only (current behavior)
2. KG only: Everything through knowledge graph
3. Unified: Full pipeline

**Ablations:**
- Without reference detection
- Without query classification (random strategy)
- Without KG (similarity for everything)
- Without episode index (similarity for temporal)

### Phase 5: Incremental Update Design

For production, the system must update incrementally:

```python
class UnifiedRetriever:
    def on_new_memory(self, memory: Memory):
        """Called when a new memory is created."""
        # Update KG with extracted facts
        facts = self.extract_facts(memory)
        self.kg.add_facts(facts)

        # Update memory index
        self.memory_index.add(memory)

        # Topic cluster assignment (batch periodically)
        self.pending_cluster_updates.append(memory)

    def on_episode_boundary(self, episode: Episode):
        """Called when an episode is detected."""
        summary = self.summarize_episode(episode)
        self.episode_index.add(summary)
```

## Deliverables

1. `unified_retriever.py` - Main pipeline
2. `components/` - Integrated components (KG, index, etc.)
3. `scale_experiment.py` - Scale testing for different approaches
4. `test_queries.json` - Evaluation dataset
5. `evaluate.py` - Evaluation pipeline
6. `FINDINGS.md` - Results and recommendations

## Success Criteria

- [ ] All components integrated into working pipeline
- [ ] Scale approaches tested with simulated decade data
- [ ] Unified beats naive baseline on all query types
- [ ] Latency < 200ms p95 (excluding LLM calls)
- [ ] Incremental update path designed
- [ ] Ready for agent integration

## Running

```bash
# Build indices from conversation
uv run python -m agent.experiments.unified_retrieval.build_indices --conversation <id>

# Run scale experiment
uv run python -m agent.experiments.unified_retrieval.scale_experiment

# Evaluate pipeline
uv run python -m agent.experiments.unified_retrieval.evaluate

# Run ablations
uv run python -m agent.experiments.unified_retrieval.ablation
```

## Notes

- Use pydantic models for all data structures
- Annotate all function signatures
- Design for testability - mock components for unit tests
- This experiment is self-contained with its own baseline classifier
