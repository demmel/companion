# Memory System Architecture

## Overview

This document describes the target architecture for the companion agent's memory system, derived from findings across five prior experiments. Three parallel experiments are now implementing and validating components of this architecture.

## Core Insight

**Different memory operations require fundamentally different retrieval strategies.**

| Query Type | Best Strategy | Evidence |
|------------|---------------|----------|
| Current state ("What is X wearing?") | KG lookup, most recent | 10x better F1 vs similarity |
| History ("Remember when...") | Similarity search | 18x better MRR vs KG |
| Entity overview ("Tell me about X") | KG with replacement/additive logic | Structured facts outperform narrative |
| Temporal ("What happened yesterday?") | Episode index + time filtering | Untested, but episodes provide natural structure |
| Continuity ("How did the interview go?") | Topic clusters + recency | Cross-action KNN finds semantic topics |

A unified system must **classify the query type first**, then route to the appropriate strategy.

## Target Architecture

```
User Input
    │
    ▼
┌─────────────────────────┐
│   Reference Detection   │  ← Extract entities, topics, time refs
│   (always-on)           │     100% recall achieved in experiments
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│   Query Classification  │  ← Determine retrieval strategy
│                         │     Target: 90%+ accuracy
└─────────────────────────┘
    │
    ├─── current_state ────────► KG Lookup
    │                            └─ Entity → Attribute → Most recent value
    │
    ├─── history ──────────────► Similarity Search
    │                            └─ Query embedding → Top-K memories
    │
    ├─── entity_overview ──────► KG Aggregation
    │                            └─ All facts (replacement + additive)
    │
    ├─── temporal ─────────────► Episode Index
    │                            └─ Time filter → Episode summaries
    │
    ├─── continuity ───────────► Topic + Recency
    │                            └─ Recent memories in same topic cluster
    │
    └─── no_retrieval ─────────► Skip
            │
            ▼
┌─────────────────────────┐
│   Context Assembly      │  ← Format for LLM consumption
└─────────────────────────┘
            │
            ▼
      Agent Response
```

## Scale Requirements

The system must handle decades of conversation (potentially millions of memories):

- **Knowledge Graph**: Bounded growth via replacement facts (current state overwrites previous)
- **Episode Summaries**: ~1000x compression (51 tokens avg per episode)
- **Topic Clusters**: Cross-action-type KNN to avoid action-type dominated clusters
- **Raw Memories**: Possibly only kept for recent window

## Experiments

Three parallel experiments are validating and implementing components:

### 1. Query Classification (`query_classification/`)

**Goal**: Build accurate, fast query type classifier

**Approach**:
- Create 250+ labeled queries with human verification
- Test LLM zero-shot, few-shot, and embedding classifiers
- Target 90%+ accuracy, <50ms for embedding classifier

**Output**: Production-ready classifier module

### 2. Unified Retrieval (`unified_retrieval/`)

**Goal**: Build end-to-end retrieval pipeline

**Approach**:
- Integrate existing components (KG, similarity, episodes, topics)
- Test scale approaches for decades of data
- Include baseline LLM classifier (swappable)

**Output**: Working pipeline ready for agent integration

### 3. Temporal Retrieval (`temporal_retrieval/`)

**Goal**: Build time-based query handling

**Approach**:
- Parse relative ("yesterday"), absolute ("January 15"), and emotional ("when I was stressed") time references
- Build time-indexed episode lookup
- Compare retrieval strategies (summaries only vs drill-down)

**Output**: Time parser + episode index + evaluation results

## Key Findings from Prior Experiments

### Episode Summaries
- Hybrid LLM + rule-based filtering: 95%+ boundary quality
- 226 episodes from 6,653 memories
- Short summaries: 921x compression

### Retrieval
- KG-aware: F1=0.707 for state queries (vs 0.077 similarity)
- Similarity: MRR=0.500 for episodic queries (vs 0.028 KG)
- Reference detection reframed to "what references exist?" achieves 100% recall

### Topic Clustering
- Cross-action-type KNN (k=15) discovers semantic topics
- Embeddings encode action type strongly; must filter

### Memory Extraction
- <1% hallucination rate
- Extraction not the bottleneck; retrieval design is

## Integration Path

After experiments complete:

1. **Query Classification** produces optimized classifier
2. **Unified Retrieval** validates pipeline architecture
3. **Temporal Retrieval** provides time handling module
4. Combine into production memory system
5. Replace current naive similarity-only retrieval

## Files

- `query_classification/PLAN.md` - Query classifier experiment
- `unified_retrieval/PLAN.md` - Pipeline experiment
- `temporal_retrieval/PLAN.md` - Time handling experiment
- Prior experiment findings in `episode_summaries/`, `retrieval/`, `topic_clustering/`, `memory_extraction/`, `dreams/`
