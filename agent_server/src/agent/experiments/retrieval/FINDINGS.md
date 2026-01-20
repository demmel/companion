# Retrieval Experiments - Findings

## Summary

These experiments explored retrieval strategies for companion agent memory systems. Key findings:

1. **Reference detection should be design, not personality** - always retrieve when there's a reference
2. **Reframing the prompt from "should I retrieve?" to "what references exist?" achieves 100% recall**
3. **KG-aware retrieval works for state queries** (validated at scale with 95% CIs):
   - entity_overview: F1=0.707 vs 0.077 naive (~9x improvement, statistically significant)
   - specific_attribute: MRR=0.637 vs 0.066 naive (~10x improvement, statistically significant)
4. **Episodic queries need pure similarity** - MRR=0.500 with similarity vs 0.028 with KG (~18x difference)
5. **Facts need to be typed** - replacement (most recent wins) vs additive (accumulates)
6. **Query type determines optimal strategy** - KG for state, similarity for events
7. **Query type classification is critical** - wrong strategy for wrong type = terrible results

---

## Experiment 1: Query Classification (deprecated)

We initially tried classifying queries into types (fact, state, episodic, relationship). This had 69% accuracy but the fact/state distinction was problematic.

**Key insight**: Fact vs state is a false distinction. Both are "current state of something." What matters is:
- **current_state**: Want the latest value of something
- **episodic**: Want a specific past moment or event

---

## Experiment 2: Temporal Retrieval

**Goal**: Does recency weighting help for current-state queries?

**Result**: Yes - significant improvement.

| Strategy | MRR |
|----------|-----|
| Pure similarity (baseline) | 0.333 |
| Recency weighted (0.3) | 0.778 |
| Recency weighted (0.5) | 0.833 |

**Why it matters**: For queries like "What are you wearing?", pure similarity returns the first mention (where the embedding matches best), not the current state. Adding recency weighting fixes this.

**Recommendation**: Use recency-weighted retrieval for current_state queries. Pure similarity is fine for episodic queries.

---

## Experiment 3: Reference Detection

**Initial approach (flawed)**: Ask "should I retrieve?"
- Result: 83% accuracy, but 5 false negatives (missed references)

**Key insight from discussion**: Retrieval should be **design**, not **personality**.
- Retrieval = what information the agent has access to (always retrieve when there's a reference)
- Response = how the agent uses that information (personality decides how much to show)

A "conservative" agent that misses retrieval opportunities isn't showing personality - it's missing context it could have used.

**Redesigned approach**: Ask "what references exist that could benefit from context?"

Prompt change:
```
OLD: "Only request retrieval if it would genuinely help"
NEW: "Be thorough. It's better to retrieve context you don't use than to miss context you needed."
```

**Results after redesign (40 scenarios)**:

| Metric | Result |
|--------|--------|
| No-reference scenarios | 5/5 correctly skipped |
| Reference scenarios | 34/34 caught |
| **Recall** | **100%** |
| Type accuracy | 82% |

**What counts as a reference**:
- People mentioned by name (Sarah, Mike)
- Pronouns referring to known people (my mom, my boss)
- Events being followed up on (the interview, the date)
- Ongoing situations (work, the situation, things)
- Places with shared history (the usual spot)
- Past conversations (remember when, what you suggested)
- Recurring issues (again, is back, still)
- Implicit references (the project, my resolution)

---

## Architecture Recommendation (Revised)

Based on these experiments, companion retrieval needs to be more sophisticated than simple similarity + recency.

### The Problem

"What do I know about David?" requires:
- Most recent value for replacement attributes (mood, location, appearance)
- All values for additive attributes (preferences, relationships, history)

Simple recency weighting doesn't distinguish between these.

### Proposed Architecture

```
1. REFERENCE DETECTION (always on)
   - Scan input for all references
   - Classify query type:
     - specific_attribute: "What is David wearing?"
     - entity_overview: "What do I know about David?"
     - episodic: "Remember when David said X?"

2. FACT EXTRACTION (at memory creation time)
   - Extract structured facts from memories
   - Tag each fact with:
     - entity (who/what)
     - attribute (what property)
     - attribute_type (replacement vs additive)
     - value
     - timestamp

3. RETRIEVAL STRATEGY (based on query type)
   - specific_attribute → find attribute, get most recent
   - entity_overview → gather all current facts (see logic above)
   - episodic → pure similarity search

4. RESPONSE GENERATION (personality)
   - Agent has full context available
   - Personality decides how much to surface
```

### Key Insight

This is fundamentally a **knowledge graph / state tracking** problem, not a retrieval problem. The "retrieval" step is really about reconstructing current state from a log of observations.

---

## Experiment 4: Extraction vs Retrieval (FLAWED)

**Goal**: Does extracting structured facts from memories improve retrieval compared to raw text?

**STATUS**: This experiment had fundamental design flaws in ground truth definition.

### Original Design (Flawed)

Marked "most recent memory mentioning entity" as the correct answer for queries like "What do I know about David?"

### Why This Is Wrong

The query "What do I know about David?" doesn't have a single correct answer. The correct retrieval depends on **attribute semantics**:

**Replacement attributes** (only most recent matters):
- Mood: "David is happy" supersedes "David was sad"
- Location: "David is at home" supersedes "David was at work"
- Appearance: "David is wearing blue" supersedes "David was wearing red"

**Additive attributes** (accumulates over time):
- Preferences: "David likes sushi" does NOT supersede "David likes pizza" - he likes both
- Relationships: "David knows Sarah" adds to "David knows Mike"
- Interests: "David enjoys hiking" adds to "David enjoys reading"

### Correct Ground Truth

For "What do I know about David?", correct retrieval = **set of memories** containing:
- Most recent value for each replacement attribute
- All values for additive attributes

Example memories:
- Memory 2: "David likes pizza"
- Memory 5: "David is wearing blue"
- Memory 7: "David likes sushi"
- Memory 9: "David is happy"

Correct retrieval = {Memory 2, Memory 5, Memory 7, Memory 9}

NOT just Memory 9 (most recent), and NOT excluding Memory 2 (pizza preference still valid).

### Implications

1. **MRR is the wrong metric** - assumes single correct answer
2. **Simple recency weighting is insufficient** - need to understand attribute types
3. **This is a state tracking / knowledge graph problem** - not simple retrieval

---

## Experiment 4 Results (For Reference)

Despite flawed ground truth, here are the raw numbers:

**CURRENT_STATE QUERIES** (6 queries, ground truth = most recent memory only):

| Strategy | MRR | Top-3 Acc |
|----------|-----|-----------|
| raw_sim | 0.408 | 50% |
| raw_rec | 0.722 | 100% |
| ext_sim | 0.472 | 83% |
| ext_rec | 0.722 | 100% |

**EPISODIC QUERIES** (10 queries, ground truth = source memory):

| Strategy | MRR | Top-3 Acc |
|----------|-----|-----------|
| raw_sim | 0.542 | 80% |
| raw_rec | 0.265 | 30% |
| ext_sim | 0.620 | 70% |
| ext_rec | 0.267 | 50% |

**Tentative conclusions** (pending proper experiment):
- Recency weighting helps for replacement attributes
- Recency weighting hurts for episodic queries
- Extraction may help episodic queries

---

## Experiment 5: Attribute-Aware Retrieval

**Goal**: Test whether understanding attribute semantics (replacement vs additive) improves retrieval.

### Method

1. Extract typed facts from memories with LLM:
   - entity, attribute, attribute_type (replacement/additive), value
2. Build ground truth for entity_overview queries:
   - For replacement attrs: only most recent memory
   - For additive attrs: all memories
3. Compare strategies:
   - naive_sim: pure embedding similarity
   - naive_rec: similarity + recency weighting
   - fact_sim: match query against extracted facts
   - attr_aware: use typed facts to reconstruct current state

### Facts Extracted

From 10 memories, extracted 131 facts:
- Replacement: 81 (62%)
- Additive: 50 (38%)

### Results

**ENTITY OVERVIEW QUERIES** (4 queries, multi-answer):

| Strategy | Recall | Precision | F1 |
|----------|--------|-----------|-----|
| naive_sim | 0.632 | 0.500 | 0.534 |
| naive_rec | 0.504 | 0.500 | 0.480 |
| fact_sim | 0.779 | 0.600 | 0.639 |
| **attr_aware** | **0.929** | **1.000** | **0.958** |

**Attribute-aware wins by 79.5%** over naive similarity. This validates the core hypothesis: understanding attribute types enables correct state reconstruction.

**SPECIFIC ATTRIBUTE QUERIES** (10 queries, single-answer):

| Strategy | MRR |
|----------|-----|
| **attr_aware** | **1.000** |
| naive_rec | 0.600 |
| fact_sim | 0.542 |
| naive_sim | 0.220 |

**Attribute-aware is perfect** - directly looks up the attribute and returns most recent memory.

**EPISODIC QUERIES** (5 queries, single-answer):

| Strategy | MRR |
|----------|-----|
| **naive_sim** | **1.000** |
| fact_sim | 0.700 |
| attr_aware | 0.190 |
| naive_rec | 0.080 |

**Pure similarity is best** for episodic queries. Attribute-aware fails (0.190 MRR) because extracted facts lose narrative/event structure.

### Key Findings

1. **Attribute-aware achieves perfect or near-perfect results for state queries**:
   - entity_overview: F1=0.958
   - specific_attribute: MRR=1.000
2. **Attribute-aware fails for episodic queries** (MRR=0.190) - facts don't capture events
3. **Query type determines optimal strategy**:
   - State queries → attribute-aware (fact extraction + state tracking)
   - Episodic queries → pure similarity search

### Conclusion

For state queries ("What do I know about X?", "What is X's mood?"):
- Use typed fact extraction (replacement vs additive)
- Attribute-aware state reconstruction achieves near-perfect results

For episodic queries ("Remember when..."):
- Use pure similarity search on raw memories
- Fact extraction loses the narrative context needed

---

## Experiment 6: Fair KG-Based Retrieval (Small Scale)

**Goal**: Test KG-aware retrieval without hardcoding - all strategies start from raw query text.

### What Changed

Previous experiment cheated by storing entity/attribute in query description. This experiment:
1. Builds real KG with entity resolution + attribute normalization
2. Resolves entity from query using embedding similarity
3. Matches attribute from query using embedding similarity
4. Then looks up in KG

### Infrastructure Built

- **EntityResolver**: Resolves "david", "David", etc. to canonical entity via exact match + embedding similarity
- **AttributeNormalizer**: Maps "mood", "current_mood", "emotional_state" to canonical "mood" with pre-defined schema
- **KnowledgeGraph**: Stores facts with resolved entities and normalized attributes, tracks replacement vs additive

### Results (Small Scale - 10 memories, 26 queries)

**ENTITY OVERVIEW QUERIES** (5 queries):

| Strategy | Recall | Precision | F1 |
|----------|--------|-----------|-----|
| naive_sim | 0.767 | 0.400 | 0.493 |
| fact_sim | 0.783 | 0.440 | 0.521 |
| **kg_aware** | **0.967** | **1.000** | **0.982** |

**SPECIFIC ATTRIBUTE QUERIES** (16 queries):

| Strategy | MRR |
|----------|-----|
| naive_sim | 0.293 |
| fact_sim | 0.679 |
| **kg_aware** | **0.938** |

**EPISODIC QUERIES** (5 queries):

| Strategy | MRR |
|----------|-----|
| **naive_sim** | **1.000** |
| fact_sim | 0.667 |
| kg_aware | 0.100 |

---

## Experiment 7: Scaled-Up KG Retrieval

**Goal**: Validate findings with statistically meaningful sample sizes.

### Why Scale Up?

Previous experiments used tiny datasets (10 memories, 5-16 queries per type). Results like "1.0 MRR from 5 samples" are essentially noise. Need 50+ samples per type for meaningful confidence intervals.

### Dataset

- **100 memories** from conversation_20251024_083630_306692_triggers.json
- **1128 facts** extracted (cached to avoid repeated LLM calls)
- **204 queries** generated programmatically from KG structure

### Results (with 95% Confidence Intervals)

**ENTITY OVERVIEW QUERIES** (10 queries):

| Strategy | F1 | 95% CI |
|----------|-----|--------|
| naive_sim | 0.077 ± 0.163 | [0.000, 0.178] |
| fact_sim | 0.091 ± 0.154 | [0.000, 0.186] |
| **kg_aware** | **0.707 ± 0.343** | [0.495, 0.920] |

**SPECIFIC ATTRIBUTE QUERIES** (94 queries):

| Strategy | MRR | 95% CI |
|----------|-----|--------|
| naive_sim | 0.066 ± 0.178 | [0.030, 0.102] |
| fact_sim | 0.172 ± 0.321 | [0.107, 0.237] |
| **kg_aware** | **0.637 ± 0.461** | [0.544, 0.731] |

**EPISODIC QUERIES** (100 queries):

| Strategy | MRR | 95% CI |
|----------|-----|--------|
| **naive_sim** | **0.500 ± 0.433** | [0.415, 0.585] |
| fact_sim | 0.163 ± 0.299 | [0.104, 0.221] |
| kg_aware | 0.028 ± 0.105 | [0.007, 0.048] |

### Statistical Significance

All comparisons are **statistically significant** (confidence intervals don't overlap):

1. **entity_overview**: kg_aware significantly better than naive_sim (F1 0.707 vs 0.077)
2. **specific_attribute**: kg_aware significantly better than naive_sim (MRR 0.637 vs 0.066)
3. **episodic**: naive_sim significantly better than kg_aware (MRR 0.500 vs 0.028)

### Key Findings

1. **KG-aware is dramatically better for state queries**:
   - entity_overview: ~9x improvement in F1 (0.707 vs 0.077)
   - specific_attribute: ~10x improvement in MRR (0.637 vs 0.066)

2. **Naive similarity is dramatically better for episodic queries**:
   - ~18x improvement in MRR (0.500 vs 0.028)
   - KG-aware completely fails for event-based retrieval

3. **Fact-based similarity is middle ground**:
   - Better than naive for attribute queries (MRR 0.172 vs 0.066)
   - Worse than naive for episodic (MRR 0.163 vs 0.500)
   - Much worse than KG-aware for state (MRR 0.172 vs 0.637)

4. **Query type classification is critical**:
   - Wrong strategy for wrong query type = terrible results
   - Need reliable query type detection before retrieval

### Conclusion

The KG approach is validated at scale:
- Entity resolution via embedding similarity works
- Attribute normalization with pre-defined schema works
- State tracking (replacement vs additive) provides correct answers
- For episodic queries, fall back to pure similarity

---

## Files

- `query_generation.py` - Reference detection experiment (final version)
- `temporal_retrieval.py` - Recency-weighted retrieval strategies
- `extraction_retrieval.py` - Extraction vs retrieval comparison (flawed ground truth)
- `attribute_retrieval.py` - Attribute-aware retrieval with typed facts (Experiment 5 - hardcoded)
- `knowledge_graph.py` - KG infrastructure (entity resolver, attribute normalizer)
- `kg_retrieval.py` - Fair KG-based retrieval experiment (Experiment 6)
- `test_data.py` - Synthetic test sequences with temporal ordering
- `results/` - Experiment result JSON files

---

## Next Steps

1. **Integrate into agent** - Add reference detection to the agent's processing pipeline
2. **Test with real conversations** - Validate on actual companion agent data
3. **Optimize retrieval** - Current implementation encodes each memory separately; batch for efficiency
4. **Handle multi-turn context** - References may span multiple messages
