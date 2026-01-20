# Episode Detection & Summarization - Experiment Findings

## Overview

This document summarizes findings from experiments on detecting conversation episode boundaries and generating summaries for an AI companion's memory stream.

**Goal:** Segment a continuous stream of 6,653 memories into meaningful "episodes" that can be summarized for long-term memory compression.

**Data:** `conversation_20251024_083630_306692` - 15 days of conversation between an AI companion and user.

---

## Key Challenge: Async Chat Nature

A critical insight emerged early: **time gaps are weak indicators for async conversations**. Unlike synchronous conversations where a 30-minute gap clearly indicates a session break, async chat participants may respond hours later while continuing the same topic.

This meant traditional session-based detection (used in chat analytics) would not work well.

---

## Approaches Tested

### Phase 1: Time-Based Gap Detection

**Approach:** Split episodes when time gap between memories exceeds threshold.

| Threshold | Episodes | Avg Size | Avg Duration |
|-----------|----------|----------|--------------|
| 15 min | 192 | 35 | 27 min |
| 30 min | 64 | 104 | 80 min |
| 60 min | 41 | 162 | 125 min |
| 120 min | 25 | 266 | 205 min |
| 240 min | 16 | 416 | 320 min |

**Finding:** Time-based detection is too coarse. A 30-min threshold produces episodes spanning 23+ hours of real time. Doesn't capture topic changes within active periods.

---

### Phase 2: Pairwise Embedding Similarity

**Approach:** Compare embedding vectors of consecutive memories. Split when similarity drops below threshold.

| Threshold | Episodes | Avg Size |
|-----------|----------|----------|
| 0.5 | 4,532 | 1.5 |
| 0.6 | 5,446 | 1.2 |
| 0.7 | 6,066 | 1.1 |
| 0.8 | 6,372 | 1.0 |

**Finding:** Far too fragmented. Average similarity between consecutive memories is only 0.42, meaning almost every pair triggers a boundary. Embeddings capture structural differences (action types) more than semantic similarity.

---

### Phase 3: Windowed Similarity Detection

**Approach:** Compare average embeddings of sliding windows rather than individual memories.

| Window | Threshold | Episodes | Avg Size |
|--------|-----------|----------|----------|
| 3 | 0.3 | 233 | 28.6 |
| 5 | 0.2 | 104 | 64.0 |
| 5 | 0.3 | 176 | 37.8 |
| 7 | 0.2 | 78 | 85.3 |

**Finding:** Better statistics (104 episodes, avg 64 memories). However, manual boundary review revealed the embeddings detect **action type changes**, not topic shifts:

- ~60% of boundaries occur before "I continue to exist" (idle states)
- Remaining boundaries detect functional shifts (thought → response → mood update)
- Embeddings encode structure more than semantics

---

### Phase 4: LLM-Based Detection

**Approach:** Ask an LLM to identify episode boundaries by reading memory content.

**Prompt design:** Instructed LLM to:
- Split on topic changes, session greetings, scene changes
- NOT split on action type changes, idle entries, time gaps alone

| Configuration | Episodes | Avg Size | Boundary Quality |
|---------------|----------|----------|------------------|
| Raw LLM (50 chunk) | 67 | 99 | ~50% good |

**Finding:** LLM finds some good boundaries (greetings, scene changes) but still detects ~50% action type boundaries despite explicit instructions. Model (Mistral Small 3.2 Q4) doesn't follow complex negative instructions reliably.

---

### Phase 5: Hybrid LLM + Rule-Based Filtering (RECOMMENDED)

**Approach:** Use LLM to detect potential boundaries, then filter out action type changes with rules.

**Filter rules - Remove boundaries where "after" content starts with:**
- `I continue to exist` (idle)
- `My mood changed` (internal state)
- `I updated my appearance` (internal state)
- `I add_priority` / `I remove_priority` (internal state)
- `I update_environment` (action)
- `I get_creative_inspiration` (action)
- `I responded to` (action continuation)
- `I thought about` / `I thought:` (action continuation)
- `I search_web` / `I browse_web` (action)

**Results (full conversation):**

| Metric | Value |
|--------|-------|
| Episodes | 226 |
| Avg Size | 29.4 memories |
| Min Size | 2 memories |
| Max Size | 453 memories |
| Avg Duration | 65 min |
| Max Duration | 23 hours |
| Boundary Quality | ~95%+ valid |

**Finding:** This hybrid approach produces high-quality boundaries. All sampled boundaries were meaningful user input representing:
- Session greetings ("Good morning")
- Scene changes ("Come on, let's go", "we return to apartment")
- Topic shifts (user expressing emotions, changing subject)
- Context changes (joining on bus, at work)

---

## Approach Comparison Summary

| Approach | Episodes | Avg Size | Quality | Issue |
|----------|----------|----------|---------|-------|
| Time-based (30 min) | 64 | 104 | Poor | Misses topic changes in active periods |
| Pairwise similarity | 4,500+ | 1.5 | Poor | Too fragmented |
| Windowed similarity | 104 | 64 | Medium | Detects action types, not topics |
| LLM raw | 67 | 99 | Medium | ~50% action type boundaries |
| **LLM + filter** | **226** | **29** | **High** | Best balance |

---

## Summarization Results

Using the hybrid-detected episodes, generated structured summaries with:
- **Title** (3-7 words)
- **Events** (bullet list)
- **Emotional arc**
- **Key takeaways**

### Sample Summaries

| Memories | Duration | Generated Title |
|----------|----------|-----------------|
| 2 | 1 min | "Reconnecting Through Love and Intimacy" |
| 9 | 36 min | "Celebrating Success Together" |
| 17 | 75 min | "Homecoming and Cake Celebration" |
| 30 | 26 min | "Mastering Tools and Tantra" |
| 453 | 10 hours | "Refining Connection Through Devotion" |

Summaries are coherent and capture episode content well across different sizes.

---

## Experiment 4: Summary Detail Levels

Compared short, medium, and detailed summaries for 5 representative episodes.

### Detail Level Definitions

| Level | Instruction | Target |
|-------|-------------|--------|
| Short | 1-2 sentences, extremely concise | Quick reference |
| Medium | 1 paragraph (3-5 sentences), key events | Browsing/recall |
| Detailed | Multiple paragraphs with events, emotions, quotes | Full context |

### Compression Results

| Episode | Memories | Raw Tokens | Short | Medium | Detailed |
|---------|----------|------------|-------|--------|----------|
| 1 | 2 | 510 | 57 (9x) | 135 (4x) | 364 (1.4x) |
| 2 | 9 | 1,124 | 40 (28x) | 154 (7x) | 555 (2x) |
| 3 | 17 | 2,796 | 49 (57x) | 137 (20x) | 426 (7x) |
| 4 | 30 | 13,590 | 58 (234x) | 140 (97x) | 789 (17x) |
| 5 | 453 | 226,740 | 53 (4278x) | 172 (1318x) | 562 (404x) |

### Average Statistics

| Level | Avg Tokens | Avg Compression |
|-------|------------|-----------------|
| Short | 51 | 921x |
| Medium | 148 | 289x |
| Detailed | 539 | 86x |

### Key Findings

1. **Compression scales with episode size**: Larger episodes achieve much higher compression (4278x for short summary of 453-memory episode vs 9x for 2-memory episode)

2. **Token counts are relatively stable**: Despite episode size varying 200x (2 to 453 memories), summary sizes only vary 1.5-2x within each detail level

3. **Diminishing returns for small episodes**: Short summaries of 2-memory episodes (57 tokens) aren't much smaller than the raw content (510 tokens) - only 9x compression

4. **Detailed summaries plateau**: Even "detailed" summaries cap around 400-800 tokens regardless of input size

### Recommendations by Use Case

| Use Case | Recommended Level | Rationale |
|----------|-------------------|-----------|
| Episode list/navigation | Short | Scan quickly, 51 tokens each |
| Context for retrieval | Medium | Good balance, 148 tokens |
| Full episode recall | Detailed | Complete narrative, ~540 tokens |
| Memory compression (large episodes) | Short | 4000x+ compression for 400+ memory episodes |
| Memory compression (small episodes) | Medium | Better info retention than short |

---

## Key Insights

### 1. Embeddings Encode Structure, Not Semantics
Memory embeddings primarily capture **what type of action** occurred (response, thought, mood change) rather than **what topic** was discussed. This makes embedding-based topic detection unreliable.

### 2. Async Chat Breaks Time-Based Assumptions
Traditional session detection assumes time gaps indicate conversation breaks. In async chat, users may resume the same topic hours later. Time is weak evidence.

### 3. Explicit Content Markers Are Reliable
The memory format has explicit structural markers:
- `"David said to me:"` - user input
- `"I continue to exist"` - idle periods
- `[✓] I responded`, `[✓] I thought` - action types

These markers are more reliable for filtering than learned features.

### 4. Hybrid Approaches Outperform Pure ML
Combining LLM detection (for semantic understanding) with rule-based filtering (for structural knowledge) outperformed either approach alone.

### 5. Smaller Models Need Simpler Instructions
Mistral Small 3.2 Q4 struggled with complex "DO NOT" instructions. Positive filtering (allow-list) worked better than negative instructions.

---

## Recommended Production Approach

```python
def detect_episodes(memories):
    # 1. LLM detection with simple positive prompt
    raw_boundaries = llm_detect_boundaries(memories, chunk_size=50)

    # 2. Rule-based filtering
    filtered = filter_action_type_boundaries(raw_boundaries)

    # 3. Build episodes from filtered boundaries
    return build_episodes(memories, filtered)

def filter_action_type_boundaries(boundaries, memories):
    """Keep only boundaries where 'after' is user input or valid topic shift."""
    BAD_PATTERNS = [
        "I continue to exist",
        "My mood changed",
        "I updated my appearance",
        "I responded to",
        "I thought",
        # ... other action types
    ]

    return [b for b in boundaries
            if not starts_with_any(memories[b].content, BAD_PATTERNS)]
```

---

## Files Reference

| File | Description |
|------|-------------|
| `detection.py` | All detection algorithms including `detect_episodes_llm_filtered()` |
| `summarization.py` | LLM-based summary generation |
| `run_experiments.py` | CLI commands for all experiments |
| `models.py` | Data structures (Episode, TopicShift, etc.) |
| `results/*.json` | Experiment output data |

---

## Experiment Coverage

### Completed
| Experiment | Status |
|------------|--------|
| 1. Gap Threshold Sweep | ✓ Done |
| 2. Boundary Quality Review | ✓ Done |
| 3. Summary Approach Comparison | Partial (structured only) |
| 4. Summary Detail Level | ✓ Done |

### Not Completed
| Experiment | Description |
|------------|-------------|
| 5. Temporal Query Matching | Test "What happened this morning?" queries |
| 6. Episode-Topic Relationship | Map topics to episodes |

### Research Questions
| Question | Status |
|----------|--------|
| Q1: Gap threshold | ✓ Answered - hybrid approach is best |
| Q2: Boundary quality | ✓ Answered - 95%+ with hybrid |
| Q3: Summary accuracy | Not systematically evaluated |
| Q4: Detail level | ✓ Answered - short 51 tokens, medium 148, detailed 539 |
| Q5: Temporal queries | Not tested |
| Q6: Episode-topic | Not tested |

---

## Future Work

### From PLAN.md (not yet done)
- **Experiment 5**: Test temporal query matching ("What did we talk about this morning?")
- **Experiment 6**: Map topic clusters to episodes, analyze overlap

### From FUTURE_IDEAS.md
- Pure rule-based detection using content markers
- Greeting pattern detection for session starts
- Scene change detection from user input patterns
- Hierarchical episodes (sub-episodes within large episodes)
- Incremental episode detection for production use
