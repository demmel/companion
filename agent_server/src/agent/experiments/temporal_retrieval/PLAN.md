# Temporal Retrieval Experiment

## Goal

Build and evaluate retrieval for time-based queries. This includes relative time ("yesterday"), absolute time ("January 15"), and emotional/contextual time ("when I was stressed"). Builds on episode summaries work.

## How This Fits

**Read first:** `../memory_architecture/ARCHITECTURE.md` - describes the overall memory system architecture.

This experiment builds the **Temporal Retrieval** component that handles time-based queries. Other parallel experiments are building:
- Query Classification (`../query_classification/`) - classifier optimization
- Unified Retrieval (`../unified_retrieval/`) - the full pipeline

The temporal retrieval module will plug into the unified pipeline for `temporal` query types.

## Context

From `episode_summaries/FINDINGS.md`:
- Hybrid LLM + rule-based detection achieves 95%+ episode boundary quality
- 226 episodes detected from 6,653 memories
- Short summaries: 51 tokens avg, 921x compression
- **Not tested:** Temporal query matching ("what happened this morning?")

**Scale requirement:** Decades of conversation. Episodes provide natural compression - instead of searching millions of memories, search thousands of episode summaries.

## Query Types

### 1. Relative Time
| Pattern | Example | Resolution |
|---------|---------|------------|
| this morning/afternoon/evening | "What did we talk about this morning?" | Today's time ranges |
| yesterday, today | "What happened yesterday?" | Calendar day |
| last week/month | "Remember last week?" | Calendar range |
| recently | "What have we been discussing recently?" | Recency window |

### 2. Absolute Time
| Pattern | Example | Resolution |
|---------|---------|------------|
| Day of week | "What happened on Tuesday?" | Most recent Tuesday |
| Date | "January 15th" | Specific date |
| Month/Year | "Back in December" | Month range |
| Named events | "After Christmas" | Event-anchored |

### 3. Emotional/Contextual Time
| Pattern | Example | Resolution |
|---------|---------|------------|
| Mood-based | "When I was stressed about work" | Episodes with matching mood |
| Event-based | "During the job search" | Topic-filtered episodes |
| Relational | "Before we talked about X" | Relative to topic |
| Implicit | "That rough period" | Requires conversation context |

## Experiment Design

### Phase 1: Time Expression Parsing

Build parser for temporal expressions:

```python
@dataclass
class TimeReference:
    raw_text: str
    ref_type: Literal["relative", "absolute", "emotional"]

    # For relative/absolute
    start_time: datetime | None
    end_time: datetime | None

    # For emotional/contextual
    mood_filter: str | None
    topic_filter: str | None
    event_anchor: str | None


def parse_time_reference(text: str, now: datetime) -> TimeReference | None:
    """Extract and resolve time references from text."""
    ...
```

**Test cases:**
- "this morning" → today 6am-12pm
- "yesterday" → previous calendar day
- "last Tuesday" → most recent Tuesday
- "when I was stressed" → mood_filter="stressed"
- "during the job interview prep" → topic_filter="job interview"

### Phase 2: Episode Index

Build time-indexed episode lookup:

```python
class EpisodeIndex:
    def __init__(self, episodes: list[Episode]):
        self.by_time: SortedDict[datetime, Episode] = ...
        self.by_topic: dict[str, list[Episode]] = ...
        self.by_mood: dict[str, list[Episode]] = ...

    def query(self, time_ref: TimeReference) -> list[Episode]:
        """Find episodes matching time reference."""
        if time_ref.start_time and time_ref.end_time:
            # Time range query
            return self.by_time.irange(time_ref.start_time, time_ref.end_time)
        elif time_ref.mood_filter:
            # Mood-based query
            return self.by_mood.get(time_ref.mood_filter, [])
        elif time_ref.topic_filter:
            # Topic-based query
            return self._topic_search(time_ref.topic_filter)
```

### Phase 3: Evaluation Dataset

Create test queries with ground truth:

```json
{
  "query": "What did we talk about yesterday morning?",
  "time_ref": {
    "type": "relative",
    "start": "2024-01-18T06:00:00",
    "end": "2024-01-18T12:00:00"
  },
  "expected_episodes": [42, 43],
  "expected_content_keywords": ["breakfast", "morning routine"]
}
```

**Target:** 100+ queries across all three types:
- Relative: 40 queries
- Absolute: 30 queries
- Emotional/contextual: 30 queries

### Phase 4: Retrieval Strategies

#### Strategy A: Episode Summary Only
1. Parse time reference
2. Find matching episodes
3. Return episode summaries

**Pro:** Fast, compressed
**Con:** May lose detail

#### Strategy B: Episode → Memories
1. Parse time reference
2. Find matching episodes
3. Retrieve raw memories from those episodes

**Pro:** Full detail
**Con:** More tokens, slower

#### Strategy C: Hybrid
1. Parse time reference
2. Find matching episodes
3. Return summaries + top-K relevant raw memories (similarity to query within episode)

**Pro:** Balance of context and detail
**Con:** More complex

#### Strategy D: Direct Memory Search (Baseline)
1. Filter memories by time range
2. Similarity search within filtered set

**Pro:** Simple
**Con:** Doesn't use episode structure

### Phase 5: Emotional Time Handling

Emotional/contextual time is harder - requires understanding episode content.

**Approach A: Episode Metadata**
Add mood and topic tags to episodes during summarization:
```python
@dataclass
class Episode:
    memories: list[Memory]
    summary: str
    start_time: datetime
    end_time: datetime
    primary_mood: str          # "stressed", "happy", "anxious"
    topics: list[str]          # ["work", "job interview", "coding"]
    emotional_intensity: float  # 0-1
```

**Approach B: Semantic Search on Summaries**
Embed episode summaries, search for "when I was stressed" against summary embeddings.

**Approach C: LLM Filtering**
Ask LLM: "Which of these episodes match 'when I was stressed'?"
```
Episode 42: "Discussion about upcoming deadline, user expressed anxiety..."
Episode 43: "Relaxed evening conversation about hobbies..."

Which episodes match "when I was stressed"? → [42]
```

### Phase 6: Metrics

| Metric | Description |
|--------|-------------|
| Time parse accuracy | % of time expressions correctly parsed |
| Episode recall | % of relevant episodes retrieved |
| Episode precision | % of retrieved episodes that are relevant |
| Content relevance | LLM-judged: does retrieved content answer the query? |
| Latency | Query time |

**Targets:**
- Time parse: 95%+ for relative/absolute, 80%+ for emotional
- Episode F1: 0.8+
- Content relevance: 85%+ queries have useful response

## Deliverables

1. `time_parser.py` - Temporal expression parsing
2. `episode_index.py` - Time-indexed episode lookup
3. `emotional_time.py` - Mood/topic-based time resolution
4. `test_queries.json` - Evaluation dataset
5. `evaluate.py` - Evaluation pipeline
6. `FINDINGS.md` - Results and recommendations

## Files to Reference

- `../episode_summaries/detection.py` - Episode detection
- `../episode_summaries/summarization.py` - Summary generation
- `../episode_summaries/FINDINGS.md` - Episode experiment results
- `../topic_clustering/FINDINGS.md` - Topic cluster results (may help with topic-based time)

## Success Criteria

- [ ] Time parser handles relative, absolute, emotional references
- [ ] Episode index built with time, topic, mood access patterns
- [ ] 100+ test queries with ground truth
- [ ] Retrieval strategies compared
- [ ] Emotional time approach validated
- [ ] Clear recommendation for production

## Running

```bash
# Build episode index from conversation
uv run python -m agent.experiments.temporal_retrieval.build_index --conversation <id>

# Test time parser
uv run python -m agent.experiments.temporal_retrieval.test_parser

# Evaluate retrieval strategies
uv run python -m agent.experiments.temporal_retrieval.evaluate

# Test emotional time handling
uv run python -m agent.experiments.temporal_retrieval.emotional_time
```

## Notes

- Use pydantic models for all data structures
- Annotate all function signatures
- Time parsing is tricky - handle timezone carefully (user's local time vs UTC)
- Emotional time may need conversation context to resolve ("that rough period" - which one?)
- This can run in parallel with query_classification
