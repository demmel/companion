# Episode Summaries Prototype

## Concept

### What is this?

An episode is a contiguous conversation session, detected by finding gaps in timestamps. When there's a long pause between memories (e.g., 30+ minutes), that indicates a session boundary.

**Example**: A day's memories might form 3 episodes:
- Episode 1 (9:00-9:45): Morning greeting, coffee discussion, daily plans
- Episode 2 (14:00-15:30): Afternoon check-in, work discussion, mood changes
- Episode 3 (21:00-22:15): Evening conversation, reflections, goodnight

An **episode summary** captures what happened during that session - the narrative arc, key events, emotional beats.

### Why does this matter?

1. **Temporal queries**: "What did we talk about this morning?" requires understanding session boundaries
2. **Narrative structure**: Conversations have beginnings, middles, ends - episodes capture this arc
3. **Context relevance**: Recent episode context is more relevant than distant memories
4. **Compression**: Summarize 50 memories into one episode description
5. **Memory navigation**: Browse by session rather than scrolling through everything

### Core tension

Time gaps don't always indicate meaningful boundaries. A 30-minute gap could be a real session break or just the user being busy. The question is: how to detect meaningful episode boundaries?

---

## Design

### Data Structures

```python
@dataclass
class Episode:
    """A contiguous conversation session."""
    id: str
    start_time: datetime
    end_time: datetime
    duration_minutes: float
    memory_ids: list[str]           # Memories in this episode
    memory_count: int

    # Generated content
    title: str | None               # Short title (3-7 words)
    summary: str | None             # Narrative summary
    key_events: list[str] | None    # Main things that happened
    participants: list[str] | None  # Who was involved
    emotional_arc: str | None       # How emotions changed
    topics_discussed: list[str] | None

@dataclass
class EpisodeDetectionResult:
    """Result of episode detection."""
    episodes: list[Episode]
    gap_threshold_used: int         # Minutes
    orphan_memories: list[str]      # Memories that didn't fit (if any)
```

### Episode Detection Approaches

**Approach A: Fixed time gap**
- Gap > N minutes = new episode
- Simple, deterministic
- May miss meaningful breaks within conversations
- May split single conversations that had a pause

**Approach B: Adaptive time gap**
- Use statistics of gap distribution
- Gaps > 2 standard deviations = new episode
- Adapts to different conversation patterns
- More complex, less predictable

**Approach C: Time gap + topic shift**
- Detect large time gaps
- Also detect when topic shifts significantly (even without gap)
- Requires embedding comparison or LLM judgment
- More expensive but potentially more accurate

**Approach D: LLM-based boundary detection**
- Show LLM a sequence of memories
- Ask: "Where would you split this into conversations?"
- Most expensive but potentially most accurate
- Hard to scale

### Summary Approaches

**Basic summary**:
```
Summarize this conversation in one paragraph.
What happened? What was discussed? How did it end?
```

**Structured summary**:
```
For this conversation:
1. Title (3-7 words)
2. Main events or topics
3. Emotional arc (how did mood change?)
4. Key takeaways
```

**Narrative summary**:
```
Tell the story of this conversation as a narrative.
What happened from beginning to end?
```

**Question-oriented summary**:
```
What questions could this conversation answer?
For each, provide a brief answer.
```

---

## Research Questions

### Q1: What gap threshold produces sensible episodes?

Test thresholds: 15, 30, 60, 120, 240 minutes
For each:
- How many episodes result?
- What's the size distribution?
- Manual review: do boundaries make sense?

### Q2: Do detected episodes correspond to real conversation sessions?

Manual review:
- Do episode boundaries feel natural?
- Are there false positives (split mid-conversation)?
- Are there false negatives (merged separate conversations)?

### Q3: How accurate are episode summaries?

For each summary, evaluate:
- **Completeness**: What important things were omitted?
- **Accuracy**: Any hallucinations or errors?
- **Narrative quality**: Does it tell a coherent story?

### Q4: What level of detail is right for summaries?

Compare:
- Short (1-2 sentences)
- Medium (1 paragraph)
- Detailed (multiple paragraphs with events)

Which is most useful? For what purpose?

### Q5: Can episode summaries answer temporal queries?

Test queries like:
- "What happened this morning?"
- "What did we discuss last time?"
- "When did we talk about X?"

Can matching episode summaries find the right episode?

### Q6: How do episodes relate to topics?

An episode might contain multiple topics.
A topic might span multiple episodes.
How to represent this relationship?

---

## Experiments

### Experiment 1: Gap Threshold Sweep

**Setup**:
- Load all memories from test data
- Run episode detection with thresholds: 15, 30, 60, 120, 240 minutes
- Analyze results

**Measure**:
- Number of episodes at each threshold
- Episode size distribution (min, max, avg memories per episode)
- Episode duration distribution

**Output**:
```
Threshold: 15 minutes
  Episodes: 12
  Sizes: min=3, max=45, avg=15
  Durations: min=5min, max=120min, avg=35min

Threshold: 30 minutes
  Episodes: 8
  Sizes: min=5, max=60, avg=22
  Durations: min=10min, max=150min, avg=50min

...

Recommendation: 30-minute threshold produces X episodes with reasonable sizes
```

### Experiment 2: Boundary Quality Review

**Setup**:
- Use recommended threshold from Experiment 1
- Manually review each episode boundary
- Mark: GOOD (natural break), FALSE_POSITIVE (split mid-convo), FALSE_NEGATIVE (should split)

**Measure**:
- Boundary accuracy rate
- False positive rate
- False negative rate

**Output**:
```
Total boundaries: 7
  GOOD: 5 (71%)
  FALSE_POSITIVE: 1 (14%) - 45-minute break was just user away
  FALSE_NEGATIVE: 1 (14%) - Topic shifted within episode

Overall boundary accuracy: 71%
```

### Experiment 3: Summary Approach Comparison

**Setup**:
- Select 5 diverse episodes
- Generate summaries using each approach (basic, structured, narrative, question)
- Manually evaluate

**Measure**:
- Completeness (1-5 scale)
- Accuracy (1-5 scale)
- Usefulness (1-5 scale)

**Output**:
```
Episode 1 "Morning Conversation":

Basic summary:
  "David and Chloe had a morning conversation about coffee and daily plans.
   David mentioned having a meeting later."
  Completeness: 3/5 (missed mood discussion)
  Accuracy: 5/5
  Usefulness: 3/5

Structured summary:
  Title: "Morning Coffee and Plans"
  Events: [coffee discussion, daily planning, meeting mention]
  Emotional arc: "Started tired, became more engaged"
  Completeness: 4/5
  Accuracy: 5/5
  Usefulness: 4/5

...
```

### Experiment 4: Summary Detail Level

**Setup**:
- Generate short, medium, and detailed summaries for same episodes
- Compare token counts and usefulness

**Measure**:
- Token count at each level
- Information captured
- Best level for different use cases

**Output**:
```
Episode "Morning Conversation" (45 memories, 3200 tokens raw):

Short summary (50 tokens):
  "Brief morning conversation about coffee, daily plans, and an upcoming meeting."
  Captures: 40% of key info

Medium summary (150 tokens):
  "David and Chloe's morning conversation covered David's coffee routine,
   his plans for the day including a 2pm meeting with Henderson account,
   and Chloe's mood transitioning from sleepy to engaged..."
  Captures: 75% of key info

Detailed summary (400 tokens):
  [Full narrative with events, quotes, emotional beats]
  Captures: 95% of key info

Compression:
  Short: 64x compression
  Medium: 21x compression
  Detailed: 8x compression
```

### Experiment 5: Temporal Query Matching

**Setup**:
- 10 temporal queries with known answers
- Match queries to episode summaries
- Check if correct episode is found

**Measure**:
- Query-to-episode accuracy
- Which queries work well / poorly

**Output**:
```
Query: "What did we talk about this morning?"
  Correct episode: Episode 1
  Matched episode: Episode 1 (score 0.72)
  Result: CORRECT

Query: "When did David mention the Henderson account?"
  Correct episode: Episode 1
  Matched episode: Episode 2 (score 0.45)
  Result: WRONG - Henderson mentioned in both, chose wrong one

Overall accuracy: 7/10 (70%)
```

### Experiment 6: Episode-Topic Relationship

**Setup**:
- Run topic clustering on all memories
- Map topics to episodes
- Analyze overlap

**Measure**:
- Topics per episode
- Episodes per topic
- Cross-cutting patterns

**Output**:
```
Episode 1 "Morning Conversation":
  Topics: [Work (40%), Daily Life (35%), Mood (25%)]

Episode 2 "Afternoon Check-in":
  Topics: [Work (60%), Emotional (30%), Relationship (10%)]

Topic "Work" appears in:
  Episode 1: 12 memories
  Episode 2: 18 memories
  Episode 3: 3 memories
```

---

## Implementation Outline

### Files to Create

```
episode_summaries/
├── PLAN.md                 # This file
├── __init__.py
├── models.py               # Episode, EpisodeDetectionResult dataclasses
├── detection.py            # Episode boundary detection algorithms
├── summarization.py        # LLM-based episode summarization
├── evaluation.py           # Quality metrics and review helpers
├── temporal_search.py      # Searching episodes by time/query
├── run_experiments.py      # Main experiment runner
└── results/                # Output directory
```

### Key Functions

```python
# detection.py
def detect_episodes_by_gap(memories: list[Memory], gap_minutes: int) -> EpisodeDetectionResult:
    """Detect episodes using fixed time gap threshold."""

def detect_episodes_adaptive(memories: list[Memory]) -> EpisodeDetectionResult:
    """Detect episodes using adaptive gap threshold."""

def find_optimal_gap(memories: list[Memory], candidates: list[int]) -> int:
    """Find gap threshold that produces best episode structure."""

# summarization.py
def generate_episode_summary(episode: Episode, memories: list[Memory], style: str) -> str:
    """Generate summary for an episode using specified style."""

def generate_episode_title(episode: Episode, memories: list[Memory]) -> str:
    """Generate a short title for the episode."""

def extract_key_events(episode: Episode, memories: list[Memory]) -> list[str]:
    """Extract main events that happened in the episode."""

# temporal_search.py
def find_episode_by_time(episodes: list[Episode], time_reference: str) -> Episode | None:
    """Find episode matching a temporal reference like 'this morning'."""

def search_episodes_by_query(query: str, episodes: list[Episode]) -> list[tuple[Episode, float]]:
    """Search episodes by matching query to summaries."""
```

---

## Open Questions

### For experimentation:

1. **Topic shifts**: Should topic shifts within a conversation create episode boundaries? Or are those sub-episodes?

2. **Overlapping episodes**: Can episodes overlap (e.g., background conversation while doing something)?

3. **Episode continuation**: If a conversation pauses and resumes, is that one episode or two?

4. **Summary freshness**: Should episode summaries be regenerated as more memories are added (for ongoing conversations)?

### For user input:

1. **Conversation patterns**: How long are typical conversations? How often are there gaps?

2. **Temporal query patterns**: What temporal queries are common? "This morning"? "Last time"? "Yesterday"?

3. **Episode use cases**: How will episodes be used? Context? Navigation? Search? Compression?

---

## Success Criteria

This prototype is successful if:

1. **Detection works**: Episodes correspond to natural conversation boundaries (>70% accuracy)
2. **Summaries are accurate**: Manual review shows <10% hallucination, <20% omission
3. **Temporal queries work**: "What happened this morning?" type queries find correct episode >70%
4. **Clear recommendation**: We know what gap threshold and summary style to use

---

## Next Steps After This Plan

1. User reviews and approves this plan
2. Implement episode detection with multiple thresholds
3. Run Experiment 1-2 (threshold sweep, boundary review)
4. Based on results, select threshold
5. Run remaining experiments
6. Document findings and recommendations
