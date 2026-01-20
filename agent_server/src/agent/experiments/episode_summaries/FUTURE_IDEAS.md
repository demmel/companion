# Future Ideas - Episode Detection

Ideas discovered during experimentation that go beyond the original PLAN.md.

## Idea: Rule-Based Content Markers

**Origin:** Observed during Phase 3 boundary review that embeddings detect action TYPE changes rather than topic shifts.

**Observation:** The memory data has explicit structural markers:
- `"David said to me:"` - user input
- `"Chloe said to me:"` - external input
- `"I continue to exist"` - idle/waiting periods
- `[✓] I responded`, `[✓] I thought`, etc. - action types

**Proposed approach:**
```
New episode starts when:
  - Time gap > 4 hours (session break), OR
  - User input ("David said") after idle period, OR
  - User input after time gap > 30 min
```

**Why it might work:**
- Leverages structure that embeddings miss
- Simple, fast, interpretable
- "David said to me" after idle = clear conversation resumption

**Status:** Partially validated. The hybrid LLM + filter approach (Phase 5) uses these markers for filtering and achieves 95%+ boundary quality.

---

## Idea: Greeting Pattern Detection

**Origin:** Observed that session greetings ("Good morning", "Hey I'm back") are reliable episode start markers.

**Observation:** User messages containing greetings almost always indicate a new conversation session, regardless of time gap.

**Proposed approach:**
```python
GREETING_PATTERNS = [
    "good morning", "good night", "goodnight",
    "hello", "hey", "hi ",
    "i'm back", "im back",
]

def is_session_greeting(user_message):
    return any(p in user_message.lower() for p in GREETING_PATTERNS)
```

**Why it might work:**
- Greetings are explicit social markers of conversation (re)start
- Works regardless of time gap
- High precision (few false positives)

**Status:** Implemented in `is_good_boundary()` but could be primary detector.

---

## Idea: Scene Change Detection

**Origin:** Observed that user messages describing location/activity changes are natural episode boundaries.

**Examples from data:**
- `"Come on, let's go. *We walk out to Elliot Bay*"`
- `"*after our encounter at the bay, we return to the apartment*"`
- `"Hey, I just hopped on the bus. Update your environment to join me"`

**Proposed approach:**
- Detect asterisk-wrapped actions (`*we go to...*`)
- Detect location change verbs (walk, go, return, arrive)
- Detect environment update requests

**Status:** Not implemented. These are currently detected by LLM but could be rule-based.

---

## Idea: Hierarchical Episodes

**Origin:** The largest detected episode has 453 memories spanning 10+ hours. These could benefit from sub-episodes.

**Proposed approach:**
- Level 1: Major episodes (current hybrid approach)
- Level 2: Sub-episodes within large episodes (time-based or topic-based)
- Generate summaries at both levels

**Benefits:**
- More granular recall for large episodes
- Progressive summarization (sub-summary → episode summary)

**Status:** Not implemented.

---

## Idea: Incremental Episode Detection

**Origin:** Current approach processes all memories at once. Production needs real-time detection.

**Proposed approach:**
- Maintain running episode state
- On new memory, check if it should start new episode:
  - Is it user input after idle?
  - Does it contain greeting?
  - Does LLM think context shifted?
- Close previous episode and start new one if yes

**Status:** Not implemented. Current code is batch-oriented.
