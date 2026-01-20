# Memory Extraction Prototype

## Concept

### What is this?

Memory extraction takes verbose, narrative memory content and distills it into structured, searchable facts.

**Example input** (raw memory):
```
David said to me: "Well I'm still having my coffee, black as usual. You know how I am in
the mornings - can't function without it. Anyway, I've got that meeting at 2pm with the
Henderson account, should be done by 4. Want to grab dinner after?"
```

**Example output** (extracted):
```
- David drinks black coffee
- David has a meeting at 2pm (Henderson account)
- David expects to finish by 4pm
- David proposed dinner together
```

### Why does this matter?

1. **Retrieval**: Query "What does David drink?" matches "black coffee" better than the verbose narrative
2. **Compression**: Store dense facts instead of rambling content
3. **Structure**: Facts can be typed (person, preference, event, relationship)
4. **Clarity**: Remove filler words, hedging, conversational artifacts

### Core tension

Extraction loses context. "David drinks black coffee" loses that this was said casually in a morning conversation. The question is: what's worth keeping?

---

## Design

### Data Structures

```python
@dataclass
class ExtractedFact:
    """A single extracted fact."""
    content: str                    # The fact itself
    fact_type: str                  # "preference", "event", "relationship", "state", etc.
    confidence: float               # How certain the extraction is
    entities: list[str]             # People/things mentioned
    source_memory_id: str           # Where this came from

@dataclass
class ExtractionResult:
    """All extractions from a single memory."""
    memory_id: str
    original_content: str
    facts: list[ExtractedFact]
    summary: str                    # One-sentence summary
    compression_ratio: float        # len(extracted) / len(original)
```

### Extraction Approaches to Try

**Approach A: Fact list extraction**
```
Extract all facts from this memory as a bullet list.
Each fact should be a standalone, searchable statement.
```

**Approach B: Structured extraction**
```
Extract information in these categories:
- People mentioned and what we learn about them
- Events or plans discussed
- Preferences or opinions expressed
- Emotional states or moods
- Questions asked or answered
```

**Approach C: Query-focused extraction**
```
What questions could this memory answer?
For each question, provide a concise answer.
```

**Approach D: Entity-centric extraction**
```
For each person/entity mentioned:
- What do we learn about them?
- What did they say or do?
- How do they relate to others?
```

**Approach E: Minimal extraction**
```
What is the single most important fact in this memory?
Answer in 10 words or less.
```

### What to Extract

Categories of information:
- **Facts**: Concrete, verifiable information (David drinks black coffee)
- **Events**: Things that happened or will happen (meeting at 2pm)
- **Preferences**: Likes, dislikes, habits (can't function without coffee)
- **Relationships**: How entities relate (David proposed dinner with Chloe)
- **States**: Emotional or situational states (morning mood, tired)
- **Questions**: Things asked but not answered

### What NOT to Extract

- Filler words and conversational padding
- Hedging language ("kind of", "sort of", "maybe")
- Redundant restatements
- Context that can be inferred from metadata (timestamps)

---

## Research Questions

### Q1: What extraction approach produces the most useful facts?

Compare approaches A-E on:
- Fact completeness (what important things were captured?)
- Fact accuracy (any hallucinations?)
- Fact searchability (do queries match well?)
- Compression ratio achieved

### Q2: How accurate is LLM extraction?

For each extracted fact, manually label:
- **CORRECT**: Fact is accurate and present in original
- **HALLUCINATED**: Fact is not supported by original
- **INFERRED**: Fact is reasonable inference but not explicit
- **OMITTED**: Important fact in original was not extracted

Target: <5% hallucination rate, <20% omission rate

### Q3: What compression ratio is achievable?

Measure: `len(all_extracted_facts) / len(original_content)`

Explore tradeoffs:
- High compression (0.2x) = more loss, simpler facts
- Low compression (0.8x) = less loss, more verbose
- What's the sweet spot?

### Q4: Does extraction improve retrieval?

Test queries against:
- Raw memories only
- Extracted facts only
- Both (hybrid)

Measure: Does the correct answer rank higher with extraction?

### Q5: Should extraction replace or augment raw content?

Options:
- **Replace**: Only store extracted facts (maximum compression)
- **Augment**: Store both, search both (maximum recall)
- **Tiered**: Store raw, generate extractions on-demand

### Q6: How does extraction quality vary by memory type?

Test on different memory types:
- Dialogue (David said...)
- Actions (I changed my appearance...)
- Reflections (I thought about...)
- Events (Something happened...)

Does extraction work better for some types?

---

## Experiments

### Experiment 1: Extraction Approach Comparison

**Setup**:
- Select 30 diverse memories from test data
- Run each extraction approach (A-E) on all 30
- Generate extractions using LLM

**Measure**:
- Number of facts extracted per memory
- Character count of extraction vs original
- Manual quality review of 10 samples per approach

**Output**:
```
Approach A (fact list):
  - Avg facts per memory: 4.2
  - Avg compression: 0.45x
  - Sample: [show 3 examples]

Approach B (structured):
  ...
```

### Experiment 2: Accuracy Annotation

**Setup**:
- Take best approach from Experiment 1
- Extract facts from 20 memories
- Manually annotate every fact: CORRECT / HALLUCINATED / INFERRED / OMITTED

**Measure**:
- Hallucination rate
- Omission rate (review original to find missed facts)
- Inference rate

**Output**:
```
Total facts extracted: 84
- CORRECT: 72 (85.7%)
- HALLUCINATED: 3 (3.6%)
- INFERRED: 9 (10.7%)

Omissions found: 12 important facts not extracted
Omission rate: ~14%
```

### Experiment 3: Retrieval Impact

**Setup**:
- 20 test queries with known answers
- Search raw memories, rank by similarity
- Search extracted facts, rank by similarity
- Search both (hybrid), rank by similarity

**Measure**:
- Mean Reciprocal Rank (MRR) for each approach
- Queries where extraction helped vs hurt

**Output**:
```
Query: "What does David drink?"
  Raw: rank 5 (score 0.42)
  Extracted: rank 1 (score 0.78)
  Hybrid: rank 1 (score 0.78)
  Winner: Extracted

Overall:
  Raw MRR: 0.45
  Extracted MRR: 0.62
  Hybrid MRR: 0.68
```

### Experiment 4: Compression vs Quality Tradeoff

**Setup**:
- Vary extraction prompt to get different compression levels
- "Extract only the 3 most important facts" (high compression)
- "Extract all facts comprehensively" (low compression)
- Measure quality at each level

**Measure**:
- Compression ratio
- Accuracy (via annotation)
- Retrieval performance

**Output**:
```
High compression (0.2x):
  - Accuracy: 95%
  - Omission rate: 40%
  - Retrieval MRR: 0.55

Medium compression (0.4x):
  - Accuracy: 90%
  - Omission rate: 20%
  - Retrieval MRR: 0.65

Low compression (0.7x):
  - Accuracy: 85%
  - Omission rate: 10%
  - Retrieval MRR: 0.60
```

### Experiment 5: Memory Type Analysis

**Setup**:
- Categorize memories by type (dialogue, action, reflection, event)
- Run extraction on each category
- Compare quality metrics by category

**Measure**:
- Accuracy by memory type
- Compression by memory type
- Identify which types benefit most from extraction

---

## Implementation Outline

### Files to Create

```
memory_extraction/
├── PLAN.md                 # This file
├── __init__.py
├── models.py               # ExtractedFact, ExtractionResult dataclasses
├── extraction.py           # Core extraction logic, multiple approaches
├── prompts.py              # LLM prompts for each extraction approach
├── evaluation.py           # Accuracy annotation helpers
├── retrieval.py            # Search raw vs extracted vs hybrid
├── run_experiments.py      # Main experiment runner
└── results/                # Output directory for experiment results
```

### Key Functions

```python
# extraction.py
def extract_facts(content: str, approach: str, llm, model) -> ExtractionResult:
    """Extract facts from a memory using specified approach."""

def extract_batch(memories: list[Memory], approach: str) -> list[ExtractionResult]:
    """Extract facts from multiple memories."""

# evaluation.py
def annotate_extraction(original: str, extraction: ExtractionResult) -> AnnotationResult:
    """Interactive annotation of extraction accuracy."""

def calculate_omissions(original: str, facts: list[ExtractedFact]) -> list[str]:
    """Find important facts in original that weren't extracted."""

# retrieval.py
def search_raw(query: str, memories: list[Memory]) -> list[SearchResult]:
    """Search raw memory content."""

def search_extracted(query: str, extractions: list[ExtractionResult]) -> list[SearchResult]:
    """Search extracted facts."""

def search_hybrid(query: str, memories: list[Memory], extractions: list[ExtractionResult]) -> list[SearchResult]:
    """Search both raw and extracted."""
```

---

## Open Questions

### For experimentation:

1. **Fact typing**: Should facts be typed (preference, event, relationship)? Does it help retrieval?

2. **Entity linking**: Should extracted facts link back to specific entities? How to handle coreference ("he", "she", "it")?

3. **Temporal context**: How to preserve when something happened without redundant timestamps?

4. **Confidence scoring**: Can the LLM reliably indicate confidence in extractions?

### For user input:

1. **Memory types**: What types of memories exist in the current system? What's their distribution?

2. **Query patterns**: What kinds of queries are common? What should extraction optimize for?

3. **Storage constraints**: Is storage a concern? Does compression matter beyond retrieval quality?

4. **Real-time vs batch**: Should extraction happen at memory creation time or on-demand?

---

## Success Criteria

This prototype is successful if:

1. **Extraction works**: LLM can extract facts with <10% hallucination rate
2. **Retrieval improves**: Hybrid search (raw + extracted) beats raw-only on test queries
3. **Compression achieved**: Can compress to <50% of original while maintaining quality
4. **Clear recommendation**: We know when to use extraction and how to configure it

---

## Next Steps After This Plan

1. User reviews and approves this plan
2. Implement core extraction with multiple approaches
3. Run Experiment 1 (approach comparison)
4. Based on results, focus on best approach(es)
5. Run remaining experiments
6. Document findings and recommendations
