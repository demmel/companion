# Memory Extraction Experiment - Findings

## Overview

This experiment tested whether extracting structured facts from verbose narrative memories could improve retrieval. The experiment had significant design flaws that limit conclusions, but produced useful learnings.

## What Worked

### Extraction Quality
- All 5 approaches achieved <1% hallucination rate (LLM-judged)
- Extraction successfully distills narrative into structured facts
- Approach A (fact list) had best balance: 87.8% accuracy, 4.6 omissions/memory

### Approach Comparison Results

| Approach | Description | Facts/Memory | Accuracy | Hallucination | Omissions |
|----------|-------------|--------------|----------|---------------|-----------|
| A | Fact list | 16.4 | 87.8% | 0.0% | 4.6 |
| B | Structured by category | 12.4 | 83.9% | 0.0% | 6.2 |
| C | Query-focused (Q&A) | 5.4 | 88.9% | 0.0% | 5.6 |
| D | Entity-centric | 23.6 | 75.4% | 0.8% | 7.0 |
| E | Minimal (1 fact) | 1.0 | 100.0% | 0.0% | 10.6 |

## What Was Flawed

### Accuracy Measurement
- Used LLM-as-judge (same model evaluating its own output)
- No human verification baseline
- 0% hallucination rate is likely unreliable

### Best Approach Selection
- Algorithm selected E (100% accuracy) ignoring that it has 10.6 omissions
- Selection criteria should weight omissions, not just accuracy

### Retrieval Experiment (Experiment 3)
The retrieval test was fundamentally broken:

1. **Initial design**: All queries pointed to `memories[0]` - invalid test
2. **Redesigned version**: Auto-generated queries from each memory, assumed source memory = correct answer
3. **Core flaw**: For state queries like "What is she wearing?", the correct answer is the *most recent* memory, not the source memory
4. **Missing foundation**: Compared two naive retrieval approaches (raw vs extracted embedding similarity) without first designing what good retrieval looks like

## Key Learnings

### Extraction is not the bottleneck
The extraction itself works reasonably well. The problem is not "can we extract facts?" but "what do we do with them?"

### Retrieval needs design first
Before testing "does extraction help retrieval?", we need:
1. Clear retrieval requirements for companion agents
2. Proper retrieval mechanisms for different query types
3. Ground truth that accounts for temporal ordering and multiple valid answers

### Companion agent retrieval needs

Different query types require different retrieval logic:

| Query Type | Example | Retrieval Need |
|------------|---------|----------------|
| Fact | "What's my dog's name?" | Find relevant fact, latest wins if updated |
| State | "What am I wearing?" | Most recent state update |
| Episodic | "Remember when I was stressed about X?" | Specific moment, emotional context |
| Relationship | "Who is Sarah?" | Structured knowledge about a person |
| Pattern | "When do I usually feel tired?" | Aggregation across memories |
| Proactive | (no query) | Surface relevant memories based on context |

## Recommendations

1. **Pivot to retrieval experiments** - Design and test retrieval mechanisms before revisiting extraction
2. **Build proper benchmark** - Create ground truth with human verification for different query types
3. **Separate concerns** - Test fact retrieval and episodic retrieval separately; they may need different approaches
4. **Consider temporal ordering** - Any retrieval system needs to understand that recent memories may override older ones

## Files

- `results/experiment_run_20260114_211315.json` - Final experiment results
- `PLAN.md` - Original experiment design
- Code in this directory implements the extraction and evaluation pipeline
