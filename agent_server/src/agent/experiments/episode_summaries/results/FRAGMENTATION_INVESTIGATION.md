# Episode Detection Fragmentation Investigation

## Problem Statement

The description-first episode detection produced 1457 episodes (avg 4.6 memories) vs a previously reported 226 episodes. The output appears highly fragmented, with one episode per "trigger" (user message, action, etc.).

## Experiment Setup

Created two detection variants to compare:
1. **JSON format**: Asks LLM to return `[{"starts_at": N, "about": "..."}]`
2. **Description-first format**: Asks LLM to return `"description" index`

Both use the same underlying logic, just different output format instructions.

## Test Run Results (5 chunks, ~250 memories)

```
Format               Episodes     Avg Size     Small(≤10)  Medium(11-50)  Large(>50)
JSON                 25           10.0         16          8              1
Description-First    35           7.1          29          5              1
```

Boundary analysis:
- Shared boundaries: 18
- JSON-only boundaries: 7
- Description-first-only boundaries: 17

## Observations (Not Conclusions)

### What the data shows

1. In this test run, description-first produced 40% more episodes than JSON format
2. 17 boundaries appeared only in description-first output
3. Sample of description-first-only boundaries:
   - Index 42: "David said to me: ..."
   - Index 54: "I thought about..."
   - Index 66: "I continue to exist..."
   - Index 122: "I updated my appearance..."
   - Index 126: "My mood changed..."

### Questions this raises

1. **Is the 5-chunk sample representative?** The full dataset has 134 chunks. Behavior might differ across the dataset.

2. **Why the discrepancy with reported numbers?**
   - Previous "226 episodes" - what parameters was this run with?
   - Current "1457 episodes" - this was a full run with max_chunks=None
   - The test comparison used max_chunks=5

3. **Are the "extra" boundaries meaningful or noise?**
   - Some appear to be at action-type changes (mood, appearance)
   - But some are at user inputs which could be valid episode boundaries
   - Need to examine more systematically

4. **Is the format actually the cause?**
   - Both formats show fragmentation
   - The difference (25 vs 35) might be within normal LLM variance
   - Would need multiple runs to establish if difference is consistent

## What We Don't Know Yet

1. Full dataset comparison results (experiment running in background)
2. Whether the 226 number was from a different prompt entirely, not just format
3. Whether running the same format twice produces consistent results
4. What the "correct" number of episodes should be for this data

## Files Created

- `detection.py`: Added `detect_episodes_llm_json()` and `detect_episodes_llm_chunk_json()`
- `run_experiments.py`: Added `experiment-baseline-json` and `experiment-format-comparison` commands

## Next Steps

1. Wait for full comparison to complete
2. Compare raw LLM responses between formats to see if parsing differs
3. Consider running same experiment multiple times to check variance
4. Review what prompt/parameters produced the original 226 number

## Commands

```bash
# Run JSON format baseline
uv run python -m agent.experiments.episode_summaries.run_experiments experiment-baseline-json

# Run format comparison
uv run python -m agent.experiments.episode_summaries.run_experiments experiment-format-comparison

# Check results
ls agent_server/src/agent/experiments/episode_summaries/results/
```
