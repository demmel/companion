# Temporal Retrieval Experiment Findings

Generated: 2026-01-21T22:40:34.433040

## Summary

**Best Strategy:** A (F1: 0.469)

## Results by Strategy

| Strategy | Precision | Recall | F1 | Latency (ms) |
|----------|-----------|--------|-----|--------------|
| A | 0.469 | 0.470 | 0.469 | 0.0 |
| B | 0.469 | 0.470 | 0.469 | 0.5 |
| C | 0.469 | 0.470 | 0.469 | 147.4 |
| D | 0.469 | 0.470 | 0.469 | 147.2 |

## Time Parsing Accuracy

- Overall: 88.9%
- Relative time: 93.3%
- Absolute time: 73.3%
- Emotional time: 100.0%

## Recommendations

- **Episode Summary Only** performs best - summaries capture key information

## Next Steps

1. Test with larger dataset
2. Evaluate content relevance with human judges
3. Optimize latency for production use
4. Test edge cases in time parsing