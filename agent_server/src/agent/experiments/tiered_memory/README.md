# Tiered Memory Experiment

An experimental implementation of hierarchical memory organization with 4 tiers of granularity, designed to test retrieval strategies across different levels of detail.

## Overview

This experiment extends the existing 2-tier memory system with 2 additional tiers:

### Memory Tiers

1. **Tier 1 - Atomic** (existing): Individual `MemoryElement` objects (triggers + actions) with embeddings
2. **Tier 2 - Trigger-Response Pairs** (existing): `TriggerHistoryEntry` with compressed summaries and embeddings
3. **Tier 3 - Temporal Conversations** (new): Contiguous sequences of trigger-response pairs grouped by conversation boundaries
4. **Tier 4 - Semantic Clusters** (new): Topic-based clusters of memories across time, grouped by embedding similarity

### Key Features

- **Multi-granularity retrieval**: Query across all tiers simultaneously
- **Drill-down capability**: Start with high-level summaries and drill into specifics as needed
- **LLM-guided selection**: Agent can choose appropriate detail level per query
- **Token efficiency**: Compare context window sizes at different granularities

## Project Structure

```
tiered_memory/
├── __init__.py                    # Package initialization
├── models.py                      # Data models for tiers 3 & 4
├── conversation_detection.py      # Tier 3: Detect conversation boundaries
├── semantic_clustering.py         # Tier 4: Create topic-based clusters
├── tiered_retrieval.py           # Multi-tier retrieval system
├── context_builder.py            # Format retrieval results
├── experiment_runner.py          # Main experiment script
├── analysis.py                   # Result analysis and reporting
└── README.md                     # This file
```

## Usage

### Running the Experiment

```bash
cd agent_server

# Run with a saved memory graph
python -m agent.experiments.tiered_memory.experiment_runner \
    --memory-graph /path/to/memory_graph.json \
    --output-dir ./experiment_results
```

### Options

- `--memory-graph`: Path to saved `MemoryGraph` JSON file (required)
- `--output-dir`: Directory to save experiment results (default: `./experiment_results`)
- `--skip-topic-detection`: Skip LLM-based topic shift detection in tier 3 (faster but less accurate)
- `--clustering-source`: Source for tier 4 clustering (`conversations`, `triggers`, or `memories`)

### Example with Existing Memory

```bash
# Assuming you have a saved memory graph from the agent
python -m agent.experiments.tiered_memory.experiment_runner \
    --memory-graph ~/.agent_data/memory_graph.json \
    --output-dir ./tiered_memory_results \
    --clustering-source conversations
```

## What the Experiment Does

1. **Load existing memory graph**: Loads tier 1 & 2 from saved state
2. **Build tier 3**: Detects conversation boundaries using:
   - Time gaps (default: 30+ minutes)
   - LLM-based topic shift detection
   - Generates compressed summaries for each conversation
3. **Build tier 4**: Creates semantic clusters using:
   - Hierarchical clustering on embeddings
   - LLM-generated topic labels and summaries
4. **Run retrieval tests**: Queries memories using different strategies:
   - `ALL_TIERS`: Search all 4 tiers simultaneously
   - `COARSE_TO_FINE`: Start with tier 4, drill down as needed
   - `LLM_SELECTED`: LLM chooses which tiers to query per query
5. **Generate analysis**: Compares strategies on:
   - Retrieval relevance (similarity scores)
   - Token efficiency (score per token)
   - Tier distribution

## Output

The experiment generates several output files:

### Per-Query Results
```
{query_slug}_{strategy}.txt  # Context windows for each query-strategy pair
```

### Analysis Reports
```
comparison_report.txt   # Comprehensive comparison of all strategies
efficiency_report.txt   # Token efficiency analysis
```

### Example Output Structure
```
experiment_results/
├── all_discussions_about_memory_systems_all_tiers.txt
├── all_discussions_about_memory_systems_coarse_to_fine.txt
├── all_discussions_about_memory_systems_llm_selected.txt
├── what_did_we_discuss_in_our_last_conversation_all_tiers.txt
├── ...
├── comparison_report.txt
└── efficiency_report.txt
```

## Interpreting Results

### Context Window Formats

1. **Standard Format**: Shows results grouped by tier with metadata and drill-down previews
2. **Drill-down Format**: Shows high-level results with expanded lower-tier details

### Analysis Metrics

- **Average Score**: Mean cosine similarity across retrieved results
- **Tokens (standard)**: Estimated token count for standard context format
- **Tokens (drill-down)**: Estimated token count with drill-down expansion
- **Efficiency**: Score per token ratio (higher = better)

### Strategy Comparison

- **ALL_TIERS**: Most comprehensive, highest token cost
- **COARSE_TO_FINE**: More efficient, focuses on high-level summaries
- **LLM_SELECTED**: Adaptive, balances relevance and efficiency

## Implementation Notes

### Reused Components

The experiment maximizes code reuse from the existing agent codebase:

- `embedding_service.py`: Embedding generation and similarity
- `similarity_scoring.py`: Cosine similarity calculations
- `memory_formation.py`: LLM summarization patterns
- `structured_llm.py`: Structured LLM calls
- `models.py`: Base memory models

### Isolation

This experiment is **completely isolated** from the main agent codebase:
- No modifications to existing memory system
- No integration into active agent
- Safe to run experiments without affecting production

### Configuration

Key parameters can be adjusted in the source files:

**conversation_detection.py**:
- `TIME_GAP_THRESHOLD_MINUTES`: Time gap for conversation boundaries (default: 30)
- `MIN_CONVERSATION_LENGTH`: Minimum trigger entries per conversation (default: 2)

**semantic_clustering.py**:
- `MIN_CLUSTER_SIZE`: Minimum elements per cluster (default: 2)
- `SIMILARITY_THRESHOLD`: Cosine similarity threshold for clustering (default: 0.6)
- `MAX_CLUSTERS`: Maximum number of clusters to create (default: 20)

**tiered_retrieval.py**:
- Similarity thresholds per tier
- Top-k results per tier

## Future Enhancements

Potential improvements to explore:

1. **Dynamic tier selection**: Train a model to predict optimal tier per query
2. **Hybrid strategies**: Mix results from multiple tiers with learned weights
3. **Temporal weighting**: Boost recent memories within clusters
4. **Cross-tier references**: Maintain bidirectional links between tiers
5. **Incremental clustering**: Update clusters as new memories arrive
6. **Query expansion**: Generate multiple queries per tier for better coverage

## Limitations

Current limitations to be aware of:

- **LLM calls**: Tier 3 & 4 construction requires many LLM calls (can be slow/expensive)
- **Fixed thresholds**: Time gaps and similarity thresholds are hardcoded
- **No persistence**: Tier 3 & 4 structures are regenerated each run
- **Memory overhead**: Stores embeddings at all tiers (4x embedding storage)
- **Static clustering**: Clusters don't update as memories evolve

## Contact

For questions or issues with this experiment, please refer to the main agent repository.
