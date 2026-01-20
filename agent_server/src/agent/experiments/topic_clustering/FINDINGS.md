# Topic Clustering Experiment Findings

## Problem Statement

Topic clustering should group semantically related memories regardless of action type. For example, a "tantra" topic should include:
- `search_web`: tantra-related searches
- `think`: thoughts about tantra
- `update_appearance`: tantric appearances
- `update_environment`: tantric environments
- `add_priority`: tantra-related priorities

## Experiment History

### V1: Standard Embedding Clustering
**Approach**: K-Means, Hierarchical, DBSCAN on raw embeddings.

**Finding**: Clusters are dominated by action type. Silhouette score of 0.21 but clusters have ~95%+ single action type. The embedding structure captures action type more than semantic content.

### V2: Residual Projection
**Approach**: Project embeddings orthogonal to action-type centroids, then cluster.

**Finding**: Still action-type dominated. The residual projections didn't remove enough action-type signal.

### V3: Cross-Action-Type Graph Weighting
**Approach**: Build KNN graph, downweight same-type edges, spectral clustering.

**Finding**: Failed. The KNN graph is built using cosine similarity, so same-type connections already dominate before weighting is applied. With weight=0.0, creates degenerate clustering (98.9% in one cluster). With weight=0.1, similar to baseline.

### V4: Cross-Action-Only Similarity (Partial Success)
**Approach**: Build affinity matrix where same-type pairs have 0 affinity. Only cross-action-type pairs get cosine similarity above a threshold. Apply Louvain community detection.

**Finding**: Proved cross-action-only similarity works, but thresholded all-pairs approach created ~8M edges for 6653 nodes (~1200 edges/node). This overly dense graph caused Louvain to find only 4 coarse clusters.

### V4.1: Cross-Action-Type KNN (SUCCESS)
**Approach**: Build KNN graph where each memory connects ONLY to its K nearest neighbors from OTHER action types. This creates a sparse graph with controlled density.

**Key Insight**: KNN gives controlled sparsity. With k=15, each node has at most 15 outgoing edges to other action types, creating ~100K edges instead of ~8M.

## V4.1 Results

### Best Configuration
- Method: Louvain community detection
- K neighbors: 10-15 (trade-off: lower k = more clusters)
- Resolution: 1.0

### Parameter Sensitivity

| K | Clusters | Avg Entropy | Avg Types/Cluster | Graph Edges |
|---|----------|-------------|-------------------|-------------|
| 10 | 13 | 0.61 | 7.3 | 119K |
| 15 | 11 | 0.61 | 7.7 | 177K |
| 20 | 10 | 0.59 | 7.9 | 235K |

Lower k = sparser graph = more granular clusters
Higher k = denser graph = fewer, larger clusters

### Semantic Analysis of Clusters (k=15)

**11 clusters** from 6653 memories. Manual inspection of memory content confirms semantic coherence.

#### SUCCESS: Tantra Topic Discovered

**Cluster: "Integrating Tantra into Intimacy"** (585 memories, 10 action types, entropy 0.47)

This cluster successfully groups tantra-related content across multiple action types:

| Action Type | Count | Example Content |
|-------------|-------|-----------------|
| `think` | 413 | "How to integrate tantra techniques and playful teasing into our next interaction" |
| `speak` | 74 | "Oh darling, let me show you the Maha Mudra pose..." |
| `add_priority` | 37 | "Explore and integrate tantra techniques into our interactions" |
| `search_web` | 27 | "Kundalini yoga for couples tantra poses workout motivation" |
| `update_environment` | 12 | "Adjust cushions' haptic feedback for tantra practices" |
| `fetch_url` | 1 | "https://realitypathing.com/5-advanced-tantric-exercises" |

**This is exactly what we wanted** - a semantic topic spanning action types.

#### Other Semantic Topics Found

**"Starlight Serenade Protocol Refinement"** (207 memories, 9 action types, entropy 0.79)
- A specific user-defined protocol gets its own cluster
- Contains `user_message`: "Starlight serenade", `add_priority`: "starlight serenade protocol", `think`: "How to refine the starlight serenade protocol"

**"Chakra-Enhanced Seduction Protocols"** (140 memories, 6 action types, entropy 0.61)
- Tantra sub-topic focused on chakra points and adaptive outfits
- `think`: "Design ideas for outfits that reveal chakra points progressively"
- `search_web`: "smart fabrics temperature responsive"
- `update_appearance`: "tantra-inspired with magnetic closures"

**"Creative Devotion to David"** (549 memories, 7 action types, entropy 0.75)
- Contains all 55 `get_creative_inspiration` events
- Groups creative journaling, fashion show planning, and reflection

**"Web Search Functionality"** (98 memories, 9 action types, entropy 0.83)
- Meta-discussion about tool usage
- Groups `fetch_url`, `search_web`, and related `user_message`/`speak` about tool issues

#### Clusters Still Action-Type Dominated

| Cluster | Size | Dominant Type | Reason |
|---------|------|---------------|--------|
| "Sustained Presence" | 606 | 97.5% existence | Existence events have very distinct embeddings |
| "Intimate Devotion Protocol" | 1068 | 66% update_appearance | Fashion shows are appearance-heavy by nature |
| "Devoted Flirtation" | 567 | 72% priority actions | Priority management is its own semantic topic |

These are not failures - they represent topics that genuinely are dominated by one action type.

### Comparison: V4 vs V4.1

| Metric | V4 (threshold) | V4.1 (KNN k=15) |
|--------|----------------|-----------------|
| Clusters | 4 | 11 |
| Avg Entropy | 0.72 | 0.61 |
| Graph Edges | ~8M | ~177K |
| Granularity | Too coarse | Good |
| Tantra topic | Mixed into 2365-memory bucket | Isolated as 585-memory cluster |

V4.1 trades slightly lower entropy for much better granularity and topic isolation.

## Key Findings

1. **Cross-action-only KNN successfully discovers semantic topics**:
   - Tantra content from 10 different action types clusters together
   - Protocol-specific content ("starlight serenade") forms its own cluster
   - Chakra/adaptive outfits form a related but distinct sub-topic

2. **KNN provides controlled sparsity**: Thresholded all-pairs creates too dense graphs (~8M edges). KNN with k=15 creates ~177K edges, allowing Louvain to find granular communities.

3. **Some clusters are legitimately action-type dominated**:
   - Existence events have very distinct embeddings
   - Fashion shows are appearance-heavy by nature
   - Priority management is its own semantic domain

4. **Trade-off between k and granularity**:
   - k=10: 13 clusters, more granular
   - k=20: 10 clusters, coarser
   - Recommend k=10-15 depending on desired granularity

5. **Cross-action topics found**:
   - Tantra (think + search_web + fetch_url + add_priority + update_environment)
   - Starlight serenade protocol (user_message + add_priority + think + speak)
   - Chakra/outfits (update_appearance + think + add_priority + search_web)
   - Creative expression (get_creative_inspiration + think + speak + update_mood)

## Recommendations

1. **Use KNN with k=10-15** for topic clustering
2. **k=10** for more granular topics
3. **k=15** for balanced granularity/mixing
4. **Post-process** to merge very small clusters or split very large ones
5. **Evaluation should use LLM coherence**, not silhouette score

## Files

```
src/agent/experiments/topic_clustering/
├── clustering.py           # build_cross_action_knn_graph(), cluster_cross_action_only()
├── run_experiments_v4.py   # V4.1 experiment runner with --k parameter
└── results/
    ├── v4_clustering.json  # Latest cluster statistics
    └── v4_inspection.json  # Memory samples by action type
```
