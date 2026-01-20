# Topic Clustering Prototype

## Concept

### What is this?

Topic clustering groups semantically similar memories together, regardless of when they occurred. It discovers themes that span across time.

**Example**: Memories from different days about "David's work" cluster together:
- Monday: "David mentioned he has a big presentation coming up"
- Wednesday: "David was stressed about the Henderson account"
- Friday: "David said the presentation went well"

These form a "David's Work" topic cluster, even though they're temporally scattered.

### Why does this matter?

1. **Thematic organization**: "What do I know about X?" queries can search a topic rather than all memories
2. **Pattern discovery**: Clusters reveal what the agent and user talk about most
3. **Compression**: A topic summary can represent many related memories
4. **Navigation**: User can browse by topic ("Show me everything about our relationship")

### Core tension

Clusters are arbitrary boundaries. Real topics overlap, blend, and resist clean categorization. The question is: how to make useful groupings despite this messiness?

---

## Design

### Data Structures

```python
@dataclass
class TopicCluster:
    """A group of semantically related memories."""
    id: str
    name: str                       # LLM-generated topic name
    description: str                # LLM-generated description
    memory_ids: list[str]           # Memories in this cluster
    centroid: list[float]           # Average embedding
    coherence_score: float          # How tight is the cluster?
    keywords: list[str]             # Key terms in this topic

@dataclass
class ClusteringResult:
    """Result of clustering all memories."""
    clusters: list[TopicCluster]
    unclustered: list[str]          # Memory IDs that didn't fit anywhere
    silhouette_score: float         # Overall clustering quality
    method: str                     # Which algorithm was used
```

### Clustering Approaches to Try

**Approach A: K-Means**
- Fixed number of clusters (K)
- Requires choosing K upfront
- Every memory assigned to exactly one cluster
- Fast, well-understood

**Approach B: Hierarchical (Agglomerative)**
- Build tree of clusters bottom-up
- Can cut at different levels for different granularity
- No need to specify K upfront
- Reveals cluster structure

**Approach C: DBSCAN**
- Density-based clustering
- Automatically determines cluster count
- Can leave outliers unclustered
- Sensitive to parameters (eps, min_samples)

**Approach D: Soft clustering (GMM)**
- Memories can belong to multiple clusters
- Probabilistic membership
- Better reflects reality of overlapping topics
- More complex to work with

**Approach E: LLM-guided clustering**
- Use LLM to judge if memories belong together
- Start with embedding clusters, refine with LLM
- More expensive but potentially more meaningful
- Can generate topic names as part of clustering

### Topic Naming Approaches

**Simple**: Let LLM name cluster after seeing sample memories
```
Here are 10 related memories. What topic do they share?
Give a 2-5 word name for this topic.
```

**Structured**: Extract common themes first
```
What themes appear in these memories?
What would you call this topic?
What keywords define it?
```

**Contrastive**: Define topic by what makes it different
```
These memories are grouped together. What makes them similar?
What distinguishes this group from other memories?
```

---

## Research Questions

### Q1: What clustering algorithm produces the most coherent topics?

Compare K-Means, Hierarchical, DBSCAN on:
- Silhouette score (quantitative coherence)
- Manual review (do clusters make sense?)
- Topic interpretability (can we name them?)

### Q2: How many topics are natural for a given memory set?

Methods to determine K:
- Silhouette score sweep (find elbow)
- Hierarchical dendrogram analysis
- DBSCAN auto-detection
- LLM judgment ("are these the same topic?")

### Q3: How coherent are the resulting clusters?

For each cluster:
- Do the memories actually relate to each other?
- Is the topic name accurate?
- Are there outliers that don't belong?
- Are there memories that should be in this cluster but aren't?

### Q4: How should overlapping topics be handled?

Some memories legitimately belong to multiple topics:
- "David talked about work stress affecting our relationship"
- → Both "David's Work" and "Relationship" topics

Options:
- Hard assignment (pick one)
- Soft assignment (probability in each)
- Multi-label (belongs to multiple)

### Q5: Do topic summaries accurately represent the cluster?

Generate summaries, evaluate:
- Does summary capture what the cluster is about?
- Does summary mention key memories?
- Would searching the summary find the right cluster?

### Q6: How stable are clusters as new memories arrive?

Simulate adding memories:
- Do existing cluster assignments change?
- Do new topics emerge?
- How often should re-clustering happen?

---

## Experiments

### Experiment 1: Algorithm Comparison

**Setup**:
- Load all memories from test data
- Run K-Means (K=5,8,10,15), Hierarchical, DBSCAN
- For each, compute silhouette score

**Measure**:
- Silhouette score
- Number of clusters found (for DBSCAN)
- Cluster size distribution

**Output**:
```
K-Means (K=8):
  Silhouette: 0.35
  Sizes: [45, 32, 28, 22, 19, 15, 12, 7]

Hierarchical (cut at 8 clusters):
  Silhouette: 0.38
  Sizes: [40, 35, 30, 25, 18, 14, 10, 8]

DBSCAN (eps=0.3):
  Silhouette: 0.42
  Clusters found: 6
  Unclustered: 23 memories
  Sizes: [52, 38, 30, 22, 15, 10]
```

### Experiment 2: Optimal K Discovery

**Setup**:
- Run K-Means for K = 3, 5, 8, 10, 12, 15, 20
- Plot silhouette score vs K
- Cut hierarchical dendrogram at different levels

**Measure**:
- Silhouette at each K
- Elbow point
- Manual assessment of cluster quality at different K

**Output**:
```
K=3:  silhouette=0.28, clusters too broad
K=5:  silhouette=0.34, reasonable groupings
K=8:  silhouette=0.38, good specificity
K=10: silhouette=0.35, some clusters too small
K=15: silhouette=0.30, over-fragmented

Recommendation: K=8 balances coherence and specificity
```

### Experiment 3: Cluster Coherence Review

**Setup**:
- Take best clustering from Experiment 1-2
- For each cluster, sample 10 memories
- Manual review: do these belong together?

**Measure**:
- Coherence rating per cluster (1-5 scale)
- Outliers identified
- Missing memories identified

**Output**:
```
Cluster 0 "David's Work":
  Coherence: 4/5
  Sample memories all about work
  1 outlier: memory about weekend plans

Cluster 1 "Emotional States":
  Coherence: 3/5
  Mix of moods - some about Chloe's mood, some about David's
  Could potentially split into two clusters

...
```

### Experiment 4: Topic Naming Quality

**Setup**:
- Generate topic names using different prompts
- Compare: simple, structured, contrastive

**Measure**:
- Name accuracy (does it fit the cluster?)
- Name specificity (is it distinctive?)
- Name usefulness (would a user understand it?)

**Output**:
```
Cluster 0:
  Simple: "Work and Career"
  Structured: "David's Professional Life"
  Contrastive: "David's Work (vs personal life)"

  Best: "David's Professional Life" - specific and clear
```

### Experiment 5: Summary Quality

**Setup**:
- Generate summaries for each cluster
- Evaluate: completeness, accuracy, searchability

**Measure**:
- Does summary mention key themes?
- Any hallucinations?
- Can queries find the right cluster via summary?

**Output**:
```
Cluster "David's Work" summary:
"This topic covers David's professional life including his job responsibilities,
work stress, meetings, and career discussions. Key events include the Henderson
account presentation and general work-related conversations."

Evaluation:
  - Mentions Henderson account: YES
  - Mentions work stress: YES
  - Hallucinations: NONE
  - Query "Henderson presentation" matches: YES (score 0.72)
```

### Experiment 6: Topic Overlap Analysis

**Setup**:
- Run soft clustering (GMM)
- Identify memories with high probability in multiple clusters
- Analyze overlap patterns

**Measure**:
- Percentage of memories in multiple clusters (>30% probability)
- Which topics overlap most?
- Does hard assignment lose important information?

**Output**:
```
Multi-topic memories: 28/180 (15.5%)

Common overlaps:
  - "Emotional States" + "Relationship": 12 memories
  - "David's Work" + "Daily Life": 8 memories

Example multi-topic memory:
  "David was stressed about work and it affected our conversation"
  → Work: 45%, Emotional: 40%, Relationship: 15%
```

---

## Implementation Outline

### Files to Create

```
topic_clustering/
├── PLAN.md                 # This file
├── __init__.py
├── models.py               # TopicCluster, ClusteringResult dataclasses
├── clustering.py           # K-Means, Hierarchical, DBSCAN implementations
├── topic_naming.py         # LLM-based topic name generation
├── topic_summary.py        # LLM-based cluster summarization
├── evaluation.py           # Coherence metrics, manual review helpers
├── visualization.py        # Cluster visualization (optional)
├── run_experiments.py      # Main experiment runner
└── results/                # Output directory
```

### Key Functions

```python
# clustering.py
def cluster_kmeans(memories: list[Memory], k: int) -> ClusteringResult:
    """Cluster using K-Means."""

def cluster_hierarchical(memories: list[Memory], n_clusters: int) -> ClusteringResult:
    """Cluster using agglomerative hierarchical clustering."""

def cluster_dbscan(memories: list[Memory], eps: float, min_samples: int) -> ClusteringResult:
    """Cluster using DBSCAN."""

def find_optimal_k(memories: list[Memory], k_range: range) -> int:
    """Find optimal K using silhouette analysis."""

# topic_naming.py
def generate_topic_name(cluster: TopicCluster, memories: list[Memory], approach: str) -> str:
    """Generate a name for the topic cluster."""

def generate_topic_keywords(cluster: TopicCluster, memories: list[Memory]) -> list[str]:
    """Extract keywords that define this topic."""

# topic_summary.py
def generate_topic_summary(cluster: TopicCluster, memories: list[Memory]) -> str:
    """Generate a comprehensive summary of the topic."""

# evaluation.py
def calculate_coherence(cluster: TopicCluster, memories: list[Memory]) -> float:
    """Calculate coherence score for a cluster."""

def find_outliers(cluster: TopicCluster, memories: list[Memory], threshold: float) -> list[str]:
    """Find memories that don't fit well in the cluster."""
```

---

## Open Questions

### For experimentation:

1. **Granularity**: Should topics be broad ("Work") or specific ("Henderson Account")? Can we have both via hierarchy?

2. **Dynamic topics**: How to handle topics that evolve over time? "Our relationship" means different things at different stages.

3. **Singleton handling**: What to do with memories that don't fit any cluster? Force assignment or leave unclustered?

4. **Embedding quality**: Are the current embeddings good enough for semantic clustering? Would different embeddings help?

### For user input:

1. **Expected topics**: What topics does the user expect to see? This can validate clustering results.

2. **Use cases**: How will topics be used? Navigation? Search? Compression? Different uses may need different granularity.

3. **Topic count**: Roughly how many topics would be useful? 5? 20? 100?

---

## Success Criteria

This prototype is successful if:

1. **Clusters are coherent**: Manual review shows >80% of clusters make sense
2. **Topics are nameable**: LLM can generate accurate, useful topic names
3. **Silhouette is reasonable**: Score >0.3 indicates meaningful structure
4. **Summaries work**: Topic summaries can be used to find relevant clusters
5. **Clear recommendation**: We know which algorithm and K to use

---

## Next Steps After This Plan

1. User reviews and approves this plan
2. Implement clustering with multiple algorithms
3. Run Experiment 1-2 (algorithm comparison, optimal K)
4. Based on results, focus on best approach
5. Run remaining experiments
6. Document findings and recommendations
