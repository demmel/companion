# Dreams Experiment Findings

## Overview

This experiment explored generating dream-like narratives by traversing the agent's memory graph. The goal: allow the autonomous agent to "dream" at night, creating new DREAM-type memories.

## Key Findings

### Traversal Strategy Matters Most

| Strategy             | Effect                                                                           |
| -------------------- | -------------------------------------------------------------------------------- |
| **CONTRAST_SEEKING** | Best for dream-like quality. Jarring thematic jumps create surreal "dream logic" |
| **SEMANTIC_DRIFT**   | Too coherent. Feels like a focused narrative, not a dream                        |
| **EDGE_FOLLOWING**   | Follows actual graph relationships. Good for connected dreams                    |
| **RANDOM_JUMP**      | Truly random (ignores edges). Unpredictable but not necessarily dreamlike        |
| **RECENCY_WEIGHTED** | Biased toward recent memories                                                    |

**Insight:** Contrast-seeking creates the weirdness. Narrative style is secondary.

### Narrative Styles

| Style        | Best For                                                 |
| ------------ | -------------------------------------------------------- |
| **FRAGMENT** | Most dream-like. Ellipses, incomplete thoughts, drifting |
| **STREAM**   | Continuous flow, decent dream quality                    |
| **POETIC**   | Beautiful/shareable, but less like actual dreams         |
| **SENSORY**  | Rich physical details, good for immersive dreams         |

### Optimal Depth

- **Depth 3**: Too short, not enough material
- **Depth 5-7**: Sweet spot
- **Depth 15**: Diminishing returns, narrative becomes repetitive

### Seed Selection

Experiment 7 tested whether the seed memory shapes the dream. Finding: **the seed has outsized influence on themes**, but this is primarily because:

1. The LLM anchors on early content when weaving narrative
2. The agent's memory corpus is thematically homogeneous (all revolves around similar themes)

Experiment 9 (ordering test) showed that reordering the same 5 memories produces similar dreams regardless of which is "first" - suggesting the corpus homogeneity matters more than prompt order.

### Graph Topology

Experiment 8 tested hub memories (30+ edges) vs peripheral memories (0 edges). Finding: **topology doesn't matter for RANDOM_JUMP traversal** since it ignores edges entirely. Would only matter for EDGE_FOLLOWING.

## Dream Modes

Based on findings, we implemented three purpose-driven modes:

### 1. TODAY (Memory Consolidation)

- **Purpose:** Process memories since last sleep
- **Traversal:** Semantic drift (stay thematically connected within today)
- **Style:** Stream
- **Use case:** End-of-day processing, like human sleep consolidation

### 2. BIZARRE (Surreal Dreams)

- **Purpose:** Generate strange, jarring dreams like humans have
- **Traversal:** Contrast-seeking
- **Style:** Fragment
- **Use case:** The weird dreams that feel significant but make no logical sense

### 3. CONNECT (Insight Discovery)

- **Purpose:** Find hidden connections between memories
- **Traversal:** Random jump (diverse sampling)
- **Output:** Dream narrative + list of discovered connections
- **Use case:** Growing the memory graph by finding relationships the agent hadn't noticed

The CONNECT mode is unique: it returns `DiscoveredConnection` objects that could become new edges in the memory graph. Types: EXPLAINS, CAUSED, CLARIFIED_BY, CONTRADICTED_BY.

## What Dreams Are For

Dreams serve functional purposes for the agent:

1. **Consolidation** - Processing recent experiences (TODAY mode)
2. **Surreal experience** - Having the strange dreams humans have (BIZARRE mode)
3. **Connection discovery** - Finding patterns and growing the memory graph (CONNECT mode)

The dream narrative becomes a new DREAM-type memory, allowing the agent to remember its dreams.

## Implementation Notes

- Naming: "random_walk" was renamed to "random_jump" since it doesn't follow edges
- The LLM generates both narrative and theme extraction
- Connection discovery uses structured JSON output from the LLM
- Temporal constraints apply to new edges (must go forward in time)

## Future Experiment Directions

### Connection Validation

CONNECT mode discovers connections but doesn't validate them. Have the LLM (or human) evaluate whether discovered connections are valid before adding to graph. Track precision/recall over time.

### Edge-Following Dreams

We haven't tested EDGE_FOLLOWING traversal much. Does walking the actual graph structure produce more coherent dreams? Compare edge-following vs random-jump for the same seeds.

### Dream-to-Memory Loop

What happens when dreams become memories and then appear in future dreams? Does the agent dream about its dreams? Could be interesting or could spiral weirdly.

### Dream Type Selection

Should the agent choose which dream mode based on the day?

- High-emotion day → BIZARRE (processing)
- Many new memories → TODAY (consolidation)
- Sparse connections in graph → CONNECT (growth)

### Recurring Dreams

Do certain seeds/themes recur naturally? If the agent dreams 10 nights in a row, what patterns emerge?

### Longer-Form Connect

CONNECT with depth=5 found 4 connections. What about depth=10 or 15? More connections, or diminishing returns?

## Integration Recommendations

If integrating dreaming into the agent:

### 1. Add DREAM Memory Type

Create a new memory type for dreams. The narrative becomes the memory content. Include metadata: mode used, memories visited, connections discovered.

### 2. Sleep Trigger

Add a `SleepTrigger` that fires during idle periods (configurable threshold). When triggered:

1. Run 1-3 dreams (mix of modes)
2. Store dream narratives as DREAM memories
3. For CONNECT mode, optionally apply discovered edges to graph

### 3. Mode Selection Logic

Simple heuristic based on recent activity:

```python
if memories_since_sleep > threshold:
    mode = TODAY  # Lots to consolidate
elif high_emotional_content_today:
    mode = BIZARRE  # Process emotions
else:
    mode = CONNECT  # Grow the graph
```

### 4. Connection Application

For CONNECT mode discoveries:

- Don't auto-apply all connections
- Apply only high-confidence ones (could add confidence scoring)
- Or queue for later validation

### 5. Dream Recall

Agent could reference its dreams in conversation: "I dreamed about X last night." The DREAM memories are searchable like any other memory.
