# Autonomous Research with Knowledge Graphs

**Status:** Experimental
**Goal:** Explore whether hypergraph knowledge structures improve agent memory and research capabilities

## Overview

This experiment investigates autonomous research powered by n-ary knowledge graphs (hypergraphs). The agent conducts multi-cycle research on topics, extracts structured facts, and builds queryable knowledge bases.

### Research Questions

1. **Does autonomous research work?** Can an agent build useful knowledge through self-directed web research?
2. **Are hypergraphs better than embeddings?** Does structured n-ary knowledge improve retrieval vs. pure semantic search?
3. **How should topic graphs integrate?** What's the best way to combine knowledge from multiple research sessions?
4. **What are the bottlenecks?** LLM calls? Graph size? Extraction quality?
5. **Is this cost-effective?** Can we get useful knowledge within reasonable LLM budget?

## Architecture

### Component Interfaces

All components use abstract interfaces (ABC) so implementations can be swapped:

- **`IKnowledgeGraph`**: Storage for n-ary facts (hyperedges)
- **`IFactExtractor`**: Extract structured facts from text
- **`IResearchOrchestrator`**: Coordinate research cycles
- **`IGraphIntegrator`**: Combine multiple topic graphs
- **`IRetriever`**: Find relevant facts for queries

**Design principle:** Interfaces matter, implementations don't. We can rewrite any component if we hit limitations.

### Current Implementations

**Knowledge Graph** (`knowledge_graph.py`):
- `SimpleHypergraph`: Dict-based with entity/predicate indexing
- Fast enough for experiments, can be rewritten if needed

**Fact Extraction** (`extraction.py`):
- `LLMFactExtractor`: Single-call extraction with structured prompts
- `ChunkedFactExtractor`: Handles large articles by chunking
- Uses **Mistral 3.2 Q4** (local) to avoid API costs

**Research** (`research.py`):
- `SequentialResearch`: Linear search → read → extract → think cycles
- Generates questions → searches web → fetches articles → extracts facts
- Multi-cycle with follow-up questions

**Integration** (`integration.py`):
- `NaiveIntegrator`: Just combine everything
- `BridgedIntegrator`: Detect entity overlaps, create bridges
- `IsolatedIntegrator`: Keep topics separated with tags
- `HierarchicalIntegrator`: Topic nodes with fact children
- `DeduplicatingIntegrator`: Merge duplicate facts

**Retrieval** (`retrieval.py`):
- `EmbeddingRetriever`: Cosine similarity baseline
- `KeywordRetriever`: Simple keyword matching
- `HybridRetriever`: Weighted combination
- `GraphTraversalRetriever`: Follow entity connections

## Usage

### Command Line (Recommended)

```bash
# Research a single topic
python -m agent.experiments.autonomous_research research "Byzantine Empire" --depth 3

# Research multiple topics
python -m agent.experiments.autonomous_research research "Quantum Computing" "Machine Learning" --depth 2

# Specify strategies
python -m agent.experiments.autonomous_research research "Coffee" \
    --integration bridged \
    --retrieval hybrid \
    --depth 2

# Compare all integration strategies
python -m agent.experiments.autonomous_research compare-integration "Byzantine Empire"

# Compare all retrieval strategies
python -m agent.experiments.autonomous_research compare-retrieval "Quantum Computing"

# Test all combinations (matrix test)
# Tests every integration strategy × retrieval strategy combination
# Shows quality metrics for each combination + best performers
python -m agent.experiments.autonomous_research matrix "Byzantine Empire" --depth 2

# List available strategies
python -m agent.experiments.autonomous_research list-strategies

# Verbose output
python -m agent.experiments.autonomous_research research "Coffee" --verbose
```

### Python API

```python
from agent.experiments.autonomous_research.experiment_runner import run_simple_experiment

# Research topics and build knowledge graph
topics = ["Byzantine Empire", "Coffee brewing methods"]
results = run_simple_experiment(topics, depth=2)
```

### Custom Experiment

```python
from agent.experiments.autonomous_research.experiment_runner import (
    ExperimentRunner,
    ExperimentConfig,
    STANDARD_RESEARCH_CONFIG,
    STANDARD_EXTRACTION_CONFIG,
)

config = ExperimentConfig(
    topics=["Quantum Computing", "Machine Learning"],
    research_depth=3,
    integration_strategy='bridged',
    retrieval_strategy='hybrid',
    research_config=STANDARD_RESEARCH_CONFIG,
    extraction_config=STANDARD_EXTRACTION_CONFIG
)

runner = ExperimentRunner(config)
results = runner.run_full_experiment()
```

### Compare Strategies

```python
from agent.experiments.autonomous_research.experiment_runner import (
    compare_integration_strategies,
    compare_retrieval_strategies,
    run_matrix_test
)

# Test all integration strategies
topics = ["Byzantine Empire"]
integration_results = compare_integration_strategies(topics, depth=2)

# Test all retrieval strategies
retrieval_results = compare_retrieval_strategies(topics, depth=2)

# Test all combinations (5 integration × 4 retrieval = 20 tests)
matrix_results = run_matrix_test(topics, depth=2)
```

### Direct Component Usage

```python
from agent.llm.router import create_llm
from agent.experiments.autonomous_research.knowledge_graph import SimpleHypergraph
from agent.experiments.autonomous_research.extraction import ChunkedFactExtractor
from agent.experiments.autonomous_research.research import SequentialResearch
from agent.experiments.autonomous_research.config import (
    get_default_research_config,
    get_default_extraction_config,
)

# Initialize
llm = create_llm()
research_config = get_default_research_config()
extraction_config = get_default_extraction_config()

extractor = ChunkedFactExtractor(llm, extraction_config)
researcher = SequentialResearch(llm, extractor, research_config)

# Research a topic
graph = researcher.research_topic("Quantum Entanglement", depth=3)

# Examine results
print(f"Built graph with {len(graph)} facts")
for fact in graph.get_all_facts():
    print(f"  {fact.predicate}: {fact.entities}")
```

## Evaluation & KPIs

### Metrics Tracked

The experiment automatically computes comprehensive quality metrics:

**Fact Quality:**
- Well-formed ratio: % of facts with complete structure
- Entity diversity: Uniqueness of entities (higher = more varied)
- Predicate diversity: Variety of relationship types
- Avg entities/fact: N-ary richness (higher = more complex facts)
- Entity reuse ratio: % entities in multiple facts (connectivity)

**Retrieval Quality:**
- Avg facts/query: How many relevant results returned
- Entity overlap: Do results contain query entities?
- Predicate diversity: Variety in result types
- Null result ratio: % queries with no results

**Graph Structure:**
- Bridge entities: Entities connecting multiple topics
- Isolated facts: Facts with no shared entities
- Redundancy: Duplicate facts detected
- Connectivity: Average entity/fact connections

**Cost Efficiency:**
- Total LLM calls and breakdown by operation
- Total time and breakdown by operation
- Facts per LLM call (extraction efficiency)
- Calls per fact (cost per fact)

### What We're Learning

*Run experiments and document findings here.*

**Questions answered by metrics:**
- Which integration preserves quality vs adds noise?
- Which retrieval finds most relevant facts?
- What's the cost per fact for different strategies?
- Does hypergraph enable better retrieval than embeddings?
- Are there unexpected bridges between topics?

### Example Matrix Test Output

```
📊 FACT QUALITY (by Integration Strategy):
Strategy             Well-Formed  Entity Div   Entity Reuse
------------------------------------------------------------
naive                     95.2%        0.67         45.3%
bridged                   95.2%        0.67         47.8%
isolated                  95.2%        0.67         45.3%
hierarchical              98.1%        0.62         52.1%
deduplicating             96.8%        0.71         48.9%

📊 RETRIEVAL QUALITY (by Retrieval Strategy):
Strategy             Facts/Query  Entity Overlap   Null Rate
------------------------------------------------------------
embedding                   4.8           1.82         0.0%
keyword                     3.2           2.14         5.0%
hybrid                      5.0           2.05         0.0%
graph_traversal             4.1           1.95         0.0%

📊 GRAPH STRUCTURE (by Integration Strategy):
Strategy             Bridge Ratio  Isolation   Redundancy
------------------------------------------------------------
naive                      15.3%       22.1%          0
bridged                    15.3%       22.1%          0
isolated                   15.3%       22.1%          0
hierarchical               15.3%        8.9%          0
deduplicating               8.2%       22.1%          7

🏆 BEST PERFORMERS:
  Fact Quality: hierarchical
  Retrieval: hybrid
  Graph Connectivity: hierarchical
```

### Known Limitations

**Already Identified:**
- Extraction quality depends on LLM (Mistral 3.2 Q4) performance
- No fact verification or contradiction detection yet
- Simple deduplication (might miss duplicates with different wording)
- Limited to text-based facts (no images, tables, etc.)
- Web search limited to DuckDuckGo HTML
- URL fetching can fail or timeout

**Possible Improvements:**
- Multi-pass extraction for better fact quality
- Fact verification against multiple sources
- Entity disambiguation (is "Turkey" the country or the bird?)
- Confidence scoring on extracted facts
- Parallel URL fetching for speed
- Better question generation for research depth

## File Structure

```
autonomous_research/
├── README.md                   # This file
├── interfaces.py               # Component ABCs
├── knowledge_graph.py          # SimpleHypergraph implementation
├── extraction.py               # LLMFactExtractor
├── research.py                 # SequentialResearch orchestrator
├── integration.py              # Graph integration strategies
├── retrieval.py                # Fact retrieval strategies
├── experiment_runner.py        # Main experiment harness
└── results/                    # Generated experiment results
    ├── experiment_YYYYMMDD_HHMMSS.json
    └── graph_YYYYMMDD_HHMMSS.json
```

## Design Decisions

### Why Hypergraphs?

Binary graphs can't represent real-world facts:
- ❌ Binary: "Byzantine Empire" → "traded_with" → "Venice"
- ✅ N-ary: Trade(trader=Byzantine Empire, partner=Venice, good=silk, time=10th century, region=Mediterranean)

Real facts involve multiple entities with specific roles.

### Why Local Models?

Using **Mistral 3.2 Q4 (local)** instead of Claude API to:
- Avoid burning through API budget on experiments
- Enable unlimited experimentation iterations
- Keep costs predictable

Trade-off: Lower quality extraction, but acceptable for learning what works.

### Why Simple Implementations?

Starting with dicts and simple algorithms because:
- Fast to implement and iterate
- Easy to understand and debug
- Good enough to test ideas
- Can rewrite with better structures when we hit bottlenecks

**Premature optimization is the enemy of experimentation.**

## Next Steps

Depending on what we learn:

1. **If extraction works well:**
   - Add fact verification
   - Try entity linking
   - Build larger knowledge bases

2. **If retrieval beats embeddings:**
   - Integrate with main agent memory system
   - Add graph reasoning capabilities
   - Explore meta-learning on graph structure

3. **If integration discovers useful bridges:**
   - Add automatic connection detection
   - Enable serendipitous discovery
   - Build cross-topic reasoning

4. **If cost is acceptable:**
   - Enable autonomous research in agent
   - Build persistent knowledge base
   - Create research projects feature

5. **If any component sucks:**
   - Rewrite that component
   - Try different approach
   - Learn from failure

## Running Experiments

```bash
# From agent_server directory
cd agent_server

# Quick experiment
python -m agent.experiments.autonomous_research research "Byzantine Empire" --depth 2

# See all options
python -m agent.experiments.autonomous_research --help

# See command-specific options
python -m agent.experiments.autonomous_research research --help
```

Check `results/` directory for:
- JSON with experiment metrics
- Serialized knowledge graph
- Extraction statistics

## Contributing

This is an experimental sandbox. Break things, try ideas, rewrite components.

When you hit a bottleneck:
1. Document what broke and why
2. Propose alternative approach
3. Implement and compare
4. Update this README with learnings

**The goal is learning, not perfection.**
