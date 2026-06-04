# Experiments

Self-contained research experiments that informed the agent's memory system. Each experiment lives in its own package and follows the standard layout below so results are reproducible and generated artifacts never end up in git.

## Standard procedure

### Layout

Every experiment is a package `experiments/<name>/` with:

```
<name>/
  PLAN.md        # design + hypotheses            (tracked)
  FINDINGS.md    # results summary + conclusions   (tracked)
  *.py           # experiment code                 (tracked)
  output/        # ALL generated artifacts         (gitignored)
    results/     #   metrics / evaluation JSON
    models/      #   trained model artifacts
    dataset/     #   generated datasets
    cache/       #   built indices / caches
```

Only `PLAN.md`, `FINDINGS.md`, and code are committed. **All generated output goes under `output/`**, which is ignored via the single rule `src/agent/experiments/*/output/` in `agent_server/.gitignore`. This is the whole reason the convention exists: one rule ignores every experiment's artifacts, regardless of how large they get.

### Writing output

Anchor paths to the experiment directory and nest them under `output/` — never hard-code a CWD-relative path:

```python
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "output"
results_dir = OUTPUT_DIR / "results"
results_dir.mkdir(parents=True, exist_ok=True)
```

If a script emits a curated `FINDINGS.md`, write it to the experiment root (`Path(__file__).parent / "FINDINGS.md"`) so it stays tracked — not into `output/`.

For variant × test-case × run experiments, reuse the shared framework in [`framework/`](framework/) (`ExperimentRunner`, `ExperimentStorage`, `ExperimentAnalyzer`) and point its `base_dir` at `OUTPUT_DIR`.

### Running

Per `agent_server/CLAUDE.md`, always use `uv run`:

```bash
uv run python -m agent.experiments.<name>.run_experiments
```

### Starting a new experiment

1. Create `experiments/<name>/` with a `PLAN.md`.
2. Write code that sends every artifact to `output/` (see above).
3. Record conclusions in `FINDINGS.md`.
4. Add a row to the index table below.

## Index

Status legend: **active** = under development · **complete** = concluded, findings stable · **exploratory** = separate research track.

| Experiment | Status | Key finding | Docs |
|---|---|---|---|
| [`memory_architecture`](memory_architecture/ARCHITECTURE.md) | design | Synthesized target memory system: classify query type first, then route (KG / similarity / episodes / topics). Consolidates the findings below. | [ARCHITECTURE](memory_architecture/ARCHITECTURE.md) |
| [`query_classification`](query_classification/FINDINGS.md) | active | Hybrid classifier ~86.9% accuracy and ~40% faster than LLM-only; embedding-only 82.5% on a held-out set. | [FINDINGS](query_classification/FINDINGS.md) · [PLAN](query_classification/PLAN.md) |
| [`unified_retrieval`](unified_retrieval/FINDINGS.md) | active | Meaningful evaluation needs ground-truth IR metrics, not "answerability"; similarity scales to 10K memories (P95 ~157ms). | [FINDINGS](unified_retrieval/FINDINGS.md) · [PLAN](unified_retrieval/PLAN.md) |
| [`temporal_retrieval`](temporal_retrieval/FINDINGS.md) | active | Episode-summary-only retrieval performs best (F1 ~0.47); time parsing 88.9% overall (emotional 100%, absolute 73%). | [FINDINGS](temporal_retrieval/FINDINGS.md) · [PLAN](temporal_retrieval/PLAN.md) |
| [`episode_summaries`](episode_summaries/FINDINGS.md) | complete | Hybrid LLM + rule-based detection reaches ~95% boundary quality; 226 episodes from 6,653 memories (~921× compression). | [FINDINGS](episode_summaries/FINDINGS.md) · [PLAN](episode_summaries/PLAN.md) · [investigation](episode_summaries/FRAGMENTATION_INVESTIGATION.md) · [future ideas](episode_summaries/FUTURE_IDEAS.md) |
| [`retrieval`](retrieval/FINDINGS.md) | complete | KG-aware retrieval F1 0.707 for state queries (vs 0.077 similarity); similarity MRR 0.500 for episodic (vs 0.028 KG); reference detection 100% recall. | [FINDINGS](retrieval/FINDINGS.md) · [PLAN](retrieval/PLAN.md) |
| [`topic_clustering`](topic_clustering/FINDINGS.md) | complete | Cross-action-type KNN (k≈15) surfaces semantic topics; embeddings strongly encode action type and must be filtered. | [FINDINGS](topic_clustering/FINDINGS.md) · [PLAN](topic_clustering/PLAN.md) |
| [`memory_extraction`](memory_extraction/FINDINGS.md) | complete | <1% hallucination rate; extraction is not the bottleneck — retrieval design is. | [FINDINGS](memory_extraction/FINDINGS.md) · [PLAN](memory_extraction/PLAN.md) |
| [`dreams`](dreams/FINDINGS.md) | complete | Contrast-seeking traversal yields the most dream-like output; optimal depth 5–7; three practical modes (TODAY / BIZARRE / CONNECT). | [FINDINGS](dreams/FINDINGS.md) · [PLAN](dreams/PLAN.md) |
| [`autonomous_research`](autonomous_research/README.md) | exploratory | Hypergraph knowledge-graph research orchestration; interfaces-first pluggable design. | [README](autonomous_research/README.md) |

## Memory system synthesis

[`memory_architecture/ARCHITECTURE.md`](memory_architecture/ARCHITECTURE.md) is the consolidated target design derived from the experiments above. The `query_classification`, `unified_retrieval`, and `temporal_retrieval` experiments are actively implementing and validating its components.

## Shared framework

[`framework/`](framework/) provides reusable, type-safe experiment infrastructure: variant/test-case definitions, a runner, JSON storage (save raw data, compute metrics later), aggregation, statistics, charts, and reports. Prefer it over bespoke runners for comparative experiments.
