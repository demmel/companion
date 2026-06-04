"""
Experiment runner for autonomous research.

Runs complete experiments: research → integrate → retrieve → evaluate.
"""

import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from agent.llm.router import create_llm

from .interfaces import IKnowledgeGraph, Fact
from .knowledge_graph import SimpleHypergraph
from .extraction import LLMFactExtractor, ChunkedFactExtractor
from .research import SequentialResearch
from .integration import get_integrator, INTEGRATION_STRATEGIES
from .retrieval import get_retriever, RETRIEVAL_STRATEGIES
from .config import ResearchConfig, ExtractionConfig
from .evaluation import (
    compute_fact_quality,
    compute_retrieval_quality,
    compute_graph_structure,
    ExperimentEvaluation,
    get_cost_tracker,
)
from .test_questions import get_test_questions

logger = logging.getLogger(__name__)


STANDARD_RESEARCH_CONFIG = ResearchConfig(
    max_sources_per_cycle=10,
    max_search_query_length=300,
    max_facts_for_followup=20,
)

STANDARD_EXTRACTION_CONFIG = ExtractionConfig(
    chunk_size=6000,
    extraction_temperature=0.1,
)


class ExperimentConfig:
    """Configuration for an experiment run"""

    def __init__(
        self,
        topics: List[str],
        research_depth: int,
        integration_strategy: str,
        retrieval_strategy: str,
        research_config: ResearchConfig,
        extraction_config: ExtractionConfig,
        output_dir: Optional[Path] = None,
    ):
        self.topics = topics
        self.research_depth = research_depth
        self.integration_strategy = integration_strategy
        self.retrieval_strategy = retrieval_strategy
        self.research_config = research_config
        self.extraction_config = extraction_config
        self.output_dir = output_dir or Path(__file__).parent / "output"


class ExperimentRunner:
    """
    Runs autonomous research experiments.

    Orchestrates the full pipeline:
    1. Research multiple topics
    2. Integrate graphs
    3. Test retrieval quality
    4. Generate reports
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.llm = create_llm()

        # Initialize components - use chunked extractor to handle long articles
        self.fact_extractor = ChunkedFactExtractor(self.llm, config.extraction_config)
        self.researcher = SequentialResearch(
            self.llm, self.fact_extractor, config.research_config
        )

        # Results
        self.topic_graphs: Dict[str, IKnowledgeGraph] = {}
        self.integrated_graph: Optional[IKnowledgeGraph] = None
        self.results: Dict[str, Any] = {}

        # Reset cost tracker for this experiment
        get_cost_tracker().reset()

    def run_full_experiment(self) -> Dict[str, Any]:
        """
        Run complete experiment pipeline.

        Returns:
            Results dictionary with metrics and findings
        """
        logger.info("=" * 80)
        logger.info("STARTING AUTONOMOUS RESEARCH EXPERIMENT")
        logger.info("=" * 80)

        start_time = datetime.now()

        # Phase 1: Research each topic
        logger.info("\n[PHASE 1] Researching Topics")
        logger.info("-" * 80)
        for topic in self.config.topics:
            logger.info(f"\nResearching: {topic}")
            graph = self.researcher.research_topic(
                topic=topic, depth=self.config.research_depth
            )
            self.topic_graphs[topic] = graph
            logger.info(
                f"✓ Built graph: {len(graph)} facts, {len(graph.get_all_entities())} entities"
            )

        # Phase 2: Integrate graphs
        logger.info("\n[PHASE 2] Integrating Knowledge Graphs")
        logger.info("-" * 80)
        logger.info(f"Strategy: {self.config.integration_strategy}")

        integrator = get_integrator(self.config.integration_strategy)
        self.integrated_graph = integrator.integrate(
            list(self.topic_graphs.values()),
            metadata={"topics": list(self.topic_graphs.keys())},
        )

        logger.info(f"✓ Integrated graph: {len(self.integrated_graph)} facts")

        # Compute graph structure metrics
        logger.info("\nComputing graph structure metrics...")
        graph_metrics = compute_graph_structure(
            self.integrated_graph,
            topic_metadata={"topics": list(self.topic_graphs.keys())},
        )

        # Phase 3: Test retrieval
        logger.info("\n[PHASE 3] Testing Retrieval")
        logger.info("-" * 80)
        logger.info(f"Strategy: {self.config.retrieval_strategy}")

        retrieval_results = self._test_retrieval()

        # Phase 4: Compute quality metrics
        logger.info("\n[PHASE 4] Computing Quality Metrics")
        logger.info("-" * 80)

        # Fact quality
        fact_quality = compute_fact_quality(self.integrated_graph)
        logger.info(f"✓ Fact quality: {fact_quality.well_formed_ratio:.1%} well-formed")

        # Get cost metrics
        cost_metrics = get_cost_tracker().metrics

        # Create evaluation
        evaluation = ExperimentEvaluation(
            fact_quality=fact_quality,
            retrieval_quality=retrieval_results["quality_metrics"],
            graph_structure=graph_metrics,
            cost=cost_metrics,
        )

        # Phase 5: Generate report
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        logger.info("\n[PHASE 5] Generating Report")
        logger.info("-" * 80)

        self.results = {
            "config": {
                "topics": self.config.topics,
                "research_depth": self.config.research_depth,
                "integration_strategy": self.config.integration_strategy,
                "retrieval_strategy": self.config.retrieval_strategy,
            },
            "timing": {
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_seconds": duration,
            },
            "graphs": {
                topic: {
                    "fact_count": len(graph),
                    "entity_count": len(graph.get_all_entities()),
                    "predicates": list(set(f.predicate for f in graph.get_all_facts())),
                }
                for topic, graph in self.topic_graphs.items()
            },
            "integrated": {
                "fact_count": len(self.integrated_graph),
                "entity_count": len(self.integrated_graph.get_all_entities()),
                "predicates": list(
                    set(f.predicate for f in self.integrated_graph.get_all_facts())
                ),
            },
            "retrieval": retrieval_results,
            "evaluation": evaluation.to_dict(),
        }

        self._save_results()
        self._print_summary()

        return self.results

    def _test_retrieval(self) -> Dict[str, Any]:
        """Test retrieval with sample queries"""
        assert self.integrated_graph is not None
        retriever = get_retriever(self.config.retrieval_strategy)

        # Generate test queries for each topic
        test_queries = []
        for topic in self.config.topics:
            # Use all test questions available
            test_q = get_test_questions(topic)
            test_queries.extend(test_q.questions)

        # Also test cross-topic queries for all topic pairs
        if len(self.config.topics) > 1:
            for i in range(len(self.config.topics) - 1):
                test_queries.append(
                    f"Connection between {self.config.topics[i]} and {self.config.topics[i+1]}"
                )

        results = []
        all_retrieved_facts = []

        for query in test_queries:
            logger.info(f"\nQuery: {query}")
            facts = retriever.retrieve(query, self.integrated_graph, top_k=5)
            all_retrieved_facts.append(facts)

            logger.info(f"Retrieved {len(facts)} facts:")
            for i, fact in enumerate(facts[:3]):  # Show top 3
                logger.info(
                    f"  {i+1}. {fact.predicate}: {', '.join(fact.entities.values())}"
                )

            results.append(
                {
                    "query": query,
                    "retrieved_count": len(facts),
                    "top_predicates": [f.predicate for f in facts[:5]],
                }
            )

        # Compute retrieval quality metrics
        quality_metrics = compute_retrieval_quality(test_queries, all_retrieved_facts)

        logger.info(
            f"\n✓ Retrieval quality: {quality_metrics.avg_retrieved_per_query:.1f} facts/query"
        )
        logger.info(f"  Entity overlap: {quality_metrics.avg_query_entity_overlap:.2f}")
        logger.info(f"  Null results: {quality_metrics.null_result_ratio:.1%}")

        return {
            "queries_tested": len(test_queries),
            "results": results,
            "quality_metrics": quality_metrics,
        }

    def _save_results(self):
        """Save experiment results to disk"""
        assert self.integrated_graph is not None
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # Save results JSON
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.config.output_dir / f"experiment_{timestamp}.json"

        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=2)

        logger.info(f"✓ Saved results to {results_file}")

        # Save integrated graph
        graph_file = self.config.output_dir / f"graph_{timestamp}.json"
        with open(graph_file, "w") as f:
            json.dump(self.integrated_graph.to_dict(), f, indent=2)

        logger.info(f"✓ Saved graph to {graph_file}")

    def _print_summary(self):
        """Print experiment summary"""
        logger.info("\n" + "=" * 80)
        logger.info("EXPERIMENT SUMMARY")
        logger.info("=" * 80)

        logger.info(f"\nTopics researched: {len(self.config.topics)}")
        for topic, stats in self.results["graphs"].items():
            logger.info(
                f"  • {topic}: {stats['fact_count']} facts, {stats['entity_count']} entities"
            )

        logger.info(f"\nIntegrated graph:")
        logger.info(f"  • Total facts: {self.results['integrated']['fact_count']}")
        logger.info(f"  • Total entities: {self.results['integrated']['entity_count']}")
        logger.info(
            f"  • Unique predicates: {len(self.results['integrated']['predicates'])}"
        )

        # Print evaluation metrics if available
        if "evaluation" in self.results:
            eval_data = self.results["evaluation"]

            logger.info(f"\nQuality Metrics:")
            fq = eval_data["fact_quality"]
            logger.info(f"  • Well-formed facts: {fq['well_formed_ratio']:.1%}")
            logger.info(f"  • Entity diversity: {fq['entity_diversity']:.2f}")
            logger.info(f"  • Avg entities/fact: {fq['avg_entities_per_fact']:.1f}")
            logger.info(f"  • Entity reuse: {fq['entity_reuse_ratio']:.1%}")

            logger.info(f"\nRetrieval Metrics:")
            rq = eval_data["retrieval_quality"]
            logger.info(f"  • Queries tested: {rq['total_queries']}")
            logger.info(f"  • Avg facts/query: {rq['avg_retrieved_per_query']:.1f}")
            logger.info(f"  • Entity overlap: {rq['avg_query_entity_overlap']:.2f}")
            logger.info(f"  • Null results: {rq['null_result_ratio']:.1%}")

            logger.info(f"\nGraph Structure:")
            gs = eval_data["graph_structure"]
            logger.info(f"  • Bridge entities: {gs['bridge_ratio']:.1%}")
            logger.info(f"  • Isolated facts: {gs['isolation_ratio']:.1%}")
            logger.info(f"  • Redundancy: {gs['redundancy_detected']} duplicates")

            logger.info(f"\nCost:")
            cost = eval_data["cost"]
            logger.info(f"  • Total LLM calls: {cost['total_llm_calls']}")
            logger.info(f"  • Total time: {cost['total_time_seconds']:.1f}s")
        else:
            logger.info(f"\nRetrieval tested:")
            logger.info(f"  • Queries: {self.results['retrieval']['queries_tested']}")

            logger.info(
                f"\nDuration: {self.results['timing']['duration_seconds']:.1f}s"
            )

        logger.info("\n" + "=" * 80)


def run_simple_experiment(topics: List[str], depth: int = 2):
    """
    Quick helper to run a simple experiment.

    Args:
        topics: List of topics to research
        depth: Research depth (number of cycles per topic)
    """
    config = ExperimentConfig(
        topics=topics,
        research_depth=depth,
        integration_strategy="bridged",
        retrieval_strategy="hybrid",
        research_config=STANDARD_RESEARCH_CONFIG,
        extraction_config=STANDARD_EXTRACTION_CONFIG,
    )

    runner = ExperimentRunner(config)
    return runner.run_full_experiment()


def compare_integration_strategies(topics: List[str], depth: int = 2):
    """
    Run experiments comparing different integration strategies.

    Args:
        topics: List of topics to research
        depth: Research depth
    """
    results = {}

    for strategy in INTEGRATION_STRATEGIES.keys():
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Testing integration strategy: {strategy}")
        logger.info(f"{'=' * 80}\n")

        config = ExperimentConfig(
            topics=topics,
            research_depth=depth,
            integration_strategy=strategy,
            retrieval_strategy="hybrid",
            research_config=STANDARD_RESEARCH_CONFIG,
            extraction_config=STANDARD_EXTRACTION_CONFIG,
        )

        runner = ExperimentRunner(config)
        results[strategy] = runner.run_full_experiment()

    # Print comparison
    logger.info("\n" + "=" * 80)
    logger.info("STRATEGY COMPARISON")
    logger.info("=" * 80)

    for strategy, result in results.items():
        logger.info(f"\n{strategy}:")
        logger.info(f"  Facts: {result['integrated']['fact_count']}")
        logger.info(f"  Entities: {result['integrated']['entity_count']}")
        logger.info(f"  Duration: {result['timing']['duration_seconds']:.1f}s")

    return results


def compare_retrieval_strategies(topics: List[str], depth: int = 2):
    """
    Run experiments comparing different retrieval strategies.

    Args:
        topics: List of topics to research
        depth: Research depth
    """
    # First, build a single integrated graph
    logger.info("Building knowledge graph for retrieval comparison...")

    # Build graphs with proper extractor
    llm = create_llm()
    fact_extractor = ChunkedFactExtractor(llm, STANDARD_EXTRACTION_CONFIG)
    researcher = SequentialResearch(llm, fact_extractor, STANDARD_RESEARCH_CONFIG)

    topic_graphs = {}
    for topic in topics:
        graph = researcher.research_topic(topic, depth)
        topic_graphs[topic] = graph

    # Integrate
    integrator = get_integrator("bridged")
    integrated_graph = integrator.integrate(
        list(topic_graphs.values()), metadata={"topics": list(topic_graphs.keys())}
    )

    # Test each retrieval strategy
    logger.info("\n" + "=" * 80)
    logger.info("RETRIEVAL STRATEGY COMPARISON")
    logger.info("=" * 80)

    # Use all test questions for all topics
    test_queries = []
    for topic in topics:
        test_q = get_test_questions(topic)
        test_queries.extend(test_q.questions)

    results = {}
    for strategy in RETRIEVAL_STRATEGIES.keys():
        logger.info(f"\nStrategy: {strategy}")
        retriever = get_retriever(strategy)

        total_retrieved = 0
        for query in test_queries:
            facts = retriever.retrieve(query, integrated_graph, top_k=5)
            total_retrieved += len(facts)

        avg_retrieved = total_retrieved / len(test_queries) if test_queries else 0
        logger.info(
            f"  Average: {avg_retrieved:.1f} facts/query across {len(test_queries)} queries"
        )

        results[strategy] = {
            "avg_retrieved": avg_retrieved,
            "queries_tested": len(test_queries),
        }

    return results


def run_matrix_test(topics: List[str], depth: int = 2):
    """
    Run full matrix test: all integration × retrieval strategy combinations.

    Args:
        topics: List of topics to research
        depth: Research depth

    Returns:
        Dict mapping (integration, retrieval) -> results
    """
    logger.info("\n" + "=" * 80)
    logger.info("MATRIX TEST: All Strategy Combinations")
    logger.info("=" * 80)
    logger.info(f"Topics: {topics}")
    logger.info(f"Integration strategies: {len(INTEGRATION_STRATEGIES)}")
    logger.info(f"Retrieval strategies: {len(RETRIEVAL_STRATEGIES)}")
    logger.info(
        f"Total combinations: {len(INTEGRATION_STRATEGIES) * len(RETRIEVAL_STRATEGIES)}"
    )
    logger.info("=" * 80)

    matrix_results = {}

    # Build topic graphs once (reuse for all combinations)
    logger.info("\n[BUILDING TOPIC GRAPHS]")
    llm = create_llm()
    fact_extractor = ChunkedFactExtractor(llm, STANDARD_EXTRACTION_CONFIG)
    researcher = SequentialResearch(llm, fact_extractor, STANDARD_RESEARCH_CONFIG)

    topic_graphs = {}
    for topic in topics:
        logger.info(f"Researching: {topic}")
        graph = researcher.research_topic(topic, depth)
        topic_graphs[topic] = graph
        logger.info(f"✓ {len(graph)} facts")

    # Test all combinations
    total = len(INTEGRATION_STRATEGIES) * len(RETRIEVAL_STRATEGIES)
    current = 0

    for int_strategy in INTEGRATION_STRATEGIES.keys():
        for ret_strategy in RETRIEVAL_STRATEGIES.keys():
            current += 1
            logger.info(
                f"\n[{current}/{total}] Testing: {int_strategy} + {ret_strategy}"
            )
            logger.info("-" * 80)

            # Integrate with this strategy
            integrator = get_integrator(int_strategy)
            integrated = integrator.integrate(
                list(topic_graphs.values()),
                metadata={"topics": list(topic_graphs.keys())},
            )

            # Compute graph quality metrics
            fact_quality = compute_fact_quality(integrated)
            graph_structure = compute_graph_structure(
                integrated, topic_metadata={"topics": topics}
            )

            # Test retrieval with this strategy
            retriever = get_retriever(ret_strategy)

            # Generate test queries using test questions
            test_queries = []
            all_retrieved = []
            for topic in topics:
                test_q = get_test_questions(topic)
                test_queries.extend(test_q.questions)  # Use all test questions

            for query in test_queries:
                facts = retriever.retrieve(query, integrated, top_k=5)
                all_retrieved.append(facts)

            # Compute retrieval quality
            retrieval_quality = compute_retrieval_quality(test_queries, all_retrieved)

            matrix_results[(int_strategy, ret_strategy)] = {
                "fact_quality": fact_quality,
                "graph_structure": graph_structure,
                "retrieval_quality": retrieval_quality,
            }

            logger.info(
                f"  Facts: {len(integrated)}, Quality: {fact_quality.well_formed_ratio:.1%}"
            )
            logger.info(
                f"  Retrieval: {retrieval_quality.avg_retrieved_per_query:.1f} facts/query, Overlap: {retrieval_quality.avg_query_entity_overlap:.2f}"
            )

    # Print comparison matrix
    logger.info("\n" + "=" * 80)
    logger.info("MATRIX TEST RESULTS")
    logger.info("=" * 80)

    # Fact Quality by Integration Strategy
    logger.info("\n📊 FACT QUALITY (by Integration Strategy):")
    logger.info(
        f"{'Strategy':<20} {'Well-Formed':<12} {'Entity Div':<12} {'Entity Reuse':<12}"
    )
    logger.info("-" * 60)
    for int_strategy in INTEGRATION_STRATEGIES.keys():
        qualities = [
            matrix_results[(int_strategy, r)]["fact_quality"]
            for r in RETRIEVAL_STRATEGIES.keys()
        ]
        avg_wellformed = sum(q.well_formed_ratio for q in qualities) / len(qualities)
        avg_diversity = sum(q.entity_diversity for q in qualities) / len(qualities)
        avg_reuse = sum(q.entity_reuse_ratio for q in qualities) / len(qualities)
        logger.info(
            f"{int_strategy:<20} {avg_wellformed:>10.1%}  {avg_diversity:>10.2f}  {avg_reuse:>10.1%}"
        )

    # Retrieval Quality by Retrieval Strategy
    logger.info("\n📊 RETRIEVAL QUALITY (by Retrieval Strategy):")
    logger.info(
        f"{'Strategy':<20} {'Facts/Query':<12} {'Entity Overlap':<15} {'Null Rate':<12}"
    )
    logger.info("-" * 60)
    for ret_strategy in RETRIEVAL_STRATEGIES.keys():
        qualities = [
            matrix_results[(int_strategy, ret_strategy)]["retrieval_quality"]
            for int_strategy in INTEGRATION_STRATEGIES.keys()
        ]
        avg_facts = sum(q.avg_retrieved_per_query for q in qualities) / len(qualities)
        avg_overlap = sum(q.avg_query_entity_overlap for q in qualities) / len(
            qualities
        )
        avg_null = sum(q.null_result_ratio for q in qualities) / len(qualities)
        logger.info(
            f"{ret_strategy:<20} {avg_facts:>10.1f}  {avg_overlap:>13.2f}  {avg_null:>10.1%}"
        )

    # Graph Structure by Integration Strategy
    logger.info("\n📊 GRAPH STRUCTURE (by Integration Strategy):")
    logger.info(
        f"{'Strategy':<20} {'Bridge Ratio':<13} {'Isolation':<11} {'Redundancy':<12}"
    )
    logger.info("-" * 60)
    for int_strategy in INTEGRATION_STRATEGIES.keys():
        structures = [
            matrix_results[(int_strategy, r)]["graph_structure"]
            for r in RETRIEVAL_STRATEGIES.keys()
        ]
        avg_bridge = sum(s.bridge_ratio for s in structures) / len(structures)
        avg_isolation = sum(s.isolation_ratio for s in structures) / len(structures)
        avg_redundancy = sum(s.redundancy_detected for s in structures) / len(
            structures
        )
        logger.info(
            f"{int_strategy:<20} {avg_bridge:>11.1%}  {avg_isolation:>9.1%}  {avg_redundancy:>10.0f}"
        )

    # Best performers
    logger.info("\n🏆 BEST PERFORMERS:")

    # Best fact quality
    best_quality = max(
        INTEGRATION_STRATEGIES.keys(),
        key=lambda s: sum(
            matrix_results[(s, r)]["fact_quality"].well_formed_ratio
            for r in RETRIEVAL_STRATEGIES.keys()
        ),
    )
    logger.info(f"  Fact Quality: {best_quality}")

    # Best retrieval
    best_retrieval = max(
        RETRIEVAL_STRATEGIES.keys(),
        key=lambda s: sum(
            matrix_results[(i, s)]["retrieval_quality"].avg_query_entity_overlap
            for i in INTEGRATION_STRATEGIES.keys()
        ),
    )
    logger.info(f"  Retrieval: {best_retrieval}")

    # Most connected graph
    best_connectivity = max(
        INTEGRATION_STRATEGIES.keys(),
        key=lambda s: sum(
            matrix_results[(s, r)]["graph_structure"].bridge_ratio
            for r in RETRIEVAL_STRATEGIES.keys()
        ),
    )
    logger.info(f"  Graph Connectivity: {best_connectivity}")

    logger.info("\n" + "=" * 80)

    return matrix_results


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Run matrix test
    topics = ["Byzantine Empire", "Quantum Computing"]
    results = run_matrix_test(topics, depth=2)

    print("\nMatrix test complete! Check results directory for detailed output.")
