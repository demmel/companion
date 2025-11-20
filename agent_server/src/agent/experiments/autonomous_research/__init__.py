"""
Autonomous Research Experiment

Explores knowledge graph-based autonomous research for AI agents.
"""

from .interfaces import (
    Fact,
    IKnowledgeGraph,
    IFactExtractor,
    IResearchOrchestrator,
    IGraphIntegrator,
    IRetriever,
)

from .knowledge_graph import SimpleHypergraph, create_fact

from .extraction import LLMFactExtractor, ChunkedFactExtractor

from .research import SequentialResearch

from .integration import (
    NaiveIntegrator,
    BridgedIntegrator,
    IsolatedIntegrator,
    HierarchicalIntegrator,
    DeduplicatingIntegrator,
    get_integrator,
    INTEGRATION_STRATEGIES,
)

from .retrieval import (
    EmbeddingRetriever,
    KeywordRetriever,
    HybridRetriever,
    GraphTraversalRetriever,
    get_retriever,
    RETRIEVAL_STRATEGIES,
)

from .experiment_runner import (
    ExperimentConfig,
    ExperimentRunner,
    run_simple_experiment,
    compare_integration_strategies,
    compare_retrieval_strategies,
    run_matrix_test,
    STANDARD_RESEARCH_CONFIG,
    STANDARD_EXTRACTION_CONFIG,
)

from .evaluation import (
    FactQualityMetrics,
    RetrievalQualityMetrics,
    GraphStructureMetrics,
    CostMetrics,
    ExperimentEvaluation,
    compute_fact_quality,
    compute_retrieval_quality,
    compute_graph_structure,
    get_cost_tracker,
)

from .test_questions import (
    TopicTestQuestions,
    TEST_QUESTIONS,
    get_test_questions,
    get_all_topics,
)

from .config import ResearchConfig, ExtractionConfig

__all__ = [
    # Interfaces
    "Fact",
    "IKnowledgeGraph",
    "IFactExtractor",
    "IResearchOrchestrator",
    "IGraphIntegrator",
    "IRetriever",
    # Knowledge Graph
    "SimpleHypergraph",
    "create_fact",
    # Extraction
    "LLMFactExtractor",
    "ChunkedFactExtractor",
    # Research
    "SequentialResearch",
    # Integration
    "NaiveIntegrator",
    "BridgedIntegrator",
    "IsolatedIntegrator",
    "HierarchicalIntegrator",
    "DeduplicatingIntegrator",
    "get_integrator",
    "INTEGRATION_STRATEGIES",
    # Retrieval
    "EmbeddingRetriever",
    "KeywordRetriever",
    "HybridRetriever",
    "GraphTraversalRetriever",
    "get_retriever",
    "RETRIEVAL_STRATEGIES",
    # Experiment Runner
    "ExperimentConfig",
    "ExperimentRunner",
    "run_simple_experiment",
    "compare_integration_strategies",
    "compare_retrieval_strategies",
    "run_matrix_test",
    "STANDARD_RESEARCH_CONFIG",
    "STANDARD_EXTRACTION_CONFIG",
    # Evaluation
    "FactQualityMetrics",
    "RetrievalQualityMetrics",
    "GraphStructureMetrics",
    "CostMetrics",
    "ExperimentEvaluation",
    "compute_fact_quality",
    "compute_retrieval_quality",
    "compute_graph_structure",
    "get_cost_tracker",
    # Test Questions
    "TopicTestQuestions",
    "TEST_QUESTIONS",
    "get_test_questions",
    "get_all_topics",
    # Configuration
    "ResearchConfig",
    "ExtractionConfig",
]
