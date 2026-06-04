"""Unified Retrieval Architecture Experiment.

This experiment implements a multi-strategy retrieval pipeline that routes queries
to different backends (KG, similarity, episodes, topics) based on query classification.
"""

from .models import (
    QueryType,
    RetrievalContext,
    DetectedReference,
    EpisodeSummary,
    QueryClassifier,
)
from .query_classifier import LLMQueryClassifier, RuleBasedQueryClassifier
from .unified_retriever import UnifiedRetriever

__all__ = [
    "QueryType",
    "RetrievalContext",
    "DetectedReference",
    "EpisodeSummary",
    "QueryClassifier",
    "LLMQueryClassifier",
    "RuleBasedQueryClassifier",
    "UnifiedRetriever",
]
