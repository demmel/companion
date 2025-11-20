"""
Configuration for autonomous research experiments.

All magic numbers and limits are defined here as configurable parameters.
"""

from dataclasses import dataclass


@dataclass
class ResearchConfig:
    """Configuration for research orchestrator behavior"""

    max_sources_per_cycle: int  # How many URLs to fetch per research cycle
    max_search_query_length: int  # Max length of search query string
    max_facts_for_followup: int  # Max facts to include in follow-up question prompt


@dataclass
class ExtractionConfig:
    """Configuration for fact extraction behavior"""

    chunk_size: int  # Characters per chunk for long articles
    extraction_temperature: float  # LLM temperature for extraction
