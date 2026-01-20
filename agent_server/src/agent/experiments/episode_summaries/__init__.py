"""
Episode Summaries Experiment

Detects conversation episode boundaries and generates LLM-based summaries.
"""

from agent.experiments.episode_summaries.models import (
    Episode,
    EpisodeDetectionResult,
    GapThresholdStats,
    GapSweepResult,
)
from agent.experiments.episode_summaries.detection import (
    detect_episodes_by_gap,
    analyze_gap_distribution,
    run_gap_threshold_sweep,
)
from agent.experiments.episode_summaries.summarization import (
    generate_episode_summary,
    generate_episode_title,
    generate_summary_at_detail_level,
    SUMMARY_STYLES,
)

__all__ = [
    # Models
    "Episode",
    "EpisodeDetectionResult",
    "GapThresholdStats",
    "GapSweepResult",
    # Detection
    "detect_episodes_by_gap",
    "analyze_gap_distribution",
    "run_gap_threshold_sweep",
    # Summarization
    "generate_episode_summary",
    "generate_episode_title",
    "generate_summary_at_detail_level",
    "SUMMARY_STYLES",
]
