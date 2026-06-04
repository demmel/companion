"""
Test data generation for temporal retrieval evaluation.

Generates test queries with ground truth based on ACTUAL episode data:
- Queries that target dates where episodes exist
- Expected episodes are the actual episodes from those dates
"""

import json
import random
from datetime import datetime, timedelta
from pathlib import Path

from agent.experiments.temporal_retrieval.episode_index import EpisodeIndex
from agent.experiments.temporal_retrieval.models import IndexedEpisode, TemporalQuery, TimeReference
from agent.llm.models import SupportedModel
from agent.llm.router import LLM


def generate_date_based_queries(
    index: EpisodeIndex,
    count: int = 30,
) -> list[TemporalQuery]:
    """
    Generate queries targeting specific dates where episodes exist.
    """
    queries: list[TemporalQuery] = []
    episodes = index.get_all_episodes()

    if not episodes:
        return queries

    # Group episodes by date
    by_date: dict[datetime, list[IndexedEpisode]] = {}
    for ep in episodes:
        date = ep.start_time.replace(hour=0, minute=0, second=0, microsecond=0)
        if date not in by_date:
            by_date[date] = []
        by_date[date].append(ep)

    dates = sorted(by_date.keys())
    if not dates:
        return queries

    # Reference time is the latest episode end
    ref_time = max(ep.end_time for ep in episodes)

    # Generate queries for each date
    templates = [
        ("What happened on {date}?", "absolute"),
        ("What did we discuss on {date}?", "absolute"),
        ("Remember {date}?", "absolute"),
        ("What was going on {date}?", "absolute"),
    ]

    for date in dates:
        date_episodes = by_date[date]

        # Format date for query
        date_str = date.strftime("%B %d")  # e.g., "October 13"
        weekday = date.strftime("%A")  # e.g., "Sunday"

        # Create a query for this date
        template, ref_type = random.choice(templates)
        query_text = template.format(date=date_str)

        time_ref = TimeReference(
            raw_text=query_text,
            ref_type=ref_type,
            start_time=date,
            end_time=date.replace(hour=23, minute=59, second=59),
        )

        queries.append(
            TemporalQuery(
                query_text=query_text,
                time_ref=time_ref,
                expected_episode_ids=[ep.id for ep in date_episodes[:5]],
                expected_content_keywords=[],
            )
        )

        # Also add weekday-based query
        query_text = f"What happened on {weekday}?"
        queries.append(
            TemporalQuery(
                query_text=query_text,
                time_ref=TimeReference(
                    raw_text=query_text,
                    ref_type="absolute",
                    start_time=date,
                    end_time=date.replace(hour=23, minute=59, second=59),
                ),
                expected_episode_ids=[ep.id for ep in date_episodes[:5]],
                expected_content_keywords=[],
            )
        )

    return queries[:count]


def generate_time_of_day_queries(
    index: EpisodeIndex,
    count: int = 20,
) -> list[TemporalQuery]:
    """
    Generate queries for time-of-day (morning, afternoon, evening).
    """
    queries: list[TemporalQuery] = []
    episodes = index.get_all_episodes()

    if not episodes:
        return queries

    # Group episodes by date and time of day
    by_date_time: dict[tuple[datetime, str], list[IndexedEpisode]] = {}

    for ep in episodes:
        date = ep.start_time.replace(hour=0, minute=0, second=0, microsecond=0)
        hour = ep.start_time.hour

        if 6 <= hour < 12:
            time_of_day = "morning"
        elif 12 <= hour < 18:
            time_of_day = "afternoon"
        else:
            time_of_day = "evening"

        key = (date, time_of_day)
        if key not in by_date_time:
            by_date_time[key] = []
        by_date_time[key].append(ep)

    templates = {
        "morning": [
            "What happened on {date} morning?",
            "What did we discuss that morning on {date}?",
        ],
        "afternoon": [
            "What happened on {date} afternoon?",
            "What did we talk about that afternoon on {date}?",
        ],
        "evening": [
            "What happened on {date} evening?",
            "What did we discuss that evening on {date}?",
        ],
    }

    hour_ranges = {
        "morning": (6, 12),
        "afternoon": (12, 18),
        "evening": (18, 24),
    }

    for (date, time_of_day), eps in by_date_time.items():
        date_str = date.strftime("%B %d")
        template = random.choice(templates[time_of_day])
        query_text = template.format(date=date_str)

        start_hour, end_hour = hour_ranges[time_of_day]
        start_time = date.replace(hour=start_hour)
        end_time = date.replace(hour=end_hour - 1, minute=59, second=59)

        queries.append(
            TemporalQuery(
                query_text=query_text,
                time_ref=TimeReference(
                    raw_text=query_text,
                    ref_type="relative",
                    start_time=start_time,
                    end_time=end_time,
                ),
                expected_episode_ids=[ep.id for ep in eps[:5]],
                expected_content_keywords=[],
            )
        )

    return queries[:count]


def generate_emotional_queries(
    index: EpisodeIndex,
    count: int = 30,
) -> list[TemporalQuery]:
    """
    Generate emotional/contextual queries based on actual episode moods/topics.
    """
    queries: list[TemporalQuery] = []
    episodes = index.get_all_episodes()

    if not episodes:
        return queries

    # Mood-based queries
    mood_templates = [
        "When I was {mood}",
        "What happened when I was {mood}?",
        "Remember when I felt {mood}?",
    ]

    for mood in index.get_all_moods()[:10]:
        matching = index.query_by_mood(mood)
        if matching:
            template = random.choice(mood_templates)
            query_text = template.format(mood=mood)

            queries.append(
                TemporalQuery(
                    query_text=query_text,
                    time_ref=TimeReference(
                        raw_text=query_text,
                        ref_type="emotional",
                        mood_filter=mood,
                    ),
                    expected_episode_ids=[ep.id for ep in matching[:5]],
                    expected_content_keywords=[],
                )
            )

    # Topic-based queries
    topic_templates = [
        "When we discussed {topic}",
        "What happened during the {topic} discussion?",
        "Remember talking about {topic}?",
    ]

    for topic in index.get_all_topics()[:10]:
        matching = index.query_by_topic(topic)
        if matching:
            template = random.choice(topic_templates)
            query_text = template.format(topic=topic)

            queries.append(
                TemporalQuery(
                    query_text=query_text,
                    time_ref=TimeReference(
                        raw_text=query_text,
                        ref_type="emotional",
                        topic_filter=topic,
                    ),
                    expected_episode_ids=[ep.id for ep in matching[:5]],
                    expected_content_keywords=[],
                )
            )

    random.shuffle(queries)
    return queries[:count]


def generate_test_dataset(
    index: EpisodeIndex,
    relative_count: int = 40,
    absolute_count: int = 30,
    emotional_count: int = 30,
    llm: LLM | None = None,
    model: SupportedModel | None = None,
) -> list[TemporalQuery]:
    """
    Generate complete test dataset based on actual episode data.
    """
    queries: list[TemporalQuery] = []

    # Date-based absolute queries
    queries.extend(generate_date_based_queries(index, absolute_count))

    # Time-of-day queries (relative)
    queries.extend(generate_time_of_day_queries(index, relative_count))

    # Emotional/contextual queries
    queries.extend(generate_emotional_queries(index, emotional_count))

    return queries


def save_test_dataset(queries: list[TemporalQuery], filepath: Path) -> None:
    """Save test dataset to JSON file."""
    data = []
    for q in queries:
        data.append(
            {
                "query_text": q.query_text,
                "time_ref": {
                    "raw_text": q.time_ref.raw_text,
                    "ref_type": q.time_ref.ref_type,
                    "start_time": (
                        q.time_ref.start_time.isoformat()
                        if q.time_ref.start_time
                        else None
                    ),
                    "end_time": (
                        q.time_ref.end_time.isoformat() if q.time_ref.end_time else None
                    ),
                    "mood_filter": q.time_ref.mood_filter,
                    "topic_filter": q.time_ref.topic_filter,
                },
                "expected_episode_ids": q.expected_episode_ids,
                "expected_content_keywords": q.expected_content_keywords,
            }
        )

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_test_dataset(filepath: Path) -> list[TemporalQuery]:
    """Load test dataset from JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    queries: list[TemporalQuery] = []
    for item in data:
        time_ref = TimeReference(
            raw_text=item["time_ref"]["raw_text"],
            ref_type=item["time_ref"]["ref_type"],
            start_time=(
                datetime.fromisoformat(item["time_ref"]["start_time"])
                if item["time_ref"]["start_time"]
                else None
            ),
            end_time=(
                datetime.fromisoformat(item["time_ref"]["end_time"])
                if item["time_ref"]["end_time"]
                else None
            ),
            mood_filter=item["time_ref"].get("mood_filter"),
            topic_filter=item["time_ref"].get("topic_filter"),
        )

        queries.append(
            TemporalQuery(
                query_text=item["query_text"],
                time_ref=time_ref,
                expected_episode_ids=item["expected_episode_ids"],
                expected_content_keywords=item.get("expected_content_keywords", []),
            )
        )

    return queries
