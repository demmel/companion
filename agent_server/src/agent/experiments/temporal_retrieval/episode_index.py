"""
Episode index for temporal retrieval.

Provides time-indexed, topic-based, and mood-based episode lookup.
"""

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from sortedcontainers import SortedDict

from agent.experiments.temporal_retrieval.models import IndexedEpisode, TimeReference


class EpisodeIndex:
    """
    Time-indexed episode lookup with topic and mood access.

    Provides efficient querying by:
    - Time range
    - Topic
    - Mood
    """

    def __init__(self) -> None:
        # Main index: start_time -> episode
        self.by_time: SortedDict[datetime, IndexedEpisode] = SortedDict()

        # Secondary indices
        self.by_topic: dict[str, list[IndexedEpisode]] = {}
        self.by_mood: dict[str, list[IndexedEpisode]] = {}

        # All episodes by ID for quick lookup
        self.by_id: dict[str, IndexedEpisode] = {}

    def add_episode(self, episode: IndexedEpisode) -> None:
        """Add an episode to all indices."""
        # Time index
        self.by_time[episode.start_time] = episode

        # ID lookup
        self.by_id[episode.id] = episode

        # Topic index
        for topic in episode.topics:
            topic_lower = topic.lower()
            if topic_lower not in self.by_topic:
                self.by_topic[topic_lower] = []
            self.by_topic[topic_lower].append(episode)

        # Mood index
        for mood in episode.moods:
            mood_lower = mood.lower()
            if mood_lower not in self.by_mood:
                self.by_mood[mood_lower] = []
            self.by_mood[mood_lower].append(episode)

    def query_by_time_range(
        self, start: datetime, end: datetime
    ) -> list[IndexedEpisode]:
        """
        Find episodes that overlap with the given time range.

        Args:
            start: Start of time range
            end: End of time range

        Returns:
            List of episodes overlapping the range
        """
        results: list[IndexedEpisode] = []

        for episode_start in self.by_time.irange(minimum=None, maximum=end):
            episode = self.by_time[episode_start]
            # Check if episode overlaps with range
            if episode.end_time >= start and episode.start_time <= end:
                results.append(episode)

        return results

    def query_by_topic(self, topic: str) -> list[IndexedEpisode]:
        """
        Find episodes matching a topic.

        Args:
            topic: Topic to search for (case-insensitive)

        Returns:
            List of matching episodes
        """
        topic_lower = topic.lower()

        # Exact match first
        if topic_lower in self.by_topic:
            return self.by_topic[topic_lower].copy()

        # Partial match
        results: list[IndexedEpisode] = []
        for stored_topic, episodes in self.by_topic.items():
            if topic_lower in stored_topic or stored_topic in topic_lower:
                results.extend(episodes)

        # Deduplicate by ID
        seen_ids: set[str] = set()
        unique_results: list[IndexedEpisode] = []
        for ep in results:
            if ep.id not in seen_ids:
                seen_ids.add(ep.id)
                unique_results.append(ep)

        return unique_results

    def query_by_mood(self, mood: str) -> list[IndexedEpisode]:
        """
        Find episodes matching a mood.

        Args:
            mood: Mood to search for (case-insensitive)

        Returns:
            List of matching episodes sorted by emotional intensity
        """
        mood_lower = mood.lower()

        # Exact match
        if mood_lower in self.by_mood:
            episodes = self.by_mood[mood_lower].copy()
            # Sort by emotional intensity (higher first)
            episodes.sort(key=lambda e: e.emotional_intensity, reverse=True)
            return episodes

        # Check for related moods
        mood_groups = {
            "stressed": ["stressed", "anxious", "worried", "tense", "nervous"],
            "happy": ["happy", "excited", "joyful", "content", "pleased"],
            "sad": ["sad", "depressed", "down", "melancholy", "unhappy"],
            "angry": ["angry", "frustrated", "upset", "irritated", "annoyed"],
            "tired": ["tired", "exhausted", "sleepy", "fatigued", "drained"],
            "calm": ["calm", "relaxed", "peaceful", "serene", "tranquil"],
        }

        # Find which group the mood belongs to
        related_moods: list[str] = []
        for group_name, group_moods in mood_groups.items():
            if mood_lower in group_moods:
                related_moods = group_moods
                break

        # Search for all related moods
        results: list[IndexedEpisode] = []
        seen_ids: set[str] = set()
        for related_mood in related_moods or [mood_lower]:
            if related_mood in self.by_mood:
                for ep in self.by_mood[related_mood]:
                    if ep.id not in seen_ids:
                        seen_ids.add(ep.id)
                        results.append(ep)

        # Sort by emotional intensity
        results.sort(key=lambda e: e.emotional_intensity, reverse=True)
        return results

    def query(self, time_ref: TimeReference) -> list[IndexedEpisode]:
        """
        Query episodes based on a time reference.

        Args:
            time_ref: Parsed time reference

        Returns:
            List of matching episodes
        """
        if time_ref.start_time and time_ref.end_time:
            return self.query_by_time_range(time_ref.start_time, time_ref.end_time)
        elif time_ref.mood_filter:
            return self.query_by_mood(time_ref.mood_filter)
        elif time_ref.topic_filter:
            return self.query_by_topic(time_ref.topic_filter)
        elif time_ref.event_anchor:
            # Event anchor requires searching summaries/content
            return self.query_by_topic(time_ref.event_anchor)

        return []

    def get_all_episodes(self) -> list[IndexedEpisode]:
        """Get all episodes sorted by time."""
        return list(self.by_time.values())

    def get_episode_by_id(self, episode_id: str) -> IndexedEpisode | None:
        """Get a specific episode by ID."""
        return self.by_id.get(episode_id)

    def get_all_topics(self) -> list[str]:
        """Get all unique topics in the index."""
        return sorted(self.by_topic.keys())

    def get_all_moods(self) -> list[str]:
        """Get all unique moods in the index."""
        return sorted(self.by_mood.keys())

    def save(self, filepath: Path) -> None:
        """
        Save the index to a JSON file.

        Args:
            filepath: Path to save the index
        """
        # Convert to serializable format
        episodes_data = []
        for episode in self.by_time.values():
            ep_dict = asdict(episode)
            # Convert datetimes to ISO format
            ep_dict["start_time"] = episode.start_time.isoformat()
            ep_dict["end_time"] = episode.end_time.isoformat()
            episodes_data.append(ep_dict)

        data = {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "episode_count": len(episodes_data),
            "topics": self.get_all_topics(),
            "moods": self.get_all_moods(),
            "episodes": episodes_data,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: Path) -> "EpisodeIndex":
        """
        Load an index from a JSON file.

        Args:
            filepath: Path to the index file

        Returns:
            Loaded EpisodeIndex
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        index = cls()

        for ep_dict in data["episodes"]:
            # Parse datetimes
            ep_dict["start_time"] = datetime.fromisoformat(ep_dict["start_time"])
            ep_dict["end_time"] = datetime.fromisoformat(ep_dict["end_time"])

            episode = IndexedEpisode(
                id=ep_dict["id"],
                start_time=ep_dict["start_time"],
                end_time=ep_dict["end_time"],
                duration_minutes=ep_dict["duration_minutes"],
                memory_ids=ep_dict["memory_ids"],
                memory_count=ep_dict["memory_count"],
                title=ep_dict.get("title"),
                summary=ep_dict.get("summary"),
                topics=ep_dict.get("topics", []),
                moods=ep_dict.get("moods", []),
                emotional_intensity=ep_dict.get("emotional_intensity", 0.0),
                key_events=ep_dict.get("key_events", []),
            )
            index.add_episode(episode)

        return index

    def __len__(self) -> int:
        return len(self.by_time)

    def __repr__(self) -> str:
        return (
            f"EpisodeIndex(episodes={len(self)}, "
            f"topics={len(self.by_topic)}, moods={len(self.by_mood)})"
        )
