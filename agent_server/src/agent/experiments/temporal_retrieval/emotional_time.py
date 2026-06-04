"""
Emotional/contextual time handling for temporal retrieval.

Implements multiple approaches for resolving queries like:
- "when I was stressed"
- "during the job search"
- "that rough period"
"""

import json
import re

from agent.experiments.episode_summaries.detection import cosine_similarity
from agent.experiments.temporal_retrieval.episode_index import EpisodeIndex
from agent.experiments.temporal_retrieval.models import IndexedEpisode
from agent.llm.models import SupportedModel
from agent.llm.router import LLM


class EmotionalTimeResolver:
    """
    Resolves emotional/contextual time references to specific episodes.

    Implements three approaches:
    A. Episode Metadata - Use pre-extracted mood and topic tags
    B. Semantic Search - Search episode summaries using embeddings
    C. LLM Filtering - Ask LLM which episodes match the query
    """

    def __init__(self, index: EpisodeIndex, llm: LLM | None = None):
        self.index = index
        self.llm = llm

    def resolve_by_metadata(
        self,
        query: str,
        mood_filter: str | None = None,
        topic_filter: str | None = None,
    ) -> list[IndexedEpisode]:
        """
        Approach A: Use episode metadata (mood, topics).

        Fast and doesn't require LLM calls, but limited to pre-extracted tags.

        Args:
            query: Original query text (for context)
            mood_filter: Mood to filter by
            topic_filter: Topic to filter by

        Returns:
            Matching episodes sorted by emotional intensity
        """
        results: list[IndexedEpisode] = []

        if mood_filter:
            results = self.index.query_by_mood(mood_filter)

        if topic_filter:
            topic_results = self.index.query_by_topic(topic_filter)
            if results:
                # Intersect with mood results
                mood_ids = {ep.id for ep in results}
                results = [ep for ep in topic_results if ep.id in mood_ids]
            else:
                results = topic_results

        # Sort by emotional intensity
        results.sort(key=lambda e: e.emotional_intensity, reverse=True)

        return results

    def resolve_by_semantic_search(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int = 10,
    ) -> list[IndexedEpisode]:
        """
        Approach B: Semantic search on episode summaries.

        Uses embedding similarity to find episodes matching the query.

        Args:
            query: Query text
            query_embedding: Embedding vector for the query
            top_k: Maximum episodes to return

        Returns:
            Matching episodes sorted by similarity
        """
        # This requires embedding the summaries - for now, we'll use
        # a simpler keyword-based approach as a fallback

        results: list[IndexedEpisode] = []
        all_episodes = self.index.get_all_episodes()

        # Extract keywords from query
        keywords = self._extract_keywords(query)

        # Score episodes by keyword matches in summary/title
        scored: list[tuple[float, IndexedEpisode]] = []
        for episode in all_episodes:
            score = self._keyword_match_score(episode, keywords)
            if score > 0:
                scored.append((score, episode))

        # Sort by score
        scored.sort(key=lambda x: x[0], reverse=True)

        return [ep for _, ep in scored[:top_k]]

    def resolve_by_llm_filter(
        self,
        query: str,
        model: SupportedModel,
        max_episodes_to_check: int = 20,
    ) -> list[IndexedEpisode]:
        """
        Approach C: LLM-based filtering.

        Asks the LLM which episodes match the emotional/contextual query.

        Args:
            query: Query text
            model: Model to use for filtering
            max_episodes_to_check: Maximum episodes to present to LLM

        Returns:
            Episodes that LLM determined match the query
        """
        if not self.llm:
            raise ValueError("LLM required for LLM-based filtering")

        # Get candidate episodes (sort by emotional intensity for emotional queries)
        all_episodes = self.index.get_all_episodes()
        all_episodes.sort(key=lambda e: e.emotional_intensity, reverse=True)
        candidates = all_episodes[:max_episodes_to_check]

        # Format episodes for LLM
        episode_descriptions = []
        for i, ep in enumerate(candidates):
            desc = f"[{i + 1}] "
            if ep.title:
                desc += f"{ep.title}: "
            if ep.summary:
                desc += ep.summary[:200]
            else:
                desc += f"Episode at {ep.start_time.strftime('%Y-%m-%d %H:%M')}"
            if ep.moods:
                desc += f" (moods: {', '.join(ep.moods)})"
            episode_descriptions.append(desc)

        episodes_text = "\n".join(episode_descriptions)

        prompt = f"""I have a collection of conversation episodes. The user is asking about a specific time period using an emotional or contextual reference.

Query: "{query}"

Here are the available episodes:
{episodes_text}

Which episode numbers (if any) match the query? Return a JSON array of matching episode numbers, e.g., [1, 3, 5]. If none match, return [].

JSON:"""

        response = self.llm.generate(
            model=model,
            prompt=prompt,
            caller="emotional_time_filtering",
        )

        # Parse response
        try:
            json_match = re.search(r"\[[\d,\s]*\]", response)
            if json_match:
                indices = json.loads(json_match.group())
                # Convert 1-indexed to 0-indexed
                results = []
                for idx in indices:
                    if 1 <= idx <= len(candidates):
                        results.append(candidates[idx - 1])
                return results
        except (json.JSONDecodeError, ValueError):
            pass

        return []

    def resolve(
        self,
        query: str,
        mood_filter: str | None = None,
        topic_filter: str | None = None,
        approach: str = "metadata",
        model: SupportedModel | None = None,
    ) -> list[IndexedEpisode]:
        """
        Resolve an emotional/contextual time reference.

        Args:
            query: Query text
            mood_filter: Optional mood filter
            topic_filter: Optional topic filter
            approach: "metadata", "semantic", or "llm"
            model: Model for LLM approach

        Returns:
            Matching episodes
        """
        if approach == "metadata":
            return self.resolve_by_metadata(query, mood_filter, topic_filter)
        elif approach == "semantic":
            # Use a placeholder embedding for now
            return self.resolve_by_semantic_search(query, [], top_k=10)
        elif approach == "llm":
            if not model:
                raise ValueError("Model required for LLM approach")
            return self.resolve_by_llm_filter(query, model)
        else:
            raise ValueError(f"Unknown approach: {approach}")

    def _extract_keywords(self, text: str) -> list[str]:
        """Extract keywords from text for matching."""
        # Simple keyword extraction
        words = re.findall(r"\b\w+\b", text.lower())

        # Filter out common words
        stopwords = {
            "i",
            "was",
            "the",
            "a",
            "an",
            "when",
            "during",
            "that",
            "about",
            "what",
            "how",
            "did",
            "we",
            "talk",
            "discuss",
            "remember",
            "felt",
            "feeling",
        }

        return [w for w in words if w not in stopwords and len(w) > 2]

    def _keyword_match_score(
        self, episode: IndexedEpisode, keywords: list[str]
    ) -> float:
        """Score an episode by keyword matches."""
        score = 0.0

        # Check title
        if episode.title:
            title_lower = episode.title.lower()
            for kw in keywords:
                if kw in title_lower:
                    score += 2.0

        # Check summary
        if episode.summary:
            summary_lower = episode.summary.lower()
            for kw in keywords:
                if kw in summary_lower:
                    score += 1.0

        # Check topics
        for topic in episode.topics:
            topic_lower = topic.lower()
            for kw in keywords:
                if kw in topic_lower or topic_lower in kw:
                    score += 1.5

        # Check moods
        for mood in episode.moods:
            mood_lower = mood.lower()
            for kw in keywords:
                if kw in mood_lower or mood_lower in kw:
                    score += 2.0

        return score


def evaluate_emotional_time_approaches(
    index: EpisodeIndex,
    llm: LLM,
    model: SupportedModel,
    test_queries: list[dict[str, str | list[str]]],
) -> dict[str, dict[str, float]]:
    """
    Evaluate different approaches for emotional time resolution.

    Args:
        index: Episode index
        llm: LLM router
        model: Model for LLM approach
        test_queries: List of test queries with expected results

    Returns:
        Metrics for each approach
    """
    resolver = EmotionalTimeResolver(index, llm)

    approaches = ["metadata", "semantic", "llm"]
    results: dict[str, dict[str, float]] = {}

    for approach in approaches:
        correct = 0
        total = 0

        for test in test_queries:
            query = str(test["query"])
            expected_ids = test.get("expected_episode_ids", [])
            mood = test.get("mood_filter")
            topic = test.get("topic_filter")

            try:
                if approach == "llm":
                    retrieved = resolver.resolve(
                        query=query,
                        mood_filter=str(mood) if mood else None,
                        topic_filter=str(topic) if topic else None,
                        approach=approach,
                        model=model,
                    )
                else:
                    retrieved = resolver.resolve(
                        query=query,
                        mood_filter=str(mood) if mood else None,
                        topic_filter=str(topic) if topic else None,
                        approach=approach,
                    )

                retrieved_ids = {ep.id for ep in retrieved}
                expected_set = set(expected_ids) if expected_ids else set()

                # Check for any overlap
                if retrieved_ids & expected_set:
                    correct += 1
                total += 1

            except Exception:
                total += 1

        accuracy = correct / total if total > 0 else 0.0
        results[approach] = {
            "accuracy": accuracy,
            "correct": float(correct),
            "total": float(total),
        }

    return results
