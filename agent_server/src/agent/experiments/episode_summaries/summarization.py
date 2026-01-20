"""
LLM-based episode summarization.

Generates summaries, titles, and key events for detected episodes.
"""

from agent.memory.dag.models import MemoryElement
from agent.experiments.episode_summaries.models import Episode
from agent.llm.router import LLM
from agent.llm.models import SupportedModel
from agent.chain_of_action.prompts import format_section


SUMMARY_STYLES = {
    "basic": """Summarize this conversation episode in one paragraph.
What happened? What was discussed? How did it end?""",
    "structured": """For this conversation episode, provide:
1. Title (3-7 words)
2. Main events or topics
3. Emotional arc (how did mood change?)
4. Key takeaways

Format as:
Title: [title]
Events: [bullet list]
Emotional arc: [description]
Key takeaways: [bullet list]""",
    "narrative": """Tell the story of this conversation as a narrative.
What happened from beginning to end? Write in past tense, describing the flow of the conversation.""",
    "question": """What questions could this conversation answer?
For each, provide a brief answer.

Format as:
Q: [question]
A: [answer]""",
}


def format_memories_for_prompt(memories: list[MemoryElement], episode: Episode) -> str:
    """Format memories for inclusion in a prompt."""
    # Get memories in order
    episode_memory_ids = set(episode.memory_ids)
    episode_memories = [m for m in memories if m.id in episode_memory_ids]
    episode_memories.sort(key=lambda m: m.timestamp)

    lines = []
    lines.append(
        f"Episode: {episode.start_time.strftime('%Y-%m-%d %H:%M')} - "
        f"{episode.end_time.strftime('%H:%M')}"
    )
    lines.append(f"Duration: {episode.duration_minutes:.1f} minutes")
    lines.append(f"Memories: {episode.memory_count}")
    lines.append("")
    lines.append("--- Conversation Content ---")
    lines.append("")

    for memory in episode_memories:
        timestamp = memory.timestamp.strftime("%H:%M:%S")
        lines.append(f"[{timestamp}] {memory.content}")

    return "\n".join(lines)


def generate_episode_summary(
    episode: Episode,
    memories: list[MemoryElement],
    llm: LLM,
    model: SupportedModel,
    style: str = "basic",
) -> str:
    """
    Generate a summary for an episode using the specified style.

    Args:
        episode: The episode to summarize
        memories: All memories (will filter to episode's memories)
        llm: LLM router instance
        model: Model to use for generation
        style: Summary style - "basic", "structured", "narrative", or "question"

    Returns:
        Generated summary text
    """
    if style not in SUMMARY_STYLES:
        raise ValueError(
            f"Unknown style: {style}. Must be one of {list(SUMMARY_STYLES.keys())}"
        )

    # Format episode content
    episode_content = format_memories_for_prompt(memories, episode)

    # Build prompt
    style_instruction = SUMMARY_STYLES[style]
    prompt = f"""{format_section("EPISODE CONTENT", episode_content)}

{format_section("TASK", style_instruction)}

Generate your response now:"""

    # Call LLM
    response = llm.generate_complete(
        model=model,
        prompt=prompt,
        caller="episode_summarization",
    )

    return response


def generate_episode_title(
    episode: Episode,
    memories: list[MemoryElement],
    llm: LLM,
    model: SupportedModel,
) -> str:
    """Generate a short title for an episode."""
    episode_content = format_memories_for_prompt(memories, episode)

    prompt = f"""{format_section("EPISODE CONTENT", episode_content)}

{format_section("TASK", "Generate a short title (3-7 words) that captures the essence of this conversation episode. Output ONLY the title, nothing else.")}

Title:"""

    response = llm.generate_complete(
        model=model,
        prompt=prompt,
        caller="episode_title_generation",
    )

    # Clean up response
    title = response.strip()
    # Remove quotes if present
    if title.startswith('"') and title.endswith('"'):
        title = title[1:-1]
    if title.startswith("'") and title.endswith("'"):
        title = title[1:-1]

    return title


def generate_summaries_for_episodes(
    episodes: list[Episode],
    memories: list[MemoryElement],
    llm: LLM,
    model: SupportedModel,
    styles: list[str] | None = None,
) -> dict[str, dict[str, str]]:
    """
    Generate summaries for multiple episodes in multiple styles.

    Args:
        episodes: List of episodes to summarize
        memories: All memories
        llm: LLM router instance
        model: Model to use
        styles: List of styles to generate. Defaults to all styles.

    Returns:
        Dict mapping episode_id -> style -> summary
    """
    if styles is None:
        styles = list(SUMMARY_STYLES.keys())

    results: dict[str, dict[str, str]] = {}

    for episode in episodes:
        results[episode.id] = {}
        for style in styles:
            summary = generate_episode_summary(
                episode=episode,
                memories=memories,
                llm=llm,
                model=model,
                style=style,
            )
            results[episode.id][style] = summary

    return results


def generate_summary_at_detail_level(
    episode: Episode,
    memories: list[MemoryElement],
    llm: LLM,
    model: SupportedModel,
    detail_level: str,
) -> str:
    """
    Generate a summary at a specific detail level.

    Args:
        episode: Episode to summarize
        memories: All memories
        llm: LLM router
        model: Model to use
        detail_level: "short" (1-2 sentences), "medium" (1 paragraph), "detailed" (multiple paragraphs)

    Returns:
        Generated summary
    """
    detail_instructions = {
        "short": "Summarize in 1-2 sentences only. Be extremely concise.",
        "medium": "Summarize in one paragraph (3-5 sentences). Include key events and topics.",
        "detailed": "Write a detailed summary in multiple paragraphs. Include events, emotions, key quotes, and narrative flow.",
    }

    if detail_level not in detail_instructions:
        raise ValueError(f"Unknown detail level: {detail_level}")

    episode_content = format_memories_for_prompt(memories, episode)
    instruction = detail_instructions[detail_level]

    prompt = f"""{format_section("EPISODE CONTENT", episode_content)}

{format_section("TASK", f"Summarize this conversation episode.\\n\\n{instruction}")}

Summary:"""

    response = llm.generate_complete(
        model=model,
        prompt=prompt,
        caller=f"episode_summary_{detail_level}",
    )

    return response.strip()
