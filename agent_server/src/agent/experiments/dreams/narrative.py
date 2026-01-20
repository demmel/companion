"""Narrative generation styles for dream creation."""

from agent.llm import LLM, SupportedModel, Message
from agent.memory.dag.models import MemoryElement


def _format_memories_for_prompt(memories: list[MemoryElement]) -> str:
    """Format memories as input for narrative generation."""
    lines = []
    for i, mem in enumerate(memories, 1):
        lines.append(f"{i}. {mem.content}")
    return "\n".join(lines)


def generate_fragment_narrative(
    memories: list[MemoryElement], llm: LLM, model: SupportedModel
) -> str:
    """
    Generate dream narrative in fragment style.

    Ellipsis-heavy, incomplete thoughts, drifting between moments.

    Args:
        memories: List of memories to weave into narrative
        llm: LLM instance for generation
        model: Model to use

    Returns:
        Dream narrative text
    """
    memory_text = _format_memories_for_prompt(memories)

    prompt = f"""You are generating a dream sequence. Dreams are fragmentary, associative, non-linear.

Here are the memories to weave into a dream:
{memory_text}

Generate a dream narrative in FRAGMENT style:
- Use ellipses (...) frequently to suggest trailing thoughts
- Leave sentences incomplete
- Jump between sensations, images, feelings
- Use indentation to show drift/transition
- Focus on impressions, not full descriptions
- Let images dissolve into each other

Example of fragment style:
...the coffee was warm in his hands...
    dissolving into the office, fluorescent lights humming...
        becoming the sound of rain against the window...
            and somewhere, a feeling of being understood...

Now generate a dream from these memories. Only output the dream narrative, nothing else."""

    messages = [Message(role="user", content=prompt)]
    result = llm.chat_complete(model, messages, caller="dream_narrative_fragment")
    return result or ""


def generate_stream_narrative(
    memories: list[MemoryElement], llm: LLM, model: SupportedModel
) -> str:
    """
    Generate dream narrative in stream of consciousness style.

    Run-on sentences, no punctuation barriers, continuous flow.

    Args:
        memories: List of memories to weave into narrative
        llm: LLM instance for generation
        model: Model to use

    Returns:
        Dream narrative text
    """
    memory_text = _format_memories_for_prompt(memories)

    prompt = f"""You are generating a dream sequence. Dreams flow without logical breaks.

Here are the memories to weave into a dream:
{memory_text}

Generate a dream narrative in STREAM OF CONSCIOUSNESS style:
- Use minimal punctuation
- Let thoughts run into each other
- Words should flow without stopping
- Repeat key words as bridges between thoughts
- Create a continuous river of consciousness
- No paragraph breaks - one flowing stream

Example of stream style:
and then the coffee and the light and the way he said my name becoming something else entirely becoming the feeling of warmth and understanding flowing into the memory of rain on windows becoming tears becoming laughter becoming

Now generate a dream from these memories. Only output the dream narrative, nothing else."""

    messages = [Message(role="user", content=prompt)]
    result = llm.chat_complete(model, messages, caller="dream_narrative_stream")
    return result or ""


def generate_poetic_narrative(
    memories: list[MemoryElement], llm: LLM, model: SupportedModel
) -> str:
    """
    Generate dream narrative in poetic style.

    Line breaks, metaphorical language, rhythmic structure.

    Args:
        memories: List of memories to weave into narrative
        llm: LLM instance for generation
        model: Model to use

    Returns:
        Dream narrative text
    """
    memory_text = _format_memories_for_prompt(memories)

    prompt = f"""You are generating a dream sequence. Dreams have a poetic, metaphorical quality.

Here are the memories to weave into a dream:
{memory_text}

Generate a dream narrative in POETIC style:
- Use line breaks for rhythm and pause
- Use metaphor and imagery
- Create visual, evocative language
- Let images transform into other images
- Use repetition for effect
- Be sparse with words - each one counts

Example of poetic style:
In the space between waking
    memories fold like paper cranes
        each one a window, slightly open

The coffee grows cold
    while time stretches thin
        as morning light through curtains

Now generate a dream from these memories. Only output the dream narrative, nothing else."""

    messages = [Message(role="user", content=prompt)]
    result = llm.chat_complete(model, messages, caller="dream_narrative_poetic")
    return result or ""


def generate_sensory_narrative(
    memories: list[MemoryElement], llm: LLM, model: SupportedModel
) -> str:
    """
    Generate dream narrative in sensory style.

    Focus on senses: warmth, smell, texture, sound, taste.

    Args:
        memories: List of memories to weave into narrative
        llm: LLM instance for generation
        model: Model to use

    Returns:
        Dream narrative text
    """
    memory_text = _format_memories_for_prompt(memories)

    prompt = f"""You are generating a dream sequence. Dreams are experienced through the senses.

Here are the memories to weave into a dream:
{memory_text}

Generate a dream narrative in SENSORY style:
- Focus on physical sensations
- Describe warmth, cold, textures
- Include smells, sounds, tastes
- Let sensations blend and transform
- Use body-centered language
- Make the dream feel embodied

Example of sensory style:
warmth of the cup against palms
    smell of rain on pavement rising
        texture of his voice, rough like gravel

the light changes - now golden, now cold
    taste of something sweet fading
        pressure of being held, then released

Now generate a dream from these memories. Only output the dream narrative, nothing else."""

    messages = [Message(role="user", content=prompt)]
    result = llm.chat_complete(model, messages, caller="dream_narrative_sensory")
    return result or ""


def generate_narrative(
    memories: list[MemoryElement], style: str, llm: LLM, model: SupportedModel
) -> str:
    """
    Generate dream narrative using the specified style.

    Args:
        memories: List of memories to weave into narrative
        style: One of 'fragment', 'stream', 'poetic', 'sensory'
        llm: LLM instance for generation
        model: Model to use

    Returns:
        Dream narrative text
    """
    if style == "fragment":
        return generate_fragment_narrative(memories, llm, model)
    elif style == "stream":
        return generate_stream_narrative(memories, llm, model)
    elif style == "poetic":
        return generate_poetic_narrative(memories, llm, model)
    elif style == "sensory":
        return generate_sensory_narrative(memories, llm, model)
    else:
        raise ValueError(f"Unknown narrative style: {style}")


def extract_themes(
    memories: list[MemoryElement], llm: LLM, model: SupportedModel
) -> list[str]:
    """
    Extract themes that emerged from the dream memories.

    Args:
        memories: List of memories from the dream
        llm: LLM instance for generation
        model: Model to use

    Returns:
        List of theme strings
    """
    memory_text = _format_memories_for_prompt(memories)

    prompt = f"""Analyze these memories and identify the main themes that emerge.

Memories:
{memory_text}

List 3-5 themes that connect these memories. Output only the themes, one per line, no numbering or explanation."""

    messages = [Message(role="user", content=prompt)]
    result = llm.chat_complete(model, messages, caller="dream_themes")

    if not result:
        return []

    # Parse themes from response
    themes = [line.strip() for line in result.strip().split("\n") if line.strip()]
    return themes[:5]  # Limit to 5 themes
