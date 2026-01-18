"""Text processing utilities for TTS generation."""

import re


def normalize_text_for_tts(text: str) -> str:
    """Normalize text for TTS - replace special characters that cause issues.

    Args:
        text: The text to normalize.

    Returns:
        Normalized text safe for TTS processing.
    """
    replacements = {
        # Curly quotes to straight
        "\u2018": "'",  # '
        "\u2019": "'",  # '
        "\u201c": '"',  # "
        "\u201d": '"',  # "
        # Dashes
        "\u2014": ", ",  # em-dash
        "\u2013": "-",  # en-dash
        # Ellipsis
        "\u2026": "...",
        # Other problematic chars
        "\u00a0": " ",  # non-breaking space
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def split_into_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs for chunked TTS generation.

    Args:
        text: The text to split.

    Returns:
        List of paragraph strings.
    """
    # Split on double newlines or single newlines
    parts = re.split(r"\n\n+|\n", text)
    # Filter out empty strings and strip whitespace
    return [p.strip() for p in parts if p.strip()]
