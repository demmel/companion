"""
Time expression parsing for temporal retrieval.

Uses LLM for natural language understanding of time references:
- Relative time ("yesterday", "last week")
- Absolute time ("January 15", "on Monday")
- Emotional/contextual time ("when I was stressed", "during the project")
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from agent.experiments.temporal_retrieval.models import TimeReference
from agent.llm.models import SupportedModel
from agent.llm.router import LLM
from agent.structured_llm import structured_llm_call


class ParsedTimeReference(BaseModel):
    """LLM output for time reference parsing."""

    ref_type: Literal["relative", "absolute", "emotional", "none"] = Field(
        description="Type of time reference: 'relative' (yesterday, last week), 'absolute' (January 15, Monday), 'emotional' (when I was stressed), or 'none' if no time reference found"
    )

    # For relative/absolute - these are offsets from reference time
    days_ago_start: int | None = Field(
        default=None,
        description="Start of time range as days before reference time (0 = today, 1 = yesterday, 7 = a week ago)"
    )
    days_ago_end: int | None = Field(
        default=None,
        description="End of time range as days before reference time (0 = today, 1 = yesterday)"
    )
    hour_start: int | None = Field(
        default=None,
        description="Start hour of day (0-23) if time of day is specified, e.g., 6 for morning, 12 for afternoon"
    )
    hour_end: int | None = Field(
        default=None,
        description="End hour of day (0-23) if time of day is specified, e.g., 12 for morning end, 18 for afternoon end"
    )

    # For emotional/contextual
    mood_filter: str | None = Field(
        default=None,
        description="Mood keyword if emotional reference (stressed, happy, tired, sad, anxious, etc.)"
    )
    topic_filter: str | None = Field(
        default=None,
        description="Topic keyword if contextual reference (project, meeting, work, etc.)"
    )


def parse_time_reference(
    text: str,
    now: datetime | None = None,
    llm: LLM | None = None,
    model: SupportedModel | None = None,
) -> TimeReference | None:
    """
    Parse a time reference from text using LLM.

    Args:
        text: Text containing a time reference
        now: Reference datetime (defaults to now)
        llm: LLM instance (required for LLM parsing)
        model: Model to use (required for LLM parsing)

    Returns:
        TimeReference or None if no time reference found
    """
    if now is None:
        now = datetime.now()

    # If no LLM provided, fall back to simple heuristics
    if llm is None or model is None:
        return _parse_time_reference_heuristic(text, now)

    return _parse_time_reference_llm(text, now, llm, model)


def _parse_time_reference_llm(
    text: str,
    now: datetime,
    llm: LLM,
    model: SupportedModel,
) -> TimeReference | None:
    """Parse time reference using LLM."""

    system_prompt = f"""Parse the time reference in the user's query.

Current reference time: {now.strftime('%A, %B %d, %Y at %H:%M')}
Current day of week: {now.strftime('%A')}

Determine:
1. ref_type: Is this relative (yesterday, last week), absolute (January 15, Monday), emotional (when I was stressed), or none?
2. For relative/absolute: Calculate days_ago_start and days_ago_end from the reference time
3. For time-of-day references: Set hour_start and hour_end (morning=6-12, afternoon=12-18, evening=18-22)
4. For emotional: Extract the mood or topic keyword

Examples:
- "yesterday" -> ref_type="relative", days_ago_start=1, days_ago_end=1
- "last week" -> ref_type="relative", days_ago_start=7, days_ago_end=1
- "this morning" -> ref_type="relative", days_ago_start=0, days_ago_end=0, hour_start=6, hour_end=12
- "on Monday" -> ref_type="absolute", calculate days to most recent Monday
- "when I was stressed" -> ref_type="emotional", mood_filter="stressed"
- "during the project" -> ref_type="emotional", topic_filter="project"
- "how are you" -> ref_type="none" (no time reference)"""

    try:
        result = structured_llm_call(
            system_prompt=system_prompt,
            user_input=text,
            response_model=ParsedTimeReference,
            model=model,
            llm=llm,
            caller="time_parser",
            temperature=0.0,
        )

        if result.ref_type == "none":
            return None

        # Convert to TimeReference
        start_time = None
        end_time = None

        if result.days_ago_start is not None:
            start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
            start_time = start_date.replace(
                day=start_date.day - result.days_ago_start
            ) if result.days_ago_start <= start_date.day else start_date

            # Handle month boundaries properly
            from datetime import timedelta
            start_time = now.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=result.days_ago_start)

            if result.hour_start is not None:
                start_time = start_time.replace(hour=result.hour_start)

        if result.days_ago_end is not None:
            from datetime import timedelta
            end_time = now.replace(hour=23, minute=59, second=59, microsecond=999999) - timedelta(days=result.days_ago_end)

            if result.hour_end is not None:
                end_time = end_time.replace(hour=result.hour_end, minute=59, second=59)

        return TimeReference(
            raw_text=text,
            ref_type="emotional" if result.ref_type == "emotional" else result.ref_type,
            start_time=start_time,
            end_time=end_time,
            mood_filter=result.mood_filter,
            topic_filter=result.topic_filter,
        )

    except Exception:
        # Fall back to heuristic parsing on LLM failure
        return _parse_time_reference_heuristic(text, now)


def _parse_time_reference_heuristic(text: str, now: datetime) -> TimeReference | None:
    """Simple heuristic fallback for time parsing without LLM."""
    from datetime import timedelta

    text_lower = text.lower()

    # Relative time patterns
    if "yesterday" in text_lower:
        start = now.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
        end = start.replace(hour=23, minute=59, second=59)
        return TimeReference(raw_text=text, ref_type="relative", start_time=start, end_time=end)

    if "today" in text_lower:
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return TimeReference(raw_text=text, ref_type="relative", start_time=start, end_time=now)

    if "last week" in text_lower:
        start = now - timedelta(days=7)
        return TimeReference(raw_text=text, ref_type="relative", start_time=start, end_time=now)

    if "this morning" in text_lower:
        start = now.replace(hour=6, minute=0, second=0, microsecond=0)
        end = now.replace(hour=12, minute=0, second=0, microsecond=0)
        return TimeReference(raw_text=text, ref_type="relative", start_time=start, end_time=end)

    # Emotional patterns
    emotional_keywords = ["stressed", "anxious", "worried", "happy", "excited", "sad", "tired", "angry"]
    for mood in emotional_keywords:
        if mood in text_lower:
            return TimeReference(raw_text=text, ref_type="emotional", mood_filter=mood)

    # Topic patterns
    if "during" in text_lower or "while" in text_lower:
        # Extract topic after "during" or "while"
        for word in ["project", "meeting", "work", "discussion"]:
            if word in text_lower:
                return TimeReference(raw_text=text, ref_type="emotional", topic_filter=word)

    return None


def test_time_parser() -> list[dict[str, str | datetime | None]]:
    """Test the time parser with various examples."""
    now = datetime(2024, 1, 19, 14, 30, 0)  # Friday afternoon

    test_cases = [
        "What happened this morning?",
        "What did we talk about yesterday?",
        "Remember last week?",
        "What have we been discussing recently?",
        "What happened on Tuesday?",
        "What about last Monday?",
        "January 15th",
        "Back in December",
        "When I was stressed about work",
        "During the job search",
        "That rough period",
        "When I felt happy",
    ]

    results = []
    for text in test_cases:
        ref = _parse_time_reference_heuristic(text, now)
        result: dict[str, str | datetime | None] = {
            "input": text,
            "ref_type": ref.ref_type if ref else None,
            "start_time": ref.start_time if ref else None,
            "end_time": ref.end_time if ref else None,
            "mood_filter": ref.mood_filter if ref else None,
            "topic_filter": ref.topic_filter if ref else None,
        }
        results.append(result)

    return results
