"""
Conversation boundary detection for tier 3 memories.

Detects natural boundaries in trigger-response sequences based on:
- Time gaps between interactions
- Topic shifts detected via LLM
- Explicit conversation endings
"""

import logging
import uuid
from datetime import datetime, timedelta
from typing import List, Tuple, Optional
from pydantic import BaseModel, Field

from agent.chain_of_action.trigger_history import TriggerHistoryEntry
from agent.embedding_service import get_embedding_service
from agent.llm import LLM, SupportedModel
from agent.structured_llm import direct_structured_llm_call

from .models import ConversationBoundary

logger = logging.getLogger(__name__)


# Configuration for boundary detection
TIME_GAP_THRESHOLD_MINUTES = 30  # Conversations separated by 30+ min are split
MIN_CONVERSATION_LENGTH = 2  # Minimum trigger entries for a conversation


class ConversationSegment(BaseModel):
    """Intermediate representation of a conversation segment."""

    start_index: int
    end_index: int  # Inclusive
    trigger_entries: List[TriggerHistoryEntry]
    reason: str = Field(description="Why this boundary was detected")


class TopicShiftAnalysis(BaseModel):
    """LLM analysis of potential topic shifts."""

    has_topic_shift: bool = Field(
        description="Whether a significant topic shift occurred"
    )
    shift_location: int = Field(
        default=-1, description="Index where the topic shift occurs (-1 if no shift)"
    )
    before_topic: str = Field(default="", description="Topic before the shift")
    after_topic: str = Field(default="", description="Topic after the shift")
    reasoning: str = Field(description="Explanation of the analysis")


class ConversationSummary(BaseModel):
    """LLM-generated summary of a conversation."""

    summary: str = Field(description="Concise summary of the conversation")
    topic_tags: List[str] = Field(
        description="Key topics discussed in the conversation"
    )


def detect_time_based_boundaries(
    trigger_entries: List[TriggerHistoryEntry],
) -> List[ConversationSegment]:
    """
    Detect conversation boundaries based on time gaps.

    Args:
        trigger_entries: Chronologically ordered trigger entries

    Returns:
        List of conversation segments based on time gaps
    """
    if not trigger_entries:
        return []

    segments = []
    current_segment_start = 0

    for i in range(1, len(trigger_entries)):
        prev_entry = trigger_entries[i - 1]
        current_entry = trigger_entries[i]

        # Calculate time gap
        time_gap = current_entry.timestamp - prev_entry.timestamp
        gap_minutes = time_gap.total_seconds() / 60

        # Check if gap exceeds threshold
        if gap_minutes >= TIME_GAP_THRESHOLD_MINUTES:
            # End current segment
            segments.append(
                ConversationSegment(
                    start_index=current_segment_start,
                    end_index=i - 1,
                    trigger_entries=trigger_entries[current_segment_start:i],
                    reason=f"Time gap of {gap_minutes:.1f} minutes",
                )
            )
            current_segment_start = i

    # Add final segment
    segments.append(
        ConversationSegment(
            start_index=current_segment_start,
            end_index=len(trigger_entries) - 1,
            trigger_entries=trigger_entries[current_segment_start:],
            reason="End of sequence",
        )
    )

    logger.info(
        f"Detected {len(segments)} time-based conversation segments "
        f"from {len(trigger_entries)} trigger entries"
    )

    return segments


def detect_topic_shifts_in_segment(
    segment: ConversationSegment,
    llm: LLM,
    model: SupportedModel,
) -> List[ConversationSegment]:
    """
    Analyze a segment for topic shifts and split if needed.

    Args:
        segment: Conversation segment to analyze
        llm: LLM instance for topic analysis
        model: Model to use

    Returns:
        List of segments (original if no shift, split if shift detected)
    """
    # Don't split very short segments
    if len(segment.trigger_entries) < 4:
        return [segment]

    # Build context for LLM analysis
    entries_text = []
    for i, entry in enumerate(segment.trigger_entries):
        timestamp = entry.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        summary = entry.compressed_summary or "No summary available"
        entries_text.append(f"[{i}] {timestamp}: {summary}")

    prompt = f"""Analyze this sequence of conversation exchanges to detect topic shifts.

Exchanges (chronologically ordered):
{chr(10).join(entries_text)}

Your task:
1. Determine if there is a SIGNIFICANT topic shift within this sequence
2. A significant shift means the conversation fundamentally changes subject
3. Minor tangents or related topics don't count - only major topic changes
4. If there is a shift, identify the index where it occurs

Consider:
- Are they discussing the same general subject throughout?
- Do later exchanges build on earlier ones, or start fresh?
- Is there a clear pivot point where the focus changes?"""

    try:
        analysis = direct_structured_llm_call(
            prompt=prompt,
            response_model=TopicShiftAnalysis,
            model=model,
            llm=llm,
            caller="topic_shift_detection",
        )

        if analysis.has_topic_shift and 0 < analysis.shift_location < len(
            segment.trigger_entries
        ):
            # Split the segment
            logger.info(
                f"Topic shift detected at index {analysis.shift_location}: "
                f"{analysis.before_topic} → {analysis.after_topic}"
            )

            return [
                ConversationSegment(
                    start_index=segment.start_index,
                    end_index=segment.start_index + analysis.shift_location - 1,
                    trigger_entries=segment.trigger_entries[: analysis.shift_location],
                    reason=f"Topic shift: {analysis.before_topic}",
                ),
                ConversationSegment(
                    start_index=segment.start_index + analysis.shift_location,
                    end_index=segment.end_index,
                    trigger_entries=segment.trigger_entries[analysis.shift_location :],
                    reason=f"Topic shift: {analysis.after_topic}",
                ),
            ]
        else:
            return [segment]

    except Exception as e:
        logger.warning(f"Topic shift detection failed: {e}")
        return [segment]


def generate_conversation_summary(
    segment: ConversationSegment,
    llm: LLM,
    model: SupportedModel,
    state,
) -> ConversationSummary:
    """
    Generate a compressed summary for a conversation segment.

    Args:
        segment: Conversation segment to summarize
        llm: LLM instance
        model: Model to use
        state: Agent state with name and role

    Returns:
        ConversationSummary with summary text and topic tags
    """
    from agent.state import build_agent_state_description
    from agent.chain_of_action.prompts import format_section

    # Build conversation context
    entries_text = []
    for entry in segment.trigger_entries:
        timestamp = entry.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        trigger_summary = entry.compressed_summary or "No summary"
        entries_text.append(f"[{timestamp}] {trigger_summary}")

    state_desc = build_agent_state_description(state)

    prompt = f"""I am {state.name}, {state.role}. I'm reviewing my conversation history to create a summary.

{state_desc}

{format_section(
    "CONVERSATION TO SUMMARIZE",
    "\n".join(entries_text)
)}

I need to create:
1. A 2-3 sentence summary written in FIRST PERSON from my perspective, capturing what we discussed and what happened
2. A list of 2-5 key topic tags for this conversation

The summary should be written as "I" or "we" - describing what I experienced, what was discussed, what I decided or accomplished. Do not use third-person language like "the individual" or "the agent"."""

    try:
        summary = direct_structured_llm_call(
            prompt=prompt,
            response_model=ConversationSummary,
            model=model,
            llm=llm,
            caller="conversation_summarization",
        )
        return summary

    except Exception as e:
        logger.warning(f"Conversation summarization failed: {e}")
        # Fallback summary
        return ConversationSummary(
            summary=f"Conversation from {segment.trigger_entries[0].timestamp} "
            f"to {segment.trigger_entries[-1].timestamp}",
            topic_tags=["general"],
        )


def detect_conversations(
    trigger_entries: List[TriggerHistoryEntry],
    llm: LLM,
    model: SupportedModel,
    state,
    use_topic_detection: bool = True,
    on_conversation_created=None,
    existing_conversations: Optional[List[ConversationBoundary]] = None,
) -> List[ConversationBoundary]:
    """
    Detect conversation boundaries in a sequence of trigger entries.

    Args:
        trigger_entries: Chronologically ordered trigger entries
        llm: LLM instance for topic analysis and summarization
        model: Model to use
        use_topic_detection: Whether to use LLM-based topic shift detection
        on_conversation_created: Callback called after each conversation is created
        existing_conversations: Previously created conversations (for resume)

    Returns:
        List of ConversationBoundary objects (tier 3)
    """
    logger.info(f"Detecting conversations from {len(trigger_entries)} trigger entries")

    # Find which trigger entries are already processed
    if existing_conversations:
        processed_entry_ids = set()
        for conv in existing_conversations:
            processed_entry_ids.update(conv.trigger_entry_ids)

        # Filter out already processed entries
        remaining_entries = [
            entry
            for entry in trigger_entries
            if entry.entry_id not in processed_entry_ids
        ]

        logger.info(
            f"Found {len(existing_conversations)} existing conversations "
            f"covering {len(processed_entry_ids)} entries. "
            f"Processing {len(remaining_entries)} remaining entries."
        )

        conversations = list(existing_conversations)
        trigger_entries = remaining_entries
    else:
        conversations = []

    if not trigger_entries:
        logger.info("No new trigger entries to process")
        return conversations

    # Step 1: Time-based segmentation
    time_segments = detect_time_based_boundaries(trigger_entries)

    # Step 2: Optional topic-based splitting with incremental conversation creation
    if use_topic_detection:
        logger.info(
            f"Analyzing {len(time_segments)} time-based segments for topic shifts..."
        )
        for i, segment in enumerate(time_segments, 1):
            logger.info(
                f"  Analyzing segment {i}/{len(time_segments)} ({len(segment.trigger_entries)} entries)"
            )
            topic_segments = detect_topic_shifts_in_segment(segment, llm, model)

            # Create conversations immediately for valid segments
            for topic_segment in topic_segments:
                if len(topic_segment.trigger_entries) >= MIN_CONVERSATION_LENGTH:
                    conversation = _create_conversation_from_segment(
                        topic_segment, llm, model, state
                    )
                    conversations.append(conversation)

                    # Call callback immediately after creating each conversation
                    if on_conversation_created:
                        on_conversation_created(conversation)
    else:
        # No topic detection - create conversations from time segments
        for segment in time_segments:
            if len(segment.trigger_entries) >= MIN_CONVERSATION_LENGTH:
                conversation = _create_conversation_from_segment(
                    segment, llm, model, state
                )
                conversations.append(conversation)

                # Call callback immediately
                if on_conversation_created:
                    on_conversation_created(conversation)

    logger.info(f"Created {len(conversations)} conversation boundaries (tier 3)")

    return conversations


def _create_conversation_from_segment(
    segment: ConversationSegment,
    llm: LLM,
    model: SupportedModel,
    state,
) -> ConversationBoundary:
    """Create a ConversationBoundary from a segment."""
    from agent.embedding_service import get_embedding_service

    summary_result = generate_conversation_summary(segment, llm, model, state)

    # Generate embedding for the summary
    embedding_service = get_embedding_service()
    embedding = embedding_service.encode(summary_result.summary)

    conversation = ConversationBoundary(
        id=str(uuid.uuid4()),
        trigger_entry_ids=[entry.entry_id for entry in segment.trigger_entries],
        start_timestamp=segment.trigger_entries[0].timestamp,
        end_timestamp=segment.trigger_entries[-1].end_timestamp
        or segment.trigger_entries[-1].timestamp,
        summary=summary_result.summary,
        embedding_vector=embedding,
        topic_tags=summary_result.topic_tags,
    )

    logger.info(
        f"Created conversation {conversation.id[:8]} with "
        f"{len(segment.trigger_entries)} entries, "
        f"duration: {conversation.duration_seconds:.0f}s, "
        f"topics: {', '.join(summary_result.topic_tags)}"
    )

    return conversation
