"""
LLM-based memory query extraction for retrieving relevant memories.
"""

from datetime import datetime, timedelta
import logging
import re
import time
from typing import List, Optional
from agent.memory.memory_query import LLMMemoryExtraction
from agent.state import State
from agent.chain_of_action.trigger import BaseTrigger
from agent.chain_of_action.trigger_history import TriggerHistory, TriggerHistoryEntry
from agent.chain_of_action.prompts import build_memory_extraction_prompt
from agent.structured_llm import direct_structured_llm_call
from agent.llm import LLM, SupportedModel

logger = logging.getLogger(__name__)


def extract_memory_queries(
    state: State,
    trigger: BaseTrigger,
    trigger_history: TriggerHistory,
    llm: LLM,
    model: SupportedModel,
) -> Optional[LLMMemoryExtraction]:
    """
    Extract memory queries from current context using structured LLM call.

    Args:
        state: Current agent state
        trigger: Current trigger being processed
        trigger_history: Agent's trigger history
        llm: LLM instance for making the call
        model: Model to use for extraction

    Returns:
        LLMMemoryExtraction with keywords and time query, or None if extraction fails
    """
    start_time = time.time()
    prompt = build_memory_extraction_prompt(state, trigger, trigger_history)

    try:
        response = direct_structured_llm_call(
            prompt=prompt,
            response_model=LLMMemoryExtraction,
            model=model,
            llm=llm,
            caller="memory_extraction",
            temperature=0.3,
        )
        elapsed = time.time() - start_time
        logger.info(
            f"Memory extraction completed in {elapsed:.3f}s - query: {response.conceptual_query if response else None}"
        )
        return response
    except Exception as e:
        elapsed = time.time() - start_time
        logger.warning(f"Memory extraction failed in {elapsed:.3f}s: {e}")
        return None


def retrieve_relevant_memories(
    memory_query: LLMMemoryExtraction,
    trigger_history: TriggerHistory,
    max_results: int = 5,
) -> List[TriggerHistoryEntry]:
    """
    Retrieve relevant memories based on extracted query.

    Args:
        memory_query: Extracted memory query with keywords and time constraints
        trigger_history: Agent's trigger history to search
        max_results: Maximum number of memories to return

    Returns:
        List of relevant trigger history entries
    """
    from datetime import datetime

    if not memory_query:
        return []

    start_time = time.time()
    total_entries = len(trigger_history.get_all_entries())

    # Parse time boundaries for filtering
    start_boundary = None
    end_boundary = None
    now = datetime.now()

    if memory_query.time_query and (
        memory_query.time_query.start_time or memory_query.time_query.end_time
    ):
        if memory_query.time_query.start_time:
            start_boundary = _parse_time_reference(
                memory_query.time_query.start_time, now
            )

        if memory_query.time_query.end_time:
            end_boundary = _parse_time_reference(memory_query.time_query.end_time, now)

    # Use similarity-based retrieval if conceptual query provided
    if memory_query.conceptual_query:
        from agent.memory.similarity_retrieval import retrieve_memories_by_similarity

        relevant_memories = retrieve_memories_by_similarity(
            query_text=memory_query.conceptual_query,
            trigger_history=trigger_history,
            max_results=max_results,
            time_filter_start=start_boundary,
            time_filter_end=end_boundary,
        )

        elapsed = time.time() - start_time
        logger.info(
            f"Memory retrieval completed in {elapsed:.3f}s - found {len(relevant_memories)} similar memories"
        )
        return relevant_memories
    else:
        # No conceptual query, just apply time filtering and return recent results
        candidate_memories = []
        for memory in trigger_history.get_all_entries():
            memory_time = memory.timestamp

            # Check if memory falls within time range
            within_range = True
            if start_boundary and memory_time < start_boundary:
                within_range = False
            if end_boundary and memory_time > end_boundary:
                within_range = False

            if within_range:
                candidate_memories.append(memory)

        results = candidate_memories[:max_results]
        elapsed = time.time() - start_time
        logger.info(
            f"Memory retrieval completed in {elapsed:.3f}s - searched {total_entries} entries, time-filtered to {len(candidate_memories)}, returned {len(results)}"
        )
        return results


def _parse_time_reference(time_str: str, now: datetime) -> datetime:
    """Parse time reference string to datetime"""
    if time_str == "now":
        return now

    # Try ISO format first
    try:
        return datetime.fromisoformat(time_str.replace("Z", "+00:00"))
    except ValueError:
        pass

    # Parse relative format like -3d, -2h, -1w
    relative_pattern = r"^-(\d+)([hdwmy])$"
    match = re.match(relative_pattern, time_str)
    if match:
        amount = int(match.group(1))
        unit = match.group(2)

        if unit == "h":
            return now - timedelta(hours=amount)
        elif unit == "d":
            return now - timedelta(days=amount)
        elif unit == "w":
            return now - timedelta(weeks=amount)
        elif unit == "m":
            return now - timedelta(days=amount * 30)  # Approximate
        elif unit == "y":
            return now - timedelta(days=amount * 365)  # Approximate

    # Fallback to now if parsing fails
    logger.warning(f"Could not parse time reference: {time_str}")
    return now
