"""Core extraction logic for memory extraction experiment."""

import logging
import uuid

from agent.llm import LLM, SupportedModel
from agent.structured_llm import direct_structured_llm_call

from .models import (
    ExtractedFact,
    ExtractionResult,
    FactType,
    FactListExtractionResponse,
    StructuredExtractionResponse,
    QueryFocusedExtractionResponse,
    EntityCentricExtractionResponse,
    MinimalExtractionResponse,
    MemorySample,
)
from .prompts import (
    FACT_LIST_PROMPT,
    STRUCTURED_PROMPT,
    QUERY_FOCUSED_PROMPT,
    ENTITY_CENTRIC_PROMPT,
    MINIMAL_PROMPT,
    HIGH_COMPRESSION_PROMPT,
    LOW_COMPRESSION_PROMPT,
)

logger = logging.getLogger(__name__)


def _parse_fact_type(type_str: str) -> FactType:
    """Parse a fact type string to enum, with fallback."""
    type_str = type_str.lower().strip()
    try:
        return FactType(type_str)
    except ValueError:
        # Try to match common variations
        if "prefer" in type_str:
            return FactType.PREFERENCE
        if "event" in type_str:
            return FactType.EVENT
        if "relation" in type_str:
            return FactType.RELATIONSHIP
        if "state" in type_str:
            return FactType.STATE
        if "appear" in type_str:
            return FactType.APPEARANCE
        if "environ" in type_str:
            return FactType.ENVIRONMENT
        if "emotion" in type_str or "feel" in type_str:
            return FactType.EMOTION
        if "value" in type_str:
            return FactType.VALUE
        if "question" in type_str:
            return FactType.QUESTION
        return FactType.FACT


def extract_facts_approach_a(
    content: str,
    memory_id: str,
    llm: LLM,
    model: SupportedModel,
) -> ExtractionResult:
    """Approach A: Fact list extraction."""
    prompt = FACT_LIST_PROMPT.format(content=content)

    response = direct_structured_llm_call(
        prompt=prompt,
        response_model=FactListExtractionResponse,
        model=model,
        llm=llm,
        caller="memory_extraction_a",
    )

    facts = [
        ExtractedFact(
            content=f.content,
            fact_type=_parse_fact_type(f.fact_type),
            confidence=f.confidence,
            entities=f.entities,
            source_memory_id=memory_id,
        )
        for f in response.facts
    ]

    extracted_len = sum(len(f.content) for f in facts)
    compression_ratio = extracted_len / len(content) if content else 0.0

    return ExtractionResult(
        memory_id=memory_id,
        original_content=content,
        facts=facts,
        summary=response.summary,
        compression_ratio=compression_ratio,
    )


def extract_facts_approach_b(
    content: str,
    memory_id: str,
    llm: LLM,
    model: SupportedModel,
) -> ExtractionResult:
    """Approach B: Structured extraction by category."""
    prompt = STRUCTURED_PROMPT.format(content=content)

    response = direct_structured_llm_call(
        prompt=prompt,
        response_model=StructuredExtractionResponse,
        model=model,
        llm=llm,
        caller="memory_extraction_b",
    )

    facts: list[ExtractedFact] = []

    # Convert structured response to facts
    for fact_content in response.people:
        facts.append(
            ExtractedFact(
                content=fact_content,
                fact_type=FactType.RELATIONSHIP,
                confidence=0.8,
                entities=[],
                source_memory_id=memory_id,
            )
        )

    for fact_content in response.events:
        facts.append(
            ExtractedFact(
                content=fact_content,
                fact_type=FactType.EVENT,
                confidence=0.8,
                entities=[],
                source_memory_id=memory_id,
            )
        )

    for fact_content in response.preferences:
        facts.append(
            ExtractedFact(
                content=fact_content,
                fact_type=FactType.PREFERENCE,
                confidence=0.8,
                entities=[],
                source_memory_id=memory_id,
            )
        )

    for fact_content in response.emotions:
        facts.append(
            ExtractedFact(
                content=fact_content,
                fact_type=FactType.EMOTION,
                confidence=0.8,
                entities=[],
                source_memory_id=memory_id,
            )
        )

    for fact_content in response.questions:
        facts.append(
            ExtractedFact(
                content=fact_content,
                fact_type=FactType.QUESTION,
                confidence=0.8,
                entities=[],
                source_memory_id=memory_id,
            )
        )

    extracted_len = sum(len(f.content) for f in facts)
    compression_ratio = extracted_len / len(content) if content else 0.0

    return ExtractionResult(
        memory_id=memory_id,
        original_content=content,
        facts=facts,
        summary=response.summary,
        compression_ratio=compression_ratio,
    )


def extract_facts_approach_c(
    content: str,
    memory_id: str,
    llm: LLM,
    model: SupportedModel,
) -> ExtractionResult:
    """Approach C: Query-focused extraction."""
    prompt = QUERY_FOCUSED_PROMPT.format(content=content)

    response = direct_structured_llm_call(
        prompt=prompt,
        response_model=QueryFocusedExtractionResponse,
        model=model,
        llm=llm,
        caller="memory_extraction_c",
    )

    facts: list[ExtractedFact] = []

    for qa in response.qa_pairs:
        # Combine Q&A into a searchable fact
        fact_content = f"{qa.question} → {qa.answer}"
        facts.append(
            ExtractedFact(
                content=fact_content,
                fact_type=FactType.FACT,
                confidence=0.8,
                entities=[],
                source_memory_id=memory_id,
            )
        )

    extracted_len = sum(len(f.content) for f in facts)
    compression_ratio = extracted_len / len(content) if content else 0.0

    return ExtractionResult(
        memory_id=memory_id,
        original_content=content,
        facts=facts,
        summary=response.summary,
        compression_ratio=compression_ratio,
    )


def extract_facts_approach_d(
    content: str,
    memory_id: str,
    llm: LLM,
    model: SupportedModel,
) -> ExtractionResult:
    """Approach D: Entity-centric extraction."""
    prompt = ENTITY_CENTRIC_PROMPT.format(content=content)

    response = direct_structured_llm_call(
        prompt=prompt,
        response_model=EntityCentricExtractionResponse,
        model=model,
        llm=llm,
        caller="memory_extraction_d",
    )

    facts: list[ExtractedFact] = []

    for entity in response.entities:
        entity_name = entity.entity_name

        for fact_content in entity.learned_facts:
            facts.append(
                ExtractedFact(
                    content=f"{entity_name}: {fact_content}",
                    fact_type=FactType.FACT,
                    confidence=0.8,
                    entities=[entity_name],
                    source_memory_id=memory_id,
                )
            )

        for action in entity.actions:
            facts.append(
                ExtractedFact(
                    content=f"{entity_name}: {action}",
                    fact_type=FactType.EVENT,
                    confidence=0.8,
                    entities=[entity_name],
                    source_memory_id=memory_id,
                )
            )

        for relationship in entity.relationships:
            facts.append(
                ExtractedFact(
                    content=f"{entity_name}: {relationship}",
                    fact_type=FactType.RELATIONSHIP,
                    confidence=0.8,
                    entities=[entity_name],
                    source_memory_id=memory_id,
                )
            )

    extracted_len = sum(len(f.content) for f in facts)
    compression_ratio = extracted_len / len(content) if content else 0.0

    return ExtractionResult(
        memory_id=memory_id,
        original_content=content,
        facts=facts,
        summary=response.summary,
        compression_ratio=compression_ratio,
    )


def extract_facts_approach_e(
    content: str,
    memory_id: str,
    llm: LLM,
    model: SupportedModel,
) -> ExtractionResult:
    """Approach E: Minimal extraction (single most important fact)."""
    prompt = MINIMAL_PROMPT.format(content=content)

    response = direct_structured_llm_call(
        prompt=prompt,
        response_model=MinimalExtractionResponse,
        model=model,
        llm=llm,
        caller="memory_extraction_e",
    )

    facts = [
        ExtractedFact(
            content=response.most_important_fact,
            fact_type=_parse_fact_type(response.fact_type),
            confidence=1.0,
            entities=[],
            source_memory_id=memory_id,
        )
    ]

    extracted_len = len(response.most_important_fact)
    compression_ratio = extracted_len / len(content) if content else 0.0

    return ExtractionResult(
        memory_id=memory_id,
        original_content=content,
        facts=facts,
        summary=response.most_important_fact,
        compression_ratio=compression_ratio,
    )


def extract_facts(
    content: str,
    approach: str,
    llm: LLM,
    model: SupportedModel,
    memory_id: str | None = None,
) -> ExtractionResult:
    """
    Extract facts from memory content using specified approach.

    Args:
        content: The memory content to extract from
        approach: One of "A", "B", "C", "D", "E"
        llm: LLM instance
        model: Model to use for extraction
        memory_id: Optional ID for the memory

    Returns:
        ExtractionResult with extracted facts
    """
    if memory_id is None:
        memory_id = str(uuid.uuid4())

    approach = approach.upper()

    if approach == "A":
        return extract_facts_approach_a(content, memory_id, llm, model)
    elif approach == "B":
        return extract_facts_approach_b(content, memory_id, llm, model)
    elif approach == "C":
        return extract_facts_approach_c(content, memory_id, llm, model)
    elif approach == "D":
        return extract_facts_approach_d(content, memory_id, llm, model)
    elif approach == "E":
        return extract_facts_approach_e(content, memory_id, llm, model)
    else:
        raise ValueError(f"Unknown approach: {approach}. Must be A, B, C, D, or E.")


def extract_batch(
    memories: list[MemorySample],
    approach: str,
    llm: LLM,
    model: SupportedModel,
) -> list[ExtractionResult]:
    """
    Extract facts from multiple memories.

    Args:
        memories: List of memory samples to extract from
        approach: Extraction approach to use
        llm: LLM instance
        model: Model to use

    Returns:
        List of ExtractionResult for each memory
    """
    results = []
    for memory in memories:
        try:
            result = extract_facts(
                content=memory.content,
                approach=approach,
                llm=llm,
                model=model,
                memory_id=memory.memory_id,
            )
            results.append(result)
            logger.info(
                f"Extracted {len(result.facts)} facts from memory {memory.memory_id}"
            )
        except Exception as e:
            logger.error(f"Failed to extract from memory {memory.memory_id}: {e}")

    return results
