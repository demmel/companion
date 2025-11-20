"""
Fact extraction from unstructured text using LLM.

Uses Mistral 3.2 Q4 (local model) to extract structured facts.
"""

import logging
from typing import List, Optional
from pydantic import BaseModel, Field

from agent.llm.router import LLM
from agent.llm.models import SupportedModel
from agent.structured_llm import structured_llm_call

from .interfaces import IFactExtractor, Fact
from .knowledge_graph import create_fact
from .config import ExtractionConfig

logger = logging.getLogger(__name__)

# Use Mistral 3.2 Q4 to avoid API costs
EXTRACTION_MODEL = SupportedModel.MISTRAL_SMALL_3_2_Q4


class ExtractedFact(BaseModel):
    """Schema for a single extracted fact"""

    predicate: str = Field(
        description="The relationship type (e.g., 'traded_with', 'ruled_by')"
    )
    entities: dict[str, str] = Field(
        description="Entity roles and IDs (e.g., {'trader': 'Byzantine Empire', 'partner': 'Venice'})"
    )
    time_period: Optional[str] = Field(
        default=None, description="Time period when this occurred"
    )
    region: Optional[str] = Field(default=None, description="Geographic region")
    confidence: Optional[str] = Field(
        default=None, description="Confidence level: high, medium, or low"
    )


class ExtractionResponse(BaseModel):
    """Schema for fact extraction response"""

    facts: List[ExtractedFact] = Field(description="List of extracted facts")


class LLMFactExtractor(IFactExtractor):
    """
    Extract structured facts from text using LLM.

    Simple implementation: single LLM call with structured output.
    If extraction quality is poor, we can iterate on the prompt or try multi-pass extraction.
    """

    def __init__(self, llm: LLM, config: ExtractionConfig):
        self.llm = llm
        self.config = config

    def extract_facts(self, text: str, context: Optional[str] = None) -> List[Fact]:
        """
        Extract facts from text.

        Prompts LLM to identify n-ary relationships and extract them as structured facts.
        """
        system_prompt = """Extract structured n-ary facts from the text.

A fact is a relationship involving multiple entities with specific roles.
Only extract facts explicitly stated in the text - be specific and precise."""

        user_input = text
        if context:
            user_input = f"{context}\n\n{text}"

        try:
            # Use structured LLM call with Pydantic schema
            response = structured_llm_call(
                system_prompt=system_prompt,
                user_input=user_input,
                response_model=ExtractionResponse,
                model=EXTRACTION_MODEL,
                llm=self.llm,
                caller="fact_extraction",
                temperature=self.config.extraction_temperature,
            )

            # Convert ExtractedFact models to Fact objects
            facts = []
            for extracted in response.facts:
                fact = create_fact(
                    predicate=extracted.predicate,
                    entities=extracted.entities,
                    time_period=extracted.time_period,
                    region=extracted.region,
                    confidence=extracted.confidence,
                )
                facts.append(fact)

            logger.info(f"Extracted {len(facts)} facts from text ({len(text)} chars)")
            return facts

        except Exception as e:
            logger.error(f"Error during fact extraction: {e}")
            return []


class ChunkedFactExtractor(IFactExtractor):
    """
    Extracts facts by chunking large text into smaller pieces.

    If we hit token limit issues with large articles, this chunks the text
    and extracts from each chunk separately.
    """

    def __init__(self, llm: LLM, config: ExtractionConfig):
        self.base_extractor = LLMFactExtractor(llm, config)
        self.config = config

    def extract_facts(self, text: str, context: Optional[str] = None) -> List[Fact]:
        """Extract facts from text, chunking if necessary"""
        # If text is short enough, use base extractor
        if len(text) <= self.config.chunk_size:
            return self.base_extractor.extract_facts(text, context)

        # Otherwise, chunk and extract from each
        chunks = self._chunk_text(text)
        logger.info(f"Chunking text into {len(chunks)} chunks for extraction")

        all_facts = []
        for i, chunk in enumerate(chunks):
            chunk_context = (
                f"{context} (chunk {i+1}/{len(chunks)})"
                if context
                else f"(chunk {i+1}/{len(chunks)})"
            )
            facts = self.base_extractor.extract_facts(chunk, chunk_context)
            all_facts.extend(facts)

        return all_facts

    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.

        Uses ~20% overlap (600 chars for 3000 char chunks) to ensure facts
        near boundaries are captured in at least one full chunk.
        """
        chunks = []
        overlap_size = int(self.config.chunk_size * 0.2)  # 20% overlap
        words = text.split()

        current_chunk = []
        current_length = 0
        overlap_words = []  # Words from end of previous chunk

        for word in words:
            current_chunk.append(word)
            current_length += len(word) + 1  # +1 for space

            if current_length >= self.config.chunk_size:
                # Save this chunk
                chunk_text = " ".join(current_chunk)
                chunks.append(chunk_text)

                # Keep last ~overlap_size chars worth of words for next chunk
                overlap_words = []
                overlap_length = 0
                for w in reversed(current_chunk):
                    overlap_length += len(w) + 1
                    overlap_words.insert(0, w)
                    if overlap_length >= overlap_size:
                        break

                # Start next chunk with overlap
                current_chunk = overlap_words.copy()
                current_length = overlap_length

        # Add remaining words as final chunk (if substantial)
        if current_chunk and current_length > overlap_size:
            chunks.append(" ".join(current_chunk))

        return chunks
