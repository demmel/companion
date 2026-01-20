"""Evaluation and annotation helpers for memory extraction experiment."""

import logging

from agent.llm import LLM, SupportedModel
from agent.structured_llm import direct_structured_llm_call

from .models import (
    AnnotationLabel,
    AnnotationResponse,
    AnnotationResult,
    ExtractedFact,
    ExtractionResult,
    FactAnnotation,
    OmissionsResponse,
)
from .prompts import ANNOTATION_PROMPT, OMISSIONS_PROMPT

logger = logging.getLogger(__name__)


def _parse_label(label_str: str) -> AnnotationLabel:
    """Parse label string to enum."""
    label_str = label_str.lower().strip()
    if "correct" in label_str:
        return AnnotationLabel.CORRECT
    elif "hallucin" in label_str:
        return AnnotationLabel.HALLUCINATED
    elif "infer" in label_str:
        return AnnotationLabel.INFERRED
    else:
        # Default to inferred if unclear
        return AnnotationLabel.INFERRED


def annotate_fact(
    original: str,
    fact: ExtractedFact,
    llm: LLM,
    model: SupportedModel,
) -> FactAnnotation:
    """
    Annotate a single extracted fact.

    Args:
        original: The original memory content
        fact: The extracted fact to annotate
        llm: LLM instance
        model: Model to use

    Returns:
        FactAnnotation with label and notes
    """
    prompt = ANNOTATION_PROMPT.format(original=original, fact=fact.content)

    response = direct_structured_llm_call(
        prompt=prompt,
        response_model=AnnotationResponse,
        model=model,
        llm=llm,
        caller="fact_annotation",
    )

    return FactAnnotation(
        fact=fact,
        label=_parse_label(response.label),
        notes=response.reasoning,
    )


def find_omissions(
    original: str,
    facts: list[ExtractedFact],
    llm: LLM,
    model: SupportedModel,
) -> list[str]:
    """
    Find important facts in the original that weren't extracted.

    Args:
        original: The original memory content
        facts: List of extracted facts
        llm: LLM instance
        model: Model to use

    Returns:
        List of omitted facts
    """
    extracted_facts_str = "\n".join(f"- {f.content}" for f in facts)
    prompt = OMISSIONS_PROMPT.format(
        original=original, extracted_facts=extracted_facts_str
    )

    response = direct_structured_llm_call(
        prompt=prompt,
        response_model=OmissionsResponse,
        model=model,
        llm=llm,
        caller="find_omissions",
    )

    return response.omitted_facts


def annotate_extraction(
    extraction: ExtractionResult,
    llm: LLM,
    model: SupportedModel,
) -> AnnotationResult:
    """
    Annotate all facts in an extraction result.

    Args:
        extraction: The extraction result to annotate
        llm: LLM instance
        model: Model to use

    Returns:
        AnnotationResult with all annotations and omissions
    """
    annotations: list[FactAnnotation] = []

    for fact in extraction.facts:
        try:
            annotation = annotate_fact(
                original=extraction.original_content,
                fact=fact,
                llm=llm,
                model=model,
            )
            annotations.append(annotation)
            logger.debug(
                f"Annotated fact: {fact.content[:50]}... -> {annotation.label}"
            )
        except Exception as e:
            logger.error(f"Failed to annotate fact: {e}")
            # Default to inferred on error
            annotations.append(
                FactAnnotation(
                    fact=fact,
                    label=AnnotationLabel.INFERRED,
                    notes=f"Annotation failed: {e}",
                )
            )

    # Find omissions
    try:
        omissions = find_omissions(
            original=extraction.original_content,
            facts=extraction.facts,
            llm=llm,
            model=model,
        )
    except Exception as e:
        logger.error(f"Failed to find omissions: {e}")
        omissions = []

    return AnnotationResult(
        extraction=extraction,
        annotations=annotations,
        omissions=omissions,
    )


def compute_metrics(annotation_results: list[AnnotationResult]) -> dict[str, float]:
    """
    Compute aggregate metrics across multiple annotation results.

    Args:
        annotation_results: List of annotation results

    Returns:
        Dictionary with computed metrics
    """
    total_facts = 0
    correct_count = 0
    hallucinated_count = 0
    inferred_count = 0
    total_omissions = 0

    for result in annotation_results:
        total_facts += len(result.annotations)
        correct_count += result.correct_count
        hallucinated_count += result.hallucinated_count
        inferred_count += result.inferred_count
        total_omissions += len(result.omissions)

    metrics = {
        "total_facts": total_facts,
        "correct_count": correct_count,
        "hallucinated_count": hallucinated_count,
        "inferred_count": inferred_count,
        "total_omissions": total_omissions,
        "accuracy_rate": correct_count / total_facts if total_facts > 0 else 0.0,
        "hallucination_rate": (
            hallucinated_count / total_facts if total_facts > 0 else 0.0
        ),
        "inference_rate": inferred_count / total_facts if total_facts > 0 else 0.0,
        "avg_omissions_per_memory": (
            total_omissions / len(annotation_results) if annotation_results else 0.0
        ),
    }

    return metrics


def print_annotation_summary(annotation_result: AnnotationResult) -> None:
    """Print a summary of an annotation result."""
    print(f"\n=== Annotation Summary for {annotation_result.extraction.memory_id} ===")
    print(
        f"Original length: {len(annotation_result.extraction.original_content)} chars"
    )
    print(f"Facts extracted: {len(annotation_result.extraction.facts)}")
    print(f"Compression ratio: {annotation_result.extraction.compression_ratio:.2f}")
    print(f"\nAnnotation breakdown:")
    print(
        f"  CORRECT: {annotation_result.correct_count} ({annotation_result.accuracy_rate:.1%})"
    )
    print(
        f"  HALLUCINATED: {annotation_result.hallucinated_count} ({annotation_result.hallucination_rate:.1%})"
    )
    print(f"  INFERRED: {annotation_result.inferred_count}")
    print(f"\nOmissions found: {len(annotation_result.omissions)}")

    if annotation_result.omissions:
        print("Omitted facts:")
        for omission in annotation_result.omissions:
            print(f"  - {omission}")
