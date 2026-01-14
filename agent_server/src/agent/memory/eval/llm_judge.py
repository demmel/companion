"""LLM-based judgment for memory evaluation."""

from typing import List

from pydantic import BaseModel, Field

from agent.llm import LLM, SupportedModel
from agent.structured_llm import direct_structured_llm_call
from .data_models import Judgment


class JudgmentResponse(BaseModel):
    """Response from LLM for a single judgment."""

    is_present: bool = Field(description="Whether the expected information is present")
    reasoning: str = Field(description="Explanation for why it is or isn't present")
    confidence: float = Field(
        description="Confidence in judgment (0.0-1.0)",
        ge=0.0,
        le=1.0,
    )


def judge_single_item(
    memory_output: str,
    expected_item: str,
    llm: LLM,
    model: SupportedModel,
) -> Judgment:
    """Judge whether a single expected item is present in memory output."""

    prompt = f"""You are evaluating whether a memory retrieval system successfully retrieved expected information.

Given this memory retrieval output:
---
{memory_output}
---

Determine if this expected information is present: "{expected_item}"

IMPORTANT:
- The information doesn't need to be worded exactly the same
- Look for semantic equivalence - the same meaning conveyed in different words
- Partial matches count as present if the core information is there
- Be generous in interpretation, but the key facts must be present

Is this information present in the output?"""

    response = direct_structured_llm_call(
        prompt=prompt,
        response_model=JudgmentResponse,
        model=model,
        llm=llm,
        caller="memory_eval_judge",
    )

    return Judgment(
        expected_item=expected_item,
        is_present=response.is_present,
        reasoning=response.reasoning,
        confidence=response.confidence,
    )


def judge_all_items(
    memory_output: str,
    expected_items: List[str],
    llm: LLM,
    model: SupportedModel,
) -> List[Judgment]:
    """Judge all expected items against memory output."""
    judgments = []

    for item in expected_items:
        judgment = judge_single_item(memory_output, item, llm, model)
        judgments.append(judgment)

    return judgments


def compute_recall(judgments: List[Judgment]) -> float:
    """Compute recall from judgments."""
    if not judgments:
        return 0.0

    present_count = sum(1 for j in judgments if j.is_present)
    return present_count / len(judgments)
