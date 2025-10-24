# Action s
from typing import Literal

from agent.chain_of_action.action.action_data import ActionData, create_context_given
from pydantic import BaseModel


class ActionStatusSuccess(BaseModel):
    type: Literal["success"] = "success"
    result: str


class ActionStatusError(BaseModel):
    type: Literal["error"] = "error"
    error: str | None = None


class ActionStatusStreaming(BaseModel):
    type: Literal["streaming"] = "streaming"
    result: str


ActionStatus = ActionStatusSuccess | ActionStatusError | ActionStatusStreaming


class BaseAction(BaseModel):
    """Base class for action s"""

    context_given: str
    reasoning: str
    status: ActionStatus
    duration_ms: int


class ThinkAction(BaseAction):
    """for think actions"""

    type: Literal["think"] = "think"


class SpeakAction(BaseAction):
    """for speak actions"""

    type: Literal["speak"] = "speak"


class UpdateAppearanceAction(BaseAction):
    """for appearance update actions"""

    type: Literal["update_appearance"] = "update_appearance"
    image_description: str
    image_url: str


class UpdateEnvironmentAction(BaseAction):
    """for environment update actions"""

    type: Literal["update_environment"] = "update_environment"
    image_description: str
    image_url: str


class UpdateMoodAction(BaseAction):
    """for mood update actions"""

    type: Literal["update_mood"] = "update_mood"


class WaitAction(BaseAction):
    """for wait actions"""

    type: Literal["wait"] = "wait"


class CreativeInspirationAction(BaseAction):
    """for creative inspiration actions"""

    type: Literal["get_creative_inspiration"] = "get_creative_inspiration"
    words: list[str]


class AddPriorityAction(BaseAction):
    """for add priority actions"""

    type: Literal["add_priority"] = "add_priority"


class RemovePriorityAction(BaseAction):
    """for remove priority actions"""

    type: Literal["remove_priority"] = "remove_priority"


class PriorityOperationResult(BaseModel):
    """Result of a priority operation"""

    operation_type: Literal["add", "remove", "merge", "refine", "reorder"]
    summary: str


class EvaluatePrioritiesAction(BaseAction):
    """for evaluate priorities actions"""

    type: Literal["evaluate_priorities"] = "evaluate_priorities"
    operations: list[PriorityOperationResult]


class FetchUrlAction(BaseAction):
    """for fetch URL actions"""

    type: Literal["fetch_url"] = "fetch_url"
    url: str
    looking_for: str


class SearchResult(BaseModel):
    """for individual search results"""

    url: str
    title: str
    snippet: str


class SearchWebAction(BaseAction):
    """for web search actions"""

    type: Literal["search_web"] = "search_web"
    query: str
    purpose: str
    search_results: list[SearchResult]


# Discriminated union for all action types
Action = (
    ThinkAction
    | SpeakAction
    | UpdateAppearanceAction
    | UpdateEnvironmentAction
    | UpdateMoodAction
    | WaitAction
    | CreativeInspirationAction
    | AddPriorityAction
    | RemovePriorityAction
    | EvaluatePrioritiesAction
    | FetchUrlAction
    | SearchWebAction
)


def convert_action_to_dto(action: ActionData) -> Action:
    """Convert backend ActionResult to"""
    from agent.chain_of_action.action.action_types import ActionType

    # Convert success/result_summary to ActionStatus
    if action.result.type == "success":
        status = ActionStatusSuccess(result=action.result.content.result_summary())
    else:
        status = ActionStatusError(error=action.result.error)

    context_given = create_context_given(action)
    base_data = {
        "context_given": context_given,
        "reasoning": action.reasoning,
        "status": status,
        "duration_ms": int(action.duration_ms) if action.duration_ms is not None else 0,
    }

    if action.type == ActionType.THINK:
        return ThinkAction(**base_data)
    elif action.type == ActionType.SPEAK:
        return SpeakAction(**base_data)
    elif action.type == ActionType.UPDATE_APPEARANCE:
        image_description = None
        image_url = None

        if action.result.type == "success":
            image_description = action.result.content.image_description
            image_url = action.result.content.image_result.image_url
        else:
            image_description = "Failed to generate image"
            image_url = ""

        return UpdateAppearanceAction(
            **base_data, image_description=image_description, image_url=image_url
        )
    elif action.type == ActionType.UPDATE_ENVIRONMENT:
        image_description = None
        image_url = None

        if action.result.type == "success":
            image_description = action.result.content.image_description
            image_url = action.result.content.image_result.image_url
        else:
            image_description = "Failed to generate image"
            image_url = ""

        return UpdateEnvironmentAction(
            **base_data, image_description=image_description, image_url=image_url
        )
    elif action.type == ActionType.UPDATE_MOOD:
        return UpdateMoodAction(**base_data)
    elif action.type == ActionType.WAIT:
        return WaitAction(**base_data)
    elif action.type == ActionType.GET_CREATIVE_INSPIRATION:
        words = []
        if action.result.type == "success":
            words = action.result.content.words
        return CreativeInspirationAction(**base_data, words=words)
    elif action.type == ActionType.ADD_PRIORITY:
        return AddPriorityAction(**base_data)
    elif action.type == ActionType.REMOVE_PRIORITY:
        return RemovePriorityAction(**base_data)
    elif action.type == ActionType.EVALUATE_PRIORITIES:
        operations = []
        if action.result.type == "success":
            operations = [
                PriorityOperationResult(
                    operation_type=op_result.operation_type, summary=op_result.summary
                )
                for op_result in action.result.content.operation_results
            ]
        return EvaluatePrioritiesAction(**base_data, operations=operations)
    elif action.type == ActionType.FETCH_URL:
        url = action.input.url
        looking_for = action.input.looking_for
        return FetchUrlAction(**base_data, url=url, looking_for=looking_for)
    elif action.type == ActionType.SEARCH_WEB:
        query = action.input.query
        purpose = action.input.purpose
        search_results = []

        if action.result.type == "success":
            # Convert SearchResult objects to SearchResult objects
            search_results = [
                SearchResult(url=result.url, title=result.title, snippet=result.snippet)
                for result in action.result.content.search_results
            ]

        return SearchWebAction(
            **base_data, query=query, purpose=purpose, search_results=search_results
        )
    else:
        raise ValueError(f"Unsupported action type: {action.action}")
