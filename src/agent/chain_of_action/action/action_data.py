from typing import Any, Literal
from agent.chain_of_action.action.action_types import ActionType
from agent.chain_of_action.action.actions.fetch_url_action import (
    FetchUrlInput,
    FetchUrlOutput,
)
from agent.chain_of_action.action.actions.priority_actions import (
    AddPriorityInput,
    AddPriorityOutput,
    RemovePriorityInput,
    RemovePriorityOutput,
)
from agent.chain_of_action.action.actions.speak_action import SpeakInput, SpeakOutput
from agent.chain_of_action.action.actions.think_action import ThinkInput, ThinkOutput
from agent.chain_of_action.action.actions.update_appearance_action import (
    UpdateAppearanceInput,
    UpdateAppearanceOutput,
)
from agent.chain_of_action.action.actions.update_mood_action import (
    UpdateMoodInput,
    UpdateMoodOutput,
)
from agent.chain_of_action.action.actions.wait_action import WaitInput, WaitOutput
from .base_action_data import BaseActionData


class FetchUrlActionData(BaseActionData[FetchUrlInput, FetchUrlOutput]):
    type: Literal[ActionType.FETCH_URL] = ActionType.FETCH_URL


class AddPriorityActionData(BaseActionData[AddPriorityInput, AddPriorityOutput]):
    type: Literal[ActionType.ADD_PRIORITY] = ActionType.ADD_PRIORITY


class RemovePriorityActionData(
    BaseActionData[RemovePriorityInput, RemovePriorityOutput]
):
    type: Literal[ActionType.REMOVE_PRIORITY] = ActionType.REMOVE_PRIORITY


class SpeakActionData(BaseActionData[SpeakInput, SpeakOutput]):
    type: Literal[ActionType.SPEAK] = ActionType.SPEAK


class ThinkActionData(BaseActionData[ThinkInput, ThinkOutput]):
    type: Literal[ActionType.THINK] = ActionType.THINK


class UpdateAppearanceActionData(
    BaseActionData[UpdateAppearanceInput, UpdateAppearanceOutput]
):
    type: Literal[ActionType.UPDATE_APPEARANCE] = ActionType.UPDATE_APPEARANCE


class UpdateMoodActionData(BaseActionData[UpdateMoodInput, UpdateMoodOutput]):
    type: Literal[ActionType.UPDATE_MOOD] = ActionType.UPDATE_MOOD


class WaitActionData(BaseActionData[WaitInput, WaitOutput]):
    type: Literal[ActionType.WAIT] = ActionType.WAIT


ActionData = (
    FetchUrlActionData
    | AddPriorityActionData
    | RemovePriorityActionData
    | SpeakActionData
    | ThinkActionData
    | UpdateAppearanceActionData
    | UpdateMoodActionData
    | WaitActionData
)


def cast_base_action_data_to_action_data(action_data: BaseActionData) -> ActionData:
    match action_data:
        case FetchUrlActionData():
            return action_data
        case AddPriorityActionData():
            return action_data
        case RemovePriorityActionData():
            return action_data
        case SpeakActionData():
            return action_data
        case ThinkActionData():
            return action_data
        case UpdateAppearanceActionData():
            return action_data
        case UpdateMoodActionData():
            return action_data
        case WaitActionData():
            return action_data

    raise ValueError(f"Unknown action data type: {action_data.__class__}")


def create_action_data(
    type: ActionType, input: Any, result: Any, duration_ms: float
) -> ActionData:
    match type:
        case ActionType.FETCH_URL:
            return FetchUrlActionData(
                input=input, result=result, duration_ms=duration_ms
            )
        case ActionType.ADD_PRIORITY:
            return AddPriorityActionData(
                input=input, result=result, duration_ms=duration_ms
            )
        case ActionType.REMOVE_PRIORITY:
            return RemovePriorityActionData(
                input=input, result=result, duration_ms=duration_ms
            )
        case ActionType.SPEAK:
            return SpeakActionData(input=input, result=result, duration_ms=duration_ms)
        case ActionType.THINK:
            return ThinkActionData(input=input, result=result, duration_ms=duration_ms)
        case ActionType.UPDATE_APPEARANCE:
            return UpdateAppearanceActionData(
                input=input, result=result, duration_ms=duration_ms
            )
        case ActionType.UPDATE_MOOD:
            return UpdateMoodActionData(
                input=input, result=result, duration_ms=duration_ms
            )
        case ActionType.WAIT:
            return WaitActionData(input=input, result=result, duration_ms=duration_ms)


def create_result_summary(action_data: BaseActionData) -> str:
    """Create a result summary for the given action data."""
    if action_data.result.type == "success":
        return action_data.result.content.result_summary()
    return action_data.result.error


def create_context_given(action_data: BaseActionData) -> str:
    """Create a context given for the given action data."""
    data = action_data.input.model_dump(mode="json")
    return ", ".join(f"{key}: {value}" for key, value in data.items())
