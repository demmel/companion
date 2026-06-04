"""
ActionData: the persisted record of an executed action.

Per-action data classes (``FooActionData``) are co-located in each action module. They
auto-collect into ``_ACTION_DATA_TYPES`` (see ``base_action_data``), so the discriminated
``ActionData`` union and the type->class lookup here are *derived* from the registry
rather than hand-maintained: adding an action requires no edit to this file.
"""

import functools
import operator
from datetime import datetime
from typing import Any, TypeGuard

from agent.chain_of_action.action.action_types import ActionType

# Importing the actions package runs every action module, which (a) registers each
# action via @register_action and (b) defines each FooActionData, collecting it into
# _ACTION_DATA_TYPES.
import agent.chain_of_action.action.actions  # noqa: F401

from .base_action_data import BaseActionData, _ACTION_DATA_TYPES


# Static type used in annotations. Concrete values are FooActionData instances; code
# narrows to a specific action via ``isinstance(x, FooActionData)`` imported from the
# action's own module.
ActionData = BaseActionData

# Runtime discriminated union (members carry distinct Literal ``type`` discriminators),
# derived from the collected per-action classes. Used by the storage TypeAdapter.
ACTION_DATA_UNION = functools.reduce(operator.or_, _ACTION_DATA_TYPES)

# type -> concrete data class, derived from the collected classes.
_ACTION_DATA_CONSTRUCTORS: dict[ActionType, type[BaseActionData]] = {
    cls.model_fields["type"].default: cls for cls in _ACTION_DATA_TYPES
}


def isinstance_of_action_data(obj: object) -> TypeGuard[BaseActionData]:
    return isinstance(obj, BaseActionData)


def cast_base_action_data_to_action_data(action_data: BaseActionData) -> BaseActionData:
    if isinstance(action_data, BaseActionData):
        return action_data
    raise ValueError(f"Unknown action data type: {action_data.__class__}")


def create_action_data(
    type: ActionType,
    reasoning: str,
    input: Any,
    result: Any,
    duration_ms: float,
    start_timestamp: datetime,
) -> ActionData:
    constructor = _ACTION_DATA_CONSTRUCTORS.get(type)
    if not constructor:
        raise ValueError(f"Unknown action type: {type}")
    return constructor(
        reasoning=reasoning,
        input=input,
        result=result,
        duration_ms=duration_ms,
        start_timestamp=start_timestamp,
    )


def create_result_summary(action_data: BaseActionData) -> str:
    """Create a result summary for the given action data."""
    if action_data.result.type == "success":
        return action_data.result.content.result_summary()
    return action_data.result.error


def create_context_given(action_data: BaseActionData) -> str:
    """Create a context given for the given action data."""
    data = action_data.input.model_dump(mode="json")
    return ", ".join(f"{key}: {value}" for key, value in data.items())
