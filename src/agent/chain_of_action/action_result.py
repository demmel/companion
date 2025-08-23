"""
Action result definitions.
"""

from typing import Generic, TypeVar
from pydantic import BaseModel, model_validator

from .action_types import ActionType

TMetadata = TypeVar("TMetadata", bound=BaseModel | None)


class ActionResult(BaseModel, Generic[TMetadata]):
    """Result of executing an action"""

    action: ActionType
    result_summary: str
    context_given: str
    duration_ms: float
    metadata: TMetadata
    success: bool = True
    error: str = ""

    @model_validator(mode="before")
    @classmethod
    def validate_metadata(cls, data):
        """Convert metadata dict to proper type based on action type"""
        if isinstance(data, dict) and "action" in data and "metadata" in data:
            action = data["action"]
            metadata = data["metadata"]

            # Only handle UPDATE_APPEARANCE for now since it's the only one using metadata
            if action == ActionType.UPDATE_APPEARANCE and isinstance(metadata, dict):
                from .actions.update_appearance_action import (
                    UpdateAppearanceActionMetadata,
                )

                data["metadata"] = UpdateAppearanceActionMetadata.model_validate(
                    metadata
                )

        return data
