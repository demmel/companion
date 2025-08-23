"""
Memory query system for retrieving relevant memories based on structured queries.
"""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator
from datetime import datetime
import re


class TimeQuery(BaseModel):
    """Structured time query format"""

    start_time: Optional[str] = Field(
        default=None,
        description="Start time in ISO format (2024-01-15T10:30:00) or relative (-3d, -2h, -1w)",
    )
    end_time: Optional[str] = Field(
        default=None,
        description="End time in ISO format (2024-01-15T10:30:00) or relative (-1d, now)",
    )

    @field_validator("start_time", "end_time")
    @classmethod
    def validate_time_format(cls, v):
        if v is None:
            return v

        # Check if it's "now"
        if v == "now":
            return v

        # Check if it's a relative time format like -3d, -2h, -1w
        relative_pattern = r"^-(\d+)([hdwmy])$"
        if re.match(relative_pattern, v):
            return v

        # Check if it's ISO format
        try:
            datetime.fromisoformat(v.replace("Z", "+00:00"))
            return v
        except ValueError:
            raise ValueError(
                f"Time must be ISO format (2024-01-15T10:30:00), relative (-3d, -2h, -1w), or 'now'. Got: {v}"
            )


class LLMMemoryExtraction(BaseModel):
    """What the LLM extracts from context for memory retrieval"""

    conceptual_query: str = Field(
        default="",
        description="A natural language description of what memories would be relevant to this situation - describe the topics, themes, or experiences that would be helpful to recall",
    )
    time_query: Optional[TimeQuery] = Field(
        default=None, description="Structured temporal query"
    )
