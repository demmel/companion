"""FETCH_URL action data types (input/output/record)."""

from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, Field, validator

from ..action_types import ActionType
from ..base_action_data import ActionOutput, BaseActionData


class FetchUrlInput(BaseModel):
    """Input for FETCH_URL action"""

    url: str = Field(
        description="The URL to fetch content from (must be a valid HTTP/HTTPS URL)"
    )
    looking_for: str = Field(
        description="What specific information I'm hoping to find or learn from this URL"
    )

    @validator("url")
    def validate_url(cls, v):
        """Validate that the URL is properly formatted and uses HTTP/HTTPS"""
        try:
            parsed = urlparse(v)
            if not parsed.scheme or parsed.scheme not in ["http", "https"]:
                raise ValueError("URL must use HTTP or HTTPS protocol")
            if not parsed.netloc:
                raise ValueError("URL must have a valid domain")
            return v
        except Exception as e:
            raise ValueError(f"Invalid URL format: {e}")


class FetchUrlOutput(ActionOutput):
    """Output for FETCH_URL action"""

    content_summary: str

    def result_summary(self) -> str:
        return self.content_summary


class FetchUrlActionData(BaseActionData[FetchUrlInput, FetchUrlOutput]):
    type: Literal[ActionType.FETCH_URL] = ActionType.FETCH_URL
