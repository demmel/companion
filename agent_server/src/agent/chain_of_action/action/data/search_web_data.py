"""SEARCH_WEB action data types (input/output/record)."""

from typing import List, Literal

from pydantic import BaseModel, Field

from ..action_types import ActionType
from ..base_action_data import ActionOutput, BaseActionData


class SearchResult(BaseModel):
    """Individual search result"""

    title: str
    url: str
    snippet: str


class SearchWebInput(BaseModel):
    """Input for SEARCH_WEB action"""

    purpose: str = Field(
        description="What specific information I'm hoping to find or learn from this search"
    )
    query: str = Field(
        description="Search query string. For best results: use specific keywords rather than questions (e.g., 'Python asyncio tutorial' not 'How do I use asyncio in Python?'), include relevant context terms, avoid overly broad searches, use quotes for exact phrases when needed"
    )


class SearchWebOutput(ActionOutput):
    """Output for SEARCH_WEB action"""

    query_used: str
    search_results: List[SearchResult]
    total_results_found: int

    def result_summary(self) -> str:
        if not self.search_results:
            return f"No results found for query: '{self.query_used}'"

        results_summary = (
            f"Found {self.total_results_found} results for '{self.query_used}':\n"
        )
        for i, result in enumerate(self.search_results, 1):
            results_summary += (
                f"{i}. {result.title} ({result.url})\n   {result.snippet}\n"
            )

        return results_summary


class SearchWebActionData(BaseActionData[SearchWebInput, SearchWebOutput]):
    type: Literal[ActionType.SEARCH_WEB] = ActionType.SEARCH_WEB
