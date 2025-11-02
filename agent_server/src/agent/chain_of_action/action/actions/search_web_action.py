"""
SEARCH_WEB action implementation.
"""

import logging
from typing import Type, List, Dict, Any
import requests
import re
from urllib.parse import quote_plus

from pydantic import BaseModel, Field

from agent.chain_of_action.context import ExecutionContext

from ..action_types import ActionType
from ..base_action import BaseAction
from ..base_action_data import (
    ActionFailureResult,
    ActionOutput,
    ActionResult,
    ActionSuccessResult,
)

from agent.state import State
from agent.llm import LLM, SupportedModel

logger = logging.getLogger(__name__)


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


class SearchWebAction(BaseAction[SearchWebInput, SearchWebOutput]):
    """Search the web for information using DuckDuckGo"""

    action_type = ActionType.SEARCH_WEB

    @classmethod
    def get_action_description(cls) -> str:
        return "Search the web for information using DuckDuckGo search engine"

    @classmethod
    def get_input_type(cls) -> Type[SearchWebInput]:
        return SearchWebInput

    def _parse_duckduckgo_results(self, html_content: str) -> List[SearchResult]:
        """Parse search results from DuckDuckGo HTML"""
        results = []

        # Extract all titles and URLs
        title_matches = re.findall(
            r'<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]*)"[^>]*>(.*?)</a>',
            html_content,
            re.IGNORECASE | re.DOTALL,
        )

        # Extract all snippets with their URLs
        snippet_matches = re.findall(
            r'<a[^>]*class="[^"]*result__snippet[^"]*"[^>]*href="([^"]*)"[^>]*>(.*?)</a>',
            html_content,
            re.IGNORECASE | re.DOTALL,
        )

        # Create a mapping of URL to snippet
        snippet_map = {url: snippet for url, snippet in snippet_matches}

        # Combine results by matching URLs
        for url, title in title_matches[:10]:  # Limit to 10 results
            snippet = snippet_map.get(url, "No snippet available")

            # Clean up HTML entities and tags
            import html

            title = html.unescape(re.sub(r"<[^>]+>", "", title).strip())
            snippet = html.unescape(re.sub(r"<[^>]+>", "", snippet).strip())

            # Handle protocol-relative URLs and extract real URL from DuckDuckGo redirect
            if url and title:
                if url.startswith("//"):
                    url = "https:" + url

                # Extract actual URL from DuckDuckGo redirect (uddg parameter)
                if "duckduckgo.com/l/?uddg=" in url:
                    from urllib.parse import urlparse, parse_qs, unquote

                    parsed = urlparse(url)
                    params = parse_qs(parsed.query)
                    if "uddg" in params:
                        url = unquote(params["uddg"][0])

                if url.startswith(("http://", "https://")):
                    results.append(
                        SearchResult(
                            url=url,
                            title=title,
                            snippet=snippet or "No snippet available",
                        )
                    )

        return results

    def execute(
        self,
        action_input: SearchWebInput,
        context: ExecutionContext,
        state: State,
        llm: LLM,
        model: SupportedModel,
        progress_callback,
    ) -> ActionResult[SearchWebOutput]:

        try:
            # Prepare the search query
            query = action_input.query.strip()
            encoded_query = quote_plus(query)

            # DuckDuckGo search URL
            search_url = f"https://html.duckduckgo.com/html/?q={encoded_query}"

            # Make the search request with proper Chrome headers
            # Note: Requires 'brotli' package to be installed for decompression
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br, zstd",
                "Sec-Ch-Ua": '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
                "Sec-Ch-Ua-Mobile": "?0",
                "Sec-Ch-Ua-Platform": '"Windows"',
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "none",
                "Sec-Fetch-User": "?1",
                "Upgrade-Insecure-Requests": "1",
            }

            logger.debug(f"Searching DuckDuckGo for: {query}")

            response = requests.get(search_url, headers=headers, timeout=10)
            response.raise_for_status()

            # Parse the results
            search_results = self._parse_duckduckgo_results(response.text)

            logger.debug(f"Found {len(search_results)} search results")

            return ActionSuccessResult(
                content=SearchWebOutput(
                    query_used=query,
                    search_results=search_results,
                    total_results_found=len(search_results),
                )
            )

        except requests.RequestException as e:
            error_msg = f"Failed to perform web search: {str(e)}"
            logger.error(error_msg)
            return ActionFailureResult(error=error_msg)

        except Exception as e:
            error_msg = f"Unexpected error during web search: {str(e)}"
            logger.error(error_msg)
            return ActionFailureResult(error=error_msg)
