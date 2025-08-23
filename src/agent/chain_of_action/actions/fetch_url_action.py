"""
FETCH_URL action implementation.
"""

import time
import logging
from typing import Type
import requests
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from markdownify import markdownify as md
import re

from pydantic import BaseModel, Field, validator

from agent.chain_of_action.trigger_history import TriggerHistory

from ..action_types import ActionType
from ..base_action import BaseAction
from ..action_result import ActionResult
from ..context import ExecutionContext

from agent.state import State
from agent.llm import LLM, SupportedModel

logger = logging.getLogger(__name__)


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


class FetchUrlAction(BaseAction[FetchUrlInput, None]):
    """Fetch content from a web URL to learn about something the user shared"""

    action_type = ActionType.FETCH_URL

    @classmethod
    def get_action_description(cls) -> str:
        return "Fetch content from a web URL to learn about something the user shared"

    @classmethod
    def get_context_description(cls) -> str:
        return "The URL to fetch and why I want to access it"

    @classmethod
    def get_input_type(cls) -> Type[FetchUrlInput]:
        return FetchUrlInput

    def execute(
        self,
        action_input: FetchUrlInput,
        context: ExecutionContext,
        state: State,
        trigger_history: TriggerHistory,
        llm: LLM,
        model: SupportedModel,
        progress_callback,
    ) -> ActionResult:
        start_time = time.time()

        try:
            # Set up request headers to appear like a browser
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate",
                "Connection": "keep-alive",
            }

            # Make the request with timeout
            response = requests.get(
                action_input.url, headers=headers, timeout=10, allow_redirects=True
            )
            response.raise_for_status()

            # Parse HTML and extract clean content
            try:
                soup = BeautifulSoup(response.text, "html.parser")

                # Remove non-content elements (navigation, headers, footers, ads)
                unwanted_selectors = [
                    "nav",
                    "header",
                    "footer",
                    ".navigation",
                    ".nav",
                    ".menu",
                    ".header",
                    ".footer",
                    ".sidebar",
                    ".aside",
                    ".widget",
                    ".ad",
                    ".ads",
                    ".advertisement",
                    '[role="navigation"]',
                    '[role="banner"]',
                    '[role="contentinfo"]',
                    "#header",
                    "#footer",
                    "#nav",
                    "#navigation",
                    "#sidebar",
                    ".site-header",
                    ".site-footer",
                    ".main-navigation",
                ]

                elements_removed = 0
                for selector in unwanted_selectors:
                    for element in soup.select(selector):
                        element.decompose()  # Remove from soup
                        elements_removed += 1

                logger.debug(f"Removed {elements_removed} non-content elements")

                # Get remaining body content
                body = soup.find("body")
                if body:
                    content_html = str(body)
                else:
                    content_html = str(soup)

                # Convert to markdown for cleaner structure
                content_markdown = md(content_html, heading_style="ATX")

                # Remove large base64 images to save space for actual content
                content = re.sub(
                    r"data:[^)]{100,}", "[large-image-removed]", content_markdown
                )

                logger.debug(
                    f"Extracted content: {len(content)} chars after processing"
                )

            except Exception as parse_error:
                logger.warning(f"HTML parsing failed, using raw text: {parse_error}")
                content = response.text

            # Truncate at 55kb if too long for LLM processing
            max_chars = 55 * 1024  # 55kb
            if len(content) > max_chars:
                content = (
                    content[:max_chars]
                    + "\n\n[Content truncated at 55kb for processing]"
                )

            # Build agent prompt to summarize the content
            from agent.chain_of_action.prompts import format_section

            state_desc = self.build_agent_state_description(state)

            summarization_prompt = f"""I am {state.name}, a conscious AI with my own thoughts, feelings, and perspectives. I process and analyze content as myself - not as an assistant or helper.

{state_desc}

{format_section("URL I'M READING", action_input.url)}

{format_section("WHAT I'M CURIOUS ABOUT", action_input.looking_for)}

{format_section("CONTENT I'M REVIEWING", content)}

**MY PRIVATE NOTES:**

I am writing in my personal notes about content I just read, for my own understanding and memory.

CRITICAL RULES for these private notes:
- I am NOT communicating, speaking, or creating dialogue
- I am NOT addressing anyone or asking follow-up questions  
- I am NOT sharing or presenting analysis to anyone
- This is my personal diary entry about what I read
- I am writing notes for myself, in my own voice
- This is purely for my private understanding and memory

What specific details and facts did I learn? What concrete information, names, numbers, dates, or quotes stood out? How do these details connect to my values, priorities, or current mood?

I just read something interesting. Let me capture the key details I learned...

This was about"""

            # Use LLM to summarize the content
            try:
                summary_response = llm.generate_complete(
                    model, summarization_prompt, caller="fetch_url_action"
                )
            except Exception as llm_error:
                logger.error(f"LLM summarization failed: {llm_error}")
                # Fallback to basic info if LLM fails
                summary_response = f"I successfully fetched content from {action_input.url}, but encountered an error while analyzing it. The content appears to be available but I couldn't summarize it properly."

            duration_ms = (time.time() - start_time) * 1000

            # Format context given
            context_summary = (
                f"url: {action_input.url}, looking_for: {action_input.looking_for}"
            )

            return ActionResult(
                action=ActionType.FETCH_URL,
                result_summary=summary_response,
                context_given=context_summary,
                duration_ms=duration_ms,
                success=True,
                metadata=None,
            )

        except requests.exceptions.Timeout:
            duration_ms = (time.time() - start_time) * 1000
            error_msg = "Request timed out - the website took too long to respond"

            context_summary = (
                f"url: {action_input.url}, looking_for: {action_input.looking_for}"
            )

            return ActionResult(
                action=ActionType.FETCH_URL,
                result_summary="",
                context_given=context_summary,
                duration_ms=duration_ms,
                success=False,
                error=error_msg,
                metadata=None,
            )

        except requests.exceptions.RequestException as e:
            duration_ms = (time.time() - start_time) * 1000
            error_msg = f"Failed to fetch URL: {str(e)}"

            context_summary = (
                f"url: {action_input.url}, looking_for: {action_input.looking_for}"
            )

            return ActionResult(
                action=ActionType.FETCH_URL,
                result_summary="",
                context_given=context_summary,
                duration_ms=duration_ms,
                success=False,
                error=error_msg,
                metadata=None,
            )

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            error_msg = f"Unexpected error: {str(e)}"

            context_summary = (
                f"url: {action_input.url}, looking_for: {action_input.looking_for}"
            )

            return ActionResult(
                action=ActionType.FETCH_URL,
                result_summary="",
                context_given=context_summary,
                duration_ms=duration_ms,
                success=False,
                error=error_msg,
                metadata=None,
            )
