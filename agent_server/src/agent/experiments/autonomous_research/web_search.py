"""
Shared web search utilities using DuckDuckGo.

Extracted from agent.chain_of_action.action.actions.search_web_action to avoid duplication.
"""

import logging
import re
import html as html_lib
from typing import List, Dict
from urllib.parse import urlparse, parse_qs, unquote, quote_plus
import requests

logger = logging.getLogger(__name__)


def parse_duckduckgo_url(url: str) -> str:
    """
    Extract actual URL from DuckDuckGo redirect URL.

    Handles:
    - Protocol-relative URLs (//example.com -> https://example.com)
    - DuckDuckGo redirect URLs (extracts uddg parameter)

    Returns the cleaned URL or empty string if invalid.
    """
    if not url:
        return ""

    # Handle protocol-relative URLs
    if url.startswith("//"):
        url = "https:" + url

    # Extract actual URL from DuckDuckGo redirect (uddg parameter)
    if "duckduckgo.com/l/?uddg=" in url:
        parsed = urlparse(url)
        params = parse_qs(parsed.query)
        if "uddg" in params:
            url = unquote(params["uddg"][0])

    # Only return valid HTTP(S) URLs
    if url.startswith(("http://", "https://")):
        return url

    return ""


def parse_duckduckgo_results(
    html_content: str, max_results: int = 10
) -> List[Dict[str, str]]:
    """
    Parse search results from DuckDuckGo HTML.

    Args:
        html_content: Raw HTML from DuckDuckGo search
        max_results: Maximum number of results to return

    Returns:
        List of dicts with 'url' and 'title' keys
    """
    results = []

    # Extract title and URL pairs
    title_matches = re.findall(
        r'<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]*)"[^>]*>(.*?)</a>',
        html_content,
        re.IGNORECASE | re.DOTALL,
    )

    for url, title in title_matches[:max_results]:
        # Clean up title
        title = html_lib.unescape(re.sub(r"<[^>]+>", "", title).strip())

        # Parse and clean URL
        url = parse_duckduckgo_url(url)

        if url and title:
            results.append({"url": url, "title": title})

    return results


def search_duckduckgo(query: str, max_results: int = 10) -> List[Dict[str, str]]:
    """
    Search DuckDuckGo and return results.

    Args:
        query: Search query string
        max_results: Maximum number of results to return

    Returns:
        List of dicts with 'url' and 'title' keys
    """
    try:
        encoded_query = quote_plus(query)
        search_url = f"https://html.duckduckgo.com/html/?q={encoded_query}"

        # Use proper Chrome headers (from search_web_action.py)
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

        response = requests.get(search_url, headers=headers, timeout=10)
        response.raise_for_status()

        return parse_duckduckgo_results(response.text, max_results)

    except Exception as e:
        logger.error(f"Error searching DuckDuckGo: {e}")
        return []
