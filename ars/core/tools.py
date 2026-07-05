"""
Phase 1 tools: just enough to let the Researcher agent pull real sources.
No caching, no reranking, no scraping fallback yet -- that's Phase 3+.
"""

import os
from typing import TypedDict

from tavily import TavilyClient


class SearchResult(TypedDict):
    title: str
    url: str
    content: str


def tavily_search(query: str, max_results: int = 3) -> list[SearchResult]:
    """
    Requires TAVILY_API_KEY in the environment.
    Content is truncated to keep the writer prompt within Groq's
    free-tier token-per-minute limits.
    """
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError(
            "TAVILY_API_KEY is not set. Get a free key at https://tavily.com "
            "and add it to your .env file."
        )

    client = TavilyClient(api_key=api_key)
    response = client.search(
        query=query,
        max_results=max_results,
        include_answer=False,
        search_depth="basic",
    )

    MAX_CONTENT_CHARS = 900

    results: list[SearchResult] = []
    for item in response.get("results", []):
        results.append(
            SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                content=item.get("content", "")[:MAX_CONTENT_CHARS],
            )
        )
    return results