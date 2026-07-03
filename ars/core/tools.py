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


def tavily_search(query: str, max_results: int = 4) -> list[SearchResult]:
    """
    Run a single web search and return raw source snippets.

    Requires TAVILY_API_KEY in the environment. Raises a clear error
    if it's missing rather than failing silently later in the graph.
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

    results: list[SearchResult] = []
    for item in response.get("results", []):
        results.append(
            SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                content=item.get("content", ""),
            )
        )
    return results
