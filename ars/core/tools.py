"""
Phase 1 tools: just enough to let the Researcher agent pull real sources.
Caching and reranking are intentionally lightweight until the core loop proves
which costs and latency are worth optimizing.
"""

import os
from typing import TypedDict

from tavily import TavilyClient

from ars.cache import TTLCache
from ars.utils import SETTINGS


class SearchResult(TypedDict):
    title: str
    url: str
    content: str


SEARCH_CACHE: TTLCache[list[SearchResult]] = TTLCache(
    ttl_seconds=SETTINGS.cache.ttl_seconds,
    max_entries=SETTINGS.cache.max_entries,
)


def _search_cache_key(query: str, max_results: int) -> str:
    return f"tavily:{max_results}:{query.strip().lower()}"


def tavily_search(
    query: str,
    max_results: int = SETTINGS.search.max_results,
) -> list[SearchResult]:
    """
    Requires TAVILY_API_KEY in the environment.
    Content is truncated to keep the writer prompt within Groq's
    free-tier token-per-minute limits.
    """
    cache_key = _search_cache_key(query, max_results)
    if SETTINGS.cache.enabled:
        cached = SEARCH_CACHE.get(cache_key)
        if cached is not None:
            return cached

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
                content=item.get("content", "")[: SETTINGS.search.max_content_chars],
            )
        )

    if SETTINGS.cache.enabled:
        SEARCH_CACHE.set(cache_key, results)

    return results
