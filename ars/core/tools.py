"""
Phase 1 tools: just enough to let the Researcher agent pull real sources.
Caching and reranking are intentionally lightweight until the core loop proves
which costs and latency are worth optimizing.
"""

import os
from typing import TypedDict

from tavily import TavilyClient

from ars.cache import TTLCache


class SearchResult(TypedDict):
    title: str
    url: str
    content: str


SEARCH_CACHE_TTL_SECONDS = int(os.environ.get("ARS_SEARCH_CACHE_TTL_SECONDS", "900"))
SEARCH_CACHE_MAX_ENTRIES = int(os.environ.get("ARS_SEARCH_CACHE_MAX_ENTRIES", "256"))
SEARCH_CACHE_ENABLED = os.environ.get("ARS_SEARCH_CACHE_ENABLED", "true").lower() not in {
    "0",
    "false",
    "no",
}

SEARCH_CACHE: TTLCache[list[SearchResult]] = TTLCache(
    ttl_seconds=SEARCH_CACHE_TTL_SECONDS,
    max_entries=SEARCH_CACHE_MAX_ENTRIES,
)


def _search_cache_key(query: str, max_results: int) -> str:
    return f"tavily:{max_results}:{query.strip().lower()}"


def tavily_search(query: str, max_results: int = 3) -> list[SearchResult]:
    """
    Requires TAVILY_API_KEY in the environment.
    Content is truncated to keep the writer prompt within Groq's
    free-tier token-per-minute limits.
    """
    cache_key = _search_cache_key(query, max_results)
    if SEARCH_CACHE_ENABLED:
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

    if SEARCH_CACHE_ENABLED:
        SEARCH_CACHE.set(cache_key, results)

    return results
