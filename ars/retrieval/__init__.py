"""Phase 3 retrieval package: source normalization and lightweight reranking."""

import re
from dataclasses import dataclass
from typing import Iterable, TypedDict


TOKEN_RE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_-]*")


class SourceDocument(TypedDict, total=False):
    """Normalized source shape passed from retrieval to the writer."""

    title: str
    url: str
    content: str
    query: str
    score: float


@dataclass(frozen=True)
class RetrievalConfig:
    """Small retrieval settings that can later move to config.yaml."""

    max_sources: int = 8
    min_token_length: int = 3


def tokenize(text: str, min_length: int = 3) -> set[str]:
    """Return normalized keyword tokens for simple lexical matching."""
    return {
        match.group(0).lower()
        for match in TOKEN_RE.finditer(text)
        if len(match.group(0)) >= min_length
    }


def normalize_sources(sources: Iterable[dict]) -> list[SourceDocument]:
    """Deduplicate and normalize raw search results by URL."""
    normalized: list[SourceDocument] = []
    seen_urls: set[str] = set()

    for source in sources:
        url = str(source.get("url", "")).strip()
        if not url or url in seen_urls:
            continue

        seen_urls.add(url)
        normalized.append(
            SourceDocument(
                title=str(source.get("title", "")).strip(),
                url=url,
                content=str(source.get("content", "")).strip(),
                query=str(source.get("query", "")).strip(),
            )
        )

    return normalized


def score_source(source: SourceDocument, query_tokens: set[str]) -> float:
    """Score a source by overlap with query terms, with title matches weighted."""
    title_tokens = tokenize(source.get("title", ""))
    content_tokens = tokenize(source.get("content", ""))
    source_query_tokens = tokenize(source.get("query", ""))

    title_hits = len(query_tokens & title_tokens)
    content_hits = len(query_tokens & content_tokens)
    query_hits = len(query_tokens & source_query_tokens)

    return (title_hits * 2.0) + content_hits + (query_hits * 0.5)


def rank_sources(
    question: str,
    queries: list[str],
    sources: Iterable[dict],
    config: RetrievalConfig | None = None,
) -> list[SourceDocument]:
    """Rank normalized sources for the writer prompt."""
    settings = config or RetrievalConfig()
    query_text = " ".join([question, *queries])
    query_tokens = tokenize(query_text, settings.min_token_length)

    ranked = []
    for source in normalize_sources(sources):
        ranked.append(
            SourceDocument(
                **source,
                score=score_source(source, query_tokens),
            )
        )

    ranked.sort(key=lambda item: item.get("score", 0.0), reverse=True)
    return ranked[: settings.max_sources]
