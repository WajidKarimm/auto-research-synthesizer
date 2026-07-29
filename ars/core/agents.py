"""
Phase 1 agents: Planner -> Researcher -> Writer.

Deliberately linear and simple. No supervisor routing, no retries,
no RAG, no guardrails. The only thing being validated here is:
does this loop produce a decent, cited research answer?
"""

import os
from pathlib import Path
from typing import Any

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from ars.core.tools import tavily_search

# openai/gpt-oss-120b is Groq's current recommended general-purpose model
# (llama-3.3-70b-versatile was deprecated June 2026 — don't use it in new code)
MODEL_NAME = os.environ.get("ARS_MODEL", "openai/gpt-oss-120b")
PROMPTS_PATH = Path(__file__).resolve().parents[2] / "config" / "prompts.yaml"


def _llm(temperature: float = 0.0) -> ChatGroq:
    return ChatGroq(model=MODEL_NAME, temperature=temperature)


def _load_prompts() -> dict[str, str]:
    """
    Load prompts from config/prompts.yaml.

    PyYAML is used when installed. The tiny fallback handles this project's
    simple top-level block scalar format so the core graph still imports in a
    fresh environment before optional config dependencies are installed.
    """
    if not PROMPTS_PATH.exists():
        raise FileNotFoundError(f"Prompt config not found: {PROMPTS_PATH}")

    text = PROMPTS_PATH.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        loaded: Any = yaml.safe_load(text)
        return {
            "planner_system": loaded["planner_system"],
            "writer_system": loaded["writer_system"],
        }
    except ImportError:
        prompts: dict[str, str] = {}
        current_key: str | None = None
        current_lines: list[str] = []

        for line in text.splitlines():
            if line.startswith("planner_system:") or line.startswith("writer_system:"):
                if current_key:
                    prompts[current_key] = "\n".join(current_lines).strip()
                current_key = line.split(":", 1)[0]
                current_lines = []
                continue
            if current_key and (line.startswith("  ") or not line.strip()):
                current_lines.append(line[2:] if line.startswith("  ") else "")

        if current_key:
            prompts[current_key] = "\n".join(current_lines).strip()

        return prompts


PROMPTS = _load_prompts()


# ---------------------------------------------------------------------------
# Planner: turns one research question into 2-4 concrete search queries
# ---------------------------------------------------------------------------

PLANNER_SYSTEM_PROMPT = PROMPTS["planner_system"]


def plan(question: str) -> list[str]:
    """Return a small list of search queries for the given question."""
    response = _llm().invoke(
        [
            SystemMessage(content=PLANNER_SYSTEM_PROMPT),
            HumanMessage(content=question),
        ]
    )
    lines = [line.strip() for line in response.content.strip().split("\n")]
    queries = [line for line in lines if line]
    return queries[:4] if queries else [question]


# ---------------------------------------------------------------------------
# Researcher: runs each query, collects sources
# ---------------------------------------------------------------------------


MAX_TOTAL_SOURCES = 8


def research(queries: list[str]) -> list[dict]:
    """
    Capped at MAX_TOTAL_SOURCES regardless of query count, to bound
    prompt size against Groq's free-tier TPM limit.
    """
    seen_urls: set[str] = set()
    all_sources: list[dict] = []

    for query in queries:
        if len(all_sources) >= MAX_TOTAL_SOURCES:
            break
        results = tavily_search(query)
        for result in results:
            if len(all_sources) >= MAX_TOTAL_SOURCES:
                break
            if result["url"] in seen_urls:
                continue
            seen_urls.add(result["url"])
            all_sources.append({**result, "query": query})

    return all_sources


# ---------------------------------------------------------------------------
# Writer: synthesizes sources into a cited answer
# ---------------------------------------------------------------------------

WRITER_SYSTEM_PROMPT = PROMPTS["writer_system"]


def write(question: str, sources: list[dict]) -> str:
    """Synthesize the final answer from collected sources."""
    if not sources:
        return (
            "I couldn't find any sources for this question. "
            "Try rephrasing it or check that TAVILY_API_KEY is set correctly."
        )

    numbered_sources = "\n\n".join(
        f"[{i+1}] {s['title']}\nURL: {s['url']}\n{s['content']}"
        for i, s in enumerate(sources)
    )

    user_prompt = (
        f"Research question: {question}\n\n"
        f"Sources:\n{numbered_sources}\n\n"
        "Write the synthesized, cited answer now."
    )

    response = _llm().invoke(
        [
            SystemMessage(content=WRITER_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]
    )
    return response.content.strip()
