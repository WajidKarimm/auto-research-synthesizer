"""
Phase 1 agents: Planner -> Researcher -> Writer.

Deliberately linear and simple. No supervisor routing, no retries,
no RAG, no guardrails. The only thing being validated here is:
does this loop produce a decent, cited research answer?
"""

import os

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from ars.core.tools import tavily_search

# openai/gpt-oss-120b is Groq's current recommended general-purpose model
# (llama-3.3-70b-versatile was deprecated June 2026 — don't use it in new code)
MODEL_NAME = os.environ.get("ARS_MODEL", "openai/gpt-oss-120b")


def _llm(temperature: float = 0.0) -> ChatGroq:
    return ChatGroq(model=MODEL_NAME, temperature=temperature)


# ---------------------------------------------------------------------------
# Planner: turns one research question into 2-4 concrete search queries
# ---------------------------------------------------------------------------

PLANNER_SYSTEM_PROMPT = """You are a research planner. Given a user's research \
question, break it into 2 to 4 specific, distinct web search queries that \
together would let someone answer the question thoroughly.

Rules:
- Output ONLY the search queries, one per line.
- No numbering, no bullets, no explanation, no preamble.
- Each query should be a realistic search engine query (short, specific).
"""


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


def research(queries: list[str]) -> list[dict]:
    """
    Run every planned query and return a flat, deduplicated list of sources.
    Each source keeps its originating query for traceability.
    """
    seen_urls: set[str] = set()
    all_sources: list[dict] = []

    for query in queries:
        results = tavily_search(query)
        for result in results:
            if result["url"] in seen_urls:
                continue
            seen_urls.add(result["url"])
            all_sources.append({**result, "query": query})

    return all_sources


# ---------------------------------------------------------------------------
# Writer: synthesizes sources into a cited answer
# ---------------------------------------------------------------------------

WRITER_SYSTEM_PROMPT = """You are a research synthesizer. You are given a \
research question and a set of source snippets pulled from the web.

Write a clear answer to the question using ONLY information present in the \
sources. Rules:
- Output plain prose paragraphs. Do NOT use markdown tables, bullet lists, \
  or headers. A few well-organized paragraphs only.
- Paraphrase source content in your own words. Do NOT use quotation marks \
  to lift exact phrasing from a source, even short phrases. If a number or \
  fact is source-specific, state it in your own sentence structure.
- Every factual claim must be traceable to a source. Cite sources inline \
  using [n] matching the numbered source list you're given.
- Do not cite the same source more than 2 times in the whole answer -- if \
  you find yourself doing that, the point is probably better made once and \
  referenced, not repeated.
- Do not state a specific number, statistic, or benchmark result unless the \
  exact source snippet contains that number. If a source only implies \
  something qualitatively, describe it qualitatively -- do not sharpen a \
  vague statement into a precise-sounding figure.
- If two sources disagree, say so explicitly rather than picking one silently.
- If the sources don't fully answer the question, say so explicitly rather \
  than filling gaps from your own knowledge.
- Do not invent sources or facts not present in the provided snippets.
- Keep it tight: 3-4 paragraphs maximum, not an essay.
""" 


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
