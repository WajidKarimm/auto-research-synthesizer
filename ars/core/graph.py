"""
Phase 1 graph: Planner -> Researcher -> Writer, linear, no persistence,
no supervisor. Run it directly to sanity-check the core loop before
touching RAG, guardrails, caching, or the API layer.

Usage:
    python -m ars.core.graph "What are the tradeoffs of pgvector vs Pinecone?"
"""

import sys
from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from ars.core.agents import plan, research, write


class ResearchState(TypedDict, total=False):
    question: str
    queries: list[str]
    sources: list[dict]
    answer: str


def planner_node(state: ResearchState) -> ResearchState:
    queries = plan(state["question"])
    print(f"[planner] {len(queries)} queries:", *[f"\n  - {q}" for q in queries])
    return {"queries": queries}


def researcher_node(state: ResearchState) -> ResearchState:
    sources = research(state["queries"])
    print(f"[researcher] collected {len(sources)} unique sources")
    return {"sources": sources}


def writer_node(state: ResearchState) -> ResearchState:
    answer = write(state["question"], state["sources"])
    return {"answer": answer}


def build_graph():
    graph = StateGraph(ResearchState)
    graph.add_node("planner", planner_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("writer", writer_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "researcher")
    graph.add_edge("researcher", "writer")
    graph.add_edge("writer", END)

    return graph.compile()


def run(question: str) -> ResearchState:
    app = build_graph()
    result = app.invoke({"question": question})
    return result


def _print_sources(sources: list[dict]) -> None:
    print("Sources:")
    for i, s in enumerate(sources):
        print(f"  [{i+1}] {s['title']}\n      {s['url']}")


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    if len(sys.argv) < 2:
        print('Usage: python -m ars.core.graph "your research question"')
        sys.exit(1)

    q = " ".join(sys.argv[1:])
    print(f"\nResearch question: {q}\n{'-' * 60}")
    final_state = run(q)
    print(f"\n{'-' * 60}\nAnswer:\n{final_state['answer']}\n")
    print(f"{'-' * 60}")
    _print_sources(final_state["sources"])
    print()
