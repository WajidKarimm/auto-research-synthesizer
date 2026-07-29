"""Streamlit app for running Auto-Research Synthesizer."""

from dotenv import load_dotenv

import streamlit as st

from ars.core.graph import run


EXAMPLE_QUESTIONS = (
    "What are the tradeoffs of pgvector vs Pinecone?",
    "Is long-context prompting replacing retrieval augmented generation?",
    "How does semantic caching reduce LLM inference cost?",
)


def _init_page() -> None:
    st.set_page_config(
        page_title="Auto-Research Synthesizer",
        layout="wide",
    )
    st.title("Auto-Research Synthesizer")
    st.caption("Ask a research question and get a sourced answer.")


def _render_sidebar() -> None:
    with st.sidebar:
        st.header("Run Settings")
        st.write("The app uses your `.env` and `config/config.yaml` settings.")
        st.divider()
        st.subheader("Examples")
        for question in EXAMPLE_QUESTIONS:
            if st.button(question, use_container_width=True):
                st.session_state["question"] = question


def _render_sources(sources: list[dict]) -> None:
    if not sources:
        st.info("No sources were collected for this run.")
        return

    st.subheader("Sources")
    for index, source in enumerate(sources, start=1):
        title = source.get("title") or "Untitled source"
        url = source.get("url") or ""
        score = source.get("score")
        label = f"[{index}] {title}"
        if score is not None:
            label = f"{label} | score {score:.1f}"

        with st.expander(label):
            if url:
                st.link_button("Open source", url)
            if source.get("query"):
                st.caption(f"Search query: {source['query']}")
            st.write(source.get("content") or "No snippet available.")


def _render_result(state: dict) -> None:
    safety_error = state.get("safety_error")
    if safety_error:
        st.warning(safety_error)
        return

    st.subheader("Answer")
    st.markdown(state.get("answer") or "No answer returned.")

    queries = state.get("queries") or []
    if queries:
        st.subheader("Planned Queries")
        for query in queries:
            st.write(f"- {query}")

    _render_sources(state.get("sources") or [])


def main() -> None:
    load_dotenv()
    _init_page()
    _render_sidebar()

    default_question = st.session_state.get("question", EXAMPLE_QUESTIONS[0])
    with st.form("research_form"):
        question = st.text_area(
            "Research question",
            value=default_question,
            height=120,
            placeholder="Ask a specific research question...",
        )
        submitted = st.form_submit_button("Run research", type="primary")

    if not submitted:
        return

    with st.status("Running research...", expanded=True) as status:
        st.write("Checking safety")
        st.write("Planning searches")
        st.write("Collecting and reranking sources")
        st.write("Writing cited answer")
        try:
            state = run(question)
        except Exception as exc:
            status.update(label="Research failed", state="error")
            st.error(str(exc))
            return
        status.update(label="Research complete", state="complete")

    _render_result(state)


if __name__ == "__main__":
    main()
