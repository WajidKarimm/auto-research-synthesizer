# Auto-Research Synthesizer

Auto-Research Synthesizer (ARS) is a small, staged LangGraph project for
turning a complex research question into a sourced answer.

The current core is intentionally linear with lightweight safety, retrieval,
caching, and API layers around it:

```text
Safety -> Planner -> Search + Cache -> Retrieval/Rerank -> Writer -> API
```

That simplicity is the point. ARS first proves that a minimal loop can plan
useful searches, collect web snippets, and write a faithful cited answer before
adding retrieval infrastructure, guardrails, caching, and an API.

## Current Status

- Phase 1: linear LangGraph loop in `ars/core/`
- Phase 2: golden eval dataset and faithfulness eval runner in `ars/eval/`
- Phase 3: source normalization and lightweight lexical reranking in
  `ars/retrieval/`
- Phase 4: local input safety checks in `ars/safety/`
- Phase 5: FastAPI app and `/research` endpoint in `ars/main.py` and `ars/api/`
- Phase 6: in-memory TTL cache for Tavily search results in `ars/cache/`

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file with at least:

```bash
GROQ_API_KEY=...
TAVILY_API_KEY=...
```

Run the research loop:

```bash
python -m ars "What are the tradeoffs of pgvector vs Pinecone?"
```

Or call the graph module directly:

```bash
python -m ars.core.graph "What are the tradeoffs of pgvector vs Pinecone?"
```

Run the API:

```bash
uvicorn ars.main:app --reload
```

Then open `http://127.0.0.1:8000/docs`.

Run the Phase 2 eval smoke test:

```bash
python -m ars.eval.run_evals --limit 3
```

## Project Layout

```text
ars/
  core/       Planner, Researcher, Writer, and Tavily search tool
  eval/       Golden dataset, faithfulness eval runner, local metric history
  retrieval/  Source normalization and lexical reranking
  safety/     Input guardrails
  api/        FastAPI routes
  models/     Shared API/domain models
  cache/      In-memory TTL cache
  utils/      Shared helpers
config/
  prompts.yaml
  config.yaml
```
