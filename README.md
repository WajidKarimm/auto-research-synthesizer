# Auto-Research Synthesizer

Auto-Research Synthesizer (ARS) is a small, staged LangGraph project for
turning a complex research question into a sourced answer.

The current core is intentionally linear:

```text
Planner -> Researcher -> Writer
```

That simplicity is the point. ARS first proves that a minimal loop can plan
useful searches, collect web snippets, and write a faithful cited answer before
adding retrieval infrastructure, guardrails, caching, and an API.

## Current Status

- Phase 1: linear LangGraph loop in `ars/core/`
- Phase 2: golden eval dataset and faithfulness eval runner in `ars/eval/`
- Phase 3: retrieval package placeholder for pgvector, embeddings, and reranking
- Phase 4: safety package placeholder for input/output guards
- Phase 5: FastAPI entrypoint placeholder in `ars/main.py`
- Phase 6: CI/cache/metrics placeholders, to expand only when the core loop
  produces useful eval signal

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
python -m ars.core.graph "What are the tradeoffs of pgvector vs Pinecone?"
```

Run the Phase 2 eval smoke test:

```bash
python -m ars.eval.run_evals --limit 3
```

## Project Layout

```text
ars/
  core/       Planner, Researcher, Writer, and Tavily search tool
  eval/       Golden dataset, faithfulness eval runner, local metric history
  retrieval/  Phase 3 retrieval work
  safety/     Phase 4 guardrails
  api/        Phase 5 API routes
  models/     Phase 5 schemas/domain models
  cache/      Phase 6 cache work, if justified
  utils/      Shared helpers
config/
  prompts.yaml
  config.yaml
```
