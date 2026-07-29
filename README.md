# Auto-Research Synthesizer

Auto-Research Synthesizer (ARS) is a small LangGraph research assistant that
turns a complex question into a sourced answer.

The current flow is:

```text
Safety -> Planner -> Search + Cache -> Retrieval/Rerank -> Writer -> API
```

ARS keeps the architecture intentionally simple: a planner creates search
queries, Tavily collects sources, a lightweight retriever deduplicates and
ranks them, and a writer produces a cited answer.

## Current Status

- Phase 1: linear LangGraph loop in `ars/core/`
- Phase 2: golden eval dataset and faithfulness eval runner in `ars/eval/`
- Phase 3: source normalization and lexical reranking in `ars/retrieval/`
- Phase 4: local input safety checks in `ars/safety/`
- Phase 5: FastAPI app and `/research` endpoint in `ars/main.py` and `ars/api/`
- Phase 6: in-memory TTL cache and runtime settings in `ars/cache/` and
  `ars/utils/settings.py`

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file with:

```bash
GROQ_API_KEY=...
TAVILY_API_KEY=...
```

Run the CLI:

```bash
python -m ars "What are the tradeoffs of pgvector vs Pinecone?"
```

Run the graph module directly:

```bash
python -m ars.core.graph "What are the tradeoffs of pgvector vs Pinecone?"
```

## Streamlit App

Start the user-friendly web app locally:

```bash
streamlit run ars/ui/streamlit_app.py
```

The app provides a question box, example questions, a sourced answer view,
planned queries, and expandable source snippets.

## Streamlit Community Cloud deploy

This repo includes `streamlit_app.py` at the root so Streamlit Cloud can launch it directly. Do not point Streamlit at `ars/__main__.py`.

1. Push your repo to GitHub.
2. Go to https://share.streamlit.io and sign in.
3. Create a new app and choose this repository.
4. Set the app file to `streamlit_app.py`.
5. In the Streamlit Cloud dashboard, add secrets for:
   - `GROQ_API_KEY`
   - `TAVILY_API_KEY`
   - `ARS_MODEL` (optional)

Deploy without a payment method using Streamlit Community Cloud.

## API

Start the FastAPI server:

```bash
uvicorn ars.main:app --reload
```

Open the docs:

```text
http://127.0.0.1:8000/docs
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Run a research request:

```bash
curl -X POST http://127.0.0.1:8000/research ^
  -H "Content-Type: application/json" ^
  -d "{\"question\":\"What are the tradeoffs of pgvector vs Pinecone?\"}"
```

## Docker

Build the container:

```bash
docker build -t ars:latest .
```

Run the API container:

```bash
docker run --rm -p 8000:8000 -e GROQ_API_KEY=... -e TAVILY_API_KEY=... ars:latest
```

Run with Docker Compose:

```bash
cp .env.example .env
# edit .env with your API keys

docker compose up --build
```

The API will be available at `http://localhost:8000`.

## Configuration

Default runtime settings live in `config/config.yaml`:

```yaml
model_name: openai/gpt-oss-120b

search:
  max_results: 3
  max_content_chars: 900

retrieval:
  max_sources: 8

cache:
  enabled: true
  ttl_seconds: 900
  max_entries: 256
```

Environment variables can override these settings:

```bash
ARS_MODEL=...
ARS_SEARCH_MAX_RESULTS=3
ARS_SEARCH_MAX_CONTENT_CHARS=900
ARS_RETRIEVAL_MAX_SOURCES=8
ARS_SEARCH_CACHE_ENABLED=true
ARS_SEARCH_CACHE_TTL_SECONDS=900
ARS_SEARCH_CACHE_MAX_ENTRIES=256
```

## Checks

Compile the main modules:

```bash
python -m py_compile ars\__init__.py ars\__main__.py ars\core\agents.py ars\core\graph.py ars\core\tools.py ars\safety\__init__.py ars\retrieval\__init__.py ars\cache\__init__.py ars\api\__init__.py ars\models\__init__.py ars\main.py ars\utils\settings.py
```

Run local checks that do not need API keys:

```bash
python -m ars hi
python -c "from ars.utils import SETTINGS; print(SETTINGS)"
python -c "from ars.safety import validate_question; print(validate_question('hi')); print(validate_question('What are the tradeoffs of pgvector vs Pinecone?'))"
python -c "from ars.cache import TTLCache; c=TTLCache[int](ttl_seconds=60, max_entries=2); c.set('a', 1); print(c.get('a')); print(c.get('x'))"
python -c "from ars.retrieval import rank_sources; sources=[{'title':'Pinecone vector database','url':'https://a','content':'managed vector search and embeddings','query':'pinecone vector db'},{'title':'Cooking notes','url':'https://b','content':'recipe only','query':'food'}]; print(rank_sources('pgvector vs pinecone tradeoffs', ['pgvector pinecone vector database'], sources)[0]['url'])"
python -c "from fastapi.testclient import TestClient; from ars.main import app; client=TestClient(app); print(client.get('/health').json()); print(client.post('/research', json={'question':'hi'}).json())"
```

Run the eval smoke test:

```bash
python -m ars.eval.run_evals --limit 3
```

## Render deployment

This repository includes `render.yaml` so you can connect it directly to Render.

1. Push your repo to GitHub.
2. In Render, create a new Web Service.
3. Choose "Connect a repo" and select this repository.
4. Render will use `render.yaml` and build the Docker image.
5. In Render service settings, set the environment variables:
   - `GROQ_API_KEY`
   - `TAVILY_API_KEY`
   - `ARS_MODEL` (optional)

Render will deploy the service and expose it on a public URL.

## Project Layout

```text
ars/
  core/       Planner, researcher, writer, graph, and Tavily search tool
  eval/       Golden dataset, faithfulness eval runner, local metric history
  retrieval/  Source normalization and lexical reranking
  safety/     Input guardrails
  api/        FastAPI routes
  models/     Shared API/domain models
  cache/      In-memory TTL cache
  ui/         Streamlit app
  utils/      Settings and shared helpers
config/
  prompts.yaml
  config.yaml
```

## Notes

- The current cache is in-memory and resets when the process exits.
- Retrieval is lexical reranking for now, not embeddings or pgvector yet.
- Safety checks are local guardrails, not a full moderation/policy system.
