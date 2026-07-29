"""FastAPI entrypoint placeholder for Phase 5."""

try:
    from fastapi import FastAPI
except ImportError:  # pragma: no cover - useful before API deps are installed
    FastAPI = None  # type: ignore


if FastAPI is not None:
    from ars.api import router as api_router

    app = FastAPI(title="Auto-Research Synthesizer")
    app.include_router(api_router)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}
else:
    app = None
