"""Phase 5 API routes and request orchestration."""

from fastapi import APIRouter, HTTPException

from ars.core.graph import run
from ars.models import ResearchRequest, ResearchResponse


router = APIRouter()


@router.post("/research", response_model=ResearchResponse)
def research_endpoint(payload: ResearchRequest) -> ResearchResponse:
    """Run the research graph for one question."""
    try:
        state = run(payload.question)
    except Exception as exc:  # pragma: no cover - API boundary protection
        raise HTTPException(
            status_code=500,
            detail=f"Research run failed: {exc}",
        ) from exc

    return ResearchResponse(
        question=state["question"],
        answer=state.get("answer", ""),
        sources=state.get("sources", []) if payload.include_sources else [],
        queries=state.get("queries", []),
        safety_error=state.get("safety_error"),
    )
