"""Phase 5 shared API/domain models."""

from typing import Any

from pydantic import BaseModel, Field


class ResearchRequest(BaseModel):
    """Request body for running the research graph."""

    question: str = Field(
        ...,
        min_length=1,
        description="Research question to plan, search, and synthesize.",
    )
    include_sources: bool = Field(
        True,
        description="Include collected source metadata in the response.",
    )


class ResearchResponse(BaseModel):
    """Response returned by the research API."""

    question: str
    answer: str
    sources: list[dict[str, Any]] = Field(default_factory=list)
    queries: list[str] = Field(default_factory=list)
    safety_error: str | None = None
