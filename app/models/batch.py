"""Batch enrichment request/response."""

from pydantic import BaseModel, Field

from app.models.inputs import EnrichmentRequest
from app.models.outputs import AccountIntelligence


class BatchEnrichmentRequest(BaseModel):
    """Request body for batch enrichment."""

    items: list[EnrichmentRequest] = Field(..., min_length=1, max_length=50)


class BatchEnrichmentItem(BaseModel):
    """Single result in batch response (success or error)."""

    success: bool
    data: AccountIntelligence | None = None
    error: str | None = None


class BatchEnrichmentResponse(BaseModel):
    """Batch enrichment response."""

    results: list[BatchEnrichmentItem]
    total: int
    succeeded: int
    failed: int
