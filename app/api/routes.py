"""Enrichment API routes."""

import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from app.logging_utils import get_logger
from app.models.inputs import EnrichmentRequest
from app.models.outputs import AccountIntelligence
from app.models.batch import BatchEnrichmentRequest, BatchEnrichmentResponse, BatchEnrichmentItem
from app.router.input_router import route_input
from app.graph.pipeline import run_pipeline
from app import storage


router = APIRouter(prefix="/api", tags=["enrichment"])
logger = get_logger("app.api.routes")


def _run_one(request: EnrichmentRequest, trace_id: str | None = None) -> BatchEnrichmentItem:
    """Run pipeline for one request; return BatchEnrichmentItem (success or error)."""
    trace_id = trace_id or uuid.uuid4().hex[:10]
    it = route_input(request)
    logger.info("[%s] enrich request received | input_type=%s company=%s", trace_id, it, (request.company_name or "").strip() or "-")
    if it == "company" and not (request.company_name or "").strip():
        return BatchEnrichmentItem(success=False, error="company_name is required for company input")
    if it == "visitor" and not request.visitor and not (request.company_name or "").strip():
        return BatchEnrichmentItem(success=False, error="visitor object or company_name required")
    try:
        result = run_pipeline(request, trace_id=trace_id)
        if not result:
            logger.error("[%s] pipeline returned empty output", trace_id)
            return BatchEnrichmentItem(success=False, error="Pipeline did not return output")
        logger.info(
            "[%s] enrich completed | company=%s intent_score=%s stage=%s",
            trace_id,
            result.get("company_name"),
            result.get("intent_score"),
            result.get("intent_stage"),
        )
        return BatchEnrichmentItem(success=True, data=AccountIntelligence.model_validate(result))
    except Exception as e:
        logger.exception("[%s] enrich failed", trace_id)
        return BatchEnrichmentItem(success=False, error=str(e))


@router.post("/enrich", response_model=AccountIntelligence)
async def enrich(request: EnrichmentRequest, save: bool = Query(True, description="Save result to storage")) -> AccountIntelligence:
    """
    Accept either visitor signal or company name/domain.
    Runs LangGraph pipeline: route -> parallel agents (enrich, web research, tech stack) -> reasoning -> output.
    Returns structured account intelligence with confidence.
    """
    trace_id = uuid.uuid4().hex[:10]
    item = _run_one(request, trace_id=trace_id)
    if not item.success:
        raise HTTPException(status_code=400 if "required" in (item.error or "") else 500, detail=item.error)
    assert item.data is not None
    
    # Save to storage
    if save:
        storage.save_enrichment(item.data.model_dump(), trace_id=trace_id)
    
    return item.data


@router.post("/enrich/batch", response_model=BatchEnrichmentResponse)
async def enrich_batch(body: BatchEnrichmentRequest, save: bool = Query(True, description="Save results to storage")) -> BatchEnrichmentResponse:
    """
    Enrich multiple companies or visitors. Max 50 items per request.
    Returns a result per item (success + data, or error).
    """
    batch_trace_id = uuid.uuid4().hex[:10]
    logger.info("[%s] batch request received | items=%d", batch_trace_id, len(body.items))
    results: list[BatchEnrichmentItem] = []
    for idx, req in enumerate(body.items, start=1):
        item = _run_one(req, trace_id=f"{batch_trace_id}-{idx}")
        results.append(item)
        
        # Save successful results to storage
        if save and item.success and item.data:
            storage.save_enrichment(item.data.model_dump(), trace_id=f"{batch_trace_id}-{idx}")
    
    succeeded = sum(1 for r in results if r.success)
    logger.info("[%s] batch completed | succeeded=%d failed=%d", batch_trace_id, succeeded, len(results) - succeeded)
    return BatchEnrichmentResponse(
        results=results,
        total=len(results),
        succeeded=succeeded,
        failed=len(results) - succeeded,
    )


# ============ Storage Endpoints ============

@router.get("/enrichments")
async def list_enrichments(limit: int = Query(50, ge=1, le=200)):
    """List recent enrichment results stored on disk."""
    return {
        "enrichments": storage.list_enrichments(limit=limit),
        "stats": storage.get_storage_stats(),
    }


@router.get("/enrichments/{filename}")
async def get_enrichment(filename: str):
    """Get a specific enrichment result by filename."""
    data = storage.get_enrichment(filename)
    if not data:
        raise HTTPException(status_code=404, detail="Enrichment not found")
    return data


@router.delete("/enrichments/{filename}")
async def delete_enrichment(filename: str):
    """Delete an enrichment result."""
    if storage.delete_enrichment(filename):
        return {"status": "deleted", "filename": filename}
    raise HTTPException(status_code=404, detail="Enrichment not found")


@router.get("/enrichments/search/{domain}")
async def search_enrichment_by_domain(domain: str):
    """Find the most recent enrichment for a domain."""
    data = storage.get_enrichment_by_domain(domain)
    if not data:
        raise HTTPException(status_code=404, detail=f"No enrichment found for domain: {domain}")
    return data
