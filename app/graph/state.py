"""LangGraph pipeline state. All values are JSON-serializable."""

from typing import TypedDict, Any


class PipelineState(TypedDict, total=False):
    """State passed through the enrichment graph. Each node returns a partial update."""

    _request: dict  # EnrichmentRequest as dict (injected at invoke)
    trace_id: str
    input_type: str  # "visitor" | "company"
    company_name: str
    domain: str | None
    visitor: dict | None  # VisitorSignal as dict
    resolved_company_name: str
    resolved_domain: str | None
    enrichment_result: dict  # CompanyProfile.model_dump()
    web_research_result: dict  # {"leadership": [...], "business_signals": [...]}
    tech_stack_result: list[str]
    reasoning_result: dict  # IntentPersonaSummaryAction.model_dump()
    final_output: dict  # AccountIntelligence.model_dump()
    errors: dict[str, str]  # node name -> error message
