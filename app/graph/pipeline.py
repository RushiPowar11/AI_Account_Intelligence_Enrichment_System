"""LangGraph pipeline: route -> parallel agents -> reasoning -> final."""

from concurrent.futures import ThreadPoolExecutor
from time import perf_counter

from langgraph.graph import StateGraph, START, END

from app.graph.state import PipelineState
from app.logging_utils import get_logger
from app.router.input_router import route_input
from app.models.inputs import EnrichmentRequest, VisitorSignal
from app.agents.ip_resolver import resolve_ip
from app.agents.enrichment import enrich_company
from app.agents.web_research import run_web_research
from app.agents.tech_stack import run_tech_stack
from app.agents.reasoning import run_reasoning
from app.models.enrichment import CompanyProfile
from app.models.outputs import AccountIntelligence
from app.models.enrichment import EnrichedField

logger = get_logger("app.graph.pipeline")


def _norm_name(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum() or ch.isspace()).strip()


def _same_company_name(a: str | None, b: str | None) -> bool:
    if not a or not b:
        return False
    na = _norm_name(a)
    nb = _norm_name(b)
    if not na or not nb:
        return False
    if na == nb or na in nb or nb in na:
        return True
    a_tokens = {t for t in na.split() if len(t) > 2}
    b_tokens = {t for t in nb.split() if len(t) > 2}
    if not a_tokens or not b_tokens:
        return False
    overlap = len(a_tokens & b_tokens)
    return overlap >= max(1, min(len(a_tokens), len(b_tokens)) // 2)


def _route_node(state: PipelineState) -> PipelineState:
    """Normalize input: set company_name, domain, visitor, input_type. For visitor, resolve IP."""
    trace_id = state.get("trace_id", "no-trace")
    request = state.get("_request")
    if not request:
        logger.error("[%s] route failed: missing _request", trace_id)
        return {"errors": {"route": "Missing _request"}}
    req = EnrichmentRequest(**request) if isinstance(request, dict) else request
    input_type = route_input(req)
    company_name = (req.company_name or "").strip()
    domain = req.domain
    visitor_dict = req.visitor.model_dump() if req.visitor else None

    if input_type == "visitor" and req.visitor and (req.visitor.ip or req.visitor.pages_visited):
        name, dom = resolve_ip(req.visitor.ip)
        company_mismatch = bool(company_name) and not _same_company_name(company_name, name)
        chosen_company = company_name or name
        chosen_domain = domain
        if not chosen_domain and not company_mismatch:
            chosen_domain = dom
        if company_mismatch:
            logger.warning(
                "[%s] visitor/company mismatch | input_company=%s resolved_company=%s resolved_domain=%s; keeping input company context",
                trace_id,
                company_name,
                name,
                dom or "-",
            )
        logger.info(
            "[%s] route visitor | input_company=%s resolved_company=%s resolved_domain=%s",
            trace_id,
            company_name or "-",
            name,
            dom or "-",
        )
        return {
            "input_type": input_type,
            "company_name": chosen_company,
            "domain": chosen_domain,
            "visitor": visitor_dict,
            "resolved_company_name": name,
            "resolved_domain": dom,
        }
    logger.info("[%s] route company | company=%s domain=%s", trace_id, company_name or "Unknown Company", domain or "-")
    return {
        "input_type": input_type,
        "company_name": company_name or "Unknown Company",
        "domain": domain,
        "visitor": visitor_dict,
    }


def _parallel_agents_node(state: PipelineState) -> PipelineState:
    """Run enrichment first, then web research and tech stack in parallel."""
    trace_id = state.get("trace_id", "no-trace")
    company_name = state.get("company_name") or state.get("resolved_company_name") or "Unknown"
    domain = state.get("domain")
    errors: dict[str, str] = dict(state.get("errors") or {})
    started = perf_counter()

    try:
        profile = enrich_company(company_name, domain)
    except Exception as e:
        errors["enrichment"] = str(e)
        profile = CompanyProfile(company_name=EnrichedField(value=company_name, confidence="low", source="input"))

    resolved_domain = domain or (profile.domain.value if profile and profile.domain else None)
    web_result = {"leadership": [], "business_signals": []}
    tech_result: list[str] = []

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {
            "web_research": pool.submit(run_web_research, company_name, resolved_domain),
            "tech_stack": pool.submit(run_tech_stack, company_name, resolved_domain),
        }
        for key, future in futures.items():
            try:
                result = future.result()
                if key == "web_research":
                    web_result = result or web_result
                else:
                    tech_result = result or []
            except Exception as e:
                errors[key] = str(e)
    elapsed_ms = int((perf_counter() - started) * 1000)
    logger.info(
        "[%s] agents done | company=%s domain=%s web_leadership=%d tech_count=%d elapsed_ms=%d",
        trace_id,
        company_name,
        resolved_domain or "-",
        len(web_result.get("leadership") or []),
        len(tech_result),
        elapsed_ms,
    )

    return {
        "enrichment_result": profile.model_dump(),
        "web_research_result": web_result,
        "tech_stack_result": tech_result,
        "errors": errors,
    }


def _reasoning_node(state: PipelineState) -> PipelineState:
    """Run Gemini for intent, persona, summary, action."""
    trace_id = state.get("trace_id", "no-trace")
    enrichment_dict = state.get("enrichment_result")
    visitor_dict = state.get("visitor")
    if not enrichment_dict:
        logger.error("[%s] reasoning skipped: no enrichment_result", trace_id)
        return {"errors": {**(state.get("errors") or {}), "reasoning": "No enrichment result"}}
    profile = CompanyProfile.model_validate(enrichment_dict)
    visitor = VisitorSignal.model_validate(visitor_dict) if visitor_dict else None
    reasoning = run_reasoning(profile, visitor=visitor)
    logger.info(
        "[%s] reasoning done | intent=%.1f stage=%s persona=%s",
        trace_id,
        reasoning.intent_score,
        reasoning.intent_stage,
        reasoning.likely_persona,
    )
    return {"reasoning_result": reasoning.model_dump()}


def _final_node(state: PipelineState) -> PipelineState:
    """Build AccountIntelligence from state."""
    trace_id = state.get("trace_id", "no-trace")
    enrichment_dict = state.get("enrichment_result") or {}
    reasoning_dict = state.get("reasoning_result") or {}
    web = state.get("web_research_result") or {}
    tech = state.get("tech_stack_result") or []
    profile = CompanyProfile.model_validate(enrichment_dict) if enrichment_dict else None

    def v(ef: EnrichedField | None) -> str | None:
        return ef.value if ef else None

    fc = {}
    if profile:
        fc["company_name"] = profile.company_name.confidence
        if profile.domain:
            fc["domain"] = profile.domain.confidence
        if profile.industry:
            fc["industry"] = profile.industry.confidence
        if profile.company_size:
            fc["company_size"] = profile.company_size.confidence
        if profile.headquarters:
            fc["headquarters"] = profile.headquarters.confidence
        if profile.founding_year:
            fc["founding_year"] = profile.founding_year.confidence

    out = AccountIntelligence(
        company_name=profile.company_name.value if profile else "Unknown",
        website=v(profile.website) if profile else None,
        domain=v(profile.domain) if profile else None,
        industry=v(profile.industry) if profile else None,
        company_size=v(profile.company_size) if profile else None,
        headquarters=v(profile.headquarters) if profile else None,
        founding_year=v(profile.founding_year) if profile else None,
        likely_persona=reasoning_dict.get("likely_persona"),
        persona_confidence=reasoning_dict.get("persona_confidence_percent"),
        intent_score=reasoning_dict.get("intent_score"),
        intent_stage=reasoning_dict.get("intent_stage"),
        intent_justification=reasoning_dict.get("intent_justification"),
        ai_summary=reasoning_dict.get("summary"),
        recommended_sales_action=reasoning_dict.get("recommended_action"),
        action_steps=reasoning_dict.get("action_steps") or [],
        technology_stack=tech,
        leadership=web.get("leadership") or [],
        leadership_contacts=[
            {"name": c.get("name", ""), "title": c.get("title", ""), "email": c.get("email"), "confidence": c.get("confidence")}
            for c in web.get("leadership_enriched") or []
        ],
        email_pattern=web.get("email_pattern") or None,
        business_signals=web.get("business_signals") or [],
        key_signals_observed=reasoning_dict.get("key_signals_observed") or [],
        field_confidence=fc,
    )
    logger.info(
        "[%s] final output | company=%s domain=%s confidence_fields=%d",
        trace_id,
        out.company_name,
        out.domain or "-",
        len(out.field_confidence),
    )
    return {"final_output": out.model_dump()}


def build_graph():
    """Build and compile the pipeline graph."""
    builder = StateGraph(PipelineState)
    builder.add_node("route", _route_node)
    builder.add_node("parallel_agents", _parallel_agents_node)
    builder.add_node("reasoning", _reasoning_node)
    builder.add_node("final", _final_node)

    builder.add_edge(START, "route")
    builder.add_edge("route", "parallel_agents")
    builder.add_edge("parallel_agents", "reasoning")
    builder.add_edge("reasoning", "final")
    builder.add_edge("final", END)

    return builder.compile()


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_pipeline(request: EnrichmentRequest, trace_id: str = "no-trace") -> dict:
    """
    Run the enrichment pipeline and return the final AccountIntelligence as dict.
    """
    req_dict = request.model_dump(mode="json")
    initial: PipelineState = {
        "_request": req_dict,
        "trace_id": trace_id,
        "input_type": "company",
        "company_name": "",
        "domain": None,
        "visitor": None,
        "errors": {},
    }
    graph = get_graph()
    result = graph.invoke(initial)
    return result.get("final_output") or {}
