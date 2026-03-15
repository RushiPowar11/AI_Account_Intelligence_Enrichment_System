"""AI reasoning: intent score, persona, summary, recommended action (Gemini + heuristics)."""

from typing import Any, Optional

import json
import re

from pydantic import BaseModel, Field

from app.config import config
from app.agents.llm_client import generate_gemini_text
from app.logging_utils import get_logger
from app.models.enrichment import CompanyProfile
from app.models.inputs import VisitorSignal

GEMINI_MODEL_PRIMARY = "gemini-2.5-flash"
GEMINI_MODEL_FALLBACK = "gemini-2.0-flash"
INTENT_STAGES = ("Awareness", "Consideration", "Evaluation", "Decision")
logger = get_logger("app.agents.reasoning")


class IntentPersonaSummaryAction(BaseModel):
    """Structured output from reasoning agent."""

    intent_score: float = Field(..., ge=0, le=10)
    intent_stage: str = Field(..., description="One of: Awareness, Consideration, Evaluation, Decision")
    intent_justification: str = ""
    likely_persona: str = ""
    persona_confidence_percent: float = Field(..., ge=0, le=100)
    summary: str = Field(..., description="2-3 sentence executive summary")
    recommended_action: str = ""
    action_steps: list[str] = Field(default_factory=list)
    key_signals_observed: list[str] = Field(default_factory=list)


def _profile_to_text(profile: CompanyProfile) -> str:
    parts = [f"Company: {profile.company_name.value}"]
    if profile.domain:
        parts.append(f"Domain: {profile.domain.value}")
    if profile.website:
        parts.append(f"Website: {profile.website.value}")
    if profile.industry:
        parts.append(f"Industry: {profile.industry.value}")
    if profile.company_size:
        parts.append(f"Company size: {profile.company_size.value}")
    if profile.headquarters:
        parts.append(f"Headquarters: {profile.headquarters.value}")
    if profile.description:
        parts.append(f"Description: {profile.description.value}")
    return "\n".join(parts)


def _visitor_to_text(visitor: Optional[VisitorSignal]) -> str:
    if not visitor:
        return "No visitor signals (company-only input)."
    parts = []
    if visitor.pages_visited:
        parts.append(f"Pages visited: {', '.join(visitor.pages_visited)}")
    if visitor.time_on_site:
        parts.append(f"Time on site: {visitor.time_on_site}")
    if visitor.visits_this_week is not None:
        parts.append(f"Visits this week: {visitor.visits_this_week}")
    if visitor.referral_source:
        parts.append(f"Referral: {visitor.referral_source}")
    if visitor.device:
        parts.append(f"Device: {visitor.device}")
    return "\n".join(parts) if parts else "No behavioral signals."


def _get_gemini_api_key() -> str:
    """Support GOOGLE_API_KEY or GEMINI_API_KEY."""
    return config.GOOGLE_API_KEY or config.GEMINI_API_KEY or ""


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clean_text(value: Any, fallback: str = "") -> str:
    if value is None:
        return fallback
    txt = str(value).strip()
    return txt if txt else fallback


def _sanitize_description(desc: str | None) -> str:
    """Remove scraped junk from description (markdown, LinkedIn artifacts, etc.)."""
    if not desc:
        return ""
    text = desc.strip()
    lower = text.lower()
    
    immediate_reject = [
        "linkedin", "crunchbase", "bitscale", "total keywords", 
        "## overview", "# fello", "company profile -",
        "associated members", "see all employees",
    ]
    if any(junk in lower for junk in immediate_reject):
        return ""
    
    if text.count("#") >= 2 or text.count("|") >= 2:
        return ""
    
    junk_patterns = [
        r"###?\s*\w+",
        r"\|[^|]+\|",
        r"\d+\s*total\s*keywords",
        r"Company Size\s+\d+",
        r"Industry\s+\w+",
        r"See all employees",
        r"View profile",
        r"Follow\s*$",
        r"^\s*\|\s*",
        r"\.\.\.\.*\s*$",
        r"Read more",
        r"Learn more",
    ]
    for pattern in junk_patterns:
        text = re.sub(pattern, "", text, flags=re.I)
    
    text = re.sub(r"\s+", " ", text).strip()
    
    if len(text) < 25:
        return ""
    
    word_count = len(text.split())
    if word_count < 6:
        return ""
    
    return text


def _to_stage(value: Any, score: float) -> str:
    stage = _clean_text(value)
    if stage in INTENT_STAGES:
        return stage
    if score >= 8:
        return "Decision"
    if score >= 6:
        return "Evaluation"
    if score >= 3:
        return "Consideration"
    return "Awareness"


def _time_to_seconds(value: str | None) -> int:
    if not value:
        return 0
    text = value.strip().lower()
    total = 0
    m = re.search(r"(\d+)\s*h", text)
    if m:
        total += int(m.group(1)) * 3600
    m = re.search(r"(\d+)\s*m", text)
    if m:
        total += int(m.group(1)) * 60
    m = re.search(r"(\d+)\s*s", text)
    if m:
        total += int(m.group(1))
    if total == 0 and text.isdigit():
        total = int(text)
    return total


def _heuristic_reasoning(profile: CompanyProfile, visitor: Optional[VisitorSignal]) -> IntentPersonaSummaryAction:
    score = 1.2
    stage = "Awareness"
    persona = "Revenue Operations Leader"
    confidence = 42.0
    signals: list[str] = []
    justification = "Company-level enrichment available; no high-intent behavior detected yet."

    pages = [p.lower() for p in (visitor.pages_visited if visitor else [])]
    dwell_seconds = _time_to_seconds(visitor.time_on_site if visitor else None)
    weekly_visits = visitor.visits_this_week if visitor and visitor.visits_this_week is not None else 0
    referral = (visitor.referral_source or "").lower() if visitor else ""

    if visitor:
        if pages:
            if any("pricing" in p for p in pages):
                score += 2.4
                signals.append("Visited pricing page")
            if any("case" in p for p in pages):
                score += 1.6
                signals.append("Viewed case studies")
            if any("demo" in p or "contact" in p for p in pages):
                score += 1.8
                signals.append("Demo/contact intent signal")
            if any("docs" in p or "api" in p for p in pages):
                score += 1.4
                signals.append("Explored docs/API")

        if dwell_seconds >= 180:
            score += 1.2
            signals.append("High dwell time")
        elif dwell_seconds >= 90:
            score += 0.7

        if weekly_visits >= 3:
            score += 1.6
            signals.append("Multiple repeat visits")
        elif weekly_visits == 2:
            score += 0.8

        if referral in {"google", "bing", "linkedin", "g2", "capterra"}:
            score += 0.5
            signals.append("High-intent referral source")

        if any("docs" in p or "api" in p for p in pages):
            persona = "Technical Evaluator / Solutions Engineer"
            confidence = 70.0
        elif any("pricing" in p for p in pages) and any("case" in p for p in pages):
            persona = "Head of Sales Operations / RevOps"
            confidence = 76.0
        elif any("blog" in p or "resource" in p for p in pages):
            persona = "Research / Demand Generation Manager"
            confidence = 60.0
        elif pages:
            persona = "Sales or GTM Decision Influencer"
            confidence = 58.0

        score = _clamp(score, 0, 10)
        stage = _to_stage("", score)
        if signals:
            justification = "Behavioral signals indicate active evaluation of sales tooling."
        else:
            justification = "Visitor data exists but contains limited intent indicators."
    else:
        score = _clamp(score, 0, 3)
        stage = "Awareness"
        signals = ["No visitor behavior provided"]

    company_context = ""
    if profile.industry:
        company_context += f" in the {profile.industry.value} space"
    if profile.headquarters:
        company_context += f", operating from {profile.headquarters.value}"
    
    clean_desc = _sanitize_description(profile.description.value if profile.description else None)
    if clean_desc and len(clean_desc) > 30:
        desc_snippet = clean_desc[:150].strip()
        if len(clean_desc) > 150 and not desc_snippet.endswith("."):
            last_space = desc_snippet.rfind(" ")
            if last_space > 100:
                desc_snippet = desc_snippet[:last_space] + "..."
        company_context = f". {desc_snippet}"
    
    summary = (
        f"{profile.company_name.value} is a company{company_context}. "
        f"Intent is currently {stage.lower()} with score {score:.1f}/10 based on observed behavioral signals."
    )
    if not visitor:
        summary = (
            f"{profile.company_name.value}{company_context}. "
            "No visitor behavior available; intent estimated from account context only and should be validated with live engagement data."
        )

    recommended_action = "Prioritize account research and run personalized outreach to identified decision makers."
    if stage in {"Evaluation", "Decision"}:
        recommended_action = "Launch fast follow-up with tailored proof points and book a discovery call."

    action_steps = [
        "Identify VP Sales, RevOps, or Growth leaders in the account.",
        "Send personalized outreach tied to the pages/topics they engaged with.",
        "Add account to high-priority sequence with 7-day follow-up cadence.",
    ]
    if stage in {"Evaluation", "Decision"}:
        action_steps = [
            "Route account to AE immediately and prepare a custom value narrative.",
            "Send use-case specific proof points and relevant case studies.",
            "Propose a 30-minute discovery or demo session within 48 hours.",
        ]

    return IntentPersonaSummaryAction(
        intent_score=score,
        intent_stage=stage,
        intent_justification=justification,
        likely_persona=persona,
        persona_confidence_percent=confidence,
        summary=summary,
        recommended_action=recommended_action,
        action_steps=action_steps,
        key_signals_observed=signals[:6],
    )


def _extract_json_blob(text: str) -> dict[str, Any] | None:
    clean = text.strip()
    if not clean:
        return None
    try:
        parsed = json.loads(clean)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    first = clean.find("{")
    last = clean.rfind("}")
    if first == -1 or last == -1 or first >= last:
        return None
    try:
        parsed = json.loads(clean[first : last + 1])
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def _to_str_list(value: Any) -> list[str]:
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            item_txt = _clean_text(item)
            if item_txt:
                out.append(item_txt)
        return out
    if isinstance(value, str):
        return [x.strip("- ").strip() for x in value.split("\n") if x.strip()]
    return []


def _build_prompt(profile: CompanyProfile, visitor: Optional[VisitorSignal], heuristic: IntentPersonaSummaryAction) -> str:
    schema = {
        "intent_score": "number 0-10",
        "intent_stage": "Awareness|Consideration|Evaluation|Decision",
        "intent_justification": "string",
        "likely_persona": "string",
        "persona_confidence_percent": "number 0-100",
        "summary": "2-3 sentence string describing the company and intent",
        "recommended_action": "one sentence string",
        "action_steps": "array of 2-4 strings",
        "key_signals_observed": "array of strings",
    }
    return (
        "You are a sales intelligence analyst. Return ONLY valid JSON and no markdown.\n"
        f"JSON schema keys required: {json.dumps(schema)}\n\n"
        "IMPORTANT RULES:\n"
        "- Use visitor behavior heavily for intent/persona scoring when available.\n"
        "- If visitor behavior is absent, keep score between 0 and 3.\n"
        "- DO NOT invent facts about the company not present in the profile.\n"
        "- The summary should mention what the company does (from description/industry), not just 'enrichment available'.\n"
        "- Make the summary specific and actionable for sales teams.\n\n"
        "Company profile:\n"
        f"{_profile_to_text(profile)}\n\n"
        "Visitor signals:\n"
        f"{_visitor_to_text(visitor)}\n\n"
        "Heuristic baseline (use as guardrails, not mandatory):\n"
        f"{heuristic.model_dump_json()}"
    )


def _llm_reasoning(prompt: str, api_key: str) -> str | None:
    return generate_gemini_text(
        prompt=prompt,
        api_key=api_key,
        model_candidates=(GEMINI_MODEL_PRIMARY, GEMINI_MODEL_FALLBACK),
        max_output_tokens=900,
        temperature=0.2,
    )


def _is_garbage_summary(summary: str) -> bool:
    """Check if summary contains scraped junk that shouldn't be shown."""
    if not summary:
        return True
    lower = summary.lower()
    
    garbage_indicators = [
        "###",
        "##",
        "# ",
        "| linkedin",
        "| company profile",
        "crunchbase",
        "bitscale",
        "associated members",
        "see all employees",
        "view profile",
        "total keywords",
        "keywords #",
        "company size 51-200",
        "company size 201-500",
        "company size 11-50",
        "industry software",
        "## overview",
        "....",
        "read more",
        "click here",
        "learn more about",
        "visit website",
        "follow us",
        "connect with",
    ]
    
    if any(ind in lower for ind in garbage_indicators):
        return True
    
    if re.search(r"\d+\s*total\s*keywords", lower):
        return True
    if re.search(r"#\s*[a-z]+\s*##", lower):
        return True
    if summary.count("|") >= 2:
        return True
    if summary.count("#") >= 2:
        return True
    
    return False


def _merge_reasoning(
    llm_data: dict[str, Any] | None,
    profile: CompanyProfile,
    visitor: Optional[VisitorSignal],
    heuristic: IntentPersonaSummaryAction,
) -> IntentPersonaSummaryAction:
    data = llm_data or {}
    score = _clamp(_to_float(data.get("intent_score"), heuristic.intent_score), 0, 10)
    stage = _to_stage(data.get("intent_stage"), score)
    intent_justification = _clean_text(data.get("intent_justification"), heuristic.intent_justification)
    persona = _clean_text(data.get("likely_persona"), heuristic.likely_persona)[:80]
    persona_conf = _clamp(_to_float(data.get("persona_confidence_percent"), heuristic.persona_confidence_percent), 0, 100)
    
    llm_summary = _clean_text(data.get("summary"), "")
    if _is_garbage_summary(llm_summary):
        summary = heuristic.summary
    else:
        summary = llm_summary
    
    recommended_action = _clean_text(data.get("recommended_action"), heuristic.recommended_action)

    action_steps = _to_str_list(data.get("action_steps"))[:5] or heuristic.action_steps
    key_signals = _to_str_list(data.get("key_signals_observed"))[:8] or heuristic.key_signals_observed

    # Keep output realistic when no visitor signal exists.
    if not visitor:
        score = _clamp(score, 0, 3)
        stage = "Awareness" if score < 3 else "Consideration"

    return IntentPersonaSummaryAction(
        intent_score=score,
        intent_stage=stage,
        intent_justification=intent_justification,
        likely_persona=persona,
        persona_confidence_percent=persona_conf,
        summary=summary,
        recommended_action=recommended_action,
        action_steps=action_steps,
        key_signals_observed=key_signals,
    )


def run_reasoning(
    profile: CompanyProfile,
    visitor: Optional[VisitorSignal] = None,
) -> IntentPersonaSummaryAction:
    """
    Produce intent score, persona, summary, and recommended action.
    Uses Gemini JSON output with deterministic visitor-signal fallback.
    """
    heuristic = _heuristic_reasoning(profile, visitor)
    api_key = _get_gemini_api_key()
    if not api_key:
        logger.info("reasoning heuristic only | company=%s (no API key)", profile.company_name.value)
        return heuristic

    prompt = _build_prompt(profile, visitor, heuristic)
    llm_text = _llm_reasoning(prompt, api_key)
    llm_data = _extract_json_blob(llm_text or "")
    out = _merge_reasoning(llm_data, profile, visitor, heuristic)
    logger.info(
        "reasoning complete | company=%s intent=%.1f stage=%s persona=%s llm_json=%s",
        profile.company_name.value,
        out.intent_score,
        out.intent_stage,
        out.likely_persona,
        bool(llm_data),
    )
    return out
