"""Structured output models for account intelligence."""

from typing import Optional

from pydantic import BaseModel, Field


class IntentScore(BaseModel):
    """Intent score 0-10 with stage and justification."""

    score: float = Field(..., ge=0, le=10)
    stage: str  # Awareness | Consideration | Evaluation | Decision
    justification: str


class PersonaInference(BaseModel):
    """Inferred visitor role with confidence."""

    likely_persona: str
    confidence_percent: float = Field(..., ge=0, le=100)


class AISummary(BaseModel):
    """3-sentence AI research summary."""

    summary: str


class RecommendedSalesAction(BaseModel):
    """Suggested next steps for sales."""

    action: str
    steps: list[str] = Field(default_factory=list)


class LeadershipContact(BaseModel):
    """Leadership contact with optional email from Hunter.io."""
    
    name: str
    title: str
    email: Optional[str] = None
    confidence: Optional[int] = None  # Email confidence 0-100


class AccountIntelligence(BaseModel):
    """Full sales-ready account intelligence output."""

    company_name: str
    website: Optional[str] = None
    domain: Optional[str] = None
    industry: Optional[str] = None
    company_size: Optional[str] = None
    headquarters: Optional[str] = None
    founding_year: Optional[str] = None
    likely_persona: Optional[str] = None
    persona_confidence: Optional[float] = None
    intent_score: Optional[float] = None
    intent_stage: Optional[str] = None
    intent_justification: Optional[str] = None
    ai_summary: Optional[str] = None
    recommended_sales_action: Optional[str] = None
    action_steps: list[str] = Field(default_factory=list)
    technology_stack: list[str] = Field(default_factory=list)
    leadership: list[str] = Field(default_factory=list)
    leadership_contacts: list[LeadershipContact] = Field(default_factory=list)  # Enriched with emails
    email_pattern: Optional[str] = None  # Company email pattern from Hunter
    business_signals: list[str] = Field(default_factory=list)
    key_signals_observed: list[str] = Field(default_factory=list)
    # Confidence metadata (optional, for debugging)
    field_confidence: dict[str, str] = Field(default_factory=dict)
