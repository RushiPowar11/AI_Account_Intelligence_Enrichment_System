"""Enrichment models with confidence layer."""

from typing import Optional

from pydantic import BaseModel, Field


class EnrichedField(BaseModel):
    """Single field with value, confidence, and source."""

    value: str
    confidence: str = Field(..., pattern="^(high|medium|low)$")
    source: str  # e.g. clearbit, tavily, builtwith, llm, input


class CompanyProfile(BaseModel):
    """Structured company profile; every field is enriched with confidence."""

    company_name: EnrichedField
    website: Optional[EnrichedField] = None
    domain: Optional[EnrichedField] = None
    industry: Optional[EnrichedField] = None
    company_size: Optional[EnrichedField] = None
    headquarters: Optional[EnrichedField] = None
    founding_year: Optional[EnrichedField] = None
    description: Optional[EnrichedField] = None
