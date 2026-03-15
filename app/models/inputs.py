"""Input models for visitor signals and company list."""

from typing import Optional

from pydantic import BaseModel, Field


class VisitorSignal(BaseModel):
    """Raw visitor activity from website analytics."""

    visitor_id: Optional[str] = None
    ip: Optional[str] = None
    pages_visited: list[str] = Field(default_factory=list)
    time_on_site: Optional[str] = None  # e.g. "3m 42s"
    visits_this_week: Optional[int] = None
    referral_source: Optional[str] = None
    device: Optional[str] = None
    location: Optional[str] = None
    visit_timestamps: list[str] = Field(default_factory=list)


class CompanyInput(BaseModel):
    """Minimal company input (name and optional domain)."""

    company_name: str
    domain: Optional[str] = None


class EnrichmentRequest(BaseModel):
    """API request: either visitor signal or company input."""

    visitor: Optional[VisitorSignal] = None
    company_name: Optional[str] = None
    domain: Optional[str] = None
