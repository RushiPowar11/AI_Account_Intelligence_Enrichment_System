"""Detect input type and route to visitor or company path."""

from typing import Literal

from app.models.inputs import EnrichmentRequest

InputType = Literal["visitor", "company"]


def route_input(body: EnrichmentRequest) -> InputType:
    """
    Determine whether the request is a visitor signal or company-only input.
    If IP or pages_visited present, treat as visitor; else company.
    """
    if body.visitor and (body.visitor.ip or body.visitor.pages_visited):
        return "visitor"
    if body.company_name:
        return "company"
    # Default: treat as company with empty name so downstream can return error
    return "company"
