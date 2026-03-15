"""Company enrichment: Apollo.io / Clearbit + Tavily fallback."""

import httpx
from app.config import config
from app.logging_utils import get_logger
from app.models.enrichment import CompanyProfile, EnrichedField
from app.agents.tavily_fallback import enrich_via_tavily

logger = get_logger("app.agents.enrichment")


def _apollo_enrich(domain: str | None, company_name: str) -> CompanyProfile | None:
    """Call Apollo.io organization enrichment API. Returns None if no key or failure."""
    if not config.APOLLO_API_KEY:
        return None
    
    api_key = config.APOLLO_API_KEY.strip().strip('"')
    
    if not domain:
        logger.debug("apollo skipped | domain required for enrichment")
        return None
    
    headers = {
        "Content-Type": "application/json",
        "Cache-Control": "no-cache",
        "x-api-key": api_key,
    }
    
    payload = {"domain": domain}
    
    url = "https://api.apollo.io/api/v1/organizations/enrich"
    
    try:
        with httpx.Client(timeout=15.0) as client:
            r = client.post(url, json=payload, headers=headers)
            
            if r.status_code != 200:
                logger.warning("apollo enrichment failed | status=%d response=%s", r.status_code, r.text[:300] if r.text else "")
                return None
            
            data = r.json()
            org = data.get("organization")
            if not org:
                logger.warning("apollo enrichment empty | no organization in response")
                return None
            
            logger.info("apollo enrichment success | domain=%s company=%s", domain, org.get("name", ""))
    except Exception as e:
        logger.warning("apollo enrichment error | error=%s", str(e))
        return None

    def ef(v: str | None, conf: str) -> EnrichedField | None:
        if not v or not str(v).strip():
            return None
        return EnrichedField(value=str(v).strip(), confidence=conf, source="apollo")

    name = org.get("name") or company_name
    
    employee_count = org.get("estimated_num_employees")
    company_size = None
    if employee_count:
        company_size = ef(f"{employee_count} employees", "high")
    
    hq_parts = []
    if org.get("city"):
        hq_parts.append(org["city"])
    if org.get("state"):
        hq_parts.append(org["state"])
    if org.get("country"):
        hq_parts.append(org["country"])
    headquarters = ef(", ".join(hq_parts), "high") if hq_parts else None
    
    industry = org.get("industry") or org.get("industry_tag_id")
    
    founded_year = org.get("founded_year")
    
    return CompanyProfile(
        company_name=EnrichedField(value=name, confidence="high", source="apollo"),
        domain=ef(org.get("primary_domain") or org.get("website_url", "").replace("https://", "").replace("http://", "").split("/")[0], "high"),
        website=ef(org.get("website_url"), "high"),
        industry=ef(industry, "high"),
        company_size=company_size,
        headquarters=headquarters,
        founding_year=ef(str(founded_year), "medium") if founded_year else None,
        description=ef(org.get("short_description") or org.get("seo_description"), "medium"),
    )


def _clearbit_enrich(domain: str | None, company_name: str) -> CompanyProfile | None:
    """Call Clearbit company API. Returns None if no key or failure."""
    if not config.CLEARBIT_API_KEY or not domain:
        return None
    url = "https://company.clearbit.com/v2/companies/find"
    params = {"domain": domain}
    headers = {"Authorization": f"Bearer {config.CLEARBIT_API_KEY}"}
    try:
        with httpx.Client(timeout=10.0) as client:
            r = client.get(url, params=params, headers=headers)
            if r.status_code != 200:
                return None
            data = r.json()
    except Exception:
        return None

    def ef(v: str | None, conf: str, src: str) -> EnrichedField | None:
        if not v or not str(v).strip():
            return None
        return EnrichedField(value=str(v).strip(), confidence=conf, source=src)

    name = data.get("name") or company_name
    return CompanyProfile(
        company_name=EnrichedField(value=name, confidence="high", source="clearbit"),
        domain=ef(data.get("domain"), "high", "clearbit"),
        website=ef(data.get("domain") and f"https://{data.get('domain')}", "high", "clearbit"),
        industry=ef(data.get("industry"), "high", "clearbit"),
        company_size=(
            ef(
                str(data["metrics"].get("employees", "")) + " employees" if data.get("metrics") and data["metrics"].get("employees") else None,
                "medium",
                "clearbit",
            )
            if data.get("metrics") and data["metrics"].get("employees")
            else None
        ),
        headquarters=_parse_hq(data),
        founding_year=ef(str(data.get("foundedYear")), "medium", "clearbit") if data.get("foundedYear") else None,
        description=ef(data.get("description"), "medium", "clearbit"),
    )


def _parse_hq(data: dict) -> EnrichedField | None:
    loc = data.get("location") or data.get("geo")
    if not loc:
        return None
    if isinstance(loc, str):
        return EnrichedField(value=loc, confidence="medium", source="clearbit")
    city = loc.get("city") or ""
    state = loc.get("state") or loc.get("stateCode") or ""
    country = loc.get("country") or loc.get("countryCode") or ""
    parts = [p for p in [city, state, country] if p]
    if not parts:
        return None
    return EnrichedField(value=", ".join(parts), confidence="medium", source="clearbit")


def enrich_company(company_name: str, domain: str | None = None) -> CompanyProfile:
    """
    Enrich company by name/domain. Tries Apollo.io first, then Clearbit, then Tavily + LLM fallback.
    Always returns a CompanyProfile with at least company_name.
    """
    # Normalize domain: strip protocol and path
    if domain:
        domain = domain.replace("https://", "").replace("http://", "").split("/")[0].strip().lower()
    if not domain and company_name and "." in company_name and " " not in company_name:
        maybe_domain = company_name.replace("https://", "").replace("http://", "").split("/")[0].strip().lower()
        if "." in maybe_domain:
            domain = maybe_domain

    # Try Apollo.io first (recommended - has free tier)
    profile = _apollo_enrich(domain, company_name)
    if profile is not None:
        logger.info("enrichment via apollo | company=%s domain=%s", company_name, domain or "-")
        return profile

    # Try Clearbit (legacy, now part of HubSpot)
    profile = _clearbit_enrich(domain, company_name)
    if profile is not None:
        logger.info("enrichment via clearbit | company=%s domain=%s", company_name, domain or "-")
        return profile

    # Fallback: Tavily search + LLM extraction. If it infers domain, retry Apollo/Clearbit.
    tavily_profile = enrich_via_tavily(company_name, domain)
    if tavily_profile is not None:
        inferred_domain = tavily_profile.domain.value if tavily_profile.domain else None
        if not domain and inferred_domain:
            # Try Apollo with inferred domain
            api_profile = _apollo_enrich(inferred_domain, company_name)
            if api_profile is None:
                api_profile = _clearbit_enrich(inferred_domain, company_name)
            if api_profile is not None:
                logger.info("enrichment via api+tavily merge | company=%s domain=%s", company_name, inferred_domain)
                return _merge_profiles(api_profile, tavily_profile)
        logger.info("enrichment via tavily fallback | company=%s domain=%s", company_name, inferred_domain or domain or "-")
        return tavily_profile

    # Minimal profile from input only
    logger.warning("enrichment minimal input fallback | company=%s domain=%s", company_name, domain or "-")
    return CompanyProfile(
        company_name=EnrichedField(value=company_name, confidence="low", source="input"),
        domain=EnrichedField(value=domain or "", confidence="low", source="input") if domain else None,
    )


def _merge_profiles(primary: CompanyProfile, secondary: CompanyProfile) -> CompanyProfile:
    """
    Merge two profiles by keeping primary fields and filling missing values from secondary.
    Intended for Clearbit(primary) + Tavily(secondary) merge.
    """

    return CompanyProfile(
        company_name=primary.company_name or secondary.company_name,
        website=primary.website or secondary.website,
        domain=primary.domain or secondary.domain,
        industry=primary.industry or secondary.industry,
        company_size=primary.company_size or secondary.company_size,
        headquarters=primary.headquarters or secondary.headquarters,
        founding_year=primary.founding_year or secondary.founding_year,
        description=primary.description or secondary.description,
    )
