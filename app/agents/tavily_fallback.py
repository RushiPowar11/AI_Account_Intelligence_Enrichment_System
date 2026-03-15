"""Tavily search + LLM fallback when Clearbit is unavailable."""

from typing import Any

import json
import re
from urllib.parse import urlparse

import httpx

from app.config import config
from app.agents.llm_client import generate_gemini_text
from app.logging_utils import get_logger
from app.models.enrichment import CompanyProfile, EnrichedField

GEMINI_MODEL_PRIMARY = "gemini-2.5-flash"
GEMINI_MODEL_FALLBACK = "gemini-2.0-flash"

_DOMAIN_BLOCKLIST = {
    "linkedin.com",
    "facebook.com",
    "x.com",
    "twitter.com",
    "instagram.com",
    "youtube.com",
    "wikipedia.org",
    "crunchbase.com",
    "zoominfo.com",
    "glassdoor.com",
    "pitchbook.com",
}

_INDUSTRY_HINTS = {
    "home services": "Home Services",
    "on-demand services": "Home Services",
    "beauty services": "Home Services",
    "cleaning services": "Home Services",
    "plumbing": "Home Services",
    "home repair": "Home Services",
    "mortgage": "Mortgage Lending",
    "real estate": "Real Estate",
    "property listing": "Real Estate",
    "property management": "Property Management",
    "fintech": "FinTech",
    "banking": "Banking",
    "insurance": "Insurance",
    "saas": "SaaS",
    "software platform": "SaaS",
    "gamification": "SaaS",
    "rewards platform": "SaaS",
    "engagement platform": "SaaS",
    "identity": "Identity & Access Management",
    "cybersecurity": "Cybersecurity",
    "healthcare": "Healthcare",
    "ecommerce": "E-commerce",
    "logistics": "Logistics",
    "edtech": "EdTech",
}

_INDUSTRY_KEYWORDS: dict[str, list[str]] = {
    "Home Services": [
        "home services",
        "on-demand services",
        "beauty services",
        "salon at home",
        "cleaning services",
        "home cleaning",
        "plumbing",
        "electrician",
        "carpenter",
        "home repair",
        "appliance repair",
        "pest control",
        "spa at home",
        "massage at home",
        "handyman",
        "home maintenance",
    ],
    "FinTech": [
        "fintech",
        "payments",
        "payment gateway",
        "payment orchestration",
        "payment infrastructure",
        "upi",
        "merchant payments",
        "card network",
        "transaction",
        "checkout",
        "banking infrastructure",
    ],
    "Mortgage Lending": [
        "mortgage",
        "home loan",
        "lending",
        "loan servicing",
    ],
    "Real Estate": [
        "real estate",
        "property listing",
        "brokerage",
        "realtor",
    ],
    "Property Management": [
        "property management",
        "tenant",
        "lease management",
    ],
    "Banking": [
        "retail banking",
        "banking",
        "commercial bank",
    ],
    "Insurance": [
        "insurance policy",
        "insurtech",
        "policyholder",
        "insurance premium",
        "insurance claim",
    ],
    "SaaS": [
        "software as a service",
        "saas",
        "subscription software",
        "software platform",
        "cloud software",
        "b2b software",
        "enterprise software",
        "gamification",
        "gamification platform",
        "rewards platform",
        "engagement platform",
        "loyalty platform",
        "retention platform",
        "customer engagement",
    ],
    "Identity & Access Management": [
        "identity and access",
        "single sign-on",
        "mfa",
        "authentication",
    ],
    "Cybersecurity": [
        "cybersecurity",
        "security platform",
        "threat detection",
    ],
    "Healthcare": [
        "healthcare",
        "hospital",
        "clinical",
    ],
    "E-commerce": [
        "e-commerce",
        "online retail",
        "marketplace",
    ],
    "Logistics": [
        "logistics",
        "supply chain",
        "freight",
        "shipping",
        "delivery",
        "warehouse",
    ],
    "EdTech": [
        "edtech",
        "online learning",
        "e-learning",
        "education platform",
        "online courses",
    ],
}

logger = get_logger("app.agents.tavily_fallback")


def _normalize_domain(value: str | None) -> str | None:
    if not value:
        return None
    raw = value.strip().lower()
    if not raw:
        return None
    if "://" not in raw:
        raw = f"https://{raw}"
    try:
        host = urlparse(raw).netloc.strip().lower()
    except Exception:
        return None
    if host.startswith("www."):
        host = host[4:]
    if ":" in host:
        host = host.split(":", 1)[0]
    if not host or "." not in host:
        return None
    return host


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize_founding_year(value: str | None) -> str:
    text = _clean_text(value)
    m = re.search(r"\b(18\d{2}|19\d{2}|20\d{2})\b", text)
    if not m:
        return ""
    year = int(m.group(1))
    return str(year) if 1800 <= year <= 2100 else ""


def _normalize_company_size(value: str | None, founding_year: str | None = None) -> str:
    text = _clean_text(value).lower()
    if not text:
        return ""
    has_employee_keyword = bool(re.search(r"\b(employees?|staff|people|workforce|headcount)\b", text))
    range_match = re.search(r"(\d[\d,]{1,6})\s*(?:-|to)\s*(\d[\d,]{1,6})\s*(?:employees?|staff|people|workforce|headcount)?", text)
    if range_match and has_employee_keyword:
        lo = int(range_match.group(1).replace(",", ""))
        hi = int(range_match.group(2).replace(",", ""))
        if lo < 5 or hi > 500000 or lo >= hi:
            return ""
        return f"{lo}-{hi} employees"
    m = re.search(r"(\d[\d,]{0,6})(?:\+)?", text)
    if not m:
        return ""
    number_txt = m.group(1).replace(",", "")
    if not number_txt.isdigit():
        return ""
    number = int(number_txt)
    if not has_employee_keyword and "+" not in text:
        if number_txt.isdigit() and number >= 100:
            if 1800 <= number <= 2100:
                return ""
            return f"{number} employees"
        return ""
    if founding_year and number_txt == founding_year:
        return ""
    if 1800 <= number <= 2100 and "founded" in text:
        return ""
    if number < 5 or number > 500000:
        return ""
    return f"{number} employees"


def _cross_validate_company_size(
    llm_size: str,
    regex_size: str,
    snippets: str,
) -> str:
    """Cross-validate company size from multiple extractions and snippets."""
    sizes: list[int] = []
    
    for size_str in [llm_size, regex_size]:
        m = re.search(r"(\d[\d,]+)", size_str or "")
        if m:
            num = int(m.group(1).replace(",", ""))
            if 10 <= num <= 500000:
                sizes.append(num)
    
    snippet_sizes: list[int] = []
    patterns = [
        r"(\d[\d,]{1,6})\s*(?:\+\s*)?employees",
        r"company\s+size[:\s]+(\d[\d,]{1,6})",
        r"headcount[:\s]+(\d[\d,]{1,6})",
        r"team\s+of\s+(\d[\d,]{1,6})",
    ]
    for pat in patterns:
        for m in re.finditer(pat, snippets.lower()):
            num = int(m.group(1).replace(",", ""))
            if 10 <= num <= 500000:
                snippet_sizes.append(num)
    
    sizes.extend(snippet_sizes)
    
    if not sizes:
        return llm_size or regex_size or ""
    
    if len(sizes) == 1:
        return f"{sizes[0]} employees"
    
    median = sorted(sizes)[len(sizes) // 2]
    
    valid_sizes = [s for s in sizes if 0.3 <= s / median <= 3.0]
    
    if valid_sizes:
        best = int(sum(valid_sizes) / len(valid_sizes))
        return f"{best} employees"
    
    return f"{median} employees"


def _normalize_headquarters(value: str | None) -> str:
    text = _clean_text(value)
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^(?:location|hq|headquarters?)\s*[:\-]?\s*", "", text, flags=re.I)
    text = re.sub(r"\b(?:at|in)\s+([A-Z])", r"\1", text)
    # Trim known noisy suffix patterns often seen in snippets.
    text = re.split(r"\b(?:powered by|offers|provides|is a|with|and)\b", text, maxsplit=1, flags=re.I)[0].strip(" ,.-")
    parts = [p.strip(" ,.-") for p in text.split(",") if p.strip(" ,.-")]
    if not parts:
        return ""
    # Keep concise location-like prefix.
    if len(parts) >= 2:
        text = ", ".join(parts[:3])
    else:
        text = parts[0]
    # Reject if it still looks like a sentence.
    if len(text.split()) > 8:
        return ""
    return text


def _looks_like_region(part: str) -> bool:
    p = part.strip().lower()
    if not p:
        return True
    region_like = {
        "karnataka",
        "tamil nadu",
        "maharashtra",
        "delhi",
        "gujarat",
        "telangana",
        "andhra pradesh",
        "uttar pradesh",
        "haryana",
        "new york",
        "california",
        "texas",
        "india",
        "united states",
        "usa",
        "uk",
        "united kingdom",
        "singapore",
        "uae",
    }
    if p in region_like:
        return True
    if p.endswith(" state") or p.endswith(" region") or p.endswith(" province"):
        return True
    return False


def _is_holding_company_location(location: str, snippets_lower: str) -> bool:
    """Check if a location is likely a holding company registration, not operational HQ."""
    loc_lower = location.lower()
    holding_indicators = [
        "registered in " + loc_lower,
        "incorporated in " + loc_lower,
        "holding company in " + loc_lower,
        loc_lower + " holding",
        "registered office",
    ]
    operational_indicators = [
        "headquartered in " + loc_lower,
        "founded in " + loc_lower,
        "started in " + loc_lower,
        "based in " + loc_lower,
        "operates from " + loc_lower,
    ]
    holding_score = sum(1 for ind in holding_indicators if ind in snippets_lower)
    operational_score = sum(1 for ind in operational_indicators if ind in snippets_lower)
    return holding_score > operational_score and holding_score > 0


def _hq_candidate_quality(location: str) -> int:
    text = _normalize_headquarters(location)
    if not text:
        return 0
    parts = [p.strip() for p in text.split(",") if p.strip()]
    score = 1
    if len(parts) >= 2:
        score += 2
    if len(parts) >= 3:
        score += 1
    # City-like first token gives better HQ quality than just region/country.
    first = parts[0] if parts else ""
    if " " in first and len(first.split()) <= 2:
        score += 1
    if _looks_like_region(first):
        score -= 2
    else:
        score += 2
    return score


def _extract_hq_candidates(text: str) -> list[tuple[str, int]]:
    candidates: list[tuple[str, int]] = []
    if not text:
        return candidates

    patterns = [
        (r"(?:headquartered in|hq in|head office in)\s+([A-Za-z .,-]{3,80})(?:[.;]|$)", 10),
        (r"(?:based in)\s+([A-Za-z .,-]{3,70})(?:[.;]|$)", 7),
        (r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?),\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?),\s*(India|United States|USA|UK|United Kingdom|Singapore|UAE|Germany|France|Canada|Australia)\b", 6),
        (r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?),\s*(India|United States|USA|UK|United Kingdom|Singapore|UAE|Germany|France|Canada|Australia)\b", 4),
    ]

    for pattern, base in patterns:
        for m in re.finditer(pattern, text, re.I):
            groups = [g for g in m.groups() if g]
            raw = ", ".join(groups) if len(groups) > 1 else groups[0]
            cleaned = _normalize_headquarters(raw)
            if cleaned:
                candidates.append((cleaned, base + _hq_candidate_quality(cleaned)))
    return candidates


def _infer_headquarters(
    ranked_results: list[dict[str, Any]],
    company_name: str,
    resolved_domain: str | None,
    llm_hq: str,
    regex_hq: str,
) -> str:
    hq_scores: dict[str, int] = {}
    all_snippets_lower = " ".join(
        f"{_clean_text(row.get('title'))} {_clean_text(row.get('content'))}".lower()
        for row in ranked_results[:12]
    )
    
    for row in ranked_results[:12]:
        url = _clean_text(row.get("url"))
        title = _clean_text(row.get("title"))
        content = _clean_text(row.get("content"))
        blob = f"{title}. {content}"
        row_score = _result_relevance(row, company_name)
        lower_blob = blob.lower()
        lower_url = url.lower()

        if "headquarter" in lower_blob or "head office" in lower_blob or "hq" in lower_blob:
            row_score += 8
        if "based in" in lower_blob:
            row_score += 3
        if "registered office" in lower_blob or "privacy policy" in lower_url:
            row_score -= 10
        if "regional headquarters" in lower_blob or "middle east headquarters" in lower_blob:
            row_score -= 6
        if "holding company" in lower_blob or "registered in" in lower_blob:
            row_score -= 8
        if "corporate office" in lower_blob and "operational" not in lower_blob:
            row_score -= 4
        if "founded in" in lower_blob or "started in" in lower_blob:
            row_score += 5
        if resolved_domain and _normalize_domain(url) == resolved_domain:
            row_score += 8

        for location, bonus in _extract_hq_candidates(blob):
            total = row_score + bonus
            if _is_holding_company_location(location, all_snippets_lower):
                total -= 15
            hq_scores[location] = hq_scores.get(location, 0) + total

    for fallback in (llm_hq, regex_hq):
        normalized = _normalize_headquarters(fallback)
        if normalized:
            base_score = 3 + _hq_candidate_quality(normalized)
            if _is_holding_company_location(normalized, all_snippets_lower):
                base_score -= 10
            hq_scores[normalized] = hq_scores.get(normalized, 0) + base_score

    if not hq_scores:
        return ""

    # Prefer higher score, then better location quality.
    best = sorted(hq_scores.items(), key=lambda x: (x[1], _hq_candidate_quality(x[0])), reverse=True)[0][0]
    return best


def _extract_hq_from_official_domain(domain: str | None) -> str:
    host = _normalize_domain(domain)
    if not host:
        return ""
    urls = [
        f"https://{host}/about",
        f"https://www.{host}/about",
        f"https://{host}",
        f"https://www.{host}",
    ]
    patterns = [
        r"headquartered in ([A-Za-z .'-]{2,50})(?:,|[.;<]|$)",
        r"hq(?:uartered)? in ([A-Za-z .'-]{2,50})(?:,|[.;<]|$)",
        r"based in ([A-Za-z .'-]{2,50})(?:,|[.;<]|$)",
        r"headquartered in ([A-Za-z .,-]{3,90})(?:[.;<]|$)",
        r"based in ([A-Za-z .,-]{3,90})(?:[.;<]|$)",
    ]
    try:
        with httpx.Client(timeout=8.0, follow_redirects=True) as client:
            for url in urls:
                try:
                    resp = client.get(url, headers={"User-Agent": "Mozilla/5.0"})
                    if resp.status_code >= 400:
                        continue
                    html = (resp.text or "")
                    compact = re.sub(r"\s+", " ", html)
                    for pattern in patterns:
                        m = re.search(pattern, compact, re.I)
                        if m:
                            hq = _normalize_headquarters(m.group(1))
                            if hq:
                                if "," not in hq and "india" in compact.lower():
                                    hq = f"{hq}, India"
                                return hq
                except Exception:
                    continue
    except Exception:
        return ""
    return ""


def _canonicalize_domain(domain: str | None) -> str | None:
    host = _normalize_domain(domain)
    if not host:
        return None
    try:
        with httpx.Client(timeout=8.0, follow_redirects=True) as client:
            resp = client.get(f"https://{host}", headers={"User-Agent": "Mozilla/5.0"})
            final = _normalize_domain(str(resp.url))
            return final or host
    except Exception:
        return host


def _extract_json(text: str) -> dict[str, Any] | None:
    clean = (text or "").strip()
    if not clean:
        return None
    try:
        out = json.loads(clean)
        return out if isinstance(out, dict) else None
    except json.JSONDecodeError:
        pass
    s = clean.find("{")
    e = clean.rfind("}")
    if s == -1 or e == -1 or s >= e:
        return None
    try:
        out = json.loads(clean[s : e + 1])
        return out if isinstance(out, dict) else None
    except json.JSONDecodeError:
        return None


def _result_relevance(row: dict[str, Any], company_name: str) -> int:
    text = " ".join(
        [
            _clean_text(row.get("title")),
            _clean_text(row.get("content")),
            _clean_text(row.get("url")),
        ]
    ).lower()
    score = 0
    for token, weight in (
        ("employee", 4),
        ("headcount", 4),
        ("team size", 3),
        ("linkedin", 3),
        ("founded", 3),
        ("headquarter", 3),
        ("company profile", 2),
        ("industry", 2),
    ):
        if token in text:
            score += weight
    for part in re.findall(r"[A-Za-z0-9]+", company_name.lower()):
        if len(part) > 2 and part in text:
            score += 1
    return score


def _guess_domain(company_name: str, results: list[dict[str, Any]]) -> str | None:
    name_parts = [p.lower() for p in re.findall(r"[a-zA-Z0-9]+", company_name) if len(p) > 2]
    candidates: dict[str, int] = {}
    for row in results:
        url = row.get("url") or ""
        host = _normalize_domain(url)
        if not host:
            continue
        if any(host.endswith(bad) for bad in _DOMAIN_BLOCKLIST):
            continue
        score = 1
        if host.split(".")[0] in name_parts:
            score += 3
        if any(part in host for part in name_parts):
            score += 1
        candidates[host] = candidates.get(host, 0) + score
    if not candidates:
        return None
    return sorted(candidates.items(), key=lambda x: x[1], reverse=True)[0][0]


def _regex_structured(snippets: str) -> dict[str, str]:
    out: dict[str, str] = {}
    text = snippets or ""
    lower = text.lower()

    for hint, industry in _INDUSTRY_HINTS.items():
        if hint in lower:
            out["industry"] = industry
            break

    m = re.search(r"(?:headquartered in|based in)\s+([A-Za-z .,-]{3,80})(?:[.;]|$)", text, re.I)
    if m:
        out["headquarters"] = _normalize_headquarters(m.group(1))
    if not out.get("headquarters"):
        m = re.search(
            r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?),\s*(India|United States|USA|UK|United Kingdom|Singapore|UAE|Germany|France|Canada|Australia)\b",
            text,
        )
        if m:
            out["headquarters"] = _normalize_headquarters(f"{m.group(1)}, {m.group(2)}")

    m = re.search(r"(?:total\s+employees|employees?)\s*[:\-]?\s*(\d[\d,]{1,6})\b", text, re.I)
    if m:
        out["company_size"] = _normalize_company_size(m.group(1))
    if not out.get("company_size"):
        m = re.search(r"(?:company\s+size)\s*[:\-]?\s*(\d[\d,]{1,6}\s*(?:-|to)\s*\d[\d,]{1,6})\s*employees?", text, re.I)
    if m:
        out["company_size"] = _normalize_company_size(m.group(1) + " employees")
    if not out.get("company_size"):
        m = re.search(r"(\d[\d,]{1,6}\s*(?:-|to)\s*\d[\d,]{1,6})\s+(?:employees|employee|staff|people|workforce|headcount)", text, re.I)
    if m:
        out["company_size"] = _normalize_company_size(m.group(1) + " employees")
    if not out.get("company_size"):
        m = re.search(r"(\d[\d,]{1,6}\+?)\s+(?:employees|employee|staff|people|workforce|headcount)", text, re.I)
    if m:
        out["company_size"] = _normalize_company_size(m.group(1))

    m = re.search(r"(?:founded in|founded|since)\s+(19\d{2}|20\d{2})", text, re.I)
    if m:
        out["founding_year"] = _normalize_founding_year(m.group(1))

    summary = re.sub(r"\s+", " ", text).strip()
    if summary:
        out["description"] = summary[:420]
    return out


def _infer_industry(
    ranked_results: list[dict[str, Any]],
    company_name: str,
    resolved_domain: str | None,
    llm_industry: str,
    regex_industry: str,
) -> tuple[str, str]:
    """
    Infer industry using weighted evidence from ranked search results.
    Returns (industry, confidence).
    """
    scores: dict[str, int] = {industry: 0 for industry in _INDUSTRY_KEYWORDS}

    for row in ranked_results[:12]:
        title = _clean_text(row.get("title")).lower()
        content = _clean_text(row.get("content")).lower()
        url = _clean_text(row.get("url"))
        text = f"{title}\n{content}"

        base = _result_relevance(row, company_name)
        if resolved_domain and _normalize_domain(url) == resolved_domain:
            base += 10

        for industry, keywords in _INDUSTRY_KEYWORDS.items():
            hit_count = 0
            for kw in keywords:
                if kw in text:
                    hit_count += 1
            if hit_count:
                scores[industry] += base + (hit_count * 3)

    # LLM and regex hints act as priors if valid.
    if llm_industry in scores:
        scores[llm_industry] += 12
    if regex_industry in scores:
        scores[regex_industry] += 6

    best_industry, best_score = max(scores.items(), key=lambda x: x[1])
    second_score = sorted(scores.values(), reverse=True)[1] if len(scores) > 1 else 0

    if best_score <= 0:
        fallback = llm_industry or regex_industry
        if fallback:
            return fallback, "low"
        return "", "low"

    gap = best_score - second_score
    confidence = "high" if gap >= 12 else "medium" if gap >= 5 else "low"
    return best_industry, confidence


_VALID_INDUSTRIES = [
    "Home Services", "FinTech", "Mortgage Lending", "Real Estate", "Property Management",
    "Banking", "Insurance", "SaaS", "Identity & Access Management", "Cybersecurity",
    "Healthcare", "E-commerce", "Logistics", "EdTech", "Software Development", 
    "Gamification", "MarTech", "HRTech", "Other",
]


def _llm_structured(company_name: str, domain: str | None, snippets: str) -> dict[str, Any] | None:
    api_key = (config.GOOGLE_API_KEY or config.GEMINI_API_KEY or "").strip()
    if not api_key:
        return None
    prompt = (
        "Extract company profile fields from the web snippets below.\n\n"
        "STRICT RULES:\n"
        "- ONLY extract information that is EXPLICITLY stated in the snippets.\n"
        "- DO NOT invent, guess, or infer any values not clearly present.\n"
        "- If a field is not clearly mentioned, return empty string for that field.\n"
        "- For industry, use ONLY one from this list: " + ", ".join(_VALID_INDUSTRIES) + "\n"
        "- For headquarters, extract the OPERATIONAL headquarters (where company was founded/operates from), NOT holding company registration location.\n"
        "- For company_size, extract actual employee count only if explicitly mentioned.\n\n"
        "Return ONLY valid JSON with keys: domain, website, industry, company_size, headquarters, founding_year, description.\n\n"
        f"Company: {company_name}\n"
        f"Known domain hint: {domain or ''}\n"
        f"Web snippets:\n{snippets[:3500]}"
    )
    text = generate_gemini_text(
        prompt=prompt,
        api_key=api_key,
        model_candidates=(GEMINI_MODEL_PRIMARY, GEMINI_MODEL_FALLBACK),
        max_output_tokens=500,
        temperature=0.1,
    )
    return _extract_json(text or "")


def _ef(value: str | None, confidence: str, source: str) -> EnrichedField | None:
    text = _clean_text(value)
    if not text:
        return None
    return EnrichedField(value=text, confidence=confidence, source=source)


def enrich_via_tavily(company_name: str, domain: str | None) -> CompanyProfile | None:
    """
    Use Tavily search to discover and structure company profile fields.
    Returns None if no Tavily key or search fails.
    """
    if not config.TAVILY_API_KEY:
        return None
    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=config.TAVILY_API_KEY)
        queries = [
            f"{company_name} official website company profile",
            f"{company_name} headquarters industry company size founded",
            f"{company_name} employee count company size",
            f"{company_name} linkedin employees",
        ]
        if domain:
            queries = [f"{company_name} {domain} company profile headquarters industry"]

        merged_results: list[dict[str, Any]] = []
        seen_urls: set[str] = set()
        for q in queries:
            response = client.search(q, max_results=5, search_depth="advanced")
            for row in response.get("results") or []:
                url = (row.get("url") or "").strip()
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                merged_results.append(row)

        if not merged_results:
            return None

        inferred_domain = _normalize_domain(domain) or _guess_domain(company_name, merged_results)
        ranked_results = sorted(merged_results, key=lambda r: _result_relevance(r, company_name), reverse=True)
        snippets = " ".join(
            f"{_clean_text(r.get('title'))}. {_clean_text(r.get('content'))}"
            for r in ranked_results[:16]
        ).strip()

        regex_data = _regex_structured(snippets)
        llm_data = _llm_structured(company_name, inferred_domain, snippets) or {}
        source = "tavily+llm" if llm_data else "tavily"

        resolved_domain = _normalize_domain(_clean_text(llm_data.get("domain"))) or inferred_domain
        resolved_domain = _canonicalize_domain(resolved_domain) or resolved_domain
        has_domain_evidence = False
        if resolved_domain:
            has_domain_evidence = any(_normalize_domain((row.get("url") or "")) == resolved_domain for row in merged_results)
        website = _clean_text(llm_data.get("website"))
        if not website and resolved_domain:
            website = f"https://{resolved_domain}"
        if website:
            website = website if website.startswith("http") else f"https://{website}"

        raw_llm_industry = _clean_text(llm_data.get("industry"))
        raw_regex_industry = regex_data.get("industry", "")
        industry, inferred_industry_conf = _infer_industry(
            ranked_results=ranked_results,
            company_name=company_name,
            resolved_domain=resolved_domain,
            llm_industry=raw_llm_industry,
            regex_industry=raw_regex_industry,
        )
        founding_year = _normalize_founding_year(_clean_text(llm_data.get("founding_year")) or regex_data.get("founding_year", ""))
        llm_company_size = _normalize_company_size(_clean_text(llm_data.get("company_size")), founding_year=founding_year)
        regex_company_size = _normalize_company_size(regex_data.get("company_size", ""), founding_year=founding_year)
        company_size = _cross_validate_company_size(llm_company_size, regex_company_size, snippets)
        llm_hq = _normalize_headquarters(_clean_text(llm_data.get("headquarters")))
        regex_hq = _normalize_headquarters(regex_data.get("headquarters", ""))
        official_hq = _extract_hq_from_official_domain(resolved_domain)
        headquarters = official_hq or _infer_headquarters(ranked_results, company_name, resolved_domain, llm_hq, regex_hq)
        description = _clean_text(llm_data.get("description")) or regex_data.get("description", "") or company_name

        industry_conf = inferred_industry_conf if industry else "low"
        domain_conf = "high" if has_domain_evidence else "medium" if resolved_domain else "low"
        website_conf = "high" if has_domain_evidence and website and resolved_domain and resolved_domain in website else "medium" if website else "low"
        company_size_conf = "medium" if company_size else "low"
        headquarters_conf = "high" if official_hq else "medium" if headquarters else "low"
        founding_year_conf = "medium" if founding_year else "low"

        logger.info(
            "tavily profile | company=%s domain=%s industry=%s size=%s hq=%s year=%s official_hq=%s",
            company_name,
            resolved_domain or "-",
            industry or "-",
            company_size or "-",
            headquarters or "-",
            founding_year or "-",
            official_hq or "-",
        )

        return CompanyProfile(
            company_name=EnrichedField(value=company_name, confidence="medium", source="tavily"),
            website=_ef(website, website_conf, source),
            domain=_ef(resolved_domain, domain_conf, source),
            industry=_ef(industry, industry_conf, source),
            company_size=_ef(company_size, company_size_conf, source),
            headquarters=_ef(headquarters, headquarters_conf, source),
            founding_year=_ef(founding_year, founding_year_conf, source),
            description=_ef(description[:500], "low", source),
        )
    except Exception:
        logger.exception("tavily enrichment failed | company=%s domain=%s", company_name, domain or "-")
        return None
