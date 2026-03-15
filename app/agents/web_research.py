"""Web research agent: leadership and business signals via Tavily + LLM + Hunter."""

from typing import Any
import re

import httpx

from app.config import config
from app.agents.llm_client import generate_gemini_text
from app.logging_utils import get_logger

logger = get_logger("app.agents.web_research")
GEMINI_MODEL_PRIMARY = "gemini-2.5-flash"
GEMINI_MODEL_FALLBACK = "gemini-2.0-flash"


def _hunter_domain_search(domain: str) -> dict[str, Any]:
    """
    Search Hunter.io for emails and contacts at a domain.
    Returns {"emails": [...], "email_pattern": "...", "organization": "..."}
    """
    if not config.HUNTER_API_KEY or not domain:
        return {}
    
    # Clean domain - remove protocol and path
    clean_domain = domain.replace("https://", "").replace("http://", "").replace("www.", "").split("/")[0].strip()
    
    url = "https://api.hunter.io/v2/domain-search"
    params = {
        "domain": clean_domain,
        "api_key": config.HUNTER_API_KEY,
        "limit": 10,
    }
    
    try:
        with httpx.Client(timeout=10.0) as client:
            r = client.get(url, params=params)
            if r.status_code != 200:
                logger.warning("hunter domain search failed | status=%d", r.status_code)
                return {}
            data = r.json().get("data", {})
            
            result = {
                "organization": data.get("organization", ""),
                "email_pattern": "",
                "emails": [],
            }
            
            pattern = data.get("pattern")
            if pattern:
                result["email_pattern"] = f"{pattern}@{clean_domain}"
            
            for email_obj in data.get("emails", [])[:10]:
                email_data = {
                    "email": email_obj.get("value", ""),
                    "first_name": email_obj.get("first_name", ""),
                    "last_name": email_obj.get("last_name", ""),
                    "position": email_obj.get("position", ""),
                    "confidence": email_obj.get("confidence", 0),
                    "department": email_obj.get("department", ""),
                }
                if email_data["email"]:
                    result["emails"].append(email_data)
            
            logger.info("hunter domain search | domain=%s emails=%d pattern=%s", 
                       domain, len(result["emails"]), result["email_pattern"] or "-")
            return result
    except Exception as e:
        logger.warning("hunter domain search error | error=%s", str(e))
        return {}


def _hunter_email_finder(domain: str, first_name: str, last_name: str) -> dict[str, Any]:
    """
    Find email for a specific person at a domain.
    Returns {"email": "...", "confidence": 0-100}
    """
    if not config.HUNTER_API_KEY or not domain or not first_name or not last_name:
        return {}
    
    # Clean domain
    clean_domain = domain.replace("https://", "").replace("http://", "").replace("www.", "").split("/")[0].strip()
    
    url = "https://api.hunter.io/v2/email-finder"
    params = {
        "domain": clean_domain,
        "first_name": first_name,
        "last_name": last_name,
        "api_key": config.HUNTER_API_KEY,
    }
    
    try:
        with httpx.Client(timeout=10.0) as client:
            r = client.get(url, params=params)
            if r.status_code != 200:
                return {}
            data = r.json().get("data", {})
            
            email = data.get("email", "")
            confidence = data.get("score", 0)
            
            if email:
                logger.info("hunter email finder | %s %s -> %s (confidence=%d)", 
                           first_name, last_name, email, confidence)
                return {"email": email, "confidence": confidence}
            return {}
    except Exception:
        return {}


def _is_valid_person_name(name: str, company_name: str = "") -> bool:
    """Check if a name looks like a real person name, not a company name."""
    if not name or len(name.split()) < 2:
        return False
    
    name_lower = name.lower()
    company_lower = company_name.lower() if company_name else ""
    
    # Reject if name contains company name or vice versa
    if company_lower and (company_lower in name_lower or name_lower in company_lower):
        return False
    
    # Reject common non-person patterns
    bad_patterns = [
        "digital", "engineering", "technologies", "solutions", "services",
        "consulting", "software", "systems", "corporation", "company",
        "group", "inc", "ltd", "llc", "limited",
    ]
    if any(pattern in name_lower for pattern in bad_patterns):
        return False
    
    return True


def _enrich_leadership_with_hunter(leadership: list[str], domain: str, company_name: str = "") -> list[dict[str, Any]]:
    """
    Enrich leadership list with emails from Hunter.io.
    Returns list of {"name": "...", "title": "...", "email": "...", "confidence": 0-100}
    """
    if not domain or not leadership:
        return [{"name": name.split(" - ")[0], "title": name.split(" - ")[1] if " - " in name else ""} 
                for name in leadership if _is_valid_person_name(name.split(" - ")[0], company_name)]
    
    hunter_data = _hunter_domain_search(domain)
    hunter_emails = {
        f"{e['first_name'].lower()} {e['last_name'].lower()}": e 
        for e in hunter_data.get("emails", []) 
        if e.get("first_name") and e.get("last_name")
    }
    
    enriched = []
    for item in leadership:
        if " - " not in item:
            if _is_valid_person_name(item, company_name):
                enriched.append({"name": item, "title": "", "email": "", "confidence": 0})
            continue
        
        name, title = item.split(" - ", 1)
        
        # Skip invalid person names
        if not _is_valid_person_name(name, company_name):
            continue
        
        name_parts = name.strip().split()
        
        entry = {
            "name": name.strip(),
            "title": title.strip(),
            "email": "",
            "confidence": 0,
        }
        
        name_lower = name.strip().lower()
        if name_lower in hunter_emails:
            hunter_entry = hunter_emails[name_lower]
            entry["email"] = hunter_entry.get("email", "")
            entry["confidence"] = hunter_entry.get("confidence", 0)
        elif len(name_parts) >= 2:
            first_name = name_parts[0]
            last_name = name_parts[-1]
            found = _hunter_email_finder(domain, first_name, last_name)
            if found:
                entry["email"] = found.get("email", "")
                entry["confidence"] = found.get("confidence", 0)
        
        enriched.append(entry)
    
    return enriched


_FAMOUS_FOUNDERS_BLOCKLIST = {
    "kunal shah",
    "elon musk",
    "jeff bezos",
    "mark zuckerberg",
    "sundar pichai",
    "satya nadella",
    "tim cook",
    "bill gates",
    "steve jobs",
    "jack dorsey",
    "brian chesky",
    "travis kalanick",
    "dara khosrowshahi",
    "reed hastings",
    "jensen huang",
    "sam altman",
}


def _clean_leadership_item(item: str, snippets_lower: str = "") -> str:
    s = re.sub(r"\s+", " ", (item or "")).strip().strip(",.-")
    if not s or "-" not in s:
        return ""
    name, title = [x.strip() for x in s.split("-", 1)]
    if not name or not title:
        return ""
    
    name_lower = name.lower()
    if name_lower in _FAMOUS_FOUNDERS_BLOCKLIST:
        return ""
    
    bad_name_tokens = {
        "former", "co", "ex", "team", "leadership", "executive", "email", "phone", "contact",
        "current", "subscribe", "subscribe", "follow", "click", "view", "read", "more",
        "company", "group", "consulting", "partners", "solutions", "services", "inc", "ltd",
        "boston", "mckinsey", "bain", "deloitte", "kpmg", "pwc", "accenture",
    }
    if any(tok in name_lower for tok in bad_name_tokens):
        return ""
    if len(name.split()) < 2:
        return ""
    if len(name.split()) > 4:
        return ""
    if not re.search(r"[A-Z][a-z]+", name):
        return ""
    if re.search(r"\d", name):
        return ""
    if any(c in name for c in ["@", "#", "$", "%", "&", "*", "="]):
        return ""
    
    if snippets_lower and name_lower not in snippets_lower:
        return ""
    
    return f"{name} - {title}"


def _finalize_leadership_items(items: list[str], company_name: str, snippets_lower: str = "") -> list[str]:
    out: list[str] = []
    company_tokens = {t.lower() for t in re.findall(r"[A-Za-z]+", company_name) if len(t) > 2}
    company_name_lower = company_name.lower()
    
    for item in items:
        cleaned = _clean_leadership_item(item, snippets_lower)
        if not cleaned:
            continue
        name, title = [x.strip() for x in cleaned.split("-", 1)]
        
        # Skip if name looks like company name
        name_lower = name.lower()
        if company_name_lower in name_lower or name_lower in company_name_lower:
            continue
        
        # Skip names that are mostly company tokens
        name_tokens = [t for t in re.findall(r"[A-Za-z]+", name) if t]
        company_token_count = sum(1 for t in name_tokens if t.lower() in company_tokens)
        if len(name_tokens) > 0 and company_token_count / len(name_tokens) > 0.5:
            continue
        
        # Drop company tokens accidentally captured as name.
        filtered = [t for t in name_tokens if t.lower() not in company_tokens]
        dedup_tokens: list[str] = []
        for tok in filtered:
            if not dedup_tokens or dedup_tokens[-1].lower() != tok.lower():
                dedup_tokens.append(tok)
        if len(dedup_tokens) >= 2:
            name = " ".join(dedup_tokens[:3])
        elif len(name_tokens) >= 2:
            name = " ".join(name_tokens[:3])
        else:
            continue
        out.append(f"{name} - {title}")
    return list(dict.fromkeys(out))


def _extract_named_regex(snippets: str) -> list[str]:
    out: list[str] = []
    snippets_lower = snippets.lower()
    patterns = [
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\s*(?:is\s+the\s+|,\s*)(CEO|Chief Executive Officer|Founder|Co-Founder|CFO|Chief Financial Officer|Chief Revenue Officer|VP Sales|Head of Sales)",
        r"(CEO|Chief Executive Officer|Founder|Co-Founder|CFO|Chief Financial Officer|Chief Revenue Officer|VP Sales|Head of Sales)\s*(?:of|at|:|-)?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})",
    ]
    for pat in patterns:
        for m in re.findall(pat, snippets):
            if len(m) != 2:
                continue
            left, right = m
            if "chief" in left.lower() or "founder" in left.lower() or "vp" in left.lower() or "head of" in left.lower() or left.upper() in {"CEO", "CFO"}:
                title, name = left, right
            else:
                name, title = left, right
            cleaned = _clean_leadership_item(f"{name} - {title}", snippets_lower)
            if cleaned:
                out.append(cleaned)
    return out


def _extract_named_leadership(snippets: str) -> list[str]:
    api_key = (config.GOOGLE_API_KEY or config.GEMINI_API_KEY or "").strip()
    if not api_key or not snippets.strip():
        return []
    
    snippets_lower = snippets.lower()
    
    prompt = (
        "Extract decision makers from this text.\n\n"
        "STRICT RULES:\n"
        "- ONLY extract names that EXPLICITLY appear in the provided text below.\n"
        "- Names MUST be real human names in 'FirstName LastName' format (2-3 words).\n"
        "- DO NOT invent or guess any names not present in the text.\n"
        "- DO NOT use your general knowledge - ONLY use names from the text.\n"
        "- DO NOT include company names, job board text, subscription prompts, or UI elements.\n"
        "- DO NOT include partial names or single words.\n"
        "- If no valid names are found, return empty array.\n\n"
        "Return ONLY valid JSON: {\"leadership\": [\"Name - Title\", ...]}\n\n"
        f"Text:\n{snippets[:3200]}"
    )
    text = generate_gemini_text(
        prompt=prompt,
        api_key=api_key,
        model_candidates=(GEMINI_MODEL_PRIMARY, GEMINI_MODEL_FALLBACK),
        max_output_tokens=400,
        temperature=0.0,
    )
    if not text:
        return []
    m = re.search(r"\{.*\}", text, re.S)
    raw = m.group(0) if m else text.strip()
    try:
        import json

        data = json.loads(raw)
        leadership = data.get("leadership") if isinstance(data, dict) else None
        if not isinstance(leadership, list):
            return []
        out = []
        for item in leadership:
            s = _clean_leadership_item(str(item), snippets_lower)
            if s:
                out.append(s)
        return out[:6]
    except Exception:
        return []


def run_web_research(company_name: str, domain: str | None) -> dict[str, Any]:
    """
    Search for leadership and business signals. Returns
    {"leadership": [...], "leadership_enriched": [...], "email_pattern": "...", "business_signals": [...]}
    """
    result: dict[str, Any] = {
        "leadership": [], 
        "leadership_enriched": [],
        "email_pattern": "",
        "business_signals": [],
    }
    if not config.TAVILY_API_KEY:
        return result
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=config.TAVILY_API_KEY)
        queries = [
            f"{company_name} leadership team CEO Founder VP Sales",
            f"{company_name} funding expansion hiring news",
        ]
        if domain:
            queries = [
                f"{company_name} {domain} leadership team executives",
                f"{company_name} {domain} funding expansion hiring",
            ]
        results = []
        seen = set()
        for q in queries:
            r = client.search(q, max_results=5, search_depth="advanced")
            for row in r.get("results") or []:
                url = (row.get("url") or "").strip()
                if not url or url in seen:
                    continue
                seen.add(url)
                results.append(row)
        snippets = " ".join((x.get("content") or "") for x in results[:8])
        lower = snippets.lower()

        # Leadership extraction from common title patterns.
        leadership: list[str] = _extract_named_leadership(snippets)
        leadership.extend(_extract_named_regex(snippets))
        patterns = [
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\s*[,|-]\s*(CEO|Chief Executive Officer|Founder|CFO|Chief Financial Officer|Chief Revenue Officer|VP Sales|Head of Sales|Head of Marketing)",
            r"(CEO|Chief Executive Officer|Founder|CFO|Chief Financial Officer|Chief Revenue Officer|VP Sales|Head of Sales|Head of Marketing)\s*[:|-]\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})",
        ]
        for pat in patterns:
            for m in re.findall(pat, snippets):
                if len(m) == 2:
                    left, right = m
                    if "chief" in left.lower() or "vp" in left.lower() or "head of" in left.lower() or left.upper() in {"CEO", "CFO"}:
                        title, name = left, right
                    else:
                        name, title = left, right
                    name_tokens = name.strip().split()
                    if name_tokens and name_tokens[0].lower() in {"cybersecurity", "identity", "access", "security", "cloud", "marketing", "sales"} and len(name_tokens) >= 2:
                        name = " ".join(name_tokens[-2:])
                    if len(name_tokens) > 3:
                        name = " ".join(name_tokens[-3:])
                    cleaned = _clean_leadership_item(f"{name.strip()} - {title.strip()}", lower)
                    if cleaned:
                        leadership.append(cleaned)

        if not leadership:
            # Targeted name lookup fallback.
            extra_queries = [f"{company_name} CEO name", f"{company_name} founder name"]
            for q in extra_queries:
                rr = client.search(q, max_results=3, search_depth="advanced")
                extra_snippets = " ".join((x.get("content") or "") for x in (rr.get("results") or [])[:3])
                leadership.extend(_extract_named_regex(extra_snippets))
                if leadership:
                    break
        if not leadership:
            for title in ("CEO", "Founder", "CFO", "Chief Revenue Officer", "VP Sales", "Head of Sales", "Head of Marketing"):
                if title.lower() in lower:
                    leadership.append(f"{title} (from web research)")
        if not leadership and snippets:
            leadership.append("Leadership team identified (web research)")
        
        finalized_leadership = _finalize_leadership_items(leadership, company_name, lower)[:6]
        result["leadership"] = finalized_leadership
        
        # Enrich leadership with Hunter.io emails
        if domain and config.HUNTER_API_KEY:
            hunter_data = _hunter_domain_search(domain)
            result["email_pattern"] = hunter_data.get("email_pattern", "")
            result["leadership_enriched"] = _enrich_leadership_with_hunter(finalized_leadership, domain, company_name)
            
            # Add decision makers from Hunter if we didn't find many
            if len(finalized_leadership) < 3:
                for hunter_email in hunter_data.get("emails", [])[:5]:
                    position = hunter_email.get("position", "").lower()
                    if any(title in position for title in ["ceo", "founder", "cto", "cfo", "vp", "director", "head"]):
                        name = f"{hunter_email.get('first_name', '')} {hunter_email.get('last_name', '')}".strip()
                        if name and len(name.split()) >= 2 and _is_valid_person_name(name, company_name):
                            entry = {
                                "name": name,
                                "title": hunter_email.get("position", "Executive"),
                                "email": hunter_email.get("email", ""),
                                "confidence": hunter_email.get("confidence", 0),
                            }
                            # Avoid duplicates
                            existing_names = [e["name"].lower() for e in result["leadership_enriched"]]
                            if name.lower() not in existing_names:
                                result["leadership_enriched"].append(entry)
        else:
            result["leadership_enriched"] = [
                {"name": item.split(" - ")[0], "title": item.split(" - ")[1] if " - " in item else "", "email": "", "confidence": 0}
                for item in finalized_leadership
            ]

        # Business signals from keyword groups.
        signal_map = {
            "hiring": "Hiring activity detected",
            "job opening": "Hiring activity detected",
            "funding": "Funding signal detected",
            "raised": "Funding signal detected",
            "acquisition": "M&A activity detected",
            "expansion": "Expansion signal detected",
            "launch": "New product launch signal",
            "partnership": "Partnership signal detected",
            "growth": "Growth signal detected",
        }
        signals: list[str] = []
        for token, label in signal_map.items():
            if token in lower:
                signals.append(label)
        if not signals and snippets:
            signals.append("Recent company activity detected")
        result["business_signals"] = list(dict.fromkeys(signals))[:6]
        logger.info(
            "web research | company=%s leaders=%d signals=%d",
            company_name,
            len(result["leadership"]),
            len(result["business_signals"]),
        )
    except Exception:
        logger.exception("web research failed | company=%s domain=%s", company_name, domain or "-")
    return result
