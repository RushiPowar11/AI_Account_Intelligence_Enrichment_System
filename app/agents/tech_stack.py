"""Tech stack detection: BuiltWith API or Tavily fallback."""

from urllib.parse import urlparse

import httpx

from app.config import config
from app.logging_utils import get_logger

logger = get_logger("app.agents.tech_stack")

TECH_KEYWORDS: dict[str, str] = {
    "salesforce": "CRM: Salesforce",
    "hubspot": "CRM/Marketing: HubSpot",
    "marketo": "Marketing Automation: Marketo",
    "pardot": "Marketing Automation: Pardot",
    "mailchimp": "Email Marketing: Mailchimp",
    "wordpress": "Website Platform: WordPress",
    "shopify": "Website Platform: Shopify",
    "drupal": "Website Platform: Drupal",
    "google analytics": "Analytics: Google Analytics",
    "segment": "Customer Data: Segment",
    "zendesk": "Support: Zendesk",
    "intercom": "Support: Intercom",
    "snowflake": "Data Platform: Snowflake",
}


def _normalize_domain(domain: str | None) -> str | None:
    if not domain:
        return None
    raw = domain.strip().lower()
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
    return host.split(":")[0] if host else None


def _detect_from_homepage(domain: str | None) -> list[str]:
    host = _normalize_domain(domain)
    if not host:
        return []
    urls = [f"https://{host}", f"https://www.{host}", f"http://{host}"]
    markers = {
        "googletagmanager.com": "Analytics: Google Tag Manager",
        "google-analytics.com": "Analytics: Google Analytics",
        "gtag(": "Analytics: Google Analytics",
        "segment": "Customer Data: Segment",
        "segment.com": "Customer Data: Segment",
        "hs-scripts.com": "CRM/Marketing: HubSpot",
        "marketo": "Marketing Automation: Marketo",
        "pardot": "Marketing Automation: Pardot",
        "intercom": "Support: Intercom",
        "zendesk": "Support: Zendesk",
        "wp-content": "Website Platform: WordPress",
        "shopify": "Website Platform: Shopify",
        "react": "Frontend Framework: React",
        "next": "Frontend Framework: Next.js",
        "cloudflare": "Infrastructure: Cloudflare",
    }
    for url in urls:
        try:
            with httpx.Client(timeout=8.0, follow_redirects=True) as client:
                resp = client.get(url, headers={"User-Agent": "Mozilla/5.0"})
                if resp.status_code >= 400:
                    continue
                html = (resp.text or "").lower()
                hits = [label for token, label in markers.items() if token in html]
                if hits:
                    return list(dict.fromkeys(hits))[:8]
        except Exception:
            continue
    return []


def run_tech_stack(company_name: str, domain: str | None) -> list[str]:
    """
    Return list of detected technologies (e.g. CRM: Salesforce, Marketing: HubSpot).
    Uses BuiltWith if key present, else Tavily search.
    """
    normalized_domain = _normalize_domain(domain)

    if config.BUILTWITH_API_KEY and domain:
        try:
            # BuiltWith company API (example endpoint; adjust to actual API)
            url = f"https://api.builtwith.com/v20/api.json?KEY={config.BUILTWITH_API_KEY}&LOOKUP={domain}"
            with httpx.Client(timeout=8.0) as client:
                r = client.get(url)
                if r.status_code == 200:
                    data = r.json()
                    paths = data.get("Results", [{}])[0].get("Paths", [])
                    techs: list[str] = []
                    for p in paths[:5]:
                        for tech in p.get("Technologies", [])[:3]:
                            name = tech.get("Name")
                            if name:
                                techs.append(name)
                    if techs:
                        out = list(dict.fromkeys(techs))[:12]
                        logger.info("tech stack via builtwith | company=%s count=%d", company_name, len(out))
                        return out
        except Exception:
            pass

    # Direct website fingerprint fallback
    homepage_tech = _detect_from_homepage(normalized_domain)
    if homepage_tech:
        logger.info("tech stack via homepage fingerprint | company=%s count=%d", company_name, len(homepage_tech))
        return homepage_tech

    # Tavily fallback
    if config.TAVILY_API_KEY:
        try:
            from tavily import TavilyClient
            client = TavilyClient(api_key=config.TAVILY_API_KEY)
            queries = [f"{company_name} technology stack CRM marketing analytics", f"{company_name} builtwith stack"]
            if normalized_domain:
                queries = [f"site:{normalized_domain} technology stack", f"{company_name} {normalized_domain} tech stack"]
            results = []
            seen = set()
            for q in queries:
                r = client.search(q, max_results=4, search_depth="advanced")
                for row in r.get("results") or []:
                    url = (row.get("url") or "").strip()
                    if url and url in seen:
                        continue
                    if url:
                        seen.add(url)
                    results.append(row)
            techs: list[str] = []
            for x in results:
                content = (x.get("content") or "").lower()
                for token, label in TECH_KEYWORDS.items():
                    if token in content and label not in techs:
                        techs.append(label)
            out = techs[:10]
            logger.info("tech stack via tavily | company=%s count=%d", company_name, len(out))
            return out
        except Exception:
            logger.exception("tech stack detection failed | company=%s domain=%s", company_name, normalized_domain or "-")
    return []
