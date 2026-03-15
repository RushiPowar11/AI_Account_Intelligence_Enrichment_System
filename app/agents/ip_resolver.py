"""IP to company/domain resolution. Simulated for demo when no external API."""

# Simulated mapping for demo (real implementation would use Clearbit IP or similar)
DEMO_IP_TO_COMPANY: dict[str, tuple[str, str]] = {
    "34.201.": ("Acme Mortgage", "acmemortgage.com"),
    "52.0.": ("Summit Realty Group", "summitrealty.com"),
    "3.0.": ("Rocket Mortgage", "rocketmortgage.com"),
}


def resolve_ip(ip: str | None) -> tuple[str, str | None]:
    """
    Resolve IP to (company_name, domain). Returns ("Unknown Company", None) if not found.
    Demo: prefix match on DEMO_IP_TO_COMPANY; production would call Clearbit IP or similar.
    """
    if not ip or not ip.strip():
        return "Unknown Company", None
    ip = ip.strip()
    for prefix, (name, domain) in DEMO_IP_TO_COMPANY.items():
        if ip.startswith(prefix):
            return name, domain
    return "Unknown Company", None
