"""JSON file storage for enrichment results."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from app.logging_utils import get_logger

logger = get_logger("app.storage")

# Storage directory
STORAGE_DIR = Path(__file__).resolve().parent.parent / "output"


def _ensure_storage_dir() -> Path:
    """Create storage directory if it doesn't exist."""
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    return STORAGE_DIR


def _sanitize_filename(name: str) -> str:
    """Sanitize company name for use in filename."""
    safe = "".join(c if c.isalnum() or c in " -_" else "_" for c in name)
    return safe.strip().replace(" ", "_")[:50]


def save_enrichment(data: dict[str, Any], trace_id: str = "") -> str | None:
    """
    Save enrichment result to JSON file.
    Returns the file path if successful, None otherwise.
    """
    try:
        _ensure_storage_dir()
        
        company_name = data.get("company_name", "unknown")
        domain = data.get("domain", "")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create filename: company_domain_timestamp.json
        safe_name = _sanitize_filename(company_name)
        safe_domain = _sanitize_filename(domain) if domain else "no-domain"
        filename = f"{safe_name}_{safe_domain}_{timestamp}.json"
        
        filepath = STORAGE_DIR / filename
        
        # Add metadata
        stored_data = {
            "metadata": {
                "stored_at": datetime.now().isoformat(),
                "trace_id": trace_id,
                "version": "1.0",
            },
            "result": data,
        }
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(stored_data, f, indent=2, ensure_ascii=False)
        
        logger.info("enrichment saved | file=%s company=%s", filename, company_name)
        return str(filepath)
    
    except Exception as e:
        logger.warning("failed to save enrichment | error=%s", str(e))
        return None


def list_enrichments(limit: int = 50) -> list[dict[str, Any]]:
    """
    List recent enrichment results.
    Returns list of {filename, company_name, domain, stored_at}.
    """
    try:
        _ensure_storage_dir()
        
        files = sorted(
            STORAGE_DIR.glob("*.json"),
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )[:limit]
        
        results = []
        for f in files:
            try:
                with open(f, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
                    result = data.get("result", {})
                    metadata = data.get("metadata", {})
                    results.append({
                        "filename": f.name,
                        "filepath": str(f),
                        "company_name": result.get("company_name", "Unknown"),
                        "domain": result.get("domain", ""),
                        "stored_at": metadata.get("stored_at", ""),
                        "intent_score": result.get("intent_score"),
                    })
            except Exception:
                continue
        
        return results
    
    except Exception as e:
        logger.warning("failed to list enrichments | error=%s", str(e))
        return []


def get_enrichment(filename: str) -> dict[str, Any] | None:
    """
    Load a specific enrichment result by filename.
    Returns the full stored data or None if not found.
    """
    try:
        filepath = STORAGE_DIR / filename
        if not filepath.exists():
            return None
        
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    
    except Exception as e:
        logger.warning("failed to load enrichment | file=%s error=%s", filename, str(e))
        return None


def delete_enrichment(filename: str) -> bool:
    """Delete an enrichment file. Returns True if successful."""
    try:
        filepath = STORAGE_DIR / filename
        if filepath.exists():
            filepath.unlink()
            logger.info("enrichment deleted | file=%s", filename)
            return True
        return False
    
    except Exception as e:
        logger.warning("failed to delete enrichment | file=%s error=%s", filename, str(e))
        return False


def get_enrichment_by_domain(domain: str) -> dict[str, Any] | None:
    """
    Find the most recent enrichment for a domain.
    Returns the result data or None if not found.
    """
    try:
        _ensure_storage_dir()
        
        domain_lower = domain.lower().replace(".", "_")
        
        # Find files matching domain
        matching_files = sorted(
            [f for f in STORAGE_DIR.glob("*.json") if domain_lower in f.name.lower()],
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )
        
        if matching_files:
            with open(matching_files[0], "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("result")
        
        return None
    
    except Exception:
        return None


def get_storage_stats() -> dict[str, Any]:
    """Get storage statistics."""
    try:
        _ensure_storage_dir()
        
        files = list(STORAGE_DIR.glob("*.json"))
        total_size = sum(f.stat().st_size for f in files)
        
        return {
            "total_files": len(files),
            "total_size_kb": round(total_size / 1024, 2),
            "storage_path": str(STORAGE_DIR),
        }
    
    except Exception:
        return {"total_files": 0, "total_size_kb": 0, "storage_path": str(STORAGE_DIR)}
