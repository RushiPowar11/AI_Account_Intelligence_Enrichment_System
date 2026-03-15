"""Load configuration from environment."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path)


def _get(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


class Config:
    """Application configuration."""

    # LLM providers (with fallback chain: Gemini -> Groq -> OpenRouter)
    GOOGLE_API_KEY: str = _get("GOOGLE_API_KEY")
    GEMINI_API_KEY: str = _get("GEMINI_API_KEY")  # Gemini LLM (either key works)
    GROQ_API_KEY: str = _get("GROQ_API_KEY")  # Groq fallback (Llama-3-70B)
    OPENROUTER_API_KEY: str = _get("OPENROUTER_API_KEY")  # OpenRouter fallback
    
    # Enrichment APIs
    CLEARBIT_API_KEY: str = _get("CLEARBIT_API_KEY")
    APOLLO_API_KEY: str = _get("APOLLO_API_KEY")  # Apollo.io enrichment
    TAVILY_API_KEY: str = _get("TAVILY_API_KEY")
    BUILTWITH_API_KEY: str = _get("BUILTWITH_API_KEY")
    HUNTER_API_KEY: str = _get("HUNTER_API_KEY")
    LOG_LEVEL: str = _get("LOG_LEVEL", "INFO")


config = Config()
