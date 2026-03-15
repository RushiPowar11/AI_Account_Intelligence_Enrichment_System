"""Multi-LLM client with Gemini, Groq, and OpenRouter fallbacks."""

from typing import Iterable
import httpx

from app.config import config
from app.logging_utils import get_logger

logger = get_logger("app.agents.llm_client")


def _call_groq(prompt: str, max_tokens: int = 900, temperature: float = 0.2) -> str | None:
    """Call Groq API with Llama-3-70B. Returns None if unavailable or fails."""
    api_key = getattr(config, "GROQ_API_KEY", "").strip()
    if not api_key:
        return None
    
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    models_to_try = ["llama-3.3-70b-versatile", "llama3-70b-8192", "mixtral-8x7b-32768"]
    
    for model in models_to_try:
        try:
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            
            with httpx.Client(timeout=30.0) as client:
                r = client.post(url, json=payload, headers=headers)
                if r.status_code == 200:
                    data = r.json()
                    text = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                    if text:
                        logger.info("groq response | model=%s", model)
                        return text
                elif r.status_code == 429:
                    logger.debug("groq rate limited | model=%s", model)
                    continue
                else:
                    logger.debug("groq error | model=%s status=%d", model, r.status_code)
                    continue
        except Exception as e:
            logger.debug("groq exception | model=%s error=%s", model, str(e))
            continue
    
    return None


def _call_openrouter(prompt: str, max_tokens: int = 900, temperature: float = 0.2) -> str | None:
    """Call OpenRouter API with free models. Returns None if unavailable or fails."""
    api_key = getattr(config, "OPENROUTER_API_KEY", "").strip()
    if not api_key:
        return None
    
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    # Free models on OpenRouter
    models_to_try = [
        "deepseek/deepseek-chat",
        "mistralai/mistral-small",
        "meta-llama/llama-3-8b-instruct:free",
    ]
    
    for model in models_to_try:
        try:
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            
            with httpx.Client(timeout=30.0) as client:
                r = client.post(url, json=payload, headers=headers)
                if r.status_code == 200:
                    data = r.json()
                    text = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                    if text:
                        logger.info("openrouter response | model=%s", model)
                        return text
                else:
                    logger.debug("openrouter error | model=%s status=%d", model, r.status_code)
                    continue
        except Exception as e:
            logger.debug("openrouter exception | model=%s error=%s", model, str(e))
            continue
    
    return None


def _call_gemini(
    prompt: str,
    api_key: str,
    model_candidates: Iterable[str],
    max_output_tokens: int = 900,
    temperature: float = 0.2,
) -> str | None:
    """
    Generate text from Gemini models.
    Tries google-genai SDK first, then falls back to google-generativeai.
    Returns None if rate limited or fails.
    """
    if not api_key:
        return None
    models = list(model_candidates)
    if not models:
        return None

    rate_limited = False

    # New SDK path (google-genai)
    try:
        from google import genai
    except Exception:
        genai = None

    if genai is not None:
        try:
            client = genai.Client(api_key=api_key)
            gen_config = {
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
            }
            for model_name in models:
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config=gen_config,
                    )
                    text = (getattr(response, "text", None) or "").strip()
                    if text:
                        logger.debug("gemini response via google-genai | model=%s", model_name)
                        return text
                except Exception as e:
                    if "429" in str(e) or "rate" in str(e).lower():
                        rate_limited = True
                    continue
        except Exception:
            pass

    # Legacy SDK fallback (google-generativeai)
    try:
        import warnings

        warnings.filterwarnings("ignore", category=FutureWarning)
        import google.generativeai as genai_legacy

        genai_legacy.configure(api_key=api_key)
        generation_config = genai_legacy.GenerationConfig(
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )
        for model_name in models:
            try:
                model = genai_legacy.GenerativeModel(model_name)
                response = model.generate_content(prompt, generation_config=generation_config)
                text = getattr(response, "text", None) or ""
                if not text and getattr(response, "candidates", None):
                    c = response.candidates[0]
                    if getattr(c, "content", None) and getattr(c.content, "parts", None):
                        part = c.content.parts[0] if c.content.parts else None
                        text = getattr(part, "text", None) or ""
                text = text.strip()
                if text:
                    logger.debug("gemini response via google-generativeai | model=%s", model_name)
                    return text
            except Exception as e:
                if "429" in str(e) or "rate" in str(e).lower():
                    rate_limited = True
                continue
    except Exception:
        pass

    if rate_limited:
        logger.debug("gemini rate limited")
    else:
        logger.debug("gemini response empty across all model candidates")
    return None


def generate_gemini_text(
    prompt: str,
    api_key: str,
    model_candidates: Iterable[str],
    max_output_tokens: int = 900,
    temperature: float = 0.2,
) -> str | None:
    """
    Generate text with multi-LLM fallback chain:
    1. Gemini (primary)
    2. Groq (fallback - fast, free)
    3. OpenRouter (fallback - many free models)
    """
    # Try Gemini first
    result = _call_gemini(prompt, api_key, model_candidates, max_output_tokens, temperature)
    if result:
        return result
    
    # Fallback to Groq
    logger.debug("falling back to groq")
    result = _call_groq(prompt, max_output_tokens, temperature)
    if result:
        return result
    
    # Fallback to OpenRouter
    logger.debug("falling back to openrouter")
    result = _call_openrouter(prompt, max_output_tokens, temperature)
    if result:
        return result
    
    logger.warning("all LLM providers failed")
    return None
