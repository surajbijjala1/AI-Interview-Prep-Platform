"""Chat model factory. Switch providers with LLM_PROVIDER=gemini|ollama."""

from functools import lru_cache

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter

from .config import Settings, get_settings


@lru_cache
def _gemini_rate_limiter(requests_per_minute: int) -> InMemoryRateLimiter:
    # One limiter shared by every Gemini model instance, so the free-tier cap applies app-wide.
    return InMemoryRateLimiter(
        requests_per_second=requests_per_minute / 60,
        check_every_n_seconds=0.1,
        max_bucket_size=1,
    )


def get_chat_model(temperature: float = 0.0, settings: Settings | None = None) -> BaseChatModel:
    settings = settings or get_settings()
    if settings.llm_provider == "ollama":
        return init_chat_model(
            settings.ollama_model,
            model_provider="ollama",
            temperature=temperature,
            base_url=settings.ollama_base_url,
        )
    return init_chat_model(
        settings.gemini_model,
        model_provider="google_genai",
        temperature=temperature,
        rate_limiter=_gemini_rate_limiter(settings.gemini_requests_per_minute),
        # The Google SDK retries 429s and transient errors with backoff.
        max_retries=6,
    )
