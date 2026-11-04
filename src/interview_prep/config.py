"""App settings, read from environment variables (and `.env`)."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    # Env vars match field names case-insensitively, e.g. LLM_PROVIDER=ollama.
    llm_provider: Literal["gemini", "ollama"] = "gemini"
    gemini_model: str = "gemini-2.5-flash"
    # Gemini free tier allows ~10-15 requests/min; stay under it app-wide.
    gemini_requests_per_minute: int = 10
    ollama_model: str = "llama3.2"
    ollama_base_url: str = "http://localhost:11434"


@lru_cache
def get_settings() -> Settings:
    # LangSmith and the Google SDK read os.environ directly, so load .env into it first.
    load_dotenv(ROOT / ".env")
    return Settings()
