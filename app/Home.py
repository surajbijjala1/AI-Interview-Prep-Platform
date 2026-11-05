import os

import httpx
import streamlit as st

from interview_prep.config import get_settings

st.set_page_config(page_title="AI Interview Prep", page_icon="🎯")
settings = get_settings()

st.title("AI Interview Prep")
st.write(
    "Upload your resume and a job description. The agent researches the company's interview process, "
    "then runs an adaptive mock interview one question at a time."
)
st.page_link("pages/1_Setup.py", label="Start: Setup", icon="➡️")
st.page_link("pages/6_Question_Bank.py", label="Try the question bank (MCP test page)", icon="🗂️")

st.subheader("Environment")


def ollama_models() -> list[str] | None:
    """Installed Ollama models, or None if Ollama isn't running."""
    try:
        response = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=1)
        return [m["name"] for m in response.json()["models"]] if response.is_success else None
    except httpx.HTTPError:
        return None


models = ollama_models()
checks = {
    f"LLM provider: **{settings.llm_provider}**": True,
    "Gemini API key set (`GOOGLE_API_KEY`)": bool(os.getenv("GOOGLE_API_KEY")),
    "Ollama running (`ollama serve`; always needed for embeddings)": models is not None,
    f"Embedding model `{settings.embedding_model}` pulled": any(
        m.split(":")[0] == settings.embedding_model for m in models or []
    ),
    "LangSmith tracing on (`LANGSMITH_TRACING=true`)": os.getenv("LANGSMITH_TRACING", "").lower() == "true",
}
for label, ok in checks.items():
    st.write(("✅ " if ok else "⚠️ ") + label)
