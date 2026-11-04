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

st.subheader("Environment")


def ollama_reachable() -> bool:
    try:
        return httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=1).is_success
    except httpx.HTTPError:
        return False


checks = {
    f"LLM provider: **{settings.llm_provider}**": True,
    "Gemini API key set (`GOOGLE_API_KEY`)": bool(os.getenv("GOOGLE_API_KEY")),
    "Ollama running (needed for `LLM_PROVIDER=ollama`)": ollama_reachable(),
    "LangSmith tracing on (`LANGSMITH_TRACING=true`)": os.getenv("LANGSMITH_TRACING", "").lower() == "true",
}
for label, ok in checks.items():
    st.write(("✅ " if ok else "⚠️ ") + label)
