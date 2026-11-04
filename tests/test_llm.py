from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

from interview_prep.config import Settings
from interview_prep.llm import get_chat_model


def test_ollama_provider():
    model = get_chat_model(settings=Settings(llm_provider="ollama", ollama_model="llama3.2"))
    assert isinstance(model, ChatOllama)
    assert model.model == "llama3.2"


def test_gemini_provider_shares_one_rate_limiter(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    settings = Settings(llm_provider="gemini", gemini_model="gemini-2.5-flash")
    a, b = get_chat_model(settings=settings), get_chat_model(temperature=0.7, settings=settings)
    assert isinstance(a, ChatGoogleGenerativeAI)
    assert a.rate_limiter is b.rate_limiter is not None
