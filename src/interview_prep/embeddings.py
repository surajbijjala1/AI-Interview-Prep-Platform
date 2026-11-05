"""Local embeddings on Ollama. No API quota, and resume text never leaves the machine for embedding."""

from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings

from interview_prep.config import Settings, get_settings

# nomic-embed-text was trained with task prefixes; leaving them out noticeably hurts retrieval.
# https://huggingface.co/nomic-ai/nomic-embed-text-v1.5#usage
NOMIC_PREFIXES = ("search_document: ", "search_query: ")


class PrefixedEmbeddings(Embeddings):
    """Adds the model's task prefixes: one for stored documents, another for search queries."""

    def __init__(self, inner: Embeddings, document_prefix: str, query_prefix: str):
        self.inner = inner
        self.document_prefix = document_prefix
        self.query_prefix = query_prefix

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.inner.embed_documents([self.document_prefix + t for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self.inner.embed_query(self.query_prefix + text)


def get_embeddings(settings: Settings | None = None) -> Embeddings:
    settings = settings or get_settings()
    model = OllamaEmbeddings(model=settings.embedding_model, base_url=settings.ollama_base_url)
    if settings.embedding_model.startswith("nomic-embed-text"):
        return PrefixedEmbeddings(model, *NOMIC_PREFIXES)
    return model
