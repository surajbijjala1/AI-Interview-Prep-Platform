import math
import re
import zlib
from pathlib import Path

import pytest
from fpdf import FPDF
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableLambda

from interview_prep.config import Settings
from interview_prep.storage import Storage, open_storage

FIXTURES = Path(__file__).parent / "fixtures"

# Fictional persona: 555 numbers and example.com are reserved for fiction/testing.
PERSONA_RESUME = """Priya Raman | priya.raman@example.com | (415) 555-0134 | linkedin.com/in/priyaraman
123 Market Street, San Francisco, CA 94105 | https://priyaraman.dev

SUMMARY
Backend engineer with 4 years building Python and Go services. Priya enjoys distributed systems.

EXPERIENCE
Software Engineer, Ledgerly - Jan 2021 to Present
- Built a payments API in FastAPI serving 10000 requests per second on PostgreSQL.
- Migrated jobs from RabbitMQ to Kafka; cut p99 latency by 40%.
- Mentored 2 junior engineers. Projects at github.com/priyar/ledger-tools.

SKILLS
Python 3.12, Go, FastAPI, Django, PostgreSQL, Redis, Kafka, Docker, Kubernetes, AWS, ASP.NET, Node.js

EDUCATION
B.S. Computer Science, State University, 2016 - 2020
"""


def make_pdf(text: str) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=10)
    for line in text.splitlines():
        pdf.multi_cell(0, 5, line or " ", new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())


class FakeStructuredLLM:
    """Stands in for a chat model: returns a canned object and records every prompt it was sent."""

    def __init__(self, response):
        self.response = response
        self.prompts: list[str] = []

    def with_structured_output(self, schema):
        def run(prompt_value):
            self.prompts.append(prompt_value.to_string())
            return self.response

        return RunnableLambda(run)


@pytest.fixture
def persona_pdf() -> bytes:
    return make_pdf(PERSONA_RESUME)


class KeywordEmbeddings(Embeddings):
    """Bag-of-words vectors: texts sharing words are similar. Lets retrieval tests run without Ollama.

    Deterministic (crc32, not Python's per-process salted hash) and also records every text it embeds.
    """

    def __init__(self, dims: int = 512):
        self.dims = dims
        self.calls: list[str] = []

    def _embed(self, text: str) -> list[float]:
        self.calls.append(text)
        vector = [0.0] * self.dims
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            if len(word) > 2:
                vector[zlib.crc32(word.encode()) % self.dims] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


@pytest.fixture
def embeddings() -> KeywordEmbeddings:
    return KeywordEmbeddings()


@pytest.fixture
def storage(tmp_path, embeddings) -> Storage:
    """Real SQLite + Chroma in a temp dir, with fake embeddings and the real question bank."""
    return open_storage(Settings(data_dir=tmp_path, embedding_model="keyword-test"), embeddings=embeddings)
