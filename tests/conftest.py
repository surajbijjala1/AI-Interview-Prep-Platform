from pathlib import Path

import pytest
from fpdf import FPDF
from langchain_core.runnables import RunnableLambda

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
