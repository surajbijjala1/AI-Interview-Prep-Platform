from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from ..llm import get_chat_model
from ..schemas import ResumeProfile
from .pdf import extract_text
from .redact import Redacted, redact_resume

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You extract structured data from resumes. Use only what the resume states. "
            "Contact details were replaced with placeholders like [NAME] or [EMAIL]; ignore them.",
        ),
        ("human", "Resume:\n---\n{resume_text}\n---"),
    ]
)


@dataclass
class ParsedResume:
    redacted: Redacted  # exactly what the LLM saw
    profile: ResumeProfile


def load_redacted_resume(pdf_bytes: bytes) -> Redacted:
    return redact_resume(extract_text(pdf_bytes))


def parse_resume(pdf_bytes: bytes, llm: BaseChatModel | None = None) -> ParsedResume:
    redacted = load_redacted_resume(pdf_bytes)
    chain = PROMPT | (llm or get_chat_model()).with_structured_output(ResumeProfile)
    return ParsedResume(redacted, chain.invoke({"resume_text": redacted.text}))
