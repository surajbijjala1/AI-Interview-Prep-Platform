from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from ..llm import get_chat_model
from ..schemas import ParsedJD
from .redact import redact_jd

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You extract structured data from job descriptions. Use only what the text states; "
            "leave optional fields null rather than guessing.",
        ),
        ("human", "Job description:\n---\n{job_description}\n---"),
    ]
)


def parse_jd(text: str, llm: BaseChatModel | None = None) -> ParsedJD:
    if not text.strip():
        raise ValueError("The job description is empty.")
    chain = PROMPT | (llm or get_chat_model()).with_structured_output(ParsedJD)
    return chain.invoke({"job_description": redact_jd(text).text})
