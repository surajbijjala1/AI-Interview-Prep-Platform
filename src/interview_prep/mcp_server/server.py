"""The `interview-prep` MCP server: question bank RAG, resume RAG and data deletion, over stdio.

It owns stored data and retrieval but never calls an LLM, so it needs no API keys. Any MCP client can use it:
this app (through langchain-mcp-adapters), Claude Desktop or Claude Code (see docs/mcp-clients.md).

stdio rule: stdout carries the protocol, so nothing here may print(). Use logging (stderr) instead.
"""

import functools
import json
import logging
from pathlib import Path
from typing import Annotated

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field

from interview_prep.ingest import load_redacted_resume
from interview_prep.ingest.redact import scrub_contact_info
from interview_prep.question_bank import BANK_TRACKS
from interview_prep.schemas import BankQuestion, QuestionHit, ResumeChunk, Round
from interview_prep.storage import Storage, check_session_id, open_storage

log = logging.getLogger(__name__)

INSTRUCTIONS = """\
Data and retrieval for the AI Interview Prep app. Search the curated interview question bank (tracks: \
backend_swe, ai_llm_engineer, data_engineer, plus shared screen/behavioral questions), extract a resume PDF with \
contact info redacted, and index/search a session's resume. Embeddings run locally on Ollama (`ollama serve`)."""

READ_ONLY = ToolAnnotations(readOnlyHint=True, openWorldHint=False)
WRITES = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=True, openWorldHint=False)
DESTRUCTIVE = ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=True, openWorldHint=False)


class RedactedResume(BaseModel):
    text: str = Field(description="Resume text with [NAME], [EMAIL], [PHONE], [URL] and [ADDRESS] placeholders.")
    redactions: dict[str, int] = Field(description="How many of each placeholder were inserted.")


class IndexResult(BaseModel):
    session_id: str
    chunks: int
    redactions: dict[str, int] = Field(description="Contact info found and removed before storing (normally none).")


class DeleteResult(BaseModel):
    sessions: int
    resume_chunks: int


def needs_ollama(fn):
    """Turn Ollama's connection error into a message that says how to fix it."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except ConnectionError as exc:
            raise ToolError("Ollama isn't running, and embeddings need it. Start it with `ollama serve`.") from exc

    return wrapper


def build_server(storage: Storage) -> FastMCP:
    mcp = FastMCP("interview-prep", instructions=INSTRUCTIONS)
    questions_by_id = {q.id: q for q in storage.bank}

    # --- question bank ---------------------------------------------------------------------------------------

    @mcp.tool(annotations=READ_ONLY)
    @needs_ollama
    def search_question_bank(
        query: Annotated[str, Field(description="What to look for, e.g. 'kafka delivery guarantees'.")],
        track: Annotated[
            str | None, Field(description="backend_swe, ai_llm_engineer or data_engineer. Shared questions always included.")
        ] = None,
        round: Annotated[Round | None, Field(description="Limit to one interview round.")] = None,
        difficulty: Annotated[int | None, Field(ge=1, le=3, description="1 junior, 2 mid, 3 senior.")] = None,
        exclude_ids: Annotated[list[str] | None, Field(description="Question IDs to skip, e.g. already asked.")] = None,
        k: Annotated[int, Field(ge=1, le=20)] = 5,
    ) -> list[QuestionHit]:
        """Semantic search over the curated interview question bank, with optional filters."""
        if track and track not in BANK_TRACKS:
            raise ToolError(f"Unknown track '{track}'. Use one of: {', '.join(BANK_TRACKS[1:])}")
        return storage.vectors.search_bank(
            storage.bank, query, track=track, round=round, difficulty=difficulty, exclude_ids=exclude_ids, k=k
        )

    @mcp.tool(annotations=READ_ONLY)
    def get_question(question_id: str) -> BankQuestion:
        """Fetch one bank question by ID, including what a good answer covers."""
        if question_id not in questions_by_id:
            raise ToolError(f"No question with ID '{question_id}'")
        return questions_by_id[question_id]

    @mcp.resource("question-bank://{track}", mime_type="application/json")
    def bank_for_track(track: str) -> str:
        """Every bank question for one track ('shared' holds screen and behavioral questions)."""
        if track not in BANK_TRACKS:
            raise ValueError(f"Unknown track '{track}'")
        return json.dumps([q.model_dump(mode="json") for q in storage.bank if q.track == track], indent=2)

    # --- resume ----------------------------------------------------------------------------------------------

    @mcp.tool(annotations=READ_ONLY)
    def extract_resume_text(
        pdf_path: Annotated[str, Field(description="Absolute path to a resume PDF on this machine.")],
    ) -> RedactedResume:
        """Extract text from a resume PDF with contact info (name, email, phone, links, address) redacted.

        Deterministic, no LLM. The PDF is read, not stored.
        """
        path = Path(pdf_path).expanduser()
        if path.suffix.lower() != ".pdf" or not path.is_file():
            raise ToolError(f"Not a PDF file: {pdf_path}")
        try:
            redacted = load_redacted_resume(path.read_bytes())
        except ValueError as exc:
            raise ToolError(str(exc)) from exc
        return RedactedResume(text=redacted.text, redactions=dict(redacted.counts))

    @mcp.tool(annotations=WRITES)
    @needs_ollama
    def index_resume(
        session_id: Annotated[str, Field(description="Session to attach the resume to; created if new.")],
        text: Annotated[str, Field(description="Redacted resume text, e.g. from extract_resume_text.")],
    ) -> IndexResult:
        """Chunk and embed a session's resume for search_resume. Replaces any resume already indexed for it."""
        check_session_id(session_id)
        if not text.strip():
            raise ToolError("Resume text is empty")
        # Defense in depth: callers should send redacted text, but never store contact info if they didn't.
        scrubbed = scrub_contact_info(text)
        chunks = storage.vectors.index_resume(session_id, scrubbed.text)  # first: fails if Ollama is down
        storage.db.ensure_session(session_id)
        return IndexResult(session_id=session_id, chunks=chunks, redactions=dict(scrubbed.counts))

    @mcp.tool(annotations=READ_ONLY)
    @needs_ollama
    def search_resume(
        session_id: str,
        query: Annotated[str, Field(description="e.g. 'experience with message queues'.")],
        k: Annotated[int, Field(ge=1, le=10)] = 3,
    ) -> list[ResumeChunk]:
        """Find the resume excerpts most relevant to a topic, for personalizing questions and citing provenance."""
        if not storage.db.session_exists(check_session_id(session_id)):
            raise ToolError(f"No session '{session_id}'. Call index_resume first.")
        return storage.vectors.search_resume(session_id, query, k=k)

    # --- deletion --------------------------------------------------------------------------------------------

    @mcp.tool(annotations=DESTRUCTIVE)
    def delete_session(session_id: str) -> DeleteResult:
        """Permanently delete one session and its resume chunks."""
        chunks = storage.vectors.delete_session(check_session_id(session_id))
        sessions = int(storage.db.delete_session(session_id))
        return DeleteResult(sessions=sessions, resume_chunks=chunks)

    @mcp.tool(annotations=DESTRUCTIVE)
    def delete_all_data(
        confirm: Annotated[bool, Field(description="Must be true. Deletes every session and resume chunk.")],
    ) -> DeleteResult:
        """Permanently delete all user data: every session and indexed resume. The question bank is kept."""
        if not confirm:
            raise ToolError("Nothing deleted: pass confirm=true to delete all data.")
        chunks = storage.vectors.delete_all_user_data()
        sessions = storage.db.delete_all()
        return DeleteResult(sessions=sessions, resume_chunks=chunks)

    return mcp


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")  # stderr
    build_server(open_storage()).run(transport="stdio")
