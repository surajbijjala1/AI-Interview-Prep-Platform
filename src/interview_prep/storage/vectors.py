"""Chroma collections for RAG: the question bank and per-session resume chunks.

Each collection records which embedding model built it (and, for the bank, a fingerprint of its content).
Vectors from different models aren't comparable, so a mismatch means the collection is dropped and rebuilt.
"""

import logging
import re
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from interview_prep.question_bank import embedding_text, fingerprint
from interview_prep.schemas import SHARED_TRACK, BankQuestion, QuestionHit, ResumeChunk, Round

log = logging.getLogger(__name__)

BANK = "question_bank"
RESUMES = "resume_chunks"

# A chunk should hold about one section or role, so a search hit is a usable excerpt. Sections are split first
# because the splitter alone packs small neighbors together (SKILLS + EDUCATION), and then the bigger merged
# chunk wins unrelated queries. Long sections are then split by size.
resume_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)
PLACEHOLDER = re.compile(r"\[(?:NAME|EMAIL|PHONE|URL|ADDRESS)\]")
SECTION_NAMES = {
    "summary", "profile", "about", "objective", "experience", "work experience", "professional experience",
    "employment", "skills", "technical skills", "education", "projects", "certifications", "publications",
    "awards", "leadership", "volunteering", "languages", "interests",
}


ALL_CAPS_HEADING = re.compile(r"[A-Z][A-Z &/]{3,40}")  # "SKILLS", "WORK EXPERIENCE", "AWARDS & HONORS"


def _is_heading(line: str) -> bool:
    words = line.strip().rstrip(":")
    return words.lower() in SECTION_NAMES or bool(ALL_CAPS_HEADING.fullmatch(words))


def chunk_resume(text: str) -> list[str]:
    """Split redacted resume text into section-sized chunks, dropping contact-header lines (pure noise)."""
    sections: list[list[str]] = [[]]
    for line in text.splitlines():
        other_words = re.findall(r"[A-Za-z]{2,}", PLACEHOLDER.sub(" ", line))
        if PLACEHOLDER.search(line) and len(other_words) < 4:
            continue  # e.g. "[NAME] | [EMAIL] | [PHONE]" or "[ADDRESS], San Francisco, [ADDRESS]"
        if _is_heading(line) and any(l.strip() for l in sections[-1]):
            sections.append([])
        sections[-1].append(line)
    return [
        chunk
        for section in sections
        if (body := "\n".join(l for l in section if l.strip()))
        for chunk in resume_splitter.split_text(body)
    ]


class VectorStore:
    def __init__(self, path: Path, embeddings: Embeddings, embedding_model: str):
        self.client = chromadb.PersistentClient(path, settings=ChromaSettings(anonymized_telemetry=False))
        self.embeddings = embeddings
        self.embedding_model = embedding_model
        self._stores: dict[str, Chroma] = {}

    # --- collections ---------------------------------------------------------------------------------------

    def _store(self, name: str, **extra_metadata: str) -> Chroma:
        """Open a collection, dropping it first if it was built by another model or from other content."""
        wanted = {"embedding_model": self.embedding_model, **extra_metadata}
        if name in self._stores and self._metadata_matches(name, wanted):
            return self._stores[name]
        if name in {c.name for c in self.client.list_collections()} and not self._metadata_matches(name, wanted):
            log.warning("Collection %s is stale (want %s); rebuilding", name, wanted)
            self.client.delete_collection(name)
        self._stores[name] = Chroma(
            client=self.client,
            collection_name=name,
            embedding_function=self.embeddings,
            collection_metadata=wanted,
            # Cosine distance, so relevance scores are 1 - distance in 0..1.
            collection_configuration={"hnsw": {"space": "cosine"}},
        )
        return self._stores[name]

    def _metadata_matches(self, name: str, wanted: dict[str, str]) -> bool:
        metadata = self.client.get_collection(name).metadata or {}
        return all(metadata.get(k) == v for k, v in wanted.items())

    def _count(self, name: str) -> int:
        return self.client.get_collection(name).count()

    # --- question bank ---------------------------------------------------------------------------------------

    def sync_bank(self, questions: list[BankQuestion] | tuple[BankQuestion, ...]) -> bool:
        """Make the index match the bank. Returns True if it had to (re)embed. Cheap when nothing changed."""
        return self._bank(questions)[1]

    def _bank(self, questions: list[BankQuestion] | tuple[BankQuestion, ...]) -> tuple[Chroma, bool]:
        store = self._store(BANK, fingerprint=fingerprint(questions))
        if self._count(BANK) == len(questions):
            return store, False
        store.reset_collection()  # empty, or a partial index from an interrupted build: start over
        store.add_documents(
            [
                Document(
                    page_content=embedding_text(q),
                    metadata={"qid": q.id, "track": q.track, "round": q.round.value, "difficulty": q.difficulty},
                )
                for q in questions
            ],
            ids=[q.id for q in questions],
        )
        log.info("Embedded %d bank questions with %s", len(questions), self.embedding_model)
        return store, True

    def search_bank(
        self,
        questions: list[BankQuestion] | tuple[BankQuestion, ...],
        query: str,
        *,
        track: str | None = None,
        round: Round | None = None,
        difficulty: int | None = None,
        exclude_ids: list[str] | None = None,
        k: int = 5,
    ) -> list[QuestionHit]:
        store, _ = self._bank(questions)
        clauses: list[dict] = []
        if track:
            # Behavioral and screen questions are shared by every track.
            clauses.append({"track": {"$in": [track, SHARED_TRACK]}})
        if round:
            clauses.append({"round": Round(round).value})
        if difficulty:
            clauses.append({"difficulty": difficulty})
        if exclude_ids:
            clauses.append({"qid": {"$nin": list(exclude_ids)}})
        where = clauses[0] if len(clauses) == 1 else {"$and": clauses} if clauses else None

        by_id = {q.id: q for q in questions}
        results = store.similarity_search_with_relevance_scores(query, k=k, filter=where)
        return [
            QuestionHit(**by_id[doc.metadata["qid"]].model_dump(), score=round_score(score))
            for doc, score in results
        ]

    # --- resumes ---------------------------------------------------------------------------------------------

    def index_resume(self, session_id: str, text: str) -> int:
        """Replace the session's resume chunks. Returns the number of chunks stored."""
        chunks = chunk_resume(text)
        if not chunks:
            raise ValueError("No resume content left to index after removing contact lines")
        store = self._store(RESUMES)
        # Upsert first, then trim leftovers from a longer previous version, so a failed embedding call
        # (Ollama down) leaves the old index intact instead of an empty one.
        store.add_documents(
            [Document(page_content=c, metadata={"session_id": session_id, "chunk_index": i}) for i, c in enumerate(chunks)],
            ids=[f"{session_id}:{i}" for i in range(len(chunks))],
        )
        store.delete(where={"$and": [{"session_id": session_id}, {"chunk_index": {"$gte": len(chunks)}}]})
        return len(chunks)

    def search_resume(self, session_id: str, query: str, k: int = 3) -> list[ResumeChunk]:
        results = self._store(RESUMES).similarity_search_with_relevance_scores(
            query, k=k, filter={"session_id": session_id}
        )
        return [
            ResumeChunk(text=doc.page_content, chunk_index=doc.metadata["chunk_index"], score=round_score(score))
            for doc, score in results
        ]

    def resume_chunk_count(self, session_id: str) -> int:
        return len(self._store(RESUMES).get(where={"session_id": session_id}, include=[])["ids"])

    def delete_session(self, session_id: str) -> int:
        store = self._store(RESUMES)
        count = self.resume_chunk_count(session_id)
        store.delete(where={"session_id": session_id})
        return count

    def delete_all_user_data(self) -> int:
        """Drop every resume chunk. The question bank is shipped content, not user data, so it stays."""
        store = self._store(RESUMES)
        count = self._count(RESUMES)
        store.reset_collection()
        return count


def round_score(score: float) -> float:
    return round(max(0.0, min(1.0, score)), 3)
