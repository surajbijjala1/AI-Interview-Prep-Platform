"""Everything the app keeps on disk: SQLite (relational) + Chroma (vectors), both under settings.data_dir."""

from dataclasses import dataclass

from langchain_core.embeddings import Embeddings

from interview_prep.config import Settings, get_settings
from interview_prep.embeddings import get_embeddings
from interview_prep.question_bank import default_bank
from interview_prep.schemas import BankQuestion
from interview_prep.storage.db import Database, check_session_id
from interview_prep.storage.vectors import VectorStore

__all__ = ["Database", "Storage", "VectorStore", "check_session_id", "open_storage"]


@dataclass
class Storage:
    db: Database
    vectors: VectorStore
    bank: tuple[BankQuestion, ...]


def open_storage(
    settings: Settings | None = None,
    *,
    embeddings: Embeddings | None = None,
    bank: tuple[BankQuestion, ...] | None = None,
) -> Storage:
    """Tests pass fake embeddings and a temp data_dir; the defaults are the real Ollama model and .data/."""
    settings = settings or get_settings()
    return Storage(
        db=Database(settings.data_dir / "interview_prep.db"),
        vectors=VectorStore(
            settings.data_dir / "chroma", embeddings or get_embeddings(settings), settings.embedding_model
        ),
        bank=bank if bank is not None else default_bank(),
    )
