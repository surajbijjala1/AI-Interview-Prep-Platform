"""Load and validate the curated question bank in data/question_bank/*.json."""

import hashlib
import json
from functools import lru_cache
from pathlib import Path

from pydantic import TypeAdapter

from interview_prep.config import ROOT
from interview_prep.schemas import SHARED_TRACK, SUPPORTED_TRACKS, BankQuestion

BANK_DIR = ROOT / "data" / "question_bank"
BANK_TRACKS = (SHARED_TRACK, *SUPPORTED_TRACKS)  # one JSON file per entry, named after it

# Bump when embedding_text() changes, so existing indexes are rebuilt.
EMBEDDING_TEXT_VERSION = 1

_questions = TypeAdapter(list[BankQuestion])


def load_bank(bank_dir: Path = BANK_DIR) -> list[BankQuestion]:
    """All questions, validated. Raises ValueError on a bad file, a wrong track or a duplicate ID."""
    questions: list[BankQuestion] = []
    for track in BANK_TRACKS:
        path = bank_dir / f"{track}.json"
        try:
            file_questions = _questions.validate_json(path.read_bytes())
        except Exception as exc:
            raise ValueError(f"{path.name}: {exc}") from exc
        for q in file_questions:
            if q.track != track:
                raise ValueError(f"{path.name}: {q.id} has track '{q.track}', expected '{track}'")
        questions.extend(file_questions)

    seen: set[str] = set()
    for q in questions:
        if q.id in seen:
            raise ValueError(f"Duplicate question ID {q.id}")
        seen.add(q.id)
    return questions


@lru_cache
def default_bank() -> tuple[BankQuestion, ...]:
    return tuple(load_bank())


def embedding_text(q: BankQuestion) -> str:
    """What gets embedded: the question plus its tags, so 'Kafka' matches a question tagged 'kafka'."""
    return f"{q.question}\nTopics: {', '.join(q.topics)}"


def fingerprint(questions: list[BankQuestion] | tuple[BankQuestion, ...]) -> str:
    """Changes whenever any question or the embedding text format changes."""
    payload = json.dumps(
        [EMBEDDING_TEXT_VERSION, [q.model_dump(mode="json") for q in questions]], sort_keys=True
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]
