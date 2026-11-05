import pytest

from interview_prep.config import Settings
from interview_prep.ingest.redact import redact_resume
from interview_prep.schemas import SHARED_TRACK, Round
from interview_prep.storage import open_storage
from interview_prep.storage.db import MIGRATIONS, Database
from interview_prep.storage.vectors import chunk_resume
from tests.conftest import PERSONA_RESUME, KeywordEmbeddings

# --- SQLite ------------------------------------------------------------------------------------------------------


def test_migrations_run_once(tmp_path):
    db = Database(tmp_path / "app.db")
    assert db.schema_version == len(MIGRATIONS)
    db.ensure_session("s1")
    db.close()
    reopened = Database(tmp_path / "app.db")  # migrations already applied: data kept, nothing re-run
    assert reopened.schema_version == len(MIGRATIONS)
    assert reopened.session_exists("s1")


def test_session_lifecycle(storage):
    storage.db.ensure_session("s1")
    storage.db.ensure_session("s1")  # idempotent
    assert storage.db.session_exists("s1")
    assert storage.db.delete_session("s1") is True
    assert storage.db.delete_session("s1") is False


@pytest.mark.parametrize("bad", ["", "a b", "../etc", "x" * 65])
def test_rejects_bad_session_ids(storage, bad):
    with pytest.raises(ValueError):
        storage.db.ensure_session(bad)


# --- question bank index ---------------------------------------------------------------------------------------


def test_bank_is_embedded_once(storage, embeddings):
    assert storage.vectors.sync_bank(storage.bank) is True
    calls = len(embeddings.calls)
    assert calls == len(storage.bank)
    assert storage.vectors.sync_bank(storage.bank) is False
    assert len(embeddings.calls) == calls


def test_bank_rebuilds_when_content_changes(storage):
    storage.vectors.sync_bank(storage.bank)
    edited = (storage.bank[0].model_copy(update={"question": "A brand new question?"}), *storage.bank[1:])
    assert storage.vectors.sync_bank(edited) is True
    hit = storage.vectors.search_bank(edited, "brand new question", k=1)[0]
    assert hit.id == edited[0].id and hit.question == "A brand new question?"


def test_bank_rebuilds_when_embedding_model_changes(tmp_path):
    first = open_storage(Settings(data_dir=tmp_path, embedding_model="model-a"), embeddings=KeywordEmbeddings())
    first.vectors.sync_bank(first.bank)
    same = open_storage(Settings(data_dir=tmp_path, embedding_model="model-a"), embeddings=KeywordEmbeddings())
    assert same.vectors.sync_bank(same.bank) is False
    other = open_storage(Settings(data_dir=tmp_path, embedding_model="model-b"), embeddings=KeywordEmbeddings())
    assert other.vectors.sync_bank(other.bank) is True


def test_search_finds_relevant_question(storage):
    hits = storage.vectors.search_bank(storage.bank, "kafka exactly once delivery semantics", track="backend_swe")
    assert hits[0].id == "be-007"
    assert 0 < hits[0].score <= 1
    assert hits == sorted(hits, key=lambda h: h.score, reverse=True)


def test_track_filter_includes_shared_questions(storage):
    hits = storage.vectors.search_bank(storage.bank, "tell me about a time", track="data_engineer", k=20)
    assert {h.track for h in hits} <= {"data_engineer", SHARED_TRACK}
    assert any(h.track == SHARED_TRACK for h in hits)


def test_round_difficulty_and_exclude_filters(storage):
    hits = storage.vectors.search_bank(
        storage.bank, "design a system", track="backend_swe", round=Round.SYSTEM_DESIGN, difficulty=3, k=20
    )
    assert hits and all(h.round == Round.SYSTEM_DESIGN and h.difficulty == 3 for h in hits)
    excluded = [h.id for h in hits[:2]]
    rest = storage.vectors.search_bank(
        storage.bank, "design a system", track="backend_swe", round="system_design", difficulty=3,
        exclude_ids=excluded, k=20,
    )
    assert not {h.id for h in rest} & set(excluded)
    assert len(rest) == len(hits) - 2


# --- resumes -----------------------------------------------------------------------------------------------------


def test_resume_index_search_and_delete(storage):
    text = redact_resume(PERSONA_RESUME).text
    chunks = storage.vectors.index_resume("s1", text)
    assert chunks >= 2
    assert storage.vectors.index_resume("s1", text) == chunks  # re-indexing replaces, doesn't duplicate
    assert storage.vectors.resume_chunk_count("s1") == chunks

    storage.vectors.index_resume("s2", "Data engineer who builds Airflow pipelines.")
    hit = storage.vectors.search_resume("s1", "kafka rabbitmq migration", k=1)[0]
    assert "Kafka" in hit.text
    assert all("Airflow" not in h.text for h in storage.vectors.search_resume("s1", "airflow pipelines", k=10))

    assert storage.vectors.delete_session("s1") == chunks
    assert storage.vectors.resume_chunk_count("s1") == 0
    assert storage.vectors.resume_chunk_count("s2") == 1


def test_delete_all_user_data_keeps_question_bank(storage):
    storage.vectors.sync_bank(storage.bank)
    storage.vectors.index_resume("s1", "Some resume text about Python.")
    assert storage.vectors.delete_all_user_data() == 1
    assert storage.vectors.resume_chunk_count("s1") == 0
    assert storage.vectors.sync_bank(storage.bank) is False


def test_resume_chunks_follow_sections_and_skip_contact_header(persona_pdf):
    from interview_prep.ingest import load_redacted_resume

    chunks = chunk_resume(load_redacted_resume(persona_pdf).text)
    assert [c.split("\n")[0] for c in chunks] == ["SUMMARY", "EXPERIENCE", "SKILLS", "EDUCATION"]
    assert not any("[EMAIL]" in c or "[PHONE]" in c for c in chunks)
    assert "Projects at [URL]" in chunks[1]  # a placeholder inside real content is kept


def test_resume_chunking_handles_title_case_headings_and_long_sections():
    bullets = "\n".join(f"- Shipped feature number {i} with measurable impact on latency." for i in range(20))
    chunks = chunk_resume(f"Experience\nEngineer, Acme\n{bullets}\nEducation:\nB.S., State University")
    assert len(chunks) > 3 and chunks[0].startswith("Experience")
    assert chunks[-1] == "Education:\nB.S., State University"
    assert all(len(c) <= 400 for c in chunks)


def test_all_caps_multiword_headings_start_sections():
    chunks = chunk_resume("WORK EXPERIENCE\nEngineer at Acme, built APIs.\nAWARDS & HONORS\nHackathon winner, 2022.")
    assert [c.split("\n")[0] for c in chunks] == ["WORK EXPERIENCE", "AWARDS & HONORS"]


def test_reindexing_a_shorter_resume_trims_old_chunks(storage):
    long_resume = "\n".join(f"SECTION{chr(65 + i)}\nDetails about area {i} of my work." for i in range(5))
    assert storage.vectors.index_resume("s1", long_resume) == 5
    assert storage.vectors.index_resume("s1", "SUMMARY\nShort resume now.") == 1
    assert storage.vectors.resume_chunk_count("s1") == 1


def test_failed_reindex_keeps_previous_chunks(storage, embeddings, monkeypatch):
    storage.vectors.index_resume("s1", "SUMMARY\nPython developer.")

    def ollama_down(texts):
        raise ConnectionError("Failed to connect to Ollama.")

    monkeypatch.setattr(embeddings, "embed_documents", ollama_down)
    with pytest.raises(ConnectionError):
        storage.vectors.index_resume("s1", "SUMMARY\nA newer resume.")
    assert storage.vectors.resume_chunk_count("s1") == 1


def test_contact_only_resume_is_rejected(storage):
    with pytest.raises(ValueError, match="No resume content"):
        storage.vectors.index_resume("s1", "[NAME] | [EMAIL] | [PHONE]")
