import json
from collections import Counter

import pytest

from interview_prep.question_bank import BANK_DIR, BANK_TRACKS, fingerprint, load_bank
from interview_prep.schemas import SHARED_TRACK, SUPPORTED_TRACKS, Round

PREFIXES = {SHARED_TRACK: "gen", "backend_swe": "be", "ai_llm_engineer": "ai", "data_engineer": "de"}


@pytest.fixture(scope="module")
def bank():
    return load_bank()


def test_bank_size_matches_spec(bank):
    counts = Counter(q.track for q in bank)
    assert counts[SHARED_TRACK] >= 30
    for track in SUPPORTED_TRACKS:
        assert counts[track] >= 25, track


def test_ids_are_unique_and_prefixed_by_track(bank):
    assert len({q.id for q in bank}) == len(bank)
    for q in bank:
        assert q.id.startswith(PREFIXES[q.track] + "-"), q.id


def test_every_track_covers_its_rounds_and_levels(bank):
    shared = [q for q in bank if q.track == SHARED_TRACK]
    assert {q.round for q in shared} == {Round.SCREEN, Round.BEHAVIORAL}
    for track in SUPPORTED_TRACKS:
        questions = [q for q in bank if q.track == track]
        assert {q.round for q in questions} == {Round.TECHNICAL, Round.SYSTEM_DESIGN}, track
        assert {q.difficulty for q in questions} == {1, 2, 3}, track


def test_topics_are_lowercase_tags(bank):
    for q in bank:
        assert all(t == t.lower().strip() and t for t in q.topics), q.id


def test_rejects_duplicate_ids(tmp_path):
    for track in BANK_TRACKS:
        (tmp_path / f"{track}.json").write_text((BANK_DIR / f"{track}.json").read_text())
    shared = json.loads((tmp_path / "shared.json").read_text())
    (tmp_path / "shared.json").write_text(json.dumps(shared + [shared[0]]))
    with pytest.raises(ValueError, match="Duplicate question ID gen-001"):
        load_bank(tmp_path)


def test_rejects_question_in_wrong_file(tmp_path):
    for track in BANK_TRACKS:
        (tmp_path / f"{track}.json").write_text((BANK_DIR / f"{track}.json").read_text())
    backend = json.loads((tmp_path / "backend_swe.json").read_text())
    backend[0]["track"] = "data_engineer"
    (tmp_path / "backend_swe.json").write_text(json.dumps(backend))
    with pytest.raises(ValueError, match="be-001 has track 'data_engineer'"):
        load_bank(tmp_path)


def test_fingerprint_changes_with_content(bank):
    edited = [bank[0].model_copy(update={"question": bank[0].question + " Why?"}), *bank[1:]]
    assert fingerprint(bank) == fingerprint(list(bank))
    assert fingerprint(edited) != fingerprint(bank)
