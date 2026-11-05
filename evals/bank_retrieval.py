"""Quick retrieval check for the question bank, on the real local embedding model.

Each query lists the question IDs a human would accept in the top 3. Run it after editing the bank or
changing embedding_text(), and compare hit@1 / hit@3 before and after:

    ollama serve                              # in another terminal
    uv run python evals/bank_retrieval.py

Uses a throwaway index in a temp dir, so it never touches .data/. Phase 6 moves evals like this to LangSmith.
"""

import tempfile
from pathlib import Path

from interview_prep.config import get_settings
from interview_prep.storage import open_storage

CASES = {
    "a time I disagreed with my manager": {"gen-016", "gen-008"},
    "conflict with a coworker": {"gen-008"},
    "biggest failure": {"gen-007"},
    "why this company": {"gen-002"},
    "kafka exactly once delivery": {"be-007", "de-019"},
    "prevent overselling inventory": {"be-006"},
    "zero downtime database migration": {"be-015"},
    "design a feed": {"be-022"},
    "how do I evaluate retrieval quality": {"ai-005"},
    "keep pii out of prompts": {"ai-015"},
    "handling events that arrive late": {"de-011"},
    "SCD type 2": {"de-007"},
}


def main() -> None:
    settings = get_settings().model_copy(update={"data_dir": Path(tempfile.mkdtemp())})
    storage = open_storage(settings)
    hit1 = hit3 = 0
    for query, accepted in CASES.items():
        hits = storage.vectors.search_bank(storage.bank, query, k=3)
        ids = [h.id for h in hits]
        hit1 += ids[0] in accepted
        hit3 += bool(accepted & set(ids))
        mark = "✓" if accepted & set(ids) else "✗"
        print(f"{mark} {query:40} {'  '.join(f'{h.id} {h.score:.2f}' for h in hits)}")
    print(f"\nhit@1 {hit1}/{len(CASES)}   hit@3 {hit3}/{len(CASES)}   ({settings.embedding_model})")


if __name__ == "__main__":
    main()
