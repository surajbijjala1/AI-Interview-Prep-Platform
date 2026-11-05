"""SQLite for relational data. The schema grows through numbered migrations tracked in PRAGMA user_version."""

import re
import sqlite3
from pathlib import Path

# Append only: never edit a migration that has shipped, add a new one instead.
# Later phases add research cache, turns, grades and reports.
MIGRATIONS = [
    # 1 (Phase 2): an interview session. setup_json holds the confirmed SessionSetup once Phase 4 creates sessions.
    """
    CREATE TABLE sessions (
        id          TEXT PRIMARY KEY,
        created_at  TEXT NOT NULL DEFAULT (datetime('now')),
        status      TEXT NOT NULL DEFAULT 'setup' CHECK (status IN ('setup', 'in_progress', 'done')),
        setup_json  TEXT
    );
    """,
]

SESSION_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def check_session_id(session_id: str) -> str:
    if not SESSION_ID.fullmatch(session_id):
        raise ValueError("session_id must be 1-64 letters, digits, '-' or '_'")
    return session_id


class Database:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._migrate()

    def _migrate(self) -> None:
        version = self.conn.execute("PRAGMA user_version").fetchone()[0]
        for number, sql in enumerate(MIGRATIONS[version:], start=version + 1):
            with self.conn:  # each migration commits atomically, or not at all
                self.conn.executescript(f"BEGIN; {sql} PRAGMA user_version = {number}; COMMIT;")

    @property
    def schema_version(self) -> int:
        return self.conn.execute("PRAGMA user_version").fetchone()[0]

    def ensure_session(self, session_id: str) -> None:
        with self.conn:
            self.conn.execute("INSERT OR IGNORE INTO sessions (id) VALUES (?)", (check_session_id(session_id),))

    def session_exists(self, session_id: str) -> bool:
        return self.conn.execute("SELECT 1 FROM sessions WHERE id = ?", (session_id,)).fetchone() is not None

    def delete_session(self, session_id: str) -> bool:
        with self.conn:
            return self.conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,)).rowcount > 0

    def delete_all(self) -> int:
        with self.conn:
            return self.conn.execute("DELETE FROM sessions").rowcount

    def close(self) -> None:
        self.conn.close()
