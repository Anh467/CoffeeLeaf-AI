"""SQLite persistence for prediction history."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pipeline import ROOT

DEFAULT_DB_PATH = ROOT / "runs" / "history" / "history.db"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class HistoryDatabase:
    """Thin SQLite wrapper storing metadata paths, never image blobs."""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = Path(db_path) if db_path is not None else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    image_path TEXT NOT NULL,
                    result_path TEXT NOT NULL,
                    processing_time REAL NOT NULL,
                    model TEXT NOT NULL,
                    summary TEXT NOT NULL
                )
                """
            )
            connection.commit()

    def insert(
        self,
        *,
        prediction_id: str,
        filename: str,
        image_path: Path,
        result_path: Path,
        processing_time: float,
        model: str,
        summary: dict[str, Any],
        created_at: str | None = None,
    ) -> dict[str, Any]:
        timestamp = created_at or _utc_now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO predictions (
                    id, filename, created_at, image_path, result_path,
                    processing_time, model, summary
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    prediction_id,
                    filename,
                    timestamp,
                    str(image_path),
                    str(result_path),
                    float(processing_time),
                    model,
                    json.dumps(summary, ensure_ascii=False),
                ),
            )
            connection.commit()
        return self.get(prediction_id)

    def list_items(self, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM predictions
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
                """,
                (int(limit), int(offset)),
            ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def count(self) -> int:
        with self._connect() as connection:
            value = connection.execute("SELECT COUNT(*) AS total FROM predictions").fetchone()
        return int(value["total"]) if value is not None else 0

    def get(self, prediction_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM predictions WHERE id = ?",
                (prediction_id,),
            ).fetchone()
        return None if row is None else self._row_to_dict(row)

    def delete(self, prediction_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM predictions WHERE id = ?",
                (prediction_id,),
            )
            connection.commit()
            return cursor.rowcount > 0

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        payload = dict(row)
        payload["summary"] = json.loads(payload["summary"])
        return payload
