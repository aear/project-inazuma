"""Bounded compact navigation index for Experience Cycles."""
from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from typing import Any, Mapping

from storage_layout import format_child_path, root_is_writable


DEFAULT_MAX_ROWS = 100_000


class ExperienceCycleIndex:
    def __init__(
        self, child: str, durable_root: Path, *, config: Mapping[str, Any] | None = None,
        enable_fast: bool = True,
    ) -> None:
        self.child = str(child)
        self.config = dict(config or {})
        raw = self.config.get("experience_cycle_storage")
        raw = raw if isinstance(raw, Mapping) else {}
        layout = self.config.get("storage_layout")
        layout = layout if isinstance(layout, Mapping) else {}
        self.max_rows = max(1000, int(raw.get("max_index_rows", DEFAULT_MAX_ROWS)))
        path = Path(durable_root) / "cycle_index.sqlite3"
        current_only = bool(layout.get("fast_runtime_current_child_only", True))
        current_child = self.config.get("current_child")
        if enable_fast and (not current_only or not current_child or str(current_child) == self.child):
            fast_index = format_child_path(layout.get("fast_index_root"), self.child)
            if fast_index is not None and root_is_writable(fast_index):
                path = fast_index / "experience_cycles.sqlite3"
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path), timeout=5.0)
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA synchronous=NORMAL")
        return connection

    def _ensure_schema(self) -> None:
        with self._connect() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS cycles ("
                "cycle_id TEXT PRIMARY KEY, parent_cycle_id TEXT, domain TEXT NOT NULL, "
                "stage TEXT NOT NULL, intent TEXT NOT NULL, references_json TEXT NOT NULL, "
                "manifest_path TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL)"
            )
            connection.execute("CREATE INDEX IF NOT EXISTS idx_cycles_domain_updated ON cycles(domain, updated_at DESC)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_cycles_parent ON cycles(parent_cycle_id)")

    def upsert(self, cycle: Mapping[str, Any], manifest_path: Path) -> None:
        cycle_id = str(cycle.get("cycle_id") or "")
        if not cycle_id:
            raise ValueError("cycle index entry needs cycle_id")
        references = list(cycle.get("payload_references") or ())[:64]
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO cycles(cycle_id,parent_cycle_id,domain,stage,intent,references_json,manifest_path,created_at,updated_at) "
                "VALUES(?,?,?,?,?,?,?,?,?) ON CONFLICT(cycle_id) DO UPDATE SET "
                "parent_cycle_id=excluded.parent_cycle_id,domain=excluded.domain,stage=excluded.stage,"
                "intent=excluded.intent,references_json=excluded.references_json,manifest_path=excluded.manifest_path,updated_at=excluded.updated_at",
                (
                    cycle_id, cycle.get("parent_cycle_id"), str(cycle.get("domain") or "unknown")[:80],
                    str(cycle.get("stage") or "intent")[:32], str(cycle.get("intent") or "")[:1000],
                    json.dumps(references, ensure_ascii=False, separators=(",", ":")), str(manifest_path),
                    str(cycle.get("created_at") or ""), str(cycle.get("updated_at") or cycle.get("created_at") or ""),
                ),
            )
            count = int(connection.execute("SELECT COUNT(*) FROM cycles").fetchone()[0])
            excess = count - self.max_rows
            if excess > 0:
                connection.execute(
                    "DELETE FROM cycles WHERE cycle_id IN (SELECT cycle_id FROM cycles ORDER BY updated_at ASC LIMIT ?)",
                    (excess,),
                )

    def recent(self, *, limit: int = 50, domain: str | None = None) -> list[dict[str, Any]]:
        bounded = max(1, min(500, int(limit)))
        query = "SELECT cycle_id,parent_cycle_id,domain,stage,intent,references_json,manifest_path,created_at,updated_at FROM cycles"
        params: tuple[Any, ...]
        if domain is None:
            query += " ORDER BY updated_at DESC LIMIT ?"
            params = (bounded,)
        else:
            query += " WHERE domain=? ORDER BY updated_at DESC LIMIT ?"
            params = (str(domain), bounded)
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [
            {
                "cycle_id": row[0], "parent_cycle_id": row[1], "domain": row[2], "stage": row[3],
                "intent": row[4], "payload_references": json.loads(row[5]), "manifest_path": row[6],
                "created_at": row[7], "updated_at": row[8],
            }
            for row in rows
        ]


__all__ = ["ExperienceCycleIndex", "DEFAULT_MAX_ROWS"]
