"""SQLite-backed storage for the learned English/native meaning map.

JSON remains an import/export compatibility format. Runtime callers use this
store once it exists so the complete mapping is no longer parsed from a large
document on every lookup pass.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable

from storage_layout import fast_runtime_path


SCHEMA_VERSION = 1


def sqlite_path_for(json_path: Path) -> Path:
    fallback = json_path.with_suffix(".sqlite")
    parts = json_path.parts
    try:
        child = parts[parts.index("AI_Children") + 1]
    except (ValueError, IndexError):
        return fallback
    return fast_runtime_path(
        child, fallback.name, fallback, subdir="index",
        root_keys=("fast_index_root", "fast_runtime_root", "fast_root"),
    )


def _values(values: Iterable[Any] | None, limit: int = 256) -> list[str]:
    result = []
    for value in values or ():
        item = str(value or "").strip().casefold()
        if item and item not in result:
            result.append(item[:500])
        if len(result) >= limit:
            break
    return result


def write_text_vocab_store(path: Path, payload: Dict[str, Any]) -> None:
    """Atomically replace the logical mapping inside one SQLite transaction."""
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=30.0)
    try:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS evaluated (
                word TEXT PRIMARY KEY,
                value_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS links (
                word TEXT NOT NULL,
                symbol TEXT NOT NULL,
                rank INTEGER NOT NULL,
                value_json TEXT NOT NULL,
                PRIMARY KEY (word, symbol)
            );
            CREATE INDEX IF NOT EXISTS links_symbol_idx ON links(symbol, rank);
            """
        )
        with connection:
            connection.execute("DELETE FROM metadata")
            connection.execute("DELETE FROM evaluated")
            connection.execute("DELETE FROM links")
            metadata = {key: value for key, value in payload.items() if key not in {"evaluated", "links"}}
            metadata["sqlite_schema_version"] = SCHEMA_VERSION
            connection.executemany(
                "INSERT INTO metadata(key, value_json) VALUES (?, ?)",
                ((key, json.dumps(value, ensure_ascii=False)) for key, value in metadata.items()),
            )
            evaluated = payload.get("evaluated") if isinstance(payload.get("evaluated"), dict) else {}
            connection.executemany(
                "INSERT INTO evaluated(word, value_json) VALUES (?, ?)",
                ((str(word), json.dumps(value, ensure_ascii=False)) for word, value in evaluated.items()),
            )
            rows = []
            for rank, link in enumerate(payload.get("links") or []):
                if not isinstance(link, dict) or not link.get("word") or not link.get("symbol"):
                    continue
                rows.append((str(link["word"]), str(link["symbol"]), rank, json.dumps(link, ensure_ascii=False)))
            connection.executemany(
                "INSERT INTO links(word, symbol, rank, value_json) VALUES (?, ?, ?, ?)", rows
            )
    finally:
        connection.close()


def load_text_vocab_store(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5.0)
    try:
        payload = {
            str(key): json.loads(value)
            for key, value in connection.execute("SELECT key, value_json FROM metadata")
            if key != "sqlite_schema_version"
        }
        payload["evaluated"] = {
            str(word): json.loads(value)
            for word, value in connection.execute("SELECT word, value_json FROM evaluated")
        }
        payload["links"] = [
            json.loads(value)
            for (value,) in connection.execute("SELECT value_json FROM links ORDER BY rank")
        ]
        return payload
    except (sqlite3.Error, ValueError, json.JSONDecodeError):
        return {}
    finally:
        connection.close()


def load_text_vocab_store_subset(
    path: Path, *, words: Iterable[Any] | None = None,
    symbols: Iterable[Any] | None = None,
) -> Dict[str, Any]:
    """Load only requested indexed mappings for latency-sensitive realisation."""
    selected_words = _values(words)
    selected_symbols = _values(symbols)
    if not selected_words and not selected_symbols or not path.exists():
        return {}
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5.0)
    try:
        payload = {
            str(key): json.loads(value)
            for key, value in connection.execute("SELECT key, value_json FROM metadata")
            if key != "sqlite_schema_version"
        }
        clauses = []
        parameters: list[str] = []
        if selected_words:
            clauses.append("word IN (" + ",".join("?" for _ in selected_words) + ")")
            parameters.extend(selected_words)
        if selected_symbols:
            clauses.append("symbol IN (" + ",".join("?" for _ in selected_symbols) + ")")
            parameters.extend(selected_symbols)
        query = "SELECT value_json FROM links WHERE " + " OR ".join(clauses) + " ORDER BY rank"
        payload["links"] = [json.loads(value) for (value,) in connection.execute(query, parameters)]
        if selected_words:
            placeholders = ",".join("?" for _ in selected_words)
            payload["evaluated"] = {
                str(word): json.loads(value)
                for word, value in connection.execute(
                    f"SELECT word, value_json FROM evaluated WHERE word IN ({placeholders})",
                    selected_words,
                )
            }
        else:
            payload["evaluated"] = {}
        payload["subset"] = {"words": selected_words, "symbols": selected_symbols}
        return payload
    except (sqlite3.Error, ValueError, json.JSONDecodeError):
        return {}
    finally:
        connection.close()
