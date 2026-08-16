"""
fragment_health.py

Lightweight scanner for Ina's on-disk fragments so she can inspect
corruption issues and decide whether to repair or remove them.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from io_utils import atomic_write_json


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tier_label(root: Path, fragment_path: Path) -> Optional[str]:
    try:
        relative = fragment_path.relative_to(root)
    except ValueError:
        return None

    parts = relative.parts
    if len(parts) <= 1:
        return "root"
    return parts[0]


def _preview_fragment(fragment_path: Path, limit: int) -> str:
    try:
        with fragment_path.open("r", encoding="utf-8", errors="replace") as handle:
            snippet = handle.read(limit)
    except Exception:
        return ""

    snippet = snippet.replace("\n", " ").replace("\r", " ")
    if len(snippet) > limit:
        snippet = snippet[:limit]
    return snippet.strip()


def _recommend_action(error_message: str, size_bytes: Optional[int]) -> str:
    lowered = (error_message or "").lower()
    if any(key in lowered for key in ("unexpected end", "unterminated", "truncated", "eof while parsing")):
        return "attempt_repair"
    if size_bytes is not None and size_bytes < 128:
        return "consider_removal"
    if "line 1 column 1" in lowered:
        return "consider_removal"
    return "inspect"


def scan_fragment_integrity(
    child: str,
    *,
    max_samples: int = 6,
    preview_chars: int = 200,
    max_records: int = 2048,
    max_seconds: float = 8.0,
) -> Optional[Dict[str, Any]]:
    """
    Check one bounded, resumable batch selected by the fragment index.

    Returns a summary dict suitable for publishing into inastate, or None
    if no fragments were found.
    """
    def _scan() -> Optional[Dict[str, Any]]:
        root = Path("AI_Children") / child / "memory" / "fragments"
        if not root.exists():
            return None

        memory_root = root.parent
        index_path = memory_root / "memory_map.sqlite"
        cursor_path = memory_root / "fragment_integrity_cursor.json"
        try:
            cursor_state = json.loads(cursor_path.read_text(encoding="utf-8")) if cursor_path.exists() else {}
        except Exception:
            cursor_state = {}
        cursor = str(cursor_state.get("frag_id") or "") if isinstance(cursor_state, dict) else ""
        try:
            with sqlite3.connect(f"file:{index_path.resolve()}?mode=ro", uri=True, timeout=0.25) as connection:
                rows = connection.execute(
                    "SELECT frag_id, tier, filename FROM fragments "
                    "WHERE frag_id > ? ORDER BY frag_id LIMIT ?",
                    (cursor, max(1, int(max_records))),
                ).fetchall()
                wrapped = bool(cursor and not rows)
                if wrapped:
                    rows = connection.execute(
                        "SELECT frag_id, tier, filename FROM fragments ORDER BY frag_id LIMIT ?",
                        (max(1, int(max_records)),),
                    ).fetchall()
        except (OSError, sqlite3.Error) as exc:
            return {
                "child": child,
                "scanned_at": _now_iso(),
                "checked_this_pass": 0,
                "corrupted_this_pass": 0,
                "total_fragments_checked": 0,
                "corrupted_count": 0,
                "status": "deferred",
                "reason": "index_unavailable",
                "error": str(exc),
            }

        if not rows:
            return {
                "child": child,
                "scanned_at": _now_iso(),
                "checked_this_pass": 0,
                "corrupted_this_pass": 0,
                "total_fragments_checked": 0,
                "corrupted_count": 0,
                "status": "empty",
                "note": "The fragment index contains no records.",
            }

        deadline = time.monotonic() + max(0.01, float(max_seconds))
        total = 0
        corrupted = 0
        samples: List[Dict[str, Any]] = []
        corrupt_entries: List[Dict[str, Any]] = []
        last_frag_id = "" if wrapped else cursor

        for frag_id, tier, filename in rows:
            if time.monotonic() >= deadline:
                break
            last_frag_id = str(frag_id)
            name = str(filename or f"frag_{frag_id}.json")
            tier_name = str(tier or "").strip()
            candidates = ([root / tier_name / name] if tier_name else []) + [root / name]
            path = next((candidate for candidate in candidates if candidate.is_file()), None)
            if path is None:
                continue

            total += 1
            try:
                with path.open("r", encoding="utf-8") as handle:
                    json.load(handle)
            except Exception as exc:
                corrupted += 1
                try:
                    stats = path.stat()
                    size_bytes = stats.st_size
                    modified = datetime.fromtimestamp(stats.st_mtime, timezone.utc).isoformat()
                except Exception:
                    size_bytes = None
                    modified = None
                entry = {
                    "id": str(frag_id),
                    "path": str(path),
                    "filename": path.name,
                    "tier": _tier_label(root, path),
                    "error": str(exc),
                    "reason": "invalid_json",
                    "size_bytes": size_bytes,
                    "modified": modified,
                    "preview": _preview_fragment(path, preview_chars),
                    "recommendation": _recommend_action(str(exc), size_bytes),
                    "detected_at": _now_iso(),
                }
                corrupt_entries.append(entry)
                if len(samples) < max_samples:
                    samples.append(entry)

        if last_frag_id:
            atomic_write_json(cursor_path, {
                "frag_id": last_frag_id,
                "updated_at": _now_iso(),
                "batch_limit": max(1, int(max_records)),
            }, indent=2)

        summary: Dict[str, Any] = {
            "child": child,
            "scanned_at": _now_iso(),
            "checked_this_pass": total,
            "corrupted_this_pass": corrupted,
            # Compatibility aliases now describe this bounded pass, not an
            # exhaustive directory-wide scan.
            "total_fragments_checked": total,
            "corrupted_count": corrupted,
            "status": "attention_needed" if corrupted else "ok",
            "cursor": last_frag_id,
            "wrapped": wrapped,
            "bounded": True,
        }

        if corrupted:
            summary["corrupted_samples"] = samples
            summary["sampled_count"] = len(samples)
            summary["corrupt_entries"] = corrupt_entries
        else:
            summary["note"] = "All scanned fragments loaded cleanly."

        return summary

    try:
        from precision_requests import precision_request
    except Exception:
        precision_request = None

    if precision_request:
        with precision_request(
            task="integrity_check",
            child=child,
            ttl_sec=4.0,
            reason="fragment_integrity_scan",
            integrity_threat=True,
            source="fragment_health",
        ):
            return _scan()

    return _scan()


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan Ina's fragments for corruption.")
    parser.add_argument(
        "--child",
        default=None,
        help="Name of the child/identity to scan (defaults to config.json current_child).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=6,
        help="Maximum number of corrupted fragment samples to record.",
    )
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=200,
        help="Preview snippet length for corrupted fragments.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to write the JSON summary (defaults to stdout only).",
    )

    args = parser.parse_args()

    child = args.child
    if not child:
        cfg_path = Path("config.json")
        if cfg_path.exists():
            try:
                with cfg_path.open("r", encoding="utf-8") as handle:
                    child = json.load(handle).get("current_child")
            except Exception:
                child = None
        if not child:
            parser.error("Unable to determine child; pass --child explicitly.")

    summary = scan_fragment_integrity(
        child,
        max_samples=max(1, int(args.max_samples or 1)),
        preview_chars=max(40, int(args.preview_chars or 40)),
    )

    if summary is None:
        print("No fragment directory found.")
        return

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
