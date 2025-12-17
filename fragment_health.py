"""
fragment_health.py

Lightweight scanner for Ina's on-disk fragments so she can inspect
corruption issues and decide whether to repair or remove them.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


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
) -> Optional[Dict[str, Any]]:
    """
    Scan the child's fragment directory for corrupted JSON fragments.

    Returns a summary dict suitable for publishing into inastate, or None
    if no fragments were found.
    """
    root = Path("AI_Children") / child / "memory" / "fragments"
    if not root.exists():
        return None

    fragment_paths = sorted(root.rglob("frag_*.json"))
    if not fragment_paths:
        return {
            "child": child,
            "scanned_at": _now_iso(),
            "total_fragments_checked": 0,
            "corrupted_count": 0,
            "status": "empty",
            "note": "No fragment files found to scan.",
        }

    total = 0
    corrupted = 0
    samples: List[Dict[str, Any]] = []

    for path in fragment_paths:
        if not path.is_file():
            continue

        total += 1
        try:
            with path.open("r", encoding="utf-8") as handle:
                json.load(handle)
        except Exception as exc:
            corrupted += 1
            if len(samples) < max_samples:
                try:
                    stats = path.stat()
                    size_bytes = stats.st_size
                    modified = datetime.fromtimestamp(stats.st_mtime, timezone.utc).isoformat()
                except Exception:
                    size_bytes = None
                    modified = None

                samples.append(
                    {
                        "path": str(path),
                        "filename": path.name,
                        "tier": _tier_label(root, path),
                        "error": str(exc),
                        "size_bytes": size_bytes,
                        "modified": modified,
                        "preview": _preview_fragment(path, preview_chars),
                        "recommendation": _recommend_action(str(exc), size_bytes),
                    }
                )

    summary: Dict[str, Any] = {
        "child": child,
        "scanned_at": _now_iso(),
        "total_fragments_checked": total,
        "corrupted_count": corrupted,
        "status": "attention_needed" if corrupted else "ok",
    }

    if corrupted:
        summary["corrupted_samples"] = samples
        summary["sampled_count"] = len(samples)
    else:
        summary["note"] = "All scanned fragments loaded cleanly."

    return summary


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
