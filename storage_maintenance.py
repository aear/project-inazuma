"""Explicit, bounded storage-maintenance opportunities for Ina."""
from __future__ import annotations

from datetime import datetime, timezone
import gzip
import hashlib
import os
from pathlib import Path
import shutil
from typing import Any

from log_policy import classify_log_path, inventory_logs


CHOICES = ("inspect", "compress_one", "defer", "decline")
MAX_COMPRESS_SOURCE_BYTES = 512 * 1024 * 1024
COPY_CHUNK_BYTES = 1024 * 1024
FREE_SPACE_RESERVE_BYTES = 64 * 1024 * 1024


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _compression_candidate(root: Path, row: dict[str, Any]) -> Path | None:
    relative = Path(str(row.get("path") or ""))
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError:
        return None
    policy = classify_log_path(relative)
    if policy is None or not policy.automatic_cleanup or not row.get("over_size_policy"):
        return None
    if path.suffix == ".gz" or not path.is_file():
        return None
    # Active logs are owned by their writer. Only closed generations or crash
    # artifacts are eligible for this explicit maintenance action.
    if relative.name in {"ina_status.log", "precision_window.log", "comms_core.jsonl"}:
        return None
    size = path.stat().st_size
    if size <= 0 or size > MAX_COMPRESS_SOURCE_BYTES:
        return None
    return path


def maintenance_opportunity(root: Path | str, *, max_files: int = 10_000) -> dict[str, Any]:
    root = Path(root).resolve()
    report = inventory_logs(root, max_files=max_files)
    candidates = []
    for row in report["files"]:
        path = _compression_candidate(root, row)
        if path is not None:
            candidates.append({"path": str(path.relative_to(root)), "bytes": path.stat().st_size})
    candidates.sort(key=lambda row: (-int(row["bytes"]), str(row["path"])))
    return {
        "available": bool(candidates),
        "kind": "storage_evidence_cleanup",
        "note": "Optional: inspect or compress one verified closed log generation.",
        "choices": list(CHOICES),
        "candidates": candidates[:16],
        "candidate_count": len(candidates),
        "inventory_truncated": bool(report["truncated"]),
        "observed_at": _now(),
    }


def _sha256(path: Path, *, compressed: bool = False) -> str:
    digest = hashlib.sha256()
    opener = gzip.open if compressed else Path.open
    with opener(path, "rb") as handle:
        while True:
            chunk = handle.read(COPY_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def compress_verified(path: Path | str) -> dict[str, Any]:
    """Compress one closed artifact and retire the source only after verification."""
    source = Path(path).resolve()
    if not source.is_file() or source.suffix == ".gz":
        raise ValueError("A regular uncompressed source file is required.")
    size = source.stat().st_size
    if size <= 0 or size > MAX_COMPRESS_SOURCE_BYTES:
        raise ValueError("Source is outside the bounded compression size.")
    usage = shutil.disk_usage(source.parent)
    if usage.free < size + FREE_SPACE_RESERVE_BYTES:
        raise RuntimeError("Insufficient free space for a recoverable compressed copy.")
    target = source.with_name(source.name + ".gz")
    temporary = source.with_name(source.name + ".gz.partial")
    if target.exists() or temporary.exists():
        raise FileExistsError("Compressed target or partial file already exists.")
    source_digest = _sha256(source)
    try:
        with source.open("rb") as incoming, gzip.open(temporary, "wb", compresslevel=6) as outgoing:
            while True:
                chunk = incoming.read(COPY_CHUNK_BYTES)
                if not chunk:
                    break
                outgoing.write(chunk)
        if _sha256(temporary, compressed=True) != source_digest:
            raise RuntimeError("Compressed copy failed hash verification.")
        os.replace(temporary, target)
        os.utime(target, (source.stat().st_atime, source.stat().st_mtime))
        source.unlink()
    except Exception:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise
    return {
        "status": "compressed",
        "source": str(source),
        "target": str(target),
        "source_bytes": size,
        "target_bytes": target.stat().st_size,
        "sha256": source_digest,
        "verified": True,
        "completed_at": _now(),
    }


def perform_choice(root: Path | str, choice: str) -> dict[str, Any]:
    root = Path(root).resolve()
    selected = str(choice or "").strip().lower()
    if selected not in CHOICES:
        raise ValueError(f"choice must be one of: {', '.join(CHOICES)}")
    opportunity = maintenance_opportunity(root)
    if selected == "inspect":
        return {"status": "inspected", "choice": selected, "opportunity": opportunity, "completed_at": _now()}
    if selected in {"defer", "decline"}:
        return {"status": "deferred" if selected == "defer" else "declined", "choice": selected, "completed_at": _now()}
    if not opportunity["candidates"]:
        return {"status": "nothing_eligible", "choice": selected, "completed_at": _now()}
    candidate = root / opportunity["candidates"][0]["path"]
    return {"choice": selected, **compress_verified(candidate)}


__all__ = ["CHOICES", "compress_verified", "maintenance_opportunity", "perform_choice"]
