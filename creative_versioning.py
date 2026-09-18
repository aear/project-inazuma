"""Bounded, inspectable version history for Ina's creative working files."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping


VERSION_SCHEMA = "ina.creative_asset_version/V1"
MAX_LABEL_LENGTH = 160


def _safe_component(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_." else "-" for ch in value)
    cleaned = cleaned.strip("-.")
    return cleaned[:MAX_LABEL_LENGTH] or "untitled"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def preserve_creative_version(
    source: str | os.PathLike[str],
    *,
    medium: str,
    version_root: str | os.PathLike[str] | None = None,
    label: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Preserve one immutable snapshot and append its provenance record.

    Identical content reuses its hash-addressed snapshot while still recording
    the save event. The working file remains the ordinary editable copy.
    """
    source_path = Path(source)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    medium_name = _safe_component(str(medium))
    asset_name = _safe_component(label or source_path.stem)
    root = Path(version_root) if version_root is not None else source_path.parent / ".versions"
    asset_root = root / medium_name / asset_name
    snapshots = asset_root / "snapshots"
    snapshots.mkdir(parents=True, exist_ok=True, mode=0o700)

    content_hash = _sha256(source_path)
    snapshot = snapshots / f"{content_hash}{source_path.suffix.lower()}"
    created_snapshot = not snapshot.exists()
    if created_snapshot:
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(dir=snapshots, prefix=".version-", delete=False) as handle:
                temporary = Path(handle.name)
            shutil.copyfile(source_path, temporary)
            os.chmod(temporary, 0o600)
            os.replace(temporary, snapshot)
        except Exception:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            raise

    timestamp = datetime.now(timezone.utc).isoformat()
    record = {
        "schema": VERSION_SCHEMA,
        "timestamp": timestamp,
        "medium": medium_name,
        "asset": asset_name,
        "working_path": str(source_path),
        "snapshot_path": str(snapshot),
        "sha256": content_hash,
        "size_bytes": source_path.stat().st_size,
        "created_snapshot": created_snapshot,
        "metadata": dict(metadata or {}),
    }
    ledger = asset_root / "versions.jsonl"
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    record["ledger_path"] = str(ledger)
    return record


__all__ = ["MAX_LABEL_LENGTH", "VERSION_SCHEMA", "preserve_creative_version"]
