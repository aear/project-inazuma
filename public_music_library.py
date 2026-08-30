"""Fail-closed provenance gate for music Ina may listen to or study."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


MANIFEST_NAME = "ina_public_music_manifest.json"
SCHEMA = "ina.public_music_manifest/V1"
ACCEPTED_LICENSES = frozenset({
    "CC0-1.0", "PDM-1.0", "CC-BY-4.0", "LOC-MUSICBOX-FREE-TO-USE"
})


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_track_record(record: Mapping[str, Any], file_path: Path) -> dict[str, Any]:
    """Return an inspectable admission decision; uncertainty always blocks."""
    license_id = str(record.get("license_id") or "").strip().upper()
    required = {
        "source_url": str(record.get("source_url") or "").strip(),
        "license_url": str(record.get("license_url") or "").strip(),
        "creator": str(record.get("creator") or "").strip(),
        "recording_rights": str(record.get("recording_rights") or "").strip(),
        "composition_rights": str(record.get("composition_rights") or "").strip(),
        "sha256": str(record.get("sha256") or "").strip().lower(),
    }
    missing = sorted(key for key, value in required.items() if not value)
    if missing:
        return {"admitted": False, "reason": "missing_provenance", "missing": missing}
    if license_id not in ACCEPTED_LICENSES:
        return {"admitted": False, "reason": "license_not_allowlisted", "license_id": license_id}
    if not file_path.is_file():
        return {"admitted": False, "reason": "file_missing"}
    actual = sha256_file(file_path)
    if actual != required["sha256"]:
        return {"admitted": False, "reason": "hash_mismatch", "actual_sha256": actual}
    return {
        "admitted": True,
        "license_id": license_id,
        "creator": required["creator"],
        "source_url": required["source_url"],
        "license_url": required["license_url"],
        "recording_rights": required["recording_rights"],
        "composition_rights": required["composition_rights"],
        "learning_scope": {
            "music": ["tone", "composition", "rhythm", "instrumentation"],
            "language": ["words", "pronunciation_evidence", "sung_phrasing"],
            "self_voice_identity": False,
            "voice_imitation_target": False,
        },
    }


def admitted_track(root: Path, relative_path: str) -> dict[str, Any]:
    manifest_path = root / MANIFEST_NAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return {"admitted": False, "reason": "manifest_unavailable"}
    if manifest.get("schema") != SCHEMA or not isinstance(manifest.get("tracks"), dict):
        return {"admitted": False, "reason": "invalid_manifest"}
    record = manifest["tracks"].get(str(relative_path).replace("\\", "/"))
    if not isinstance(record, dict):
        return {"admitted": False, "reason": "track_not_manifested"}
    return validate_track_record(record, root / relative_path)


__all__ = ["ACCEPTED_LICENSES", "MANIFEST_NAME", "SCHEMA", "admitted_track", "sha256_file", "validate_track_record"]
