"""Minimal, non-forcing capture of material that surfaces before an intention."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
import uuid


CAPTURE_SCHEMA = "ina.emergence_capture/V1"
INTERPRETATION_SCHEMA = "ina.emergence_interpretation/V1"
MODALITIES = {"text", "image", "sound", "rhythm", "symbol", "relation", "impulse", "mixed"}
MAX_CONTENT_BYTES = 16 * 1024


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _references(rows: Sequence[Mapping[str, Any]] | None, *, limit: int = 16) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for row in rows or ():
        reference = {
            key: str(row[key])[:500]
            for key in ("id", "path", "kind", "role") if row.get(key) is not None
        }
        if reference and (reference.get("id") or reference.get("path")):
            result.append(reference)
        if len(result) >= limit:
            break
    return result


def _bounded_content(content: Any) -> Any:
    try:
        encoded = json.dumps(content, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("emergent content must be JSON-compatible") from exc
    if len(encoded) > MAX_CONTENT_BYTES:
        raise ValueError(
            f"emergent content exceeds {MAX_CONTENT_BYTES} bytes; store large domain material externally and reference it"
        )
    return content


def capture_emergence(
    content: Any, *, modality: str, source: str = "spontaneous",
    context_references: Sequence[Mapping[str, Any]] | None = None,
    salience: float | None = None,
) -> dict[str, Any]:
    """Preserve surfaced material without manufacturing intent or meaning."""
    selected_modality = str(modality).strip().lower()
    if selected_modality not in MODALITIES:
        raise ValueError(f"modality must be one of: {', '.join(sorted(MODALITIES))}")
    if content is None or (isinstance(content, str) and not content.strip()):
        raise ValueError("emergent content must not be empty")
    record: dict[str, Any] = {
        "schema": CAPTURE_SCHEMA,
        "capture_id": str(uuid.uuid4()),
        "captured_at": _now(),
        "content": _bounded_content(content),
        "modality": selected_modality,
        "source": str(source or "unknown")[:80],
        "context_references": _references(context_references),
        "meaning_status": "unresolved",
        "meaning": None,
        "requires_interpretation": False,
        "requires_action": False,
    }
    if salience is not None:
        record["salience"] = max(0.0, min(1.0, float(salience)))
    return record


def propose_interpretation(
    capture: Mapping[str, Any], hypothesis: str, *,
    evidence_references: Sequence[Mapping[str, Any]] | None = None,
    confidence: float = 0.0,
) -> dict[str, Any]:
    """Create a linked hypothesis; never rewrite the captured source material."""
    capture_id = str(capture.get("capture_id") or "")
    if capture.get("schema") != CAPTURE_SCHEMA or not capture_id:
        raise ValueError("a valid emergence capture is required")
    text = str(hypothesis or "").strip()
    if not text:
        raise ValueError("interpretation hypothesis must not be empty")
    return {
        "schema": INTERPRETATION_SCHEMA,
        "interpretation_id": str(uuid.uuid4()),
        "capture_id": capture_id,
        "created_at": _now(),
        "hypothesis": text[:2000],
        "confidence": max(0.0, min(1.0, float(confidence))),
        "evidence_references": _references(evidence_references),
        "status": "candidate",
    }


def append_record(path: Path | str, record: Mapping[str, Any]) -> Path:
    """Append one bounded capture or interpretation record to a JSONL ledger."""
    schema = record.get("schema")
    if schema not in {CAPTURE_SCHEMA, INTERPRETATION_SCHEMA}:
        raise ValueError("unsupported emergence record schema")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(dict(record), ensure_ascii=False, separators=(",", ":"))
    if len(encoded) > 65536:
        raise ValueError("emergence record exceeds 65536 characters")
    with target.open("a", encoding="utf-8") as handle:
        handle.write(encoded + "\n")
    return target


__all__ = [
    "CAPTURE_SCHEMA", "INTERPRETATION_SCHEMA", "MODALITIES", "MAX_CONTENT_BYTES",
    "capture_emergence", "propose_interpretation", "append_record",
]
