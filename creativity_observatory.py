"""Multidimensional creativity evidence without an aggregate creativity score."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable, Mapping
import uuid

from io_utils import file_lock, flush_for_durability


SCHEMA = "ina.creativity_evidence/V1"
REPORT_SCHEMA = "ina.creativity_profile/V1"
CREATIVE_DOMAINS = ("music", "image", "code_game", "model_3d")
MATERIAL_MODES = ("intended", "emergent", "mixed", "unknown")
DIMENSIONS = {
    "coherence": {"description": "Parts form a legible whole without requiring conventionality.", "human": True, "objective": False},
    "originality": {"description": "Meaningful difference from an explicit reference set, not noise alone.", "human": True, "objective": True},
    "purposeful_surprise": {"description": "Unexpected choices add expressive or functional value.", "human": True, "objective": False},
    "transformation": {"description": "Sources are recombined or transformed rather than merely reproduced.", "human": False, "objective": True},
    "range": {"description": "Distinct briefs produce materially distinct structures and approaches.", "human": False, "objective": True},
    "development": {"description": "Revision preserves strengths while making meaningful change.", "human": True, "objective": True},
}


def evidence_path(root: Path | str = ".") -> Path:
    return Path(root) / "benchmark_results" / "creativity_evidence.jsonl"


def make_evidence(
    domain: str, dimension: str, score: float, *, artifact_id: str,
    origin: str, method: str, evaluator_type: str, material_mode: str = "unknown",
    reference_set_id: str = "", source_references: Iterable[str] = (), notes: str = "",
    observed_at: str | None = None,
) -> dict[str, Any]:
    if domain not in CREATIVE_DOMAINS:
        raise ValueError(f"domain must be one of: {', '.join(CREATIVE_DOMAINS)}")
    if dimension not in DIMENSIONS:
        raise ValueError(f"unknown creativity dimension: {dimension}")
    value = float(score)
    if not 0 <= value <= 1:
        raise ValueError("score must be between 0 and 1")
    evaluator = str(evaluator_type).lower().strip()
    if evaluator not in {"human", "objective"}:
        raise ValueError("evaluator_type must be human or objective")
    mode = str(material_mode).lower().strip()
    if mode not in MATERIAL_MODES:
        raise ValueError(f"material_mode must be one of: {', '.join(MATERIAL_MODES)}")
    if any(not str(item).strip() for item in (artifact_id, origin, method)):
        raise ValueError("artifact_id, origin, and method are required")
    references = tuple(str(item)[:240] for item in source_references if str(item).strip())
    if len(references) > 16:
        raise ValueError("source_references is bounded to 16 entries")
    if dimension == "originality" and not str(reference_set_id).strip():
        raise ValueError("originality evidence requires an explicit reference_set_id")
    return {
        "schema": SCHEMA, "evidence_id": str(uuid.uuid4()), "domain": domain,
        "dimension": dimension, "score": value, "artifact_id": str(artifact_id)[:240],
        "origin": str(origin)[:160], "method": str(method)[:160],
        "evaluator_type": evaluator, "material_mode": mode,
        "reference_set_id": str(reference_set_id)[:240],
        "source_references": list(references), "notes": str(notes)[:2000],
        "observed_at": observed_at or datetime.now(timezone.utc).isoformat(),
    }


def append_evidence(record: Mapping[str, Any], *, root: Path | str = ".") -> Path:
    if record.get("schema") != SCHEMA:
        raise ValueError("unsupported creativity evidence schema")
    make_evidence(**{key: record[key] for key in (
        "domain", "dimension", "score", "artifact_id", "origin", "method",
        "evaluator_type", "material_mode", "reference_set_id", "source_references",
        "notes", "observed_at",
    )})
    path = evidence_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(path.with_suffix(path.suffix + ".lock")):
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(record), sort_keys=True, ensure_ascii=False) + "\n")
            flush_for_durability(handle, path)
    return path


def load_evidence(path: Path | str, *, limit: int = 2048) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return []
    retained = []
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                item = json.loads(line)
            except (ValueError, TypeError):
                continue
            if isinstance(item, dict) and item.get("schema") == SCHEMA:
                retained.append(item)
                if len(retained) > max(1, int(limit)):
                    retained.pop(0)
    return retained


def build_profile(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    retained = list(records)
    domains = {}
    for domain in CREATIVE_DOMAINS:
        dimensions = {}
        domain_rows = [row for row in retained if row.get("domain") == domain]
        for name, policy in DIMENSIONS.items():
            rows = [row for row in domain_rows if row.get("dimension") == name]
            origins = sorted({str(row.get("origin")) for row in rows if row.get("origin")})
            types = {str(row.get("evaluator_type")) for row in rows}
            mean = sum(float(row.get("score", 0)) for row in rows) / len(rows) if rows else None
            blockers = []
            if len(rows) < 2: blockers.append("needs at least 2 evidence records")
            if len(origins) < 2: blockers.append("needs at least 2 independent origins")
            if policy["human"] and "human" not in types: blockers.append("needs human evidence")
            if policy["objective"] and "objective" not in types: blockers.append("needs objective evidence")
            dimensions[name] = {
                "description": policy["description"], "score": round(mean, 4) if mean is not None else None,
                "evidence_count": len(rows), "origins": origins, "evidenced": not blockers,
                "blockers": blockers,
            }
        evidenced = sum(item["evidenced"] for item in dimensions.values())
        domains[domain] = {
            "status": "not_assessed" if not domain_rows else ("multidimensionally_evidenced" if evidenced == len(DIMENSIONS) else "partial"),
            "evidence_count": len(domain_rows), "evidenced_dimensions": evidenced,
            "dimensions": dimensions,
        }
    return {
        "schema": REPORT_SCHEMA, "generated_at": datetime.now(timezone.utc).isoformat(),
        "domains": domains,
        "policy": {
            "single_creativity_score": False, "intent_required": False,
            "novelty_alone_is_creativity": False, "automatic_promotion": False,
            "private_rationale_required": False,
        },
    }


__all__ = [
    "CREATIVE_DOMAINS", "DIMENSIONS", "SCHEMA", "append_evidence", "build_profile",
    "evidence_path", "load_evidence", "make_evidence",
]
