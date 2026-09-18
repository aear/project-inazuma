"""Bounded runtime measurements and review-only benchmark proposals.

Dynamic measurements describe observed work; they do not silently become tests,
move files, or authorize source changes.  Benchmark proposals preserve the gap
that Ina noticed and enough structure for a human to review a future benchmark.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Callable, Iterable, Mapping
import uuid

from io_utils import atomic_write_json


OBSERVATION_SCHEMA = "ina.dynamic_benchmark_observation/V1"
STORAGE_REPORT_SCHEMA = "ina.dynamic_storage_report/V1"
PROPOSAL_SCHEMA = "ina.benchmark_proposal/V1"

DEFAULT_POLICY = {
    "max_observations_per_subject": 32,
    "min_observations": 3,
    "min_independent_origins": 2,
    "min_benefit_ratio": 1.5,
    "min_saved_seconds": 0.02,
    "max_probe_files": 8,
    "max_probe_bytes": 8 * 1024 * 1024,
    "max_probe_seconds": 3.0,
    "proposal_path": "AI_Children/{child}/memory/benchmark_proposals.jsonl",
    "observation_path": "AI_Children/{child}/memory/dynamic_benchmark_observations.jsonl",
}


def evidence_route(*, material_uncertainty: bool, operation_can_teach: bool) -> dict[str, str]:
    """Apply the rule: benchmark when uncertain; otherwise learn in operation."""
    if material_uncertainty:
        return {
            "route": "propose_bounded_benchmark", "authority": "review_required",
            "reason": "material_uncertainty_requires_controlled_comparison",
        }
    if operation_can_teach:
        return {
            "route": "observe_normal_operation", "authority": "observation_only",
            "reason": "normal_operation_can_supply_relevant_evidence",
        }
    return {"route": "defer", "authority": "none",
            "reason": "no_material_uncertainty_or_available_operational_evidence"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _policy(config: Mapping[str, Any] | None) -> dict[str, Any]:
    result = dict(DEFAULT_POLICY)
    raw = config.get("dynamic_benchmark_policy") if isinstance(config, Mapping) else None
    if isinstance(raw, Mapping):
        result.update(raw)
    return result


def _child_path(template: object, child: str) -> Path:
    return Path(str(template).format(child=child))


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), sort_keys=True) + "\n")


def _bounded_tail(path: Path, *, max_bytes: int = 256 * 1024) -> list[dict[str, Any]]:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes))
            raw = handle.read(max_bytes)
    except OSError:
        return []
    records: list[dict[str, Any]] = []
    for line in raw.decode("utf-8", errors="replace").splitlines():
        try:
            item = json.loads(line)
        except (TypeError, ValueError):
            continue
        if isinstance(item, dict):
            records.append(item)
    return records


def record_measurement(
    child: str,
    subject: str,
    operation: str,
    config: Mapping[str, Any],
    *,
    origin: str,
    tier: str,
    elapsed_seconds: float,
    bytes_processed: int,
    success: bool = True,
    cache_state: str = "unknown",
    user_visible: bool = False,
    evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Retain one operation-local measurement without drawing a conclusion."""
    if not str(subject).strip() or not str(operation).strip() or not str(origin).strip():
        raise ValueError("subject, operation, and origin are required")
    if tier not in {"durable", "fast", "other"}:
        raise ValueError("tier must be durable, fast, or other")
    record = {
        "schema": OBSERVATION_SCHEMA,
        "observation_id": uuid.uuid4().hex,
        "observed_at": _now(),
        "child": child,
        "subject": str(subject),
        "operation": str(operation),
        "origin": str(origin),
        "tier": tier,
        "elapsed_seconds": max(0.0, float(elapsed_seconds)),
        "bytes_processed": max(0, int(bytes_processed)),
        "success": bool(success),
        "cache_state": str(cache_state),
        "user_visible": bool(user_visible),
        "evidence": dict(evidence or {}),
    }
    policy = _policy(config)
    _append_jsonl(_child_path(policy["observation_path"], child), record)
    return record


def storage_recommendation(
    child: str,
    subject: str,
    config: Mapping[str, Any],
    *,
    durable_authoritative: bool,
    rebuildable_fast_copy: bool,
    fast_free_bytes: int | None = None,
    subject_bytes: int | None = None,
) -> dict[str, Any]:
    """Compare retained observations and return a recommendation, never a move."""
    policy = _policy(config)
    path = _child_path(policy["observation_path"], child)
    limit = max(1, int(policy["max_observations_per_subject"]))
    observations = [
        item for item in _bounded_tail(path)
        if item.get("schema") == OBSERVATION_SCHEMA
        and item.get("child") == child and item.get("subject") == subject
        and item.get("success") is True
    ][-limit:]
    groups = {
        tier: [item for item in observations if item.get("tier") == tier]
        for tier in ("durable", "fast")
    }
    minimum = max(2, int(policy["min_observations"]))
    origins = {str(item.get("origin")) for item in observations if item.get("origin")}
    comparable = all(len(groups[tier]) >= minimum for tier in groups)
    independent = len(origins) >= max(2, int(policy["min_independent_origins"]))
    means = {
        tier: (sum(float(item["elapsed_seconds"]) for item in rows) / len(rows) if rows else None)
        for tier, rows in groups.items()
    }
    durable_mean, fast_mean = means["durable"], means["fast"]
    ratio = (durable_mean / max(fast_mean, 1e-9)) if comparable else None
    saved = (durable_mean - fast_mean) if comparable else None
    capacity_ok = not (
        fast_free_bytes is not None and subject_bytes is not None
        and int(subject_bytes) > max(0, int(fast_free_bytes) - max(64 * 1024 * 1024, int(fast_free_bytes * 0.08)))
    )
    strong = bool(
        comparable and independent and capacity_ok and rebuildable_fast_copy
        and ratio is not None and ratio >= float(policy["min_benefit_ratio"])
        and saved is not None and saved >= float(policy["min_saved_seconds"])
    )
    action = "keep_durable_source_and_add_verified_fast_copy" if strong else "no_change"
    blockers = []
    if not comparable:
        blockers.append("insufficient_paired_observations")
    if not independent:
        blockers.append("insufficient_independent_origins")
    if not rebuildable_fast_copy:
        blockers.append("fast_copy_not_declared_rebuildable")
    if not capacity_ok:
        blockers.append("fast_capacity_reserve")
    if comparable and not strong and capacity_ok and rebuildable_fast_copy:
        blockers.append("measured_benefit_below_threshold")
    return {
        "schema": STORAGE_REPORT_SCHEMA,
        "generated_at": _now(),
        "child": child,
        "subject": subject,
        "recommendation": action,
        "strong": strong,
        "automatic_migration_authorized": False,
        "durable_source_must_remain": bool(durable_authoritative),
        "sample_counts": {tier: len(rows) for tier, rows in groups.items()},
        "independent_origins": sorted(origins),
        "mean_seconds": means,
        "benefit_ratio": round(ratio, 6) if ratio is not None else None,
        "mean_seconds_saved": round(saved, 6) if saved is not None else None,
        "subject_bytes": subject_bytes,
        "fast_free_bytes": fast_free_bytes,
        "blockers": blockers,
        "next_step": "human_review_then_hash_verified_migration" if strong else "collect_or_review_evidence",
    }


def bounded_read_probe(
    files: Iterable[Path], *, max_files: int = 8, max_bytes: int = 8 * 1024 * 1024,
    max_seconds: float = 3.0, clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Explicitly invoked, content-blind read probe with byte/time ceilings."""
    started = clock()
    digest = hashlib.sha256()
    read_bytes = 0
    read_files = 0
    errors: list[str] = []
    for path in files:
        if read_files >= max(1, int(max_files)) or read_bytes >= max(4096, int(max_bytes)):
            break
        if clock() - started >= max(0.01, float(max_seconds)):
            break
        try:
            with Path(path).open("rb") as handle:
                while read_bytes < max_bytes and clock() - started < max_seconds:
                    chunk = handle.read(min(256 * 1024, max_bytes - read_bytes))
                    if not chunk:
                        break
                    digest.update(chunk)
                    read_bytes += len(chunk)
            read_files += 1
        except OSError as exc:
            errors.append(f"{Path(path).name}: {exc}")
    elapsed = max(0.0, clock() - started)
    return {
        "elapsed_seconds": elapsed, "bytes_processed": read_bytes,
        "files_processed": read_files, "digest_sha256": digest.hexdigest(),
        "errors": errors[:8], "bounded": True,
    }


def propose_benchmark(
    child: str,
    config: Mapping[str, Any],
    *,
    capability: str,
    reason: str,
    uncertainty: str,
    signals: list[Mapping[str, Any]],
    dimensions: list[str],
    baseline_version: str,
    candidate_version: str,
    deterministic_cases: list[str],
    held_out_cases: list[str],
    adversarial_cases: list[str],
    run_budget: int = 1,
) -> dict[str, Any]:
    """Append a deduplicated review proposal; it cannot execute a benchmark."""
    capability = str(capability).strip()
    reason = str(reason).strip()
    uncertainty = str(uncertainty).strip()
    if not capability or not reason or not uncertainty:
        raise ValueError("capability, reason, and uncertainty are required")
    origins = {str(item.get("origin") or "").strip() for item in signals} - {""}
    if len(signals) < 2 or len(origins) < 2:
        raise ValueError("benchmark proposals require two independent signals")
    if not dimensions or not deterministic_cases or not held_out_cases or not adversarial_cases:
        raise ValueError("dimensions plus deterministic, held-out, and adversarial cases are required")
    if not str(baseline_version).startswith("V") or not str(candidate_version).startswith("V"):
        raise ValueError("baseline and candidate benchmark versions must be explicit")
    if not 1 <= int(run_budget) <= 5:
        raise ValueError("run_budget must be 1..5")
    policy = _policy(config)
    path = _child_path(policy["proposal_path"], child)
    fingerprint_payload = {
        "capability": capability,
        "dimensions": sorted({str(item) for item in dimensions}),
        "baseline_version": str(baseline_version),
        "candidate_version": str(candidate_version),
    }
    fingerprint = hashlib.sha256(json.dumps(fingerprint_payload, sort_keys=True).encode()).hexdigest()
    previous = next((item for item in reversed(_bounded_tail(path)) if item.get("fingerprint") == fingerprint), None)
    if previous:
        return {**previous, "created": False, "duplicate_of": previous.get("proposal_id")}
    proposal = {
        "schema": PROPOSAL_SCHEMA,
        "proposal_id": uuid.uuid4().hex,
        "fingerprint": fingerprint,
        "created_at": _now(),
        "child": child,
        "capability": capability,
        "reason": reason,
        "uncertainty": uncertainty,
        "signals": [dict(item) for item in signals],
        "independent_origins": sorted(origins),
        "dimensions": sorted({str(item) for item in dimensions}),
        "baseline_version": str(baseline_version),
        "candidate_version": str(candidate_version),
        "cases": {
            "deterministic": [str(item) for item in deterministic_cases],
            "held_out": [str(item) for item in held_out_cases],
            "adversarial": [str(item) for item in adversarial_cases],
        },
        "run_budget": int(run_budget),
        "trigger": "explicit_only",
        "status": "review_required",
        "execution_authorized": False,
        "source_changes_authorized": False,
    }
    _append_jsonl(path, proposal)
    return {**proposal, "created": True}


__all__ = [
    "OBSERVATION_SCHEMA", "STORAGE_REPORT_SCHEMA", "PROPOSAL_SCHEMA",
    "record_measurement", "storage_recommendation", "bounded_read_probe",
    "propose_benchmark", "evidence_route",
]
