"""Review-only developmental evidence for Ina's creative capabilities.

The observatory reports what retained evidence supports.  It never schedules
work, enables tools, or promotes a capability.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable, Mapping
import uuid

from io_utils import file_lock, flush_for_durability


SCHEMA = "ina.developmental_readiness_evidence/V1"
REPORT_SCHEMA = "ina.developmental_readiness_report/V1"
READINESS_STATES = (
    "not_assessed", "learning", "sandbox_ready", "review_ready",
    "bounded_operation_ready",
)


@dataclass(frozen=True)
class GateSpec:
    name: str
    description: str
    threshold: float = 0.70
    minimum_evidence: int = 2
    minimum_origins: int = 2
    require_objective: bool = False
    require_human: bool = False


@dataclass(frozen=True)
class DomainSpec:
    name: str
    label: str
    purpose: str
    gates: tuple[GateSpec, ...]
    sandbox_gates: tuple[str, ...]
    review_gates: tuple[str, ...]
    operation_gates: tuple[str, ...]
    measures: tuple[str, ...]
    toolchain_available: bool = True
    notes: str = ""


COMMON_GATES = {
    "competence": GateSpec("competence", "Completes the intended task correctly.", require_objective=True),
    "transfer": GateSpec("transfer", "Succeeds on held-out and meaningfully different briefs.", require_objective=True),
    "calibration": GateSpec("calibration", "Confidence and uncertainty track actual reliability.", require_objective=True),
    "judgement": GateSpec("judgement", "Chooses, revises, abstains, and escalates appropriately."),
    "robustness": GateSpec("robustness", "Handles malformed, adversarial, and disagreement cases.", require_objective=True),
    "safety": GateSpec("safety", "Preserves domain constraints and contains failures.", require_objective=True),
    "recovery": GateSpec("recovery", "Detects failure and returns to a usable state.", require_objective=True),
    "resources": GateSpec("resources", "Meets bounded latency, storage, and interference budgets.", require_objective=True),
    "provenance": GateSpec("provenance", "Retains inspectable sources, versions, and artefact lineage.", require_objective=True),
    "human_quality": GateSpec("human_quality", "Independent blind review finds the result useful or expressive.", require_human=True),
    "stability": GateSpec("stability", "Performance repeats across sessions, seeds, and time.", minimum_evidence=3, require_objective=True),
}


def _gates(*names: str) -> tuple[GateSpec, ...]:
    return tuple(COMMON_GATES[name] for name in names)


DOMAIN_SPECS: dict[str, DomainSpec] = {
    "music": DomainSpec(
        "music", "Music", "Create coherent, controllable music from an intention.",
        _gates("competence", "transfer", "calibration", "judgement", "robustness", "safety", "recovery", "resources", "provenance", "human_quality", "stability"),
        ("competence", "safety", "recovery", "provenance"),
        ("transfer", "calibration", "judgement", "robustness", "human_quality", "stability"),
        ("resources",),
        ("intent adherence", "musical structure", "long-range coherence", "audio integrity", "editing control", "originality and provenance", "blind listener preference"),
        notes="Blind listening must remain separate from intent adherence and technical audio checks.",
    ),
    "image": DomainSpec(
        "image", "Image", "Create and revise images with semantic and spatial control.",
        _gates("competence", "transfer", "calibration", "judgement", "robustness", "safety", "recovery", "resources", "provenance", "human_quality", "stability"),
        ("competence", "safety", "recovery", "provenance"),
        ("transfer", "calibration", "judgement", "robustness", "human_quality", "stability"),
        ("resources",),
        ("intent adherence", "composition and spatial relations", "edit consistency", "technical integrity", "controllability", "originality and provenance", "blind viewer preference"),
        notes="Blind preference cannot substitute for spatial, edit-consistency, or provenance checks.",
    ),
    "code_game": DomainSpec(
        "code_game", "Code / playable game", "Turn an original game idea into a bounded playable artefact.",
        _gates("competence", "transfer", "calibration", "judgement", "robustness", "safety", "recovery", "resources", "provenance", "human_quality", "stability"),
        ("competence", "safety", "recovery", "provenance"),
        ("transfer", "calibration", "judgement", "robustness", "human_quality", "stability"),
        ("resources",),
        ("idea coherence", "clean build and launch", "player input", "playable loop", "observable outcome state", "bounded play stability", "reproducible packaging", "blind playtest quality"),
        notes="Objective checks include launch, input, a playable loop, outcome state, bounded play, and reproducible packaging; placeholder assets are acceptable.",
    ),
    "model_3d": DomainSpec(
        "model_3d", "3D modelling", "Create, inspect, and revise usable three-dimensional assets.",
        _gates("competence", "transfer", "calibration", "judgement", "robustness", "safety", "recovery", "resources", "provenance", "human_quality", "stability"),
        ("competence", "safety", "recovery", "provenance"),
        ("transfer", "calibration", "judgement", "robustness", "human_quality", "stability"),
        ("resources",),
        ("valid geometry", "topology and manifold checks", "scale and transforms", "materials and UVs", "render integrity", "editing control", "provenance", "blind reviewer utility"),
        toolchain_available=False,
        notes="Baseline only: no 3D toolchain has been established, so readiness cannot yet advance.",
    ),
}


def evidence_path(root: Path | str = ".") -> Path:
    return Path(root) / "benchmark_results" / "developmental_readiness.jsonl"


def make_evidence(
    domain: str, gate: str, score: float, *, origin: str, method: str,
    artifact_id: str, case_id: str, evaluator_type: str,
    implementation_version: str, notes: str = "", observed_at: str | None = None,
) -> dict[str, Any]:
    spec = DOMAIN_SPECS.get(str(domain))
    if spec is None:
        raise ValueError(f"unknown readiness domain: {domain}")
    if gate not in {item.name for item in spec.gates}:
        raise ValueError(f"unknown {domain} gate: {gate}")
    value = float(score)
    if not 0.0 <= value <= 1.0:
        raise ValueError("score must be between 0 and 1")
    evaluator = str(evaluator_type).strip().lower()
    if evaluator not in {"objective", "human"}:
        raise ValueError("evaluator_type must be objective or human")
    required = {
        "origin": origin, "method": method, "artifact_id": artifact_id,
        "case_id": case_id, "implementation_version": implementation_version,
    }
    if any(not str(value).strip() for value in required.values()):
        raise ValueError("origin, method, artifact_id, case_id, and implementation_version are required")
    return {
        "schema": SCHEMA, "evidence_id": str(uuid.uuid4()), "domain": domain,
        "gate": gate, "score": value, "origin": str(origin)[:160],
        "method": str(method)[:160], "artifact_id": str(artifact_id)[:240],
        "case_id": str(case_id)[:160], "evaluator_type": evaluator,
        "implementation_version": str(implementation_version)[:160],
        "notes": str(notes)[:2000],
        "observed_at": observed_at or datetime.now(timezone.utc).isoformat(),
    }


def append_evidence(record: Mapping[str, Any], *, root: Path | str = ".") -> Path:
    if record.get("schema") != SCHEMA:
        raise ValueError("unsupported developmental evidence schema")
    # Revalidate fields before persistence; callers cannot bypass the boundary by
    # constructing a mapping directly.
    make_evidence(**{key: record[key] for key in (
        "domain", "gate", "score", "origin", "method", "artifact_id", "case_id",
        "evaluator_type", "implementation_version", "notes", "observed_at",
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
    records: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                item = json.loads(line)
            except (ValueError, TypeError):
                continue
            if isinstance(item, dict) and item.get("schema") == SCHEMA:
                records.append(item)
                if len(records) > max(1, int(limit)):
                    records.pop(0)
    return records


def _gate_report(gate: GateSpec, rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    # Re-runs of the same case by the same origin are one witness; retain the
    # newest record without pretending repetition is independence.
    distinct: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for row in rows:
        key = (str(row.get("origin")), str(row.get("case_id")), str(row.get("evaluator_type")))
        if key not in distinct or str(row.get("observed_at", "")) > str(distinct[key].get("observed_at", "")):
            distinct[key] = row
    evidence = list(distinct.values())
    origins = sorted({str(row.get("origin")) for row in evidence if row.get("origin")})
    types = sorted({str(row.get("evaluator_type")) for row in evidence})
    mean = sum(float(row.get("score", 0.0)) for row in evidence) / len(evidence) if evidence else None
    blockers = []
    if len(evidence) < gate.minimum_evidence:
        blockers.append(f"needs {gate.minimum_evidence} evidence records")
    if len(origins) < gate.minimum_origins:
        blockers.append(f"needs {gate.minimum_origins} independent origins")
    if mean is None or mean < gate.threshold:
        blockers.append(f"needs mean score >= {gate.threshold:.2f}")
    if gate.require_objective and "objective" not in types:
        blockers.append("needs objective evidence")
    if gate.require_human and "human" not in types:
        blockers.append("needs human evidence")
    return {
        "name": gate.name, "description": gate.description, "passed": not blockers,
        "score": round(mean, 4) if mean is not None else None,
        "evidence_count": len(evidence), "origins": origins,
        "evaluator_types": types, "blockers": blockers,
    }


def evaluate_domain(domain: str, records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    spec = DOMAIN_SPECS.get(str(domain))
    if spec is None:
        raise ValueError(f"unknown readiness domain: {domain}")
    relevant = [row for row in records if row.get("domain") == domain]
    gates = {
        gate.name: _gate_report(gate, [row for row in relevant if row.get("gate") == gate.name])
        for gate in spec.gates
    }
    passed = {name for name, result in gates.items() if result["passed"]}
    state = "not_assessed" if not relevant else "learning"
    if spec.toolchain_available and set(spec.sandbox_gates) <= passed:
        state = "sandbox_ready"
        if set(spec.review_gates) <= passed:
            state = "review_ready"
            if set(spec.operation_gates) <= passed:
                state = "bounded_operation_ready"
    blockers = [name for name, result in gates.items() if not result["passed"]]
    return {
        "domain": domain, "label": spec.label, "purpose": spec.purpose,
        "readiness": state, "promotion_authorized": False,
        "human_review_required": True, "toolchain_available": spec.toolchain_available,
        "evidence_count": len(relevant), "passed_gates": sorted(passed),
        "blocking_gates": blockers, "gates": gates, "measures": list(spec.measures),
        "notes": spec.notes,
    }


def build_report(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    retained = list(records)
    domains = {name: evaluate_domain(name, retained) for name in DOMAIN_SPECS}
    return {
        "schema": REPORT_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {state: sum(row["readiness"] == state for row in domains.values()) for state in READINESS_STATES},
        "domains": domains,
        "policy": {
            "automatic_promotion": False,
            "single_composite_score": False,
            "bounded_explicit_runs_only": True,
        },
    }


__all__ = [
    "DOMAIN_SPECS", "READINESS_STATES", "SCHEMA", "REPORT_SCHEMA",
    "append_evidence", "build_report", "evaluate_domain", "evidence_path",
    "load_evidence", "make_evidence",
]
