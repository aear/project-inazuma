"""Deterministic, explicit comparisons between retained module versions."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Mapping

from discourse_context import build_discourse_context, resolution_for


@dataclass(frozen=True)
class ModuleVersion:
    module: str
    version: str
    description: str
    evaluate: Callable[[], dict[str, Any]]


@dataclass(frozen=True)
class ModuleBenchmarkResult:
    module: str
    version: str
    benchmark_version: str
    accuracy: float
    correct: int
    total: int
    elapsed_seconds: float
    cases: tuple[dict[str, Any], ...]
    run_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_DISCOURSE_CASES = (
    ("I found the key.", "i", "sakura"),
    ("Careful with your memory use.", "your", "self"),
    ("You remembered it.", "you", "self"),
    ("My note is here.", "my", "sakura"),
    ("We can inspect this.", "we", "sakura"),
    ("We can inspect this.", "we", "self"),
    ("They moved it.", "they", "rowan"),
    ("That was the garden.", "that", "garden"),
)


def _legacy_discourse() -> dict[str, Any]:
    """V1 baseline: discourse terms were lexical stopwords with no role model."""
    rows = [{"case": text, "surface": surface, "expected": expected, "actual": None,
             "correct": False} for text, surface, expected in _DISCOURSE_CASES]
    return {"correct": 0, "total": len(rows), "cases": rows}


def _role_aware_discourse() -> dict[str, Any]:
    rows = []
    for text, surface, expected in _DISCOURSE_CASES:
        context = build_discourse_context(
            text, speaker={"id": "sakura", "name": "Sakura"},
            addressee={"id": "ina", "name": "Ina", "is_self": True},
            self_identity={"id": "ina", "name": "Ina", "is_self": True},
            current_subject="inspection", mentioned_entities=("Rowan",),
            prior_referent="garden",
        )
        resolved = resolution_for(context, surface) or {}
        actual_ids = [str(item.get("id")) for item in resolved.get("referents") or () if isinstance(item, Mapping)]
        correct = expected in actual_ids
        rows.append({"case": text, "surface": surface, "expected": expected,
                     "actual": actual_ids, "correct": correct})
    return {"correct": sum(row["correct"] for row in rows), "total": len(rows), "cases": rows}


_REGISTRY = {
    "discourse": (
        ModuleVersion("discourse", "V1", "Legacy lexical stopword behavior", _legacy_discourse),
        ModuleVersion("discourse", "V2", "Speaker/addressee and deictic role resolution", _role_aware_discourse),
    ),
}


def list_benchmark_modules() -> dict[str, tuple[ModuleVersion, ...]]:
    return dict(_REGISTRY)


def benchmark_module(module: str, versions: tuple[str, ...] | None = None) -> tuple[ModuleBenchmarkResult, ...]:
    specs = _REGISTRY.get(str(module))
    if not specs:
        raise ValueError(f"unknown benchmark module: {module}")
    selected = set(versions or ())
    results = []
    for spec in specs:
        if selected and spec.version not in selected:
            continue
        started = time.perf_counter()
        outcome = spec.evaluate()
        elapsed = time.perf_counter() - started
        total = int(outcome.get("total") or 0)
        correct = int(outcome.get("correct") or 0)
        results.append(ModuleBenchmarkResult(
            module=spec.module, version=spec.version, benchmark_version="V1",
            accuracy=round(correct / total, 6) if total else 0.0,
            correct=correct, total=total, elapsed_seconds=round(elapsed, 6),
            cases=tuple(outcome.get("cases") or ()),
            run_at=datetime.now(timezone.utc).isoformat(),
        ))
    return tuple(results)


__all__ = ["ModuleBenchmarkResult", "ModuleVersion", "benchmark_module", "list_benchmark_modules"]
