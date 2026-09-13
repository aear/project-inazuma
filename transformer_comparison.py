"""Evidence-gated, task-specific comparison of Ina and conventional Transformers."""
from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Mapping

EVIDENTIARY_PROTOCOLS = frozenset({"procedural-generative", "blind-held-out"})
FAMILIES = ("federated_ina", "conventional_transformer")
MAX_HISTORY_BYTES = 32 * 1024 * 1024
MAX_RECORDS = 512


def load_benchmark_history(path: Path) -> list[dict[str, Any]]:
    try:
        if not path.is_file() or path.stat().st_size > MAX_HISTORY_BYTES:
            return []
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    value = json.loads(line)
                except (ValueError, TypeError):
                    continue
                if isinstance(value, dict):
                    rows.append(value)
        return rows[-MAX_RECORDS:]
    except OSError:
        return []


def _witness_id(row: Mapping[str, Any]) -> str:
    protocol = str(row.get("evaluation_protocol") or "")
    fingerprint = str(row.get("seed_fingerprint") or row.get("suite_fingerprint") or "")
    return f"{protocol}:{fingerprint}" if protocol in EVIDENTIARY_PROTOCOLS and fingerprint else ""


def compare_transformer_families(
    records: Iterable[Mapping[str, Any]], *, minimum_witnesses: int = 3,
    minimum_accuracy_gain: float = 0.05, maximum_latency_ratio: float = 2.0,
) -> dict[str, Any]:
    """Recommend review only from repeated matched evaluations and three dimensions."""
    grouped: dict[tuple[str, str, str], dict[str, dict[str, dict[str, float]]]] = {}
    for row in tuple(records)[-MAX_RECORDS:]:
        family = str(row.get("model_family") or "")
        witness = _witness_id(row)
        if family not in FAMILIES or not witness:
            continue
        benchmark = str(row.get("benchmark") or "unknown")
        version = str(row.get("benchmark_version") or "unknown")
        categories = row.get("categories") if isinstance(row.get("categories"), Mapping) else {}
        elapsed = float(row.get("elapsed_seconds") or 0.0)
        total = max(1, int(row.get("total") or 1))
        for category, detail in categories.items():
            if not isinstance(detail, Mapping):
                continue
            key = (benchmark, version, str(category))
            grouped.setdefault(key, {}).setdefault(witness, {})[family] = {
                "accuracy": float(detail.get("accuracy") or 0.0),
                "margin": float(detail.get("mean_margin") or 0.0),
                "seconds_per_case": elapsed / total,
                "trained": float(bool(row.get("trained_weights", family != "conventional_transformer"))),
            }
    comparisons = []
    for (benchmark, version, category), witnesses in sorted(grouped.items()):
        paired = [value for value in witnesses.values() if all(family in value for family in FAMILIES)]
        conventional = [pair["conventional_transformer"] for pair in paired]
        federated = [pair["federated_ina"] for pair in paired]
        count = len(paired)
        accuracy_gain = fmean(row["accuracy"] for row in conventional) - fmean(
            row["accuracy"] for row in federated
        ) if paired else 0.0
        margin_gain = fmean(row["margin"] for row in conventional) - fmean(
            row["margin"] for row in federated
        ) if paired else 0.0
        ina_latency = fmean(row["seconds_per_case"] for row in federated) if paired else 0.0
        conventional_latency = fmean(row["seconds_per_case"] for row in conventional) if paired else 0.0
        latency_ratio = conventional_latency / ina_latency if ina_latency > 0 else math.inf
        paired_wins = sum(c["accuracy"] > i["accuracy"] for c, i in zip(conventional, federated))
        trained = bool(conventional) and all(row["trained"] for row in conventional)
        sufficient = count >= max(2, int(minimum_witnesses))
        recommend = (
            sufficient and trained and accuracy_gain >= minimum_accuracy_gain
            and margin_gain > 0.0 and paired_wins / count >= 2 / 3
            and latency_ratio <= maximum_latency_ratio
        )
        if recommend:
            status = "consider_conventional_for_task"
        elif not sufficient or not trained:
            status = "insufficient_evidence"
        elif accuracy_gain <= -minimum_accuracy_gain:
            status = "federated_advantage"
        else:
            status = "mixed_evidence"
        comparisons.append({
            "benchmark": benchmark, "benchmark_version": version, "task": category,
            "status": status, "review_recommended": recommend,
            "matched_independent_witnesses": count,
            "signals": {
                "mean_accuracy_gain": round(accuracy_gain, 6),
                "mean_margin_gain": round(margin_gain, 6),
                "paired_accuracy_wins": paired_wins,
                "latency_ratio": round(latency_ratio, 6) if math.isfinite(latency_ratio) else None,
            },
            "improvement_route": (
                "inspect this task for a missing federated capability before considering a bounded council role"
            ),
            "automatic_promotion": False,
        })
    return {
        "schema": "ina.transformer_comparison/V1",
        "comparisons": comparisons,
        "recommendations": [row for row in comparisons if row["review_recommended"]],
        "policy": {
            "minimum_matched_independent_witnesses": max(2, int(minimum_witnesses)),
            "requires_accuracy_margin_and_latency_signals": True,
            "public_smoke_is_evidence": False,
            "automatic_promotion": False,
        },
    }
