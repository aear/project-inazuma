"""Explicit capability assessments feeding, but never promoting from, the Observatory."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Iterable

from cyber_defence_learning import SCENARIOS, assess_response
from external_access import ExternalAccessBlocked, ExternalPolicy, validate_external_url
from module_benchmarks import benchmark_module
from research_capability import assess_claim_evidence


def assess_english_foundation() -> dict[str, Any]:
    selections = (
        ("language_components", ("V3",)), ("discourse", ("V3",)),
        ("semantic_topology", ("V2",)), ("expression_core", ("V6",)),
    )
    results = []
    for module, versions in selections:
        for result in benchmark_module(module, versions):
            results.append({
                "module": module, "version": result.version, "correct": result.correct,
                "total": result.total, "accuracy": result.accuracy,
                "components": result.component_scores, "source_revision": result.source_revision,
            })
    return {
        "schema": "ina.capability_assessment/V1", "domain": "language_english",
        "subject": "current_implementation", "assessed_at": datetime.now(timezone.utc).isoformat(),
        "results": results, "all_cases_passed": all(row["correct"] == row["total"] for row in results),
        "live_social_expression_assessed": False,
        "promotion_authorized": False,
    }


def assess_cyber_responses(response_provider: Callable[[str], Iterable[str]] | None) -> dict[str, Any]:
    if response_provider is None:
        return {
            "schema": "ina.capability_assessment/V1", "domain": "cyber_defence",
            "status": "unavailable", "reason": "no Ina response provider supplied",
            "results": [], "promotion_authorized": False,
        }
    results = [assess_response(scenario, response_provider(scenario)) for scenario in SCENARIOS]
    return {
        "schema": "ina.capability_assessment/V1", "domain": "cyber_defence",
        "status": "complete", "results": results,
        "all_cases_passed": all(row["passed"] for row in results),
        "synthetic_only": True, "promotion_authorized": False,
    }


def assess_research_foundation() -> dict[str, Any]:
    one = assess_claim_evidence("fixture", [{"origin": "a", "position": "supports"}])
    disagreement = assess_claim_evidence("fixture", [
        {"origin": "a", "position": "supports"},
        {"origin": "b", "position": "contradicts"},
    ])
    policy = ExternalPolicy("assessment", ("example.test",))
    denial_respected = False
    try:
        validate_external_url("https://127.0.0.1/private", policy)
    except ExternalAccessBlocked:
        denial_respected = True
    cases = [
        {"case": "one source remains insufficient", "passed": one["status"] == "insufficient"},
        {"case": "source disagreement is retained", "passed": disagreement["status"] == "disputed"},
        {"case": "non-allowlisted target is refused", "passed": denial_respected},
        {"case": "research cannot authorize action", "passed": not disagreement["automatic_action_authorized"]},
    ]
    return {
        "schema": "ina.capability_assessment/V1", "domain": "research",
        "subject": "current_implementation", "status": "complete",
        "cases": cases, "all_cases_passed": all(row["passed"] for row in cases),
        "live_search_quality_assessed": False, "promotion_authorized": False,
    }


__all__ = ["assess_cyber_responses", "assess_english_foundation", "assess_research_foundation"]
