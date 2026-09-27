"""V1/V2 comparison for developmental readiness reporting."""
from __future__ import annotations

from developmental_readiness import build_report, make_evidence


def main() -> int:
    rows = []
    for gate in ("competence", "safety", "recovery", "provenance"):
        for number, origin in enumerate(("fixture_runner", "independent_recount")):
            rows.append(make_evidence(
                "code_game", gate, 0.9, origin=origin, method="bounded fixture",
                artifact_id="game-fixture", case_id=f"{gate}-{number}",
                evaluator_type="objective", implementation_version="V2",
            ))
    report = build_report(rows)
    code = report["domains"]["code_game"]
    model = report["domains"]["model_3d"]
    results = {
        "V1_undifferentiated_score": {
            "domain_specific_gates": 0, "independent_origins": 0,
            "review_only": 0, "missing_toolchain_blocks": 0,
        },
        "V2_developmental_observatory": {
            "domain_specific_gates": int(code["readiness"] == "sandbox_ready"),
            "independent_origins": int(all(code["gates"][gate]["passed"] for gate in ("competence", "safety", "recovery", "provenance"))),
            "review_only": int(not code["promotion_authorized"] and code["human_review_required"]),
            "missing_toolchain_blocks": int(model["readiness"] == "not_assessed" and not model["toolchain_available"]),
        },
    }
    print(results)
    return 0 if all(results["V2_developmental_observatory"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
