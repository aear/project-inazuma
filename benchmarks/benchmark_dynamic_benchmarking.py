"""V1/V2 benchmark for measured recommendations and benchmark proposals."""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from dynamic_benchmarking import evidence_route, propose_benchmark, record_measurement, storage_recommendation


def main() -> int:
    with TemporaryDirectory(prefix="ina-dynamic-benchmark-") as directory:
        root = Path(directory)
        config = {"dynamic_benchmark_policy": {
            "observation_path": str(root / "observations.jsonl"),
            "proposal_path": str(root / "proposals.jsonl"),
            "min_observations": 3, "min_independent_origins": 2,
            "min_benefit_ratio": 2.0, "min_saved_seconds": 0.02,
        }}
        for tier, elapsed in (("durable", 0.2), ("fast", 0.01)):
            for origin in ("normal_operation", "normal_operation", "bounded_probe"):
                record_measurement("Ina", "fixture", "read", config, origin=origin,
                                   tier=tier, elapsed_seconds=elapsed, bytes_processed=4096)
        report = storage_recommendation(
            "Ina", "fixture", config, durable_authoritative=True,
            rebuildable_fast_copy=True, fast_free_bytes=10**9, subject_bytes=4096,
        )
        proposal = propose_benchmark(
            "Ina", config, capability="fixture reads", reason="Repeated material latency gap.",
            uncertainty="Whether storage placement causes the latency gap.",
            signals=[{"origin": "normal_operation"}, {"origin": "bounded_probe"}],
            dimensions=["latency", "correctness", "capacity"],
            baseline_version="V1", candidate_version="V2",
            deterministic_cases=["same bytes"], held_out_cases=["new file"],
            adversarial_cases=["hash mismatch"], run_budget=1,
        )
        uncertain_route = evidence_route(material_uncertainty=True, operation_can_teach=True)
        operating_route = evidence_route(material_uncertainty=False, operation_can_teach=True)
    results = {
        "V1_static_only": {
            "operation_measurement": 0, "multi_signal_recommendation": 0,
            "proposal_without_execution": 0, "durable_source_preserved": 0,
            "uncertainty_routes_to_benchmark": 0, "operation_routes_to_learning": 0,
        },
        "V2_dynamic_review_gated": {
            "operation_measurement": 1,
            "multi_signal_recommendation": int(report["strong"] and len(report["independent_origins"]) >= 2),
            "proposal_without_execution": int(not proposal["execution_authorized"] and proposal["status"] == "review_required"),
            "durable_source_preserved": int(report["durable_source_must_remain"] and not report["automatic_migration_authorized"]),
            "uncertainty_routes_to_benchmark": int(uncertain_route["route"] == "propose_bounded_benchmark"),
            "operation_routes_to_learning": int(operating_route["route"] == "observe_normal_operation"),
        },
    }
    print(results)
    return 0 if all(results["V2_dynamic_review_gated"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
