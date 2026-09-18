import json
from pathlib import Path

import pytest

from dynamic_benchmarking import (
    bounded_read_probe,
    evidence_route,
    propose_benchmark,
    record_measurement,
    storage_recommendation,
)


def _config(tmp_path: Path):
    return {
        "dynamic_benchmark_policy": {
            "min_observations": 3,
            "min_independent_origins": 2,
            "min_benefit_ratio": 2.0,
            "min_saved_seconds": 0.02,
            "observation_path": str(tmp_path / "{child}_observations.jsonl"),
            "proposal_path": str(tmp_path / "{child}_proposals.jsonl"),
        }
    }


def test_dynamic_storage_report_needs_paired_independent_evidence(tmp_path):
    config = _config(tmp_path)
    for tier, elapsed in (("durable", 0.12), ("fast", 0.01)):
        for index in range(3):
            record_measurement(
                "Ina", "public_music", "read_excerpt", config,
                origin="playback" if index % 2 == 0 else "explicit_probe",
                tier=tier, elapsed_seconds=elapsed, bytes_processed=256_000,
                cache_state="mixed", user_visible=tier == "durable",
            )
    report = storage_recommendation(
        "Ina", "public_music", config, durable_authoritative=True,
        rebuildable_fast_copy=True, fast_free_bytes=2_000_000_000,
        subject_bytes=160_000_000,
    )
    assert report["strong"] is True
    assert report["recommendation"] == "keep_durable_source_and_add_verified_fast_copy"
    assert report["automatic_migration_authorized"] is False
    assert report["durable_source_must_remain"] is True
    assert report["benefit_ratio"] == 12.0


def test_one_origin_or_capacity_pressure_abstains(tmp_path):
    config = _config(tmp_path)
    for tier, elapsed in (("durable", 1.0), ("fast", 0.01)):
        for _ in range(3):
            record_measurement(
                "Ina", "index", "lookup", config, origin="same_probe", tier=tier,
                elapsed_seconds=elapsed, bytes_processed=4096,
            )
    report = storage_recommendation(
        "Ina", "index", config, durable_authoritative=True,
        rebuildable_fast_copy=True, fast_free_bytes=70_000_000, subject_bytes=20_000_000,
    )
    assert report["strong"] is False
    assert "insufficient_independent_origins" in report["blockers"]
    assert "fast_capacity_reserve" in report["blockers"]


def test_read_probe_is_bounded_and_content_blind(tmp_path):
    files = []
    for index in range(4):
        path = tmp_path / f"{index}.bin"
        path.write_bytes(bytes([index]) * 4096)
        files.append(path)
    result = bounded_read_probe(files, max_files=2, max_bytes=6000, max_seconds=1.0)
    assert result["bounded"] is True
    assert result["files_processed"] <= 2
    assert result["bytes_processed"] <= 6000
    assert result["digest_sha256"]
    assert "content" not in result


def test_ina_can_propose_but_not_execute_a_versioned_benchmark(tmp_path):
    config = _config(tmp_path)
    kwargs = {
        "capability": "public music playback storage",
        "reason": "Repeated playback and a bounded probe disagree with the current placement.",
        "uncertainty": "Whether storage placement materially delays playback.",
        "signals": [
            {"origin": "playback", "finding": "startup latency"},
            {"origin": "explicit_probe", "finding": "tier comparison"},
        ],
        "dimensions": ["latency", "correctness", "capacity", "human_visible_quality"],
        "baseline_version": "V1",
        "candidate_version": "V2",
        "deterministic_cases": ["same excerpt and byte range"],
        "held_out_cases": ["unseen excerpt"],
        "adversarial_cases": ["missing fast copy", "hash mismatch"],
        "run_budget": 1,
    }
    proposal = propose_benchmark("Ina", config, **kwargs)
    duplicate = propose_benchmark("Ina", config, **kwargs)
    assert proposal["created"] is True
    assert proposal["status"] == "review_required"
    assert proposal["execution_authorized"] is False
    assert proposal["source_changes_authorized"] is False
    assert duplicate["created"] is False
    records = (tmp_path / "Ina_proposals.jsonl").read_text().splitlines()
    assert len(records) == 1
    assert json.loads(records[0])["run_budget"] == 1


def test_benchmark_proposal_rejects_single_signal_or_missing_cases(tmp_path):
    config = _config(tmp_path)
    with pytest.raises(ValueError, match="two independent"):
        propose_benchmark(
            "Ina", config, capability="lookup", reason="slow",
            uncertainty="Whether storage is causal.",
            signals=[{"origin": "timer"}, {"origin": "timer"}], dimensions=["latency"],
            baseline_version="V1", candidate_version="V2",
            deterministic_cases=["known"], held_out_cases=["new"], adversarial_cases=["missing"],
        )


def test_uncertainty_routes_to_benchmark_and_normal_work_routes_to_learning():
    uncertain = evidence_route(material_uncertainty=True, operation_can_teach=True)
    operating = evidence_route(material_uncertainty=False, operation_can_teach=True)
    idle = evidence_route(material_uncertainty=False, operation_can_teach=False)
    assert uncertain["route"] == "propose_bounded_benchmark"
    assert uncertain["authority"] == "review_required"
    assert operating["route"] == "observe_normal_operation"
    assert operating["authority"] == "observation_only"
    assert idle["route"] == "defer"
