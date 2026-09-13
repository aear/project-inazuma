import json

import monitoring_dashboard
from module_benchmarks import benchmark_module
from transformer_comparison import compare_transformer_families


def _records(*, trained=True):
    rows = []
    for seed in range(3):
        for family, accuracy, margin, elapsed in (
            ("federated_ina", .5, .1, 1.0),
            ("conventional_transformer", .8, .3, 1.2),
        ):
            rows.append({
                "benchmark": "fresh", "benchmark_version": "2",
                "evaluation_protocol": "procedural-generative",
                "seed_fingerprint": f"seed-{seed}", "model_family": family,
                "trained_weights": trained if family == "conventional_transformer" else True,
                "elapsed_seconds": elapsed, "total": 10,
                "categories": {"pragmatics": {"accuracy": accuracy, "mean_margin": margin}},
            })
    return rows


def test_comparison_recommends_only_a_specific_task_from_repeated_signals():
    report = compare_transformer_families(_records())
    recommendation = report["recommendations"][0]
    assert recommendation["task"] == "pragmatics"
    assert recommendation["matched_independent_witnesses"] == 3
    assert recommendation["signals"]["mean_accuracy_gain"] == .3
    assert recommendation["automatic_promotion"] is False


def test_untrained_or_single_smoke_result_cannot_trigger_recommendation():
    assert compare_transformer_families(_records(trained=False))["recommendations"] == []
    smoke = dict(_records()[0], evaluation_protocol="public-smoke")
    assert compare_transformer_families([smoke])["recommendations"] == []


def test_monitor_highlights_review_candidate(tmp_path):
    path = tmp_path / "history.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in _records()), encoding="utf-8")
    cards, rows = monitoring_dashboard._transformer_benchmarks(path)
    assert ("Review candidates", "1") in cards
    candidate = next(row for row in rows if row[0].startswith("pragmatics"))
    assert "highlight" in candidate[2]
    assert "missing federated capability" in candidate[4]


def test_versioned_comparison_and_social_expression_benchmarks():
    comparison_v1, comparison_v2 = benchmark_module("transformer_comparison")
    meaning_v1, meaning_v2, meaning_v3 = benchmark_module("communicative_meaning")
    assert (comparison_v1.correct, comparison_v2.correct) == (0, 4)
    assert meaning_v3.correct == meaning_v3.total == 6
