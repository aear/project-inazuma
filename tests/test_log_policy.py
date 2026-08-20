from pathlib import Path

from log_policy import classify_log_path, inventory_logs


def test_sneaky_jsonl_cases_are_classified_by_role_not_extension():
    assert classify_log_path("logs/comms_core.jsonl").category == "operational"
    assert classify_log_path("benchmark_results/history.jsonl").category == "benchmark"
    assert classify_log_path("benchmarks/persistent_cognition_v1.jsonl").category == "fixture"
    assert classify_log_path("AI_Children/Ina/memory/emotion_log.jsonl").category == "memory_adjacent"
    assert classify_log_path("AI_Children/Ina/memory/self_read_incidents.jsonl").category == "audit"
    assert classify_log_path("crashes/core.123").category == "diagnostic"
    assert classify_log_path("logs/ina_status.log.1").category == "operational"
    assert classify_log_path("logs/ina_status.log.2.gz").category == "operational"
    assert classify_log_path("alignment/core.py") is None
    assert classify_log_path("memory/history.py") is None
    assert classify_log_path("REFACTORING_AUDIT.md") is None


def test_inventory_is_bounded_and_skips_memory_tree(tmp_path):
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "ina_status.log").write_text("status\n", encoding="utf-8")
    memory = tmp_path / "AI_Children" / "Ina" / "memory"
    memory.mkdir(parents=True)
    (memory / "emotion_log.jsonl").write_text("{}\n", encoding="utf-8")

    report = inventory_logs(tmp_path, max_files=50)

    assert [row["path"] for row in report["files"]] == ["logs/ina_status.log"]
    assert report["excluded_directories"]
    assert report["truncated"] is False


def test_inventory_reports_size_policy_without_mutating_file(tmp_path):
    path = tmp_path / "precision_window.log"
    path.write_bytes(b"x" * (8 * 1024 * 1024 + 1))

    report = inventory_logs(tmp_path)

    assert report["files"][0]["over_size_policy"] is True
    assert path.stat().st_size == 8 * 1024 * 1024 + 1
