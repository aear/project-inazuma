import json
import os
from pathlib import Path

import emotion_symbol_store as store


def _write_source(child: str, count: int) -> Path:
    path = store.source_path(child)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "symbols": [
            {
                "symbol_word_id": f"sym_{index}",
                "symbol": f"S{index}",
                "summary": f"state {index}",
                "average_emotion": {"trust": index / 10},
                "vector": [index / 10],
            }
            for index in range(count)
        ]
    }, indent=2), encoding="utf-8")
    return path


def test_resumable_emotion_database_migration_retains_json(tmp_path):
    prior = Path.cwd()
    os.chdir(tmp_path)
    try:
        source = _write_source("Ina", 5)
        config = {"current_child": "Ina", "storage_layout": {"fast_runtime_enabled": False}}

        first = store.migration_step("Ina", max_records=2, max_seconds=30, config=config)
        second = store.migration_step("Ina", max_records=2, max_seconds=30, config=config)
        final = store.migration_step("Ina", max_records=2, max_seconds=30, config=config)

        assert first["status"] == "copying"
        assert second["status"] == "copying"
        assert final["status"] == "complete"
        assert final["verification"] == "ok"
        assert final["source_retained"] is True
        assert source.is_file() and not source.is_symlink()
        assert store.database_ready("Ina", config)
        assert store.symbol_count("Ina", config) == 5
        assert [item["symbol_word_id"] for item in store.iter_symbol_payloads("Ina", config=config)] == [
            "sym_0", "sym_1", "sym_2", "sym_3", "sym_4"
        ]
        candidates = list(store.iter_candidate_payloads("Ina", [0.2], config=config))
        assert candidates
        assert len(candidates) <= 5
    finally:
        os.chdir(prior)


def test_source_change_blocks_resume(tmp_path):
    prior = Path.cwd()
    os.chdir(tmp_path)
    try:
        source = _write_source("Ina", 3)
        config = {"current_child": "Ina", "storage_layout": {"fast_runtime_enabled": False}}
        first = store.migration_step("Ina", max_records=1, max_seconds=30, config=config)
        assert first["status"] == "copying"
        source.write_text(source.read_text(encoding="utf-8") + " ", encoding="utf-8")

        failed = store.migration_step("Ina", max_records=1, max_seconds=30, config=config)
        assert failed["status"] == "failed"
        assert failed["error"] == "source_changed_during_migration"
    finally:
        os.chdir(prior)

