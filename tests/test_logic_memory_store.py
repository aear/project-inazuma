import os
from pathlib import Path

import logic_memory_store as store


def test_logic_store_is_sparse_durable_and_bounded_on_read(tmp_path):
    prior = Path.cwd()
    os.chdir(tmp_path)
    try:
        config = {"current_child": "Ina", "storage_layout": {"fast_runtime_enabled": False}}
        entries = [
            {"timestamp": f"2026-01-01T00:00:0{i}+00:00", "description": f"logic {i}"}
            for i in range(4)
        ]
        vectors = [[1.0, 0.0], [0.99, 0.01], [0.0, 1.0], [0.98, 0.02]]
        results = [
            store.store_logic_entry("Ina", entry, vector, config=config)
            for entry, vector in zip(entries, vectors)
        ]

        assert all(result["status"] == "stored" for result in results)
        assert store.entry_count("Ina", config) == 4
        counts = store.graph_counts("Ina", config)
        assert counts["edges"] >= 1
        recent = store.recent_entries("Ina", 2, config=config)
        assert [entry["description"] for entry in recent] == ["logic 2", "logic 3"]

        duplicate = store.store_logic_entry("Ina", entries[-1], vectors[-1], config=config)
        assert duplicate["inserted"] is False
        assert store.entry_count("Ina", config) == 4
    finally:
        os.chdir(prior)


def test_logic_capacity_is_independent_of_active_window(tmp_path):
    config = {
        "logic_store_policy": {"max_entries": 2_000_000},
        "logic_map_policy": {"burst": 120},
    }
    assert store._policy(config)["max_entries"] == 2_000_000

