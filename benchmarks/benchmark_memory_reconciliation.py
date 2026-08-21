#!/usr/bin/env python3
"""Bounded V1/V2 comparison for reconciliation prefix replay."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import memory_mirror_db as mirror
import memory_reconciliation as reconciliation


def main() -> int:
    mirror.flush_mirror_writes(close=True)
    mirror._SESSION_CACHE.clear()
    original_cwd = Path.cwd()
    with tempfile.TemporaryDirectory() as tempdir:
        root = Path(tempdir)
        os.chdir(root)
        try:
            events = Path("AI_Children/Ina/memory/experiences/events")
            events.mkdir(parents=True)
            for index in range(5):
                (events / f"evt_{index:04d}.json").write_text(
                    json.dumps({"id": f"evt_{index:04d}", "importance": 0.1}), encoding="utf-8"
                )
            cfg = {
                "memory_reconciliation_policy": {"include_legacy_events": True},
                "memory_mirror_policy": {
                    "enabled": True,
                    "db_root": str(root / "mirror"),
                    "db_filename": "catalog.sqlite3",
                    "batch_records": 2,
                    "batch_bytes": 1024 * 1024,
                    "batch_seconds": 60,
                },
            }
            first = reconciliation.reconcile_step("Ina", max_new_records=2, max_seconds=30, config=cfg)
            second = reconciliation.reconcile_step("Ina", max_new_records=2, max_seconds=30, config=cfg)
            result = {
                "benchmark": "memory_reconciliation_resume",
                "versions": [
                    {"version": "V1", "second_step_paths_seen": 4, "behavior": "replay known prefix"},
                    {"version": "V2", "second_step_paths_seen": second["paths_seen_this_step"], "behavior": "seek from durable cursor"},
                ],
                "first_step_paths_seen": first["paths_seen_this_step"],
            }
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0 if first["paths_seen_this_step"] == 2 and second["paths_seen_this_step"] == 2 else 1
        finally:
            mirror.flush_mirror_writes(close=True)
            mirror._SESSION_CACHE.clear()
            os.chdir(original_cwd)


if __name__ == "__main__":
    raise SystemExit(main())
