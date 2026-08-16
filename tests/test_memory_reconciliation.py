import json
import os
import tempfile
import unittest
from pathlib import Path

import memory_mirror_db as mirror
import memory_reconciliation as reconciliation


class MemoryReconciliationTests(unittest.TestCase):
    def setUp(self):
        mirror.flush_mirror_writes(close=True)
        mirror._SESSION_CACHE.clear()
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.old_cwd = Path.cwd()
        os.chdir(self.root)
        self.events = Path("AI_Children/Ina/memory/experiences/events")
        self.events.mkdir(parents=True)
        self.config = {
            "memory_reconciliation_policy": {
                "include_legacy_events": True,
                "scan_fragments": False,
            },
            "memory_mirror_policy": {
                "enabled": True,
                "mirror_on_read": True,
                "db_root": str(self.root / "mirror"),
                "db_filename": "catalog.sqlite3",
                "batch_records": 2,
                "batch_bytes": 1024 * 1024,
                "batch_seconds": 60,
                "wal_autocheckpoint_pages": 1000,
                "synchronous": "NORMAL",
                "remove_json_after_verified": False,
                "quarantine_json_after_verified": False,
            }
        }

    def tearDown(self):
        mirror.flush_mirror_writes(close=True)
        mirror._SESSION_CACHE.clear()
        os.chdir(self.old_cwd)
        self.tempdir.cleanup()

    def _event(self, index: int) -> Path:
        path = self.events / f"evt_20260101T00000{index}Z.json"
        path.write_text(json.dumps({
            "id": path.stem,
            "timestamp": f"2026-01-01T00:00:0{index}+00:00",
            "importance": 0.1,
            "narrative": "benchmark candidate " + ("x" * 128),
        }), encoding="utf-8")
        return path

    def test_bounded_steps_resume_through_verified_catalogue(self):
        paths = [self._event(index) for index in range(3)]

        first = reconciliation.reconcile_step(
            "Ina", max_new_records=2, max_seconds=30, config=self.config
        )
        second = reconciliation.reconcile_step(
            "Ina", max_new_records=2, max_seconds=30, config=self.config
        )

        self.assertEqual(first["catalogued_this_step"], 2)
        self.assertFalse(first["completed"])
        self.assertEqual(second["catalogued_this_step"], 1)
        self.assertTrue(second["completed"])
        for path in paths:
            self.assertTrue(
                mirror.catalog_path_is_current(
                    "Ina", "experience_event", path, config=self.config
                )
            )

    def test_catalogue_supplies_graph_candidates_without_directory_walk(self):
        self._event(0)
        reconciliation.reconcile_step(
            "Ina", max_new_records=10, max_seconds=30, config=self.config
        )
        candidates = mirror.experience_catalog_candidates(
            "Ina",
            limit=10,
            min_age_hours=0,
            max_importance=0.2,
            min_size_bytes=1,
            config=self.config,
        )
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["kind"], "experience_event")


if __name__ == "__main__":
    unittest.main()
