import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import memory_mirror_db as mirror


def _config(tmp_path: Path, **overrides):
    policy = {
        "enabled": True,
        "mirror_on_read": True,
        "db_root": str(tmp_path / "mirror"),
        "db_filename": "test.sqlite3",
        "batch_records": 256,
        "batch_bytes": 16 * 1024 * 1024,
        "batch_seconds": 60,
        "wal_autocheckpoint_pages": 4096,
        "synchronous": "NORMAL",
        "remove_json_after_verified": False,
        "quarantine_json_after_verified": False,
    }
    policy.update(overrides)
    return {"memory_mirror_policy": policy}


def _record(tmp_path: Path, name: str, value: int) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps({"id": name, "value": value}), encoding="utf-8")
    return path


def _stored_row(db_path: Path, item_id: str):
    conn = sqlite3.connect(str(db_path))
    try:
        return conn.execute(
            "SELECT verified_at, removal_eligible, payload_json "
            "FROM mirrored_json WHERE item_id = ?",
            (item_id,),
        ).fetchone()
    finally:
        conn.close()


class MirrorBatchingTests(unittest.TestCase):
    def setUp(self):
        mirror.flush_mirror_writes(close=True)
        mirror._SESSION_CACHE.clear()
        self.tempdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tempdir.name)

    def tearDown(self):
        mirror.flush_mirror_writes(close=True)
        mirror._SESSION_CACHE.clear()
        self.tempdir.cleanup()

    def test_mirror_queues_then_flushes_and_verifies(self):
        cfg = _config(self.tmp_path)
        source = _record(self.tmp_path, "one.json", 1)
        db_path = mirror.mirror_db_path("Ina", cfg)

        result = mirror.mirror_json_file("Ina", "fragment", source, config=cfg)

        self.assertEqual(result["status"], "queued_for_verification")
        self.assertFalse(result["verified"])
        self.assertTrue(source.exists())

        self.assertEqual(mirror.flush_mirror_writes(db_path), 1)
        row = _stored_row(db_path, "one.json")
        self.assertIsNotNone(row)
        self.assertTrue(row[0])
        self.assertEqual(row[1], 1)
        self.assertEqual(json.loads(row[2])["value"], 1)

    def test_batch_threshold_verifies_all_rows_with_one_reused_session(self):
        cfg = _config(self.tmp_path, batch_records=2)
        first = _record(self.tmp_path, "first.json", 1)
        second = _record(self.tmp_path, "second.json", 2)

        first_result = mirror.mirror_json_file("Ina", "fragment", first, config=cfg)
        second_result = mirror.mirror_json_file("Ina", "fragment", second, config=cfg)

        self.assertFalse(first_result["verified"])
        self.assertTrue(second_result["verified"])
        self.assertEqual(len(mirror._MIRROR_SESSIONS), 1)
        db_path = mirror.mirror_db_path("Ina", cfg)
        self.assertEqual(_stored_row(db_path, "first.json")[1], 1)
        self.assertEqual(_stored_row(db_path, "second.json")[1], 1)

    def test_source_removal_forces_durable_verification(self):
        cfg = _config(self.tmp_path, remove_json_after_verified=True)
        source = _record(self.tmp_path, "remove.json", 3)

        result = mirror.mirror_json_file("Ina", "fragment", source, config=cfg)

        self.assertEqual(result["status"], "json_removed")
        self.assertTrue(result["verified"])
        self.assertTrue(result["removal_eligible"])
        self.assertFalse(source.exists())
        db_path = mirror.mirror_db_path("Ina", cfg)
        row = _stored_row(db_path, "remove.json")
        self.assertTrue(row[0])
        self.assertEqual(row[1], 1)

    def test_close_flushes_pending_rows_and_drops_session(self):
        cfg = _config(self.tmp_path)
        source = _record(self.tmp_path, "close.json", 4)
        db_path = mirror.mirror_db_path("Ina", cfg)
        mirror.mirror_json_file("Ina", "fragment", source, config=cfg)

        self.assertEqual(mirror.flush_mirror_writes(db_path, close=True), 1)
        self.assertFalse(mirror._MIRROR_SESSIONS)
        self.assertEqual(_stored_row(db_path, "close.json")[1], 1)


    def test_verified_row_is_a_persistent_restart_checkpoint(self):
        cfg = _config(self.tmp_path)
        source = _record(self.tmp_path, "checkpoint.json", 5)
        payload = {"id": "checkpoint.json", "value": 5}
        db_path = mirror.mirror_db_path("Ina", cfg)
        mirror.mirror_json_file("Ina", "fragment", source, payload=payload, config=cfg)
        mirror.flush_mirror_writes(db_path, close=True)
        mirror._SESSION_CACHE.clear()

        with mock.patch.object(Path, "read_text", side_effect=AssertionError("source reread")):
            result = mirror.mirror_json_file(
                "Ina", "fragment", source, payload=payload, config=cfg
            )

        self.assertEqual(result["status"], "cached_verified")
        self.assertTrue(result["verified"])
        session = next(iter(mirror._MIRROR_SESSIONS.values()))
        self.assertFalse(session["pending"])
if __name__ == "__main__":
    unittest.main()
