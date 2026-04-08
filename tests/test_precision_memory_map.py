import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import precision_memory_map as pmm


CHILD = "TestPrecisionMemoryMap"


def _context(module="memory_graph", queue=0.25, focus=0.7):
    return {
        "active_modules": [module],
        "queue_pressure": queue,
        "emotion_state": {"focus": focus, "stress": 0.2},
        "energy": 0.8,
    }


class PrecisionMemoryMapTests(unittest.TestCase):
    def test_log_event_appends_jsonl_and_calculates_regret(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir)
            cost = {"cpu": {"memory_graph": 20.0, "logic": 40.0}, "ram": {"memory_graph": 1.0, "logic": 3.0}}
            outcome = {"status": "stable", "duration": 2.5, "completion": 0.75}
            precision = {"global": 32.123456789, "per_module": {"memory_graph": 31.987654321}}

            entry = pmm.log_event(_context(), cost, precision, outcome, child=CHILD, base_path=base_path)

            path = pmm.precision_memory_log_path(CHILD, base_path=base_path)
            lines = path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 1)
            persisted = json.loads(lines[0])
            self.assertEqual(persisted, entry)
            self.assertEqual(persisted["precision"]["global"], 32.123456789)
            self.assertEqual(persisted["precision"]["per_module"]["memory_graph"], 31.987654321)
            self.assertTrue(math.isclose(persisted["outcome"]["regret"], 4.7))

    def test_get_similar_contexts_prefers_module_emotion_and_queue_match(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir)
            pmm.log_event(
                _context(module="audio", queue=0.9, focus=0.1),
                {"cpu": {"audio": 10.0}, "ram": {"audio": 0.5}},
                {"global": 16.0, "per_module": {"audio": 16.0}},
                {"status": "efficient", "duration": 1.0, "completion": 1.0, "regret": 0.0},
                child=CHILD,
                base_path=base_path,
            )
            pmm.log_event(
                _context(module="memory_graph", queue=0.3, focus=0.68),
                {"cpu": {"memory_graph": 12.0}, "ram": {"memory_graph": 1.0}},
                {"global": 48.0, "per_module": {"memory_graph": 47.5}},
                {"status": "stable", "duration": 1.0, "completion": 0.9, "regret": 0.1},
                child=CHILD,
                base_path=base_path,
            )

            similar = pmm.get_similar_contexts(
                _context(module="memory_graph", queue=0.28, focus=0.7),
                k=2,
                child=CHILD,
                base_path=base_path,
            )

            self.assertEqual([event["precision"]["global"] for event in similar], [48.0, 16.0])
            self.assertGreater(similar[0]["similarity"], similar[1]["similarity"])

    def test_suggest_precision_averages_successful_neighbors_only(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir)
            for global_precision, module_precision, status, completion, regret in [
                (40.0, 41.0, "stable", 1.0, 0.1),
                (50.0, 49.0, "efficient", 0.9, 0.05),
                (8.0, 8.0, "overload", 0.2, 9.0),
            ]:
                pmm.log_event(
                    _context(module="memory_graph", queue=0.25, focus=0.7),
                    {"cpu": {"memory_graph": 10.0}, "ram": {"memory_graph": 1.0}},
                    {"global": global_precision, "per_module": {"memory_graph": module_precision}},
                    {"status": status, "duration": 1.0, "completion": completion, "regret": regret},
                    child=CHILD,
                    base_path=base_path,
                )

            suggestion = pmm.suggest_precision(
                _context(module="memory_graph", queue=0.25, focus=0.7),
                child=CHILD,
                base_path=base_path,
            )

            self.assertEqual(suggestion["basis"], "nearest_neighbors")
            self.assertEqual(suggestion["successful_events"], 2)
            self.assertGreater(suggestion["global"], 40.0)
            self.assertLess(suggestion["global"], 50.0)
            self.assertGreater(suggestion["per_module"]["memory_graph"], 41.0)
            self.assertLess(suggestion["per_module"]["memory_graph"], 49.0)
            self.assertGreater(suggestion["confidence"], 0.0)

    def test_suggest_precision_returns_advisory_empty_when_no_successful_memory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir)
            pmm.log_event(
                _context(),
                {"cpu": {"memory_graph": 90.0}, "ram": {"memory_graph": 8.0}},
                {"global": 8.0, "per_module": {"memory_graph": 8.0}},
                {"status": "stalled", "duration": 5.0, "completion": 0.1, "regret": 12.0},
                child=CHILD,
                base_path=base_path,
            )

            suggestion = pmm.suggest_precision(_context(), child=CHILD, base_path=base_path)

            self.assertIsNone(suggestion["global"])
            self.assertEqual(suggestion["per_module"], {})
            self.assertEqual(suggestion["basis"], "insufficient_successful_memory")


if __name__ == "__main__":
    unittest.main()
