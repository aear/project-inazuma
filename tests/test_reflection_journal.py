import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import precision_memory_map as pmm
import reflection_journal as rj


CHILD = "TestReflectionJournal"


def _context(module="memory_graph", intensity=0.4):
    return {
        "active_modules": [module],
        "emotion_state": {"intensity": intensity, "stress": 0.2},
        "energy": 0.7,
    }


class ReflectionJournalTests(unittest.TestCase):
    def test_write_note_preserves_content_and_recent_notes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir)
            content = "prediction cost high, outcome low -- inefficient pattern"
            symbolic = "\u2218\u03c6 unstable when memory load high"

            note = rj.write_note(
                content,
                context=_context(),
                tags=["pattern", "symbolic"],
                freeform={"symbolic": symbolic},
                child=CHILD,
                base_path=base_path,
            )
            rj.write_reflection(
                "reflection scratch",
                context=_context(module="logic_engine"),
                tags=["scratch"],
                child=CHILD,
                base_path=base_path,
            )

            notes = rj.get_recent_notes(child=CHILD, base_path=base_path)
            self.assertEqual(len(notes), 1)
            self.assertEqual(notes[0], note)
            self.assertEqual(notes[0]["content"], content)
            self.assertEqual(notes[0]["freeform"], {"symbolic": symbolic})
            self.assertEqual(notes[0]["context"]["active_modules"], ["memory_graph"])

            raw = rj.reflection_journal_path(CHILD, base_path=base_path).read_text(encoding="utf-8")
            self.assertIn(symbolic, raw)

    def test_precision_high_cost_event_writes_linked_note_and_reflects(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir)
            event = pmm.log_event(
                _context(),
                {"cpu": {"memory_graph": 91.0}, "ram": {"memory_graph": 10.0}},
                {"global": 32.0, "per_module": {"memory_graph": 32.0}},
                {"status": "stalled", "duration": 6.0, "completion": 0.1, "regret": 12.0},
                child=CHILD,
                base_path=base_path,
            )

            notes = rj.get_recent_notes(child=CHILD, base_path=base_path)
            self.assertEqual(len(notes), 1)
            self.assertIn("precision decision note", notes[0]["content"])
            self.assertIn("high_regret", notes[0]["tags"])
            self.assertEqual(notes[0]["linked_events"], [event["id"]])

            reflection = rj.reflect_on_event(event["id"], child=CHILD, base_path=base_path)
            self.assertEqual(reflection["type"], "reflection")
            self.assertIn("what_happened", reflection["content"])
            self.assertIn("what_worked", reflection["content"])
            self.assertIn("what_did_not_work", reflection["content"])
            self.assertEqual(reflection["linked_events"], [event["id"]])

    def test_generate_journal_summarizes_recent_signals_and_persists(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir)
            memory = base_path / CHILD / "memory"
            memory.mkdir(parents=True, exist_ok=True)
            now = datetime.now(timezone.utc).isoformat()
            with (memory / "emotion_log.jsonl").open("w", encoding="utf-8") as fh:
                fh.write(json.dumps({"timestamp": now, "mode": "awake", "values": {"intensity": 0.8, "stress": 0.5}}, ensure_ascii=False) + "\n")

            event = pmm.log_event(
                _context(module="predictive_layer", intensity=0.8),
                {"cpu": {"predictive_layer": 88.0}, "ram": {"predictive_layer": 3.0}},
                {"global": 48.0, "per_module": {"predictive_layer": 48.0}},
                {"status": "overload", "duration": 4.0, "completion": 0.2, "regret": 9.0},
                child=CHILD,
                base_path=base_path,
            )

            content = rj.generate_journal(period="daily", child=CHILD, base_path=base_path)
            self.assertIn("dominant modules: predictive_layer", content)
            self.assertIn("emotional patterns:", content)
            self.assertIn("high-cost decisions:", content)
            self.assertIn(event["id"], content)

            journals = rj.get_recent_entries(
                entry_types="journal",
                child=CHILD,
                base_path=base_path,
            )
            self.assertEqual(len(journals), 1)
            self.assertEqual(journals[0]["content"], content)
            self.assertEqual(journals[0]["linked_events"], [event["id"]])

    def test_emotion_snapshot_reflection_threshold(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir)
            low = rj.reflect_on_emotion_snapshot(
                {"mode": "awake", "values": {"intensity": 0.2}},
                child=CHILD,
                base_path=base_path,
            )
            self.assertIsNone(low)

            high = rj.reflect_on_emotion_snapshot(
                {"mode": "awake", "values": {"intensity": 0.9, "stress": 0.7}},
                context={"active_modules": ["emotion_engine"], "energy": 0.4},
                tags=["test_spike"],
                child=CHILD,
                base_path=base_path,
            )
            self.assertIsNotNone(high)
            assert high is not None
            self.assertEqual(high["type"], "reflection")
            self.assertIn("emotion_spike", high["tags"])
            self.assertIn("test_spike", high["tags"])
            self.assertIn("emotion spike", high["content"])


if __name__ == "__main__":
    unittest.main()
