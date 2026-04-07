import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import early_comm as ec


def _write_json(path, payload, mtime):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    os.utime(path, (mtime, mtime))


def test_newest_matching_paths_keeps_only_recent_matches(tmp_path):
    for index in range(5):
        _write_json(tmp_path / f"evt_{index}.json", {"index": index}, 100 + index)
    _write_json(tmp_path / "other.json", {"index": 99}, 999)

    newest = ec._newest_matching_paths(tmp_path, "evt_*.json", 2)

    assert [path.name for path in newest] == ["evt_4.json", "evt_3.json"]


def test_load_recent_heard_words_uses_bounded_recent_event_window(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    events_dir = tmp_path / "AI_Children" / "Ina" / "memory" / "experiences" / "events"
    _write_json(
        events_dir / "evt_old.json",
        {
            "timestamp": "old",
            "word_usage": [{"speaker": "Sakura", "words": ["oldword"], "utterance": "oldword"}],
        },
        100,
    )
    _write_json(
        events_dir / "evt_mid.json",
        {
            "timestamp": "mid",
            "word_usage": [{"speaker": "Sakura", "words": ["midword"], "utterance": "midword"}],
        },
        200,
    )
    _write_json(
        events_dir / "evt_new.json",
        {
            "timestamp": "new",
            "word_usage": [{"speaker": "Sakura", "words": ["newword"], "utterance": "newword"}],
        },
        300,
    )

    heard = ec.load_recent_heard_words("Ina", limit=4, event_scan_limit=2)

    assert [item["word"] for item in heard] == ["newword", "midword"]


def test_early_comm_import_does_not_load_model_manager():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import early_comm; print('model_manager' in sys.modules)",
        ],
        cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"
