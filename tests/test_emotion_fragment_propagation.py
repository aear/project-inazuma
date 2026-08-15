import json
import sqlite3

import emotion_engine
import model_manager
from cognition_runtime.default_capabilities import build_task_profiles


def _build_index(path, rows):
    with sqlite3.connect(str(path)) as connection:
        connection.execute(
            "CREATE TABLE fragments(frag_id TEXT PRIMARY KEY, tier TEXT, filename TEXT)"
        )
        connection.executemany("INSERT INTO fragments VALUES (?, ?, ?)", rows)


def test_emotion_fragment_propagation_is_indexed_bounded_and_resumable(tmp_path, monkeypatch):
    child = "benchmark"
    memory = tmp_path / child / "memory"
    fragments = memory / "fragments"
    fragments.mkdir(parents=True)
    rows = []
    for index in range(5):
        frag_id = f"f{index}"
        filename = f"frag_{index}.json"
        (fragments / filename).write_text(json.dumps({"id": frag_id}), encoding="utf-8")
        rows.append((frag_id, "", filename))
    _build_index(memory / "memory_map.sqlite", rows)

    state = {}
    monkeypatch.setattr(emotion_engine, "AI_CHILDREN_ROOT", tmp_path)
    monkeypatch.setattr(emotion_engine, "EMOTION_FRAGMENT_BATCH_LIMIT", 2)
    monkeypatch.setattr(emotion_engine, "get_inastate", lambda key, default=None: state.get(key, default))
    monkeypatch.setattr(emotion_engine, "update_inastate", lambda key, value: state.__setitem__(key, value))
    snapshot = emotion_engine.EmotionSnapshot(dict(emotion_engine.DEFAULT_BASELINE), "awake", "now")

    first = emotion_engine.tag_all_fragments(child, snapshot)
    second = emotion_engine.tag_all_fragments(child, snapshot)

    assert first == {"updated": 2, "skipped": 0, "inspected": 2, "deferred": None}
    assert second == {"updated": 2, "skipped": 0, "inspected": 2, "deferred": None}
    assert state[emotion_engine._EMOTION_FRAGMENT_CURSOR_KEY]["frag_id"] == "f3"
    assert "emotions" in json.loads((fragments / "frag_3.json").read_text(encoding="utf-8"))
    assert "emotions" not in json.loads((fragments / "frag_4.json").read_text(encoding="utf-8"))


def test_emotion_fragment_propagation_defers_without_index(tmp_path, monkeypatch):
    child = "benchmark"
    (tmp_path / child / "memory" / "fragments").mkdir(parents=True)
    monkeypatch.setattr(emotion_engine, "AI_CHILDREN_ROOT", tmp_path)
    monkeypatch.setattr(emotion_engine, "get_inastate", lambda key, default=None: default)

    snapshot = emotion_engine.EmotionSnapshot(dict(emotion_engine.DEFAULT_BASELINE), "awake", "now")
    result = emotion_engine.tag_all_fragments(child, snapshot)

    assert result["deferred"] == "index_unavailable"
    assert result["inspected"] == 0


def test_emotion_engine_has_individual_and_emergency_memory_guards():
    profile = build_task_profiles("benchmark")["emotion_engine_run"]

    assert profile["memory_limit_gb"] == 1.5
    assert "emotion_engine.py" in model_manager._default_shed_patterns()
