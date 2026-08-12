from pathlib import Path
from types import SimpleNamespace

import model_manager as mm


def test_recall_touch_never_loads_or_rewrites_full_json(monkeypatch, tmp_path):
    manager = SimpleNamespace(
        child="Ina",
        index_path=tmp_path / "memory_map.json",
        load_map=lambda: (_ for _ in ()).throw(AssertionError("full map loaded")),
        save_map=lambda: (_ for _ in ()).throw(AssertionError("full map rewritten")),
    )
    db_path = tmp_path / "memory_map.sqlite"
    calls = []
    monkeypatch.setattr(mm, "_memory_index_db_path", lambda child: db_path)
    monkeypatch.setattr(mm, "ensure_memory_index_db", lambda json_path, target: True)
    monkeypatch.setattr(
        mm,
        "touch_fragments",
        lambda target, ids, timestamp: calls.append((target, ids, timestamp)) or len(ids),
    )

    mm._MemoryIndexUpdater(manager).ingest_fragments([{"id": "a"}, {"id": "b"}])

    assert calls
    assert calls[0][0] == db_path
    assert calls[0][1] == ["a", "b"]
