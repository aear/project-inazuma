import json
import os
import time
from pathlib import Path

from experience_archive import archive_step, load_archived_experience
from experience_logger import ExperienceLogger


def _event(event_id: str, narrative: str) -> dict:
    return {
        "id": event_id,
        "timestamp": "2026-01-01T00:00:00+00:00",
        "situation_tags": ["test", "memory"],
        "perceived_entities": [],
        "actions": [],
        "outcome": {"remembered": True},
        "internal_state": {},
        "narrative": narrative,
        "episode_id": None,
        "word_usage": [],
    }


def test_archive_step_is_lossless_bounded_and_reported(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    child = "Ina"
    events = Path("AI_Children") / child / "memory" / "experiences" / "events"
    events.mkdir(parents=True)
    archive = tmp_path / "durable" / "experience_archive.sqlite3"
    state = Path("AI_Children") / child / "memory" / "experience_archive_state.json"
    config = {
        "current_child": child,
        "experience_archive_policy": {
            "enabled": True,
            "batch_files": 2,
            "max_seconds": 10,
            "min_age_hours": 1,
            "compression_level": 6,
            "archive_path": str(archive),
            "state_path": str(state),
        },
    }
    Path("config.json").write_text(json.dumps(config), encoding="utf-8")

    old_paths = []
    for index in range(3):
        path = events / f"evt_20260101T00000{index}000000Z.json"
        path.write_text(json.dumps(_event(path.stem, "A repeated but exact memory. " * 20), indent=2), encoding="utf-8")
        old = time.time() - 7200
        os.utime(path, (old, old))
        old_paths.append(path)
    recent = events / "evt_20260807T100000000000Z.json"
    recent.write_text(json.dumps(_event(recent.stem, "recent")), encoding="utf-8")

    first = archive_step(child, config=config)
    assert first["run"]["archived"] == 2
    assert first["run"]["saved_bytes"] > 0
    assert len(first["history"]) == 1
    assert sum(path.exists() for path in old_paths) == 1
    assert recent.exists()

    second = archive_step(child, config=config)
    assert second["archive"]["records"] == 3
    assert second["cumulative"]["archived"] == 3
    assert len(second["history"]) == 2
    restored = load_archived_experience(child, "experience_event", old_paths[0].stem, config=config)
    assert restored == _event(old_paths[0].stem, "A repeated but exact memory. " * 20)

    logger = ExperienceLogger(child=child, base_path=Path("AI_Children"))
    recalled = logger._load_event(old_paths[0].stem)
    assert recalled.narrative.startswith("A repeated but exact memory")
