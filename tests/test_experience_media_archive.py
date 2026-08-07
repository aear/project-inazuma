import json
import os
import time
from pathlib import Path

from experience_media_archive import load_archived_media, media_archive_step
from experience_storage import sharded_media_dir


def test_media_archive_preserves_exact_bytes_and_retires_flat_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    child = "Ina"
    media = Path("AI_Children") / child / "memory" / "experiences" / "live_media"
    media.mkdir(parents=True)
    archive = tmp_path / "durable" / "live_media.sqlite3"
    state = Path("AI_Children") / child / "memory" / "experience_media_archive_state.json"
    config = {
        "experience_archive_policy": {
            "batch_files": 10,
            "max_seconds": 10,
            "min_age_hours": 1,
            "compression_level": 6,
            "media_archive_path": str(archive),
            "media_state_path": str(state),
        }
    }
    event_id = "evt_20260101T000000000000Z"
    payloads = {
        f"{event_id}_screen.json": json.dumps({"event_id": event_id, "recognized_text": ["hello"]}, indent=2).encode(),
        f"{event_id}_screen.png": b"\x89PNG\r\n\x1a\n" + bytes(range(128)),
    }
    for filename, raw in payloads.items():
        path = media / filename
        path.write_bytes(raw)
        old = time.time() - 7200
        os.utime(path, (old, old))

    result = media_archive_step(child, config=config)
    assert result["run"]["archived"] == 2
    assert result["archive"]["records"] == 2
    assert len(result["history"]) == 1
    for filename, raw in payloads.items():
        assert not (media / filename).exists()
        assert load_archived_media(child, filename, config=config) == raw

    shard = sharded_media_dir(media, event_id)
    assert shard.parts[-5:] == ("by_time", "2026", "01", "01", "00")
