import json
import os
import time

from discord_retention import BoundedIdSet, compact_jsonl_tail, prune_buffer_files, tail_jsonl_entries
from module_benchmarks import benchmark_module


def test_history_tail_and_seen_ids_stay_bounded(tmp_path):
    history = tmp_path / "history.jsonl"
    with history.open("w", encoding="utf-8") as handle:
        for index in range(2000):
            handle.write(json.dumps({"id": str(index)}) + "\n")
    old_size = history.stat().st_size
    result = compact_jsonl_tail(history, max_bytes=1024, keep_lines=100, tail_bytes=8192)
    entries = tail_jsonl_entries(history, max_lines=100, max_tail_bytes=8192)
    seen = BoundedIdSet(32, (entry["id"] for entry in entries))
    assert result["compacted"] is True
    assert history.stat().st_size < old_size
    assert len(entries) <= 100
    assert len(seen) == 32
    assert "1999" in seen


def test_voice_buffer_retention_enforces_all_bounds(tmp_path):
    voice = tmp_path / "voice"; voice.mkdir()
    for index in range(6):
        path = voice / f"{index}.pcm"; path.write_bytes(b"x" * 16)
        os.utime(path, (time.time() + index, time.time() + index))
    result = prune_buffer_files(voice, max_files=3, max_bytes=64, max_age_hours=24)
    assert result["removed_files"] == 3
    assert result["remaining_files"] == 3
    assert result["remaining_bytes"] == 48


def test_discord_retention_benchmark_beats_history_version():
    v1, v2 = benchmark_module("discord_retention")
    assert v2.accuracy > v1.accuracy
    assert set(v2.component_scores) == {"history_io", "memory", "buffers"}
