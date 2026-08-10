import json
from pathlib import Path

import emotion_map
import monitoring_dashboard


def test_emotion_status_counts_without_loading_map(tmp_path, monkeypatch):
    source = tmp_path / "emotion_symbol_map.json"
    source.write_text(json.dumps({
        "symbols": [
            {"symbol_word_id": "one"},
            {"symbol_word_id": "two"},
            {"symbol_word_id": "three"},
        ]
    }), encoding="utf-8")
    status_path = tmp_path / "emotion_symbol_map_status.json"
    monkeypatch.setattr(emotion_map, "_map_path", lambda child: source)
    monkeypatch.setattr(emotion_map, "_status_path", lambda child: status_path)

    refreshed = emotion_map.emotion_map_status("Ina", refresh=True)
    cached = emotion_map.emotion_map_status("Ina", refresh=False)

    assert refreshed["symbol_count"] == 3
    assert cached["symbol_count"] == 3
    assert status_path.is_file()


def test_emotion_builder_stops_at_vocabulary_cap(monkeypatch):
    monkeypatch.setattr(emotion_map, "load_config", lambda: {})
    monkeypatch.setattr(
        emotion_map, "emotion_map_status",
        lambda child, refresh=False: {"symbol_count": 10_000_000},
    )
    monkeypatch.setattr(
        emotion_map, "load_existing_symbols",
        lambda child: (_ for _ in ()).throw(AssertionError("large JSON loaded")),
    )
    monkeypatch.setattr(emotion_map, "log_to_statusbox", lambda message: None)

    assert emotion_map.build_emotion_map("Ina") is None


def test_monitor_reports_oversized_emotion_map_without_json_load(tmp_path, monkeypatch):
    source = tmp_path / "emotion_symbol_map.json"
    source.write_bytes(b"{}")
    monkeypatch.setattr(monitoring_dashboard, "MAX_JSON_BYTES", 1)
    original_safe_json = monitoring_dashboard._safe_json

    def safe_json(path: Path, default=None):
        if path == source:
            raise AssertionError("oversized map was deserialized")
        return original_safe_json(path, default)

    monkeypatch.setattr(monitoring_dashboard, "_safe_json", safe_json)
    count, label, state = monitoring_dashboard._emotion_map_summary(source)

    assert count is None
    assert "large JSON" in label
    assert state == "metadata pending"

