import json

import monitoring_dashboard


def test_english_mapping_monitor_benchmark_v1_oversize_zero_vs_v2_compact_status(tmp_path, monkeypatch):
    memory = tmp_path / "memory"
    memory.mkdir()
    links = memory / "text_vocab_links.json"
    links.write_bytes(b"x" * 2048)
    stat = links.stat()
    (memory / "text_vocab_links_status.json").write_text(json.dumps({
        "linked_word_count": 24974,
        "link_count": 99895,
        "evaluated_count": 9502,
        "remaining": 15498,
        "queue_by_source": {"discord": 68},
        "last_batch": {"mode": "new", "new_mappings": 500, "revisited_mappings": 0},
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
    }), encoding="utf-8")
    (memory / "text_vocab.json").write_text('{"vocab":{}}', encoding="utf-8")
    monkeypatch.setattr(monitoring_dashboard, "_child_memory", lambda: memory)
    monkeypatch.setattr(monitoring_dashboard, "load_config", lambda: {})
    monkeypatch.setattr(monitoring_dashboard, "MAX_JSON_BYTES", 1024)

    _cards, rows = monitoring_dashboard._mind()

    mapping = next(row for row in rows if row[0] == "English mappings")
    assert mapping[1] == "24,974 mapped · 99,895 ranked links"
    review = next(row for row in rows if row[0] == "Current-revision review")
    assert review[1] == "9,502 reviewed · 15,498 awaiting review"
    assert next(row for row in rows if row[0] == "Average links per word")[1] == "0.00"
