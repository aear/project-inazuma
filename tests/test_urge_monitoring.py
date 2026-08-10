import monitoring_dashboard


def test_urges_keep_typing_explanations_separate(tmp_path, monkeypatch):
    memory = tmp_path / "memory"
    memory.mkdir()
    (memory / "inastate.json").write_text(
        """{
          "urge_to_type": {
            "level": 0.7,
            "adjusted_level": 0.4,
            "timestamp": "2026-01-02T00:00:00+00:00",
            "drivers": {"clarity": 0.2, "fuzziness": 0.8},
            "arbitration": {"allowed": false}
          },
          "emotion_snapshot": {"values": {"interest": 0.9}},
          "text_expression_intent": {
            "strategy": "silence",
            "created_at": "2026-01-02T00:00:00+00:00"
          },
          "meta_arbitration": {"status": "conflict"}
        }""",
        encoding="utf-8",
    )
    monkeypatch.setattr(monitoring_dashboard, "_child_memory", lambda: memory)
    monkeypatch.setattr(
        monitoring_dashboard,
        "load_config",
        lambda: {"min_urge_to_type": 0.35, "min_urge_to_speak": 0.25},
    )

    cards, rows = monitoring_dashboard._urges()

    assert cards[0] == ("Type", "40%")
    assert cards[3] == ("Arbitration", "conflict")
    type_urge = next(row for row in rows if row[0] == "Urge to type")
    assert type_urge[1] == "70% base → 40% adjusted"
    assert type_urge[2] == "present · held by arbitration"
    assert next(row for row in rows if row[0] == "Typing · content to express")[1] == "not observed"
    assert next(row for row in rows if row[0] == "Typing · expression access")[1] == "possible difficulty signal"
    assert next(row for row in rows if row[0] == "Typing · uncertainty")[1] == "80%"
    assert next(row for row in rows if row[0] == "Typing · interest")[1] == "raised signal · +0.90"
    assert next(row for row in rows if row[0] == "Typing · response choice")[1] == "explicit silence"


def test_urges_do_not_infer_a_reason_from_low_typing_urge(tmp_path, monkeypatch):
    memory = tmp_path / "memory"
    memory.mkdir()
    (memory / "inastate.json").write_text(
        '{"urge_to_type": {"level": 0.1, "drivers": {}}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(monitoring_dashboard, "_child_memory", lambda: memory)
    monkeypatch.setattr(monitoring_dashboard, "load_config", lambda: {})

    _cards, rows = monitoring_dashboard._urges()

    assert next(row for row in rows if row[0] == "Urge to type")[2] == "below action threshold"
    assert next(row for row in rows if row[0] == "Typing · content to express")[1] == "not observed"
    choice = next(row for row in rows if row[0] == "Typing · response choice")
    assert choice[1] == "no explicit choice reported"
    assert "must not be used to infer" in choice[4]
