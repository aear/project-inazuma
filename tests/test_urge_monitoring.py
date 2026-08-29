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
    monkeypatch.setattr(monitoring_dashboard.time, "time", lambda: 1767312030.0)
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


def test_urge_monitor_benchmark_v1_stale_percentage_vs_v2_age_qualified_signal(tmp_path, monkeypatch):
    """V2 preserves old evidence without presenting it as current motivation."""
    memory = tmp_path / "memory"
    memory.mkdir()
    (memory / "inastate.json").write_text(
        '{"urge_to_voice":{"level":0.919,"adjusted_level":1.0,'
        '"timestamp":"2026-01-02T00:00:00+00:00"}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(monitoring_dashboard, "_child_memory", lambda: memory)
    monkeypatch.setattr(monitoring_dashboard, "load_config", lambda: {"urge_signal_stale_seconds": 300})
    monkeypatch.setattr(monitoring_dashboard.time, "time", lambda: 1767312601.0)

    cards, rows = monitoring_dashboard._urges()

    assert ("Voice", "stale (100%)") in cards
    voice = next(row for row in rows if row[0] == "Urge to voice")
    assert voice[1] == "92% base → 100% adjusted · stale"
    assert voice[2] == "stale · not a current action signal"
    assert '"current_action_signal": false' in voice[4]


def test_speaking_monitor_benchmark_v1_latched_boolean_vs_v2_fresh_gateway_signal(tmp_path, monkeypatch):
    memory = tmp_path / "memory"
    memory.mkdir()
    (memory / "runtime_services.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(monitoring_dashboard, "_child_memory", lambda: memory)
    monkeypatch.setattr(monitoring_dashboard.time, "time", lambda: 1767312601.0)
    states = {
        "currently_speaking": False,
        "discord_voice_speaking": {
            "active": True,
            "status": "signalled",
            "timestamp": "2026-01-02T00:00:00+00:00",
        },
    }
    monkeypatch.setattr(monitoring_dashboard, "get_inastate", lambda key: states.get(key))

    cards, rows = monitoring_dashboard._communication()

    assert ("Speaking", "stale (last yes)") in cards
    assert next(row for row in rows if row[0] == "Speaking now")[1] == "stale (last yes)"
    assert next(row for row in rows if row[0] == "Discord speaking indicator")[1] == "signalled"
