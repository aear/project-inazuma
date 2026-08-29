import json

import voice_cognition_trace as trace


def test_voice_cognition_trace_benchmark_v1_transient_state_vs_v2_bounded_private_trace(tmp_path, monkeypatch):
    path = tmp_path / "voice.jsonl"
    monkeypatch.setattr(trace, "voice_cognition_trace_path", lambda child: path)

    assert trace.record_voice_cognition("Ina", "urge_evaluated", {
        "level": 0.73,
        "conversation_evidence": {"reason": "reply_invitation"},
        "oversized": "x" * 900,
    }) is True

    row = json.loads(path.read_text(encoding="utf-8"))
    assert row["version"] == 1
    assert row["event"] == "urge_evaluated"
    assert row["payload"]["level"] == 0.73
    assert len(row["payload"]["oversized"]) == 512
    assert path.stat().st_mode & 0o077 == 0


def test_voice_cognition_trace_compaction_keeps_complete_recent_rows(tmp_path, monkeypatch):
    path = tmp_path / "voice.jsonl"
    monkeypatch.setattr(trace, "voice_cognition_trace_path", lambda child: path)
    monkeypatch.setattr(trace, "MAX_TRACE_BYTES", 700)
    monkeypatch.setattr(trace, "RETAIN_TRACE_BYTES", 350)

    for index in range(20):
        assert trace.record_voice_cognition("Ina", "snapshot", {"index": index, "pad": "x" * 60})

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows
    assert rows[-1]["payload"]["index"] == 19
    assert all(row["event"] == "snapshot" for row in rows)
