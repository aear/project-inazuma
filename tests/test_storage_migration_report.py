from datetime import datetime, timedelta, timezone

import storage_migration_report as smr


def _report(*, legacy_files=0, failed=0, fast_score=3.0):
    return {
        "date": "2026-09-17", "adaptive_state_updated_at": "now",
        "decisions": {"index": {"tier": "fast", "fast_score": fast_score, "durable_score": 2.0}},
        "devices": {"fast": {"failures": 0, "free_ratio": 0.8}},
        "directories": {"fragment_root": {"files": legacy_files, "sample_truncated": False}},
        "memory_tiers": {},
        "recent_migrations": ([{"status": "error", "failed": failed}] if failed else []),
        "recommendations": [], "safety": "recommendation_only",
    }


def test_report_signal_ignores_score_drift_but_keeps_actionable_changes():
    first = smr._report_signal(_report(fast_score=3.0))
    drifted = smr._report_signal(_report(fast_score=9.0))
    attention = smr._report_signal(_report(legacy_files=2))
    assert first == drifted
    assert first != attention
    assert not smr._signal_is_actionable(first)
    assert smr._signal_is_actionable(attention)


def test_unchanged_daily_state_is_recorded_without_queueing(tmp_path, monkeypatch):
    child = "Ina"
    state_path = tmp_path / "report_state.json"
    report = _report(legacy_files=2)
    signal = smr._report_signal(report)
    state_path.write_text(
        '{"last_report_at":"2026-09-15T00:00:00+00:00","report_signal":' +
        __import__("json").dumps(signal) + "}", encoding="utf-8",
    )
    policy = {
        **smr.DEFAULT_POLICY, "enabled": True, "interval_hours": 1,
        "state_path": str(state_path), "preference_path": str(tmp_path / "preferences.json"),
    }
    cfg = {"storage_migration_reporting": policy}
    calls = []
    monkeypatch.setattr(smr, "build_daily_migration_report", lambda *_args, **_kwargs: report)
    monkeypatch.setattr(smr, "report_github_finding", lambda *_args, **_kwargs: calls.append(1) or {"queued": True})

    result = smr.maybe_queue_daily_migration_report(
        child, cfg, now=datetime(2026, 9, 17, tzinfo=timezone.utc),
    )
    assert result["queued"] is False
    assert result["reason"] == "unchanged"
    assert calls == []
