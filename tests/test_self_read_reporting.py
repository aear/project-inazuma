import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import self_read_reporting as srr


def _cleanup_child(child: str) -> None:
    shutil.rmtree(Path("AI_Children") / child, ignore_errors=True)



def test_report_self_read_broken_pipe_dedupes_issue_queue(monkeypatch):
    child = "TestSelfReadBrokenPipe"
    _cleanup_child(child)
    calls = []
    try:
        def fake_queue(**kwargs):
            calls.append(kwargs)
            return "github_test_entry"

        monkeypatch.setattr(srr, "_queue_broken_pipe_issue", fake_queue)

        first = srr.report_self_read_broken_pipe(
            child=child,
            component="status_pipe",
            operation="status_log_write",
            error=BrokenPipeError(32, "Broken pipe"),
            source_message="[SelfRead] PROCESSING demo.py [text]",
            path_text="/tmp/ina_status.pipe",
        )
        second = srr.report_self_read_broken_pipe(
            child=child,
            component="status_pipe",
            operation="status_log_write",
            error=BrokenPipeError(32, "Broken pipe"),
            source_message="[SelfRead] PROCESSING demo.py [text]",
            path_text="/tmp/ina_status.pipe",
        )

        assert first["issue_entry_id"] == "github_test_entry"
        assert second["issue_entry_id"] is None
        assert len(calls) == 1

        incident_path = srr.self_read_incident_log_path(child)
        entries = [json.loads(line) for line in incident_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(entries) == 2
        assert entries[0]["duplicate_within_cooldown"] is False
        assert entries[1]["duplicate_within_cooldown"] is True
    finally:
        _cleanup_child(child)



def test_explain_self_read_broken_pipe_mentions_status_pipe():
    explanation = srr.explain_self_read_broken_pipe("status_pipe", "status_log_write")
    assert "GUI status pipe" in explanation
    assert "reader had already closed it" in explanation


def test_unresolved_broken_pipe_entry_stays_deduplicated_after_cooldown(monkeypatch):
    child = "TestSelfReadPendingIssue"
    _cleanup_child(child)
    calls = []
    try:
        monkeypatch.setattr(srr, "_queue_broken_pipe_issue", lambda **kwargs: calls.append(kwargs) or "github_pending")
        first = srr.report_self_read_broken_pipe(
            child=child, component="status_pipe", operation="status_log_write", error=BrokenPipeError(32, "Broken pipe")
        )
        state_path = srr.self_read_incident_state_path(child)
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state["broken_pipe"][first["fingerprint"]]["last_reported_at"] = "2020-01-01T00:00:00+00:00"
        state_path.write_text(json.dumps(state), encoding="utf-8")

        second = srr.report_self_read_broken_pipe(
            child=child, component="status_pipe", operation="status_log_write", error=BrokenPipeError(32, "Broken pipe")
        )

        assert len(calls) == 1
        assert second["issue_entry_id"] is None
        assert second["duplicate_within_cooldown"] is True
    finally:
        _cleanup_child(child)
