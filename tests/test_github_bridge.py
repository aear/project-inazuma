import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import github_bridge as gb
import github_submission as gs


def _cleanup_child(child: str) -> None:
    shutil.rmtree(Path("AI_Children") / child, ignore_errors=True)


def test_process_once_skips_without_token_without_failed_history(monkeypatch):
    child = "TestGitHubBridgeNoToken"
    _cleanup_child(child)
    try:
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        cfg = {
            "current_child": child,
            "github_submission": {
                "enabled": True,
                "delivery_mode": "issues",
                "repo_full_name": "owner/repo",
                "max_batch": 10,
            },
        }
        entry_id = gs.append_github_issue_entry(child, "No token", "Keep this queued.")
        submit_calls = []

        monkeypatch.setattr(gb, "load_config", lambda: cfg)
        monkeypatch.setattr(gb, "submit_issue", lambda *args, **kwargs: submit_calls.append(args) or {})

        assert entry_id
        assert gb.process_once() == 0
        assert submit_calls == []
        assert not gs.github_outbox_history_path(child).exists()
        assert [entry["id"] for entry in gs.read_pending_entries(child, cfg=cfg)] == [entry_id]
    finally:
        _cleanup_child(child)
