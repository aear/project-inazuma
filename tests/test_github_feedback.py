import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import github_feedback as gf
import github_submission as gs


def _cleanup_child(child: str) -> None:
    shutil.rmtree(Path("AI_Children") / child, ignore_errors=True)


def test_get_github_feedback_config_parses_overrides():
    policy = gf.get_github_feedback_config({
        "github_feedback": {
            "enabled": True,
            "poll_interval_sec": 120,
            "max_issues_per_check": 3,
            "max_comments_per_issue": 7,
            "ignore_authors": ["bot", "bot", ""],
        }
    })
    assert policy["enabled"] is True
    assert policy["poll_interval_sec"] == 120
    assert policy["max_issues_per_check"] == 3
    assert policy["max_comments_per_issue"] == 7
    assert policy["ignore_authors"] == ["bot"]


def test_resolve_github_token_reads_config_without_env(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    token = gs.resolve_github_token({"github_submission": {"token": "cfg-token"}})
    assert token == "cfg-token"


def test_sync_issue_feedback_logs_new_comments(monkeypatch):
    child = "TestGitHubFeedback"
    _cleanup_child(child)
    try:
        gs.log_history(
            child,
            "github_entry",
            "submitted",
            issue_number=7,
            issue_url="https://github.com/owner/repo/issues/7",
            title="[Ina] Test",
        )

        monkeypatch.setattr(gf, "resolve_github_token", lambda cfg, policy: "token")
        monkeypatch.setattr(
            gf,
            "fetch_issue_comments",
            lambda **kwargs: [
                {
                    "id": 123,
                    "html_url": "https://github.com/owner/repo/issues/7#issuecomment-123",
                    "user": {"login": "reviewer"},
                    "author_association": "OWNER",
                    "created_at": "2026-04-07T08:00:00Z",
                    "updated_at": "2026-04-07T08:00:00Z",
                    "body": "Looks right.",
                }
            ],
        )

        result = gf.sync_issue_feedback({
            "current_child": child,
            "github_submission": {
                "enabled": True,
                "delivery_mode": "issues",
                "repo_full_name": "owner/repo",
            },
            "github_feedback": {"enabled": True},
        })

        assert result["checked"] is True
        assert result["new_comments"] == 1
        feedback_log = gf.github_issue_feedback_log_path(child)
        records = [json.loads(line) for line in feedback_log.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert records[0]["comment_id"] == "123"
        assert records[0]["author"] == "reviewer"

        second = gf.sync_issue_feedback({
            "current_child": child,
            "github_submission": {
                "enabled": True,
                "delivery_mode": "issues",
                "repo_full_name": "owner/repo",
            },
            "github_feedback": {"enabled": True},
        })
        assert second["new_comments"] == 0
    finally:
        _cleanup_child(child)
