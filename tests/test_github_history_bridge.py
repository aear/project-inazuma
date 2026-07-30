import subprocess

import github_history_bridge as ghb


def test_read_commit_history_parses_metadata_and_numstat(monkeypatch, tmp_path):
    log_output = (
        "\x1eabc123\x1fabc1234\x1f2026-07-30T10:00:00+01:00"
        "\x1fIna\x1fina@example.test\x1fparent1\x1fAdd memory lanes"
        "\x1fExplain why.\n7\t2\tnot-a-stat\nMore context.\x1d\n4\t1\tmodel_manager.py\n-\t-\timage.png"
    )

    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, log_output, "")

    monkeypatch.setattr(ghb.subprocess, "run", fake_run)
    commits = ghb.read_commit_history(tmp_path, limit=500)

    assert len(commits) == 1
    assert commits[0]["subject"] == "Add memory lanes"
    assert commits[0]["body"] == "Explain why.\n7\t2\tnot-a-stat\nMore context."
    assert commits[0]["file_count"] == 2
    assert commits[0]["insertions"] == 4
    assert commits[0]["deletions"] == 1
    assert commits[0]["files"][1]["insertions"] is None


def test_commit_as_text_describes_evolution_without_diff():
    text = ghb.commit_as_text(
        {
            "short_hash": "abc1234",
            "subject": "Refine curiosity",
            "authored_at": "2026-07-30T10:00:00+01:00",
            "author": "Ina",
            "parents": ["parent"],
            "file_count": 1,
            "insertions": 8,
            "deletions": 3,
            "files": [{"path": "eq_engine.py"}],
            "body": "",
        }
    )

    assert "Refine curiosity" in text
    assert "eq_engine.py" in text
    assert "8 insertions and 3 deletions" in text
