import json

import raw_file_manager as rfm
from github_history_materializer import materialize_commit_history


def test_preferences_migrate_github_history_choice(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "AI_Children" / "Ina" / "memory" / "self_read_preferences.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "source_choices": {
                    "code": True,
                    "music": True,
                    "books": True,
                    "venv": False,
                },
                "skip_files": ["swapfile"],
            }
        ),
        encoding="utf-8",
    )

    prefs = rfm.load_self_read_preferences("Ina")

    assert prefs["source_choices"]["github_history"] is True
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored["source_choices"]["github_history"] is True


def test_materializer_writes_stable_summary(monkeypatch, tmp_path):
    commit = {
        "hash": "a" * 40,
        "short_hash": "aaaaaaa",
        "subject": "Shape memory",
        "authored_at": "2026-07-30T10:00:00+01:00",
        "author": "Ina",
        "parents": [],
        "file_count": 1,
        "insertions": 2,
        "deletions": 0,
        "files": [{"path": "memory.py"}],
        "body": "",
    }
    monkeypatch.setattr(
        "github_history_materializer.read_commit_history", lambda root, limit: [commit]
    )

    paths = materialize_commit_history(tmp_path, tmp_path / "history")

    assert paths == [tmp_path / "history" / f"{commit['hash']}.txt"]
    assert "Shape memory" in paths[0].read_text(encoding="utf-8")
