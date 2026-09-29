import subprocess

import pytest

from harness_git import HarnessGit, HarnessGitError


def _git(root, *args):
    return subprocess.run(["git", *args], cwd=root, check=True, stdout=subprocess.PIPE, text=True).stdout.strip()


def _repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-b", "main")
    _git(root, "config", "user.name", "Harness Test")
    _git(root, "config", "user.email", "harness@example.test")
    (root / "tracked.txt").write_text("one\n")
    _git(root, "add", "tracked.txt")
    _git(root, "commit", "-m", "initial")
    _git(root, "remote", "add", "origin", "https://github.com/example/project.git")
    return root


def test_commit_requires_unchanged_reviewed_diff(tmp_path):
    root = _repo(tmp_path)
    (root / "tracked.txt").write_text("two\n")
    git = HarnessGit(root)
    receipt = git.prepare_commit(["tracked.txt"], "reviewed change")
    assert "-one" in receipt["diff"] and "+two" in receipt["diff"]
    result = git.commit(receipt["receipt_id"], receipt["diff_sha256"])
    assert result["committed"] is True
    assert result["pushed"] is False
    assert _git(root, "show", "HEAD:tracked.txt") == "two"


def test_commit_rejects_changed_content_and_path_escape(tmp_path):
    root = _repo(tmp_path)
    (root / "tracked.txt").write_text("two\n")
    git = HarnessGit(root)
    receipt = git.prepare_commit(["tracked.txt"], "reviewed change")
    (root / "tracked.txt").write_text("three\n")
    with pytest.raises(HarnessGitError, match="differs"):
        git.commit(receipt["receipt_id"], receipt["diff_sha256"])
    with pytest.raises(HarnessGitError, match="leaves"):
        git.prepare_commit(["../outside"], "no")


def test_push_requires_exact_confirmation_head_and_github_remote(tmp_path, monkeypatch):
    root = _repo(tmp_path)
    git = HarnessGit(root)
    head = _git(root, "rev-parse", "HEAD")
    with pytest.raises(HarnessGitError, match="confirmation"):
        git.push(remote="origin", expected_head=head, confirmation="yes")
    _git(root, "remote", "set-url", "origin", "https://example.test/owner/repo.git")
    with pytest.raises(HarnessGitError, match="github.com"):
        git.push(remote="origin", expected_head=head, confirmation="PUSH REVIEWED COMMIT")


def test_status_never_authorizes_automatic_commit_or_push(tmp_path):
    status = HarnessGit(_repo(tmp_path)).status()
    assert status["automatic_commit"] is False
    assert status["automatic_push"] is False
    assert status["remotes"][0]["url"] == "https://github.com/example/project.git"
