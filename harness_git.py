"""Review-receipted Git operations for the standalone Codex harness."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
import secrets
import shutil
import subprocess
import tempfile
import time
from typing import Any, Iterable
from urllib.parse import urlsplit

from external_access import ExternalAccessBlocked, resolve_public_addresses


MAX_GIT_OUTPUT = 256 * 1024
RECEIPT_TTL_SECONDS = 30 * 60


class HarnessGitError(RuntimeError):
    pass


@dataclass(frozen=True)
class CommitReceipt:
    receipt_id: str
    head: str
    branch: str
    paths: tuple[str, ...]
    message: str
    diff_sha256: str
    created_monotonic: float


class HarnessGit:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).resolve()
        self._receipts: dict[str, CommitReceipt] = {}


    def _run(self, args: list[str], *, env: dict[str, str] | None = None, timeout: float = 30) -> str:
        command = ["git", *args]
        process_env = dict(os.environ)
        process_env.update({"GIT_TERMINAL_PROMPT": "0", "GIT_CONFIG_NOSYSTEM": "1"})
        if env:
            process_env.update(env)
        result = subprocess.run(
            command, cwd=self.root, env=process_env, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, check=False,
        )
        output = (result.stdout + result.stderr)[:MAX_GIT_OUTPUT].decode("utf-8", "replace")
        if result.returncode:
            raise HarnessGitError(output.strip() or f"git exited {result.returncode}")
        return output


    def _head(self) -> str:
        return self._run(["rev-parse", "HEAD"]).strip()


    def _branch(self) -> str:
        branch = self._run(["branch", "--show-current"]).strip()
        if not branch or not re.fullmatch(r"[A-Za-z0-9._/-]{1,200}", branch):
            raise HarnessGitError("a normal bounded branch is required")
        return branch


    def _paths(self, paths: Iterable[Any]) -> tuple[str, ...]:
        selected: list[str] = []
        for raw in paths:
            text = str(raw or "").strip().replace("\\", "/")
            candidate = (self.root / text).resolve()
            try:
                relative = candidate.relative_to(self.root).as_posix()
            except ValueError as exc:
                raise HarnessGitError("commit path leaves the workspace") from exc
            if not relative or relative == ".git" or relative.startswith(".git/"):
                raise HarnessGitError("Git metadata cannot be selected as content")
            if relative not in selected:
                selected.append(relative)
            if len(selected) > 100:
                raise HarnessGitError("at most 100 paths may be committed together")
        if not selected:
            raise HarnessGitError("at least one explicit path is required")
        return tuple(selected)


    def _github_remote(self, name: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9._-]{1,100}", str(name or "")):
            raise HarnessGitError("invalid remote name")
        url = self._run(["remote", "get-url", name]).strip()
        parsed = urlsplit(url)
        if parsed.scheme != "https" or parsed.hostname != "github.com" or parsed.port not in (None, 443) or parsed.username or parsed.password:
            raise HarnessGitError("harness sync permits only HTTPS github.com remotes without inline credentials")
        path = parsed.path.lstrip("/")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:\.git)?", path):
            raise HarnessGitError("remote must identify one GitHub owner/repository")
        return url


    def _preflight_github(self, remote: str) -> str:
        url = self._github_remote(remote)
        try:
            resolve_public_addresses("github.com")
        except ExternalAccessBlocked as exc:
            raise HarnessGitError(f"GitHub DNS preflight failed: {exc}") from exc
        return url


    def status(self) -> dict[str, Any]:
        branch = self._branch()
        porcelain = self._run(["status", "--porcelain=v1", "--untracked-files=normal"])
        remotes = []
        for name in self._run(["remote"]).splitlines()[:20]:
            try:
                remotes.append({"name": name, "url": self._github_remote(name)})
            except HarnessGitError:
                remotes.append({"name": name, "url": "blocked_non_github_remote"})
        ahead = behind = None
        try:
            counts = self._run(["rev-list", "--left-right", "--count", "HEAD...@{upstream}"]).split()
            ahead, behind = int(counts[0]), int(counts[1])
        except (HarnessGitError, ValueError, IndexError):
            pass
        return {
            "branch": branch, "head": self._head(), "dirty": bool(porcelain),
            "changes": porcelain.splitlines()[:200], "ahead": ahead, "behind": behind,
            "remotes": remotes, "automatic_commit": False, "automatic_push": False,
        }


    def fetch(self, remote: str) -> dict[str, Any]:
        url = self._preflight_github(remote)
        output = self._run(["-c", "http.followRedirects=false", "fetch", "--no-tags", "--", remote], timeout=120)
        return {"fetched": True, "remote": remote, "url": url, "output": output[-12000:], "status": self.status()}


    def prepare_commit(self, paths: Iterable[Any], message: str) -> dict[str, Any]:
        selected = self._paths(paths)
        commit_message = str(message or "").strip()
        if not 1 <= len(commit_message) <= 4000 or "\x00" in commit_message:
            raise HarnessGitError("commit message must be 1..4000 characters")
        git_dir = Path(self._run(["rev-parse", "--git-dir"]).strip())
        git_dir = git_dir if git_dir.is_absolute() else self.root / git_dir
        with tempfile.TemporaryDirectory(prefix="ina_harness_git_") as temporary:
            temporary_index = Path(temporary) / "index"
            live_index = git_dir / "index"
            if live_index.exists():
                shutil.copy2(live_index, temporary_index)
            env = {"GIT_INDEX_FILE": str(temporary_index)}
            self._run(["add", "--", *selected], env=env)
            diff = self._run(["diff", "--cached", "--binary", "--", *selected], env=env)
        if not diff:
            raise HarnessGitError("selected paths produce no commit diff")
        digest = hashlib.sha256(diff.encode("utf-8")).hexdigest()
        receipt = CommitReceipt(
            secrets.token_urlsafe(18), self._head(), self._branch(), selected,
            commit_message, digest, time.monotonic(),
        )
        self._receipts[receipt.receipt_id] = receipt
        return {
            "receipt_id": receipt.receipt_id, "head": receipt.head, "branch": receipt.branch,
            "paths": list(selected), "message": commit_message, "diff_sha256": digest,
            "diff": diff[:MAX_GIT_OUTPUT], "expires_in_seconds": RECEIPT_TTL_SECONDS,
        }


    def commit(self, receipt_id: str, expected_diff_sha256: str) -> dict[str, Any]:
        receipt = self._receipts.pop(str(receipt_id or ""), None)
        if receipt is None or time.monotonic() - receipt.created_monotonic > RECEIPT_TTL_SECONDS:
            raise HarnessGitError("commit review receipt is missing or expired")
        if self._head() != receipt.head or self._branch() != receipt.branch:
            raise HarnessGitError("HEAD or branch changed after review")
        if not secrets.compare_digest(receipt.diff_sha256, str(expected_diff_sha256 or "")):
            raise HarnessGitError("reviewed diff hash confirmation does not match")
        self._run(["add", "--", *receipt.paths])
        staged = self._run(["diff", "--cached", "--binary", "--", *receipt.paths])
        if hashlib.sha256(staged.encode("utf-8")).hexdigest() != receipt.diff_sha256:
            raise HarnessGitError("staged content differs from the reviewed diff")
        self._run(["commit", "--no-verify", "--no-gpg-sign", "-m", receipt.message, "--", *receipt.paths], timeout=60)
        return {"committed": True, "head": self._head(), "branch": receipt.branch, "pushed": False}


    def push(self, *, remote: str, expected_head: str, confirmation: str) -> dict[str, Any]:
        if confirmation != "PUSH REVIEWED COMMIT":
            raise HarnessGitError("explicit push confirmation phrase is required")
        self._preflight_github(remote)
        head, branch = self._head(), self._branch()
        if not re.fullmatch(r"[0-9a-f]{40,64}", str(expected_head or "")) or head != expected_head:
            raise HarnessGitError("expected commit is not the current HEAD")
        output = self._run(["-c", "http.followRedirects=false", "push", "--porcelain", "--", remote, f"HEAD:refs/heads/{branch}"], timeout=120)
        return {"pushed": True, "remote": remote, "head": head, "branch": branch, "output": output[-12000:]}


__all__ = ["HarnessGit", "HarnessGitError"]
