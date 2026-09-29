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
import threading
import copy
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
    tree: str


class HarnessGit:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).resolve()
        self._receipts: dict[str, CommitReceipt] = {}
        self._status_cache = None
        self._status_lock = threading.Lock()


    def _run(self, args: list[str], *, env: dict[str, str] | None = None, timeout: float = 30) -> str:
        command = ["git", "--literal-pathspecs", "-c", "core.hooksPath=/dev/null", "-c", "core.fsmonitor=false", *args]
        process_env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
        process_env.update({"GIT_TERMINAL_PROMPT": "0", "GIT_CONFIG_NOSYSTEM": "1"})
        if env:
            process_env.update(env)
        result = subprocess.run(
            command, cwd=self.root, env=process_env, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, check=False,
        )
        if result.returncode:
            raise HarnessGitError(result.stderr[:4000].decode("utf-8", "replace").strip() or f"git exited {result.returncode}")
        if len(result.stdout) > MAX_GIT_OUTPUT:
            raise HarnessGitError("Git output exceeds 256 KiB; narrow the selected review")
        # Never hash a lossy or truncated representation of the review.
        try:
            return result.stdout.decode("utf-8", "strict")
        except UnicodeDecodeError as exc:
            raise HarnessGitError("review output is not valid UTF-8") from exc


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
            if relative in {"", ".", ".git"} or relative.startswith(".git/"):
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
        # get-url expands insteadOf/pushInsteadOf. Validate every effective
        # fetch and push destination; never pass the remote name to transport.
        fetch_urls = self._run(["remote", "get-url", "--all", name]).splitlines()
        push_urls = self._run(["remote", "get-url", "--push", "--all", name]).splitlines()
        if len(fetch_urls) != 1 or push_urls != fetch_urls:
            raise HarnessGitError("fetch and push destinations must be the same single GitHub URL")
        url = fetch_urls[0]
        parsed = urlsplit(url)
        if parsed.scheme != "https" or parsed.hostname != "github.com" or parsed.port not in (None, 443) or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise HarnessGitError("harness sync permits only HTTPS github.com remotes without inline credentials")
        path = parsed.path.lstrip("/")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:\.git)?", path):
            raise HarnessGitError("remote must identify one GitHub owner/repository")
        return url


    def _preflight_github(self, remote: str) -> str:
        config = self._run(["config", "--list", "--name-only"])
        if any(key.lower().startswith("url.") and key.lower().endswith((".insteadof", ".pushinsteadof")) for key in config.splitlines()):
            raise HarnessGitError("URL rewrite rules are not permitted for harness transport")
        url = self._github_remote(remote)
        try:
            resolve_public_addresses("github.com")
        except ExternalAccessBlocked as exc:
            raise HarnessGitError(f"GitHub DNS preflight failed: {exc}") from exc
        return url


    def status(self, *, refresh: bool = False) -> dict[str, Any]:
        # No timer-triggered scans: explicit refresh and successful mutations
        # invalidate this snapshot. Polling only reads it.
        with self._status_lock:
            if refresh or self._status_cache is None:
                self._status_cache = self._scan_status()
            return copy.deepcopy(self._status_cache)

    def invalidate_status(self) -> None:
        with self._status_lock:
            self._status_cache = None

    def _scan_status(self) -> dict[str, Any]:
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
        output = self._run(["-c", "http.followRedirects=false", "-c", "http.sslVerify=true", "fetch", "--no-tags", "--", url], timeout=120)
        self.invalidate_status()
        return {"fetched": True, "remote": remote, "url": url, "output": output[-12000:], "status": self.status()}


    def prepare_commit(self, paths: Iterable[Any], message: str) -> dict[str, Any]:
        selected = self._paths(paths)
        commit_message = str(message or "").strip()
        if not 1 <= len(commit_message) <= 4000 or "\x00" in commit_message:
            raise HarnessGitError("commit message must be 1..4000 characters")
        head, branch = self._head(), self._branch()
        diff, tree = self._review_snapshot(selected, head)
        if not diff:
            raise HarnessGitError("selected paths produce no commit diff")
        digest = hashlib.sha256(diff.encode("utf-8")).hexdigest()
        receipt = CommitReceipt(
            secrets.token_urlsafe(18), head, branch, selected,
            commit_message, digest, time.monotonic(), tree,
        )
        # Keep review state finite and expire receipts before admitting more.
        self._receipts = {key: value for key, value in self._receipts.items()
                          if time.monotonic() - value.created_monotonic <= RECEIPT_TTL_SECONDS}
        if len(self._receipts) >= 32:
            raise HarnessGitError("too many pending commit reviews")
        self._receipts[receipt.receipt_id] = receipt
        return {
            "receipt_id": receipt.receipt_id, "head": receipt.head, "branch": receipt.branch,
            "paths": list(selected), "message": commit_message, "diff_sha256": digest,
            "diff": diff, "expires_in_seconds": RECEIPT_TTL_SECONDS,
        }


    def _review_snapshot(self, selected: tuple[str, ...], head: str) -> tuple[str, str]:
        config = self._run(["config", "--list", "--name-only"])
        if any(key.lower().startswith("filter.") and key.lower().endswith((".clean", ".process")) for key in config.splitlines()):
            raise HarnessGitError("executable Git filters require a separate reviewed workflow")
        with tempfile.TemporaryDirectory(prefix="ina_harness_git_") as temporary:
            temporary_index = Path(temporary) / "index"
            env = {"GIT_INDEX_FILE": str(temporary_index)}
            self._run(["read-tree", head], env=env)
            self._run(["add", "--", *selected], env=env)
            diff = self._run(["diff", "--no-ext-diff", "--no-textconv", "--cached", "--binary", head, "--", *selected], env=env)
            tree = self._run(["write-tree"], env=env).strip()
        return diff, tree


    def commit(self, receipt_id: str, expected_diff_sha256: str) -> dict[str, Any]:
        receipt = self._receipts.pop(str(receipt_id or ""), None)
        if receipt is None or time.monotonic() - receipt.created_monotonic > RECEIPT_TTL_SECONDS:
            raise HarnessGitError("commit review receipt is missing or expired")
        if self._head() != receipt.head or self._branch() != receipt.branch:
            raise HarnessGitError("HEAD or branch changed after review")
        if not secrets.compare_digest(receipt.diff_sha256, str(expected_diff_sha256 or "")):
            raise HarnessGitError("reviewed diff hash confirmation does not match")
        staged, tree = self._review_snapshot(receipt.paths, receipt.head)
        if tree != receipt.tree or hashlib.sha256(staged.encode("utf-8")).hexdigest() != receipt.diff_sha256:
            raise HarnessGitError("staged content differs from the reviewed diff")
        # Commit immutable reviewed objects, never reread the worktree after
        # verification. CAS prevents moving a branch changed concurrently.
        head = self._run(["-c", "commit.gpgSign=false", "commit-tree", tree, "-p", receipt.head, "-m", receipt.message]).strip()
        self._run(["update-ref", "-m", "harness reviewed commit", f"refs/heads/{receipt.branch}", head, receipt.head])
        self._run(["reset", "--quiet", head, "--", *receipt.paths])
        self.invalidate_status()
        return {"committed": True, "head": head, "branch": receipt.branch, "pushed": False}


    def push(self, *, remote: str, expected_head: str, confirmation: str) -> dict[str, Any]:
        if confirmation != "PUSH REVIEWED COMMIT":
            raise HarnessGitError("explicit push confirmation phrase is required")
        url = self._preflight_github(remote)
        head, branch = self._head(), self._branch()
        if not re.fullmatch(r"[0-9a-f]{40,64}", str(expected_head or "")) or head != expected_head:
            raise HarnessGitError("expected commit is not the current HEAD")
        output = self._run(["-c", "http.followRedirects=false", "-c", "http.sslVerify=true", "push", "--no-verify", "--porcelain", "--", url, f"{head}:refs/heads/{branch}"], timeout=120)
        self.invalidate_status()
        return {"pushed": True, "remote": remote, "head": head, "branch": branch, "output": output[-12000:]}


__all__ = ["HarnessGit", "HarnessGitError"]
