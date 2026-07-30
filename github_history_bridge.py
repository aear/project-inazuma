"""Read-only bridge from a GitHub-backed checkout's commit history to Ina.

The bridge intentionally uses only local Git metadata. It never fetches,
checks out, writes refs, or sends anything to GitHub.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional


FIELD_SEPARATOR = "\x1f"
RECORD_SEPARATOR = "\x1e"
METADATA_SEPARATOR = "\x1d"


def _run_git(repo_root: Path, args: List[str], timeout: float = 10.0) -> str:
    command = ["git", "-C", str(repo_root), *args]
    result = subprocess.run(
        command, check=False, capture_output=True, text=True, timeout=timeout
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(detail or f"git exited with status {result.returncode}")
    return result.stdout


def repository_context(repo_root: Path) -> Dict[str, Optional[str]]:
    """Return stable local repository context without contacting its remote."""
    root = Path(_run_git(repo_root, ["rev-parse", "--show-toplevel"]).strip())
    remote = ""
    for remote_name in ("project-inazuma", "origin"):
        try:
            remote = _run_git(
                root, ["config", "--get", f"remote.{remote_name}.url"]
            ).strip()
        except RuntimeError:
            continue
        if remote:
            break
    return {"root": str(root), "remote": remote or None}


def read_commit_history(repo_root: Path, limit: int = 24) -> List[Dict[str, Any]]:
    """Read bounded first-parent history summaries from the local checkout."""
    bounded_limit = max(1, min(int(limit), 1000))
    pretty = "%x1e%H%x1f%h%x1f%aI%x1f%an%x1f%ae%x1f%P%x1f%s%x1f%b%x1d"
    output = _run_git(
        repo_root,
        [
            "log", "--first-parent", f"--max-count={bounded_limit}",
            f"--pretty=format:{pretty}", "--numstat", "--no-renames",
        ],
    )

    commits: List[Dict[str, Any]] = []
    for raw_record in output.split(RECORD_SEPARATOR):
        record = raw_record.strip()
        if not record:
            continue
        metadata, separator, stats_text = record.partition(METADATA_SEPARATOR)
        if not separator:
            continue
        fields = metadata.split(FIELD_SEPARATOR, 7)
        if len(fields) != 8:
            continue
        commit_hash, short_hash, authored_at, author, email, parents, subject, body = fields
        files: List[Dict[str, Any]] = []
        insertions = 0
        deletions = 0
        for line in stats_text.splitlines():
            parts = line.split("\t", 2)
            if len(parts) != 3:
                continue
            added_text, deleted_text, path = parts
            added = int(added_text) if added_text.isdigit() else None
            deleted = int(deleted_text) if deleted_text.isdigit() else None
            if added is not None:
                insertions += added
            if deleted is not None:
                deletions += deleted
            files.append({"path": path, "insertions": added, "deletions": deleted})
        commits.append(
            {
                "hash": commit_hash, "short_hash": short_hash,
                "authored_at": authored_at, "author": author,
                "author_email": email,
                "parents": [value for value in parents.split() if value],
                "subject": subject.strip(), "body": body.strip(), "files": files,
                "file_count": len(files), "insertions": insertions,
                "deletions": deletions,
            }
        )
    return commits


def commit_as_text(commit: Dict[str, Any]) -> str:
    """Render evolution-focused prose, without including source diffs."""
    files = commit.get("files") or []
    paths = [str(item.get("path")) for item in files[:12] if item.get("path")]
    merge_note = "merge commit" if len(commit.get("parents") or []) > 1 else "commit"
    lines = [
        f"Project evolution {merge_note} {commit.get('short_hash')}: {commit.get('subject')}",
        f"Authored {commit.get('authored_at')} by {commit.get('author')}.",
        f"Changed {commit.get('file_count', 0)} files with {commit.get('insertions', 0)} insertions and {commit.get('deletions', 0)} deletions.",
    ]
    if paths:
        lines.append("Areas touched: " + ", ".join(paths) + ".")
    body = str(commit.get("body") or "").strip()[:1000]
    if body:
        lines.append("Commit notes: " + " ".join(body.split()))
    return "\n".join(lines)
