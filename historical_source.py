"""Load a repository file from Git history without retaining a duplicate implementation."""
from __future__ import annotations

import hashlib
import subprocess
import sys
import types
from pathlib import Path
from functools import lru_cache
from typing import Optional

_REPOSITORY = Path(__file__).resolve().parent


@lru_cache(maxsize=16)
def resolve_revision(revision: str = "HEAD") -> str:
    """Resolve a Git revision to the immutable commit identifier used by a run."""
    result = subprocess.run(
        ["git", "rev-parse", "--verify", str(revision)], cwd=_REPOSITORY,
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    return result.stdout.decode("ascii").strip()


def working_tree_identity() -> dict[str, str]:
    """Identify bounded tracked Python changes and untracked Python source.

    No source content is exported. This identifies code, not private learned
    state, runtime configuration, or a reconstructed dependency environment.
    """
    try:
        head = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"], cwd=_REPOSITORY,
            check=True, capture_output=True, timeout=10,
        ).stdout.decode("ascii").strip()
        diff = subprocess.run(
            ["git", "diff", "--no-ext-diff", "--no-textconv", "--binary", "HEAD", "--", "*.py"],
            cwd=_REPOSITORY, check=True, capture_output=True, timeout=10,
        ).stdout
        extra = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", "-z", "--", "*.py"],
            cwd=_REPOSITORY, check=True, capture_output=True, timeout=10,
        ).stdout.split(b"\0")
        if len(diff) > 16 * 1024 * 1024 or len(extra) > 1025:
            return {"status": "unavailable", "reason": "source_identity_budget"}
        digest = hashlib.sha256(diff)
        total = len(diff)
        for raw in sorted(filter(None, extra)):
            path = _REPOSITORY / raw.decode("utf-8")
            if path.is_symlink():
                return {"status": "unavailable", "reason": "untracked_source_symlink"}
            size = path.stat().st_size
            total += size
            if total > 16 * 1024 * 1024:
                return {"status": "unavailable", "reason": "source_identity_budget"}
            digest.update(raw + b"\0" + path.read_bytes() + b"\0")
        return {"status": "recorded", "head": head, "changes_sha256": digest.hexdigest(),
                "scope": "tracked_python_diff_and_untracked_python; excludes learned state and configuration"}
    except (OSError, ValueError, subprocess.SubprocessError):
        return {"status": "unavailable", "reason": "git_or_source_read_unavailable"}


def historical_text(path: str | Path, revision: str = "HEAD") -> str:
    """Return UTF-8 source exactly as committed at ``revision``."""
    relative = Path(path).as_posix().lstrip("/")
    result = subprocess.run(
        ["git", "show", f"{revision}:{relative}"], cwd=_REPOSITORY,
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    return result.stdout.decode("utf-8")


def historical_module(
    path: str | Path, revision: str = "HEAD", *, package: Optional[str] = None,
) -> types.ModuleType:
    """Execute one trusted historical source file against current dependencies."""
    relative = Path(path).as_posix().lstrip("/")
    source = historical_text(relative, revision)
    digest = hashlib.sha256(f"{revision}:{relative}".encode("utf-8")).hexdigest()[:12]
    name = f"_ina_history_{Path(relative).stem}_{digest}"
    module = types.ModuleType(name)
    module.__file__ = f"git:{revision}:{relative}"
    module.__package__ = package if package is not None else (Path(relative).parent.as_posix().replace("/", ".") if "/" in relative else "")
    sys.modules[name] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


__all__ = ["historical_module", "historical_text", "resolve_revision"]
