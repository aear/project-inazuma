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
