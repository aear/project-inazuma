from contextlib import contextmanager
import json
import os
import tempfile
from pathlib import Path
from functools import lru_cache
from typing import Any, Dict, Iterator, Optional, TextIO


def load_json_dict(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


@contextmanager
def file_lock(lock_path: Path) -> Iterator[None]:
    try:
        import fcntl  # type: ignore
    except Exception:
        yield
        return

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as fh:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        except Exception:
            yield
            return
        try:
            yield
        finally:
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass


def _unescape_mount_path(value: str) -> str:
    for escaped, literal in (("\\040", " "), ("\\011", "\t"), ("\\012", "\n"), ("\\134", "\\")):
        value = value.replace(escaped, literal)
    return value


@lru_cache(maxsize=64)
def filesystem_type(path: str) -> Optional[str]:
    """Return the filesystem type using mountinfo without spawning a process."""
    probe = Path(path)
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        resolved = str(probe.resolve())
        lines = Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines()
    except Exception:
        return None
    best = (0, None)
    for line in lines:
        try:
            before, after = line.split(" - ", 1)
            mountpoint = _unescape_mount_path(before.split()[4])
            fstype = after.split()[0].lower()
        except (IndexError, ValueError):
            continue
        prefix = mountpoint.rstrip("/") + "/"
        if resolved == mountpoint or resolved.startswith(prefix):
            if len(mountpoint) >= best[0]:
                best = (len(mountpoint), fstype)
    return best[1]


def should_fsync(path: Path, *, durable: bool = True) -> bool:
    """Choose foreground fsync policy; atomic rename is always retained."""
    if not durable:
        return False
    mode = str(os.getenv("INA_FSYNC_MODE") or "auto").strip().lower()
    if mode in {"always", "on", "true", "1"}:
        return True
    if mode in {"never", "off", "false", "0"}:
        return False
    # A Btrfs fsync can wait on a filesystem-wide log commit. Ina keeps the
    # atomic replace guarantee but lets the kernel flush these writes later.
    return filesystem_type(str(path)) != "btrfs"


def flush_for_durability(handle: TextIO, path: Path, *, durable: bool = True) -> bool:
    handle.flush()
    if not should_fsync(path, durable=durable):
        return False
    os.fsync(handle.fileno())
    return True


def atomic_write_json(
    path: Path,
    payload: Any,
    *,
    indent: int = 2,
    ensure_ascii: bool = True,
) -> None:
    """
    Write JSON through a same-directory temp file and atomically replace the
    target. Foreground fsync is skipped on Btrfs to avoid filesystem-wide log
    commit stalls; INA_FSYNC_MODE=always or never overrides auto detection.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=indent, ensure_ascii=ensure_ascii)
            flush_for_durability(fh, path)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        raise
