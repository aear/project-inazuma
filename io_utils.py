import json
import os
import tempfile
from pathlib import Path
from typing import Any


def atomic_write_json(
    path: Path,
    payload: Any,
    *,
    indent: int = 2,
    ensure_ascii: bool = True,
) -> None:
    """
    Write JSON atomically: write to a temp file in the same directory, fsync,
    then replace the target. This prevents empty or partially-written files
    when a process is interrupted mid-write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=indent, ensure_ascii=ensure_ascii)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        raise
