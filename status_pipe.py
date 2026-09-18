"""Small POSIX status-FIFO primitives shared by the GUI and tests."""
from __future__ import annotations

import os
from pathlib import Path
from typing import IO


def open_persistent_fifo_reader(path: str | os.PathLike[str]) -> IO[str]:
    """Open both FIFO ends so writer disconnects do not turn into reader EOF.

    Linux FIFO readers opened read-only see EOF whenever the last short-lived
    writer closes.  Reopening creates a race where another writer can open just
    before the reader closes and then receive EPIPE.  Keeping a local write end
    open makes the GUI reader stable across independent status messages.
    """
    fd = os.open(Path(path), os.O_RDWR)
    try:
        return os.fdopen(fd, "r", encoding="utf-8", errors="replace")
    except Exception:
        os.close(fd)
        raise


__all__ = ["open_persistent_fifo_reader"]
