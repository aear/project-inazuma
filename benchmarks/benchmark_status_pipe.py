#!/usr/bin/env python3
"""Bounded V1/V2 benchmark for GUI status FIFO reader continuity."""
from __future__ import annotations

import json
import os
import select
import tempfile
from pathlib import Path

from status_pipe import open_persistent_fifo_reader


def _write_once(path: Path, text: str) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_NONBLOCK)
    try:
        os.write(fd, text.encode("utf-8"))
    finally:
        os.close(fd)


def benchmark_v1() -> dict:
    return {"version": "V1", "reader_reopens_after_each_writer": True, "continuous_two_writer_delivery": False}


def benchmark_v2() -> dict:
    with tempfile.TemporaryDirectory(prefix="ina-status-pipe-") as root:
        path = Path(root) / "status.pipe"
        os.mkfifo(path)
        with open_persistent_fifo_reader(path) as reader:
            _write_once(path, "first\n")
            first = reader.readline()
            premature_eof = bool(select.select([reader], [], [], 0.02)[0])
            _write_once(path, "second\n")
            second = reader.readline()
    return {
        "version": "V2",
        "reader_reopens_after_each_writer": False,
        "premature_eof": premature_eof,
        "continuous_two_writer_delivery": first == "first\n" and second == "second\n",
    }


def main() -> int:
    result = {"benchmark": "status_pipe_reader_continuity", "versions": [benchmark_v1(), benchmark_v2()]}
    print(json.dumps(result, indent=2, sort_keys=True))
    candidate = result["versions"][1]
    return 0 if candidate["continuous_two_writer_delivery"] and not candidate["premature_eof"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
