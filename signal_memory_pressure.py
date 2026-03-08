#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def _load_config() -> Dict[str, Any]:
    path = Path("config.json")
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _inastate_path(child: str) -> Path:
    return Path("AI_Children") / child / "memory" / "inastate.json"


def _load_json_dict(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=4, ensure_ascii=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        raise


def _with_lock(lock_path: Path):
    try:
        import fcntl  # type: ignore
    except Exception:
        class _NoOp:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, tb):
                return False

        return _NoOp()

    class _LockCtx:
        def __init__(self, p: Path):
            self._path = p
            self._fh = None

        def __enter__(self):
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(self._path, "w")
            fcntl.flock(self._fh, fcntl.LOCK_EX)
            return None

        def __exit__(self, exc_type, exc, tb):
            if self._fh:
                try:
                    fcntl.flock(self._fh, fcntl.LOCK_UN)
                except Exception:
                    pass
                self._fh.close()
            return False

    return _LockCtx(lock_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Signal Ina to shed memory now.")
    parser.add_argument(
        "--action",
        default="too_much_memory",
        choices=["too_much_memory", "memory_too_high", "shed_memory_now"],
        help="Operator action sent to model_manager.",
    )
    parser.add_argument("--note", default="manual operator signal", help="Optional note for the signal payload.")
    parser.add_argument("--source", default="hotkey", help="Signal source label.")
    parser.add_argument("--child", default=None, help="Optional child override.")
    args = parser.parse_args()

    cfg = _load_config()
    child = args.child or cfg.get("current_child") or "Inazuma_Yagami"
    inastate = _inastate_path(str(child))
    lock_path = inastate.with_name("inastate.lock")

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": args.action,
        "source": args.source,
        "note": args.note,
    }

    with _with_lock(lock_path):
        state = _load_json_dict(inastate)
        state["operator_memory_signal"] = payload
        _atomic_write_json(inastate, state)

    print(json.dumps({"ok": True, "child": child, "signal": payload}, ensure_ascii=True))


if __name__ == "__main__":
    main()
