"""Compile and run Ina's paint UI from rebuildable fast storage."""
from __future__ import annotations

import fcntl
import importlib.machinery
import importlib.util
import json
import os
import py_compile
import sys
from pathlib import Path
from typing import Any, Dict

from storage_layout import fast_runtime_path, load_config


PROJECT_ROOT = Path(__file__).resolve().parent
PAINT_SOURCE = PROJECT_ROOT / "paint_window.py"


def _run_bytecode(path: Path) -> None:
    """Execute a compiled .pyc as the main module."""
    loader = importlib.machinery.SourcelessFileLoader("__main__", str(path))
    code = loader.get_code("__main__")
    if code is None:
        raise ImportError(f"Could not load bytecode from {path}")
    namespace = {
        "__name__": "__main__",
        "__file__": str(path),
        "__cached__": str(path),
        "__loader__": loader,
        "__package__": None,
        "__spec__": None,
    }
    exec(code, namespace)


def paint_runtime_paths(child: str, config: Dict[str, Any] | None = None, runtime_root: Path | None = None) -> Dict[str, Path]:
    fallback = runtime_root or (PROJECT_ROOT / "AI_Children" / child / "memory" / "fast_runtime" / "paint")
    marker = fallback / "runtime.marker" if runtime_root else fast_runtime_path(
        child, "runtime.marker", fallback / "runtime.marker", subdir="paint", config=config,
    )
    root = marker.parent
    return {
        "root": root,
        "bytecode": root / "paint_window.pyc",
        "manifest": root / "paint_window.runtime.json",
        "lock": root / "paint_window.lock",
        "staging": root / "drawings",
    }


def _source_stamp(source: Path) -> Dict[str, Any]:
    stat = source.stat()
    return {
        "source": str(source),
        "mtime_ns": stat.st_mtime_ns,
        "size": stat.st_size,
        "python_cache_tag": sys.implementation.cache_tag,
        "python_magic": importlib.util.MAGIC_NUMBER.hex(),
    }


def ensure_paint_runtime(child: str, *, source: Path = PAINT_SOURCE, config: Dict[str, Any] | None = None, runtime_root: Path | None = None) -> Dict[str, Any]:
    paths = paint_runtime_paths(child, config, runtime_root)
    paths["root"].mkdir(parents=True, exist_ok=True)
    stamp = _source_stamp(source)
    current = {}
    try:
        current = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    except Exception:
        pass
    bytecode_compatible = (
        paths["bytecode"].exists()
        and paths["bytecode"].read_bytes()[:4] == importlib.util.MAGIC_NUMBER
    )
    rebuilt = current != stamp or not bytecode_compatible
    if rebuilt:
        temporary = paths["bytecode"].with_suffix(".pyc.tmp")
        py_compile.compile(str(source), cfile=str(temporary), doraise=True)
        os.replace(temporary, paths["bytecode"])
        manifest_tmp = paths["manifest"].with_suffix(".json.tmp")
        manifest_tmp.write_text(json.dumps(stamp, indent=2), encoding="utf-8")
        os.replace(manifest_tmp, paths["manifest"])
    return {**paths, "rebuilt": rebuilt, "source_stamp": stamp}


def paint_runtime_is_running(child: str, config: Dict[str, Any] | None = None) -> bool:
    paths = paint_runtime_paths(child, config)
    paths["root"].mkdir(parents=True, exist_ok=True)
    handle = paths["lock"].open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return True
    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    handle.close()
    return False


def run_paint_runtime() -> int:
    config = load_config(PROJECT_ROOT / "config.json")
    child = str(config.get("current_child") or "default_child")
    paths = paint_runtime_paths(child, config)
    paths["root"].mkdir(parents=True, exist_ok=True)
    lock_handle = paths["lock"].open("a+")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock_handle.close()
        return 0
    runtime = ensure_paint_runtime(child, config=config)
    lock_handle.seek(0)
    lock_handle.truncate()
    lock_handle.write(str(os.getpid()))
    lock_handle.flush()
    os.environ["INA_PAINT_FAST_STAGING"] = str(runtime["staging"])
    os.environ["INA_PAINT_RUNTIME_REBUILT"] = "1" if runtime["rebuilt"] else "0"
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    try:
        _run_bytecode(runtime["bytecode"])
    finally:
        lock_handle.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(run_paint_runtime())
