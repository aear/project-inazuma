"""Bounded NVMe hot tier with verified durable-HDD draining for Experience Cycles."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
from typing import Any, Mapping

from config_layers import load_config
from io_utils import atomic_write_json, load_json_dict
from storage_layout import format_child_path, root_is_writable


DEFAULT_MAX_HOT_BYTES = 512 * 1024 * 1024
DEFAULT_MAX_HOT_FILES = 20_000
DEFAULT_MIN_FREE_BYTES = 64 * 1024 * 1024 * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CycleTierPolicy:
    def __init__(
        self, child: str, durable_root: Path, *, config: Mapping[str, Any] | None = None,
        enable_hot: bool = True,
    ) -> None:
        self.child = str(child)
        self.durable_root = Path(durable_root)
        self.config = dict(config or load_config())
        raw = self.config.get("experience_cycle_storage")
        raw = raw if isinstance(raw, Mapping) else {}
        layout = self.config.get("storage_layout")
        layout = layout if isinstance(layout, Mapping) else {}
        self.max_hot_bytes = max(1024 * 1024, int(raw.get("max_hot_bytes", DEFAULT_MAX_HOT_BYTES)))
        self.max_hot_files = max(100, int(raw.get("max_hot_files", DEFAULT_MAX_HOT_FILES)))
        self.min_free_bytes = max(1024 * 1024 * 1024, int(raw.get("min_free_bytes", DEFAULT_MIN_FREE_BYTES)))
        self.hot_root: Path | None = None
        current_only = bool(layout.get("fast_runtime_current_child_only", True))
        current_child = self.config.get("current_child")
        fast_enabled = bool(layout.get("fast_runtime_enabled", True)) and bool(raw.get("enabled", True))
        if enable_hot and fast_enabled and (not current_only or not current_child or str(current_child) == self.child):
            fast = format_child_path(layout.get("fast_runtime_root"), self.child)
            mount = format_child_path(layout.get("fast_mount"), self.child)
            if fast is not None and (mount is None or mount.is_mount()) and root_is_writable(fast):
                self.hot_root = fast / "experience_cycles"

    @property
    def state_path(self) -> Path | None:
        return self.hot_root / "usage.json" if self.hot_root is not None else None

    def roots_for_read(self) -> tuple[Path, ...]:
        return ((self.hot_root,) if self.hot_root is not None else ()) + (self.durable_root,)

    def _usage(self) -> dict[str, int]:
        state = load_json_dict(self.state_path) if self.state_path is not None else {}
        return {"written_bytes": max(0, int(state.get("written_bytes", 0))), "written_files": max(0, int(state.get("written_files", 0)))}

    def choose_write_root(self, estimated_bytes: int = 64 * 1024) -> Path:
        hot = self.hot_root
        if hot is None:
            return self.durable_root
        usage = self._usage()
        estimate = max(1, int(estimated_bytes))
        try:
            probe = hot
            while not probe.exists() and probe != probe.parent:
                probe = probe.parent
            disk = shutil.disk_usage(probe)
            reserve = max(self.min_free_bytes, int(disk.total * 0.10))
            enough_free = disk.free - estimate >= reserve
        except OSError:
            enough_free = False
        if (
            enough_free
            and usage["written_bytes"] + estimate <= self.max_hot_bytes
            and usage["written_files"] + 1 <= self.max_hot_files
        ):
            return hot
        return self.durable_root

    def record_write(self, root: Path, size_bytes: int) -> None:
        if self.hot_root is None or Path(root) != self.hot_root:
            return
        usage = self._usage()
        usage["written_bytes"] += max(0, int(size_bytes))
        usage["written_files"] += 1
        atomic_write_json(self.state_path, usage, indent=2, ensure_ascii=False)

    def reconcile_hot_usage(self) -> dict[str, int]:
        if self.hot_root is None or not self.hot_root.exists():
            return {"written_bytes": 0, "written_files": 0}
        files = 0
        size = 0
        for directory in ("manifests", "attempts", "decisions"):
            folder = self.hot_root / directory
            if not folder.is_dir():
                continue
            for entry in os.scandir(folder):
                if entry.is_file(follow_symlinks=False):
                    files += 1
                    size += entry.stat(follow_symlinks=False).st_size
                    if files >= self.max_hot_files:
                        break
        state = {"written_bytes": size, "written_files": files}
        atomic_write_json(self.state_path, state, indent=2, ensure_ascii=False)
        return state

    def drain_to_durable(self, *, max_files: int = 256, max_bytes: int = 16 * 1024 * 1024) -> dict[str, Any]:
        """Copy, hash-verify, then retire a bounded set of hot files."""
        if self.hot_root is None or not self.hot_root.exists():
            return {"moved_files": 0, "moved_bytes": 0, "remaining": False}
        file_limit = max(1, min(2048, int(max_files)))
        byte_limit = max(1024, min(256 * 1024 * 1024, int(max_bytes)))
        moved_files = moved_bytes = 0
        moved_paths: list[str] = []
        remaining = False
        for directory in ("attempts", "decisions", "manifests"):
            source_dir = self.hot_root / directory
            if not source_dir.is_dir():
                continue
            for source in sorted(source_dir.glob("*.json"), key=lambda path: path.name):
                size = source.stat().st_size
                if moved_files >= file_limit or (moved_files and moved_bytes + size > byte_limit):
                    remaining = True
                    break
                target = self.durable_root / directory / source.name
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_suffix(target.suffix + ".copying")
                shutil.copy2(source, temporary)
                if _sha256(source) != _sha256(temporary):
                    temporary.unlink(missing_ok=True)
                    raise IOError(f"cycle tier verification failed: {source.name}")
                os.replace(temporary, target)
                source.unlink()
                moved_paths.append(str(Path(directory) / source.name))
                moved_files += 1
                moved_bytes += size
            if remaining:
                break
        usage = self.reconcile_hot_usage()
        return {"moved_files": moved_files, "moved_bytes": moved_bytes, "moved_paths": moved_paths, "remaining": remaining or usage["written_files"] > 0, "hot_usage": usage}


__all__ = ["CycleTierPolicy", "DEFAULT_MAX_HOT_BYTES", "DEFAULT_MAX_HOT_FILES", "DEFAULT_MIN_FREE_BYTES"]
