"""Capability-scoped virtual drives: media is read-only, Ina's HDD is writable data."""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping
import uuid


MAX_LIST_ENTRIES = 500
MAX_READ_BYTES = 8 * 1024 * 1024
MAX_WRITE_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True)
class VirtualDrive:
    id: str
    label: str
    root: Path
    writable: bool = False
    source: str = "media"

    def payload(self) -> dict[str, Any]:
        result = asdict(self)
        result["root"] = str(self.root)
        result["capabilities"] = ["list", "read"] + (["write", "mkdir", "rename"] if self.writable else [])
        result["execution_allowed"] = False
        return result


def configured_drives(config: Mapping[str, Any], child: str, *, project_root: Path | str = ".") -> tuple[VirtualDrive, ...]:
    roots = (
        ("books", "Books", config.get("book_folder_path"), "books"),
        ("music", "Music & video", config.get("music_folder_path"), "music"),
        ("code", "Ina's reading", config.get("ina_work_path") or Path(project_root), "code"),
        ("history", "Git history", Path(project_root) / "AI_Children" / str(child) / "memory" / "github_history", "github_history"),
    )
    drives = [VirtualDrive(identifier, label, Path(value), False, source) for identifier, label, value, source in roots if value]
    storage = config.get("storage_layout") if isinstance(config.get("storage_layout"), Mapping) else {}
    durable_mount = storage.get("durable_mount")
    writable = config.get("ina_hdd_writable_path")
    if not writable and durable_mount:
        writable = Path(str(durable_mount)) / "Ina Files" / str(child)
    if not writable:
        writable = Path(project_root) / "AI_Children" / str(child) / "files"
    drives.append(VirtualDrive("ina_hdd", "Ina HDD", Path(str(writable).format(child=child)), True, "personal"))
    return tuple(drives)


class VirtualFileSystem:
    def __init__(self, drives: tuple[VirtualDrive, ...]) -> None:
        self.drives = {drive.id: drive for drive in drives}

    def describe(self) -> list[dict[str, Any]]:
        return [drive.payload() for drive in self.drives.values()]

    def ensure_writable_roots(self) -> None:
        for drive in self.drives.values():
            if drive.writable:
                drive.root.mkdir(parents=True, exist_ok=True, mode=0o700)

    def _drive(self, drive_id: str) -> VirtualDrive:
        try:
            return self.drives[str(drive_id)]
        except KeyError as exc:
            raise ValueError(f"unknown drive: {drive_id}") from exc

    def _path(self, drive: VirtualDrive, relative: str = "", *, existing: bool = False) -> Path:
        root = drive.root.resolve(strict=False)
        candidate = root / str(relative or ".")
        resolved = candidate.resolve(strict=existing)
        if resolved != root and root not in resolved.parents:
            raise PermissionError("path leaves the selected drive")
        if candidate.is_symlink() or any(part.is_symlink() for part in [candidate, *candidate.parents] if part != root and root in part.resolve(strict=False).parents):
            raise PermissionError("symbolic links are not available through Ina's explorer")
        return resolved

    def list(self, drive_id: str, relative: str = "") -> list[dict[str, Any]]:
        drive = self._drive(drive_id)
        path = self._path(drive, relative, existing=True)
        if not path.is_dir():
            raise NotADirectoryError(relative)
        rows = []
        for item in sorted(path.iterdir(), key=lambda value: (not value.is_dir(), value.name.casefold())):
            if item.is_symlink():
                continue
            rows.append({"name": item.name, "directory": item.is_dir(), "size_bytes": item.stat().st_size if item.is_file() else None})
            if len(rows) >= MAX_LIST_ENTRIES:
                break
        return rows

    def read(self, drive_id: str, relative: str) -> bytes:
        drive = self._drive(drive_id)
        path = self._path(drive, relative, existing=True)
        if not path.is_file():
            raise FileNotFoundError(relative)
        if path.stat().st_size > MAX_READ_BYTES:
            raise ValueError("file exceeds the explorer read limit")
        return path.read_bytes()

    def write(self, drive_id: str, relative: str, data: bytes | str) -> Path:
        drive = self._drive(drive_id)
        if not drive.writable:
            raise PermissionError("this drive is read-only")
        payload = data.encode("utf-8") if isinstance(data, str) else bytes(data)
        if len(payload) > MAX_WRITE_BYTES:
            raise ValueError("file exceeds the explorer write limit")
        path = self._path(drive, relative)
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
        temporary.write_bytes(payload)
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        return path

    def mkdir(self, drive_id: str, relative: str) -> Path:
        drive = self._drive(drive_id)
        if not drive.writable:
            raise PermissionError("this drive is read-only")
        path = self._path(drive, relative)
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
        return path

    def rename(self, drive_id: str, relative: str, new_name: str) -> Path:
        drive = self._drive(drive_id)
        if not drive.writable:
            raise PermissionError("this drive is read-only")
        source = self._path(drive, relative, existing=True)
        if Path(new_name).name != new_name or new_name in {"", ".", ".."}:
            raise ValueError("new name must be one filename")
        target = self._path(drive, str(Path(relative).parent / new_name))
        if target.exists():
            raise FileExistsError(f"target already exists: {new_name}")
        source.rename(target)
        return target

    def execute(self, *_args: Any, **_kwargs: Any) -> None:
        raise PermissionError("file execution is not a capability of Ina's explorer")


__all__ = ["VirtualDrive", "VirtualFileSystem", "configured_drives", "MAX_LIST_ENTRIES", "MAX_READ_BYTES", "MAX_WRITE_BYTES"]
