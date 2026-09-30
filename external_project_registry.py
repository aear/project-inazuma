"""Private, capability-free pointers to separately owned projects."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
import stat
import hashlib
from itertools import islice
from pathlib import Path
from typing import Any


DEFAULT_REGISTRY = Path(".private/external_projects.json")
DEFAULT_NOTES_DIRECTORY = Path(".private/project_notes")


class ProjectRegistryError(ValueError):
    pass


@dataclass(frozen=True)
class ExternalProject:
    name: str
    path: str
    notes_file: str
    read_authorized: bool = False
    write_authorized: bool = False
    execute_authorized: bool = False
    handover_authorized: bool = False


def _safe_project_path(path: Path | str, *, inazuma_root: Path | str) -> Path:
    if not str(path).strip():
        raise ProjectRegistryError("project path is absent")
    candidate = Path(path).expanduser().resolve()
    root = Path(inazuma_root).resolve()
    forbidden = {Path("/").resolve(), Path.home().resolve(), root}
    if candidate in forbidden or root in candidate.parents:
        raise ProjectRegistryError("external project path is absent, broad, or inside Project Inazuma")
    return candidate


def register_project(
    *, name: str, path: Path | str, inazuma_root: Path | str,
    registry_path: Path | str = DEFAULT_REGISTRY,
    notes_directory: Path | str = DEFAULT_NOTES_DIRECTORY,
) -> ExternalProject:
    project_name = str(name or "").strip()
    if not project_name or any(char in project_name for char in "/\\\x00"):
        raise ProjectRegistryError("project name must be a simple non-empty name")
    project_path = _safe_project_path(path, inazuma_root=inazuma_root)
    registry = Path(inazuma_root) / registry_path
    notes_dir = Path(inazuma_root) / notes_directory
    registry.parent.mkdir(parents=True, exist_ok=True)
    notes_dir.mkdir(parents=True, exist_ok=True)
    notes_file = notes_dir / f"{project_name.lower().replace(' ', '_')}.md"
    if not notes_file.exists():
        notes_file.write_text(
            f"# Private notes for {project_name}\n\n"
            "This file is local-only. Record handover context here; do not store credentials.\n",
            encoding="utf-8",
        )
    records: dict[str, Any] = {}
    if registry.exists():
        loaded = json.loads(registry.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ProjectRegistryError("project registry must contain an object")
        records = loaded
    record = ExternalProject(project_name, str(project_path), str(notes_file.resolve()))
    records[project_name] = asdict(record)
    temporary = registry.with_suffix(registry.suffix + ".tmp")
    temporary.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(registry)
    return record


def project_status(
    name: str, *, inazuma_root: Path | str,
    registry_path: Path | str = DEFAULT_REGISTRY,
) -> dict[str, Any]:
    """Return deliberately redacted status; never return a project path or notes."""
    registry = Path(inazuma_root) / registry_path
    if not registry.exists():
        return {"name": name, "configured": False, "reachable": False, "capabilities": []}
    loaded = json.loads(registry.read_text(encoding="utf-8"))
    raw = loaded.get(name) if isinstance(loaded, dict) else None
    if not isinstance(raw, dict) or not str(raw.get("path", "")).strip():
        return {"name": name, "configured": False, "reachable": False, "capabilities": []}
    candidate = _safe_project_path(raw.get("path", ""), inazuma_root=inazuma_root)
    capabilities = [key.removesuffix("_authorized") for key in (
        "read_authorized", "write_authorized", "execute_authorized", "handover_authorized"
    ) if raw.get(key) is True]
    return {
        "name": name, "configured": True, "reachable": candidate.is_dir(),
        "capabilities": capabilities,
    }


SOURCE_SUFFIXES = frozenset({'.py', '.rs', '.c', '.h', '.cpp', '.hpp', '.js', '.ts', '.jsx', '.tsx', '.css', '.html', '.md', '.toml'})
EXCLUDED_DIRECTORIES = frozenset({'node_modules', 'venv', 'env', '__pycache__', 'target', 'build', 'dist'})


def read_project_source(name: str, *, inazuma_root: Path | str, relative_path: str = '.',
                        offset: int = 0) -> dict[str, Any]:
    """One voluntary directory listing or 64-KiB code excerpt; no execution.

    Resolve every component with directory descriptors and O_NOFOLLOW so symlink
    swaps cannot escape the configured root. Registry consent is reread each call.
    Hidden entries, environment/config JSON and private notes are excluded.
    """
    registry = Path(inazuma_root) / DEFAULT_REGISTRY
    if registry.stat().st_size > 64 * 1024:
        raise ProjectRegistryError('project registry exceeds the size budget')
    records = json.loads(registry.read_text(encoding='utf-8'))
    record = records.get(name) if isinstance(records, dict) else None
    if not isinstance(record, dict) or record.get('read_authorized') is not True:
        raise ProjectRegistryError('project read access is disabled')
    root = _safe_project_path(record.get('path', ''), inazuma_root=inazuma_root)
    path = Path(relative_path)
    if path.is_absolute() or any(part.startswith('.') or part in EXCLUDED_DIRECTORIES for part in path.parts):
        raise ProjectRegistryError('only visible project-relative source paths are readable')
    if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
        raise ProjectRegistryError('offset must be a nonnegative integer')
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for index, part in enumerate(path.parts):
            flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
            if index < len(path.parts) - 1:
                flags |= os.O_DIRECTORY
            child = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        metadata = os.fstat(descriptor)
        base = {'schema':'ina.external_project_self_read/V1', 'project':name,
                'relative_path':path.as_posix(), 'instructions_authorized':False,
                'automatic_memory_write_authorized':False, 'execution_authorized':False,
                'source':'external_project_code', 'private':True}
        if stat.S_ISDIR(metadata.st_mode):
            entries = []
            inspected = 0
            with os.scandir(descriptor) as iterator:
                for entry in islice(iterator, 257):
                    inspected += 1
                    if inspected > 256:
                        break
                    if entry.name.startswith('.') or entry.name in EXCLUDED_DIRECTORIES or entry.is_symlink():
                        continue
                    directory = entry.is_dir(follow_symlinks=False)
                    if directory or entry.is_file(follow_symlinks=False) and Path(entry.name).suffix.lower() in SOURCE_SUFFIXES:
                        entries.append({'name':entry.name, 'kind':'directory' if directory else 'source'})
            return {**base, 'kind':'directory', 'entries':sorted(entries, key=lambda row:row['name']),
                    'truncated':inspected > 256, 'inspection_limit':256}
        if not stat.S_ISREG(metadata.st_mode) or path.suffix.lower() not in SOURCE_SUFFIXES:
            raise ProjectRegistryError('only regular source files are readable')
        os.lseek(descriptor, offset, os.SEEK_SET)
        content = os.read(descriptor, 64 * 1024)
        return {**base, 'kind':'source_excerpt', 'text':content.decode('utf-8', 'replace'),
                'offset':offset, 'next_offset':offset+len(content),
                'has_more':offset+len(content) < metadata.st_size,
                'chunk_sha256':hashlib.sha256(content).hexdigest()}
    except OSError as exc:
        raise ProjectRegistryError('source is unavailable or crosses a symlink boundary') from exc
    finally:
        os.close(descriptor)


__all__ = ["ExternalProject", "ProjectRegistryError", "project_status", "register_project"]
