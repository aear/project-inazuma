"""Private, capability-free pointers to separately owned projects."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
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


__all__ = ["ExternalProject", "ProjectRegistryError", "project_status", "register_project"]
