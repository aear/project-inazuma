import json
from pathlib import Path

import pytest

from external_project_registry import ProjectRegistryError, project_status, register_project


def test_mercury_pointer_and_notes_are_private_and_capability_free(tmp_path):
    root = tmp_path / "inazuma"
    mercury = tmp_path / "mercury"
    root.mkdir()
    mercury.mkdir()
    record = register_project(name="Project Mercury", path=mercury, inazuma_root=root)
    assert record.read_authorized is False
    assert record.write_authorized is False
    assert record.execute_authorized is False
    registry = json.loads((root / ".private/external_projects.json").read_text())
    assert registry["Project Mercury"]["path"] == str(mercury.resolve())
    status = project_status("Project Mercury", inazuma_root=root)
    assert status == {"name": "Project Mercury", "configured": True, "reachable": True, "capabilities": []}
    assert str(mercury) not in repr(status)


def test_rejects_inazuma_and_broad_paths(tmp_path):
    root = tmp_path / "inazuma"
    root.mkdir()
    for path in (root, root / "nested", Path("/"), Path.home()):
        with pytest.raises(ProjectRegistryError):
            register_project(name="Project Mercury", path=path, inazuma_root=root)
