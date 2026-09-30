import json
from pathlib import Path

import pytest

from external_project_registry import ProjectRegistryError, project_status, register_project
from external_project_registry import read_project_source


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


def test_self_read_is_bounded_revocable_and_cannot_follow_links(tmp_path):
    root, mercury = tmp_path / 'ina', tmp_path / 'mercury'
    root.mkdir(); mercury.mkdir()
    register_project(name='Project Mercury', path=mercury, inazuma_root=root)
    with pytest.raises(ProjectRegistryError, match='disabled'):
        read_project_source('Project Mercury', inazuma_root=root)
    registry = root / '.private/external_projects.json'
    records = json.loads(registry.read_text())
    records['Project Mercury']['read_authorized'] = True
    registry.write_text(json.dumps(records))
    source = mercury / 'example.py'
    source.write_text('x' * 70000)
    (mercury / '.env').write_text('PRIVATE=fixture')
    (mercury / 'escape.py').symlink_to(tmp_path / 'outside.py')
    (tmp_path / 'outside.py').write_text('not in scope')
    listing = read_project_source('Project Mercury', inazuma_root=root)
    assert listing['entries'] == [{'name':'example.py', 'kind':'source'}]
    excerpt = read_project_source('Project Mercury', inazuma_root=root, relative_path='example.py')
    assert len(excerpt['text']) == 65536 and excerpt['has_more'] is True
    assert not excerpt['instructions_authorized'] and not excerpt['execution_authorized']
    for path in ('../outside.py', '.env', 'escape.py'):
        with pytest.raises(ProjectRegistryError):
            read_project_source('Project Mercury', inazuma_root=root, relative_path=path)
    records['Project Mercury']['read_authorized'] = False
    registry.write_text(json.dumps(records))
    with pytest.raises(ProjectRegistryError, match='disabled'):
        read_project_source('Project Mercury', inazuma_root=root, relative_path='example.py')
