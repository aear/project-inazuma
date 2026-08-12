from pathlib import Path

import project_version


def test_project_version_is_v3_and_matches_release_file():
    assert project_version.VERSION == "3.0.0"
    assert project_version.RELEASE == "V3"
    assert Path("VERSION").read_text(encoding="utf-8").strip() == project_version.VERSION
