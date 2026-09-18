import json

from creative_versioning import preserve_creative_version
from daw_engine import DawProject, save_project


def test_hash_addressed_versions_preserve_lineage_and_deduplicate(tmp_path):
    source = tmp_path / "drawing.png"
    source.write_bytes(b"first")
    first = preserve_creative_version(source, medium="drawing", label="canvas")
    repeated = preserve_creative_version(source, medium="drawing", label="canvas")
    source.write_bytes(b"second")
    second = preserve_creative_version(source, medium="drawing", label="canvas")

    assert first["created_snapshot"] is True
    assert repeated["created_snapshot"] is False
    assert first["snapshot_path"] == repeated["snapshot_path"]
    assert second["snapshot_path"] != first["snapshot_path"]
    assert open(first["snapshot_path"], "rb").read() == b"first"
    records = [json.loads(line) for line in open(first["ledger_path"], encoding="utf-8")]
    assert len(records) == 3
    assert all(record["schema"] == "ina.creative_asset_version/V1" for record in records)


def test_daw_save_records_successive_project_versions(tmp_path):
    path = tmp_path / "song.ina-daw.json"
    project = DawProject(name="First Song")
    save_project(project, path)
    project.bpm = 90
    save_project(project, path)

    ledger = tmp_path / ".versions" / "music" / "First-Song" / "versions.jsonl"
    records = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 2
    assert records[0]["sha256"] != records[1]["sha256"]
    assert all(record["metadata"]["project_schema_version"] == 2 for record in records)
