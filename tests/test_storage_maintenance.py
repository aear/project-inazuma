import gzip
from pathlib import Path

from storage_maintenance import compress_verified, maintenance_opportunity, perform_choice


def test_model_manager_exposes_choice_backed_maintenance_seam():
    source = Path("model_manager.py").read_text(encoding="utf-8")
    assert "def request_storage_maintenance(" in source
    assert '_update_storage_maintenance_opportunity()' in source
    assert 'get_inastate("storage_maintenance_request")' in source


def test_opportunity_surfaces_choice_without_modifying_candidate(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    candidate = logs / "ina_status.log.1"
    candidate.write_bytes(b"repeatable evidence\n" * 1_000_000)

    opportunity = maintenance_opportunity(tmp_path)

    assert opportunity["available"] is True
    assert opportunity["choices"] == ["inspect", "compress_one", "defer", "decline"]
    assert candidate.exists()


def test_decline_and_defer_never_modify_candidate(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    candidate = logs / "ina_status.log.1"
    candidate.write_bytes(b"evidence\n" * 2_000_000)

    assert perform_choice(tmp_path, "defer")["status"] == "deferred"
    assert perform_choice(tmp_path, "decline")["status"] == "declined"
    assert candidate.exists()


def test_compress_verified_preserves_content_before_retiring_source(tmp_path):
    source = tmp_path / "diagnostic.log.1"
    payload = b"bounded diagnostic evidence\n" * 200
    source.write_bytes(payload)

    result = compress_verified(source)

    target = tmp_path / "diagnostic.log.1.gz"
    assert result["verified"] is True
    assert not source.exists()
    assert gzip.open(target, "rb").read() == payload
