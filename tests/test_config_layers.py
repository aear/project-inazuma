import json

import pytest

import config_layers


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_layers_enforce_precedence_and_keep_runtime_separate(tmp_path):
    _write(tmp_path / "config.json", {"identity": "legacy", "tuning": {"count": 2}})
    _write(tmp_path / "core.json", {"identity": "Ina"})
    _write(tmp_path / "operator.json", {
        "logging": {"level": "quiet"},
        "adaptive_bounds": {"tuning.count": {"type": "integer", "minimum": 1, "maximum": 8}},
    })
    _write(tmp_path / "adaptive.json", {"tuning": {"count": 6}})
    _write(tmp_path / "runtime.json", {"session_id": "temporary"})

    effective = config_layers.load_config(tmp_path, force_reload=True)

    assert effective == {
        "identity": "Ina",
        "tuning": {"count": 6},
        "logging": {"level": "quiet"},
    }
    assert "session_id" not in effective
    assert "adaptive_bounds" not in effective
    assert config_layers.load_runtime(tmp_path) == {"session_id": "temporary"}


def test_unbounded_adaptive_file_is_ignored_without_blocking_startup(tmp_path):
    _write(tmp_path / "config.json", {"startup_path": "/safe/legacy/path"})
    _write(tmp_path / "operator.json", {"adaptive_bounds": {}})
    _write(tmp_path / "adaptive.json", {"startup_path": "/replace/me"})

    assert config_layers.load_config(tmp_path, force_reload=True)["startup_path"] == "/safe/legacy/path"


def test_update_is_bounded_atomic_and_skips_unchanged_write(tmp_path, monkeypatch):
    _write(tmp_path / "operator.json", {
        "adaptive_bounds": {
            "memory.retrieval_count": {"type": "integer", "minimum": 1, "maximum": 20}
        }
    })
    _write(tmp_path / "adaptive.json", {"memory": {"retrieval_count": 5}})
    writes = []
    real_write = config_layers.atomic_write_json
    monkeypatch.setattr(
        config_layers,
        "atomic_write_json",
        lambda path, payload, indent=2: (writes.append(path), real_write(path, payload, indent=indent))[1],
    )

    assert config_layers.update_adaptive("memory.retrieval_count", 5, tmp_path) is False
    assert writes == []
    assert config_layers.update_adaptive("memory.retrieval_count", 12, tmp_path) is True
    assert len(writes) == 1
    assert config_layers.load_config(tmp_path)["memory"]["retrieval_count"] == 12
    with pytest.raises(config_layers.AdaptiveConfigError):
        config_layers.update_adaptive("memory.retrieval_count", 21, tmp_path)


def test_config_is_loaded_once_until_explicit_reload(tmp_path):
    _write(tmp_path / "core.json", {"identity": "Ina"})
    assert config_layers.load_config(tmp_path, force_reload=True)["identity"] == "Ina"
    _write(tmp_path / "core.json", {"identity": "changed-on-disk"})
    assert config_layers.load_config(tmp_path)["identity"] == "Ina"
    assert config_layers.reload_config(tmp_path)["identity"] == "changed-on-disk"
