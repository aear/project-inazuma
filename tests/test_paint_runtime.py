import os
from pathlib import Path

from paint_runtime import _run_bytecode, ensure_paint_runtime


def test_runtime_rebuilds_only_when_source_stamp_changes(tmp_path: Path):
    source = tmp_path / "paint.py"
    runtime = tmp_path / "runtime"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    first = ensure_paint_runtime("Ina", source=source, runtime_root=runtime)
    assert first["rebuilt"] is True
    assert first["bytecode"].exists()
    second = ensure_paint_runtime("Ina", source=source, runtime_root=runtime)
    assert second["rebuilt"] is False
    source.write_text("VALUE = 1000\n", encoding="utf-8")
    third = ensure_paint_runtime("Ina", source=source, runtime_root=runtime)
    assert third["rebuilt"] is True


def test_runtime_detects_size_change_when_mtime_is_restored(tmp_path: Path):
    source = tmp_path / "paint.py"
    runtime = tmp_path / "runtime"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    first = ensure_paint_runtime("Ina", source=source, runtime_root=runtime)
    original_mtime = source.stat().st_mtime_ns
    source.write_text("VALUE = 1000\n", encoding="utf-8")
    os.utime(source, ns=(original_mtime, original_mtime))
    second = ensure_paint_runtime("Ina", source=source, runtime_root=runtime)
    assert first["source_stamp"]["size"] != second["source_stamp"]["size"]
    assert second["rebuilt"] is True


def test_compiled_runtime_can_be_executed(tmp_path: Path):
    source = tmp_path / "paint.py"
    runtime = tmp_path / "runtime"
    marker = tmp_path / "executed"
    source.write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text(__name__)\n",
        encoding="utf-8",
    )
    compiled = ensure_paint_runtime("Ina", source=source, runtime_root=runtime)
    _run_bytecode(compiled["bytecode"])
    assert marker.read_text(encoding="utf-8") == "__main__"
