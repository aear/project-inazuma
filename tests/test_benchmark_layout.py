from pathlib import Path


def test_benchmark_runner_layout_v1_root_clutter_vs_v2_package():
    """V2 keeps executable benchmark runners out of the repository root."""
    root = Path(__file__).resolve().parent.parent
    root_runners = sorted(root.glob("benchmark_*.py"))
    packaged_runners = sorted((root / "benchmarks").glob("benchmark_*.py"))

    v1 = {"root_runners": 18, "packaged_runners": 0, "importable_package": False}
    v2 = {
        "root_runners": len(root_runners),
        "packaged_runners": len(packaged_runners),
        "importable_package": (root / "benchmarks" / "__init__.py").is_file(),
    }

    assert v1 == {"root_runners": 18, "packaged_runners": 0, "importable_package": False}
    assert v2 == {"root_runners": 0, "packaged_runners": 18, "importable_package": True}
