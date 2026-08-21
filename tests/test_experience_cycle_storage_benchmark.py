from benchmarks.benchmark_experience_cycle_storage import run_benchmark


def test_storage_benchmark_compares_history_and_cycle_on_both_devices(tmp_path):
    hdd = tmp_path / "hdd"
    nvme = tmp_path / "nvme"
    hdd.mkdir()
    nvme.mkdir()

    result = run_benchmark(hdd_root=hdd, nvme_root=nvme, samples=3)

    assert result["historical_revision"] != "working-tree"
    assert result["results"]["hdd"]["V1"]["samples"] == 3
    assert result["results"]["hdd"]["V2"]["median_latency_ms"] >= 0
    assert result["results"]["nvme"]["V2"]["mean_storage_bytes"] > 0
    assert "nvme_vs_hdd_median_ratio" in result["results"]["comparison"]["V2"]
