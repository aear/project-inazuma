import os
from types import SimpleNamespace
import sys

from background_interference import (
    MAX_PHASE_SECONDS,
    MAX_RAW_SAMPLES,
    BackgroundInterferenceBenchmark,
    LinuxSystemProbe,
    PipeWireErrorProbe,
)


def test_bounded_idle_loaded_run_reports_human_visible_and_thread_metrics(tmp_path):
    audio_samples = iter(({"sink": 10}, {"sink": 10}, {"sink": 10}, {"sink": 12}))

    benchmark = BackgroundInterferenceBenchmark(
        phase_seconds=0.1,
        sample_interval_seconds=0.01,
        audio_error_probe=lambda: next(audio_samples),
        input_probe=lambda: None,
        frame_probe=lambda: None,
    )
    result = benchmark.run(
        [sys.executable, "-c", "import time; time.sleep(2)"],
        working_directory=tmp_path,
        environment={"OMP_NUM_THREADS": "2"},
    )

    assert result["bounds"]["phase_seconds"] == 0.1
    assert len(result["loaded"]["raw_samples"]) <= MAX_RAW_SAMPLES
    assert result["loaded"]["audio"]["error_delta"] == 2
    assert result["loaded"]["input_latency"]["method"] == "provided_input_roundtrip"
    assert result["loaded"]["desktop_frame_latency"]["available"] is True
    assert result["loaded"]["context_switches_per_second"] >= 0
    assert result["loaded"]["involuntary_context_switches_per_second"] >= 0
    assert result["loaded"]["writeback_pressure"]["io_stall_ms_per_second"] >= 0
    assert "per_core_busy_percent" in result["loaded"]["per_core"]
    assert result["loaded"]["threads"]["task_peak"]["thread_count"] >= 1
    assert "runnable_thread_count" in result["loaded"]["threads"]["task_peak"]
    assert result["task"]["thread_environment"]["OMP_NUM_THREADS"] == "2"
    assert result["task"]["numerical_thread_limits_explicit"] is True
    assert result["task"]["returncode"] is not None


def test_default_input_metric_is_honestly_labelled_scheduler_proxy(tmp_path):
    benchmark = BackgroundInterferenceBenchmark(
        phase_seconds=0.1,
        sample_interval_seconds=0.01,
        audio_error_probe=lambda: {},
    )
    result = benchmark.run(
        [sys.executable, "-c", "import time; time.sleep(1)"],
        working_directory=tmp_path,
    )

    assert result["loaded"]["input_latency"]["available"] is True
    assert result["loaded"]["input_latency"]["method"] == "scheduler_wakeup_proxy"
    assert result["loaded"]["desktop_frame_latency"]["available"] is False


def test_phase_and_raw_sample_bounds_are_hard_limits():
    benchmark = BackgroundInterferenceBenchmark(
        phase_seconds=MAX_PHASE_SECONDS * 4,
        sample_interval_seconds=0.00001,
        audio_error_probe=lambda: {},
    )

    assert benchmark.phase_seconds == MAX_PHASE_SECONDS
    assert benchmark.sample_interval_seconds == 0.002
    assert MAX_RAW_SAMPLES == 512


def test_pipewire_error_probe_parses_cumulative_error_column(monkeypatch):
    output = """
R 42 128 48000 10.0us 20.0us 0.00 0 17 node-name
S 43 128 48000 10.0us 20.0us 0.00 0 3 other-node
"""
    import background_interference
    monkeypatch.setattr(
        background_interference.subprocess, "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=output),
    )

    assert PipeWireErrorProbe()() == {"42": 17, "43": 3}


def test_linux_thread_probe_separates_total_and_runnable_threads():
    probe = LinuxSystemProbe()
    system = probe.system_threads()
    target = probe.target_threads(os.getpid())

    assert system["logical_cpu_count"] >= 1
    assert system["thread_count"] >= system["process_count"]
    assert system["max_threads_per_process"] >= 1
    assert target["thread_count"] >= 1
    assert target["runnable_thread_count"] >= 1
    assert target["threads_per_logical_cpu"] > 0


def test_thread_fanout_is_scored_separately_from_cpu_totals():
    benchmark = BackgroundInterferenceBenchmark(audio_error_probe=lambda: {})
    baseline = {
        "audio": {}, "input_latency": {}, "desktop_frame_latency": {},
        "context_switches_per_second": 1, "involuntary_context_switches_per_second": 1,
        "writeback_pressure": {}, "per_core": {},
        "threads": {"system_end": {"thread_count": 100}},
    }
    loaded = {
        "audio": {}, "input_latency": {}, "desktop_frame_latency": {},
        "context_switches_per_second": 1, "involuntary_context_switches_per_second": 1,
        "writeback_pressure": {}, "per_core": {},
        "threads": {
            "system_end": {"thread_count": 110},
            "task_peak": {"threads_per_logical_cpu": 2.0},
        },
    }

    comparison = benchmark._comparison(baseline, loaded)

    assert comparison["regressions"]["thread_fanout"] is True
    assert comparison["loaded_minus_baseline"]["system_thread_count"] == 10


def test_background_interference_family_uses_git_history_for_v1():
    from module_benchmarks import benchmark_module

    v1, v2 = benchmark_module("background_interference")

    assert v1.version == "V1"
    assert v1.source_revision != "working-tree"
    assert v2.version == "V2"
    assert v2.correct == v2.total == 9
    assert v2.component_scores["threads"] == {"correct": 2, "total": 2}

