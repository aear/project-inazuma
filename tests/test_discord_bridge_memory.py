import subprocess
import sys

from module_benchmarks import benchmark_module


def test_discord_bridge_import_does_not_load_model_manager():
    probe = subprocess.run(
        [sys.executable, "-c", "import discord_bridge,sys; print(int('model_manager' in sys.modules))"],
        check=True, capture_output=True, text=True, timeout=60,
    )
    assert probe.stdout.strip() == "0"


def test_discord_bridge_memory_benchmark_compares_versions():
    v1, v2 = benchmark_module("discord_bridge_memory")
    assert v2.accuracy > v1.accuracy
    assert set(v2.component_scores) == {"isolation", "state_reuse", "fallback", "guard"}
