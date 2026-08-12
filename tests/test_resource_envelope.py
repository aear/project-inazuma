from pathlib import Path

import resource_envelope as re


def test_desired_limits_are_half_ram_and_half_swap(tmp_path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal:       1024 kB\nSwapTotal:      4096 kB\n", encoding="utf-8")
    totals = re.system_memory_totals(meminfo)
    limits = re.desired_limits(
        {"resource_envelope": {"ram_fraction": 0.5, "swap_fraction": 0.5}},
        totals=totals,
    )
    assert limits["ram_limit_bytes"] == 512 * 1024
    assert limits["swap_limit_bytes"] == 2048 * 1024


def test_cgroup_status_verifies_actual_kernel_files(tmp_path):
    root = tmp_path / "cgroup"
    group = root / "ina.scope"
    group.mkdir(parents=True)
    (root / "cgroup.controllers").write_text("memory cpu\n", encoding="utf-8")
    (group / "memory.max").write_text("500\n", encoding="utf-8")
    (group / "memory.swap.max").write_text("1000\n", encoding="utf-8")
    (group / "memory.current").write_text("123\n", encoding="utf-8")
    (group / "memory.swap.current").write_text("45\n", encoding="utf-8")
    proc = tmp_path / "self.cgroup"
    proc.write_text("0::/ina.scope\n", encoding="utf-8")
    config = {"resource_envelope": {"ram_fraction": 0.5, "swap_fraction": 0.5}}
    status = re.cgroup_status(
        config,
        cgroup_root=root,
        proc_cgroup_path=proc,
        totals={"ram_total_bytes": 1000, "swap_total_bytes": 2000},
    )
    assert status["enforced"] is True
    assert status["ram_current_bytes"] == 123
    assert status["swap_current_bytes"] == 45
    assert status["kernel_ram_limit_bytes"] == 500
    assert status["kernel_swap_limit_bytes"] == 1000


def test_service_command_sets_independent_ram_and_swap_limits(monkeypatch):
    monkeypatch.setattr(re.os, "getpid", lambda: 42)
    command = re.systemd_service_command(
        ["python", "GUI.py"],
        {
            "unit_prefix": "ina-runtime",
            "ram_limit_bytes": 100,
            "swap_limit_bytes": 250,
        },
    )
    assert "--unit=ina-runtime-42.service" in command
    assert "--scope" not in command
    assert "--service-type=exec" in command
    assert "--same-dir" in command
    assert "--property=MemoryMax=100" in command
    assert "--property=MemorySwapMax=250" in command
    assert command[-2:] == ["python", "GUI.py"]
