from pathlib import Path

import pytest

import kernel_lab
from kernel_lab import KernelLabError, comparison_record, source_manifest, vm_plan


def test_source_requires_digest_and_signature_before_build_authority():
    unsigned = source_manifest("6.18.54", sha256="a" * 64, signature_verified=False)
    signed = source_manifest("6.18.54", sha256="a" * 64, signature_verified=True)
    assert unsigned["build_authorized"] is False
    assert signed["build_authorized"] is False
    assert signed["caller_claimed_signature_verified"] is True
    assert signed["host_install_authorized"] is False


def test_vm_is_offline_snapshot_tcg_and_workspace_confined(tmp_path, monkeypatch):
    monkeypatch.setattr(kernel_lab.shutil, "which", lambda _name: "/usr/bin/qemu-system-x86_64")
    for name in ("bzImage", "initramfs.img", "disk.qcow2"):
        (tmp_path / name).write_bytes(b"fixture")
    plan = vm_plan(tmp_path, kernel_image=tmp_path / "bzImage", initramfs=tmp_path / "initramfs.img", disk_image=tmp_path / "disk.qcow2")
    assert plan["network"] == "none"
    assert plan["disk_mode"] == "read_only_snapshot"
    assert plan["kvm"] is False
    assert "-sandbox" in plan["argv"]
    with pytest.raises(KernelLabError, match="leaves"):
        vm_plan(tmp_path, kernel_image=Path("/outside"), initramfs=tmp_path / "initramfs.img", disk_image=tmp_path / "disk.qcow2")


def test_os_comparison_never_collapses_or_promotes():
    result = comparison_record({"name": "Ina OS"}, [{"name": "Fedora"}, {"name": "Debian"}])
    assert result["single_score_authorized"] is False
    assert result["promotion_authorized"] is False
    assert set(result["required_signals"]) == {"functional", "security", "resource", "recovery", "reproducibility"}
