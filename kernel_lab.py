"""Reproducible, offline-by-default Linux kernel and VM experiment plans."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import shutil
from typing import Any, Mapping


class KernelLabError(RuntimeError):
    pass


def source_manifest(version: str, *, sha256: str, signature_verified: bool) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", str(version)):
        raise KernelLabError("kernel version must be a full stable release")
    digest = str(sha256 or "").lower()
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise KernelLabError("kernel source requires a SHA-256 digest")
    series = version.split(".")[:2]
    filename = f"linux-{version}.tar.xz"
    return {
        "schema": "ina.kernel_source_manifest/V1", "version": version,
        "release_line": ".".join(series), "source_kind": "kernel.org_longterm",
        "source_url": f"https://cdn.kernel.org/pub/linux/kernel/v{series[0]}.x/{filename}",
        "signature_url": f"https://cdn.kernel.org/pub/linux/kernel/v{series[0]}.x/linux-{version}.tar.sign",
        "filename": filename, "sha256": digest, "signature_verified": bool(signature_verified),
        "build_authorized": bool(signature_verified), "host_install_authorized": False,
    }


def vm_plan(
    workspace: Path | str, *, kernel_image: Path | str, initramfs: Path | str,
    disk_image: Path | str, memory_mib: int = 2048, cpus: int = 2,
) -> dict[str, Any]:
    root = Path(workspace).resolve()
    qemu = shutil.which("qemu-system-x86_64")
    inputs = {name: Path(value).resolve() for name, value in {
        "kernel": kernel_image, "initramfs": initramfs, "disk": disk_image,
    }.items()}
    for name, path in inputs.items():
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise KernelLabError(f"{name} leaves the kernel lab workspace") from exc
    bounded_memory = max(512, min(4096, int(memory_mib)))
    bounded_cpus = max(1, min(4, int(cpus)))
    args = [] if not qemu else [
        qemu, "-machine", "q35,accel=tcg", "-cpu", "max", "-m", str(bounded_memory),
        "-smp", str(bounded_cpus), "-kernel", str(inputs["kernel"]),
        "-initrd", str(inputs["initramfs"]), "-drive", f"file={inputs['disk']},if=virtio,format=qcow2,readonly=on",
        "-append", "console=ttyS0 panic=1 oops=panic", "-nic", "none", "-snapshot",
        "-sandbox", "on,obsolete=deny,elevateprivileges=deny,spawn=deny,resourcecontrol=deny",
        "-nodefaults", "-no-reboot", "-nographic",
    ]
    return {
        "schema": "ina.kernel_vm_plan/V1", "workspace": str(root),
        "available": qemu is not None, "unavailable_reason": None if qemu else "qemu-system-x86_64 is not installed",
        "argv": args, "acceleration": "tcg", "network": "none", "disk_mode": "read_only_snapshot",
        "host_shares": [], "usb_passthrough": False, "kvm": False,
        "memory_mib": bounded_memory, "cpus": bounded_cpus,
        "production_boot_or_install_authorized": False,
    }


def comparison_record(candidate: Mapping[str, Any], baselines: list[Mapping[str, Any]]) -> dict[str, Any]:
    dimensions = (
        "boot_success", "test_failures", "known_vulnerabilities", "attack_surface",
        "default_services", "syscall_surface", "memory_mib", "boot_seconds",
        "reproducible_build", "update_latency", "recovery_success",
    )
    return {
        "schema": "ina.secure_os_comparison/V1", "candidate": dict(candidate),
        "baselines": [dict(row) for row in baselines[:12]], "dimensions": list(dimensions),
        "single_score_authorized": False, "promotion_authorized": False,
        "required_signals": ["functional", "security", "resource", "recovery", "reproducibility"],
    }


def write_manifest(path: Path | str, payload: Mapping[str, Any]) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(dict(payload), indent=2, sort_keys=True) + "\n"
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(destination)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


__all__ = ["KernelLabError", "comparison_record", "source_manifest", "vm_plan", "write_manifest"]
