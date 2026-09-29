"""Reproducible, offline-by-default Linux kernel and VM experiment plans."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import hmac
import secrets
import lzma
from typing import Any, Mapping


class KernelLabError(RuntimeError):
    pass


_VERIFICATION_KEY = secrets.token_bytes(32)


def verify_kernel_signature(source: Path | str, signature: Path | str, *,
                            keyring: Path | str, trusted_fingerprints: tuple[str, ...]) -> dict[str, Any]:
    """Verify the uncompressed tar signature with an operator-pinned keyring.

    No key downloads, trust-on-first-use, shell commands, or live keyring writes.
    The returned process-local receipt binds the compressed source digest too.
    """
    trusted = {str(item).upper() for item in trusted_fingerprints}
    if not trusted or any(not re.fullmatch(r"[0-9A-F]{40,64}", item) for item in trusted):
        raise KernelLabError("explicit trusted signing fingerprints are required")
    source = Path(source)
    with tempfile.TemporaryDirectory(prefix="ina_kernel_verify_") as directory:
        snapshot = Path(directory) / "source"
        digest = hashlib.sha256()
        size = 0
        with source.open("rb") as reader, snapshot.open("wb") as writer:
            while chunk := reader.read(1024 * 1024):
                size += len(chunk)
                if size > 512 * 1024 * 1024:
                    raise KernelLabError("source exceeds verification byte budget")
                digest.update(chunk)
                writer.write(chunk)
        tar = snapshot
        if source.suffix == ".xz":
            tar = Path(directory) / "source.tar"
            size = 0
            with lzma.open(snapshot, "rb") as reader, tar.open("wb") as writer:
                while chunk := reader.read(1024 * 1024):
                    size += len(chunk)
                    if size > 2 * 1024 * 1024 * 1024:
                        raise KernelLabError("expanded source exceeds verification byte budget")
                    writer.write(chunk)
        result = subprocess.run(["gpgv", "--homedir", directory, "--status-fd", "1",
                                 "--keyring", str(Path(keyring).resolve()),
                                 str(Path(signature).resolve()), str(tar)],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60, check=False)
        signers = []
        for line in result.stdout.decode("ascii", "replace").splitlines():
            fields = line.split()
            if fields[:2] == ["[GNUPG:]", "VALIDSIG"] and len(fields) >= 12:
                signers.append((fields[2], fields[-1]))
        if result.returncode or not any(primary in trusted or signer in trusted for signer, primary in signers):
            raise KernelLabError("cryptographic signature verification failed or signer is not pinned")
        receipt = {"sha256": digest.hexdigest(), "signers": signers, "verifier": "gpgv"}
        receipt["seal"] = hmac.new(_VERIFICATION_KEY, json.dumps(receipt, sort_keys=True).encode(), hashlib.sha256).hexdigest()
        return receipt


def source_manifest(version: str, *, sha256: str, signature_verified: bool = False,
                    verification: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", str(version)):
        raise KernelLabError("kernel version must be a full stable release")
    digest = str(sha256 or "").lower()
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise KernelLabError("kernel source requires a SHA-256 digest")
    series = version.split(".")[:2]
    filename = f"linux-{version}.tar.xz"
    receipt = dict(verification or {})
    seal = str(receipt.pop("seal", ""))
    verified = bool(receipt) and receipt.get("sha256") == digest and hmac.compare_digest(
        seal, hmac.new(_VERIFICATION_KEY, json.dumps(receipt, sort_keys=True).encode(), hashlib.sha256).hexdigest())
    return {
        "schema": "ina.kernel_source_manifest/V2", "version": version,
        "release_line": ".".join(series), "source_kind": "kernel.org_release",
        "source_url": f"https://cdn.kernel.org/pub/linux/kernel/v{series[0]}.x/{filename}",
        "signature_url": f"https://cdn.kernel.org/pub/linux/kernel/v{series[0]}.x/linux-{version}.tar.sign",
        "filename": filename, "sha256": digest, "signature_verified": verified,
        "caller_claimed_signature_verified": bool(signature_verified),
        "verification_method": "gpgv_pinned_signer" if verified else "unverified",
        "build_authorized": verified, "host_install_authorized": False,
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
        if not path.is_file() or "," in str(path):
            raise KernelLabError(f"{name} requires an existing regular file without QEMU option delimiters")
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
