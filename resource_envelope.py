"""Kernel-backed resource envelope for the complete Ina runtime tree.

The cooperative scheduler is useful for graceful shedding, but it is not a
hard limit.  This module boots the top-level runtime in a systemd cgroup and
verifies the limits from cgroup v2 files.  Children inherit that cgroup, so a
new or unregistered subprocess cannot bypass the aggregate ceiling.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


DEFAULT_POLICY: Dict[str, Any] = {
    "enabled": True,
    "required": True,
    "ram_fraction": 0.5,
    "swap_fraction": 0.5,
    "unit_prefix": "ina-runtime",
}


def _fraction(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return min(1.0, max(0.01, number))


def load_policy(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if config is None:
        try:
            with Path("config.json").open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            config = loaded if isinstance(loaded, dict) else {}
        except Exception:
            config = {}
    raw = config.get("resource_envelope") if isinstance(config, dict) else None
    policy = dict(DEFAULT_POLICY)
    if isinstance(raw, dict):
        for key in policy:
            if key in raw:
                policy[key] = raw[key]
    policy["enabled"] = bool(policy["enabled"])
    policy["required"] = bool(policy["required"])
    policy["ram_fraction"] = _fraction(policy["ram_fraction"], 0.5)
    policy["swap_fraction"] = _fraction(policy["swap_fraction"], 0.5)
    prefix = "".join(ch for ch in str(policy["unit_prefix"]) if ch.isalnum() or ch in "-_")
    policy["unit_prefix"] = prefix or "ina-runtime"
    return policy


def system_memory_totals(meminfo_path: Path = Path("/proc/meminfo")) -> Dict[str, int]:
    values: Dict[str, int] = {}
    try:
        with meminfo_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                key, separator, rest = line.partition(":")
                if not separator or key not in {"MemTotal", "SwapTotal"}:
                    continue
                fields = rest.strip().split()
                if fields:
                    values[key] = max(0, int(fields[0])) * 1024
    except (OSError, ValueError):
        pass
    return {
        "ram_total_bytes": values.get("MemTotal", 0),
        "swap_total_bytes": values.get("SwapTotal", 0),
    }


def desired_limits(
    config: Optional[Dict[str, Any]] = None,
    *,
    totals: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    policy = load_policy(config)
    memory = totals or system_memory_totals()
    ram_total = max(0, int(memory.get("ram_total_bytes", 0)))
    swap_total = max(0, int(memory.get("swap_total_bytes", 0)))
    return {
        **policy,
        "ram_total_bytes": ram_total,
        "swap_total_bytes": swap_total,
        "ram_limit_bytes": int(ram_total * policy["ram_fraction"]),
        "swap_limit_bytes": int(swap_total * policy["swap_fraction"]),
    }


def _current_cgroup_relative(proc_cgroup_path: Path = Path("/proc/self/cgroup")) -> Optional[str]:
    try:
        for line in proc_cgroup_path.read_text(encoding="utf-8").splitlines():
            hierarchy, controllers, relative = line.split(":", 2)
            if hierarchy == "0" and controllers == "":
                return relative or "/"
    except (OSError, ValueError):
        return None
    return None


def _read_limit(path: Path) -> Optional[int]:
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if raw == "max":
        return None
    try:
        return max(0, int(raw))
    except ValueError:
        return None


def _read_counter(path: Path) -> Optional[int]:
    value = _read_limit(path)
    return value if value is not None else None


def cgroup_status(
    config: Optional[Dict[str, Any]] = None,
    *,
    cgroup_root: Path = Path("/sys/fs/cgroup"),
    proc_cgroup_path: Path = Path("/proc/self/cgroup"),
    totals: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    wanted = desired_limits(config, totals=totals)
    relative = _current_cgroup_relative(proc_cgroup_path)
    group = cgroup_root / str(relative or "").lstrip("/") if relative is not None else None
    ram_max = _read_limit(group / "memory.max") if group is not None else None
    swap_max = _read_limit(group / "memory.swap.max") if group is not None else None
    ram_current = _read_counter(group / "memory.current") if group is not None else None
    swap_current = _read_counter(group / "memory.swap.current") if group is not None else None
    ram_wanted = int(wanted["ram_limit_bytes"])
    swap_wanted = int(wanted["swap_limit_bytes"])
    ram_ok = ram_wanted > 0 and ram_max is not None and ram_max <= ram_wanted
    # A machine with no swap is correctly constrained with a zero swap limit.
    swap_ok = (swap_wanted == 0 and swap_max == 0) or (
        swap_wanted > 0 and swap_max is not None and swap_max <= swap_wanted
    )
    enforced = bool(wanted["enabled"] and relative is not None and ram_ok and swap_ok)
    return {
        **wanted,
        "platform": sys.platform,
        "cgroup_v2": bool((cgroup_root / "cgroup.controllers").exists()),
        "cgroup_path": str(group) if group is not None else None,
        "cgroup_relative_path": relative,
        "ram_current_bytes": ram_current,
        "swap_current_bytes": swap_current,
        "kernel_ram_limit_bytes": ram_max,
        "kernel_swap_limit_bytes": swap_max,
        "ram_limit_verified": ram_ok,
        "swap_limit_verified": swap_ok,
        "enforced": enforced,
        "verification": "verified" if enforced else "unverified",
    }


def systemd_scope_command(command: Iterable[str], limits: Dict[str, Any]) -> list[str]:
    unit = f"{limits['unit_prefix']}-{os.getpid()}.scope"
    return [
        "systemd-run",
        "--user",
        "--scope",
        "--quiet",
        "--collect",
        f"--unit={unit}",
        "--property=MemoryAccounting=yes",
        f"--property=MemoryMax={int(limits['ram_limit_bytes'])}",
        f"--property=MemorySwapMax={int(limits['swap_limit_bytes'])}",
        "--property=OOMPolicy=stop",
        *[str(part) for part in command],
    ]


def ensure_runtime_hard_limit(command: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    """Verify the current cgroup or re-exec the entry point in a strict scope.

    When ``required`` is true this fails closed: a runtime is never silently
    started with only the cooperative Python guard.
    """
    limits = desired_limits()
    if not limits["enabled"]:
        return {**limits, "enforced": False, "verification": "disabled"}
    if limits["ram_limit_bytes"] <= 0:
        raise RuntimeError("Cannot determine physical RAM; refusing an unenforced Ina runtime.")
    status = cgroup_status()
    if status["enforced"]:
        os.environ["INA_RESOURCE_ENVELOPE_VERIFIED"] = "1"
        return status

    already_attempted = os.environ.get("INA_RESOURCE_ENVELOPE_BOOTSTRAP") == "1"
    runner = shutil.which("systemd-run")
    if command is not None and runner and not already_attempted and sys.platform.startswith("linux"):
        os.environ["INA_RESOURCE_ENVELOPE_BOOTSTRAP"] = "1"
        argv = systemd_scope_command(command, limits)
        os.execv(runner, argv)

    reason = "systemd-run unavailable" if not runner else "kernel cgroup limits did not verify"
    status["reason"] = reason
    if limits["required"]:
        raise RuntimeError(
            f"Ina hard resource envelope is required but {reason}. "
            "Start from a user session with systemd cgroup v2 support."
        )
    return status
