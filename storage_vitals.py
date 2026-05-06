"""Storage capacity and directory-index telemetry for Ina.

The kernel exposes free inode counts through statvfs for fixed-inode
filesystems such as XFS. Some filesystems, notably btrfs, allocate inode-like
metadata dynamically and report zero total/free inodes, so the payload names
that model explicitly instead of pretending there is a numeric counter.
"""
from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

from storage_layout import format_child_path, load_config, root_is_writable, storage_layout


DEFAULT_SAFE_DIRECTORY_ENTRY_LIMIT = 50000


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return default


def _safe_ratio(used: Optional[int], total: Optional[int]) -> Optional[float]:
    if not total or used is None:
        return None
    return round(max(0.0, min(1.0, float(used) / float(total))), 6)


def _existing_probe_path(path: Path) -> Path:
    probe = Path(path)
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return probe if probe.exists() else Path(".")


def _findmnt(path: Path) -> Dict[str, Any]:
    try:
        result = subprocess.run(
            ["findmnt", "-J", "-T", str(path), "-o", "TARGET,SOURCE,FSTYPE,OPTIONS"],
            capture_output=True,
            text=True,
            check=False,
            timeout=3,
        )
    except Exception:
        return {}
    if result.returncode != 0 or not result.stdout.strip():
        return {}
    try:
        payload = json.loads(result.stdout)
    except Exception:
        return {}
    filesystems = payload.get("filesystems") if isinstance(payload, dict) else None
    if not filesystems or not isinstance(filesystems, list) or not isinstance(filesystems[0], dict):
        return {}
    fs = filesystems[0]
    return {
        "target": fs.get("target"),
        "source": fs.get("source"),
        "fstype": fs.get("fstype"),
        "options": fs.get("options"),
    }


def _statvfs(path: Path) -> Dict[str, Any]:
    st = os.statvfs(path)
    block_size = int(st.f_frsize or st.f_bsize or 0)
    total_bytes = int(st.f_blocks * block_size) if block_size else None
    free_bytes = int(st.f_bfree * block_size) if block_size else None
    available_bytes = int(st.f_bavail * block_size) if block_size else None
    used_bytes = None
    if total_bytes is not None and free_bytes is not None:
        used_bytes = max(0, total_bytes - free_bytes)

    total_inodes = int(st.f_files) if int(st.f_files or 0) > 0 else None
    free_inodes = int(st.f_ffree) if int(st.f_ffree or 0) > 0 else None
    available_inodes = int(st.f_favail) if int(st.f_favail or 0) > 0 else None
    used_inodes = None
    if total_inodes is not None and free_inodes is not None:
        used_inodes = max(0, total_inodes - free_inodes)

    return {
        "block_size": block_size,
        "name_max": int(st.f_namemax),
        "total_bytes": total_bytes,
        "free_bytes": free_bytes,
        "available_bytes": available_bytes,
        "used_bytes": used_bytes,
        "used_ratio": _safe_ratio(used_bytes, total_bytes),
        "total_inodes": total_inodes,
        "free_inodes": free_inodes,
        "free_inodes_available": available_inodes,
        "used_inodes": used_inodes,
        "inode_used_ratio": _safe_ratio(used_inodes, total_inodes),
        "inode_model": "fixed" if total_inodes is not None else "dynamic_or_not_reported",
    }


def _directory_index_profile(
    fstype: str,
    options: str,
    stat: Dict[str, Any],
    *,
    safe_limit: int,
) -> Dict[str, Any]:
    fs = str(fstype or "unknown").strip().lower()
    free_inodes = _safe_int(stat.get("free_inodes_available"))
    options_text = str(options or "")

    if fs == "xfs":
        return {
            "index_type": "btree",
            "reported_hard_entry_limit": free_inodes,
            "limit_basis": "bounded by available inodes, free space, and XFS metadata; no smaller per-directory index ceiling is exposed by statvfs/findmnt",
            "limit_confidence": "bounded_estimate" if free_inodes is not None else "not_reported",
            "safe_shard_entry_limit": safe_limit,
            "sharding_recommended": True,
        }
    if fs == "btrfs":
        return {
            "index_type": "btree",
            "reported_hard_entry_limit": None,
            "limit_basis": "btrfs uses dynamic metadata and reports no fixed inode or per-directory index entry limit here; practical limit is metadata/free-space bounded",
            "limit_confidence": "not_exposed_by_filesystem",
            "safe_shard_entry_limit": safe_limit,
            "sharding_recommended": True,
        }
    if fs in {"ext2", "ext3", "ext4"}:
        large_dir_hint = "large_dir" in options_text
        estimated = 2000000000 if large_dir_hint else 10000000
        if free_inodes is not None:
            estimated = min(estimated, free_inodes)
        return {
            "index_type": "htree" if fs == "ext4" else "linear_or_htree",
            "reported_hard_entry_limit": estimated,
            "limit_basis": "ext directory index capacity estimate; exact ceiling depends on block size, filename length, and filesystem features",
            "limit_confidence": "estimate",
            "safe_shard_entry_limit": safe_limit,
            "sharding_recommended": True,
        }
    return {
        "index_type": "unknown",
        "reported_hard_entry_limit": free_inodes,
        "limit_basis": "filesystem-specific directory index limit is not exposed; using available inodes when reported",
        "limit_confidence": "generic" if free_inodes is not None else "not_reported",
        "safe_shard_entry_limit": safe_limit,
        "sharding_recommended": True,
    }


def _path_sample(role: str, raw_path: Any, child: str, safe_limit: int) -> Optional[Dict[str, Any]]:
    path = format_child_path(raw_path, child)
    if path is None:
        return None
    probe = _existing_probe_path(path)
    try:
        stat = _statvfs(probe)
    except Exception as exc:
        return {
            "role": role,
            "path": str(path),
            "stat_path": str(probe),
            "available": False,
            "error": str(exc),
        }
    mount = _findmnt(probe)
    directory_index = _directory_index_profile(
        str(mount.get("fstype") or "unknown"),
        str(mount.get("options") or ""),
        stat,
        safe_limit=safe_limit,
    )
    return {
        "role": role,
        "path": str(path),
        "path_exists": path.exists(),
        "stat_path": str(probe),
        "writable": root_is_writable(path),
        "available": True,
        "mount": mount,
        "bytes": {
            "total": stat.get("total_bytes"),
            "free": stat.get("free_bytes"),
            "available": stat.get("available_bytes"),
            "used": stat.get("used_bytes"),
            "used_ratio": stat.get("used_ratio"),
        },
        "inodes": {
            "model": stat.get("inode_model"),
            "total": stat.get("total_inodes"),
            "free": stat.get("free_inodes"),
            "available": stat.get("free_inodes_available"),
            "used": stat.get("used_inodes"),
            "used_ratio": stat.get("inode_used_ratio"),
        },
        "directory_index": directory_index,
        "name_max": stat.get("name_max"),
        "block_size": stat.get("block_size"),
    }


def _role_paths(cfg: Dict[str, Any], child: str) -> Iterable[Tuple[str, Any]]:
    layout = storage_layout(cfg)
    cold_policy = cfg.get("cold_storage_policy") if isinstance(cfg, dict) else None
    yield "project", layout.get("durable_project_root") or Path.cwd()
    yield "durable", layout.get("durable_mount") or layout.get("cold_mount")
    yield "fast", layout.get("fast_mount") or layout.get("hot_mount")
    yield "cold_storage", (cold_policy or {}).get("storage_root") if isinstance(cold_policy, dict) else layout.get("cold_storage_root")
    yield "fast_runtime", layout.get("fast_runtime_root") or layout.get("fast_root")


def sample_storage_vitals(
    child: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cfg = config if isinstance(config, dict) else load_config()
    current_child = child or str(cfg.get("current_child") or "Inazuma_Yagami")
    layout = storage_layout(cfg)
    try:
        safe_limit = int(layout.get("directory_entry_soft_limit") or DEFAULT_SAFE_DIRECTORY_ENTRY_LIMIT)
    except Exception:
        safe_limit = DEFAULT_SAFE_DIRECTORY_ENTRY_LIMIT
    safe_limit = max(1000, safe_limit)

    roles: Dict[str, Dict[str, Any]] = {}
    for role, raw_path in _role_paths(cfg, current_child):
        if raw_path is None:
            continue
        sample = _path_sample(role, raw_path, current_child, safe_limit)
        if sample is not None:
            roles[role] = sample

    numeric_free = [
        sample.get("inodes", {}).get("available")
        for sample in roles.values()
        if isinstance(sample.get("inodes"), dict) and isinstance(sample.get("inodes", {}).get("available"), int)
    ]
    fixed_limits = [
        sample.get("directory_index", {}).get("reported_hard_entry_limit")
        for sample in roles.values()
        if isinstance(sample.get("directory_index"), dict) and isinstance(sample.get("directory_index", {}).get("reported_hard_entry_limit"), int)
    ]

    return {
        "available": bool(roles),
        "updated_at": _now_iso(),
        "child": current_child,
        "directory_entry_soft_limit": safe_limit,
        "summary": {
            "min_free_inodes_available": min(numeric_free) if numeric_free else None,
            "min_reported_directory_index_limit": min(fixed_limits) if fixed_limits else None,
            "dynamic_inode_roles": [
                role for role, sample in roles.items()
                if sample.get("inodes", {}).get("model") != "fixed"
            ],
            "sharding_recommended": True,
        },
        "roles": roles,
    }


__all__ = ["sample_storage_vitals"]
