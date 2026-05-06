"""Operator-facing permission request helpers.

Ina should never repair privileged storage permissions silently. These helpers
build explicit request payloads that can be surfaced to Sakura with exact
commands, scope, and risk before any sudo-level action is taken.
"""
from __future__ import annotations

import getpass
import grp
import hashlib
import os
import shlex
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from storage_layout import format_child_path, root_is_writable, storage_layout


OPERATOR_PERMISSION_KEY = "operator_permission_request"
FAST_RUNTIME_PERMISSION_TYPE = "fast_runtime_write_access"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _current_owner() -> str:
    user = getpass.getuser() or "sakura"
    try:
        group = grp.getgrgid(os.getgid()).gr_name
    except Exception:
        group = user
    return f"{user}:{group}"


def _request_id(child: str, fast_root: Path, runtime_root: Path) -> str:
    digest = hashlib.sha1(
        f"{child}|{fast_root}|{runtime_root}|{FAST_RUNTIME_PERMISSION_TYPE}".encode("utf-8")
    ).hexdigest()[:12]
    return f"operator_permission:{FAST_RUNTIME_PERMISSION_TYPE}:{digest}"


def _role(storage_vitals: Optional[Dict[str, Any]], role_name: str) -> Dict[str, Any]:
    if not isinstance(storage_vitals, dict):
        return {}
    roles = storage_vitals.get("roles")
    if not isinstance(roles, dict):
        return {}
    role_payload = roles.get(role_name)
    return role_payload if isinstance(role_payload, dict) else {}


def build_fast_runtime_write_request(
    child: str,
    config: Optional[Dict[str, Any]] = None,
    storage_vitals: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Return a pending operator request when the fast runtime root is blocked."""

    cfg = config if isinstance(config, dict) else {}
    layout = storage_layout(cfg)
    if not layout or not bool(layout.get("fast_runtime_enabled", True)):
        return None

    fast_root = format_child_path(layout.get("fast_root") or layout.get("hot_root"), child)
    runtime_root = format_child_path(
        layout.get("fast_runtime_root") or layout.get("fast_neural_root") or layout.get("fast_root"),
        child,
    )
    fast_mount = format_child_path(layout.get("fast_mount") or layout.get("hot_mount"), child)
    if fast_root is None or runtime_root is None:
        return None

    if root_is_writable(runtime_root):
        return None

    fast_role = _role(storage_vitals, "fast")
    runtime_role = _role(storage_vitals, "fast_runtime")
    mount_seen = fast_mount.exists() if fast_mount is not None else None
    mount_is_mount = fast_mount.is_mount() if fast_mount is not None and fast_mount.exists() else False

    owner = _current_owner()
    mkdir_command = (
        "sudo install -d -o "
        f"{shlex.quote(owner.split(':', 1)[0])} -g {shlex.quote(owner.split(':', 1)[1])} "
        f"-m 0755 {shlex.quote(str(fast_root))} {shlex.quote(str(runtime_root))}"
    )
    verify_command = f"test -w {shlex.quote(str(runtime_root))}"

    request = {
        "id": _request_id(str(child), fast_root, runtime_root),
        "request_type": FAST_RUNTIME_PERMISSION_TYPE,
        "status": "pending_operator_authorization",
        "approval_required": True,
        "auto_execute": False,
        "created_at": _now_iso(),
        "title": "Allow Ina to write rebuildable fast-runtime files on the NVME",
        "summary": (
            "The fast runtime path is configured but not writable, so rebuildable "
            "high-I/O files are staying on the HDD."
        ),
        "why": (
            "Ina can boot from durable HDD storage and use the Sabrent/NVME only for "
            "fast cache, spool, snapshot, and index files once this directory is owned "
            "by the current operator user."
        ),
        "risk_level": "medium",
        "risk_notes": [
            "Requires sudo because the NVME mount root is not writable by the current user.",
            "Creates/chowns only Ina's configured fast storage directories.",
            "Does not format, delete, recursively chown the whole drive, or move durable memory.",
        ],
        "target": {
            "child": str(child),
            "device": layout.get("fast_device") or layout.get("hot_device"),
            "mount": str(fast_mount) if fast_mount is not None else None,
            "fast_root": str(fast_root),
            "runtime_root": str(runtime_root),
            "owner": owner,
        },
        "detected": {
            "mount_path_exists": mount_seen,
            "mount_is_mount": mount_is_mount,
            "fast_role_writable": fast_role.get("writable"),
            "runtime_role_writable": runtime_role.get("writable"),
            "runtime_stat_path": runtime_role.get("stat_path"),
            "fast_mount_source": (fast_role.get("mount") or {}).get("source")
            if isinstance(fast_role.get("mount"), dict) else None,
            "fast_mount_fstype": (fast_role.get("mount") or {}).get("fstype")
            if isinstance(fast_role.get("mount"), dict) else None,
        },
        "commands": [
            {
                "label": "Create and assign Ina's fast-runtime directory",
                "command": mkdir_command,
                "requires_sudo": True,
                "purpose": "Prepare only the configured Ina_Fast_Storage paths on the NVME.",
            },
            {
                "label": "Verify Ina can write there",
                "command": verify_command,
                "requires_sudo": False,
                "purpose": "Confirm the runtime root is writable before Ina uses it.",
            },
        ],
        "feedback": {
            "channel": "inastate",
            "response_path": "operator_permission_request.operator_response",
            "history_key": "operator_permission_feedback_history",
            "prompt": "Approve or deny this permission request and say why.",
            "reason_required": True,
            "decisions": [
                {
                    "value": "approved",
                    "label": "Approve",
                    "result_status": "approved_pending_manual_execution",
                    "meaning": "The operator accepts the command scope and may run it manually.",
                },
                {
                    "value": "denied",
                    "label": "Deny",
                    "result_status": "denied_by_operator",
                    "meaning": "Ina should keep using the HDD fallback and not ask again for this exact request.",
                },
            ],
        },
        "operator_response": {
            "decision": None,
            "approved": None,
            "reason": "",
            "responded_at": None,
            "responded_by": None,
            "instructions": "Run the sudo command yourself if you approve, then let Ina resample storage vitals.",
        },
    }
    if fast_mount is not None and mount_seen and not mount_is_mount:
        request["risk_notes"].append(
            "The configured fast mount path exists but is not currently a mountpoint; verify the NVME is mounted before approving."
        )
    return request


def attach_storage_permission_requests(
    storage_vitals: Dict[str, Any],
    child: str,
    config: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Attach pending storage permission requests to a storage vitals payload."""

    request = build_fast_runtime_write_request(child, config, storage_vitals)
    if not request:
        return None
    storage_vitals["operator_permission_requests"] = [request]
    summary = storage_vitals.setdefault("summary", {})
    if isinstance(summary, dict):
        summary["pending_operator_permission_count"] = 1
        summary["pending_operator_permission_types"] = [request["request_type"]]
    return request


__all__ = [
    "FAST_RUNTIME_PERMISSION_TYPE",
    "OPERATOR_PERMISSION_KEY",
    "attach_storage_permission_requests",
    "build_fast_runtime_write_request",
]
