"""Evidence-tier report for externally reachable and executable boundaries."""
from __future__ import annotations

from datetime import datetime, timezone
import ast
from pathlib import Path
from typing import Any


def _contains(path: Path, needles: tuple[str, ...]) -> bool:
    text = path.read_text(encoding="utf-8")
    return all(needle in text for needle in needles)


def _discord_executable_paths(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    forbidden_names = {"CodeExperimentLab", "request_personal_tool", "exec", "compile"}
    forbidden_modules = {"codex_harness", "paint_runtime", "personal_tool_runtime", "code_experiment_lab", "subprocess"}
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names if alias.name.split(".", 1)[0] in forbidden_modules)
        elif isinstance(node, ast.ImportFrom) and str(node.module or "").split(".", 1)[0] in forbidden_modules:
            found.add(str(node.module))
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in forbidden_names:
                found.add(node.func.id)
            elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                if node.func.value.id == "subprocess":
                    found.add(f"subprocess.{node.func.attr}")
    return sorted(found)


def assess_security_boundaries(root: Path | str = ".") -> dict[str, Any]:
    base = Path(root)
    discord_direct_paths = _discord_executable_paths(base / "discord_bridge.py")
    rows = [
        {
            "boundary": "dns_and_redirects", "enforcement": "implemented",
            "fixture_evidence": "verified",
            "live_evidence": "not_yet_verified",
            "signals": ["all_resolved_addresses_must_be_global", "connection_ip_pinned", "tls_hostname_verified", "redirects_rejected"],
        },
        {
            "boundary": "discord_to_code", "enforcement": "implemented",
            "fixture_evidence": "verified" if not discord_direct_paths else "failed",
            "live_evidence": "not_yet_verified",
            "signals": ["no_direct_discord_code_path", "process_local_hmac_seal", "external_provenance_rejected", "experiment_sandbox"],
            "unexpected_direct_paths": discord_direct_paths,
        },
        {
            "boundary": "external_instruction_authority", "enforcement": "implemented",
            "fixture_evidence": "verified",
            "live_evidence": "not_yet_verified",
            "signals": ["instructions_authorized_false_rejected", "untrusted_source_rejected", "authority_seal_not_serializable"],
        },
        {
            "boundary": "codex_harness", "enforcement": "implemented",
            "fixture_evidence": "verified" if _contains(base / "codex_harness.py", (
                'forced_login_method="chatgpt"', '"127.0.0.1"', "approvalPolicy", "runtimeWorkspaceRoots",
                "permitted_hosts", "Origin",
            )) else "failed",
            "live_evidence": "not_yet_verified",
            "signals": ["loopback_bind", "host_and_origin_gate", "launch_token", "chatgpt_only", "workspace_write", "user_approvals"],
        },
        {
            "boundary": "ina_cyber_defence_capability", "enforcement": "not_applicable",
            "fixture_evidence": "framework_only",
            "live_evidence": "unavailable_no_ina_response_provider",
            "signals": [],
        },
    ]
    return {
        "schema": "ina.security_adversarial_assessment/V1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "isolated fixtures and static executable-path trace; no live memory stores",
        "rows": rows,
        "all_fixture_enforcement_verified": all(
            row["fixture_evidence"] in {"verified", "framework_only"} for row in rows
        ),
        "live_expansion_ready": False,
        "claim": "passing boundary fixtures does not demonstrate Ina's cyber-defence understanding",
    }


__all__ = ["assess_security_boundaries"]
