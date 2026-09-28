"""Non-serializable authority and taint checks for executable actions.

External text can describe an action, but it cannot mint the process-local
capability required to queue that action.  Queue seals are deliberately lost
on restart so stale or manually injected state fails closed.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import secrets
from typing import Any, Mapping


class InstructionAuthorityError(PermissionError):
    pass


class _LocalCodeAuthority:
    __slots__ = ()


LOCAL_VOLUNTARY_CODE_AUTHORITY = _LocalCodeAuthority()
_SEAL_KEY = secrets.token_bytes(32)
_EXECUTABLE_ACTIONS = frozenset({"experiment_create", "experiment_run", "experiment_judge"})
_UNTRUSTED_SOURCES = frozenset({"discord", "discord_message", "external_research", "untrusted_external_data"})


def executable_action(action: Any) -> bool:
    return str(action or "").strip().lower() in _EXECUTABLE_ACTIONS


def reject_external_instruction(payload: Mapping[str, Any]) -> None:
    """Reject data whose provenance explicitly denies instruction authority."""
    if payload.get("instructions_authorized") is False:
        raise InstructionAuthorityError("external data is not authorized to instruct executable actions")
    source = str(payload.get("source") or payload.get("trust") or "").strip().lower()
    if source in _UNTRUSTED_SOURCES:
        raise InstructionAuthorityError(f"untrusted source cannot instruct executable actions: {source}")
    provenance = payload.get("provenance")
    if isinstance(provenance, (list, tuple, set)) and any(
        str(item).strip().lower() in _UNTRUSTED_SOURCES for item in provenance
    ):
        raise InstructionAuthorityError("external provenance cannot instruct executable actions")


def seal_code_command(command: Mapping[str, Any], authority: object | None) -> dict[str, Any]:
    payload = dict(command)
    if not executable_action(payload.get("action")):
        return payload
    reject_external_instruction(payload)
    if str(payload.get("source") or "").strip().lower() != "ina_voluntary_choice":
        raise InstructionAuthorityError("code actions require explicit Ina voluntary-choice provenance")
    if authority is not LOCAL_VOLUNTARY_CODE_AUTHORITY:
        raise InstructionAuthorityError("a local voluntary code capability is required")
    payload.pop("_code_authority_seal", None)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    payload["_code_authority_seal"] = hmac.new(_SEAL_KEY, canonical, hashlib.sha256).hexdigest()
    return payload


def verify_code_command(command: Mapping[str, Any]) -> None:
    if not executable_action(command.get("action")):
        return
    payload = dict(command)
    supplied = str(payload.pop("_code_authority_seal", ""))
    reject_external_instruction(payload)
    if str(payload.get("source") or "").strip().lower() != "ina_voluntary_choice":
        raise InstructionAuthorityError("code actions require explicit Ina voluntary-choice provenance")
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    expected = hmac.new(_SEAL_KEY, canonical, hashlib.sha256).hexdigest()
    if not supplied or not hmac.compare_digest(supplied, expected):
        raise InstructionAuthorityError("code command lacks a valid process-local authority seal")


__all__ = [
    "InstructionAuthorityError", "LOCAL_VOLUNTARY_CODE_AUTHORITY", "executable_action",
    "reject_external_instruction", "seal_code_command", "verify_code_command",
]
