from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from github_submission import get_current_child, get_github_submission_config, load_config

INCIDENT_LOG_FILENAME = "self_read_incidents.jsonl"
INCIDENT_STATE_FILENAME = "self_read_incident_state.json"
BROKEN_PIPE_COOLDOWN_MINUTES = 180


def self_read_incident_log_path(child: str) -> Path:
    return Path("AI_Children") / child / "memory" / INCIDENT_LOG_FILENAME


def self_read_incident_state_path(child: str) -> Path:
    return Path("AI_Children") / child / "memory" / INCIDENT_STATE_FILENAME


def is_broken_pipe_error(error: Optional[BaseException] = None, message: Optional[str] = None) -> bool:
    if isinstance(error, BrokenPipeError):
        return True
    parts = [str(error or ""), str(message or "")]
    text = " ".join(part for part in parts if part).strip().lower()
    return "broken pipe" in text or "errno 32" in text


def explain_self_read_broken_pipe(component: str, operation: str) -> str:
    base = "Broken pipe means Ina wrote to a pipe or FIFO after the reader had already closed it."
    component_key = str(component or "").strip().lower()
    operation_key = str(operation or "").strip().lower()
    if component_key == "status_pipe":
        return (
            f"{base} In this self-read pass the likeliest source is the GUI status pipe disconnecting "
            f"while `{operation_key or 'status_log_write'}` was still streaming updates, so the scan can continue "
            "but live status output is no longer attached."
        )
    if component_key == "training_pipeline":
        return (
            f"{base} In this self-read pass that points to the training-side consumer disappearing "
            f"while `{operation_key or 'training'}` was still writing."
        )
    return (
        f"{base} In this self-read pass that usually points to a disconnected helper or logging consumer "
        f"during `{operation_key or 'self_read'}`, not unreadable source material."
    )


def _memory_path(child: str) -> Path:
    return Path("AI_Children") / child / "memory"


def _load_incident_state(child: str) -> Dict[str, Any]:
    path = self_read_incident_state_path(child)
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _save_incident_state(child: str, payload: Dict[str, Any]) -> None:
    path = self_read_incident_state_path(child)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
    except Exception:
        pass


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _parse_timestamp(value: Any) -> Optional[datetime]:
    try:
        stamp = datetime.fromisoformat(str(value))
    except Exception:
        return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp


def _incident_fingerprint(component: str, operation: str, error_text: str) -> str:
    basis = "|".join(
        [
            "self_read_broken_pipe",
            str(component or "").strip().lower(),
            str(operation or "").strip().lower(),
            str(error_text or "").strip().lower(),
        ]
    )
    return hashlib.sha1(basis.encode("utf-8")).hexdigest()[:16]


def _queue_broken_pipe_issue(
    *,
    child: str,
    component: str,
    operation: str,
    error_text: str,
    explanation: str,
    path_text: Optional[str],
    source_message: Optional[str],
    fingerprint: str,
) -> Optional[str]:
    cfg = load_config()
    policy = get_github_submission_config(cfg)
    if not bool(policy.get("enabled", False)):
        return None

    from model_manager import queue_github_submission

    component_key = str(component or "").strip().lower() or "self_read"
    operation_key = str(operation or "").strip().lower() or "self_read"
    title = f"Self-read broken pipe while {operation_key.replace('_', ' ')}"
    summary_lines = [
        f"A self-read pass for `{child}` hit a broken pipe in `{component_key}` while `{operation_key}` was active.",
        explanation,
    ]
    if path_text:
        summary_lines.append(f"Related path: `{path_text}`.")
    summary = " ".join(line for line in summary_lines if line)
    source_preview = str(source_message or "").strip()
    if len(source_preview) > 280:
        source_preview = source_preview[:277].rstrip() + "..."

    evidence = [
        f"component={component_key}",
        f"operation={operation_key}",
        f"error={error_text}",
        f"incident_fingerprint={fingerprint}",
    ]
    if path_text:
        evidence.append(f"path={path_text}")
    if source_preview:
        evidence.append(f"status_message={source_preview}")

    touched_files = ["raw_file_manager.py"]
    if component_key == "status_pipe":
        touched_files = ["gui_hook.py", "GUI.py", "raw_file_manager.py"]

    review_notes = [
        "Verify whether the status pipe reader or another IPC consumer disconnected during self-read.",
        "Keep transient pipe disconnects from looking like source-read failures.",
    ]

    return queue_github_submission(
        title,
        summary,
        kind="request",
        submission_mode="explain",
        suggestion=(
            "Reconnect or harden the pipe consumer, and downgrade repeated broken-pipe noise into a clearer "
            "self-read diagnostic that distinguishes UI disconnects from scan failures."
        ),
        evidence=evidence,
        touched_files=touched_files,
        review_notes=review_notes,
        confidence=0.63,
        metadata={
            "source": "self_read_broken_pipe",
            "component": component_key,
            "operation": operation_key,
            "incident_fingerprint": fingerprint,
            "notify_discord_on_submit": True,
            "discord_reason": "self_read_broken_pipe",
            "discord_explanation": explanation,
        },
    )


def report_self_read_broken_pipe(
    *,
    component: str,
    operation: str,
    error: Optional[BaseException] = None,
    source_message: Optional[str] = None,
    path_text: Optional[str] = None,
    child: Optional[str] = None,
) -> Dict[str, Any]:
    if not is_broken_pipe_error(error=error, message=source_message):
        return {"reported": False, "reason": "not_broken_pipe"}

    cfg = load_config()
    child_name = str(child or get_current_child(cfg) or "Inazuma_Yagami")
    error_text = str(error or source_message or "broken pipe").strip() or "broken pipe"
    explanation = explain_self_read_broken_pipe(component, operation)
    fingerprint = _incident_fingerprint(component, operation, error_text)
    now = datetime.now(timezone.utc)
    state = _load_incident_state(child_name)
    incidents = state.get("broken_pipe") if isinstance(state.get("broken_pipe"), dict) else {}
    prior = incidents.get(fingerprint) if isinstance(incidents, dict) else None

    should_queue_issue = True
    if isinstance(prior, dict):
        prior_ts = _parse_timestamp(prior.get("last_reported_at"))
        if prior_ts and (now - prior_ts) < timedelta(minutes=BROKEN_PIPE_COOLDOWN_MINUTES):
            should_queue_issue = False

    issue_entry_id = None
    if should_queue_issue:
        issue_entry_id = _queue_broken_pipe_issue(
            child=child_name,
            component=component,
            operation=operation,
            error_text=error_text,
            explanation=explanation,
            path_text=path_text,
            source_message=source_message,
            fingerprint=fingerprint,
        )

    incident = {
        "kind": "self_read_broken_pipe",
        "child": child_name,
        "component": str(component or "").strip().lower() or "self_read",
        "operation": str(operation or "").strip().lower() or "self_read",
        "error": error_text,
        "source_message": str(source_message or "").strip() or None,
        "path": str(path_text or "").strip() or None,
        "fingerprint": fingerprint,
        "explanation": explanation,
        "github_entry_id": issue_entry_id,
        "duplicate_within_cooldown": not should_queue_issue,
        "timestamp": now.isoformat(),
    }
    _append_jsonl(self_read_incident_log_path(child_name), incident)

    incidents[fingerprint] = {
        "last_reported_at": now.isoformat(),
        "last_issue_entry_id": issue_entry_id or (prior.get("last_issue_entry_id") if isinstance(prior, dict) else None),
        "component": incident["component"],
        "operation": incident["operation"],
        "error": error_text,
    }
    state["broken_pipe"] = incidents
    _save_incident_state(child_name, state)

    return {
        "reported": True,
        "child": child_name,
        "fingerprint": fingerprint,
        "issue_entry_id": issue_entry_id,
        "explanation": explanation,
        "duplicate_within_cooldown": not should_queue_issue,
        "incident_log_path": str(self_read_incident_log_path(child_name)),
        "memory_path": str(_memory_path(child_name)),
    }


__all__ = [
    "BROKEN_PIPE_COOLDOWN_MINUTES",
    "INCIDENT_LOG_FILENAME",
    "INCIDENT_STATE_FILENAME",
    "explain_self_read_broken_pipe",
    "is_broken_pipe_error",
    "report_self_read_broken_pipe",
    "self_read_incident_log_path",
    "self_read_incident_state_path",
]
