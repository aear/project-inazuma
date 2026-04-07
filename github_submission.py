from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set
from urllib import error as urlerror
from urllib import request as urlrequest

DEFAULT_GITHUB_SUBMISSION: Dict[str, Any] = {
    "enabled": False,
    "delivery_mode": "queue_only",  # queue_only | issues
    "repo_full_name": "",
    "api_base": "https://api.github.com",
    "token_env": "GITHUB_TOKEN",
    "issue_title_prefix": "[Ina]",
    "labels": ["ina-suggestion", "needs-review"],
    "optimization_labels": ["ina-suggestion", "optimization", "needs-review"],
    "feature_labels": ["ina-suggestion", "feature-request", "needs-review"],
    "max_batch": 2,
    "max_age_minutes": 1440,
    "poll_interval_sec": 60.0,
    "daily_issue_cap": 4,
    "cooldown_minutes": 180,
    "min_resource_trend_pressure": 0.74,
    "max_patch_excerpt_chars": 4000,
    "max_body_chars": 12000,
    "auto_submit_optimization_requests": True,
}

_COMPLETED_STATUSES = {"submitted", "archived", "dropped"}


def load_config() -> Dict[str, Any]:
    path = Path("config.json")
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def get_current_child(cfg: Optional[Dict[str, Any]] = None) -> str:
    payload = cfg if isinstance(cfg, dict) else load_config()
    child = payload.get("current_child") if isinstance(payload, dict) else None
    return str(child or "Inazuma_Yagami")


def _coerce_int(value: Any, default: int, *, minimum: int = 0) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return parsed if parsed >= minimum else minimum


def _coerce_float(value: Any, default: float, *, minimum: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return parsed if parsed >= minimum else minimum


def _clean_labels(values: Any, fallback: Iterable[str]) -> List[str]:
    if not isinstance(values, list):
        values = list(fallback)
    seen: Dict[str, None] = {}
    for item in values:
        label = str(item or "").strip()
        if label:
            seen[label] = None
    return list(seen.keys())


def get_github_submission_config(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload = cfg if isinstance(cfg, dict) else load_config()
    raw = payload.get("github_submission") if isinstance(payload, dict) else None
    raw = raw if isinstance(raw, dict) else {}

    policy = DEFAULT_GITHUB_SUBMISSION.copy()
    policy["enabled"] = bool(raw.get("enabled", policy["enabled"]))
    delivery_mode = str(raw.get("delivery_mode", policy["delivery_mode"]) or policy["delivery_mode"]).strip().lower()
    policy["delivery_mode"] = delivery_mode if delivery_mode in {"queue_only", "issues"} else policy["delivery_mode"]
    policy["repo_full_name"] = str(raw.get("repo_full_name") or policy["repo_full_name"]).strip()
    policy["api_base"] = str(raw.get("api_base") or policy["api_base"]).strip().rstrip("/")
    policy["token_env"] = str(raw.get("token_env") or policy["token_env"]).strip()
    policy["issue_title_prefix"] = str(raw.get("issue_title_prefix") or policy["issue_title_prefix"]).strip()
    policy["labels"] = _clean_labels(raw.get("labels"), policy["labels"])
    policy["optimization_labels"] = _clean_labels(raw.get("optimization_labels"), policy["optimization_labels"])
    policy["feature_labels"] = _clean_labels(raw.get("feature_labels"), policy["feature_labels"])
    policy["max_batch"] = _coerce_int(raw.get("max_batch"), int(policy["max_batch"]), minimum=1)
    policy["max_age_minutes"] = _coerce_int(raw.get("max_age_minutes"), int(policy["max_age_minutes"]), minimum=1)
    policy["daily_issue_cap"] = _coerce_int(raw.get("daily_issue_cap"), int(policy["daily_issue_cap"]), minimum=1)
    policy["cooldown_minutes"] = _coerce_int(raw.get("cooldown_minutes"), int(policy["cooldown_minutes"]), minimum=1)
    policy["poll_interval_sec"] = _coerce_float(raw.get("poll_interval_sec"), float(policy["poll_interval_sec"]), minimum=5.0)
    policy["min_resource_trend_pressure"] = min(1.0, _coerce_float(raw.get("min_resource_trend_pressure"), float(policy["min_resource_trend_pressure"]), minimum=0.0))
    policy["max_patch_excerpt_chars"] = _coerce_int(raw.get("max_patch_excerpt_chars"), int(policy["max_patch_excerpt_chars"]), minimum=256)
    policy["max_body_chars"] = _coerce_int(raw.get("max_body_chars"), int(policy["max_body_chars"]), minimum=512)
    policy["auto_submit_optimization_requests"] = bool(
        raw.get("auto_submit_optimization_requests", policy["auto_submit_optimization_requests"])
    )
    return policy


def _configured_token_from_section(section: Any) -> str:
    if not isinstance(section, dict):
        return ""
    for key in ("token", "access_token", "github_token", "pat"):
        value = str(section.get(key) or "").strip()
        if value:
            return value
    for key in ("token_file", "token_path"):
        path_text = str(section.get(key) or "").strip()
        if not path_text:
            continue
        try:
            token = Path(path_text).expanduser().read_text(encoding="utf-8").strip()
        except Exception:
            token = ""
        if token:
            return token
    return ""


def resolve_github_token(cfg: Optional[Dict[str, Any]] = None, policy: Optional[Dict[str, Any]] = None) -> str:
    payload = cfg if isinstance(cfg, dict) else load_config()
    active_policy = policy if isinstance(policy, dict) else get_github_submission_config(payload)
    token_env = str(active_policy.get("token_env") or "").strip()
    if token_env:
        token = os.environ.get(token_env, "").strip()
        if token:
            return token

    raw_submission = payload.get("github_submission") if isinstance(payload, dict) else None
    token = _configured_token_from_section(raw_submission)
    if token:
        return token

    raw_github = payload.get("github") if isinstance(payload, dict) else None
    token = _configured_token_from_section(raw_github)
    if token:
        return token

    raise RuntimeError(
        f"GitHub token not found; set {token_env or 'GITHUB_TOKEN'} or configure github_submission.token/token_file"
    )


def github_outbox_path(child: str) -> Path:
    return Path("AI_Children") / child / "memory" / "github_outbox.jsonl"


def github_outbox_history_path(child: str) -> Path:
    return Path("AI_Children") / child / "memory" / "github_outbox_history.jsonl"


def github_outbox_archive_path(child: str) -> Path:
    return Path("AI_Children") / child / "memory" / "github_outbox_archive.jsonl"


def github_attachment_dir(child: str) -> Path:
    return Path("AI_Children") / child / "memory" / "github_proposals"


def github_bridge_lock_path(child: str) -> Path:
    return Path("AI_Children") / child / "memory" / "github_bridge.lock"


def typed_outbox_path(child: str) -> Path:
    return Path("AI_Children") / child / "memory" / "typed_outbox.jsonl"


def append_typed_outbox_notice(
    child: str,
    text: str,
    *,
    target: str = "owner_dm",
    user_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    allow_empty: bool = False,
    attachment_path: Optional[str] = None,
) -> Optional[str]:
    payload = "" if text is None else str(text)
    if not allow_empty and not payload.strip() and not attachment_path:
        return None
    entry = {
        "id": f"typed_{uuid.uuid4().hex}",
        "text": payload,
        "target": str(target or "owner_dm").strip() or "owner_dm",
        "user_id": str(user_id) if user_id is not None else None,
        "metadata": metadata or {},
        "allow_empty": bool(allow_empty),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if attachment_path:
        entry["attachment_path"] = str(attachment_path)
    return entry["id"] if _append_jsonl(typed_outbox_path(child), entry) else None


def maybe_queue_submission_discord_notice(
    child: str,
    entry: Dict[str, Any],
    result: Dict[str, Any],
    cfg: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
    if not bool(metadata.get("notify_discord_on_submit")):
        return None

    payload = cfg if isinstance(cfg, dict) else load_config()
    discord_cfg = payload.get("discord") if isinstance(payload, dict) else None
    if not isinstance(discord_cfg, dict) or not bool(discord_cfg.get("enabled", False)):
        return None

    title = str(result.get("title") or entry.get("title") or "GitHub issue").strip() or "GitHub issue"
    issue_number = result.get("issue_number")
    issue_url = str(result.get("issue_url") or "").strip()
    explanation = str(metadata.get("discord_explanation") or "").strip()
    source = str(metadata.get("discord_reason") or metadata.get("source") or "github_submission").strip() or "github_submission"

    header = f"Ina filed GitHub issue #{issue_number}: {title}" if issue_number is not None else f"Ina filed a GitHub issue: {title}"
    lines = [header]
    if explanation:
        lines.append(explanation)
    if issue_url:
        lines.append(issue_url)

    return append_typed_outbox_notice(
        child,
        "\n".join(lines),
        target=str(metadata.get("discord_target") or "owner_dm").strip() or "owner_dm",
        user_id=str(metadata.get("discord_user_id")) if metadata.get("discord_user_id") is not None else None,
        metadata={
            "source": "github_submission",
            "reason": source,
            "issue_number": issue_number,
            "issue_url": issue_url or None,
            "title": title,
        },
    )


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return True
    except Exception:
        return False


def _write_attachment(child: str, entry_id: str, patch_text: str) -> Optional[str]:
    patch_text = str(patch_text or "")
    if not patch_text.strip():
        return None
    path = github_attachment_dir(child) / f"{entry_id}.diff"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(patch_text, encoding="utf-8")
        return str(path)
    except Exception:
        return None


def append_github_issue_entry(
    child: str,
    title: str,
    body: str,
    *,
    kind: str = "request",
    labels: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    attachment_path: Optional[str] = None,
    patch_text: Optional[str] = None,
) -> Optional[str]:
    clean_title = str(title or "").strip()
    clean_body = str(body or "").strip()
    if not clean_title or (not clean_body and not attachment_path and not patch_text):
        return None

    entry_id = f"github_{uuid.uuid4().hex}"
    stored_attachment = str(attachment_path).strip() if attachment_path else None
    if patch_text:
        stored_attachment = _write_attachment(child, entry_id, patch_text) or stored_attachment

    entry = {
        "id": entry_id,
        "title": clean_title,
        "body": clean_body,
        "kind": str(kind or "request").strip().lower() or "request",
        "labels": _clean_labels(labels or [], []),
        "metadata": metadata or {},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if stored_attachment:
        entry["attachment_path"] = stored_attachment
    return entry_id if _append_jsonl(github_outbox_path(child), entry) else None


def _entry_timestamp(entry: Dict[str, Any]) -> Optional[datetime]:
    created_at = entry.get("created_at")
    if not created_at:
        return None
    try:
        stamp = datetime.fromisoformat(str(created_at))
    except Exception:
        return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp


def _load_history_entries(child: str) -> List[Dict[str, Any]]:
    path = github_outbox_history_path(child)
    if not path.exists():
        return []
    entries: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except Exception:
                    continue
                if isinstance(payload, dict):
                    entries.append(payload)
    except Exception:
        return []
    return entries


def load_completed_history_ids(child: str) -> Set[str]:
    completed: Set[str] = set()
    for entry in _load_history_entries(child):
        status = str(entry.get("status") or "").strip().lower()
        if status not in _COMPLETED_STATUSES:
            continue
        entry_id = str(entry.get("id") or entry.get("entry_id") or "").strip()
        if entry_id:
            completed.add(entry_id)
    return completed


def load_submitted_count_for_day(child: str, day: Optional[str] = None) -> int:
    target_day = str(day or datetime.now(timezone.utc).date().isoformat())
    total = 0
    for entry in _load_history_entries(child):
        if str(entry.get("status") or "").strip().lower() != "submitted":
            continue
        stamp = str(entry.get("timestamp") or "")
        if stamp.startswith(target_day):
            total += 1
    return total


def read_pending_entries(child: str, cfg: Optional[Dict[str, Any]] = None, seen_ids: Optional[Set[str]] = None) -> List[Dict[str, Any]]:
    path = github_outbox_path(child)
    if not path.exists():
        return []
    policy = get_github_submission_config(cfg)
    if seen_ids is None:
        seen_ids = load_completed_history_ids(child)
    pending: List[Dict[str, Any]] = []
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=int(policy["max_age_minutes"]))
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except Exception:
                    continue
                if not isinstance(entry, dict):
                    continue
                entry_id = str(entry.get("id") or "").strip()
                if not entry_id or entry_id in seen_ids:
                    continue
                stamp = _entry_timestamp(entry)
                if stamp is not None and stamp < cutoff:
                    pending.append({**entry, "_stale": True})
                else:
                    pending.append(entry)
                if len(pending) >= int(policy["max_batch"]):
                    break
    except Exception:
        return []
    return pending


def log_history(child: str, entry_id: str, status: str, **extra: Any) -> bool:
    payload = {
        "id": str(entry_id or "").strip(),
        "status": str(status or "").strip().lower(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    payload.update({key: value for key, value in extra.items() if value is not None})
    if not payload["id"]:
        return False
    return _append_jsonl(github_outbox_history_path(child), payload)


def archive_entry(child: str, entry: Dict[str, Any], reason: str, **extra: Any) -> bool:
    payload = dict(entry)
    payload["archive_reason"] = str(reason or "").strip().lower()
    payload["archived_at"] = datetime.now(timezone.utc).isoformat()
    payload.update({key: value for key, value in extra.items() if value is not None})
    return _append_jsonl(github_outbox_archive_path(child), payload)


def _truncate_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = text[: max(0, limit - 64)].rstrip()
    return f"{head}\n\n[truncated for GitHub issue body]"


def _attachment_excerpt(path_text: Optional[str], limit: int) -> str:
    if not path_text:
        return ""
    path = Path(path_text)
    if not path.exists() or not path.is_file():
        return ""
    try:
        data = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    return _truncate_text(data, limit)


def build_issue_title(entry: Dict[str, Any], cfg: Optional[Dict[str, Any]] = None) -> str:
    policy = get_github_submission_config(cfg)
    prefix = str(policy.get("issue_title_prefix") or "").strip()
    title = str(entry.get("title") or "").strip()
    if prefix and not title.startswith(prefix):
        title = f"{prefix} {title}"
    return title[:120]


def labels_for_kind(kind: str, cfg: Optional[Dict[str, Any]] = None) -> List[str]:
    policy = get_github_submission_config(cfg)
    kind_key = str(kind or "request").strip().lower()
    if kind_key in {"feature", "feature_request", "feature_patch", "feature_proposal"}:
        return list(policy.get("feature_labels", []))
    if kind_key in {"optimization", "optimization_request", "optimization_patch", "patch_attempt"}:
        return list(policy.get("optimization_labels", []))
    return list(policy.get("labels", []))


def build_issue_body(entry: Dict[str, Any], cfg: Optional[Dict[str, Any]] = None) -> str:
    policy = get_github_submission_config(cfg)
    metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
    labels = entry.get("labels") if isinstance(entry.get("labels"), list) else []
    touched_files = metadata.get("touched_files") if isinstance(metadata.get("touched_files"), list) else []
    evidence = metadata.get("evidence") if isinstance(metadata.get("evidence"), list) else []
    review_notes = metadata.get("review_notes") if isinstance(metadata.get("review_notes"), list) else []
    confidence = metadata.get("confidence")
    source = str(metadata.get("source") or "internal")
    submission_mode = str(metadata.get("submission_mode") or "").strip().lower()
    attachment_path = str(entry.get("attachment_path") or "").strip()
    attachment_excerpt = _attachment_excerpt(attachment_path, int(policy["max_patch_excerpt_chars"]))

    lines: List[str] = []
    lines.append("## Ina Submission")
    lines.append(f"- kind: `{str(entry.get('kind') or 'request')}`")
    lines.append(f"- source: `{source}`")
    if submission_mode:
        lines.append(f"- submission_mode: `{submission_mode}`")
    lines.append(f"- created_at: `{str(entry.get('created_at') or '')}`")
    if confidence is not None:
        lines.append(f"- confidence: `{confidence}`")
    if labels:
        lines.append(f"- suggested_labels: `{', '.join(str(label) for label in labels if label)}`")

    body_text = str(entry.get("body") or "").strip()
    if body_text:
        lines.append("")
        lines.append("## Summary")
        lines.append(body_text)

    if evidence:
        lines.append("")
        lines.append("## Evidence")
        for item in evidence:
            text = str(item or "").strip()
            if text:
                lines.append(f"- {text}")

    if touched_files:
        lines.append("")
        lines.append("## Touched Files")
        for item in touched_files:
            text = str(item or "").strip()
            if text:
                lines.append(f"- `{text}`")

    if review_notes:
        lines.append("")
        lines.append("## Review Notes")
        for item in review_notes:
            text = str(item or "").strip()
            if text:
                lines.append(f"- {text}")

    if attachment_excerpt:
        info = "diff" if attachment_path.endswith((".diff", ".patch")) else "text"
        lines.append("")
        lines.append("## Attachment Excerpt")
        lines.append(f"Local attachment path: `{attachment_path}`")
        lines.append(f"```{info}")
        lines.append(attachment_excerpt)
        lines.append("```")
    elif attachment_path:
        lines.append("")
        lines.append("## Attachment")
        lines.append(f"Local attachment path: `{attachment_path}`")

    lines.append("")
    lines.append("## Review Guard")
    lines.append("- Human review required before merge, execution, or deployment.")
    lines.append("- Treat any patch excerpt as a proposal, not an approved change.")

    body = "\n".join(lines).strip()
    return _truncate_text(body, int(policy["max_body_chars"]))


def submit_issue(entry: Dict[str, Any], cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    policy = get_github_submission_config(cfg)
    if policy["delivery_mode"] != "issues":
        raise RuntimeError("delivery_mode is not set to 'issues'")
    repo_full_name = str(policy.get("repo_full_name") or "").strip()
    if not repo_full_name:
        raise RuntimeError("repo_full_name is not configured")
    token = resolve_github_token(cfg, policy)

    url = f"{policy['api_base']}/repos/{repo_full_name}/issues"
    payload = {
        "title": build_issue_title(entry, policy),
        "body": build_issue_body(entry, policy),
        "labels": _clean_labels(entry.get("labels"), policy.get("labels", [])),
    }
    req = urlrequest.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": "project-inazuma-github-bridge",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with urlrequest.urlopen(req, timeout=20) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urlerror.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace") if hasattr(exc, "read") else str(exc)
        raise RuntimeError(f"GitHub issue create failed: HTTP {exc.code}: {body[:400]}") from exc
    except Exception as exc:
        raise RuntimeError(f"GitHub issue create failed: {exc}") from exc

    issue_number = result.get("number")
    issue_url = result.get("html_url") or result.get("url")
    return {
        "issue_number": int(issue_number) if issue_number is not None else None,
        "issue_url": str(issue_url or "").strip() or None,
        "title": payload["title"],
    }


__all__ = [
    "append_github_issue_entry",
    "archive_entry",
    "build_issue_body",
    "build_issue_title",
    "labels_for_kind",
    "DEFAULT_GITHUB_SUBMISSION",
    "append_typed_outbox_notice",
    "get_current_child",
    "get_github_submission_config",
    "github_attachment_dir",
    "github_bridge_lock_path",
    "github_outbox_archive_path",
    "github_outbox_history_path",
    "github_outbox_path",
    "load_completed_history_ids",
    "load_config",
    "load_submitted_count_for_day",
    "log_history",
    "maybe_queue_submission_discord_notice",
    "read_pending_entries",
    "resolve_github_token",
    "submit_issue",
    "typed_outbox_path",
]
