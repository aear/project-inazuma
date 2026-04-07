from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib import parse as urlparse
from urllib import request as urlrequest

from github_submission import (
    get_current_child,
    get_github_submission_config,
    github_outbox_history_path,
    load_config,
    resolve_github_token,
)

DEFAULT_GITHUB_FEEDBACK: Dict[str, Any] = {
    "enabled": False,
    "poll_interval_sec": 900.0,
    "max_issues_per_check": 20,
    "max_comments_per_issue": 50,
    "ignore_authors": [],
}


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


def _clean_string_list(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    seen: Dict[str, None] = {}
    for item in values:
        text = str(item or "").strip()
        if text:
            seen[text] = None
    return list(seen.keys())


def get_github_feedback_config(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload = cfg if isinstance(cfg, dict) else load_config()
    raw = payload.get("github_feedback") if isinstance(payload, dict) else None
    raw = raw if isinstance(raw, dict) else {}

    policy = DEFAULT_GITHUB_FEEDBACK.copy()
    policy["enabled"] = bool(raw.get("enabled", policy["enabled"]))
    policy["poll_interval_sec"] = _coerce_float(
        raw.get("poll_interval_sec"), float(policy["poll_interval_sec"]), minimum=60.0
    )
    policy["max_issues_per_check"] = _coerce_int(
        raw.get("max_issues_per_check"), int(policy["max_issues_per_check"]), minimum=1
    )
    policy["max_comments_per_issue"] = min(
        100,
        _coerce_int(
            raw.get("max_comments_per_issue"),
            int(policy["max_comments_per_issue"]),
            minimum=1,
        ),
    )
    policy["ignore_authors"] = _clean_string_list(raw.get("ignore_authors"))
    return policy


def github_feedback_state_path(child: str) -> Path:
    return Path("AI_Children") / child / "memory" / "github_feedback_state.json"


def github_issue_feedback_log_path(child: str) -> Path:
    return Path("AI_Children") / child / "memory" / "github_issue_feedback.jsonl"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


def load_submitted_issue_refs(child: str, *, limit: int = 20) -> List[Dict[str, Any]]:
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
                if not isinstance(payload, dict):
                    continue
                if str(payload.get("status") or "").strip().lower() != "submitted":
                    continue
                issue_number = payload.get("issue_number")
                try:
                    issue_number = int(issue_number)
                except (TypeError, ValueError):
                    continue
                entries.append({
                    "entry_id": str(payload.get("id") or "").strip(),
                    "issue_number": issue_number,
                    "issue_url": str(payload.get("issue_url") or "").strip() or None,
                    "title": str(payload.get("title") or "").strip() or None,
                    "submitted_at": str(payload.get("timestamp") or "").strip() or None,
                })
    except Exception:
        return []

    recent: List[Dict[str, Any]] = []
    seen = set()
    for entry in reversed(entries):
        issue_number = entry["issue_number"]
        if issue_number in seen:
            continue
        seen.add(issue_number)
        recent.append(entry)
        if len(recent) >= limit:
            break
    return recent


def _github_get_json(url: str, token: str) -> Any:
    req = urlrequest.Request(
        url,
        method="GET",
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "User-Agent": "project-inazuma-github-feedback",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urlrequest.urlopen(req, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_issue_comments(
    *,
    repo_full_name: str,
    issue_number: int,
    token: str,
    api_base: str,
    per_page: int,
) -> List[Dict[str, Any]]:
    query = urlparse.urlencode({"per_page": max(1, min(100, per_page))})
    url = f"{api_base.rstrip('/')}/repos/{repo_full_name}/issues/{issue_number}/comments?{query}"
    payload = _github_get_json(url, token)
    return payload if isinstance(payload, list) else []


def _truncate(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 32)].rstrip() + "\n[truncated]"


def _record_feedback_event(child: str, record: Dict[str, Any]) -> Optional[str]:
    try:
        from experience_logger import ExperienceLogger

        logger = ExperienceLogger(child=child)
        return logger.log_event(
            situation_tags=["github_feedback", "issue_comment", "ina_suggestion"],
            perceived_entities=[
                {
                    "type": "github_issue",
                    "issue_number": record.get("issue_number"),
                    "issue_url": record.get("issue_url"),
                    "title": record.get("issue_title"),
                    "source_entry_id": record.get("source_entry_id"),
                },
                {
                    "type": "github_comment",
                    "comment_id": record.get("comment_id"),
                    "comment_url": record.get("comment_url"),
                    "author": record.get("author"),
                    "author_association": record.get("author_association"),
                },
            ],
            outcome={
                "feedback_received": True,
                "issue_number": record.get("issue_number"),
                "comment_id": record.get("comment_id"),
            },
            internal_state={
                "source": "github_feedback_sync",
                "body_excerpt": record.get("body_excerpt"),
            },
            narrative=(
                f"GitHub feedback arrived on Ina issue #{record.get('issue_number')}: "
                f"{record.get('body_excerpt') or '(empty comment)'}"
            ),
        )
    except Exception:
        return None


def sync_issue_feedback(
    cfg: Optional[Dict[str, Any]] = None,
    *,
    issue_refs: Optional[Iterable[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    payload = cfg if isinstance(cfg, dict) else load_config()
    child = get_current_child(payload)
    feedback_policy = get_github_feedback_config(payload)
    if not feedback_policy.get("enabled", False):
        return {"checked": False, "reason": "disabled", "new_comments": 0}

    submission_policy = get_github_submission_config(payload)
    repo_full_name = str(submission_policy.get("repo_full_name") or "").strip()
    if not repo_full_name:
        return {"checked": False, "reason": "repo_not_configured", "new_comments": 0}

    try:
        token = resolve_github_token(payload, submission_policy)
    except RuntimeError:
        return {"checked": False, "reason": "missing_token", "new_comments": 0}

    refs = list(issue_refs) if issue_refs is not None else load_submitted_issue_refs(
        child, limit=int(feedback_policy["max_issues_per_check"])
    )
    if not refs:
        state = _load_json(github_feedback_state_path(child))
        state["last_checked_at"] = _now_iso()
        state["last_result"] = "no_submitted_issues"
        _write_json(github_feedback_state_path(child), state)
        return {"checked": True, "reason": "no_submitted_issues", "new_comments": 0}

    state = _load_json(github_feedback_state_path(child))
    seen_comments = state.get("seen_comments") if isinstance(state.get("seen_comments"), dict) else {}
    ignore_authors = {
        str(author).strip()
        for author in feedback_policy.get("ignore_authors", [])
        if str(author).strip()
    }
    new_comments = 0
    checked_issues = 0

    for ref in refs[: int(feedback_policy["max_issues_per_check"])]:
        try:
            issue_number = int(ref.get("issue_number"))
        except (TypeError, ValueError):
            continue
        issue_key = str(issue_number)
        seen_for_issue = {str(item) for item in seen_comments.get(issue_key, []) if str(item)}
        comments = fetch_issue_comments(
            repo_full_name=repo_full_name,
            issue_number=issue_number,
            token=token,
            api_base=str(submission_policy.get("api_base") or "https://api.github.com"),
            per_page=int(feedback_policy["max_comments_per_issue"]),
        )
        checked_issues += 1
        for comment in comments:
            if not isinstance(comment, dict):
                continue
            comment_id = str(comment.get("id") or "").strip()
            if not comment_id or comment_id in seen_for_issue:
                continue
            user = comment.get("user") if isinstance(comment.get("user"), dict) else {}
            author = str(user.get("login") or "").strip()
            if author and author in ignore_authors:
                seen_for_issue.add(comment_id)
                continue
            body = str(comment.get("body") or "").strip()
            record = {
                "type": "github_issue_comment",
                "received_at": _now_iso(),
                "source_entry_id": ref.get("entry_id"),
                "issue_number": issue_number,
                "issue_url": ref.get("issue_url"),
                "issue_title": ref.get("title"),
                "comment_id": comment_id,
                "comment_url": str(comment.get("html_url") or comment.get("url") or "").strip() or None,
                "author": author or None,
                "author_association": str(comment.get("author_association") or "").strip() or None,
                "created_at": str(comment.get("created_at") or "").strip() or None,
                "updated_at": str(comment.get("updated_at") or "").strip() or None,
                "body_excerpt": _truncate(body),
            }
            event_id = _record_feedback_event(child, record)
            if event_id:
                record["experience_event_id"] = event_id
            _append_jsonl(github_issue_feedback_log_path(child), record)
            seen_for_issue.add(comment_id)
            new_comments += 1
        seen_comments[issue_key] = list(seen_for_issue)[-500:]

    state.update({
        "last_checked_at": _now_iso(),
        "last_result": "checked",
        "checked_issues": checked_issues,
        "new_comments": new_comments,
        "seen_comments": seen_comments,
    })
    _write_json(github_feedback_state_path(child), state)
    return {"checked": True, "reason": "checked", "checked_issues": checked_issues, "new_comments": new_comments}


__all__ = [
    "DEFAULT_GITHUB_FEEDBACK",
    "fetch_issue_comments",
    "get_github_feedback_config",
    "github_feedback_state_path",
    "github_issue_feedback_log_path",
    "load_submitted_issue_refs",
    "sync_issue_feedback",
]
