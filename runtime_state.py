import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from gui_hook import log_to_statusbox
from io_utils import atomic_write_json, file_lock


def load_config() -> Dict[str, Any]:
    path = Path("config.json")
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _current_child() -> str:
    return str(load_config().get("current_child", "Inazuma_Yagami") or "Inazuma_Yagami")


def _memory_path(child: Optional[str] = None) -> Path:
    return Path("AI_Children") / (child or _current_child()) / "memory"


def _inastate_path(child: Optional[str] = None) -> Path:
    return _memory_path(child) / "inastate.json"


def _inastate_lock_path(child: Optional[str] = None) -> Path:
    return _memory_path(child) / "inastate.lock"


def _self_questions_path(child: Optional[str] = None) -> Path:
    return _memory_path(child) / "self_questions.json"


def _typed_outbox_path(child: Optional[str] = None) -> Path:
    return _memory_path(child) / "typed_outbox.jsonl"


def _load_inastate_state(child: Optional[str] = None) -> Dict[str, Any]:
    path = _inastate_path(child)
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def get_inastate(key: str, default: Any = None) -> Any:
    state = _load_inastate_state()
    if not isinstance(state, dict):
        return default
    return state.get(key, default)


def update_inastate(key: str, value: Any) -> None:
    child = _current_child()
    with file_lock(_inastate_lock_path(child)):
        state = _load_inastate_state(child)
        state[key] = value
        atomic_write_json(_inastate_path(child), state, indent=4)


def _load_self_question_entries(child: Optional[str] = None) -> List[Dict[str, Any]]:
    path = _self_questions_path(child)
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except Exception:
        return []

    entries: List[Dict[str, Any]] = []
    if isinstance(raw, list):
        for entry in raw:
            if not isinstance(entry, dict) or not entry.get("question"):
                continue
            now = datetime.now(timezone.utc).isoformat()
            first = entry.get("first_asked") or entry.get("timestamp") or now
            last = entry.get("last_updated") or entry.get("timestamp") or first
            count = int(entry.get("count", entry.get("times", 1)) or 1)
            normalized = {
                "question": entry.get("question"),
                "first_asked": first,
                "last_updated": last,
                "count": count,
            }
            if entry.get("resolved_at"):
                normalized["resolved_at"] = entry.get("resolved_at")
            if entry.get("resolved_reason"):
                normalized["resolved_reason"] = entry.get("resolved_reason")
            if entry.get("resolution_history"):
                normalized["resolution_history"] = entry.get("resolution_history")
            entries.append(normalized)
    return entries


def _save_self_question_entries(entries: List[Dict[str, Any]], child: Optional[str] = None) -> None:
    path = _self_questions_path(child)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(entries, fh, indent=4)


def seed_self_question(question: str) -> None:
    if not question:
        return
    child = _current_child()
    entries = _load_self_question_entries(child)
    now_iso = datetime.now(timezone.utc).isoformat()
    normalized_question = question.strip()
    existing = None
    for entry in entries:
        if entry.get("question") == normalized_question:
            existing = entry
            break

    if existing:
        existing["count"] = int(existing.get("count", 1) or 1) + 1
        existing["last_updated"] = now_iso
        existing.pop("resolved_at", None)
        existing.pop("resolved_reason", None)
    else:
        entries.append(
            {
                "question": normalized_question,
                "first_asked": now_iso,
                "last_updated": now_iso,
                "count": 1,
            }
        )

    entries.sort(key=lambda item: item.get("first_asked", now_iso))
    entries = entries[-100:]
    _save_self_question_entries(entries, child)
    log_to_statusbox(f"[Manager] Self-question seeded: {normalized_question}")


def append_typed_outbox_entry(
    text: Optional[str],
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
        "target": target,
        "user_id": str(user_id) if user_id is not None else None,
        "metadata": metadata or {},
        "allow_empty": allow_empty,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if attachment_path:
        entry["attachment_path"] = attachment_path

    try:
        path = _typed_outbox_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return entry["id"]
    except Exception as exc:
        log_to_statusbox(f"[Manager] Failed to append typed outbox entry: {exc}")
        return None
