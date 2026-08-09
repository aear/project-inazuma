import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from discord_runtime import typed_outbox_path
from gui_hook import log_to_statusbox
from io_utils import atomic_write_json, file_lock
from config_layers import load_config


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
    return typed_outbox_path(child or _current_child(), load_config())


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


def get_inastate(key: str, default: Any = None, *, child: Optional[str] = None) -> Any:
    """Read one runtime-state value, optionally for a child other than the active one."""
    state = _load_inastate_state(child)
    if not isinstance(state, dict):
        return default
    return state.get(key, default)


def update_inastate(key: str, value: Any, *, child: Optional[str] = None) -> None:
    """Atomically replace one runtime-state value for the selected child."""
    target_child = child or _current_child()
    with file_lock(_inastate_lock_path(target_child)):
        state = _load_inastate_state(target_child)
        state[key] = value
        atomic_write_json(_inastate_path(target_child), state, indent=4)


def set_text_expression_intent(
    strategy: str,
    *,
    pointers: Optional[List[Any]] = None,
    song_path: Optional[str] = None,
    caption: Optional[str] = None,
    max_emotion_sliders: Optional[int] = None,
    max_code_pointers: Optional[int] = None,
    once: bool = True,
    child: Optional[str] = None,
) -> Dict[str, Any]:
    """Offer Ina's next text turn an explicit, inspectable expression choice."""
    aliases = {
        "reply": "respond",
        "original": "respond",
        "mimic": "mirror",
        "practice": "mirror",
        "state": "emotion",
        "feelings": "emotion",
        "module": "code_pointer",
        "modules": "code_pointer",
        "code": "code_pointer",
        "music": "song",
        "track": "song",
        "quiet": "silence",
    }
    normalized = str(strategy or "").strip().lower()
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"respond", "mirror", "emotion", "code_pointer", "song", "silence"}:
        raise ValueError(f"Unsupported text expression strategy: {strategy!r}")
    payload: Dict[str, Any] = {
        "strategy": normalized,
        "once": bool(once),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if pointers is not None:
        payload["pointers"] = list(pointers)[:24]
    if song_path is not None:
        payload["song_path"] = str(song_path)
    if caption is not None:
        payload["caption"] = str(caption)[:500]
    if max_emotion_sliders is not None:
        payload["max_emotion_sliders"] = max(1, min(24, int(max_emotion_sliders)))
    if max_code_pointers is not None:
        payload["max_code_pointers"] = max(1, min(8, int(max_code_pointers)))
    update_inastate("text_expression_intent", payload, child=child)
    return payload


def _positive_queue_limit(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _normalize_inastate_queue(raw: Any) -> tuple[List[Any], bool]:
    """Return a mutable FIFO plus whether malformed stored state was discarded."""
    if raw is None or raw == "":
        return [], False
    if isinstance(raw, dict):
        return [raw], False
    if isinstance(raw, list):
        return list(raw), False
    return [], True


def drain_inastate_queue(
    key: str,
    *,
    batch_limit: int,
    queue_limit: int,
    child: Optional[str] = None,
) -> Dict[str, Any]:
    """Atomically claim the oldest bounded batch from a runtime-state queue.

    Legacy single-object queues remain supported. If older code left more than
    ``queue_limit`` entries, the oldest bounded prefix is preserved and newer
    overflow is reported as dropped. Malformed queue state is cleared and
    reported through ``invalid``.
    """
    batch_limit = _positive_queue_limit(batch_limit, "batch_limit")
    queue_limit = _positive_queue_limit(queue_limit, "queue_limit")
    target_child = child or _current_child()

    with file_lock(_inastate_lock_path(target_child)):
        state = _load_inastate_state(target_child)
        raw = state.get(key)
        queue, invalid = _normalize_inastate_queue(raw)
        dropped = max(0, len(queue) - queue_limit)
        queue = queue[:queue_limit]
        batch = queue[:batch_limit]
        remaining = queue[batch_limit:]

        if invalid or batch or dropped:
            state[key] = remaining
            atomic_write_json(_inastate_path(target_child), state, indent=4)

    return {
        "batch": batch,
        "remaining": len(remaining),
        "dropped": dropped,
        "invalid": invalid,
    }


def append_inastate_queue(
    key: str,
    item: Any,
    *,
    queue_limit: int,
    child: Optional[str] = None,
) -> Dict[str, Any]:
    """Atomically append one item while preserving a bounded FIFO.

    Pending entries are never displaced by a newer item. An enqueue attempted
    against a full queue is rejected and counted in ``dropped``. Any legacy
    overflow beyond the bound is truncated before that decision.
    """
    queue_limit = _positive_queue_limit(queue_limit, "queue_limit")
    target_child = child or _current_child()

    with file_lock(_inastate_lock_path(target_child)):
        state = _load_inastate_state(target_child)
        raw = state.get(key)
        queue, invalid = _normalize_inastate_queue(raw)
        dropped = max(0, len(queue) - queue_limit)
        queue = queue[:queue_limit]
        queued = len(queue) < queue_limit
        if queued:
            queue.append(item)
        else:
            dropped += 1

        if invalid or queued or dropped:
            state[key] = queue
            atomic_write_json(_inastate_path(target_child), state, indent=4)

    return {
        "queued": queued,
        "remaining": len(queue),
        "dropped": dropped,
        "invalid": invalid,
    }


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
