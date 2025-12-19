from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

try:
    from gui_hook import log_to_statusbox
except Exception:  # pragma: no cover - GUI not required
    def log_to_statusbox(msg: str) -> None:
        print(msg)

from social_map import get_high_trust_contacts, get_owner_user_id

try:
    from fragmentation_engine import make_fragment, store_fragment
except Exception:  # pragma: no cover - fallback to no-op if fragments unavailable
    make_fragment = None  # type: ignore[assignment]
    store_fragment = None  # type: ignore[assignment]

HUMOR_TAGS = ["benign_violation", "resolved_contradiction", "self_parody"]

_CONFIG_PATH = Path("config.json")
_DEFAULT_CHILD = "Inazuma_Yagami"
_HUMOR_INVITE_KEY = "humor_expression_invite"
_HUMOR_INVITE_TTL = 90.0
_HUMOR_INVITE_COOLDOWN = 180.0


def _log(msg: str) -> None:
    log_to_statusbox(f"[Humor] {msg}")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _iso_plus(seconds: float) -> str:
    return datetime.fromtimestamp(time.time() + seconds, timezone.utc).isoformat()


def _load_config() -> Dict[str, Any]:
    if not _CONFIG_PATH.exists():
        return {}
    try:
        return json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        _log("Failed to read config.json; using defaults.")
        return {}


def _current_child() -> str:
    cfg = _load_config()
    if isinstance(cfg, dict):
        return str(cfg.get("current_child", _DEFAULT_CHILD))
    return _DEFAULT_CHILD


def _inastate_path() -> Path:
    return Path("AI_Children") / _current_child() / "memory" / "inastate.json"


def _read_inastate() -> Dict[str, Any]:
    path = _inastate_path()
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        _log("Failed to read inastate; returning empty state.")
        return {}


def _write_inastate(state: Dict[str, Any]) -> None:
    path = _inastate_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def _get_inastate(key: str, default: Any = None) -> Any:
    state = _read_inastate()
    return state.get(key, default)


def _update_inastate(key: str, value: Any) -> None:
    state = _read_inastate()
    state[key] = value
    _write_inastate(state)


def _playfulness_snapshot() -> Dict[str, Any]:
    state = _get_inastate("emotion_playfulness_state") or {}
    return state if isinstance(state, dict) else {}


def _playfulness_level(default: float = 0.0) -> float:
    snap = _playfulness_snapshot()
    try:
        return float(snap.get("value", default))
    except (TypeError, ValueError):
        return default


def ensure_humor_tags(existing: Optional[Sequence[str]] = None) -> List[str]:
    """
    Merge HUMOR_TAGS into an existing tag list without duplicates.
    """
    tags = [str(tag) for tag in existing] if existing else []
    seen = {tag.lower() for tag in tags}
    for required in HUMOR_TAGS:
        if required not in seen:
            tags.append(required)
            seen.add(required)
    return tags


def record_benign_humor_event(
    summary: str,
    *,
    detail: Optional[str] = None,
    payload: Optional[Dict[str, Any]] = None,
    tags: Optional[Sequence[str]] = None,
    context: Optional[Dict[str, Any]] = None,
    importance: float = 0.1,
) -> Optional[Dict[str, Any]]:
    """
    Persist a lightweight fragment describing a benign violation that resolved.
    The fragment is automatically tagged so other modules can notice patterns.
    """
    if make_fragment is None or store_fragment is None:
        _log("Fragmentation engine unavailable; cannot log humor fragment.")
        return None

    tag_list = ensure_humor_tags(tags)
    payload_data = dict(payload or {})
    if detail:
        payload_data.setdefault("detail", detail)

    context_data = dict(context or {})
    context_data.setdefault("humor_tags", tag_list)
    context_data.setdefault("playfulness_level", round(_playfulness_level(), 4))

    fragment = make_fragment(  # type: ignore[operator]
        frag_type="humor_trace",
        source="humor_engine",
        summary=summary,
        tags=tag_list,
        importance=float(importance),
        emotions=None,
        payload=payload_data,
        context=context_data,
    )
    store_fragment(fragment, reason="humor_trace")  # type: ignore[operator]
    return fragment


def _summaries_for_contacts(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summary: List[Dict[str, Any]] = []
    for entry in entries:
        summary.append(
            {
                "user_id": entry.get("user_id"),
                "display_name": entry.get("display_name"),
                "trust_hint": entry.get("trust_hint"),
                "tags": entry.get("tags"),
                "last_interaction": entry.get("last_interaction"),
            }
        )
    return summary


def _set_invite_block(reason: str, level: float) -> None:
    _update_inastate(
        _HUMOR_INVITE_KEY,
        {
            "ready": False,
            "reason": reason,
            "level": round(level, 4),
            "timestamp": _now_iso(),
        },
    )


def maybe_prepare_expression_invite(
    *,
    min_playfulness: float = 0.35,
    min_trust: str = "high",
) -> Optional[Dict[str, Any]]:
    """
    Surface an optional invitation to externalise humour when:
      - Playfulness is above min_playfulness.
      - A recent benign violation resolved (tags remembered in state).
      - At least one high-trust contact is available.
      - The social cooldown has elapsed.
    """
    snapshot = _playfulness_snapshot()
    level = _playfulness_level()
    last_trigger = snapshot.get("last_trigger") if isinstance(snapshot, dict) else None
    trigger_tags = last_trigger.get("tags") if isinstance(last_trigger, dict) else None

    if level < min_playfulness:
        _set_invite_block("insufficient_playfulness", level)
        return None
    if not trigger_tags:
        _set_invite_block("no_recent_benign_violation", level)
        return None

    gate_state = _get_inastate(_HUMOR_INVITE_KEY) or {}
    try:
        cooldown_until = float(gate_state.get("cooldown_until") or 0.0)
    except (TypeError, ValueError):
        cooldown_until = 0.0
    if cooldown_until and time.time() < cooldown_until:
        return None

    config = _load_config()
    contacts = get_high_trust_contacts(config=config, min_level=min_trust, limit=3)
    if not contacts:
        _set_invite_block("no_high_trust_contacts", level)
        return None

    owner_user_id = get_owner_user_id(config)
    invite = {
        "ready": True,
        "level": round(level, 3),
        "timestamp": _now_iso(),
        "expires_at": _iso_plus(_HUMOR_INVITE_TTL),
        "note": "Optional: share only if the amusement still feels safe.",
        "contacts": _summaries_for_contacts(contacts),
        "owner_user_id": owner_user_id,
        "last_trigger": last_trigger,
        "cooldown_until": time.time() + _HUMOR_INVITE_COOLDOWN,
    }
    _update_inastate(_HUMOR_INVITE_KEY, invite)
    _log(f"Humor bridge open (playfulness {level:.2f}).")
    return invite


__all__ = [
    "HUMOR_TAGS",
    "ensure_humor_tags",
    "record_benign_humor_event",
    "maybe_prepare_expression_invite",
]

