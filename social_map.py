from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

CONFIG_PATH = Path("config.json")
DEFAULT_CHILD = "Inazuma_Yagami"
OWNER_TAGS = {"owner", "mother", "mum", "mom", "mama", "mommy", "guardian", "sakura"}
OWNER_FRIEND_TAGS = {
    "owner_friend",
    "owners_friend",
    "owner's friend",
    "friend_of_owner",
    "friend of owner",
    "mother_friend",
    "mothers_friend",
    "mother's friend",
    "mum_friend",
    "mums_friend",
    "mum's friend",
    "guardian_friend",
    "guardian's friend",
}


def _load_root_config() -> dict:
    """
    Lightweight loader for config.json so Ina can read social_map.json
    without pulling in heavier modules.
    """
    if not CONFIG_PATH.exists():
        return {}
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("Failed to read config at %s", CONFIG_PATH)
        return {}


def resolve_social_map_path(config: Optional[dict] = None) -> Path:
    """
    Resolve where the social map lives. Prefers config.json -> discord.social_map_path,
    otherwise defaults to the current child's memory folder.
    """
    cfg = config or _load_root_config()
    discord_cfg = cfg.get("discord") if isinstance(cfg, dict) else None
    raw_path = discord_cfg.get("social_map_path") if isinstance(discord_cfg, dict) else None
    if raw_path:
        return Path(raw_path)

    child = cfg.get("current_child", DEFAULT_CHILD) if isinstance(cfg, dict) else DEFAULT_CHILD
    return Path("AI_Children") / child / "memory" / "social_map.json"


def load_social_map(config: Optional[dict] = None) -> List[Dict[str, Any]]:
    """
    Read-only loader for Ina. Returns a list of entries with at least:
    user_id, display_name, tags, trust_hint, last_interaction.
    """
    path = resolve_social_map_path(config)
    if not path.exists():
        return []

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("Failed to read social map at %s", path)
        return []

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        people = data.get("people")
        if isinstance(people, list):
            return people
        return [data]
    return []


def _normalize_tags(raw_tags: Any) -> set[str]:
    if isinstance(raw_tags, list):
        return {str(tag).lower() for tag in raw_tags if tag is not None}
    if isinstance(raw_tags, str):
        return {raw_tags.lower()}
    return set()


def _json_safe_text(value: Any, *, max_len: int = 120) -> str:
    """
    Remove control characters so Ina can safely store names/labels in JSON.
    """
    if value is None:
        return ""
    text = str(value)
    cleaned = "".join(ch for ch in text if ch.isprintable())
    return cleaned[:max_len].strip()


def _sanitize_tags(raw_tags: Any) -> List[str]:
    if isinstance(raw_tags, list):
        candidates = raw_tags
    elif raw_tags is None:
        return []
    else:
        candidates = [raw_tags]

    safe: List[str] = []
    for tag in candidates:
        cleaned = _json_safe_text(tag)
        if cleaned:
            safe.append(cleaned)
    return safe


def _merge_tags(existing: Any, new_tags: List[str]) -> List[str]:
    merged: List[str] = []
    seen = set()
    existing_iterable = existing if isinstance(existing, list) else ([existing] if existing else [])
    for tag in existing_iterable:
        tag_str = str(tag)
        key = tag_str.lower()
        if key not in seen:
            merged.append(tag_str)
            seen.add(key)
    for tag in new_tags:
        key = tag.lower()
        if key not in seen:
            merged.append(tag)
            seen.add(key)
    return merged


def _write_social_map(entries: List[Dict[str, Any]], path: Path) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(entries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return True
    except Exception:
        logger.exception("Failed to write social map to %s", path)
        return False


def find_social_entry(
    user_id: str,
    *,
    social_map: Optional[List[Dict[str, Any]]] = None,
    config: Optional[dict] = None,
) -> Optional[Dict[str, Any]]:
    """Lookup helper for pulling a single entry by user_id."""
    entries = social_map if social_map is not None else load_social_map(config)
    for entry in entries:
        try:
            if str(entry.get("user_id")) == str(user_id):
                return entry
        except Exception:
            continue
    return None


def get_owner_user_id(config: Optional[dict] = None) -> Optional[int]:
    """
    Try to infer the primary owner from the social map (entry tagged with any OWNER_TAGS).
    Returns None if not found or not an int.
    """
    entries = load_social_map(config)
    for entry in entries:
        tag_set = _normalize_tags(entry.get("tags"))
        if OWNER_TAGS.intersection(tag_set):
            try:
                return int(entry.get("user_id"))
            except (TypeError, ValueError):
                logger.warning("Owner entry has non-integer user_id: %s", entry.get("user_id"))
                return None
    return None


def record_dm_attempt(
    user_id: str | int,
    display_name: Optional[str],
    *,
    config: Optional[dict] = None,
    tags: Optional[List[str]] = None,
    trust_hint: str = "unknown",
) -> bool:
    """
    Add or refresh a social map entry when someone DMs Ina.

    Returns True if the file was updated.
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    path = resolve_social_map_path(config)
    entries = load_social_map(config)
    entry = find_social_entry(user_id, social_map=entries)

    tags = tags if tags is not None else ["dm_attempt"]
    normalized_tags = _sanitize_tags(tags)
    name_safe = _json_safe_text(display_name) if display_name else None
    trust_hint_safe = _json_safe_text(trust_hint) if trust_hint is not None else None
    default_trust = trust_hint_safe or "unknown"

    updated = False
    if entry:
        if name_safe and name_safe != entry.get("display_name"):
            entry["display_name"] = name_safe
            updated = True
        if normalized_tags:
            entry["tags"] = _merge_tags(entry.get("tags"), normalized_tags)
            updated = True
        if entry.get("last_interaction") != timestamp:
            updated = True
        entry["last_interaction"] = timestamp
        if trust_hint_safe and (trust_hint != "unknown" or not entry.get("trust_hint")):
            entry["trust_hint"] = trust_hint_safe
            updated = True
    else:
        entries.append(
            {
                "user_id": str(user_id),
                "display_name": name_safe or str(user_id),
                "tags": normalized_tags,
                "trust_hint": default_trust,
                "last_interaction": timestamp,
            }
        )
        updated = True

    if not updated:
        return False

    return _write_social_map(entries, path)


def update_social_entry(
    user_id: str | int,
    *,
    display_name: Optional[str] = None,
    tags: Optional[List[str]] = None,
    trust_hint: Optional[str] = None,
    last_interaction: Optional[str] = None,
    config: Optional[dict] = None,
) -> bool:
    """
    General updater so Ina can evolve names/tags/trust hints over time.

    Returns True if the file was updated.
    """
    path = resolve_social_map_path(config)
    entries = load_social_map(config)
    entry = find_social_entry(user_id, social_map=entries)

    name_safe = _json_safe_text(display_name) if display_name else None
    new_tags = _sanitize_tags(tags)
    trust_hint_safe = _json_safe_text(trust_hint) if trust_hint is not None else None
    timestamp = last_interaction or datetime.now(timezone.utc).isoformat()

    updated = False
    if entry:
        if name_safe and name_safe != entry.get("display_name"):
            entry["display_name"] = name_safe
            updated = True
        if new_tags:
            entry["tags"] = _merge_tags(entry.get("tags"), new_tags)
            updated = True
        if trust_hint_safe and trust_hint_safe != entry.get("trust_hint"):
            entry["trust_hint"] = trust_hint_safe
            updated = True
        if last_interaction is not None or entry.get("last_interaction") != timestamp:
            entry["last_interaction"] = timestamp
            updated = True
    else:
        entries.append(
            {
                "user_id": str(user_id),
                "display_name": name_safe or str(user_id),
                "tags": new_tags,
                "trust_hint": trust_hint_safe or "unknown",
                "last_interaction": timestamp,
            }
        )
        updated = True

    if not updated:
        return False

    return _write_social_map(entries, path)


def is_owner_friend(
    user_id: str | int,
    *,
    social_map: Optional[List[Dict[str, Any]]] = None,
    config: Optional[dict] = None,
) -> bool:
    """
    Returns True if the user has any tag matching OWNER_FRIEND_TAGS.
    """
    entry = find_social_entry(str(user_id), social_map=social_map, config=config)
    if not entry:
        return False
    tag_set = _normalize_tags(entry.get("tags"))
    return bool(OWNER_FRIEND_TAGS.intersection(tag_set))


def _trust_score(trust_hint: Any) -> int:
    """
    Map textual trust hints into a simple score for ordering.
    """
    if trust_hint is None:
        return -1
    hint = str(trust_hint).strip().lower()
    table = {
        "very_high": 3,
        "very high": 3,
        "high": 2,
        "medium": 1,
        "med": 1,
        "low": 0,
        "unknown": -1,
    }
    return table.get(hint, 0 if hint else -1)


def is_high_trust(
    user_id: str | int,
    *,
    social_map: Optional[List[Dict[str, Any]]] = None,
    config: Optional[dict] = None,
    min_level: str = "high",
) -> bool:
    """
    Returns True if the user meets or exceeds the min_level trust hint.
    """
    entry = find_social_entry(str(user_id), social_map=social_map, config=config)
    if not entry:
        return False
    score = _trust_score(entry.get("trust_hint"))
    min_score = _trust_score(min_level)
    if score >= min_score and min_score >= 0:
        return True
    tags = _normalize_tags(entry.get("tags"))
    return "trusted" in tags and min_score <= _trust_score("high")


def get_high_trust_contacts(
    *,
    social_map: Optional[List[Dict[str, Any]]] = None,
    config: Optional[dict] = None,
    min_level: str = "high",
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Return entries meeting the min_level trust threshold, sorted by trust then recency.
    """
    entries = social_map if social_map is not None else load_social_map(config)
    min_score = _trust_score(min_level)

    def _entry_key(entry: Dict[str, Any]):
        score = _trust_score(entry.get("trust_hint"))
        ts_raw = entry.get("last_interaction")
        try:
            ts_val = datetime.fromisoformat(ts_raw).timestamp() if ts_raw else 0.0
        except Exception:
            ts_val = 0.0
        return (-score, -ts_val, entry.get("display_name") or entry.get("user_id") or "")

    filtered = [
        entry for entry in entries
        if _trust_score(entry.get("trust_hint")) >= min_score or "trusted" in _normalize_tags(entry.get("tags"))
    ]
    filtered.sort(key=_entry_key)
    return filtered[:limit] if limit else filtered
