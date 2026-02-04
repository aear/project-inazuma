from __future__ import annotations

import time
from typing import Any, Dict, Iterable, Optional, Tuple

try:
    import psutil  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    psutil = None

try:
    from model_manager import load_config, get_inastate, update_inastate  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    def load_config() -> Dict[str, Any]:  # type: ignore[redefinition]
        return {}

    def get_inastate(key: str, default=None):  # type: ignore[redefinition]
        return default

    def update_inastate(key: str, value) -> None:  # type: ignore[redefinition]
        return None


_FRAGMENT_LIMITS_DEFAULTS = {
    "enabled": True,
    "window_minutes": 60,
    "max_per_window": 300,
    "drop_on_memory_pressure": "hard",
    "allow_tags": ["system_event", "self_reflection", "high_emotion", "critical", "exception"],
}

_MEMORY_GUARD_DEFAULTS = {
    "enabled": True,
    "ram_soft_percent": 35.0,
    "ram_hard_percent": 45.0,
    "swap_soft_percent": 5.0,
    "swap_hard_percent": 10.0,
    "min_available_gb": 8.0,
}

_FRAGMENT_STATE_KEY = "fragment_budget"
_memory_guard_cache: Dict[str, Any] = {"level": "unknown", "timestamp": 0.0}
_fragment_state_cache: Dict[str, Any] = {}


def _coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return default


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_tags(tags: Optional[Iterable[str]]) -> set[str]:
    if not tags:
        return set()
    return {str(tag).strip().lower() for tag in tags if tag}


def _extract_fragment_tags(fragment: Optional[Dict[str, Any]]) -> set[str]:
    if not isinstance(fragment, dict):
        return set()
    tags = _normalize_tags(fragment.get("tags"))
    meta = fragment.get("meta")
    if isinstance(meta, dict):
        tags.update(_normalize_tags(meta.get("flags")))
        tags.update(_normalize_tags(meta.get("tags")))
    metadata = fragment.get("metadata")
    if isinstance(metadata, dict):
        tags.update(_normalize_tags(metadata.get("flags")))
        tags.update(_normalize_tags(metadata.get("tags")))
    return tags


def _load_fragment_limits(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = cfg or load_config()
    raw = cfg.get("fragment_limits") if isinstance(cfg, dict) else None
    limits = _FRAGMENT_LIMITS_DEFAULTS.copy()
    if isinstance(raw, dict):
        if "enabled" in raw:
            limits["enabled"] = _coerce_bool(raw.get("enabled"), limits["enabled"])
        if "window_minutes" in raw:
            limits["window_minutes"] = max(0, _coerce_int(raw.get("window_minutes"), limits["window_minutes"]))
        if "max_per_window" in raw:
            limits["max_per_window"] = max(0, _coerce_int(raw.get("max_per_window"), limits["max_per_window"]))
        if "drop_on_memory_pressure" in raw:
            drop_value = raw.get("drop_on_memory_pressure")
            if isinstance(drop_value, str):
                limits["drop_on_memory_pressure"] = drop_value.strip().lower()
            elif isinstance(drop_value, bool):
                limits["drop_on_memory_pressure"] = "hard" if drop_value else "off"
        if "allow_tags" in raw and isinstance(raw.get("allow_tags"), (list, tuple)):
            limits["allow_tags"] = [str(tag) for tag in raw.get("allow_tags", []) if tag]
    return limits


def _load_memory_guard_limits(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = cfg or load_config()
    raw = cfg.get("memory_guard") if isinstance(cfg, dict) else None
    limits = _MEMORY_GUARD_DEFAULTS.copy()
    if isinstance(raw, dict):
        if "enabled" in raw:
            limits["enabled"] = _coerce_bool(raw.get("enabled"), limits["enabled"])
        for key in ("ram_soft_percent", "ram_hard_percent", "swap_soft_percent", "swap_hard_percent"):
            if key in raw:
                limits[key] = max(0.0, min(100.0, _coerce_float(raw.get(key), limits[key])))
        if "min_available_gb" in raw:
            limits["min_available_gb"] = max(0.0, _coerce_float(raw.get("min_available_gb"), limits["min_available_gb"]))
    limits["ram_soft_percent"] = min(limits["ram_soft_percent"], limits["ram_hard_percent"])
    limits["swap_soft_percent"] = min(limits["swap_soft_percent"], limits["swap_hard_percent"])
    return limits


def get_memory_guard_level() -> str:
    now = time.time()
    if (now - _memory_guard_cache.get("timestamp", 0.0)) < 5.0:
        return str(_memory_guard_cache.get("level", "unknown"))

    state = get_inastate("memory_guard") if callable(get_inastate) else None
    if isinstance(state, dict):
        level = str(state.get("level", "")).lower()
        if level in {"soft", "hard", "disabled"}:
            _memory_guard_cache.update({"level": level, "timestamp": now})
            return level

    if psutil is None:
        _memory_guard_cache.update({"level": "unknown", "timestamp": now})
        return "unknown"

    limits = _load_memory_guard_limits()
    if not limits.get("enabled", True):
        _memory_guard_cache.update({"level": "disabled", "timestamp": now})
        return "disabled"

    try:
        vm = psutil.virtual_memory()
        swap = psutil.swap_memory()
    except Exception:
        _memory_guard_cache.update({"level": "unknown", "timestamp": now})
        return "unknown"

    available_gb = vm.available / (1024.0 ** 3)
    if (
        limits["min_available_gb"] > 0
        and available_gb <= limits["min_available_gb"]
        or (limits["ram_hard_percent"] > 0 and vm.percent >= limits["ram_hard_percent"])
        or (limits["swap_hard_percent"] > 0 and swap.percent >= limits["swap_hard_percent"])
    ):
        _memory_guard_cache.update({"level": "hard", "timestamp": now})
        return "hard"

    if (
        (limits["ram_soft_percent"] > 0 and vm.percent >= limits["ram_soft_percent"])
        or (limits["swap_soft_percent"] > 0 and swap.percent >= limits["swap_soft_percent"])
    ):
        _memory_guard_cache.update({"level": "soft", "timestamp": now})
        return "soft"

    _memory_guard_cache.update({"level": "ok", "timestamp": now})
    return "ok"


def should_accept_fragment(
    fragment: Optional[Dict[str, Any]] = None,
    *,
    tags: Optional[Iterable[str]] = None,
) -> Tuple[bool, str]:
    policy = _load_fragment_limits()
    if not policy.get("enabled", True):
        return True, "limits_disabled"

    all_tags = _normalize_tags(tags)
    all_tags.update(_extract_fragment_tags(fragment))
    allow_tags = _normalize_tags(policy.get("allow_tags"))
    if allow_tags and all_tags.intersection(allow_tags):
        return True, "allow_tag"

    drop_on_pressure = str(policy.get("drop_on_memory_pressure", "off")).lower()
    if drop_on_pressure in {"soft", "hard"}:
        level = get_memory_guard_level()
        if level == "hard" or (level == "soft" and drop_on_pressure == "soft"):
            _note_fragment_drop(reason=f"memory_guard_{level}")
            return False, f"memory_guard_{level}"

    window_minutes = max(0, int(policy.get("window_minutes") or 0))
    max_per_window = max(0, int(policy.get("max_per_window") or 0))
    if window_minutes <= 0 or max_per_window <= 0:
        return True, "no_rate_limit"

    window_seconds = window_minutes * 60
    state = _load_fragment_state(window_seconds)
    if state["count"] >= max_per_window:
        _note_fragment_drop(reason="rate_limit")
        return False, "rate_limit"

    state["count"] += 1
    _save_fragment_state(state)
    return True, "accepted"


def _load_fragment_state(window_seconds: int) -> Dict[str, Any]:
    now = time.time()
    state = {}
    if callable(get_inastate):
        state = get_inastate(_FRAGMENT_STATE_KEY) or {}
    if not isinstance(state, dict):
        state = {}
    if not state:
        state = _fragment_state_cache.copy()
    start = float(state.get("window_start", 0.0) or 0.0)
    if start <= 0 or (now - start) >= window_seconds:
        state = {"window_start": now, "count": 0, "dropped": 0, "window_seconds": window_seconds}
    return state


def _save_fragment_state(state: Dict[str, Any]) -> None:
    _fragment_state_cache.update(state)
    if callable(update_inastate):
        update_inastate(_FRAGMENT_STATE_KEY, state)


def _note_fragment_drop(reason: str) -> None:
    state = _load_fragment_state(int(_fragment_state_cache.get("window_seconds", 0) or 3600))
    state["dropped"] = int(state.get("dropped", 0) or 0) + 1
    state["last_drop_reason"] = reason
    _save_fragment_state(state)
