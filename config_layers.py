"""Low-I/O, ownership-aware configuration layers.

``config.json`` remains a read-only compatibility base while callers migrate.
The four small layer files are loaded once per process and refreshed only by an
explicit reload or a successful update through this module.
"""
from __future__ import annotations

from copy import deepcopy
import logging
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Mapping, Tuple

from io_utils import atomic_write_json, load_json_dict


LEGACY_NAME = "config.json"
CORE_NAME = "core.json"
OPERATOR_NAME = "operator.json"
ADAPTIVE_NAME = "adaptive.json"
RUNTIME_NAME = "runtime.json"

_BOUNDS_KEY = "adaptive_bounds"
_logger = logging.getLogger(__name__)
_cache: Dict[Path, Tuple[Dict[str, Any], Dict[str, Any]]] = {}
_lock = RLock()


class AdaptiveConfigError(ValueError):
    """Raised when an adaptive value is not explicitly operator-authorized."""


def _merge(base: Dict[str, Any], overlay: Mapping[str, Any]) -> Dict[str, Any]:
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = deepcopy(value)
    return base


def _flatten(payload: Mapping[str, Any], prefix: str = "") -> Dict[str, Any]:
    leaves: Dict[str, Any] = {}
    for key, value in payload.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            leaves.update(_flatten(value, path))
        else:
            leaves[path] = value
    return leaves


def _rule_accepts(value: Any, rule: Any) -> bool:
    if not isinstance(rule, Mapping):
        return False
    expected = rule.get("type")
    type_ok = {
        "boolean": lambda item: isinstance(item, bool),
        "integer": lambda item: isinstance(item, int) and not isinstance(item, bool),
        "number": lambda item: isinstance(item, (int, float)) and not isinstance(item, bool),
        "string": lambda item: isinstance(item, str),
    }
    if expected not in type_ok or not type_ok[expected](value):
        return False
    if "enum" in rule and value not in rule.get("enum", []):
        return False
    if expected in {"integer", "number"}:
        if "minimum" in rule and value < rule["minimum"]:
            return False
        if "maximum" in rule and value > rule["maximum"]:
            return False
    if expected == "string":
        if "min_length" in rule and len(value) < int(rule["min_length"]):
            return False
        if "max_length" in rule and len(value) > int(rule["max_length"]):
            return False
    return True


def validate_adaptive(adaptive: Mapping[str, Any], operator: Mapping[str, Any]) -> None:
    """Reject every adaptive leaf not covered by an exact operator rule."""
    rules = operator.get(_BOUNDS_KEY)
    rules = rules if isinstance(rules, Mapping) else {}
    for path, value in _flatten(adaptive).items():
        if not _rule_accepts(value, rules.get(path)):
            raise AdaptiveConfigError(f"adaptive value is not permitted: {path}")


def _read_layers(base_dir: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    legacy = load_json_dict(base_dir / LEGACY_NAME)
    core = load_json_dict(base_dir / CORE_NAME)
    operator = load_json_dict(base_dir / OPERATOR_NAME)
    adaptive = load_json_dict(base_dir / ADAPTIVE_NAME)
    try:
        validate_adaptive(adaptive, operator)
    except AdaptiveConfigError as exc:
        # Self-tuning must never gain the power to prevent recovery/startup.
        _logger.warning("Ignoring invalid adaptive configuration: %s", exc)
        adaptive = {}

    # Later layers have stronger ownership. Operator metadata is not runtime
    # policy, and runtime.json is intentionally never merged here.
    effective = deepcopy(legacy)
    _merge(effective, adaptive)
    operator_policy = {key: value for key, value in operator.items() if key != _BOUNDS_KEY}
    _merge(effective, operator_policy)
    _merge(effective, core)
    runtime = load_json_dict(base_dir / RUNTIME_NAME)
    return effective, runtime


def load_config(base_dir: Path = Path("."), *, force_reload: bool = False) -> Dict[str, Any]:
    """Return a defensive copy of the cached effective policy configuration."""
    root = Path(base_dir).resolve()
    with _lock:
        if force_reload or root not in _cache:
            _cache[root] = _read_layers(root)
        return deepcopy(_cache[root][0])


def load_runtime(base_dir: Path = Path("."), *, force_reload: bool = False) -> Dict[str, Any]:
    """Return ephemeral state separately so it cannot silently become policy."""
    root = Path(base_dir).resolve()
    with _lock:
        if force_reload or root not in _cache:
            _cache[root] = _read_layers(root)
        return deepcopy(_cache[root][1])


def reload_config(base_dir: Path = Path(".")) -> Dict[str, Any]:
    return load_config(base_dir, force_reload=True)


def update_adaptive(path: str, value: Any, base_dir: Path = Path(".")) -> bool:
    """Validate and atomically persist one adaptive leaf.

    Returns ``False`` without touching disk when the requested value is already
    present. Successful writes refresh the in-process cache explicitly.
    """
    root = Path(base_dir).resolve()
    target = root / ADAPTIVE_NAME
    with _lock:
        operator = load_json_dict(root / OPERATOR_NAME)
        rules = operator.get(_BOUNDS_KEY)
        rules = rules if isinstance(rules, Mapping) else {}
        if not _rule_accepts(value, rules.get(path)):
            raise AdaptiveConfigError(f"adaptive value is not permitted: {path}")
        adaptive = load_json_dict(target)
        cursor = adaptive
        parts = [part for part in str(path).split(".") if part]
        if not parts:
            raise AdaptiveConfigError("adaptive path must not be empty")
        for part in parts[:-1]:
            existing = cursor.get(part)
            if existing is None:
                cursor[part] = {}
            elif not isinstance(existing, dict):
                raise AdaptiveConfigError(f"adaptive path conflicts at: {part}")
            cursor = cursor[part]
        if cursor.get(parts[-1]) == value:
            return False
        cursor[parts[-1]] = value
        validate_adaptive(adaptive, operator)
        atomic_write_json(target, adaptive, indent=2)
        _cache[root] = _read_layers(root)
        return True


def clear_cache() -> None:
    """Testing and process-lifecycle hook; does no I/O."""
    with _lock:
        _cache.clear()


__all__ = [
    "AdaptiveConfigError",
    "clear_cache",
    "load_config",
    "load_runtime",
    "reload_config",
    "update_adaptive",
    "validate_adaptive",
]
