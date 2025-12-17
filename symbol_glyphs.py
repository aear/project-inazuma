"""
symbol_glyphs.py
----------------

Runtime-editable glyph registry so Ina can redefine the symbols used for
emotion/modulation/concept parts without touching code.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional, Tuple

DEFAULT_SYMBOL_GLYPHS: Dict[str, Dict[str, str]] = {
    "emotion": {
        "calm": "∘",
        "tension": "∇",
        "curiosity": "μ",
        "trust": "λ",
        "fear": "ψ",
        "anger": "Ω",
        "care": "σ",
    },
    "modulation": {
        "soft": "·",
        "moderate": "⇌",
        "sharp": "∴",
        "pulse": "∆",
        "spiral": "⊙",
    },
    "concept": {
        "self": "ν",
        "pattern": "Ξ",
        "truth": "φ",
        "change": "∵",
        "unknown": "∅",
        "connection": "∪",
    },
}

GLYPH_TYPES: Tuple[str, ...] = ("emotion", "modulation", "concept")
_CACHE: Dict[str, Dict[str, object]] = {}


def _resolve_child(child: Optional[str]) -> str:
    if child:
        return child
    config_path = Path("config.json")
    if config_path.exists():
        try:
            with config_path.open("r", encoding="utf-8") as fh:
                cfg = json.load(fh)
                if isinstance(cfg, dict):
                    return cfg.get("current_child", "Inazuma_Yagami")
        except Exception:
            pass
    return "Inazuma_Yagami"


def _glyph_file(child: str) -> Path:
    return Path("AI_Children") / child / "memory" / "symbol_glyphs.json"


def _load_overrides(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    filtered: Dict[str, Dict[str, str]] = {}
    for gtype, entries in data.items():
        if gtype not in GLYPH_TYPES or not isinstance(entries, dict):
            continue
        filtered[gtype] = {str(k): str(v) for k, v in entries.items()}
    return filtered


def _write_overrides(child: str, overrides: Dict[str, Dict[str, str]]) -> None:
    path = _glyph_file(child)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {gtype: entries for gtype, entries in overrides.items() if entries}
    if payload:
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
    elif path.exists():
        path.unlink()
    _CACHE.pop(child, None)


def get_symbol_glyph_maps(child: Optional[str] = None) -> Dict[str, Dict[str, str]]:
    """
    Return merged glyph maps (defaults + overrides) for the selected child.
    """
    active_child = _resolve_child(child)
    path = _glyph_file(active_child)
    mtime = path.stat().st_mtime if path.exists() else None
    cached = _CACHE.get(active_child)
    if cached and cached.get("mtime") == mtime:
        return deepcopy(cached["maps"])  # type: ignore[arg-type]

    base = {gtype: dict(entries) for gtype, entries in DEFAULT_SYMBOL_GLYPHS.items()}
    overrides = _load_overrides(path)
    for gtype, entries in overrides.items():
        base[gtype].update(entries)

    _CACHE[active_child] = {"mtime": mtime, "maps": base}
    return deepcopy(base)


def list_symbol_glyphs(child: Optional[str] = None) -> Dict[str, Dict[str, str]]:
    """
    Convenience alias for get_symbol_glyph_maps; returns a copy so callers can
    inspect the current mapping.
    """
    return get_symbol_glyph_maps(child)


def update_symbol_glyph(
    glyph_type: str,
    key: str,
    glyph: str,
    *,
    child: Optional[str] = None,
) -> Dict[str, Dict[str, str]]:
    """
    Override a single glyph value (e.g., set emotion 'care' to 'care').
    Returns the updated maps.
    """
    if glyph_type not in GLYPH_TYPES:
        raise ValueError(f"glyph_type must be one of {GLYPH_TYPES}")
    active_child = _resolve_child(child)
    current_overrides = _load_overrides(_glyph_file(active_child))
    bucket = current_overrides.setdefault(glyph_type, {})
    bucket[str(key)] = str(glyph)
    _write_overrides(active_child, current_overrides)
    return get_symbol_glyph_maps(active_child)


def remove_symbol_glyph(
    glyph_type: str,
    key: str,
    *,
    child: Optional[str] = None,
) -> Dict[str, Dict[str, str]]:
    """
    Remove an override so the glyph falls back to the default definition.
    """
    if glyph_type not in GLYPH_TYPES:
        raise ValueError(f"glyph_type must be one of {GLYPH_TYPES}")
    active_child = _resolve_child(child)
    current_overrides = _load_overrides(_glyph_file(active_child))
    bucket = current_overrides.get(glyph_type)
    if bucket and key in bucket:
        bucket.pop(key, None)
        if not bucket:
            current_overrides.pop(glyph_type, None)
        _write_overrides(active_child, current_overrides)
    return get_symbol_glyph_maps(active_child)


def reset_symbol_glyphs(
    *,
    child: Optional[str] = None,
    glyph_type: Optional[str] = None,
) -> Dict[str, Dict[str, str]]:
    """
    Reset all overrides (or only a single glyph type) back to defaults.
    """
    active_child = _resolve_child(child)
    if glyph_type is not None and glyph_type not in GLYPH_TYPES:
        raise ValueError(f"glyph_type must be one of {GLYPH_TYPES}")

    path = _glyph_file(active_child)
    if not path.exists():
        return get_symbol_glyph_maps(active_child)

    if glyph_type is None:
        path.unlink(missing_ok=True)
    else:
        overrides = _load_overrides(path)
        overrides.pop(glyph_type, None)
        _write_overrides(active_child, overrides)

    _CACHE.pop(active_child, None)
    return get_symbol_glyph_maps(active_child)
