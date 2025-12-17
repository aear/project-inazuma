# === emotion_engine.py (Slider-Based Emotion Engine, v2) ===
"""
Emotion Engine

This module defines Ina's *emotional infrastructure* rather than fixed
labels like "joy" or "fear".

Core ideas:
- 24 continuous sliders, each in [-1.0, 1.0].
- Every "emotion state" is a full 24D vector.
- No hard-coded emotion names; interpretation lives elsewhere
  (e.g. eq_engine, emotion_map, meaning_map).
- The engine:
    * Maintains a baseline and current snapshot.
    * Applies gentle drift based on mode (awake/dream/meditation/etc.).
    * Tags memory fragments with the current vector.
    * Logs snapshots over time for later reinterpretation.

It is intentionally conservative: this is the *spine* other systems lean on.
You can extend behaviour (dreamstate drift, symbolic drift, etc.) without
having to change the core structure.
"""

from __future__ import annotations

import json
import random
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from body_schema import get_default_body_schema, snapshot_default_body
from model_manager import (
    load_config,
    update_inastate,
    get_inastate,
)
from gui_hook import log_to_statusbox

# ---------------------------------------------------------------------------
# Slider definition
# ---------------------------------------------------------------------------

# 24-dimensional emotional vector. These are *axes*, not labels.
SLIDERS: List[str] = [
    "intensity",     # overall emotional amplitude
    "attention",     # focus vs scatter
    "trust",
    "care",
    "curiosity",
    "novelty",
    "familiarity",
    "stress",
    "risk",
    "negativity",
    "positivity",
    "simplicity",
    "complexity",
    "interest",
    "clarity",
    "fuzziness",
    "alignment",
    "safety",
    "threat",
    "presence",
    "isolation",
    "connection",
    "ownership",
    "externality",
]

# Reasonable default baseline: mostly neutral, slightly open/curious.
DEFAULT_BASELINE: Dict[str, float] = {
    "intensity": 0.1,
    "attention": 0.0,
    "trust": 0.0,
    "care": 0.0,
    "curiosity": 0.1,
    "novelty": 0.0,
    "familiarity": 0.0,
    "stress": 0.0,
    "risk": -0.1,
    "negativity": 0.0,
    "positivity": 0.0,
    "simplicity": 0.0,
    "complexity": 0.0,
    "interest": 0.0,
    "clarity": 0.0,
    "fuzziness": 0.0,
    "alignment": 0.0,
    "safety": 0.0,
    "threat": 0.0,
    "presence": 0.0,
    "isolation": 0.0,
    "connection": 0.0,
    "ownership": 0.0,
    "externality": 0.0,
}


# ---------------------------------------------------------------------------
# Paths and helpers
# ---------------------------------------------------------------------------

AI_CHILDREN_ROOT = Path("AI_Children")
BODY_INTENSITY_THRESHOLD = 0.6


def _log(msg: str) -> None:
    try:
        log_to_statusbox(f"[Emotion] {msg}")
    except Exception:
        print(f"[Emotion] {msg}")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clamp(v: float, lo: float = -1.0, hi: float = 1.0) -> float:
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def _emotion_dir(child: str) -> Path:
    base = AI_CHILDREN_ROOT / child / "memory"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _baseline_path(child: str) -> Path:
    return _emotion_dir(child) / "emotion_baseline.json"


def _log_path(child: str) -> Path:
    # JSONL-style log of snapshots over time.
    return _emotion_dir(child) / "emotion_log.jsonl"


_BODY_STATE_CACHE: Optional[Dict[str, Dict[str, float]]] = None
_BODY_SCHEMA_WARNED = False


def _set_body_state_cache(state: Dict[str, Dict[str, float]]) -> None:
    global _BODY_STATE_CACHE
    _BODY_STATE_CACHE = copy.deepcopy(state)


def _get_latest_body_state() -> Optional[Dict[str, Dict[str, float]]]:
    """
    Try the in-memory cache, fall back to inastate or neutral snapshot.
    """
    if _BODY_STATE_CACHE:
        return copy.deepcopy(_BODY_STATE_CACHE)

    try:
        state = get_inastate("body_state")
        if isinstance(state, dict):
            _set_body_state_cache(state)
            return copy.deepcopy(state)
    except Exception:
        pass

    try:
        fallback = snapshot_default_body()
        if fallback:
            return fallback
    except Exception:
        pass

    return None


def _update_body_schema_from_emotion(emotion_values: Dict[str, float]) -> Optional[Dict[str, Dict[str, float]]]:
    """
    Drive the shared body schema from the current emotional sliders.
    """
    global _BODY_SCHEMA_WARNED
    schema = get_default_body_schema()
    if schema is None:
        if not _BODY_SCHEMA_WARNED:
            _log("Body schema unavailable; skipping body update.")
            _BODY_SCHEMA_WARNED = True
        return None

    try:
        schema.update_from_emotion(emotion_values)
        snapshot = schema.snapshot()
        _BODY_SCHEMA_WARNED = False
        _set_body_state_cache(snapshot)
        return snapshot
    except Exception as exc:
        if not _BODY_SCHEMA_WARNED:
            _log(f"Failed to update body schema: {exc}")
            _BODY_SCHEMA_WARNED = True
        return None


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class EmotionSnapshot:
    """
    A full 24D emotional vector + minimal context.
    """
    values: Dict[str, float]
    mode: str
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "mode": self.mode,
            "values": self.values,
        }


# ---------------------------------------------------------------------------
# Baseline + snapshot I/O
# ---------------------------------------------------------------------------

def load_baseline(child: str) -> Dict[str, float]:
    """
    Load (or create) the baseline emotional vector for a child.
    """
    path = _baseline_path(child)
    if not path.exists():
        _log(f"No baseline found for {child}, using DEFAULT_BASELINE.")
        return dict(DEFAULT_BASELINE)

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        vec = {k: float(data.get(k, DEFAULT_BASELINE.get(k, 0.0))) for k in SLIDERS}
        return vec
    except Exception as e:
        _log(f"Failed to read baseline for {child}: {e}")
        return dict(DEFAULT_BASELINE)


def save_baseline(child: str, baseline: Dict[str, float]) -> None:
    """
    Persist updated baseline to disk.
    """
    path = _baseline_path(child)
    try:
        cleaned = {k: _clamp(float(baseline.get(k, 0.0))) for k in SLIDERS}
        with path.open("w", encoding="utf-8") as f:
            json.dump(cleaned, f, indent=2, ensure_ascii=False)
    except Exception as e:
        _log(f"Failed to save baseline for {child}: {e}")


def load_last_snapshot(child: str) -> Optional[EmotionSnapshot]:
    """
    Read the most recent snapshot from the log (if available).
    """
    path = _log_path(child)
    if not path.exists():
        return None
    try:
        # Read last non-empty line
        last_line = None
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    last_line = line
        if not last_line:
            return None
        data = json.loads(last_line)
        values = {k: float(data["values"].get(k, 0.0)) for k in SLIDERS}
        mode = data.get("mode", "unknown")
        ts = data.get("timestamp", _now_iso())
        return EmotionSnapshot(values=values, mode=mode, timestamp=ts)
    except Exception as e:
        _log(f"Failed to load last snapshot for {child}: {e}")
        return None


def log_emotion_snapshot(child: str, snapshot: EmotionSnapshot) -> None:
    """
    Append a snapshot to the log file.
    """
    path = _log_path(child)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(snapshot.to_dict(), ensure_ascii=False))
            f.write("\n")
    except Exception as e:
        _log(f"Failed to log emotion snapshot for {child}: {e}")


# ---------------------------------------------------------------------------
# Query helpers (lightweight, no drift)
# ---------------------------------------------------------------------------

def _normalize_values(raw: Dict[str, float]) -> Dict[str, float]:
    return {k: _clamp(float(raw.get(k, 0.0))) for k in SLIDERS}


def get_current_emotion_state(child: Optional[str] = None) -> Dict[str, float]:
    """
    Lightweight accessor used by audio/shadow modules to grab the latest
    24D emotion vector without recomputing drift. Prefers inastate, falls
    back to the last logged snapshot, then baseline.
    """
    try:
        snap = get_inastate("emotion_snapshot")
        if isinstance(snap, dict):
            vals = snap.get("values") if "values" in snap else snap
            if isinstance(vals, dict):
                return _normalize_values(vals)
    except Exception:
        pass

    try:
        cfg = load_config()
        child = child or cfg.get("current_child", "Inazuma_Yagami")
    except Exception:
        child = child or "Inazuma_Yagami"

    last = load_last_snapshot(child)
    if last:
        return _normalize_values(last.values)

    return _normalize_values(load_baseline(child))


# ---------------------------------------------------------------------------
# Drift + update logic
# ---------------------------------------------------------------------------

def _mode_from_inastate() -> str:
    """
    Pull Ina's current mode from inastate if available.
    """
    mode = get_inastate("mode", "awake")
    return mode or "awake"


def _drift_amount_for_mode(mode: str) -> float:
    """
    How "mobile" the emotional vector should be in a given mode.
    """
    mode = (mode or "awake").lower()
    if mode in ("sleep", "dream"):
        return 0.05  # more drift during dreams
    if mode in ("meditation",):
        return 0.03
    if mode in ("boredom", "exploration"):
        return 0.02
    # default: awake / focused
    return 0.01


def _drift_vector(
    baseline: Dict[str, float],
    last_values: Dict[str, float],
    mode: str,
    tags: Optional[List[str]] = None,
    quantum_bias: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Compute a new emotional vector by gently drifting the last snapshot
    toward the baseline and adding some RNG jitter.

    `tags` can bias certain sliders (e.g. trust, curiosity, stress) but
    remains deliberately weak so no single tag dominates.
    """
    tags = tags or []
    drift_strength = _drift_amount_for_mode(mode)

    result: Dict[str, float] = {}
    q_bias = quantum_bias or {}

    for k in SLIDERS:
        base = float(baseline.get(k, 0.0))
        prev = float(last_values.get(k, base))

        # 1) Relax slightly toward baseline
        relaxed = prev + (base - prev) * 0.1

        # 2) Jitter: small random noise
        jitter = (random.random() * 2.0 - 1.0) * drift_strength

        # 3) Tag bias
        bias = 0.0
        if "symbolic_drift" in tags and k in ("fuzziness", "novelty"):
            bias += drift_strength * 0.5
        if "clarity_loss" in tags and k in ("clarity", "fuzziness"):
            bias += (-drift_strength * 0.5 if k == "clarity" else drift_strength * 0.5)
        if "fragment_decay" in tags and k in ("stress", "risk"):
            bias += drift_strength * 0.25
        if "trust_up" in tags and k == "trust":
            bias += drift_strength * 0.5
        if "curiosity_up" in tags and k in ("curiosity", "novelty"):
            bias += drift_strength * 0.5
        if "fear" in tags and k in ("risk", "threat", "negativity", "stress"):
            bias += drift_strength * 0.75

        q_nudge = _clamp(float(q_bias.get(k, 0.0) or 0.0), -0.3, 0.3)
        value = _clamp(relaxed + jitter + bias + q_nudge)
        result[k] = value

    return result


def compute_emotion_snapshot(
    child: str,
    context_tags: Optional[List[str]] = None,
) -> EmotionSnapshot:
    """
    High-level entry point:
    - Load baseline.
    - Pull last snapshot (or baseline).
    - Drift based on mode + tags.
    - Return a new snapshot.
    """
    baseline = load_baseline(child)
    last = load_last_snapshot(child)
    mode = _mode_from_inastate()
    quantum_bias = get_inastate("quantum_emotion_bias") or {}
    if not isinstance(quantum_bias, dict):
        quantum_bias = {}

    if last is None:
        values = {k: _clamp(float(baseline.get(k, 0.0)) + float(quantum_bias.get(k, 0.0) or 0.0)) for k in SLIDERS}
        _log(f"First emotion snapshot for {child} (mode={mode}).")
    else:
        values = _drift_vector(baseline, last.values, mode, tags=context_tags, quantum_bias=quantum_bias)
        _log(f"Drifted emotion snapshot for {child} (mode={mode}).")

    ts = _now_iso()
    snapshot = EmotionSnapshot(values=values, mode=mode, timestamp=ts)
    return snapshot


# ---------------------------------------------------------------------------
# Fragment tagging
# ---------------------------------------------------------------------------

def _should_attach_body_state(values: Dict[str, float]) -> bool:
    try:
        intensity = float(values.get("intensity", 0.0))
    except Exception:
        intensity = 0.0
    return abs(intensity) >= BODY_INTENSITY_THRESHOLD


def _decay_quantum_bias_state(decay: float = 0.35) -> None:
    bias = get_inastate("quantum_emotion_bias")
    if not isinstance(bias, dict) or not bias:
        return
    decayed = {}
    for key, value in bias.items():
        try:
            reduced = float(value) * decay
        except Exception:
            continue
        if abs(reduced) < 0.005:
            continue
        decayed[key] = round(_clamp(reduced), 4)
    update_inastate("quantum_emotion_bias", decayed)


def tag_fragment_emotions(
    fragment: Dict[str, Any],
    child: Optional[str] = None,
    context_tags: Optional[List[str]] = None,
    body_state: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, Any]:
    """
    Convenience wrapper: compute a fresh snapshot and tag the fragment with it.

    If ``child`` is not provided, it will use the current child from config.
    Context tags default to the fragment's tags to bias drift.
    """
    config = load_config()
    active_child = child or config.get("current_child", "default_child")
    tags = context_tags or fragment.get("tags") or []
    snapshot = compute_emotion_snapshot(active_child, context_tags=list(tags))
    return tag_fragment(fragment, snapshot, body_state=body_state or _get_latest_body_state())


def tag_fragment(
    fragment: Dict[str, Any],
    snapshot: EmotionSnapshot,
    body_state: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, Any]:
    """
    Attach the current emotional vector to a fragment.

    We store both:
      - full 24D vector under fragment["emotions"]["sliders"]
      - a shallow "summary" key for quick filtering if needed later

    This function is *pure*: it returns the updated fragment dict.
    """
    emotions = fragment.get("emotions") or {}

    # Full vector
    emotions["sliders"] = {k: float(snapshot.values.get(k, 0.0)) for k in SLIDERS}

    # Simple summaries (can be extended later)
    emotions["summary"] = {
        "intensity": snapshot.values.get("intensity", 0.0),
        "valence": snapshot.values.get("positivity", 0.0) - snapshot.values.get("negativity", 0.0),
        "stress": snapshot.values.get("stress", 0.0),
        "trust": snapshot.values.get("trust", 0.0),
        "novelty": snapshot.values.get("novelty", 0.0),
    }

    fragment["emotions"] = emotions

    if _should_attach_body_state(snapshot.values):
        state_to_attach = body_state if body_state is not None else _get_latest_body_state()
        if state_to_attach is not None:
            fragment["body_state"] = copy.deepcopy(state_to_attach)

    return fragment


def tag_all_fragments(
    child: str,
    snapshot: EmotionSnapshot,
    body_state: Optional[Dict[str, Dict[str, float]]] = None,
) -> None:
    """
    Walk AI_Children/<child>/memory/fragments and tag any fragment JSON
    that does not yet have an 'emotions' field.

    This keeps disk impact modest and avoids rewriting everything constantly.
    """
    fragments_dir = AI_CHILDREN_ROOT / child / "memory" / "fragments"
    if not fragments_dir.exists():
        _log(f"No fragments directory for {child} at {fragments_dir}")
        return

    updated = 0
    skipped = 0

    for fpath in fragments_dir.glob("*.json"):
        try:
            with fpath.open("r", encoding="utf-8") as f:
                frag = json.load(f)
        except Exception as e:
            _log(f"Failed to read fragment {fpath.name}: {e}")
            continue

        if "emotions" in frag:
            skipped += 1
            continue

        frag = tag_fragment(frag, snapshot, body_state=body_state)

        try:
            with fpath.open("w", encoding="utf-8") as f:
                json.dump(frag, f, indent=2, ensure_ascii=False)
            updated += 1
        except Exception as e:
            _log(f"Failed to write updated fragment {fpath.name}: {e}")

    _log(f"Tagged fragments for {child}: updated={updated}, skipped={skipped}")


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

def run_emotion_engine(context_tags: Optional[List[str]] = None) -> None:
    """
    Main entrypoint.

    - Resolve current child from config.
    - Compute a new emotion snapshot.
    - Update inastate.
    - Log the snapshot.
    - Optionally tag fragments.
    """
    try:
        config = load_config()
    except Exception as e:
        _log(f"Failed to load config: {e}")
        config = {}

    child = config.get("current_child", "Inazuma_Yagami")

    snapshot = compute_emotion_snapshot(child, context_tags=context_tags)

    from emotion_processor import process_emotion

    # inside run_emotion_engine(), after snapshot = calculate_emotion_state(fragments)
    processed_values = process_emotion(snapshot, mode=snapshot.mode)
    snapshot.values = processed_values  # keep snapshot metadata/timestamp while using processed sliders
    body_state_snapshot = _update_body_schema_from_emotion(snapshot.values)

    # Map current emotions to nearest symbolic emotions (for multi-neuron style use).
    try:
        from emotion_map import rank_emotion_symbols
        symbol_matches = rank_emotion_symbols(snapshot.values, child=child, top_n=2)
        update_inastate("emotion_symbol_matches", symbol_matches)
    except Exception as e:
        _log(f"Failed to map emotions to symbols: {e}")

    if body_state_snapshot is not None:
        update_inastate("body_state", body_state_snapshot)
        update_inastate("last_body_update", snapshot.timestamp)

    # Update inastate
    update_inastate("emotion_snapshot", snapshot.to_dict())
    update_inastate("last_emotion_update", snapshot.timestamp)
    _decay_quantum_bias_state()

    # Persist
    log_emotion_snapshot(child, snapshot)
    save_baseline(child, snapshot.values)

    # Tag fragments (can be disabled later or made more selective)
    tag_all_fragments(child, snapshot, body_state=body_state_snapshot)

    _log("Emotion snapshot stored and propagated.")


if __name__ == "__main__":
    run_emotion_engine()
