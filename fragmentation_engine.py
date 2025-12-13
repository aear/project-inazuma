# === fragmentation_engine.py ===
"""
Fragmentation engine for Ina.

Turns pipeline outputs (audio/vision/text/logs) into standardised fragment
dicts and submits them to the memory layer via memory_gatekeeper, with a
safe on-disk fallback.

Design goals:
- Fragments are *lightweight metadata + paths*, not big blobs in RAM.
- Compatible with multiple fragment producers (audio_digest, vision_digest,
  raw_file_manager, boot summaries, etc.).
- Robust to partial installs: if memory_gatekeeper or GUI are missing,
  this still writes disk fragments rather than crashing.
"""

from __future__ import annotations

import copy
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# --- Soft imports: GUI + config + gatekeeper ---------------------------------

try:
    from gui_hook import log_to_statusbox  # type: ignore
except Exception:  # noqa: BLE001
    def log_to_statusbox(msg: str) -> None:  # type: ignore[redefinition]
        # Fallback: just print
        print(msg)


try:
    from model_manager import load_config, get_inastate  # type: ignore
except Exception:  # noqa: BLE001
    def load_config() -> Dict[str, Any]:  # type: ignore[redefinition]
        # Minimal fallback if model_manager isn't available
        return {"current_child": "default_child"}
    def get_inastate(key: str, default=None):  # type: ignore[redefinition]
        return default

try:
    from body_schema import snapshot_default_body  # type: ignore
except Exception:  # noqa: BLE001
    snapshot_default_body = None  # type: ignore


try:
    import memory_gatekeeper  # type: ignore
except Exception:  # noqa: BLE001
    memory_gatekeeper = None  # type: ignore


# === Helpers =================================================================


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _current_child() -> str:
    cfg = load_config()
    return cfg.get("current_child", "default_child")


def _fragments_root(child: Optional[str] = None) -> Path:
    if child is None:
        child = _current_child()
    root = Path("AI_Children") / child / "memory" / "fragments"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _make_fragment_id(prefix: str) -> str:
    """
    Use a human-inspectable prefix plus a compact UUID-based suffix.

    This keeps IDs in line with the existing style like:
    - frag_audio_digest_...
    - frag_vision_...
    - frag_boot_summary_...
    """
    suffix = uuid.uuid4().hex[:12]
    return f"{prefix}_{suffix}"


BODY_INTENSITY_THRESHOLD = 0.6


def _emotion_intensity(emotions: Optional[Dict[str, Any]]) -> float:
    if not isinstance(emotions, dict):
        return 0.0
    values = emotions.get("values") if isinstance(emotions.get("values"), dict) else emotions
    try:
        return float(values.get("intensity", 0.0))
    except Exception:
        return 0.0


def _current_body_state() -> Optional[Dict[str, Dict[str, float]]]:
    """
    Prefer the live body state from inastate; fall back to a neutral snapshot.
    """
    try:
        state = get_inastate("body_state")
        if isinstance(state, dict):
            return copy.deepcopy(state)
    except Exception:
        pass

    if snapshot_default_body is not None:
        try:
            return snapshot_default_body()
        except Exception:
            return None
    return None


def _maybe_attach_body_state(fragment: Dict[str, Any], emotions: Optional[Dict[str, Any]]) -> None:
    if abs(_emotion_intensity(emotions)) < BODY_INTENSITY_THRESHOLD:
        return
    state = _current_body_state()
    if state is not None:
        fragment["body_state"] = state


# === Gatekeeper integration ===================================================


def _submit_to_gatekeeper(fragment: Dict[str, Any], reason: str = "") -> Optional[Path]:
    """
    Try to hand the fragment off to memory_gatekeeper if available.

    We support several possible function names to stay compatible with
    earlier iterations of memory_gatekeeper.py. If none are present,
    fall back to a local on-disk write.
    """
    if memory_gatekeeper is not None:
        try:
            if hasattr(memory_gatekeeper, "submit_fragment"):
                return memory_gatekeeper.submit_fragment(fragment, reason=reason)  # type: ignore[attr-defined]
            if hasattr(memory_gatekeeper, "admit_fragment"):
                return memory_gatekeeper.admit_fragment(fragment, reason=reason)  # type: ignore[attr-defined]
            if hasattr(memory_gatekeeper, "store_fragment"):
                return memory_gatekeeper.store_fragment(fragment, reason=reason)  # type: ignore[attr-defined]
        except Exception as e:  # noqa: BLE001
            log_to_statusbox(f"[Fragments] Gatekeeper error, using fallback: {e}")

    # Fallback: write directly to disk as a normal fragment
    return _write_fragment_direct(fragment, subdir="pending", reason=reason)


def _write_fragment_direct(
    fragment: Dict[str, Any],
    subdir: str = "pending",
    reason: str = "",
) -> Path:
    """
    Minimal on-disk writer. Does not try to be clever: just drops JSON
    into AI_Children/<child>/memory/fragments/<subdir>/<id>.json
    """
    child = _current_child()
    root = _fragments_root(child) / subdir
    root.mkdir(parents=True, exist_ok=True)

    frag_id = fragment.get("id") or _make_fragment_id("frag_generic")
    fragment["id"] = frag_id
    fragment.setdefault("child", child)
    fragment.setdefault("written_at", _now_iso())
    if reason:
        fragment.setdefault("meta", {}).setdefault("reasons", []).append(reason)

    path = root / f"{frag_id}.json"
    try:
        with path.open("w", encoding="utf-8") as f:
            json.dump(fragment, f, indent=2)
        log_to_statusbox(f"[Fragments] Wrote fragment {frag_id} → {path}")
    except Exception as e:  # noqa: BLE001
        log_to_statusbox(f"[Fragments] FAILED to write fragment {frag_id}: {e}")
    return path


def store_fragment(fragment: Dict[str, Any], reason: str = "") -> Optional[Path]:
    """
    Public sink for any producer that already has a fragment dict.

    This enforces:
    - required top-level fields (id/type/source/timestamp),
    - dispatch to memory_gatekeeper or disk fallback.
    """
    # Ensure minimal fields
    fragment.setdefault("timestamp", _now_iso())
    fragment.setdefault("type", "generic")
    fragment.setdefault("source", "unknown")
    fragment.setdefault("tags", [])
    fragment.setdefault("importance", 0.0)
    fragment.setdefault("meta", {})

    if "id" not in fragment:
        fragment["id"] = _make_fragment_id(f"frag_{fragment['type']}")

    return _submit_to_gatekeeper(fragment, reason=reason)


# === Generic fragment builders ===============================================


def make_fragment(
    *,
    frag_type: str,
    source: str,
    summary: str,
    tags: Optional[List[str]] = None,
    importance: float = 0.0,
    emotions: Optional[Dict[str, float]] = None,
    symbols: Optional[List[str]] = None,
    payload: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Generic factory for a standard Ina fragment dict.

    This is intentionally simple and schema-agnostic – downstream systems
    (meaning_map, proto_qualia_engine, etc.) can interpret the fields. The
    important part is consistency and keeping heavy data out of RAM.
    """
    if tags is None:
        tags = []
    if payload is None:
        payload = {}
    if context is None:
        context = {}

    fragment: Dict[str, Any] = {
        "id": _make_fragment_id(f"frag_{frag_type}"),
        "type": frag_type,
        "source": source,
        "timestamp": _now_iso(),
        "summary": summary,
        "tags": tags,
        "importance": float(importance),
        "emotions": emotions or {},      # 24D slider vector lives here
        "symbols": symbols or [],        # any symbol ids / handles
        "payload": payload,              # lightweight, with file paths not blobs
        "context": context,              # extra metadata, not for indexing
    }
    _maybe_attach_body_state(fragment, emotions)
    return fragment


# Convenience wrappers if you want them later ---------------------------------


def fragment_text(
    text: str,
    *,
    origin: str = "raw_file",
    tags: Optional[List[str]] = None,
    importance: float = 0.05,
    emotions: Optional[Dict[str, float]] = None,
    symbols: Optional[List[str]] = None,
    extra_meta: Optional[Dict[str, Any]] = None,
    store: bool = True,
) -> Dict[str, Any]:
    """
    Build a text fragment (e.g. from raw_file_manager, boot logs, notes).

    `text` itself should NOT be massive; for large files, store a short
    excerpt here and keep file paths / offsets in payload.
    """
    if tags is None:
        tags = []
    if extra_meta is None:
        extra_meta = {}

    payload = {
        "text": text,
        "origin": origin,
    }
    fragment = make_fragment(
        frag_type="text",
        source=origin,
        summary=text[:200],
        tags=list(set(tags + ["text"])),
        importance=importance,
        emotions=emotions,
        symbols=symbols,
        payload=payload,
        context=extra_meta,
    )

    if store:
        store_fragment(fragment, reason="text_fragment")
    return fragment


def fragment_audio_digest(
    *,
    clip_path: str,
    label: str,
    analysis: Dict[str, Any],
    tags: Optional[List[str]] = None,
    importance: float = 0.1,
    emotions: Optional[Dict[str, float]] = None,
    symbols: Optional[List[str]] = None,
    store: bool = True,
) -> Dict[str, Any]:
    """
    Build a fragment representing a processed audio clip.

    Intended to be called from audio_digest.generate_fragment(...) or similar.
    `analysis` can contain whatever the audio pipeline extracted: MFCCs,
    loudness stats, speech detection flags, etc.
    """
    if tags is None:
        tags = []
    tags = list(set(tags + ["audio", "audio_digest", label]))

    payload = {
        "clip_path": clip_path,
        "device_label": label,
        "analysis": analysis,
    }

    summary = analysis.get("summary") or f"Audio clip from {label} ({clip_path})"

    fragment = make_fragment(
        frag_type="audio_digest",
        source=f"audio:{label}",
        summary=summary,
        tags=tags,
        importance=importance,
        emotions=emotions,
        symbols=symbols,
        payload=payload,
        context={},
    )

    if store:
        store_fragment(fragment, reason="audio_digest")
    return fragment


# === Device log fragmentation ================================================


def fragment_device_log(
    label: str,
    *,
    importance: float = 0.05,
    emotions: Optional[Dict[str, float]] = None,
    symbols: Optional[List[str]] = None,
    store: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Consolidate the audio_session log for a given device label into a
    single lightweight fragment.

    Expects audio_listener.py to be writing:
    AI_Children/<child>/memory/audio_session/<label>_log.json

    Each log entry:
    - path: str (clip path)
    - timestamp: ISO-8601
    - duration: seconds

    We do *not* load the audio; we just build a summary + metadata.
    """
    child = _current_child()
    log_dir = Path("AI_Children") / child / "memory" / "audio_session"
    log_file = log_dir / f"{label}_log.json"

    if not log_file.exists():
        log_to_statusbox(f"[Fragments] No session log for {label} at {log_file}")
        return None

    try:
        with log_file.open("r", encoding="utf-8") as f:
            entries: List[Dict[str, Any]] = json.load(f)
    except Exception as e:  # noqa: BLE001
        log_to_statusbox(f"[Fragments] Failed to read device log {log_file}: {e}")
        return None

    if not entries:
        log_to_statusbox(f"[Fragments] Empty session log for {label}")
        return None

    # Basic stats
    count = len(entries)
    total_duration = sum(float(e.get("duration", 0.0)) for e in entries)
    timestamps = sorted(e.get("timestamp") for e in entries if e.get("timestamp"))
    first_ts = timestamps[0] if timestamps else None
    last_ts = timestamps[-1] if timestamps else None

    summary = (
        f"Audio session summary for {label}: {count} clips, "
        f"{int(total_duration)}s total, "
        f"from {first_ts or 'unknown'} to {last_ts or 'unknown'}."
    )

    tags = ["audio_session", label]
    payload = {
        "device_label": label,
        "entries": entries,
        "clip_count": count,
        "total_duration": total_duration,
        "first_timestamp": first_ts,
        "last_timestamp": last_ts,
        "log_file": str(log_file),
    }

    fragment = make_fragment(
        frag_type="audio_session",
        source=f"audio_session:{label}",
        summary=summary,
        tags=tags,
        importance=importance,
        emotions=emotions,
        symbols=symbols,
        payload=payload,
        context={},
    )

    if store:
        store_fragment(fragment, reason="audio_session_log")
    return fragment


# === (Optional) vision/session helpers =======================================


def fragment_vision_summary(
    *,
    frames: List[Dict[str, Any]],
    tags: Optional[List[str]] = None,
    importance: float = 0.1,
    emotions: Optional[Dict[str, float]] = None,
    symbols: Optional[List[str]] = None,
    store: bool = True,
) -> Dict[str, Any]:
    """
    Generic hook for vision_digest.py or future vision_window logging to
    summarise a batch of screenshots / webcam frames.

    `frames` is expected to contain *paths* or pre-computed features, not raw pixels.
    """
    if tags is None:
        tags = []
    tags = list(set(tags + ["vision"]))

    count = len(frames)
    paths = [f.get("path") for f in frames if f.get("path")]
    summary = f"Vision batch: {count} frames ({len(paths)} paths recorded)."

    payload = {
        "frames": frames,
        "frame_count": count,
        "paths": paths,
    }

    fragment = make_fragment(
        frag_type="vision_batch",
        source="vision_digest",
        summary=summary,
        tags=tags,
        importance=importance,
        emotions=emotions,
        symbols=symbols,
        payload=payload,
        context={},
    )

    if store:
        store_fragment(fragment, reason="vision_batch")
    return fragment


if __name__ == "__main__":
    # Tiny manual smoke test: create a dummy text fragment and write it.
    frag = fragment_text(
        "Fragmentation engine self-test fragment.",
        origin="fragmentation_engine.selftest",
        tags=["selftest"],
        store=True,
    )
    print("Self-test fragment id:", frag["id"])
