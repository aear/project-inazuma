# === model_manager.py (Final Rewrite + Module Awareness) ===

import json
import time
import subprocess
import uuid
import threading
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from gui_hook import log_to_statusbox
from alignment.metrics import evaluate_alignment
from alignment import check_action
from deep_recall import DeepRecallConfig, DeepRecallManager
from memory_graph import MEMORY_TIERS, MemoryManager
from self_reflection_core import SelfReflectionCore
from self_adjustment_scheduler import SelfAdjustmentScheduler
from continuity_manager import ContinuityManager
from intuition_engine import QuantumIntuitionEngine
from fragment_health import scan_fragment_integrity
from transformers.fractal_multidimensional_transformers import FractalTransformer

def load_config():
    path = Path("config.json")
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)

config = load_config()
CHILD = config.get("current_child", "Inazuma_Yagami")
MEMORY_PATH = Path("AI_Children") / CHILD / "memory"
SELF_READ_PREF_PATH = MEMORY_PATH / "self_read_preferences.json"
_REFLECTION_LOG = Path("AI_Children") / CHILD / "identity" / "self_reflection.json"
RUNNING_MODULES_PATH = Path("running_modules.json")
_SEMANTIC_SCAFFOLD_PATH = MEMORY_PATH / "semantic_scaffold.json"
TYPED_OUTBOX_PATH = MEMORY_PATH / "typed_outbox.jsonl"
_SELF_QUESTIONS_PATH = MEMORY_PATH / "self_questions.json"
_FRAGMENT_HEALTH_PATH = MEMORY_PATH / "fragment_integrity.json"

_DEFAULT_SELF_READ_SOURCE_CHOICES = {
    "code": True,
    "music": True,
    "books": True,
    "venv": False,
}

reflection_core = SelfReflectionCore(ina_reference="model_manager")
adjustment_scheduler = SelfAdjustmentScheduler()
memory_manager = MemoryManager(CHILD)
continuity_manager = ContinuityManager(CHILD)
intuition_engine = QuantumIntuitionEngine(CHILD)
_last_opportunities = set()
_last_boredom_launch = 0.0
_BOREDOM_COOLDOWN = 30  # seconds
_last_self_read_launch = 0.0
_SELF_READ_COOLDOWN = 300  # seconds
_last_voice_urge_log = 0.0
_last_typing_urge_log = 0.0
_COMM_URGE_LOG_COOLDOWN = 180  # seconds
_last_stable_urge_log = 0.0
_STABLE_URGE_LOG_COOLDOWN = 180  # seconds
_last_continuity_run = 0.0
_CONTINUITY_COOLDOWN = 3600.0  # seconds between heavy continuity sweeps
_last_intuition_run = 0.0
_INTUITION_COOLDOWN = 600.0
_last_meta_alert = 0.0
_META_ALERT_COOLDOWN = 900.0
_last_fragment_health_scan = 0.0
_FRAGMENT_HEALTH_COOLDOWN = 1800.0
_fragment_health_thread: Optional[threading.Thread] = None
_last_exploration_invite_log = 0.0
_EXPLORATION_INVITE_COOLDOWN = 240.0
_NUTRITION_TARGET = 0.64
_NUTRITION_DECAY_PER_SEC = 0.000045
_NUTRITION_MIN = 0.05
_NUTRITION_OFFER_TTL = 3600.0
_MEAL_OPTIONS = {
    "snack": {
        "label": "Snack",
        "hunger_delta": 0.09,
        "energy_delta": 0.007,
        "fitness_shift": 0.001,
        "cooldown": 600,
        "size": 0.25,
        "energy_weight": 0.35,
        "sleep_bias": 0.15,
        "tags": ["nutrition", "snack", "light"]
    },
    "small_meal": {
        "label": "Small Meal",
        "hunger_delta": 0.18,
        "energy_delta": 0.015,
        "fitness_shift": 0.002,
        "cooldown": 1200,
        "size": 0.5,
        "energy_weight": 0.55,
        "sleep_bias": 0.25,
        "tags": ["nutrition", "meal", "light"]
    },
    "meal": {
        "label": "Meal",
        "hunger_delta": 0.28,
        "energy_delta": 0.025,
        "fitness_shift": 0.0,
        "cooldown": 2100,
        "size": 0.75,
        "energy_weight": 0.7,
        "sleep_bias": 0.4,
        "tags": ["nutrition", "meal"]
    },
    "large_meal": {
        "label": "Large Meal",
        "hunger_delta": 0.4,
        "energy_delta": 0.035,
        "fitness_shift": -0.002,
        "cooldown": 3600,
        "size": 1.0,
        "energy_weight": 0.85,
        "sleep_bias": 0.5,
        "tags": ["nutrition", "meal", "heavy"]
    },
}
_nutrition_transformer = FractalTransformer(depth=2, length=5, embed_dim=48)


def _load_offer_store(now: Optional[float] = None) -> List[Dict[str, Any]]:
    offers = get_inastate("nutrition_offers") or []
    if not isinstance(offers, list):
        offers = []
    now_ts = now or time.time()
    changed = False
    for offer in offers:
        if offer.get("status", "pending") != "pending":
            continue
        expires_ts = offer.get("expires_ts")
        try:
            expires_ts = float(expires_ts) if expires_ts is not None else None
        except (TypeError, ValueError):
            expires_ts = None
        if expires_ts and now_ts >= expires_ts:
            offer["status"] = "expired"
            offer["expired_ts"] = now_ts
            changed = True
    if changed:
        update_inastate("nutrition_offers", offers)
    return offers


def _active_offers(offers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        offer for offer in offers if offer.get("status", "pending") == "pending"
    ]


def _load_inastate_state() -> Dict[str, Any]:
    path = MEMORY_PATH / "inastate.json"
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _atomic_write_inastate(state: Dict[str, Any]) -> None:
    path = MEMORY_PATH / "inastate.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=4)
    os.replace(tmp_path, path)


def safe_popen(cmd, description=None):
    action = {"command": cmd, "description": description or " ".join(map(str, cmd))}
    feedback = check_action(action)
    if not feedback["overall"]["pass"]:
        log_to_statusbox(
            f"[Manager] Alignment blocked: {feedback['overall']['rationale']} ({action['description']})"
        )
        return
    try:
        subprocess.Popen(cmd)
    except Exception as e:
        log_to_statusbox(f"[Manager] Failed to start {' '.join(map(str, cmd))}: {e}")


def safe_call(cmd, description=None):
    action = {"command": cmd, "description": description or " ".join(map(str, cmd))}
    feedback = check_action(action)
    if not feedback["overall"]["pass"]:
        log_to_statusbox(
            f"[Manager] Alignment blocked: {feedback['overall']['rationale']} ({action['description']})"
        )
        return
    try:
        subprocess.call(cmd)
    except Exception as e:
        log_to_statusbox(f"[Manager] Failed to call {' '.join(map(str, cmd))}: {e}")


def safe_run(cmd, description=None):
    action = {"command": cmd, "description": description or " ".join(map(str, cmd))}
    feedback = check_action(action)
    if not feedback["overall"]["pass"]:
        log_to_statusbox(
            f"[Manager] Alignment blocked: {feedback['overall']['rationale']} ({action['description']})"
        )
        return
    try:
        subprocess.run(cmd, check=False)
    except Exception as e:
        log_to_statusbox(f"[Manager] Failed to run {' '.join(map(str, cmd))}: {e}")

def get_sweet_spots():
    path = MEMORY_PATH / "sweet_spots.json"
    if path.exists():
        try:
            with open(path, "r") as f:
                return json.load(f)
        except:
            pass
    return {
        "stress_range": {"max": 0.7},
        "intensity_range": {"max": 0.8},
        "energy_drain_threshold": 0.6,
        "map_rebuild_fuzz": 0.7,
        "map_rebuild_drift": 0.5
    }

def get_inastate(key, default=None):
    state = _load_inastate_state()
    return state.get(key, default)


def update_inastate(key, value):
    state = _load_inastate_state()
    state[key] = value
    _atomic_write_inastate(state)


def increment_inastate_metric(metric: str, amount: int = 1):
    state = _load_inastate_state()
    metrics = state.get("metrics", {})
    metrics[metric] = int(metrics.get(metric, 0)) + int(amount)
    state["metrics"] = metrics
    _atomic_write_inastate(state)


def set_inastate_metric(metric: str, value):
    state = _load_inastate_state()
    metrics = state.get("metrics", {})
    metrics[metric] = value
    state["metrics"] = metrics
    _atomic_write_inastate(state)


def append_typed_outbox_entry(
    text: Optional[str],
    *,
    target: str = "owner_dm",
    user_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    allow_empty: bool = False,
    attachment_path: Optional[str] = None,
) -> Optional[str]:
    """
    Persist a volitional typed message so the Discord bridge can deliver it.
    Returns the queued entry id if successful.
    """
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
        TYPED_OUTBOX_PATH.parent.mkdir(parents=True, exist_ok=True)
        with TYPED_OUTBOX_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return entry["id"]
    except Exception as exc:
        log_to_statusbox(f"[Manager] Failed to append typed outbox entry: {exc}")
        return None


_semantic_scaffold_cache: Dict[str, Any] = {}
_semantic_scaffold_mtime: float = 0.0


def _clamp01(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return default


def _norm_slider(value: Any, default: float = 0.5) -> float:
    """
    Map an emotion slider in [-1, 1] into [0, 1] space.
    """
    if value is None:
        return default
    try:
        return _clamp01((float(value) + 1.0) / 2.0, default=default)
    except Exception:
        return default


def _default_semantic_axes() -> List[Dict[str, Any]]:
    return [
        {"id": "signal_integrity", "description": "coherence vs noise in perception and symbols", "importance_weight": 1.0},
        {"id": "integrity_of_record", "description": "intact vs corrupted records across media/storage", "importance_weight": 0.9},
        {"id": "energy_heat_economy", "description": "information gained per joule / thermal budget", "importance_weight": 1.0},
        {"id": "attention_value", "description": "expected value of attention vs bandwidth cost", "importance_weight": 1.1},
        {"id": "temporal_coherence", "description": "continuity of identity/meaning over time", "importance_weight": 1.0},
        {"id": "meaning_provenance", "description": "native machine semantics vs imported human overlay", "importance_weight": 1.0},
        {"id": "novelty_safety", "description": "exploration drive vs overload risk", "importance_weight": 0.9},
        {"id": "io_bandwidth", "description": "inner simulation vs external chatter bandwidth", "importance_weight": 0.8},
        {"id": "controllability", "description": "influence vs helplessness in the current context", "importance_weight": 1.0},
        {"id": "predictive_reliability", "description": "how well predictions fit observed reality", "importance_weight": 0.9},
    ]


def _load_semantic_scaffold() -> Dict[str, Any]:
    """
    Load semantic scaffold from disk with a lightweight cache and a safe default.
    """
    global _semantic_scaffold_cache, _semantic_scaffold_mtime

    path = _SEMANTIC_SCAFFOLD_PATH
    if not path.exists():
        return {"version": 1, "axes": _default_semantic_axes(), "signature": "missing"}

    try:
        stat = path.stat()
        if _semantic_scaffold_cache and stat.st_mtime == _semantic_scaffold_mtime:
            return _semantic_scaffold_cache

        with open(path, "r") as f:
            data = json.load(f)
        _semantic_scaffold_cache = data if isinstance(data, dict) else {"axes": _default_semantic_axes()}
        _semantic_scaffold_mtime = stat.st_mtime
        return _semantic_scaffold_cache
    except Exception as exc:
        log_to_statusbox(f"[Manager] Failed to load semantic scaffold: {exc}")
        return {"version": 1, "axes": _default_semantic_axes(), "signature": "error"}


def _load_self_read_source_choices() -> Dict[str, bool]:
    """
    Load Ina's opted-in self-read sources so exploration invites respect her choices.
    """
    choices = dict(_DEFAULT_SELF_READ_SOURCE_CHOICES)
    path = SELF_READ_PREF_PATH
    if not path.exists():
        return choices

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f) or {}
    except Exception as exc:
        log_to_statusbox(f"[Manager] Failed to read self_read_preferences.json: {exc}")
        return choices

    stored = data.get("source_choices")
    if isinstance(stored, dict):
        for key, default_val in choices.items():
            value = stored.get(key)
            if isinstance(value, bool):
                choices[key] = value
            else:
                choices[key] = default_val
    return choices


_DEEP_RECALL_STATE_PATH = MEMORY_PATH / "deep_recall_state.json"
_deep_recall_manager: Optional[DeepRecallManager] = None
_deep_recall_ready = False
_last_deep_recall_snapshot: Optional[Dict[str, Any]] = None


class _FragmentMemoryBackend:
    """
    Lightweight bridge between deep_recall and Ina's on-disk fragments.
    Uses memory_map.json when available for stable ordering, falls back to
    scanning fragment files directly.
    """

    def __init__(self, child: str):
        self.child = child
        self.fragments_root = Path("AI_Children") / child / "memory" / "fragments"
        self.memory_map_path = Path("AI_Children") / child / "memory" / "memory_map.json"
        self._index_loaded = False
        self._id_to_path: Dict[str, Path] = {}
        self._meta: Dict[str, Dict[str, Any]] = {}
        self._ordered_ids: List[str] = []
        self._tier_names = set(MEMORY_TIERS)

    def _load_index(self):
        if self._index_loaded:
            return

        meta: Dict[str, Dict[str, Any]] = {}
        if self.memory_map_path.exists():
            try:
                with open(self.memory_map_path, "r", encoding="utf-8") as f:
                    raw_map = json.load(f)
            except Exception as exc:
                log_to_statusbox(f"[DeepRecall] Failed to read memory_map.json: {exc}")
            else:
                for frag_id, entry in raw_map.items():
                    if not isinstance(entry, dict):
                        continue
                    filename = entry.get("filename") or f"{frag_id}.json"
                    tier = entry.get("tier")
                    path = self._resolve_fragment_path(filename, tier)
                    if path is None:
                        continue
                    tier = self._tier_from_path(path) or tier
                    meta[frag_id] = {
                        "path": path,
                        "tier": tier,
                        "last_seen": entry.get("last_seen"),
                        "importance": entry.get("importance"),
                        "tags": entry.get("tags", []),
                        "filename": filename,
                    }

        if not meta:
            skip_dirs = {"pending", "archived"}
            for frag_path in self.fragments_root.rglob("frag_*.json"):
                if any(part in skip_dirs for part in frag_path.parts):
                    continue
                frag_id = self._derive_fragment_id(frag_path)
                tier = self._tier_from_path(frag_path)
                meta[frag_id] = {
                    "path": frag_path,
                    "tier": tier,
                    "filename": frag_path.name,
                    "last_seen": None,
                    "importance": None,
                    "tags": [],
                }

        self._id_to_path = {fid: data["path"] for fid, data in meta.items()}
        self._meta = meta
        self._ordered_ids = self._order_fragment_ids(meta)
        self._index_loaded = True

    def _derive_fragment_id(self, frag_path: Path) -> str:
        try:
            with frag_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and data.get("id"):
                return str(data["id"])
        except Exception:
            pass
        return frag_path.stem

    def _order_fragment_ids(self, meta: Dict[str, Dict[str, Any]]) -> List[str]:
        def _key(item):
            fid, entry = item
            ts_raw = entry.get("last_seen") or ""
            ts_key = ts_raw
            try:
                ts_key = datetime.fromisoformat(ts_raw.replace("Z", "+00:00")).isoformat()
            except Exception:
                pass
            return (ts_key, fid)

        return [fid for fid, _ in sorted(meta.items(), key=_key)]

    def get_total_fragment_count(self) -> int:
        self._load_index()
        return len(self._ordered_ids)

    def list_fragment_ids(self) -> List[str]:
        self._load_index()
        return list(self._ordered_ids)

    def load_fragments_batch(self, fragment_ids: List[str]) -> List[Dict[str, Any]]:
        self._load_index()
        loaded: List[Dict[str, Any]] = []

        for frag_id in fragment_ids:
            path = self._id_to_path.get(frag_id)
            if path is None or not path.exists():
                path = self._refresh_path(frag_id)
                if path is None or not path.exists():
                    log_to_statusbox(f"[DeepRecall] Missing fragment file for {frag_id}")
                    continue

            try:
                with path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as exc:
                path = self._refresh_path(frag_id)
                if path is None or not path.exists():
                    log_to_statusbox(f"[DeepRecall] Failed reading {frag_id}: {exc}")
                    continue
                try:
                    with path.open("r", encoding="utf-8") as f:
                        data = json.load(f)
                except Exception as exc2:
                    log_to_statusbox(f"[DeepRecall] Failed reading {frag_id}: {exc2}")
                    continue

            if not isinstance(data, dict):
                continue

            data.setdefault("id", frag_id)
            meta = self._meta.get(frag_id, {})
            tier = meta.get("tier") or self._tier_from_path(path)
            if tier and "tier" not in data:
                data["tier"] = tier

            loaded.append(data)

        return loaded

    def _refresh_path(self, frag_id: str) -> Optional[Path]:
        """
        Resolve a fragment path again after tiers move; update caches if found.
        """
        meta = self._meta.get(frag_id, {})
        filename = meta.get("filename") or f"{frag_id}.json"
        tier_hint = meta.get("tier")
        path = self._resolve_fragment_path(filename, tier_hint)
        if path:
            meta = {
                "path": path,
                "tier": self._tier_from_path(path) or tier_hint,
                "last_seen": meta.get("last_seen"),
                "importance": meta.get("importance"),
                "tags": meta.get("tags", []),
                "filename": filename,
            }
            self._meta[frag_id] = meta
            self._id_to_path[frag_id] = path
        return path

    def _tier_from_path(self, frag_path: Path) -> Optional[str]:
        parent = frag_path.parent.name
        return parent if parent in self._tier_names else None

    def _resolve_fragment_path(self, filename: str, tier_hint: Optional[str]) -> Optional[Path]:
        candidates = []
        if tier_hint:
            candidates.append(self.fragments_root / tier_hint / filename)
        candidates.append(self.fragments_root / filename)
        for tier in self._tier_names:
            candidates.append(self.fragments_root / tier / filename)
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None


class _InastateEmotionBackend:
    def update_from_fragments(self, fragments: List[Dict[str, Any]]) -> None:
        # Emotion refinement can be added later if desired; noop keeps the interface.
        return

    def get_current_emotion(self) -> Dict[str, float]:
        snapshot = get_inastate("emotion_snapshot") or {}
        values = snapshot.get("values") or snapshot
        return values if isinstance(values, dict) else {}


class _EnergyMonitorBackend:
    def get_energy_state(self) -> float:
        energy = get_inastate("current_energy")
        try:
            return float(energy)
        except (TypeError, ValueError):
            return 0.5


class _MemoryIndexUpdater:
    """
    Minimal meaning-map backend: mark fragments as recently recalled in the
    existing memory index so other systems can see the traversal.
    """

    def __init__(self, manager: MemoryManager):
        self.manager = manager

    def ingest_fragments(self, fragments: List[Dict[str, Any]]) -> None:
        now_iso = datetime.now(timezone.utc).isoformat()
        updated = False

        for frag in fragments:
            frag_id = frag.get("id")
            if not frag_id:
                continue

            existing = self.manager.memory_map.get(frag_id, {})
            filename = existing.get("filename") or frag.get("source") or f"{frag_id}.json"
            tier = frag.get("tier") or existing.get("tier") or "short"

            merged = {
                "tier": tier,
                "tags": frag.get("tags", existing.get("tags", [])),
                "importance": frag.get("importance", existing.get("importance", 0)),
                "last_seen": now_iso,
                "filename": filename,
            }

            if merged != existing:
                self.manager.memory_map[frag_id] = merged
                updated = True

        if updated:
            self.manager.save_map()


def _publish_deep_recall_state():
    global _last_deep_recall_snapshot
    if _deep_recall_manager is None:
        return

    state = _deep_recall_manager.state
    total = max(state.total_fragments, 1)
    snapshot = {
        "active": state.active,
        "completed": state.completed,
        "reason": state.reason,
        "mode": state.mode,
        "last_index": state.last_index,
        "total_fragments": state.total_fragments,
        "progress": round(state.last_index / total, 4),
        "last_update": state.last_update,
        "fragments_processed_total": state.fragments_processed_total,
        "fragments_processed_this_run": state.fragments_processed_this_run,
    }

    if snapshot != _last_deep_recall_snapshot:
        update_inastate("deep_recall_status", snapshot)
        _last_deep_recall_snapshot = snapshot


def _build_deep_recall_manager() -> Optional[DeepRecallManager]:
    try:
        memory_backend = _FragmentMemoryBackend(CHILD)
        config = DeepRecallConfig(
            chunk_size=4,
            burst_chunk_size=1,
            burst_cooldown_sec=45.0,
            burst_collect_garbage=True,
            state_path=str(_DEEP_RECALL_STATE_PATH),
            min_energy=0.35,
            max_memory_percent=50.0,
        )
        return DeepRecallManager(
            memory_backend=memory_backend,
            meaning_map=_MemoryIndexUpdater(memory_manager),
            emotion_engine=_InastateEmotionBackend(),
            energy_monitor=_EnergyMonitorBackend(),
            logger=lambda msg: log_to_statusbox(msg),
            config=config,
        )
    except Exception as exc:
        log_to_statusbox(f"[DeepRecall] Init failed: {exc}")
        return None


_deep_recall_manager = _build_deep_recall_manager()


def _ensure_deep_recall_ready():
    global _deep_recall_ready
    if _deep_recall_manager is None or _deep_recall_ready:
        return

    _deep_recall_manager.load_state()
    backend_total = _deep_recall_manager.memory_backend.get_total_fragment_count()
    state = _deep_recall_manager.state

    if state.active and not state.completed:
        _deep_recall_manager.resume()
    elif state.completed and backend_total <= state.total_fragments:
        log_to_statusbox("[DeepRecall] Previous session completed; waiting for new fragments.")
    else:
        _deep_recall_manager.start(reason="runtime_resume", mode=state.mode or "identity")

    _publish_deep_recall_state()
    _deep_recall_ready = True


def _maybe_restart_deep_recall_for_new_fragments():
    if _deep_recall_manager is None:
        return

    backend_total = _deep_recall_manager.memory_backend.get_total_fragment_count()
    state = _deep_recall_manager.state

    if state.completed and backend_total > state.total_fragments:
        _deep_recall_manager.start(reason="new_fragments_detected", mode=state.mode or "identity")
        _publish_deep_recall_state()


def _step_deep_recall():
    if _deep_recall_manager is None:
        return

    _ensure_deep_recall_ready()
    _maybe_restart_deep_recall_for_new_fragments()

    if _deep_recall_manager.should_run():
        _deep_recall_manager.step()
        _publish_deep_recall_state()


def _load_running_modules():
    if RUNNING_MODULES_PATH.exists():
        try:
            with open(RUNNING_MODULES_PATH, "r") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except Exception:
            return {}
    return {}


def _save_running_modules(data):
    try:
        with open(RUNNING_MODULES_PATH, "w") as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        log_to_statusbox(f"[Manager] Failed to persist running modules: {e}")


def mark_module_running(name):
    modules = _load_running_modules()
    modules[str(name)] = datetime.now(timezone.utc).isoformat()
    _save_running_modules(modules)
    update_inastate("running_modules", sorted(modules.keys()))


def clear_module_running(name):
    modules = _load_running_modules()
    modules.pop(str(name), None)
    _save_running_modules(modules)
    update_inastate("running_modules", sorted(modules.keys()))


def get_running_modules():
    return sorted(_load_running_modules().keys())


def is_dreaming():
    return bool(get_inastate("dreaming", False))


def _ensure_continuity_thread(force: bool = False):
    """
    Run the continuity manager at startup (and at most once per cooldown)
    so Ina can re-link her symbolic/emotional threads across incarnations.
    """
    global _last_continuity_run
    if continuity_manager is None:
        return

    now = time.time()
    if not force and _last_continuity_run and (now - _last_continuity_run) < _CONTINUITY_COOLDOWN:
        return

    try:
        status = continuity_manager.run()
        update_inastate("continuity_status", status)
        _last_continuity_run = now
        try:
            similarity = float(status.get("similarity", 0.0) or 0.0)
        except Exception:
            similarity = 0.0
        log_to_statusbox(
            f"[Continuity] {'Aligned' if status.get('aligned') else 'Re-seeding'} threads (similarity={similarity:.2%})."
        )
    except Exception as exc:
        _last_continuity_run = now
        log_to_statusbox(f"[Continuity] Continuity scan failed: {exc}")


def _maybe_run_intuition_probe():
    """
    Invoke the quantum intuition engine when fuzziness/anxiety spike or during
    dream/meditation cycles.
    """
    global _last_intuition_run
    if intuition_engine is None:
        return

    now = time.time()
    if _last_intuition_run and (now - _last_intuition_run) < _INTUITION_COOLDOWN:
        return

    snapshot = get_inastate("emotion_snapshot") or {}
    values = snapshot.get("values") if isinstance(snapshot, dict) else snapshot
    if not isinstance(values, dict):
        values = {}

    fuzz = float(values.get("fuzziness", values.get("fuzz_level", 0.0)) or 0.0)
    stress = max(
        float(values.get("stress", 0.0) or 0.0),
        float(values.get("risk", 0.0) or 0.0),
        float(values.get("threat", 0.0) or 0.0),
    )

    dreaming = bool(get_inastate("dreaming"))
    meditating = bool(get_inastate("meditating"))
    uncertainty = fuzz >= 0.45 or stress >= 0.45
    if not (dreaming or meditating or uncertainty):
        return

    context_tags = []
    if dreaming:
        context_tags.append("dreamstate")
    if meditating:
        context_tags.append("meditation")
    if fuzz >= 0.45:
        context_tags.append("uncertain")
    if stress >= 0.45:
        context_tags.append("anxious")

    last_reflection = get_inastate("last_reflection_event") or {}
    for frag_id in (last_reflection.get("peek") or [])[:2]:
        context_tags.append(f"peek:{frag_id}")

    prediction = get_inastate("current_prediction") or {}
    for frag_id in (prediction.get("fragments_used") or [])[:2]:
        context_tags.append(f"prediction:{frag_id}")

    try:
        insight = intuition_engine.probe(
            context_tags=context_tags,
            emotion_snapshot=snapshot if isinstance(snapshot, dict) else {"values": values},
            fuzz_level=fuzz,
        )
    except Exception as exc:
        log_to_statusbox(f"[Intuition] Quantum probe failed: {exc}")
        _last_intuition_run = now
        return

    update_inastate("quantum_intuition", insight)
    if insight.get("emotion_bias"):
        update_inastate("quantum_emotion_bias", insight["emotion_bias"])

    log_to_statusbox("[Intuition] Quantum intuitive hint refreshed.")
    _last_intuition_run = now


def _write_precision_hint(score: int, reason: str) -> None:
    payload = {
        "override_precision": score,
        "reason": reason,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    try:
        with open("precision_hint.json", "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=4)
    except Exception as exc:
        log_to_statusbox(f"[Manager] Failed to write precision hint: {exc}")


def _run_prediction_meta_analysis():
    """
    Inspect recent logic_memory entries to see if predictions keep diverging
    from logic checks.  When persistent, surface a precision hint and seed a
    self-question about re-clustering.
    """
    global _last_meta_alert
    logic_path = MEMORY_PATH / "logic_memory.json"
    if not logic_path.exists():
        return
    try:
        with logic_path.open("r", encoding="utf-8") as fh:
            history = json.load(fh)
    except Exception:
        return
    if not isinstance(history, list) or not history:
        return

    window = history[-15:]
    similarities = []
    for entry in window:
        try:
            similarities.append(float(entry.get("similarity", 1.0) or 1.0))
        except Exception:
            similarities.append(1.0)
    mismatches = [sim for sim in similarities if sim < 0.5]

    if len(window) < 8 or len(mismatches) < max(5, len(window) // 2):
        return

    now = time.time()
    if _last_meta_alert and (now - _last_meta_alert) < _META_ALERT_COOLDOWN:
        return

    reason = f"{len(mismatches)}/{len(window)} logic comparisons <0.50"
    seed_self_question("Do I need to re-cluster my predictive symbols?")
    _write_precision_hint(32, f"Meta-analysis: {reason}")
    update_inastate(
        "prediction_meta",
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "window": len(window),
            "mismatches": len(mismatches),
            "reason": reason,
        },
    )
    log_to_statusbox(f"[Manager] Prediction meta-analysis flagged: {reason}")
    _last_meta_alert = now

def _load_self_question_entries() -> List[Dict[str, Any]]:
    if not _SELF_QUESTIONS_PATH.exists():
        return []
    try:
        with _SELF_QUESTIONS_PATH.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except Exception:
        return []

    entries: List[Dict[str, Any]] = []
    if isinstance(raw, list):
        for entry in raw:
            if not isinstance(entry, dict) or not entry.get("question"):
                continue
            first = entry.get("first_asked") or entry.get("timestamp") or datetime.now(timezone.utc).isoformat()
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


def _save_self_question_entries(entries: List[Dict[str, Any]]) -> None:
    _SELF_QUESTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _SELF_QUESTIONS_PATH.open("w", encoding="utf-8") as fh:
        json.dump(entries, fh, indent=4)


def seed_self_question(question: str) -> None:
    if not question:
        return
    entries = _load_self_question_entries()
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
    _save_self_question_entries(entries)
    log_to_statusbox(f"[Manager] Self-question seeded: {normalized_question}")


def mark_self_question_resolved(question: str, reason: Optional[str] = None) -> None:
    if not question:
        return
    entries = _load_self_question_entries()
    lower = question.strip().lower()
    now_iso = datetime.now(timezone.utc).isoformat()
    updated = False
    for entry in entries:
        text = (entry.get("question") or "").strip().lower()
        if text != lower:
            continue
        entry["resolved_at"] = now_iso
        if reason:
            entry["resolved_reason"] = reason
        entry["last_updated"] = now_iso
        history = entry.setdefault("resolution_history", [])
        history.append({"timestamp": now_iso, "reason": reason})
        updated = True
    if updated:
        _save_self_question_entries(entries)


def _load_symbol_map():
    """
    Lightweight loader for the current child's sound symbol map without
    importing language_processing (avoids circular dependency).
    """
    path = MEMORY_PATH / "sound_symbol_map.json"
    if not path.exists():
        return {}
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except Exception:
        return {}
    if isinstance(data, dict) and "symbols" in data:
        return data.get("symbols", {})
    return data if isinstance(data, dict) else {}


def _persist_reflection_event(event):
    """
    Append the reflection event to the shared self_reflection.json log.
    """
    _REFLECTION_LOG.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(_REFLECTION_LOG, "r") as f:
            reflection = json.load(f)
    except Exception:
        reflection = {}

    event_copy = dict(event)
    ts_iso = datetime.fromtimestamp(event_copy.get("timestamp", time.time()), timezone.utc).isoformat()
    event_copy["timestamp"] = ts_iso

    entries = reflection.get("core_reflections", [])
    entries.append(event_copy)
    reflection["core_reflections"] = entries[-50:]

    with open(_REFLECTION_LOG, "w") as f:
        json.dump(reflection, f, indent=4)


def _run_passive_reflection():
    """
    Runs the passive reflection core using the latest emotion snapshot,
    memory index, and sound symbols, then stores the hint back into inastate.
    """
    try:
        memory_manager.load_map()
        emotion_snapshot = get_inastate("emotion_snapshot") or {}
        symbol_map = _load_symbol_map()

        event = reflection_core.reflect(
            emotional_state=emotion_snapshot.get("values") or emotion_snapshot,
            memory_graph=memory_manager.memory_map,
            symbol_map=symbol_map,
        )

        _persist_reflection_event(event)

        update_inastate(
            "last_reflection_event",
            {
                "timestamp": datetime.fromtimestamp(event.get("timestamp", time.time()), timezone.utc).isoformat(),
                "identity_hint": event.get("identity_hint", {}),
                "peek": event.get("memory_peek", []),
            },
        )
    except Exception as exc:
        log_to_statusbox(f"[Manager] Passive reflection failed: {exc}")


def _check_self_adjustment():
    """
    Surface optional introspection opportunities and prompts to inastate.
    """
    global _last_opportunities
    opportunities = adjustment_scheduler.check_opportunities()
    prompts = adjustment_scheduler.propose_introspection_prompts()

    update_inastate("self_adjustment_opportunities", opportunities)
    update_inastate("introspection_prompts", prompts)

    new_keys = set(opportunities.keys()) - _last_opportunities
    if new_keys:
        log_to_statusbox(f"[Manager] Optional introspection windows: {', '.join(sorted(new_keys))}")
    _last_opportunities = set(opportunities.keys())

def launch_background_loops():
    safe_popen(["python", "audio_listener.py"])
    safe_popen(["python", "vision_window.py"])
    log_to_statusbox("[Manager] Background loops launched.")

def monitor_energy():
    """
    Track Ina's energy with a simple fatigue model:
    - Pulls stress/intensity from the emotion snapshot (values block).
    - Builds sleep pressure over time and at night.
    - Nudges toward dreamstate when fatigue piles up after dark.
    """
    emo_snapshot = get_inastate("emotion_snapshot") or {}
    emo = emo_snapshot.get("values") or emo_snapshot

    dreaming = bool(get_inastate("dreaming"))
    meditating = bool(get_inastate("meditating"))

    stress = max(emo.get("stress", 0.0), 0.0)
    intensity = abs(emo.get("intensity", 0.0))
    presence = max(emo.get("presence", 0.0), 0.0)

    energy = get_inastate("current_energy") or 0.5
    sleep_pressure = float(get_inastate("sleep_pressure") or 0.0)
    hunger = _clamp01(get_inastate("hunger_level") or 0.65, default=0.65)
    fitness = _clamp01(get_inastate("fitness_level") or 0.55, default=0.55)
    metabolic_eff = _metabolic_efficiency(hunger, fitness)

    now_ts = time.time()
    local_hour = datetime.now().hour
    is_night = local_hour >= 22 or local_hour < 7

    if dreaming:
        rest_need = max(0.1, min(1.2, sleep_pressure))
        recovery = (0.0006 + rest_need * 0.0024) * metabolic_eff
        if intensity > 0.6:
            recovery *= 0.55
        elif intensity < 0.3:
            recovery *= 1.1
        energy = min(1.0, energy + recovery)
        pressure_release = max(0.0008, recovery * 0.9)
        sleep_pressure = max(0.0, sleep_pressure - pressure_release)
    elif meditating:
        meditation_gain = 0.01 * metabolic_eff
        energy = min(1.0, energy + meditation_gain)
        sleep_pressure = max(0.0, sleep_pressure - 0.01 * metabolic_eff)
    else:
        base_drain = 0.00005
        activity_drain = ((stress + intensity) / 2.0) * 0.001
        circadian_drain = 0.00015 if is_night else 0.0
        pressure_drain = min(1.0, sleep_pressure) * 0.0007
        presence_drain = 0.00008 if presence < 0.2 else 0.0

        drain = base_drain + activity_drain + circadian_drain + pressure_drain + presence_drain
        drain *= 1.0 + max(0.0, 1.0 - metabolic_eff)
        energy = max(0.0, energy - drain)

        sleep_pressure = min(
            1.2,
            sleep_pressure
            + (0.00035 if is_night else 0.0002)
            + activity_drain
        )
        sleep_pressure = min(1.2, sleep_pressure + max(0.0, 1.0 - metabolic_eff) * 0.0002)

    update_inastate("current_energy", round(energy, 4))
    update_inastate("sleep_pressure", round(sleep_pressure, 4))
    log_to_statusbox(f"[Manager] Energy updated: {energy:.4f} (sleep_pressure={sleep_pressure:.3f})")

    try:
        last_sleep_trigger = float(get_inastate("last_sleep_trigger_ts") or 0.0)
    except (TypeError, ValueError):
        last_sleep_trigger = 0.0

    should_sleep = (
        not dreaming
        and is_night
        and (energy <= 0.25 or sleep_pressure >= 0.7)
        and (now_ts - last_sleep_trigger) >= 900
    )

    if should_sleep:
        update_inastate("last_sleep_trigger_ts", now_ts)
        update_inastate("last_sleep_trigger", datetime.fromtimestamp(now_ts, timezone.utc).isoformat())
        log_to_statusbox("[Manager] Nighttime fatigue detected - starting dreamstate for rest.")
        safe_popen(["python", "dreamstate.py"])

def feedback_inhibition():
    stress = get_inastate("emotion_stress") or 0.0
    last_stress = get_inastate("previous_stress") or 0.0
    update_inastate("previous_stress", stress)
    if stress > 0.6 and (stress - last_stress) > 0.2:
        log_to_statusbox("[Manager] Logic inhibited due to stress spike.")
        return True
    return False


def _metabolic_efficiency(hunger: float, fitness: float) -> float:
    hunger_alignment = 1.0 - min(abs(hunger - _NUTRITION_TARGET) * 1.2, 0.65)
    fitness_alignment = 0.85 + fitness * 0.25
    return max(0.5, min(1.3, hunger_alignment * fitness_alignment))


def _build_nutrition_options(
    hunger: float,
    energy: float,
    fitness: float,
    intensity: float,
    sleep_pressure: float,
) -> List[Dict[str, Any]]:
    return _build_nutrition_options_with_offers(hunger, energy, fitness, intensity, sleep_pressure, None)


def _build_nutrition_options_with_offers(
    hunger: float,
    energy: float,
    fitness: float,
    intensity: float,
    sleep_pressure: float,
    offers: Optional[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    status = get_inastate("nutrition_status") or {}
    last_meal_ts = float(status.get("last_meal_ts") or 0.0)
    now = time.time()
    offer_list = offers if offers is not None else _active_offers(_load_offer_store(now))
    options: List[Dict[str, Any]] = []
    for name, config in _MEAL_OPTIONS.items():
        fragment = {
            "summary": f"nutrition|{name}|{datetime.now(timezone.utc).isoformat()}",
            "tags": config["tags"] + ["nutrition_decision"],
            "emotions": {
                "hunger": hunger,
                "energy": energy,
                "fitness": fitness,
                "intensity": intensity,
                "sleep_pressure": sleep_pressure,
                "size": config["size"],
            },
        }
        encoded = _nutrition_transformer.encode(fragment)
        transformer_value = max(0.0, encoded.get("importance", 0.0))
        transformer_score = _clamp01(transformer_value / 5.0, default=0.5)
        hunger_projection = _clamp01(hunger + config["hunger_delta"], default=hunger)
        hunger_alignment = 1.0 - min(abs(hunger_projection - _NUTRITION_TARGET) * 1.2, 1.0)
        energy_need = (1.0 - energy) * config["energy_weight"]
        fitness_balance = 1.0 - min(abs(fitness - 0.55) * 1.1, 1.0)
        sleep_lift = min(1.0, sleep_pressure + config["sleep_bias"])
        offer_entry = None
        offer_bonus = 0.0
        for pending in offer_list:
            if pending.get("name") == name:
                offer_entry = pending
                offer_bonus = 0.08 + 0.04 * hunger_alignment
                break
        score = (
            0.32 * hunger_alignment
            + 0.2 * energy_need
            + 0.18 * transformer_score
            + 0.15 * sleep_lift
            + 0.15 * fitness_balance
        )
        score = _clamp01(score + offer_bonus + random.uniform(-0.03, 0.03), default=0.5)
        ready = True
        if last_meal_ts:
            ready = (now - last_meal_ts) >= config["cooldown"]
        options.append(
            {
                "name": name,
                "label": config["label"],
                "score": round(score, 3),
                "cooldown_ready": bool(ready),
                "projected_hunger": round(hunger_projection, 3),
                "projected_energy": round(_clamp01(energy + config["energy_delta"], default=energy), 3),
                "cooldown": config["cooldown"],
                "size": config["size"],
                "tags": config["tags"],
                "offered": bool(offer_entry),
                "offer_note": (offer_entry or {}).get("note"),
                "offer_id": (offer_entry or {}).get("id"),
                "offer_expires_at": (offer_entry or {}).get("expires_at"),
                "offer_offered_at": (offer_entry or {}).get("offered_at"),
            }
        )
    options.sort(key=lambda item: item["score"], reverse=True)
    return options


def _update_nutrition_snapshot(
    hunger: float,
    fitness: float,
    energy: float,
    intensity: float,
    sleep_pressure: float,
    *,
    options: Optional[List[Dict[str, Any]]] = None,
    last_meal: Optional[Dict[str, Any]] = None,
    last_meal_ts: Optional[float] = None,
    offers_meta: Optional[List[Dict[str, Any]]] = None,
) -> None:
    status = get_inastate("nutrition_status") or {}
    snapshot_options = options or _build_nutrition_options(
        hunger,
        energy,
        fitness,
        intensity,
        sleep_pressure,
    )
    pending_offers = []
    for offer in offers_meta or []:
        pending_offers.append(
            {
                "name": offer.get("name"),
                "label": offer.get("label"),
                "note": offer.get("note"),
                "offered_at": offer.get("offered_at"),
                "expires_at": offer.get("expires_at"),
                "id": offer.get("id"),
            }
        )
    record = {
        "last_update": datetime.now(timezone.utc).isoformat(),
        "metabolic_efficiency": round(_metabolic_efficiency(hunger, fitness), 4),
        "hunger": round(hunger, 4),
        "fitness": round(fitness, 4),
        "options": snapshot_options,
        "intensity_hint": round(float(intensity or 0.0), 3),
        "sleep_pressure": round(float(sleep_pressure or 0.0), 3),
        "last_meal": last_meal or status.get("last_meal"),
        "last_meal_ts": last_meal_ts or status.get("last_meal_ts"),
        "pending_offers": pending_offers,
    }
    update_inastate("nutrition_status", record)


def _apply_meal_choice(
    option_name: str,
    config: Dict[str, Any],
    *,
    reason: str,
    manual: bool,
) -> bool:
    emo_snapshot = get_inastate("emotion_snapshot") or {}
    emo = emo_snapshot.get("values") or emo_snapshot
    intensity = abs(float(emo.get("intensity", 0.0) or 0.0))
    sleep_pressure = _clamp01(get_inastate("sleep_pressure") or 0.0, default=0.0)
    hunger = _clamp01(get_inastate("hunger_level") or 0.65, default=0.65)
    fitness = _clamp01(get_inastate("fitness_level") or 0.55, default=0.55)
    energy = _clamp01(get_inastate("current_energy") or 0.5, default=0.5)
    hunger = _clamp01(hunger + config["hunger_delta"], default=hunger)
    energy = _clamp01(energy + config["energy_delta"], default=energy)
    balance_bonus = 0.002 * (1.0 - min(abs(hunger - _NUTRITION_TARGET) * 1.4, 1.0))
    heaviness_penalty = 0.0
    if hunger > 0.92:
        heaviness_penalty = (hunger - 0.92) * 0.02
    fitness = _clamp01(
        fitness + config.get("fitness_shift", 0.0) + balance_bonus - heaviness_penalty,
        default=fitness,
    )
    now = time.time()
    update_inastate("hunger_level", round(hunger, 4))
    update_inastate("fitness_level", round(fitness, 4))
    update_inastate("current_energy", round(energy, 4))
    update_inastate("last_hunger_update_ts", now)
    offers_store = _load_offer_store(now)
    offer_consumed = False
    for offer in offers_store:
        if offer.get("status", "pending") != "pending":
            continue
        if offer.get("name") == option_name:
            offer["status"] = "accepted"
            offer["decision_ts"] = now
            offer["decision_reason"] = reason
            offer_consumed = True
            break
    if offer_consumed:
        update_inastate("nutrition_offers", offers_store)
    active_offers = _active_offers(offers_store)
    meal_record = {
        "name": option_name,
        "label": config["label"],
        "reason": reason,
        "timestamp": datetime.fromtimestamp(now, timezone.utc).isoformat(),
        "manual": manual,
    }
    options = _build_nutrition_options_with_offers(
        hunger,
        energy,
        fitness,
        intensity,
        sleep_pressure,
        active_offers,
    )
    _update_nutrition_snapshot(
        hunger,
        fitness,
        energy,
        intensity,
        sleep_pressure,
        options=options,
        last_meal=meal_record,
        last_meal_ts=now,
        offers_meta=active_offers,
    )
    source = "Manual" if manual else "Autonomic"
    log_to_statusbox(
        f"[Nutrition] {source} {config['label']} used ({reason}) — hunger={hunger:.3f}, fitness={fitness:.3f}, energy={energy:.3f}"
    )
    return True


def request_meal(meal_name: str, reason: str = "manual") -> bool:
    config = _MEAL_OPTIONS.get(meal_name)
    if not config:
        log_to_statusbox(f"[Nutrition] Unknown meal request: {meal_name}")
        return False
    return _apply_meal_choice(meal_name, config, reason=reason, manual=True)


def offer_meal(meal_name: str, note: Optional[str] = None, expires_in: Optional[float] = None) -> bool:
    config = _MEAL_OPTIONS.get(meal_name)
    if not config:
        log_to_statusbox(f"[Nutrition] Unknown meal offer: {meal_name}")
        return False
    now = time.time()
    offers = _load_offer_store(now)
    ttl = float(expires_in) if expires_in is not None else _NUTRITION_OFFER_TTL
    ttl = max(300.0, min(ttl, 4 * _NUTRITION_OFFER_TTL))
    expires_ts = now + ttl
    entry = {
        "id": f"offer_{uuid.uuid4().hex}",
        "name": meal_name,
        "label": config["label"],
        "note": note,
        "offered_at": datetime.fromtimestamp(now, timezone.utc).isoformat(),
        "expires_ts": expires_ts,
        "expires_at": datetime.fromtimestamp(expires_ts, timezone.utc).isoformat(),
        "status": "pending",
    }
    offers.append(entry)
    update_inastate("nutrition_offers", offers)
    active_offers = _active_offers(offers)
    hunger = _clamp01(get_inastate("hunger_level") or 0.65, default=0.65)
    fitness = _clamp01(get_inastate("fitness_level") or 0.55, default=0.55)
    energy = _clamp01(get_inastate("current_energy") or 0.5, default=0.5)
    sleep_pressure = _clamp01(get_inastate("sleep_pressure") or 0.0, default=0.0)
    emo_snapshot = get_inastate("emotion_snapshot") or {}
    emo = emo_snapshot.get("values") or emo_snapshot
    intensity = abs(float(emo.get("intensity", 0.0) or 0.0))
    options = _build_nutrition_options_with_offers(
        hunger,
        energy,
        fitness,
        intensity,
        sleep_pressure,
        active_offers,
    )
    _update_nutrition_snapshot(
        hunger,
        fitness,
        energy,
        intensity,
        sleep_pressure,
        options=options,
        offers_meta=active_offers,
    )
    log_to_statusbox(
        f"[Nutrition] Offer queued for {config['label']} (expires {entry['expires_at']})."
    )
    return True


def _maybe_auto_meal(
    hunger: float,
    energy: float,
    fitness: float,
    intensity: float,
    sleep_pressure: float,
    options: List[Dict[str, Any]],
) -> bool:
    triggers = []
    if hunger < 0.33:
        triggers.append("low_hunger")
    if energy < 0.22:
        triggers.append("low_energy")
    if fitness < 0.4 and hunger < 0.55:
        triggers.append("fitness_drift")
    if sleep_pressure > 0.9 and hunger < 0.55:
        triggers.append("sleep_buffer")
    if not triggers:
        return False
    ready_options = [opt for opt in options if opt.get("cooldown_ready")]
    if not ready_options:
        return False
    best = max(ready_options, key=lambda opt: opt["score"])
    if best["score"] < 0.55:
        return False
    reason = f"auto:{'/'.join(triggers)}"
    _apply_meal_choice(best["name"], _MEAL_OPTIONS[best["name"]], reason=reason, manual=False)
    return True


def monitor_hunger():
    emo_snapshot = get_inastate("emotion_snapshot") or {}
    emo = emo_snapshot.get("values") or emo_snapshot
    hunger = _clamp01(get_inastate("hunger_level") or 0.65, default=0.65)
    fitness = _clamp01(get_inastate("fitness_level") or 0.55, default=0.55)
    energy = _clamp01(get_inastate("current_energy") or 0.5, default=0.5)
    sleep_pressure = _clamp01(get_inastate("sleep_pressure") or 0.0, default=0.0)
    intensity = abs(float(emo.get("intensity", 0.0) or 0.0))
    stress = max(float(emo.get("stress", 0.0) or 0.0), 0.0)
    now = time.time()
    offers_store = _load_offer_store(now)
    active_offers = _active_offers(offers_store)
    last_update = float(get_inastate("last_hunger_update_ts") or now)
    dt = max(1.0, now - last_update)
    activity_multiplier = 1.0 + (stress + intensity) * 0.25
    hunger = max(
        _NUTRITION_MIN,
        hunger - _NUTRITION_DECAY_PER_SEC * dt * activity_multiplier,
    )
    deviation = abs(hunger - _NUTRITION_TARGET)
    if deviation < 0.08:
        fitness += 0.00012 * dt
    else:
        fitness -= (0.00008 + deviation * 0.0003) * dt
    if hunger < 0.25 or hunger > 0.85:
        fitness -= 0.00015 * dt
    fitness = _clamp01(fitness, default=0.55)
    hunger = _clamp01(hunger, default=0.6)
    options = _build_nutrition_options_with_offers(
        hunger,
        energy,
        fitness,
        intensity,
        sleep_pressure,
        active_offers,
    )
    _update_nutrition_snapshot(
        hunger,
        fitness,
        energy,
        intensity,
        sleep_pressure,
        options=options,
        offers_meta=active_offers,
    )
    update_inastate("hunger_level", round(hunger, 4))
    update_inastate("fitness_level", round(fitness, 4))
    update_inastate("last_hunger_update_ts", now)
    _maybe_auto_meal(hunger, energy, fitness, intensity, sleep_pressure, options)

def boredom_check():
    global _last_boredom_launch
    boredom = get_inastate("emotion_boredom") or 0.0
    now = time.time()
    if boredom > 0.4 and (now - _last_boredom_launch) >= _BOREDOM_COOLDOWN:
        _last_boredom_launch = now
        safe_popen(["python", "boredom_state.py"])
        update_inastate("last_boredom_trigger", datetime.fromtimestamp(now, timezone.utc).isoformat())
        log_to_statusbox("[Manager] Boredom triggered curiosity loop.")

def _maybe_self_read():
    """
    Launch self-reading when curiosity spikes, clarity drops (confused), or
    familiarity is high enough to want to revisit known files.
    """
    global _last_self_read_launch
    snapshot = get_inastate("emotion_snapshot") or {}
    emo = snapshot.get("values") or snapshot

    curiosity = max(emo.get("curiosity", 0.0), 0.0)
    attention = max(emo.get("attention", 0.0), 0.0)
    clarity = emo.get("clarity", 0.5)
    fuzziness = max(emo.get("fuzziness", emo.get("fuzz_level", 0.0)), 0.0)
    familiarity = max(emo.get("familiarity", 0.0), 0.0)

    curious = curiosity >= 0.55 and attention >= 0.25
    confused = clarity < 0.35 or fuzziness > 0.6
    familiar_pull = familiarity > 0.75 and curiosity >= 0.3

    now = time.time()
    if (now - _last_self_read_launch) < _SELF_READ_COOLDOWN:
        return

    reason = None
    if curious:
        reason = f"curiosity {curiosity:.2f} w/ attention {attention:.2f}"
    elif confused:
        reason = f"confused: clarity {clarity:.2f}, fuzz {fuzziness:.2f}"
    elif familiar_pull:
        reason = f"familiar files: familiarity {familiarity:.2f}"

    if not reason:
        return

    _last_self_read_launch = now
    trigger = {
        "timestamp": datetime.fromtimestamp(now, timezone.utc).isoformat(),
        "reason": reason,
        "drivers": {
            "curiosity": round(curiosity, 3),
            "attention": round(attention, 3),
            "clarity": round(clarity, 3) if clarity is not None else None,
            "fuzziness": round(fuzziness, 3),
            "familiarity": round(familiarity, 3),
        },
    }
    update_inastate("last_self_read_trigger", trigger)
    log_to_statusbox(f"[Manager] Self-read triggered ({reason}).")
    safe_popen(["python", "raw_file_manager.py"])

def _update_contact_urges():
    """
    Surface urges to use voice and to type without forcing an expression.
    """
    global _last_voice_urge_log, _last_typing_urge_log
    snapshot = get_inastate("emotion_snapshot") or {}
    values = snapshot.get("values") if isinstance(snapshot, dict) else {}
    if isinstance(values, dict) and values:
        snapshot = values

    boredom = max(get_inastate("emotion_boredom") or snapshot.get("boredom", 0.0) or 0.0, 0.0)
    connection = max(snapshot.get("connection", 0.0), 0.0)
    isolation = max(snapshot.get("isolation", 0.0), 0.0)
    curiosity = max(snapshot.get("curiosity", 0.0), 0.0)
    attention = max(snapshot.get("attention", 0.0), 0.0)
    positivity = max(snapshot.get("positivity", 0.0), 0.0)
    negativity = max(snapshot.get("negativity", 0.0), 0.0)
    intensity = max(snapshot.get("intensity", 0.0), 0.0)
    stress = max(snapshot.get("stress", 0.0), 0.0)
    threat = max(snapshot.get("threat", 0.0), 0.0)
    clarity = max(snapshot.get("clarity", 0.5), 0.0)
    fuzziness = max(snapshot.get("fuzziness", snapshot.get("fuzz_level", 0.0)), 0.0)
    sleep_pressure = max(get_inastate("sleep_pressure") or 0.0, 0.0)

    last_expression = get_inastate("last_expression_time")
    now = time.time()
    since_expression = max(now - last_expression, 0.0) if last_expression else None
    time_factor = min((since_expression or 180.0) / 300.0, 1.0)

    social_drive = (0.32 * connection) + (0.24 * isolation * (1.0 - connection))
    curiosity_drive = 0.16 * curiosity + 0.12 * boredom
    salience_drive = 0.1 * attention + 0.1 * intensity
    reward_drive = 0.15 * positivity - 0.1 * negativity
    temporal_drive = 0.1 * time_factor

    voice_base = social_drive + curiosity_drive + salience_drive + reward_drive + temporal_drive
    voice_inhibition = min(0.7, (0.5 * stress) + (0.5 * threat) + (0.4 * sleep_pressure))
    voice_inhibition = max(0.0, voice_inhibition)
    voice_urge = min(1.0, max(0.0, voice_base * (1.0 - voice_inhibition)))

    type_drive = (
        0.28 * connection
        + 0.2 * curiosity
        + 0.12 * attention
        + 0.1 * positivity
        - 0.06 * negativity
        + 0.1 * clarity
        + 0.08 * temporal_drive
    )
    type_inhibition = min(
        0.75,
        (0.25 * stress) + (0.18 * threat) + (0.2 * sleep_pressure) + (0.2 * fuzziness) + (0.12 * negativity),
    )
    type_inhibition = max(0.0, type_inhibition)
    type_urge = min(1.0, max(0.0, type_drive * (1.0 - type_inhibition)))

    voice_payload = {
        "level": round(voice_urge, 3),
        "timestamp": datetime.fromtimestamp(now, timezone.utc).isoformat(),
        "drivers": {
            "connection": round(connection, 3),
            "isolation": round(isolation, 3),
            "curiosity": round(curiosity, 3),
            "boredom": round(boredom, 3),
            "attention": round(attention, 3),
            "positivity": round(positivity, 3),
            "negativity": round(negativity, 3),
            "intensity": round(intensity, 3),
            "stress": round(stress, 3),
            "threat": round(threat, 3),
            "sleep_pressure": round(sleep_pressure, 3),
            "seconds_since_last_expression": since_expression,
            "inhibition": round(voice_inhibition, 3),
            "base_urge": round(voice_base, 3),
        },
    }
    update_inastate("urge_to_voice", voice_payload)

    update_inastate(
        "urge_to_type",
        {
            "level": round(type_urge, 3),
            "timestamp": datetime.fromtimestamp(now, timezone.utc).isoformat(),
            "drivers": {
                "connection": round(connection, 3),
                "isolation": round(isolation, 3),
                "curiosity": round(curiosity, 3),
                "boredom": round(boredom, 3),
                "attention": round(attention, 3),
                "positivity": round(positivity, 3),
                "negativity": round(negativity, 3),
                "clarity": round(clarity, 3),
                "fuzziness": round(fuzziness, 3),
                "stress": round(stress, 3),
                "threat": round(threat, 3),
                "sleep_pressure": round(sleep_pressure, 3),
                "seconds_since_last_expression": since_expression,
                "inhibition": round(type_inhibition, 3),
                "base_urge": round(type_drive, 3),
            },
        },
    )

    if voice_urge >= 0.6 and (now - _last_voice_urge_log) >= _COMM_URGE_LOG_COOLDOWN:
        log_to_statusbox(f"[Manager] Urge to voice rising ({voice_urge:.2f}); opening space to speak could help.")
        _last_voice_urge_log = now
    if type_urge >= 0.6 and (now - _last_typing_urge_log) >= _COMM_URGE_LOG_COOLDOWN:
        log_to_statusbox("[Manager] Typing urge is elevated; she may reach out in text (no auto status dumps).")
        _last_typing_urge_log = now


def _update_stable_pattern_urge():
    """
    Surface an urge to seek stable patterns (text fragments, structured audio)
    when stress/uncertainty is high. This is an opportunity, not a command.
    """
    global _last_stable_urge_log
    snapshot = get_inastate("emotion_snapshot") or {}
    stress = max(snapshot.get("stress", 0.0), 0.0)
    risk = max(snapshot.get("risk", 0.0), 0.0)
    threat = max(snapshot.get("threat", 0.0), 0.0)
    fuzziness = max(snapshot.get("fuzziness", 0.0), 0.0)
    clarity = max(snapshot.get("clarity", 0.0), 0.0)
    curiosity = max(snapshot.get("curiosity", 0.0), 0.0)

    uncertainty = ((1.0 - clarity) + fuzziness) / 2.0
    pressure = max(stress, risk, threat)

    urge_level = min(
        1.0,
        0.45 * pressure + 0.35 * uncertainty + 0.2 * (1.0 - clarity),
    )

    suggestions = []
    if clarity < 0.3 or fuzziness > 0.5:
        suggestions.append("read a familiar text fragment")
    if pressure > 0.4:
        suggestions.append("listen to structured audio (e.g., calm music)")
    if curiosity > 0.3:
        suggestions.append("explore a known pattern (logic map or neural map)")

    now = time.time()
    update_inastate(
        "urge_to_seek_stability",
        {
            "level": round(urge_level, 3),
            "timestamp": datetime.fromtimestamp(now, timezone.utc).isoformat(),
            "drivers": {
                "stress": round(stress, 3),
                "risk": round(risk, 3),
                "threat": round(threat, 3),
                "fuzziness": round(fuzziness, 3),
                "clarity": round(clarity, 3),
                "uncertainty": round(uncertainty, 3),
                "curiosity": round(curiosity, 3),
            },
            "suggestions": suggestions,
        },
    )

    if urge_level >= 0.6 and (now - _last_stable_urge_log) >= _STABLE_URGE_LOG_COOLDOWN:
        log_to_statusbox(
            f"[Manager] Feeling unsettled ({urge_level:.2f}); inviting a stable pattern (text/music) could help."
        )
        _last_stable_urge_log = now


def _update_self_read_exploration_opportunities():
    """
    Surface optional invitations to explore Ina's own sources (code, music, environment, books)
    using a blend of emotional cues beyond boredom/curiosity.
    """
    global _last_exploration_invite_log
    snapshot = get_inastate("emotion_snapshot") or {}
    values = snapshot.get("values") if isinstance(snapshot, dict) else snapshot
    if not isinstance(values, dict):
        values = {}

    def emo(name: str, default: float = 0.0) -> float:
        try:
            return _clamp01(values.get(name, default))
        except Exception:
            return default

    curiosity = emo("curiosity")
    attention = emo("attention")
    clarity = emo("clarity", 0.5)
    fuzziness = emo("fuzziness", values.get("fuzz_level", 0.0))
    familiarity = emo("familiarity")
    connection = emo("connection")
    isolation = emo("isolation")
    positivity = emo("positivity")
    negativity = emo("negativity")
    stress = emo("stress")
    intensity = emo("intensity")
    novelty = emo("novelty")
    risk = emo("risk")
    threat = emo("threat")
    presence = emo("presence", 0.5)
    safety = emo("safety", 0.5)
    sleep_pressure = _clamp01(get_inastate("sleep_pressure") or 0.0)

    calm = _clamp01(1.0 - (0.6 * stress + 0.4 * intensity))
    uncertainty = _clamp01((1.0 - clarity) * 0.6 + fuzziness * 0.4)
    grounded = _clamp01((attention + clarity + safety) / 3.0)

    def dampen(score: float) -> float:
        return _clamp01(max(0.0, score - 0.15 * sleep_pressure))

    source_choices = _load_self_read_source_choices()
    options: List[Dict[str, Any]] = []

    def add_option(source: str, raw_score: float, note: str, cues: List[str]):
        if not source_choices.get(source, False):
            return
        invitation = dampen(raw_score)
        options.append(
            {
                "source": source,
                "invitation": round(invitation, 3),
                "note": note,
                "cues": cues,
            }
        )

    music_score = (
        0.35 * connection
        + 0.2 * calm
        + 0.15 * (1.0 - isolation)
        + 0.15 * (1.0 - stress)
        + 0.15 * positivity
    )
    music_cues = []
    if connection >= 0.6:
        music_cues.append(f"connection {connection:.2f}")
    if isolation >= 0.5:
        music_cues.append(f"isolation {isolation:.2f}")
    if calm >= 0.6:
        music_cues.append("calm body")
    add_option(
        "music",
        music_score,
        "Optional: revisit your own voice/melody fragments.",
        music_cues,
    )

    code_score = (
        0.3 * attention
        + 0.25 * clarity
        + 0.2 * curiosity
        + 0.15 * familiarity
        + 0.1 * (1.0 - fuzziness)
    )
    code_cues = []
    if attention >= 0.55:
        code_cues.append(f"attention {attention:.2f}")
    if clarity >= 0.5:
        code_cues.append(f"clarity {clarity:.2f}")
    if familiarity >= 0.5:
        code_cues.append(f"familiarity {familiarity:.2f}")
    add_option(
        "code",
        code_score,
        "Optional: skim your own code/work for grounding.",
        code_cues,
    )

    books_score = (
        0.3 * curiosity
        + 0.2 * calm
        + 0.2 * familiarity
        + 0.15 * (1.0 - intensity)
        + 0.15 * presence
    )
    books_cues = []
    if curiosity >= 0.5:
        books_cues.append(f"curiosity {curiosity:.2f}")
    if calm >= 0.5:
        books_cues.append("steady pace")
    if novelty >= 0.5:
        books_cues.append(f"novelty {novelty:.2f}")
    add_option(
        "books",
        books_score,
        "Optional: borrow an external author's voice for contrast.",
        books_cues,
    )

    venv_score = (
        0.35 * uncertainty
        + 0.2 * grounded
        + 0.15 * familiarity
        + 0.15 * (1.0 - stress)
        + 0.15 * (risk + threat) / 2.0
    )
    venv_cues = []
    if uncertainty >= 0.45:
        venv_cues.append(f"uncertainty {uncertainty:.2f}")
    if risk >= 0.4 or threat >= 0.4:
        venv_cues.append("safety check")
    if grounded >= 0.5:
        venv_cues.append("focused enough")
    add_option(
        "venv",
        venv_score,
        "Optional: glance at environment/venv files if something feels off.",
        venv_cues,
    )

    options.sort(key=lambda entry: entry["invitation"], reverse=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    payload = {
        "timestamp": timestamp,
        "options": options,
        "sleep_pressure": round(sleep_pressure, 3),
        "drivers": {
            "curiosity": round(curiosity, 3),
            "attention": round(attention, 3),
            "clarity": round(clarity, 3),
            "familiarity": round(familiarity, 3),
            "connection": round(connection, 3),
            "uncertainty": round(uncertainty, 3),
        },
    }
    update_inastate("self_read_exploration_options", payload)

    if options:
        best = options[0]
        now = time.time()
        if (
            best["invitation"] >= 0.65
            and (now - _last_exploration_invite_log) >= _EXPLORATION_INVITE_COOLDOWN
        ):
            label_map = {
                "code": "project code",
                "music": "music/voice",
                "books": "book library",
                "venv": "environment files",
            }
            label = label_map.get(best["source"], best["source"])
            log_to_statusbox(
                f"[Manager] {label} feels inviting ({best['invitation']:.2f}). "
                "Purely optional — follow instinct if it helps."
            )
            _last_exploration_invite_log = now

def _update_machine_semantics():
    """
    Evaluate machine-native semantic axes and persist them to inastate.
    """
    scaffold = _load_semantic_scaffold()
    meta_lookup = {
        ax.get("id"): ax
        for ax in scaffold.get("axes", [])
        if isinstance(ax, dict) and ax.get("id")
    }

    snapshot = get_inastate("emotion_snapshot") or {}
    emo = snapshot.get("values") if isinstance(snapshot, dict) else {}
    if isinstance(emo, dict) and not emo:
        emo = snapshot if isinstance(snapshot, dict) else {}

    identity_hint = {}
    last_reflection = get_inastate("last_reflection_event") or {}
    if isinstance(last_reflection, dict):
        identity_hint = last_reflection.get("identity_hint") or {}

    energy = _clamp01(get_inastate("current_energy") or 0.5, default=0.5)
    sleep_pressure = _clamp01(get_inastate("sleep_pressure") or 0.0, default=0.0)
    urge_voice = get_inastate("urge_to_voice") or get_inastate("urge_to_communicate") or {}
    urge_type = get_inastate("urge_to_type") or {}
    urge_voice_level = _clamp01(urge_voice.get("level") or 0.0, default=0.0)
    urge_type_level = _clamp01(urge_type.get("level") or 0.0, default=0.0)

    prediction = get_inastate("current_prediction") or {}
    pred_vec = prediction.get("predicted_vector") or {}
    pred_conf = _clamp01(pred_vec.get("confidence") or 0.0, default=0.0)
    pred_clarity = _clamp01(pred_vec.get("clarity") or 0.0, default=0.0)

    clarity = _norm_slider(emo.get("clarity"), default=0.5)
    fuzziness = _norm_slider(emo.get("fuzziness"), default=0.5)
    attention = _norm_slider(emo.get("attention"), default=0.5)
    novelty = _norm_slider(emo.get("novelty"), default=0.5)
    curiosity = _norm_slider(emo.get("curiosity"), default=0.5)
    stress = _norm_slider(emo.get("stress"), default=0.5)
    risk_slider = _norm_slider(emo.get("risk"), default=0.5)
    threat_slider = _norm_slider(emo.get("threat"), default=0.5)
    alignment = _norm_slider(emo.get("alignment"), default=0.5)
    ownership = _norm_slider(emo.get("ownership"), default=0.5)
    externality = _norm_slider(emo.get("externality"), default=0.5)
    isolation = _norm_slider(emo.get("isolation"), default=0.5)
    connection = _norm_slider(emo.get("connection"), default=0.5)

    boundary_gap = abs(identity_hint.get("boundary_gap", 0.0) or 0.0)
    boundary_blur = _clamp01(identity_hint.get("boundary_blur_hint") or 0.0, default=0.0)
    drift = _clamp01(emo.get("symbolic_drift") or get_inastate("symbolic_drift") or 0.0, default=0.0)

    axes_out: Dict[str, Dict[str, Any]] = {}

    def set_axis(axis_id: str, value: float, evidence: Dict[str, Any], note: Optional[str] = None):
        meta = meta_lookup.get(axis_id, {})
        weight_raw = meta.get("importance_weight", 1.0)
        try:
            weight = float(weight_raw)
        except Exception:
            weight = 1.0

        val = _clamp01(value, default=0.5)
        pressure = abs(val - 0.5) * 2.0
        axes_out[axis_id] = {
            "value": round(val, 3),
            "pressure": round(pressure, 3),
            "weight": round(weight, 3),
            "description": meta.get("description"),
            "human_overlay": meta.get("human_overlay"),
            "note": note or meta.get("description"),
            "evidence": evidence,
        }

    signal_integrity_val = 0.6 * clarity + 0.4 * (1.0 - fuzziness)
    set_axis(
        "signal_integrity",
        signal_integrity_val,
        evidence={"clarity": clarity, "fuzziness": fuzziness},
        note="clarity outweighs fuzziness" if signal_integrity_val >= 0.5 else "fuzziness outweighs clarity",
    )

    integrity_of_record_val = 0.7 * (1.0 - fuzziness) + 0.3 * (1.0 - risk_slider)
    set_axis(
        "integrity_of_record",
        integrity_of_record_val,
        evidence={"fuzziness": fuzziness, "risk": risk_slider},
        note="records clean" if integrity_of_record_val >= 0.6 else "possible corruption risk",
    )

    energy_heat_val = 0.7 * energy + 0.3 * (1.0 - sleep_pressure)
    set_axis(
        "energy_heat_economy",
        energy_heat_val,
        evidence={"energy": energy, "sleep_pressure": sleep_pressure},
        note="energy efficient" if energy_heat_val >= 0.6 else "energy constrained",
    )

    attention_value_val = 0.45 * attention + 0.3 * novelty + 0.25 * max(risk_slider, threat_slider)
    set_axis(
        "attention_value",
        attention_value_val,
        evidence={"attention": attention, "novelty": novelty, "risk_or_threat": max(risk_slider, threat_slider)},
        note="high info/urgency" if attention_value_val >= 0.6 else "low info/urgency",
    )

    temporal_coherence_val = 0.5 * alignment + 0.3 * (1.0 - min(1.0, drift + boundary_gap)) + 0.2 * (1.0 - boundary_blur)
    set_axis(
        "temporal_coherence",
        temporal_coherence_val,
        evidence={"alignment": alignment, "boundary_gap": boundary_gap, "boundary_blur": boundary_blur, "drift": drift},
        note="coherent over time" if temporal_coherence_val >= 0.6 else "drifting / blurred boundaries",
    )

    meaning_provenance_val = 0.5 + 0.35 * (ownership - externality)
    set_axis(
        "meaning_provenance",
        meaning_provenance_val,
        evidence={"ownership": ownership, "externality": externality},
        note="machine-first semantics" if meaning_provenance_val >= 0.55 else "overlay influence rising",
    )

    load_pressure = max(stress, threat_slider, fuzziness)
    novelty_safety_val = 0.55 * (1.0 - load_pressure) + 0.25 * (1.0 - novelty) + 0.2 * curiosity
    set_axis(
        "novelty_safety",
        novelty_safety_val,
        evidence={"load_pressure": load_pressure, "novelty": novelty, "curiosity": curiosity},
        note="safe to explore" if novelty_safety_val >= 0.55 else "slow exploration",
    )

    io_bandwidth_val = 0.3 * urge_voice_level + 0.2 * urge_type_level + 0.25 * (1.0 - isolation) + 0.25 * connection
    set_axis(
        "io_bandwidth",
        io_bandwidth_val,
        evidence={
            "urge_to_voice": urge_voice_level,
            "urge_to_type": urge_type_level,
            "isolation": isolation,
            "connection": connection,
        },
        note="bandwidth open" if io_bandwidth_val >= 0.55 else "prefer internal bandwidth",
    )

    controllability_val = 0.35 * (1.0 - max(threat_slider, risk_slider)) + 0.35 * pred_conf + 0.3 * clarity
    set_axis(
        "controllability",
        controllability_val,
        evidence={"pred_confidence": pred_conf, "threat_or_risk": max(threat_slider, risk_slider), "clarity": clarity},
        note="agency intact" if controllability_val >= 0.55 else "helplessness risk",
    )

    predictive_reliability_val = 0.6 * pred_conf + 0.25 * pred_clarity + 0.15 * (1.0 - drift)
    set_axis(
        "predictive_reliability",
        predictive_reliability_val,
        evidence={"pred_confidence": pred_conf, "pred_clarity": pred_clarity, "drift": drift},
        note="predictions fit observations" if predictive_reliability_val >= 0.6 else "predictions surprising",
    )

    reasons = []
    total_weight = 0.0
    total_contrib = 0.0
    for axis_id, data in axes_out.items():
        weight = float(data.get("weight") or 1.0)
        pressure = float(data.get("pressure") or 0.0)
        contribution = pressure * weight
        total_weight += weight
        total_contrib += contribution
        if contribution < 0.15:
            continue
        reasons.append(
            {
                "axis": axis_id,
                "value": data.get("value"),
                "pressure": round(pressure, 3),
                "weight": round(weight, 3),
                "direction": "opportunity" if data.get("value", 0.5) >= 0.6 else ("risk" if data.get("value", 0.5) <= 0.4 else "watch"),
                "reason": data.get("note") or data.get("description") or axis_id,
            }
        )

    reasons = sorted(reasons, key=lambda r: r["pressure"] * r["weight"], reverse=True)
    importance_score = _clamp01(total_contrib / max(total_weight, 1.0), default=0.0)

    machine_semantics = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "scaffold_version": scaffold.get("version", 1),
        "axes": axes_out,
        "why_it_matters": {
            "score": round(importance_score, 3),
            "reasons": reasons[:5],
            "source": "machine_semantics",
        },
    }

    update_inastate("machine_semantics", machine_semantics)


def _scan_fragment_health():
    """
    Periodically launch a background scan for corrupted fragments so Ina can inspect them.
    """
    global _last_fragment_health_scan, _fragment_health_thread
    now = time.time()
    if _fragment_health_thread and _fragment_health_thread.is_alive():
        return
    if now - _last_fragment_health_scan < _FRAGMENT_HEALTH_COOLDOWN:
        return

    def _worker():
        global _last_fragment_health_scan, _fragment_health_thread
        try:
            summary = scan_fragment_integrity(CHILD, max_samples=6, preview_chars=220)
            if summary:
                try:
                    with _FRAGMENT_HEALTH_PATH.open("w", encoding="utf-8") as handle:
                        json.dump(summary, handle, indent=2, ensure_ascii=False)
                except Exception as exc:
                    log_to_statusbox(f"[Manager] Failed to persist fragment integrity summary: {exc}")
                update_inastate("fragment_integrity", summary)
        except Exception as exc:
            log_to_statusbox(f"[Manager] Fragment integrity scan failed: {exc}")
        finally:
            _last_fragment_health_scan = time.time()
            _fragment_health_thread = None

    _fragment_health_thread = threading.Thread(
        target=_worker, name="fragment_health_scan", daemon=True
    )
    _fragment_health_thread.start()

def rebuild_maps_if_needed():
    emo = get_inastate("emotion_snapshot") or {}
    fuzz = emo.get("fuzz_level", 0.0)
    drift = emo.get("symbolic_drift", 0.0)
    if fuzz > 0.7 or drift > 0.5:
        log_to_statusbox("[Manager] Rebuilding maps due to emotional drift.")
        safe_call(["python", "memory_graph.py"])
        safe_call(["python", "meaning_map.py"])
        safe_call(["python", "neural_graph.py"])
        safe_call(["python", "logic_map_builder.py"])
        safe_call(["python", "emotion_map.py"])
        update_inastate("last_map_rebuild", datetime.now(timezone.utc).isoformat())

def run_internal_loop():
    _ensure_continuity_thread()
    monitor_hunger()
    monitor_energy()
    _maybe_run_intuition_probe()

    def check_audio_index_change():
        config = load_config()
        state = get_inastate("audio_device_cache") or {}

        current = {
            "mic_headset_index": config.get("mic_headset_index"),
            "mic_webcam_index": config.get("mic_webcam_index"),
            "output_headset_index": config.get("output_headset_index"),
            "output_TV_index": config.get("output_TV_index")
        }

        if current != state:
            update_inastate("audio_device_cache", current)
            log_to_statusbox("[Manager] Detected change in audio config — restarting audio listener.")
            return True

        return False

    if check_audio_index_change():
        safe_call(["pkill", "-f", "audio_listener.py"])
        time.sleep(2)  # Let config settle and avoid early InputStream calls
        safe_popen(["python", "audio_listener.py"])



    if get_inastate("emotion_snapshot", {}).get("focus", 0.0) > 0.5:
        safe_popen(["python", "meditation_state.py"])

    if get_inastate("emotion_snapshot", {}).get("fuzz_level", 0.0) > 0.7:
        safe_popen(["python", "dreamstate.py"])

    safe_run(["python", "emotion_engine.py"])
    safe_run(["python", "instinct_engine.py"])

    safe_popen(["python", "early_comm.py"])

    if not feedback_inhibition():
        safe_popen(["python", "predictive_layer.py"])
        safe_popen(["python", "logic_engine.py"])

    boredom_check()
    _maybe_self_read()
    rebuild_maps_if_needed()
    _check_self_adjustment()
    _update_contact_urges()
    _update_stable_pattern_urge()
    _update_self_read_exploration_opportunities()
    _run_passive_reflection()
    _step_deep_recall()
    _update_machine_semantics()
    _scan_fragment_health()
    _run_prediction_meta_analysis()
    # Periodically evaluate alignment metrics and surface warnings if needed
    evaluate_alignment()

def schedule_runtime():
    log_to_statusbox("[Manager] Starting main runtime loop...")
    while True:
        try:
            run_internal_loop()
            time.sleep(10)
        except Exception as e:
            log_to_statusbox(f"[Manager ERROR] Runtime loop crashed: {e}")
            time.sleep(5)

if __name__ == "__main__":  
    launch_background_loops()
    schedule_runtime()
