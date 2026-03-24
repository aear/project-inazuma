# === model_manager.py (Final Rewrite + Module Awareness) ===

import json
import time
import subprocess
import uuid
import threading
import os
import random
import gc
import traceback
import math
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
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
from fragment_repair import process_corrupt_queue
from transformers.fractal_multidimensional_transformers import FractalTransformer

try:
    import psutil  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    psutil = None

def load_config():
    path = Path("config.json")
    if not path.exists():
        return {}
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}

config = load_config()
CHILD = config.get("current_child", "Inazuma_Yagami")
MEMORY_PATH = Path("AI_Children") / CHILD / "memory"
SELF_READ_PREF_PATH = MEMORY_PATH / "self_read_preferences.json"
_REFLECTION_LOG = Path("AI_Children") / CHILD / "identity" / "self_reflection.json"
RUNNING_MODULES_PATH = Path("running_modules.json")
_SEMANTIC_SCAFFOLD_PATH = MEMORY_PATH / "semantic_scaffold.json"
TYPED_OUTBOX_PATH = MEMORY_PATH / "typed_outbox.jsonl"
_DECISION_PANIC_LOG_PATH = MEMORY_PATH / "decision_panic_log.jsonl"
_SELF_QUESTIONS_PATH = MEMORY_PATH / "self_questions.json"
_FRAGMENT_HEALTH_PATH = MEMORY_PATH / "fragment_integrity.json"
_INASTATE_LOCK_PATH = MEMORY_PATH / "inastate.lock"

_DEFAULT_SELF_READ_SOURCE_CHOICES = {
    "code": True,
    "music": True,
    "books": True,
    "venv": False,
}

_MEMORY_GUARD_DEFAULTS = {
    "enabled": True,
    "ram_soft_percent": 35.0,
    "ram_hard_percent": 45.0,
    "swap_soft_percent": 5.0,
    "swap_hard_percent": 10.0,
    "min_available_gb": 8.0,
    "ina_soft_gb": 0.0,
    "ina_hard_gb": 0.0,
    "shed_on_soft": False,
    "shed_on_hard": True,
    "shed_cooldown_sec": 45.0,
    "shed_process_patterns": [],
    "queue_enabled": False,
    "queue_ram_used_gb": 0.0,
    "queue_swap_used_gb": 0.0,
    "queue_max_items": 64,
    "queue_event_cooldown_sec": 20.0,
    "queue_auto_shed": True,
    "queue_process_cooldown_sec": 30.0,
    "log_cooldown_sec": 120.0,
}
_META_ARBITRATION_DEFAULTS = {
    "enabled": True,
    "activation_threshold": 0.35,
    "conflict_margin": 0.12,
    "conflict_min_level": 0.3,
    "indecision_horizon_sec": 180.0,
    "resolution_decay": 0.55,
    "narrowing_gain": 0.35,
    "narrowing_band_high": 0.24,
    "narrowing_band_low": 0.08,
    "boost_gain": 0.22,
    "suppression_gain": 0.18,
    "stall_action_window_sec": 30.0,
    "log_discomfort_threshold": 0.65,
    "log_cooldown_sec": 90.0,
    "panic_enabled": True,
    "panic_discomfort_threshold": 0.82,
    "panic_conflict_threshold": 0.55,
    "panic_indecision_sec": 240.0,
    "panic_repeat_sec": 90.0,
    "panic_popup_enabled": True,
    "panic_popup_cooldown_sec": 300.0,
    "need_aliases_enabled": True,
    "auto_generate_need_aliases": True,
    "alias_prefix": "sym_need_",
}
_NEED_CANONICAL_VARIABLES = {
    "reversibility_estimate": {
        "description": "Estimated reversibility of candidate actions before committing.",
        "default_probe": "simulate action on shadow state",
        "default_provider_modules": ["logic_engine.py", "predictive_layer.py"],
        "default_output": "scalar [-1..1] where +1 is reversible",
    },
    "urgency_signal": {
        "description": "Relative urgency gradient to break ties across active urges.",
        "default_probe": "compare delayed-cost delta across candidate actions",
        "default_provider_modules": ["predictive_layer.py", "emotion_engine.py"],
        "default_output": "scalar [0..1] where higher means immediate action cost of delay",
    },
    "context_scope": {
        "description": "How complete and bounded the current decision context is.",
        "default_probe": "measure coverage of relevant world/self context channels",
        "default_provider_modules": ["memory_graph.py", "logic_engine.py"],
        "default_output": "scalar [0..1] where higher means sufficient context coverage",
    },
}
_MEMORY_GUARD_CHECK_INTERVAL = 5.0
_last_memory_guard_check = 0.0
_last_memory_guard_log = 0.0
_last_memory_guard_key: Optional[tuple] = None
_memory_guard_state: Dict[str, Any] = {"level": "unknown"}
_last_memory_shed_ts = 0.0
_last_memory_queue_event_ts = 0.0
_last_memory_queue_process_ts = 0.0
_RUNTIME_HEARTBEAT_INTERVAL = 30.0
_last_runtime_heartbeat = 0.0
_BUNDLE_POLICY_DEFAULTS = {
    "enabled": False,
    "allow_apply": False,
    "cooldown_seconds": 3600.0,
    "max_files_per_bundle": 10_000,
    "max_bundle_bytes": 512 * 1024 * 1024,
    "max_file_bytes": 256 * 1024,
    "bundle_dir": "bundles",
    "write_manifest": True,
    "follow_symlinks": False,
    "default_include": [],
    "default_exclude": [],
    "allowed_roots": [],
    "exclude_dirs": [],
}
_FRAGMENT_REPAIR_DEFAULTS = {
    "enabled": False,
    "mode": "quarantine",  # quarantine | delete | repair | inspect
    "require_intent": True,
    "consume_intent": True,
    "cooldown_seconds": 900.0,
    "max_actions_per_pass": 3,
    "max_repair_bytes": 256 * 1024,
    "quarantine_dir": "fragments/corrupt",
    "pending_delete_dir": "fragments/pending_delete",
}

def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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


def _clamp_percent(value: Any, default: float) -> float:
    coerced = _coerce_float(value, default)
    return max(0.0, min(100.0, coerced))


def _clamp_positive(value: Any, default: float) -> float:
    coerced = _coerce_float(value, default)
    return coerced if coerced >= 0.0 else default


def _coerce_str_list(value: Any) -> List[str]:
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if item]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False


def _ina_process_tree_rss_gb() -> float:
    if psutil is None:
        return 0.0
    try:
        root = psutil.Process(os.getpid())
    except Exception:
        return 0.0
    procs = [root]
    try:
        procs.extend(root.children(recursive=True))
    except Exception:
        pass

    rss_bytes = 0
    for proc in procs:
        try:
            rss_bytes += int(proc.memory_info().rss)
        except Exception:
            continue
    return rss_bytes / (1024.0 ** 3)


def _trim_allocator_memory() -> bool:
    """
    Best-effort trim for glibc allocators on Linux to return free arenas.
    Safe no-op when unavailable.
    """
    try:
        import ctypes  # local import to avoid hard dependency

        libc = ctypes.CDLL("libc.so.6")
        if hasattr(libc, "malloc_trim"):
            libc.malloc_trim(0)
            return True
    except Exception:
        return False
    return False


def _memory_guard_limits(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = cfg or load_config()
    guard_raw = cfg.get("memory_guard") if isinstance(cfg, dict) else None
    guard = guard_raw if isinstance(guard_raw, dict) else {}

    limits = _MEMORY_GUARD_DEFAULTS.copy()
    limits["enabled"] = _coerce_bool(guard.get("enabled", limits["enabled"]), limits["enabled"])

    for key in ("ram_soft_percent", "ram_hard_percent", "swap_soft_percent", "swap_hard_percent"):
        if key in guard:
            limits[key] = _clamp_percent(guard.get(key), limits[key])

    if "min_available_gb" in guard:
        limits["min_available_gb"] = _clamp_positive(guard.get("min_available_gb"), limits["min_available_gb"])
    if "ina_soft_gb" in guard:
        limits["ina_soft_gb"] = _clamp_positive(guard.get("ina_soft_gb"), limits["ina_soft_gb"])
    if "ina_hard_gb" in guard:
        limits["ina_hard_gb"] = _clamp_positive(guard.get("ina_hard_gb"), limits["ina_hard_gb"])
    if "shed_on_soft" in guard:
        limits["shed_on_soft"] = _coerce_bool(guard.get("shed_on_soft"), limits["shed_on_soft"])
    if "shed_on_hard" in guard:
        limits["shed_on_hard"] = _coerce_bool(guard.get("shed_on_hard"), limits["shed_on_hard"])
    if "shed_cooldown_sec" in guard:
        limits["shed_cooldown_sec"] = _clamp_positive(guard.get("shed_cooldown_sec"), limits["shed_cooldown_sec"])
    if "shed_process_patterns" in guard:
        limits["shed_process_patterns"] = _coerce_str_list(guard.get("shed_process_patterns"))
    if "queue_enabled" in guard:
        limits["queue_enabled"] = _coerce_bool(guard.get("queue_enabled"), limits["queue_enabled"])
    if "queue_ram_used_gb" in guard:
        limits["queue_ram_used_gb"] = _clamp_positive(guard.get("queue_ram_used_gb"), limits["queue_ram_used_gb"])
    if "queue_swap_used_gb" in guard:
        limits["queue_swap_used_gb"] = _clamp_positive(guard.get("queue_swap_used_gb"), limits["queue_swap_used_gb"])
    if "queue_max_items" in guard:
        limits["queue_max_items"] = max(1, _coerce_int(guard.get("queue_max_items"), int(limits["queue_max_items"])))
    if "queue_event_cooldown_sec" in guard:
        limits["queue_event_cooldown_sec"] = _clamp_positive(
            guard.get("queue_event_cooldown_sec"), limits["queue_event_cooldown_sec"]
        )
    if "queue_auto_shed" in guard:
        limits["queue_auto_shed"] = _coerce_bool(guard.get("queue_auto_shed"), limits["queue_auto_shed"])
    if "queue_process_cooldown_sec" in guard:
        limits["queue_process_cooldown_sec"] = _clamp_positive(
            guard.get("queue_process_cooldown_sec"), limits["queue_process_cooldown_sec"]
        )
    if "log_cooldown_sec" in guard:
        limits["log_cooldown_sec"] = _clamp_positive(guard.get("log_cooldown_sec"), limits["log_cooldown_sec"])

    if "ram_hard_percent" not in guard:
        try:
            fraction = float(cfg.get("max_ram_fraction"))
        except (TypeError, ValueError):
            fraction = None
        if fraction is not None and fraction > 0:
            hard = max(0.0, min(100.0, fraction * 100.0))
            limits["ram_hard_percent"] = hard
            limits["ram_soft_percent"] = min(limits["ram_soft_percent"], hard * 0.85)

    limits["ram_soft_percent"] = min(limits["ram_soft_percent"], limits["ram_hard_percent"])
    limits["swap_soft_percent"] = min(limits["swap_soft_percent"], limits["swap_hard_percent"])
    if limits["ina_hard_gb"] > 0:
        if limits["ina_soft_gb"] <= 0:
            limits["ina_soft_gb"] = max(0.0, limits["ina_hard_gb"] * 0.8)
        else:
            limits["ina_soft_gb"] = min(limits["ina_soft_gb"], limits["ina_hard_gb"])
    return limits


def _meta_arbitration_limits(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = cfg or load_config()
    raw = cfg.get("meta_arbitration") if isinstance(cfg, dict) else None
    section = raw if isinstance(raw, dict) else {}

    limits = _META_ARBITRATION_DEFAULTS.copy()
    limits["enabled"] = _coerce_bool(section.get("enabled", limits["enabled"]), limits["enabled"])
    limits["panic_enabled"] = _coerce_bool(section.get("panic_enabled", limits["panic_enabled"]), limits["panic_enabled"])
    limits["panic_popup_enabled"] = _coerce_bool(
        section.get("panic_popup_enabled", limits["panic_popup_enabled"]),
        limits["panic_popup_enabled"],
    )
    limits["need_aliases_enabled"] = _coerce_bool(
        section.get("need_aliases_enabled", limits["need_aliases_enabled"]),
        limits["need_aliases_enabled"],
    )
    limits["auto_generate_need_aliases"] = _coerce_bool(
        section.get("auto_generate_need_aliases", limits["auto_generate_need_aliases"]),
        limits["auto_generate_need_aliases"],
    )

    for key in (
        "activation_threshold",
        "conflict_margin",
        "conflict_min_level",
        "resolution_decay",
        "narrowing_gain",
        "narrowing_band_high",
        "narrowing_band_low",
        "boost_gain",
        "suppression_gain",
        "log_discomfort_threshold",
        "panic_discomfort_threshold",
        "panic_conflict_threshold",
    ):
        if key in section:
            limits[key] = _clamp01(section.get(key), default=limits[key])

    for key in (
        "indecision_horizon_sec",
        "stall_action_window_sec",
        "log_cooldown_sec",
        "panic_indecision_sec",
        "panic_repeat_sec",
        "panic_popup_cooldown_sec",
    ):
        if key in section:
            limits[key] = _clamp_positive(section.get(key), limits[key])

    limits["conflict_margin"] = max(0.01, limits["conflict_margin"])
    limits["indecision_horizon_sec"] = max(5.0, limits["indecision_horizon_sec"])
    limits["stall_action_window_sec"] = max(1.0, limits["stall_action_window_sec"])
    limits["panic_indecision_sec"] = max(10.0, limits["panic_indecision_sec"])
    limits["panic_repeat_sec"] = max(10.0, limits["panic_repeat_sec"])
    limits["panic_popup_cooldown_sec"] = max(30.0, limits["panic_popup_cooldown_sec"])
    limits["resolution_decay"] = min(1.0, max(0.0, limits["resolution_decay"]))
    alias_prefix = section.get("alias_prefix")
    if isinstance(alias_prefix, str) and alias_prefix.strip():
        limits["alias_prefix"] = alias_prefix.strip().lower()
    if not str(limits.get("alias_prefix", "")).startswith("sym_need_"):
        limits["alias_prefix"] = "sym_need_"
    limits["narrowing_band_high"] = max(0.01, limits["narrowing_band_high"])
    limits["narrowing_band_low"] = max(0.0, limits["narrowing_band_low"])
    if limits["narrowing_band_low"] > limits["narrowing_band_high"]:
        limits["narrowing_band_low"], limits["narrowing_band_high"] = (
            limits["narrowing_band_high"],
            limits["narrowing_band_low"],
        )
    return limits


def _publish_memory_guard_state(state: Dict[str, Any]) -> None:
    global _last_memory_guard_key
    key = (
        state.get("level"),
        state.get("ram_percent"),
        state.get("swap_percent"),
        state.get("ram_available_gb"),
        state.get("ina_rss_gb"),
    )
    if key == _last_memory_guard_key:
        return
    update_inastate("memory_guard", state)
    _last_memory_guard_key = key


def _maybe_log_memory_guard_state(state: Dict[str, Any], limits: Dict[str, Any]) -> None:
    global _last_memory_guard_log
    level = state.get("level")
    if level not in {"soft", "hard"}:
        return
    now = time.time()
    cooldown = float(limits.get("log_cooldown_sec", _MEMORY_GUARD_DEFAULTS["log_cooldown_sec"]))
    if _last_memory_guard_log and (now - _last_memory_guard_log) < cooldown:
        return
    _last_memory_guard_log = now
    ina_rss = state.get("ina_rss_gb")
    ina_text = f", Ina RSS {ina_rss}GB" if isinstance(ina_rss, (int, float)) else ""
    log_to_statusbox(
        "[Manager] Memory pressure "
        f"{level}: RAM {state.get('ram_percent')}% "
        f"(avail {state.get('ram_available_gb')}GB), "
        f"swap {state.get('swap_percent')}%{ina_text}. Deferring non-critical work."
    )


def _refresh_memory_guard_state(force: bool = False) -> Dict[str, Any]:
    global _memory_guard_state, _last_memory_guard_check
    now = time.time()
    if not force and _memory_guard_state and (now - _last_memory_guard_check) < _MEMORY_GUARD_CHECK_INTERVAL:
        return _memory_guard_state

    _last_memory_guard_check = now
    timestamp = datetime.now(timezone.utc).isoformat()
    limits = _memory_guard_limits()

    if psutil is None:
        state = {"timestamp": timestamp, "level": "unknown", "reason": "psutil_unavailable"}
    elif not limits.get("enabled", True):
        state = {"timestamp": timestamp, "level": "disabled", "limits": limits}
    else:
        vm = psutil.virtual_memory()
        swap = psutil.swap_memory()
        available_gb = vm.available / (1024.0 ** 3)
        ina_rss_gb = _ina_process_tree_rss_gb()
        triggers = []
        level = "ok"

        hard = False
        if limits["min_available_gb"] > 0 and available_gb <= limits["min_available_gb"]:
            triggers.append("available_low")
            hard = True
        if limits["ram_hard_percent"] > 0 and vm.percent >= limits["ram_hard_percent"]:
            triggers.append("ram_hard")
            hard = True
        if limits["swap_hard_percent"] > 0 and swap.percent >= limits["swap_hard_percent"]:
            triggers.append("swap_hard")
            hard = True
        if limits["ina_hard_gb"] > 0 and ina_rss_gb >= limits["ina_hard_gb"]:
            triggers.append("ina_hard")
            hard = True

        if hard:
            level = "hard"
        else:
            soft = False
            if limits["ram_soft_percent"] > 0 and vm.percent >= limits["ram_soft_percent"]:
                triggers.append("ram_soft")
                soft = True
            if limits["swap_soft_percent"] > 0 and swap.percent >= limits["swap_soft_percent"]:
                triggers.append("swap_soft")
                soft = True
            if limits["ina_soft_gb"] > 0 and ina_rss_gb >= limits["ina_soft_gb"]:
                triggers.append("ina_soft")
                soft = True
            if soft:
                level = "soft"

        state = {
            "timestamp": timestamp,
            "level": level,
            "ram_percent": round(vm.percent, 1),
            "ram_used_gb": round(vm.used / (1024.0 ** 3), 2),
            "ram_available_gb": round(available_gb, 2),
            "swap_percent": round(swap.percent, 1),
            "swap_used_gb": round(swap.used / (1024.0 ** 3), 2),
            "ina_rss_gb": round(ina_rss_gb, 2),
            "limits": {
                "ram_soft_percent": limits["ram_soft_percent"],
                "ram_hard_percent": limits["ram_hard_percent"],
                "swap_soft_percent": limits["swap_soft_percent"],
                "swap_hard_percent": limits["swap_hard_percent"],
                "min_available_gb": limits["min_available_gb"],
                "ina_soft_gb": limits["ina_soft_gb"],
                "ina_hard_gb": limits["ina_hard_gb"],
                "queue_enabled": limits["queue_enabled"],
                "queue_ram_used_gb": limits["queue_ram_used_gb"],
                "queue_swap_used_gb": limits["queue_swap_used_gb"],
            },
            "triggers": triggers,
        }

    _memory_guard_state = state
    if isinstance(state, dict) and state.get("level") in {"soft", "hard"}:
        _maybe_log_memory_guard_state(state, limits)
    _publish_memory_guard_state(state)
    return state


def _default_shed_patterns() -> List[str]:
    return [
        "dreamstate.py",
        "meditation_state.py",
        "early_comm.py",
        "predictive_layer.py",
        "logic_engine.py",
        "paint_window.py",
        "boredom_state.py",
        "trauma_processor.py",
    ]


def _shed_memory_now(
    *,
    level: str,
    limits: Dict[str, Any],
    reason: str,
    respect_cooldown: bool = True,
) -> bool:
    global _last_memory_shed_ts
    now = time.time()
    cooldown = float(limits.get("shed_cooldown_sec", _MEMORY_GUARD_DEFAULTS["shed_cooldown_sec"]))
    if respect_cooldown and _last_memory_shed_ts and (now - _last_memory_shed_ts) < cooldown:
        return False

    gc.collect()
    trimmed = _trim_allocator_memory()

    patterns = limits.get("shed_process_patterns") or _default_shed_patterns()
    stopped: List[str] = []
    for pattern in patterns:
        if not pattern:
            continue
        if _is_process_running(pattern):
            safe_call(["pkill", "-f", pattern], description=f"memory_shed_stop:{pattern}")
            stopped.append(pattern)

    _last_memory_shed_ts = now
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "level": level,
        "reason": reason,
        "trimmed_allocator": bool(trimmed),
        "stopped_processes": stopped,
    }
    update_inastate("memory_shed_last", payload)
    log_to_statusbox(
        f"[Manager] Memory shed ({level}/{reason}): gc run, allocator_trim={bool(trimmed)}, stopped={len(stopped)}."
    )
    return True


def _maybe_shed_memory_pressure(level: str, limits: Dict[str, Any]) -> None:
    if level not in {"soft", "hard"}:
        return
    if level == "soft" and not limits.get("shed_on_soft", False):
        return
    if level == "hard" and not limits.get("shed_on_hard", True):
        return
    _shed_memory_now(level=level, limits=limits, reason="memory_guard", respect_cooldown=True)


def _push_memory_pressure_queue_event(event: Dict[str, Any], limits: Dict[str, Any]) -> None:
    queue = get_inastate("memory_pressure_queue") or []
    if not isinstance(queue, list):
        queue = []
    queue.append(event)
    max_items = max(1, int(limits.get("queue_max_items", _MEMORY_GUARD_DEFAULTS["queue_max_items"])))
    queue = queue[-max_items:]
    update_inastate("memory_pressure_queue", queue)
    update_inastate(
        "memory_pressure_queue_state",
        {
            "pending": len(queue),
            "last_event": event,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )


def _maybe_enqueue_memory_pressure_event(state: Dict[str, Any], limits: Dict[str, Any]) -> None:
    global _last_memory_queue_event_ts
    if not limits.get("queue_enabled", False):
        return
    if not isinstance(state, dict):
        return

    ram_used = _coerce_float(state.get("ram_used_gb"), 0.0)
    swap_used = _coerce_float(state.get("swap_used_gb"), 0.0)
    ram_limit = _coerce_float(limits.get("queue_ram_used_gb"), 0.0)
    swap_limit = _coerce_float(limits.get("queue_swap_used_gb"), 0.0)

    reasons = []
    if ram_limit > 0 and ram_used >= ram_limit:
        reasons.append("ram_used")
    if swap_limit > 0 and swap_used >= swap_limit:
        reasons.append("swap_used")
    if not reasons:
        return

    now = time.time()
    event_cooldown = float(
        limits.get("queue_event_cooldown_sec", _MEMORY_GUARD_DEFAULTS["queue_event_cooldown_sec"])
    )
    if _last_memory_queue_event_ts and (now - _last_memory_queue_event_ts) < event_cooldown:
        return

    event = {
        "id": uuid.uuid4().hex[:12],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": "memory_guard",
        "reasons": reasons,
        "ram_used_gb": ram_used,
        "swap_used_gb": swap_used,
        "ina_rss_gb": _coerce_float(state.get("ina_rss_gb"), 0.0),
        "limits": {
            "queue_ram_used_gb": ram_limit,
            "queue_swap_used_gb": swap_limit,
        },
    }
    _push_memory_pressure_queue_event(event, limits)
    _last_memory_queue_event_ts = now
    log_to_statusbox(
        f"[Manager] Memory event queued: ram={ram_used:.2f}GB swap={swap_used:.2f}GB reasons={','.join(reasons)}."
    )


def _process_memory_pressure_queue(limits: Dict[str, Any]) -> None:
    global _last_memory_queue_process_ts
    queue = get_inastate("memory_pressure_queue") or []
    if not isinstance(queue, list) or not queue:
        return
    if not limits.get("queue_auto_shed", True):
        return

    now = time.time()
    process_cooldown = float(
        limits.get("queue_process_cooldown_sec", _MEMORY_GUARD_DEFAULTS["queue_process_cooldown_sec"])
    )
    if _last_memory_queue_process_ts and (now - _last_memory_queue_process_ts) < process_cooldown:
        return

    event = queue[0]
    did_shed = _shed_memory_now(level="queue", limits=limits, reason="queued_event", respect_cooldown=True)
    if not did_shed:
        return

    queue.pop(0)
    update_inastate("memory_pressure_queue", queue)
    update_inastate(
        "memory_pressure_queue_state",
        {
            "pending": len(queue),
            "last_processed": event,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )
    _last_memory_queue_process_ts = now


def _consume_operator_memory_signal(limits: Dict[str, Any]) -> None:
    signal = get_inastate("operator_memory_signal")
    if not isinstance(signal, dict):
        return
    action = str(signal.get("action") or "").strip().lower()
    if action not in {"too_much_memory", "memory_too_high", "shed_memory_now"}:
        return

    update_inastate("operator_memory_signal", None)
    state = _refresh_memory_guard_state(force=True)
    event = {
        "id": uuid.uuid4().hex[:12],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": str(signal.get("source") or "operator"),
        "action": action,
        "note": signal.get("note"),
        "ram_used_gb": _coerce_float(state.get("ram_used_gb"), 0.0),
        "swap_used_gb": _coerce_float(state.get("swap_used_gb"), 0.0),
        "ina_rss_gb": _coerce_float(state.get("ina_rss_gb"), 0.0),
        "reasons": ["operator_signal"],
    }
    _push_memory_pressure_queue_event(event, limits)
    _shed_memory_now(level="operator", limits=limits, reason="operator_signal", respect_cooldown=False)


def _bundle_policy(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = cfg or load_config()
    raw = cfg.get("bundle_policy") if isinstance(cfg, dict) else None
    raw = raw if isinstance(raw, dict) else {}

    policy = _BUNDLE_POLICY_DEFAULTS.copy()
    policy["enabled"] = _coerce_bool(raw.get("enabled", policy["enabled"]), policy["enabled"])
    policy["allow_apply"] = _coerce_bool(raw.get("allow_apply", policy["allow_apply"]), policy["allow_apply"])
    policy["write_manifest"] = _coerce_bool(raw.get("write_manifest", policy["write_manifest"]), policy["write_manifest"])
    policy["follow_symlinks"] = _coerce_bool(raw.get("follow_symlinks", policy["follow_symlinks"]), policy["follow_symlinks"])
    policy["cooldown_seconds"] = _clamp_positive(raw.get("cooldown_seconds", policy["cooldown_seconds"]), policy["cooldown_seconds"])

    policy["max_files_per_bundle"] = max(
        1, _coerce_int(raw.get("max_files_per_bundle", policy["max_files_per_bundle"]), policy["max_files_per_bundle"])
    )
    policy["max_bundle_bytes"] = max(
        1, _coerce_int(raw.get("max_bundle_bytes", policy["max_bundle_bytes"]), policy["max_bundle_bytes"])
    )
    policy["max_file_bytes"] = max(
        1, _coerce_int(raw.get("max_file_bytes", policy["max_file_bytes"]), policy["max_file_bytes"])
    )

    policy["bundle_dir"] = str(raw.get("bundle_dir", policy["bundle_dir"]) or policy["bundle_dir"])
    policy["default_include"] = _coerce_str_list(raw.get("default_include"))
    policy["default_exclude"] = _coerce_str_list(raw.get("default_exclude"))
    policy["allowed_roots"] = _coerce_str_list(raw.get("allowed_roots"))
    policy["exclude_dirs"] = _coerce_str_list(raw.get("exclude_dirs"))
    return policy


def _fragment_repair_policy(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = cfg or load_config()
    raw = cfg.get("fragment_repair_policy") if isinstance(cfg, dict) else None
    raw = raw if isinstance(raw, dict) else {}

    policy = _FRAGMENT_REPAIR_DEFAULTS.copy()
    policy["enabled"] = _coerce_bool(raw.get("enabled", policy["enabled"]), policy["enabled"])
    policy["require_intent"] = _coerce_bool(raw.get("require_intent", policy["require_intent"]), policy["require_intent"])
    policy["consume_intent"] = _coerce_bool(raw.get("consume_intent", policy["consume_intent"]), policy["consume_intent"])
    policy["cooldown_seconds"] = _clamp_positive(raw.get("cooldown_seconds", policy["cooldown_seconds"]), policy["cooldown_seconds"])

    mode = str(raw.get("mode", policy["mode"]) or policy["mode"]).lower()
    if mode not in {"quarantine", "delete", "repair", "inspect"}:
        mode = policy["mode"]
    policy["mode"] = mode

    policy["max_actions_per_pass"] = max(
        1, _coerce_int(raw.get("max_actions_per_pass", policy["max_actions_per_pass"]), policy["max_actions_per_pass"])
    )
    policy["max_repair_bytes"] = max(
        0, _coerce_int(raw.get("max_repair_bytes", policy["max_repair_bytes"]), policy["max_repair_bytes"])
    )
    policy["quarantine_dir"] = str(raw.get("quarantine_dir", policy["quarantine_dir"]) or policy["quarantine_dir"])
    policy["pending_delete_dir"] = str(raw.get("pending_delete_dir", policy["pending_delete_dir"]) or policy["pending_delete_dir"])
    return policy


def _resolve_bundle_roots(policy: Dict[str, Any], child: str) -> List[Path]:
    roots: List[Path] = []
    for entry in policy.get("allowed_roots", []) or []:
        if not entry:
            continue
        try:
            formatted = entry.format(child=child)
        except Exception:
            formatted = entry
        roots.append(Path(formatted).expanduser())
    if not roots:
        roots = [Path("AI_Children") / child / "memory"]
    return [root.resolve() for root in roots]


def _bounded_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    if value is None:
        return default
    coerced = _coerce_int(value, default)
    return max(minimum, min(coerced, maximum))


def _maybe_bundle_memory(defer_optional: bool) -> None:
    global _last_bundle_launch, _bundle_thread
    if defer_optional:
        return
    if get_inastate("dreaming") or get_inastate("meditating"):
        return

    policy = _bundle_policy()
    if not policy.get("enabled", False):
        return
    if _bundle_thread and _bundle_thread.is_alive():
        return

    request = get_inastate("bundle_request")
    if not request:
        return
    if not isinstance(request, dict):
        update_inastate("bundle_request", None)
        update_inastate(
            "bundle_status",
            {
                "status": "error",
                "reason": "bundle_request_not_dict",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )
        return

    now = time.time()
    cooldown = float(policy.get("cooldown_seconds", _BUNDLE_POLICY_DEFAULTS["cooldown_seconds"]))
    if _last_bundle_launch and (now - _last_bundle_launch) < cooldown:
        return

    child = load_config().get("current_child", CHILD)
    allowed_roots = _resolve_bundle_roots(policy, child)
    root_value = request.get("root") or None
    if root_value:
        root = Path(str(root_value)).expanduser()
    else:
        root = allowed_roots[0]
    root = root.resolve()
    if not any(_is_relative_to(root, allowed) for allowed in allowed_roots):
        update_inastate("bundle_request", None)
        update_inastate(
            "bundle_status",
            {
                "status": "blocked",
                "reason": "root_not_allowed",
                "root": str(root),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )
        return
    if not root.exists():
        update_inastate("bundle_request", None)
        update_inastate(
            "bundle_status",
            {
                "status": "error",
                "reason": "root_missing",
                "root": str(root),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )
        return
    if not root.is_dir():
        update_inastate("bundle_request", None)
        update_inastate(
            "bundle_status",
            {
                "status": "error",
                "reason": "root_not_dir",
                "root": str(root),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )
        return

    bundle_dir_value = request.get("bundle_dir") or policy.get("bundle_dir")
    if bundle_dir_value:
        bundle_dir = Path(str(bundle_dir_value)).expanduser()
        if not bundle_dir.is_absolute():
            bundle_dir = root / bundle_dir
    else:
        bundle_dir = root / "bundles"
    bundle_dir_resolved = bundle_dir.resolve() if bundle_dir.exists() else bundle_dir
    if not _is_relative_to(bundle_dir_resolved, root):
        update_inastate("bundle_request", None)
        update_inastate(
            "bundle_status",
            {
                "status": "blocked",
                "reason": "bundle_dir_not_under_root",
                "bundle_dir": str(bundle_dir),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )
        return

    include_globs = _coerce_str_list(request.get("include")) or policy.get("default_include", [])
    exclude_globs = list(policy.get("default_exclude", []))
    for entry in _coerce_str_list(request.get("exclude")):
        if entry not in exclude_globs:
            exclude_globs.append(entry)

    max_files_per_bundle = _bounded_int(
        request.get("max_files_per_bundle"),
        policy["max_files_per_bundle"],
        1,
        policy["max_files_per_bundle"],
    )
    max_bundle_bytes = _bounded_int(
        request.get("max_bundle_bytes"),
        policy["max_bundle_bytes"],
        1,
        policy["max_bundle_bytes"],
    )
    max_file_bytes = _bounded_int(
        request.get("max_file_bytes"),
        policy["max_file_bytes"],
        1,
        policy["max_file_bytes"],
    )
    max_file_bytes = min(max_file_bytes, max_bundle_bytes)

    apply_requested = _coerce_bool(request.get("apply", False), False)
    apply_allowed = apply_requested and policy.get("allow_apply", False)
    apply_note = "apply_not_allowed" if apply_requested and not apply_allowed else None
    follow_symlinks = _coerce_bool(request.get("follow_symlinks", policy["follow_symlinks"]), policy["follow_symlinks"])
    write_manifest = _coerce_bool(request.get("write_manifest", policy["write_manifest"]), policy["write_manifest"])

    action_desc = f"memory bundle {'apply' if apply_allowed else 'plan'}"
    feedback = check_action({"description": action_desc})
    if not feedback.get("overall", {}).get("pass", False):
        update_inastate("bundle_request", None)
        update_inastate(
            "bundle_status",
            {
                "status": "blocked",
                "reason": feedback.get("overall", {}).get("rationale"),
                "action": action_desc,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )
        return

    update_inastate("bundle_request", None)
    _last_bundle_launch = now
    started_at = datetime.now(timezone.utc).isoformat()

    def _worker():
        global _bundle_thread
        try:
            from memory_bundler import apply_bundle, plan_bundle

            if apply_allowed:
                report = apply_bundle(
                    root=root,
                    bundle_dir=bundle_dir,
                    include_globs=include_globs,
                    exclude_globs=exclude_globs,
                    follow_symlinks=follow_symlinks,
                    max_file_bytes=max_file_bytes,
                    max_bundle_bytes=max_bundle_bytes,
                    max_files_per_bundle=max_files_per_bundle,
                    exclude_dirs=policy.get("exclude_dirs"),
                    write_manifest=write_manifest,
                )
                status = "applied"
            else:
                report = plan_bundle(
                    root=root,
                    bundle_dir=bundle_dir,
                    include_globs=include_globs,
                    exclude_globs=exclude_globs,
                    follow_symlinks=follow_symlinks,
                    max_file_bytes=max_file_bytes,
                    max_bundle_bytes=max_bundle_bytes,
                    max_files_per_bundle=max_files_per_bundle,
                    exclude_dirs=policy.get("exclude_dirs"),
                    write_manifest=write_manifest,
                )
                status = "planned"

            payload = report.to_dict()
            payload.update(
                {
                    "status": status,
                    "apply_requested": apply_requested,
                    "apply_allowed": apply_allowed,
                    "apply_note": apply_note,
                    "root": str(root),
                    "bundle_dir": str(bundle_dir),
                    "include": include_globs,
                    "exclude": exclude_globs,
                    "max_files_per_bundle": max_files_per_bundle,
                    "max_bundle_bytes": max_bundle_bytes,
                    "max_file_bytes": max_file_bytes,
                    "started_at": started_at,
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            update_inastate("bundle_status", payload)
            update_inastate(
                "bundle_prune_ready",
                {
                    "ready": status == "applied",
                    "status": status,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "root": str(root),
                    "bundle_dir": str(bundle_dir),
                },
            )
            log_to_statusbox(
                f"[Manager] Memory bundle {status}: {report.total_files} -> {report.new_file_count}"
            )
        except Exception as exc:
            update_inastate(
                "bundle_status",
                {
                    "status": "failed",
                    "reason": str(exc),
                    "root": str(root),
                    "bundle_dir": str(bundle_dir),
                    "started_at": started_at,
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            update_inastate(
                "bundle_prune_ready",
                {
                    "ready": False,
                    "status": "failed",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "root": str(root),
                    "bundle_dir": str(bundle_dir),
                },
            )
            log_to_statusbox(f"[Manager] Memory bundling failed: {exc}")
        finally:
            _bundle_thread = None

    _bundle_thread = threading.Thread(target=_worker, name="memory_bundle", daemon=True)
    _bundle_thread.start()

reflection_core = SelfReflectionCore(ina_reference="model_manager")
adjustment_scheduler = SelfAdjustmentScheduler()
memory_manager = MemoryManager(CHILD)
continuity_manager = ContinuityManager(CHILD)
intuition_engine = QuantumIntuitionEngine(CHILD)
_last_opportunities = set()
_last_boredom_launch = 0.0
_BOREDOM_COOLDOWN = 30  # seconds
_last_paint_launch = 0.0
_PAINT_COOLDOWN = 600  # seconds
_last_self_read_launch = 0.0
_SELF_READ_COOLDOWN = 300  # seconds
_last_self_read_hold_log = 0.0
_SELF_READ_HOLD_LOG_COOLDOWN = 120.0
_last_voice_urge_log = 0.0
_last_typing_urge_log = 0.0
_COMM_URGE_LOG_COOLDOWN = 180  # seconds
_last_stable_urge_log = 0.0
_STABLE_URGE_LOG_COOLDOWN = 180  # seconds
_last_motor_intent_ts = 0.0
_MOTOR_INTENT_COOLDOWN = 12.0
_MOTOR_URGE_THRESHOLD = 0.6
_last_walk_to_marker_ts = 0.0
_walk_to_marker_attempt: Optional[Dict[str, Any]] = None
_ground_fault_streak = 0
_ground_fault_clear_streak = 0
_ground_fault_active = False
_ground_fault_window_id: Optional[str] = None
_ground_fault_window_started_at: Optional[str] = None
_last_ground_fault_log = 0.0
_GROUND_FAULT_LOG_COOLDOWN = 60.0
_last_continuity_run = 0.0
_CONTINUITY_COOLDOWN = 3600.0  # seconds between heavy continuity sweeps
_last_intuition_run = 0.0
_INTUITION_COOLDOWN = 600.0
_last_meta_alert = 0.0
_META_ALERT_COOLDOWN = 900.0
_last_meta_arbitration_log = 0.0
_last_fragment_health_scan = 0.0
_FRAGMENT_HEALTH_COOLDOWN = 1800.0
_fragment_health_thread: Optional[threading.Thread] = None
_last_fragment_repair_run = 0.0
_last_bundle_launch = 0.0
_bundle_thread: Optional[threading.Thread] = None
_last_world_sense_mode: Optional[str] = None
_last_exploration_invite_log = 0.0
_EXPLORATION_INVITE_COOLDOWN = 240.0
_last_humor_bridge_log = 0.0
_HUMOR_BRIDGE_LOG_COOLDOWN = 240.0
_last_trauma_run = 0.0
_TRAUMA_COOLDOWN = 900.0
_last_ina_client_check = 0.0
_INA_CLIENT_CHECK_COOLDOWN = 300.0
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


@contextmanager
def _inastate_lock():
    try:
        import fcntl  # Unix-only; best-effort on other platforms.
    except Exception:
        yield
        return
    _INASTATE_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_INASTATE_LOCK_PATH, "w") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX)
        except Exception:
            yield
            return
        try:
            yield
        finally:
            try:
                fcntl.flock(lock_handle, fcntl.LOCK_UN)
            except Exception:
                pass


def _load_inastate_state() -> Dict[str, Any]:
    path = MEMORY_PATH / "inastate.json"
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _atomic_write_inastate(state: Dict[str, Any]) -> None:
    path = MEMORY_PATH / "inastate.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=4)
    os.replace(tmp_path, path)


def safe_popen(cmd, description=None, **popen_kwargs):
    action = {"command": cmd, "description": description or " ".join(map(str, cmd))}
    feedback = check_action(action)
    if not feedback["overall"]["pass"]:
        log_to_statusbox(
            f"[Manager] Alignment blocked: {feedback['overall']['rationale']} ({action['description']})"
        )
        return None
    try:
        return subprocess.Popen(cmd, **popen_kwargs)
    except Exception as e:
        log_to_statusbox(f"[Manager] Failed to start {' '.join(map(str, cmd))}: {e}")
        return None


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


def _is_process_running(pattern: str) -> bool:
    try:
        result = subprocess.run(
            ["pgrep", "-f", pattern],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return result.returncode == 0
    except Exception:
        return False


def _apply_world_sense_override() -> None:
    global _last_world_sense_mode
    world_connected = bool(get_inastate("world_connected", False))
    target_mode = "world" if world_connected else "default"
    if target_mode == _last_world_sense_mode:
        return
    _last_world_sense_mode = target_mode

    update_inastate("sensory_mode", target_mode)
    update_inastate("vision_mode", target_mode)
    update_inastate("audio_mode", target_mode)

    if world_connected:
        safe_call(["pkill", "-f", "audio_listener.py"])
        if not _is_process_running("vision_window.py"):
            safe_popen(["python", "vision_window.py"])
        try:
            auto_launch = bool(load_config().get("auto_launch_ina_viewer", False))
        except Exception:
            auto_launch = False
        if auto_launch and not _is_process_running("ina_arch_viewer.py"):
            safe_popen(["python", "ina_arch_viewer.py"])
        log_to_statusbox("[Manager] World connected: overriding senses with 3D mode.")
    else:
        if not _is_process_running("audio_listener.py"):
            safe_popen(["python", "audio_listener.py"])
        if not _is_process_running("vision_window.py"):
            safe_popen(["python", "vision_window.py"])
        log_to_statusbox("[Manager] World disconnected: restoring default senses.")


def _maybe_ensure_ina_client() -> None:
    global _last_ina_client_check
    now = time.time()
    try:
        interval = float(load_config().get("ina_client_check_interval", _INA_CLIENT_CHECK_COOLDOWN))
    except Exception:
        interval = _INA_CLIENT_CHECK_COOLDOWN
    if interval <= 0:
        return
    if _last_ina_client_check and (now - _last_ina_client_check) < interval:
        return
    _last_ina_client_check = now
    if not _is_process_running("ina_client.py"):
        safe_popen(["python", "ina_client.py", "--daemon"])
        log_to_statusbox("[Manager] Restarted ina_client (was not running).")

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
    if not isinstance(state, dict):
        return default
    return state.get(key, default)


def update_inastate(key, value):
    with _inastate_lock():
        state = _load_inastate_state()
        state[key] = value
        _atomic_write_inastate(state)


def _maybe_update_runtime_heartbeat() -> None:
    global _last_runtime_heartbeat
    now = time.time()
    if _last_runtime_heartbeat and (now - _last_runtime_heartbeat) < _RUNTIME_HEARTBEAT_INTERVAL:
        return
    update_inastate("runtime_heartbeat", datetime.now(timezone.utc).isoformat())
    _last_runtime_heartbeat = now


def increment_inastate_metric(metric: str, amount: int = 1):
    with _inastate_lock():
        state = _load_inastate_state()
        metrics = state.get("metrics", {})
        metrics[metric] = int(metrics.get(metric, 0)) + int(amount)
        state["metrics"] = metrics
        _atomic_write_inastate(state)


def set_inastate_metric(metric: str, value):
    with _inastate_lock():
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


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=True) + "\n")
        return True
    except Exception as exc:
        log_to_statusbox(f"[Manager] Failed to append JSONL at {path}: {exc}")
        return False


def _sanitize_need_alias(value: Any, *, prefix: str = "sym_need_") -> str:
    raw = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    cleaned = "".join(ch for ch in raw if ("a" <= ch <= "z") or ("0" <= ch <= "9") or ch == "_")
    if not cleaned:
        cleaned = f"{prefix}unnamed"
    if not cleaned.startswith(prefix):
        cleaned = f"{prefix}{cleaned}"
    cleaned = cleaned[:64]
    return cleaned if len(cleaned) > len(prefix) else f"{prefix}{uuid.uuid4().hex[:6]}"


def _resolve_need_symbol_aliases(
    canonical_variables: List[str],
    *,
    limits: Dict[str, Any],
    now_iso: str,
) -> Dict[str, str]:
    if not limits.get("need_aliases_enabled", True):
        return {}

    prefix = str(limits.get("alias_prefix") or "sym_need_")
    registry_raw = get_inastate("need_symbol_aliases")
    registry = registry_raw if isinstance(registry_raw, dict) else {}
    changed = False

    # Optional alias override channel so Ina/operator can rename tags over time.
    pending_raw = get_inastate("need_alias_requests")
    pending = pending_raw if isinstance(pending_raw, dict) else {}
    if pending:
        used_aliases = {
            str(entry.get("alias"))
            for entry in registry.values()
            if isinstance(entry, dict) and entry.get("alias")
        }
        for canonical, requested in pending.items():
            canonical_name = str(canonical or "").strip()
            if not canonical_name:
                continue
            alias = _sanitize_need_alias(requested, prefix=prefix)
            if alias in used_aliases:
                alias = _sanitize_need_alias(f"{alias}_{uuid.uuid4().hex[:4]}", prefix=prefix)
            used_aliases.add(alias)
            entry = registry.get(canonical_name) if isinstance(registry.get(canonical_name), dict) else {}
            entry["alias"] = alias
            entry["canonical_variable"] = canonical_name
            entry["last_updated"] = now_iso
            entry.setdefault("created_at", now_iso)
            entry["source"] = "request"
            registry[canonical_name] = entry
            changed = True
        update_inastate("need_alias_requests", None)

    used_aliases = {
        str(entry.get("alias"))
        for entry in registry.values()
        if isinstance(entry, dict) and entry.get("alias")
    }

    for canonical in canonical_variables:
        canonical_name = str(canonical or "").strip()
        if not canonical_name:
            continue
        entry = registry.get(canonical_name) if isinstance(registry.get(canonical_name), dict) else {}
        alias = entry.get("alias")
        if not alias and limits.get("auto_generate_need_aliases", True):
            seed = canonical_name.replace("_", "")[:10] or "need"
            alias = _sanitize_need_alias(f"{prefix}{seed}_{uuid.uuid4().hex[:4]}", prefix=prefix)
            while alias in used_aliases:
                alias = _sanitize_need_alias(f"{prefix}{seed}_{uuid.uuid4().hex[:4]}", prefix=prefix)
            used_aliases.add(alias)
            entry["alias"] = alias
            entry["canonical_variable"] = canonical_name
            entry["created_at"] = now_iso
            entry["last_updated"] = now_iso
            entry["source"] = "auto_generated"
            changed = True
        elif alias:
            normalized_alias = _sanitize_need_alias(alias, prefix=prefix)
            if normalized_alias != alias:
                alias = normalized_alias
                entry["alias"] = alias
                entry["last_updated"] = now_iso
                changed = True
            entry.setdefault("canonical_variable", canonical_name)
            entry.setdefault("created_at", now_iso)
            registry[canonical_name] = entry
        if entry and canonical_name not in registry:
            registry[canonical_name] = entry

    if changed:
        update_inastate("need_symbol_aliases", registry)

    resolved: Dict[str, str] = {}
    for canonical in canonical_variables:
        entry = registry.get(canonical)
        if isinstance(entry, dict) and isinstance(entry.get("alias"), str):
            resolved[canonical] = entry["alias"]
    return resolved


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


def _resource_pressure_value(level: Any) -> float:
    normalized = str(level or "").strip().lower()
    if normalized == "hard":
        return 1.0
    if normalized == "soft":
        return 0.72
    if normalized == "stable":
        return 0.25
    return 0.0


def _extract_resource_context(resource_vitals: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload = resource_vitals if isinstance(resource_vitals, dict) else (get_inastate("resource_vitals") or {})
    if not isinstance(payload, dict):
        payload = {}

    trend = payload.get("trend") if isinstance(payload.get("trend"), dict) else {}
    short = trend.get("short") if isinstance(trend.get("short"), dict) else {}
    long = trend.get("long") if isinstance(trend.get("long"), dict) else {}
    top_modules = payload.get("top_modules") if isinstance(payload.get("top_modules"), list) else []
    top_modules = [item for item in top_modules[:3] if isinstance(item, dict)]

    pressure_level = str(payload.get("pressure_level") or "").strip().lower()
    current_pressure = _resource_pressure_value(pressure_level)
    short_direction = str(short.get("direction") or "unknown").strip().lower()
    long_direction = str(long.get("direction") or "unknown").strip().lower()
    short_delta_gb = max(0.0, _coerce_float(short.get("ram_delta_bytes"), 0.0) / (1024.0 ** 3))
    long_delta_gb = max(0.0, _coerce_float(long.get("ram_delta_bytes"), 0.0) / (1024.0 ** 3))
    system_ram_delta = _coerce_float(long.get("system_ram_delta_percent"), 0.0)

    direction_pressure = 0.0
    if short_direction == "rising":
        direction_pressure += 0.55
    elif short_direction == "stable":
        direction_pressure += 0.15
    elif short_direction not in {"falling", "unknown"}:
        direction_pressure += 0.05
    if long_direction == "rising":
        direction_pressure += 0.35
    elif long_direction == "stable":
        direction_pressure += 0.10
    elif long_direction not in {"falling", "unknown"}:
        direction_pressure += 0.05

    growth_pressure = _clamp01((0.65 * (short_delta_gb / 4.0)) + (0.35 * (long_delta_gb / 8.0)), default=0.0)
    system_pressure = _clamp01(max(0.0, system_ram_delta) / 12.0, default=0.0)
    trend_pressure = _clamp01(
        (0.45 * current_pressure) + (0.35 * direction_pressure) + (0.15 * growth_pressure) + (0.05 * system_pressure),
        default=0.0,
    )

    largest = top_modules[0] if top_modules else {}
    trend_summary = str(trend.get("summary") or "").strip()
    summary = str(payload.get("summary") or "").strip()
    optimization_hint = str(payload.get("optimization_hint") or "").strip()

    return {
        "available": bool(payload),
        "pressure_level": pressure_level or "unknown",
        "current_pressure": round(current_pressure, 3),
        "trend_pressure": round(trend_pressure, 3),
        "short_direction": short_direction,
        "long_direction": long_direction,
        "short_ram_delta_gb": round(short_delta_gb, 3),
        "long_ram_delta_gb": round(long_delta_gb, 3),
        "system_ram_delta_percent": round(system_ram_delta, 1),
        "summary": summary,
        "trend_summary": trend_summary,
        "optimization_hint": optimization_hint,
        "top_modules": top_modules,
        "largest_module": str(largest.get("name") or ""),
        "largest_module_ram": str(largest.get("ram_human") or ""),
        "samples": int(_coerce_float(trend.get("samples"), 0.0)),
    }


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
        guard = _memory_guard_limits()
        max_memory_percent = guard.get("ram_hard_percent", 50.0)
        max_swap_percent = guard.get("swap_hard_percent", 0.0)
        min_available_gb = guard.get("min_available_gb", 0.0)
        if not guard.get("enabled", True):
            max_memory_percent = 50.0
            max_swap_percent = 0.0
            min_available_gb = 0.0
        config = DeepRecallConfig(
            chunk_size=4,
            burst_chunk_size=1,
            burst_cooldown_sec=45.0,
            burst_collect_garbage=True,
            state_path=str(_DEEP_RECALL_STATE_PATH),
            min_energy=0.35,
            max_memory_percent=max_memory_percent,
            max_swap_percent=max_swap_percent,
            min_available_gb=min_available_gb,
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
    if not _is_process_running("ina_client.py"):
        safe_popen(["python", "ina_client.py", "--daemon"])
    if not _is_process_running("audio_listener.py"):
        safe_popen(["python", "audio_listener.py"])
    if not _is_process_running("vision_window.py"):
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


def paint_check():
    global _last_paint_launch
    if get_inastate("paint_window_open"):
        return

    now = time.time()
    if (now - _last_paint_launch) < _PAINT_COOLDOWN:
        return

    if get_inastate("paint_request"):
        update_inastate("paint_request", False)
        _last_paint_launch = now
        safe_popen(["python", "paint_window.py"])
        update_inastate(
            "last_paint_trigger",
            {
                "timestamp": datetime.fromtimestamp(now, timezone.utc).isoformat(),
                "reason": "request",
            },
        )
        log_to_statusbox("[Manager] Paint window opened (request).")
        return

    if get_inastate("dreaming") or get_inastate("meditating"):
        return

    snapshot = get_inastate("emotion_snapshot") or {}
    emo = snapshot.get("values") or snapshot
    curiosity = float(emo.get("curiosity", 0.0) or 0.0)
    joy = float(emo.get("joy", 0.0) or 0.0)
    stress = float(emo.get("stress", 0.0) or 0.0)
    boredom = float(get_inastate("emotion_boredom") or 0.0)

    playfulness = get_inastate("emotion_playfulness_level")
    if playfulness is None:
        playfulness = (get_inastate("emotion_playfulness_state") or {}).get("value", 0.0)
    playfulness = float(playfulness or 0.0)

    creative_trigger = (
        (curiosity > 0.55 and stress < 0.35 and (joy > 0.2 or playfulness > 0.08))
        or (playfulness > 0.2 and stress < 0.25)
        or (boredom > 0.7 and curiosity > 0.4)
    )

    if not creative_trigger:
        return

    _last_paint_launch = now
    safe_popen(["python", "paint_window.py"])
    update_inastate(
        "last_paint_trigger",
        {
            "timestamp": datetime.fromtimestamp(now, timezone.utc).isoformat(),
            "reason": "creative_urge",
            "drivers": {
                "curiosity": round(curiosity, 3),
                "joy": round(joy, 3),
                "playfulness": round(playfulness, 3),
                "stress": round(stress, 3),
                "boredom": round(boredom, 3),
            },
        },
    )
    log_to_statusbox(
        "[Manager] Creative urge triggered paint "
        f"(curiosity={curiosity:.2f}, joy={joy:.2f}, play={playfulness:.2f}, stress={stress:.2f})."
    )


def _get_last_self_read_source() -> Optional[str]:
    last = get_inastate("last_self_read_source")
    if isinstance(last, dict):
        return last.get("source")
    if isinstance(last, str):
        return last
    return None


def _pick_self_read_source(meta_arbitration: Optional[Dict[str, Any]] = None) -> Optional[str]:
    state = get_inastate("self_read_exploration_options") or {}
    options = state.get("options") if isinstance(state, dict) else None
    if not isinstance(options, list) or not options:
        return None

    arbitration = meta_arbitration if isinstance(meta_arbitration, dict) else (get_inastate("meta_arbitration") or {})
    allowed_signals: List[str] = []
    if isinstance(arbitration, dict):
        raw_allowed = arbitration.get("allowed_signals")
        if isinstance(raw_allowed, list):
            allowed_signals = [str(item) for item in raw_allowed if item]
    if allowed_signals and "self_read" not in set(allowed_signals):
        return None

    top_signal = str(arbitration.get("top_signal") or "") if isinstance(arbitration, dict) else ""
    discomfort = _clamp01(arbitration.get("discomfort"), default=0.0) if isinstance(arbitration, dict) else 0.0
    allowed_set = set(allowed_signals)

    source_choices = _load_self_read_source_choices()
    last_source = _get_last_self_read_source()
    candidates = []
    total_weight = 0.0

    for option in options:
        if not isinstance(option, dict):
            continue
        source = option.get("source")
        if not source or not source_choices.get(source, False):
            continue
        try:
            weight = float(option.get("invitation", 0.0))
        except (TypeError, ValueError):
            continue
        if weight <= 0.0:
            continue

        # Arbitration-informed source shaping: do not force an outcome,
        # but gently bias source choice toward the current narrowed lane.
        if top_signal == "rest":
            if source == "music":
                weight *= 1.25
            elif source == "books":
                weight *= 1.1
            elif source == "code":
                weight *= 0.8
            elif source == "venv":
                weight *= 0.75
        elif top_signal in {"stability", "self_read"}:
            if source in {"code", "books"}:
                weight *= 1.2
            elif source == "venv":
                weight *= 1.08

        if discomfort >= 0.7 and source in {"code", "books", "venv"}:
            weight *= 1.0 + (0.16 * discomfort)
        if "stability" in allowed_set and source in {"code", "books"}:
            weight *= 1.08
        if "rest" in allowed_set and source == "music":
            weight *= 1.12

        if last_source and source == last_source:
            weight *= 0.45
        candidates.append((source, weight))
        total_weight += weight

    if not candidates or total_weight <= 0.0:
        return None

    roll = random.random() * total_weight
    for source, weight in candidates:
        roll -= weight
        if roll <= 0:
            return source
    return candidates[-1][0]


def _maybe_self_read():
    """
    Launch self-reading when curiosity spikes, clarity drops (confused), or
    familiarity is high enough to want to revisit known files.
    """
    global _last_self_read_launch, _last_self_read_hold_log
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

    arbitration = get_inastate("meta_arbitration") or {}
    allowed_signals = arbitration.get("allowed_signals") if isinstance(arbitration, dict) else []
    if isinstance(allowed_signals, list) and allowed_signals and "self_read" not in {str(item) for item in allowed_signals if item}:
        if (now - _last_self_read_hold_log) >= _SELF_READ_HOLD_LOG_COOLDOWN:
            top_signal = arbitration.get("top_signal") if isinstance(arbitration, dict) else None
            log_to_statusbox(
                f"[Manager] Self-read urge noted but held by arbitration"
                f" (top signal: {top_signal or 'unknown'})."
            )
            _last_self_read_hold_log = now
        update_inastate(
            "last_self_read_deferral",
            {
                "timestamp": datetime.fromtimestamp(now, timezone.utc).isoformat(),
                "reason": reason,
                "deferred_by": "meta_arbitration",
                "top_signal": arbitration.get("top_signal") if isinstance(arbitration, dict) else None,
                "allowed_signals": allowed_signals if isinstance(allowed_signals, list) else [],
            },
        )
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
    source_pick = _pick_self_read_source(meta_arbitration=arbitration if isinstance(arbitration, dict) else None)
    popen_kwargs = {}
    if source_pick:
        trigger["source_pick"] = source_pick
    if isinstance(arbitration, dict):
        trigger["arbitration"] = {
            "status": arbitration.get("status"),
            "top_signal": arbitration.get("top_signal"),
            "allowed_signals": arbitration.get("allowed_signals") if isinstance(arbitration.get("allowed_signals"), list) else [],
            "discomfort": arbitration.get("discomfort"),
        }
    if not source_pick and isinstance(arbitration, dict):
        trigger["source_pick_blocked_by_arbitration"] = bool(
            isinstance(allowed_signals, list) and allowed_signals and "self_read" not in {str(item) for item in allowed_signals if item}
        )
    if source_pick:
        update_inastate(
            "last_self_read_source",
            {
                "source": source_pick,
                "timestamp": datetime.fromtimestamp(now, timezone.utc).isoformat(),
                "reason": reason,
            },
        )
        env = dict(os.environ)
        env["SELF_READ_SOURCE"] = source_pick
        popen_kwargs["env"] = env
        log_to_statusbox(f"[Manager] Self-read source pick: {source_pick}.")
    update_inastate("last_self_read_trigger", trigger)
    log_to_statusbox(f"[Manager] Self-read triggered ({reason}).")
    safe_popen(["python", "raw_file_manager.py"], **popen_kwargs)

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


def _age_seconds(raw: Any, *, now: Optional[float] = None) -> Optional[float]:
    if now is None:
        now = time.time()
    if raw is None:
        return None
    if isinstance(raw, dict):
        raw = raw.get("timestamp")
    if isinstance(raw, (int, float)):
        try:
            return max(0.0, now - float(raw))
        except Exception:
            return None
    if isinstance(raw, str):
        try:
            ts = datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
            return max(0.0, now - ts)
        except Exception:
            return None
    return None


def _extract_level_from_signal_payload(payload: Any, *, key: str = "level", default: float = 0.0) -> float:
    if isinstance(payload, dict):
        return _clamp01(payload.get(key), default=default)
    return _clamp01(payload, default=default)


def _resolve_meta_adjusted_level(payload: Any, default: float = 0.0) -> float:
    if not isinstance(payload, dict):
        return _clamp01(payload, default=default)
    base = _extract_level_from_signal_payload(payload, default=default)
    adjusted = payload.get("adjusted_level")
    if adjusted is None:
        return base
    return _clamp01(adjusted, default=base)


def _best_self_read_invitation(payload: Any) -> Dict[str, Any]:
    best_level = 0.0
    best_source: Optional[str] = None
    if isinstance(payload, dict):
        options = payload.get("options")
        if isinstance(options, list):
            for option in options:
                if not isinstance(option, dict):
                    continue
                level = _clamp01(option.get("invitation"), default=0.0)
                if level <= best_level:
                    continue
                best_level = level
                source = option.get("source")
                if source is not None:
                    best_source = str(source)
    return {"level": best_level, "source": best_source}


def _update_meta_arbitration_signal(memory_guard: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Build a meta-signal over cross-system urges so unresolved conflict can
    escalate organically instead of forcing a fixed winner policy.
    """
    global _last_meta_arbitration_log

    now = time.time()
    now_iso = datetime.fromtimestamp(now, timezone.utc).isoformat()
    limits = _meta_arbitration_limits()

    if not limits.get("enabled", True):
        disabled_state = {
            "updated_at": now_iso,
            "enabled": False,
            "status": "disabled",
            "signals": [],
            "drivers": {},
            "allowed_signals": [],
            "adjusted_levels": {},
            "indecision_seconds": 0.0,
            "indecision_cost": 0.0,
            "discomfort": 0.0,
        }
        update_inastate("meta_arbitration", disabled_state)
        return disabled_state

    voice_state = get_inastate("urge_to_voice") or {}
    type_state = get_inastate("urge_to_type") or {}
    move_state = get_inastate("urge_to_move") or {}
    stability_state = get_inastate("urge_to_seek_stability") or {}
    self_read_state = get_inastate("self_read_exploration_options") or {}
    self_read_best = _best_self_read_invitation(self_read_state)

    snapshot = get_inastate("emotion_snapshot") or {}
    values = snapshot.get("values") if isinstance(snapshot, dict) else {}
    if isinstance(values, dict) and values:
        snapshot = values
    if not isinstance(snapshot, dict):
        snapshot = {}

    energy = _clamp01(get_inastate("current_energy") or 0.5, default=0.5)
    sleep_pressure = _clamp01(get_inastate("sleep_pressure") or 0.0, default=0.0)
    clarity = _clamp01(snapshot.get("clarity"), default=0.5)
    fuzziness = _clamp01(snapshot.get("fuzziness", snapshot.get("fuzz_level", 0.0)), default=0.5)
    alignment = _clamp01(snapshot.get("alignment"), default=0.5)
    drift = _clamp01(snapshot.get("symbolic_drift") or get_inastate("symbolic_drift") or 0.0, default=0.0)

    guard_state = memory_guard if isinstance(memory_guard, dict) else (get_inastate("memory_guard") or {})
    guard_level = ""
    if isinstance(guard_state, dict):
        guard_level = str(guard_state.get("level") or "").strip().lower()
    memory_pressure = 0.0
    if guard_level == "soft":
        memory_pressure = 0.65
    elif guard_level == "hard":
        memory_pressure = 1.0
    elif guard_level in {"unknown", "disabled"}:
        memory_pressure = 0.2
    if isinstance(guard_state, dict):
        ram_percent = _clamp01(_coerce_float(guard_state.get("ram_percent"), 0.0) / 100.0, default=0.0)
        memory_pressure = max(memory_pressure, ram_percent)
    resource_context = _extract_resource_context()
    resource_current_pressure = _clamp01(resource_context.get("current_pressure"), default=0.0)
    resource_trend_pressure = _clamp01(resource_context.get("trend_pressure"), default=0.0)
    memory_pressure = max(memory_pressure, resource_current_pressure)
    resource_trend_summary = resource_context.get("trend_summary") or resource_context.get("summary") or ""
    resource_pressure_level = str(resource_context.get("pressure_level") or "unknown")

    coherence_risk = _clamp01(
        (0.45 * (1.0 - clarity)) + (0.3 * fuzziness) + (0.15 * drift) + (0.1 * (1.0 - alignment)),
        default=0.0,
    )
    energy_instability = _clamp01((0.6 * (1.0 - energy)) + (0.4 * sleep_pressure), default=0.0)
    global_load = _clamp01(
        (0.32 * memory_pressure) + (0.18 * resource_trend_pressure) + (0.30 * coherence_risk) + (0.20 * energy_instability),
        default=0.0,
    )

    rest_drive = _clamp01((0.7 * sleep_pressure) + (0.3 * (1.0 - energy)), default=0.0)
    candidates = [
        {"id": "move", "source": "urge_to_move", "level": _extract_level_from_signal_payload(move_state)},
        {"id": "voice", "source": "urge_to_voice", "level": _extract_level_from_signal_payload(voice_state)},
        {"id": "type", "source": "urge_to_type", "level": _extract_level_from_signal_payload(type_state)},
        {"id": "stability", "source": "urge_to_seek_stability", "level": _extract_level_from_signal_payload(stability_state)},
        {"id": "self_read", "source": "self_read_exploration_options", "level": self_read_best["level"]},
        {"id": "rest", "source": "derived:energy_stability", "level": rest_drive},
    ]
    candidates.sort(key=lambda item: item["level"], reverse=True)

    activation_threshold = limits["activation_threshold"]
    active = [item for item in candidates if item["level"] >= activation_threshold]
    active_ids = {item["id"] for item in active}
    active_count = len(active)
    active_mean = (
        _clamp01(sum(item["level"] for item in active) / active_count, default=0.0)
        if active_count
        else 0.0
    )

    top = candidates[0] if candidates else None
    runner = candidates[1] if len(candidates) > 1 else None
    top_level = top["level"] if top else 0.0
    runner_level = runner["level"] if runner else 0.0
    winner_margin = max(0.0, top_level - runner_level)

    tie_pressure = 0.0
    if (
        active_count >= 2
        and top_level >= limits["conflict_min_level"]
        and winner_margin < limits["conflict_margin"]
    ):
        tie_pressure = _clamp01(
            1.0 - (winner_margin / max(limits["conflict_margin"], 1e-6)),
            default=0.0,
        )
    breadth_pressure = _clamp01((active_count - 1) / max(len(candidates) - 1, 1), default=0.0)
    conflict_score = _clamp01(
        (0.5 * tie_pressure * max(top_level, runner_level))
        + (0.2 * active_mean)
        + (0.15 * breadth_pressure)
        + (0.15 * global_load),
        default=0.0,
    )

    action_ages = [
        _age_seconds(get_inastate("last_expression_time"), now=now),
        _age_seconds(get_inastate("last_motor_intent"), now=now),
        _age_seconds(get_inastate("last_self_read_trigger"), now=now),
    ]
    valid_action_ages = [age for age in action_ages if age is not None]
    recent_action_age = min(valid_action_ages) if valid_action_ages else None
    stall_window = max(1.0, limits["stall_action_window_sec"])
    stall_pressure = (
        1.0 if recent_action_age is None else _clamp01(recent_action_age / stall_window, default=0.0)
    )

    previous = get_inastate("meta_arbitration") or {}
    if not isinstance(previous, dict):
        previous = {}
    prev_indecision = _clamp_positive(previous.get("indecision_seconds"), 0.0)
    age_since_last_meta = _age_seconds(previous.get("updated_at"), now=now)
    dt = max(1.0, min(30.0, age_since_last_meta if age_since_last_meta is not None else 10.0))

    unresolved_conflict = (
        active_count >= 2
        and top_level >= limits["conflict_min_level"]
        and winner_margin < limits["conflict_margin"]
    )
    if unresolved_conflict:
        indecision_seconds = min(limits["indecision_horizon_sec"] * 4.0, prev_indecision + dt)
    else:
        decay = dt * (0.35 + (0.35 if winner_margin >= limits["conflict_margin"] else 0.0))
        indecision_seconds = max(0.0, prev_indecision * limits["resolution_decay"] - decay)
        if conflict_score >= 0.45 and stall_pressure >= 0.85:
            indecision_seconds = min(limits["indecision_horizon_sec"] * 4.0, indecision_seconds + (0.4 * dt))

    indecision_ratio = _clamp01(indecision_seconds / max(5.0, limits["indecision_horizon_sec"]), default=0.0)
    indecision_cost = _clamp01((0.55 * indecision_ratio) + (0.45 * conflict_score), default=0.0)
    discomfort = _clamp01(
        (0.5 * indecision_cost) + (0.3 * global_load) + (0.2 * stall_pressure),
        default=0.0,
    )

    band_high = limits["narrowing_band_high"]
    band_low = limits["narrowing_band_low"]
    narrowing_band = band_high - discomfort * (band_high - band_low)
    narrowing_band = max(band_low, min(band_high, narrowing_band))
    dynamic_floor = _clamp01(
        activation_threshold + (limits["narrowing_gain"] * discomfort),
        default=activation_threshold,
    )
    gate = max(dynamic_floor, top_level - narrowing_band) if top else dynamic_floor
    allowed = [item for item in candidates if item["level"] >= gate]
    if not allowed and top:
        allowed = [top]
    allowed_ids = {item["id"] for item in allowed}

    adjusted_levels: Dict[str, float] = {}
    for item in candidates:
        level = item["level"]
        proximity = _clamp01(level / max(top_level, 1e-6), default=0.0) if top_level > 0 else 0.0
        if item["id"] in allowed_ids:
            adjusted = _clamp01(level + (limits["boost_gain"] * discomfort * proximity), default=level)
        else:
            adjusted = _clamp01(level - (limits["suppression_gain"] * discomfort), default=level)
        adjusted_levels[item["id"]] = round(adjusted, 3)

    signal_rows: List[Dict[str, Any]] = []
    for item in candidates:
        signal_rows.append(
            {
                "id": item["id"],
                "source": item["source"],
                "level": round(item["level"], 3),
                "adjusted_level": adjusted_levels.get(item["id"], round(item["level"], 3)),
                "active": item["id"] in active_ids,
                "allowed": item["id"] in allowed_ids,
            }
        )

    prev_discomfort = _clamp01(previous.get("discomfort"), default=0.0)
    if unresolved_conflict and discomfort > (prev_discomfort + 0.02):
        status = "escalating"
    elif unresolved_conflict:
        status = "conflicted"
    elif discomfort < max(0.0, prev_discomfort - 0.03):
        status = "cooling"
    else:
        status = "resolved"

    world_connected = bool(get_inastate("world_connected", False))
    prediction_state = get_inastate("current_prediction") or {}
    if not isinstance(prediction_state, dict):
        prediction_state = {}
    prediction_vec = prediction_state.get("predicted_vector") or {}
    if not isinstance(prediction_vec, dict):
        prediction_vec = {}
    prediction_confidence = _clamp01(prediction_vec.get("confidence") or 0.0, default=0.0)

    machine_semantics = get_inastate("machine_semantics") or {}
    axes = machine_semantics.get("axes") if isinstance(machine_semantics, dict) else {}
    if not isinstance(axes, dict):
        axes = {}

    def axis_value(axis_id: str, default: float = 0.5) -> float:
        axis = axes.get(axis_id)
        if isinstance(axis, dict):
            return _clamp01(axis.get("value"), default=default)
        return default

    predictive_reliability = axis_value("predictive_reliability", default=0.5)
    controllability = axis_value("controllability", default=0.5)
    signal_integrity = axis_value("signal_integrity", default=0.5)

    checks: List[Dict[str, Any]] = [
        {
            "id": "signal_conflict",
            "value": round(conflict_score, 3),
            "status": "high" if conflict_score >= 0.65 else ("medium" if conflict_score >= 0.45 else "low"),
            "note": "Competition between action channels.",
        },
        {
            "id": "context_clarity",
            "value": round(1.0 - coherence_risk, 3),
            "status": "low" if coherence_risk >= 0.6 else ("medium" if coherence_risk >= 0.45 else "high"),
            "note": "How understandable the current context seems.",
        },
        {
            "id": "prediction_fit",
            "value": round(predictive_reliability, 3),
            "status": "low" if predictive_reliability < 0.45 else ("medium" if predictive_reliability < 0.6 else "high"),
            "note": "Whether prediction context supports a choice.",
        },
        {
            "id": "resource_budget",
            "value": round(1.0 - memory_pressure, 3),
            "status": "low" if memory_pressure >= 0.8 else ("medium" if memory_pressure >= 0.6 else "high"),
            "note": "Memory/compute room for optional actions.",
        },
        {
            "id": "resource_trend",
            "value": (round(1.0 - resource_trend_pressure, 3) if resource_context.get("available") else None),
            "status": (
                "unknown"
                if not resource_context.get("available")
                else ("low" if resource_trend_pressure >= 0.75 else ("medium" if resource_trend_pressure >= 0.5 else "high"))
            ),
            "note": resource_trend_summary or "Whether RAM pressure is stabilizing or worsening over time.",
        },
        {
            "id": "feedback_loop",
            "value": round(1.0 - stall_pressure, 3),
            "status": "low" if stall_pressure >= 0.85 else ("medium" if stall_pressure >= 0.6 else "high"),
            "note": "Recent action feedback available to break indecision.",
        },
        {
            "id": "controllability",
            "value": round(controllability, 3),
            "status": "low" if controllability < 0.45 else ("medium" if controllability < 0.6 else "high"),
            "note": "Sense of agency in the current situation.",
        },
        {
            "id": "signal_integrity",
            "value": round(signal_integrity, 3),
            "status": "low" if signal_integrity < 0.45 else ("medium" if signal_integrity < 0.6 else "high"),
            "note": "Clarity vs. noise in incoming signals.",
        },
    ]

    missing_inputs: List[Dict[str, Any]] = []

    def add_missing(
        missing_id: str,
        severity: str,
        reason: str,
        *,
        canonical_variable: str,
        suggested_probe: str,
        where_to_obtain: str,
        provider_modules: List[str],
        expected_output: str,
    ) -> None:
        for item in missing_inputs:
            if item.get("id") == missing_id:
                return
        canonical_meta = _NEED_CANONICAL_VARIABLES.get(canonical_variable, {})
        providers = [str(name) for name in provider_modules if isinstance(name, str) and name]
        providers = list(dict.fromkeys(providers))
        missing_inputs.append(
            {
                "id": missing_id,
                "severity": severity,
                "canonical_variable": canonical_variable,
                "canonical_description": canonical_meta.get("description"),
                "reason": reason,
                "suggested_probe": suggested_probe,
                "where_to_obtain": where_to_obtain,
                "provider_modules": providers,
                "source_location": where_to_obtain,
                "expected_output": expected_output,
                # Backward-compatible field so existing consumers still render a probe line.
                "probe": suggested_probe,
            }
        )

    if unresolved_conflict and winner_margin <= (limits["conflict_margin"] * 0.5):
        add_missing(
            "tie_break_signal",
            "high" if conflict_score >= 0.7 else "medium",
            "Top options are too close together to resolve naturally.",
            canonical_variable="urgency_signal",
            suggested_probe="compare delayed-cost delta across candidate actions",
            where_to_obtain="prediction rollouts + machine_semantics.attention_value",
            provider_modules=["predictive_layer.py", "logic_engine.py"],
            expected_output="scalar [0..1] where higher means delay is costly",
        )
    if coherence_risk >= 0.58:
        add_missing(
            "context_clarity",
            "high" if coherence_risk >= 0.72 else "medium",
            "Context model is noisy (clarity/fuzziness mismatch).",
            canonical_variable="context_scope",
            suggested_probe="measure context coverage across world, memory, and active goal channels",
            where_to_obtain="world_heartbeat + machine_semantics + memory_graph context links",
            provider_modules=["logic_engine.py", "memory_graph.py", "predictive_layer.py"],
            expected_output="scalar [0..1] where higher means context is sufficiently bounded",
        )
    if predictive_reliability < 0.45 or prediction_confidence < 0.35:
        add_missing(
            "reversibility_estimate",
            "medium",
            "Outcome reversibility is under-modeled for this decision frame.",
            canonical_variable="reversibility_estimate",
            suggested_probe="simulate action on shadow state",
            where_to_obtain="counterfactual rollout in predictive + logic shadow simulation",
            provider_modules=["logic_engine.py", "predictive_layer.py"],
            expected_output="scalar [-1..1] where +1 is reversible",
        )
    if not world_connected and (top_level >= activation_threshold or unresolved_conflict):
        add_missing(
            "world_feedback",
            "medium",
            "World channel is disconnected during active arbitration.",
            canonical_variable="context_scope",
            suggested_probe="reacquire world heartbeat and verify sensor freshness",
            where_to_obtain="world_server heartbeat + touch/motor feedback timestamps",
            provider_modules=["ina_client.py", "motor_controls.py", "model_manager.py"],
            expected_output="bool freshness + scalar [0..1] context confidence",
        )
    if memory_pressure >= 0.8:
        add_missing(
            "resource_budget",
            "high" if memory_pressure >= 0.9 else "medium",
            "Memory pressure is constraining option execution.",
            canonical_variable="context_scope",
            suggested_probe="estimate executable option budget under current guard limits",
            where_to_obtain="memory_guard + module pressure queue state",
            provider_modules=["model_manager.py", "fragment_limits.py"],
            expected_output="scalar [0..1] where higher means enough budget to branch",
        )
    if resource_context.get("available") and (
        resource_trend_pressure >= 0.68
        or (
            memory_pressure >= 0.6
            and (
                resource_context.get("short_direction") == "rising"
                or resource_context.get("long_direction") == "rising"
            )
        )
    ):
        add_missing(
            "resource_trend",
            "high" if resource_trend_pressure >= 0.82 else "medium",
            resource_trend_summary or "Resource pressure is trending upward over time.",
            canonical_variable="context_scope",
            suggested_probe="profile the largest live RAM holders and defer optional branching until the trend cools",
            where_to_obtain="resource_vitals + resource_vitals_history + module RAM table",
            provider_modules=["GUI.py", "model_manager.py"],
            expected_output="ranked module RAM list + scalar [0..1] headroom estimate under the current trend",
        )
    if stall_pressure >= 0.85:
        add_missing(
            "action_feedback",
            "medium",
            "No recent action feedback to resolve competing urges.",
            canonical_variable="reversibility_estimate",
            suggested_probe="run low-risk probe action and score reversibility from feedback",
            where_to_obtain="motor intent feedback + expression outcomes over short horizon",
            provider_modules=["model_manager.py", "predictive_layer.py", "early_comm.py"],
            expected_output="scalar [-1..1] where +1 confirms safe reversible exploration",
        )
    if active_count == 0 or (top_level < activation_threshold and discomfort >= 0.55):
        add_missing(
            "salience_signal",
            "medium",
            "No signal is confidently above activation while pressure remains elevated.",
            canonical_variable="urgency_signal",
            suggested_probe="derive urgency from unresolved-cost slope over recent loops",
            where_to_obtain="meta_arbitration indecision trajectory + conflict trend",
            provider_modules=["model_manager.py", "emotion_engine.py"],
            expected_output="scalar [0..1] where higher indicates decisive urgency",
        )
    if "self_read" not in allowed_ids and coherence_risk >= 0.58 and active_count >= 2:
        add_missing(
            "inquiry_channel",
            "medium",
            "Conflict may require context gathering but self-read is currently suppressed.",
            canonical_variable="context_scope",
            suggested_probe="open a constrained self-read lane and test context reduction",
            where_to_obtain="self_read_exploration_options + post-read coherence delta",
            provider_modules=["model_manager.py", "raw_file_manager.py"],
            expected_output="scalar [0..1] context gain after inquiry",
        )

    canonical_missing = sorted(
        {
            str(item.get("canonical_variable"))
            for item in missing_inputs
            if isinstance(item, dict) and item.get("canonical_variable")
        }
    )
    alias_map = _resolve_need_symbol_aliases(canonical_missing, limits=limits, now_iso=now_iso)
    for item in missing_inputs:
        canonical = str(item.get("canonical_variable") or "")
        alias = alias_map.get(canonical)
        if alias:
            item["symbolic_tag"] = alias
            item["symbolic_phrase"] = f"{alias}::{canonical}"

    self_diagnosis_lines: List[str] = []
    if missing_inputs:
        for item in missing_inputs[:6]:
            symbol = item.get("symbolic_tag")
            canonical = item.get("canonical_variable")
            prefix = f"{symbol} ({canonical})" if symbol else str(canonical or item.get("id"))
            self_diagnosis_lines.append(
                f"{prefix}: {item.get('reason')} Probe: {item.get('suggested_probe')}"
            )
    else:
        self_diagnosis_lines.append(
            "No explicit missing-input class detected; conflict currently appears intensity-driven."
        )

    recommended_actions: List[Dict[str, Any]] = []
    for item in missing_inputs:
        recommended_actions.append(
            {
                "symbolic_tag": item.get("symbolic_tag"),
                "canonical_variable": item.get("canonical_variable"),
                "suggested_probe": item.get("suggested_probe"),
                "where_to_obtain": item.get("where_to_obtain"),
                "provider_modules": item.get("provider_modules"),
                "expected_output": item.get("expected_output"),
            }
        )

    recommended_tools = []
    for action in recommended_actions:
        probe = action.get("suggested_probe")
        providers = action.get("provider_modules") if isinstance(action.get("provider_modules"), list) else []
        provider_text = ", ".join(str(name) for name in providers if name)
        if probe and provider_text:
            recommended_tools.append(f"{probe} via {provider_text}")
        elif probe:
            recommended_tools.append(str(probe))
    recommended_tools = list(dict.fromkeys(recommended_tools))

    symbolic_needs = [item.get("symbolic_tag") for item in missing_inputs if item.get("symbolic_tag")]

    diagnostics_payload = {
        "missing_inputs": missing_inputs,
        "checks": checks,
        "self_diagnosis": self_diagnosis_lines,
        "recommended_tools": recommended_tools,
        "recommended_actions": recommended_actions,
        "canonical_seed_variables": list(_NEED_CANONICAL_VARIABLES.keys()),
        "symbolic_needs": symbolic_needs,
        "canonical_to_symbolic_map": alias_map,
        "resource_vitals": {
            "available": resource_context.get("available", False),
            "pressure_level": resource_pressure_level,
            "trend_pressure": round(resource_trend_pressure, 3),
            "summary": resource_context.get("summary"),
            "trend_summary": resource_trend_summary,
            "optimization_hint": resource_context.get("optimization_hint"),
            "top_modules": resource_context.get("top_modules"),
        },
    }

    advocacy_payload = {
        "updated_at": now_iso,
        "mode": "symbolic_need_report",
        "canonical_variables": canonical_missing,
        "symbolic_tags": symbolic_needs,
        "canonical_to_symbolic_map": alias_map,
        "entries": recommended_actions,
        "self_diagnosis": self_diagnosis_lines,
        "language_bridge": "symbolic -> canonical -> human",
        "alias_update_channel": "need_alias_requests",
    }
    update_inastate("decision_need_advocacy", advocacy_payload)

    payload = {
        "updated_at": now_iso,
        "enabled": True,
        "status": status,
        "top_signal": top["id"] if top else None,
        "runner_up_signal": runner["id"] if runner else None,
        "winner_margin": round(winner_margin, 3),
        "active_count": active_count,
        "conflict_score": round(conflict_score, 3),
        "indecision_seconds": round(indecision_seconds, 2),
        "indecision_cost": round(indecision_cost, 3),
        "discomfort": round(discomfort, 3),
        "dynamic_floor": round(gate, 3),
        "narrowing_band": round(narrowing_band, 3),
        "allowed_signals": [item["id"] for item in allowed],
        "adjusted_levels": adjusted_levels,
        "signals": signal_rows,
        "drivers": {
            "memory_pressure": round(memory_pressure, 3),
            "resource_trend_pressure": round(resource_trend_pressure, 3),
            "resource_pressure_level": resource_pressure_level,
            "resource_short_direction": resource_context.get("short_direction"),
            "resource_long_direction": resource_context.get("long_direction"),
            "resource_optimization_hint": resource_context.get("optimization_hint"),
            "coherence_risk": round(coherence_risk, 3),
            "energy_instability": round(energy_instability, 3),
            "global_load": round(global_load, 3),
            "stall_pressure": round(stall_pressure, 3),
            "recent_action_age_sec": round(recent_action_age, 2) if recent_action_age is not None else None,
            "best_self_read_source": self_read_best.get("source"),
            "world_connected": world_connected,
            "prediction_confidence": round(prediction_confidence, 3),
            "predictive_reliability": round(predictive_reliability, 3),
        },
        "diagnostics": diagnostics_payload,
    }

    panic_prev = get_inastate("decision_panic") or {}
    if not isinstance(panic_prev, dict):
        panic_prev = {}
    prev_panic_active = bool(panic_prev.get("active"))
    prev_panic_arbitration = panic_prev.get("arbitration") if isinstance(panic_prev.get("arbitration"), dict) else {}
    prev_panic_discomfort = _clamp01(prev_panic_arbitration.get("discomfort"), default=0.0)
    prev_panic_indecision = _clamp_positive(prev_panic_arbitration.get("indecision_seconds"), 0.0)
    last_panic_log_age = _age_seconds(panic_prev.get("last_logged_at"), now=now)
    panic_active = bool(
        limits.get("panic_enabled", True)
        and unresolved_conflict
        and discomfort >= limits["panic_discomfort_threshold"]
        and conflict_score >= limits["panic_conflict_threshold"]
        and indecision_seconds >= limits["panic_indecision_sec"]
        and (stall_pressure >= 0.65 or bool(missing_inputs))
    )
    panic_event: Optional[str] = None
    should_log_panic = False
    if panic_active and not prev_panic_active:
        panic_event = "panic_entered"
        should_log_panic = True
    elif panic_active:
        escalating = discomfort >= (prev_panic_discomfort + 0.015) or indecision_seconds >= (prev_panic_indecision + 15.0)
        if (last_panic_log_age is None or last_panic_log_age >= limits["panic_repeat_sec"]) and escalating:
            panic_event = "panic_escalating"
            should_log_panic = True
    elif prev_panic_active:
        panic_event = "panic_resolved"
        should_log_panic = True

    if panic_active or prev_panic_active:
        prior_episode_id = panic_prev.get("episode_id")
        if panic_active and prev_panic_active and prior_episode_id:
            episode_id = str(prior_episode_id)
        elif panic_active:
            episode_id = f"panic_{uuid.uuid4().hex[:10]}"
        else:
            episode_id = str(prior_episode_id or f"panic_{uuid.uuid4().hex[:10]}")

        started_at = panic_prev.get("started_at") if prev_panic_active else now_iso
        panic_payload = {
            "active": panic_active,
            "episode_id": episode_id,
            "status": "active" if panic_active else "resolved",
            "updated_at": now_iso,
            "started_at": started_at,
            "resolved_at": (None if panic_active else now_iso),
            "event": panic_event,
            "arbitration": {
                "status": status,
                "top_signal": top["id"] if top else None,
                "runner_up_signal": runner["id"] if runner else None,
                "winner_margin": round(winner_margin, 3),
                "conflict_score": round(conflict_score, 3),
                "indecision_seconds": round(indecision_seconds, 2),
                "indecision_cost": round(indecision_cost, 3),
                "discomfort": round(discomfort, 3),
                "allowed_signals": [item["id"] for item in allowed],
            },
            "missing_inputs": missing_inputs,
            "diagnostic_checks": checks,
            "self_diagnosis": self_diagnosis_lines,
            "recommended_tools": recommended_tools,
            "last_logged_at": now_iso if should_log_panic else panic_prev.get("last_logged_at"),
            "last_popup_at": panic_prev.get("last_popup_at"),
        }

        popup_triggered = False
        if panic_active and should_log_panic and limits.get("panic_popup_enabled", True):
            last_popup_age = _age_seconds(panic_prev.get("last_popup_at"), now=now)
            if last_popup_age is None or last_popup_age >= limits["panic_popup_cooldown_sec"]:
                popup_triggered = True
                panic_payload["last_popup_at"] = now_iso
        panic_payload["popup_triggered"] = popup_triggered
        update_inastate("decision_panic", panic_payload)

        if should_log_panic:
            log_entry = {
                "timestamp": now_iso,
                "event": panic_event,
                "episode_id": episode_id,
                "arbitration": panic_payload["arbitration"],
                "missing_inputs": missing_inputs,
                "self_diagnosis": self_diagnosis_lines,
            }
            _append_jsonl(_DECISION_PANIC_LOG_PATH, log_entry)
            if panic_event == "panic_resolved":
                log_to_statusbox("[Manager] Decision panic resolved; arbitration pressure cooled.")
            else:
                missing_ids = ", ".join(item.get("id", "") for item in missing_inputs[:4]) or "unknown"
                log_to_statusbox(
                    f"[Manager] Decision panic detected ({panic_event}); likely missing: {missing_ids}."
                )
        if popup_triggered:
            safe_popen(["python", "decision_panic_window.py"])

    update_inastate("meta_arbitration", payload)

    for key, signal_id in (
        ("urge_to_voice", "voice"),
        ("urge_to_type", "type"),
        ("urge_to_move", "move"),
        ("urge_to_seek_stability", "stability"),
    ):
        signal_payload = get_inastate(key)
        if not isinstance(signal_payload, dict):
            continue
        patched = dict(signal_payload)
        patched["adjusted_level"] = adjusted_levels.get(signal_id, round(_extract_level_from_signal_payload(signal_payload), 3))
        patched["arbitration"] = {
            "updated_at": now_iso,
            "status": status,
            "discomfort": round(discomfort, 3),
            "indecision_cost": round(indecision_cost, 3),
            "allowed": signal_id in allowed_ids,
            "top_signal": top["id"] if top else None,
        }
        update_inastate(key, patched)

    if discomfort >= limits["log_discomfort_threshold"]:
        cooldown = limits["log_cooldown_sec"]
        if _last_meta_arbitration_log == 0.0 or (now - _last_meta_arbitration_log) >= cooldown:
            allow_text = ", ".join(item["id"] for item in allowed[:3]) if allowed else "none"
            log_to_statusbox(
                f"[Manager] Arbitration pressure {discomfort:.2f} "
                f"(cost {indecision_cost:.2f}) - narrowing options to {allow_text}."
            )
            _last_meta_arbitration_log = now

    return payload


def _motor_ground_sense_snapshot() -> Dict[str, Any]:
    touch_feedback = get_inastate("touch_feedback") or {}
    touch_world = get_inastate("touch_world") or {}
    motor_feedback = get_inastate("motor_feedback") or {}

    grounded_signals: List[bool] = []
    support_scores: List[float] = []
    sources: List[str] = []
    stance: Optional[str] = None
    contact_count = 0

    if isinstance(touch_feedback, dict) and touch_feedback:
        sources.append("touch_feedback")
        grounded_signals.append(bool(touch_feedback.get("grounded")))
        support_scores.append(_clamp01(touch_feedback.get("surface_solidity"), default=0.0))
        support_scores.append(_clamp01(touch_feedback.get("foot_pressure"), default=0.0))
        stance_raw = touch_feedback.get("stance")
        if isinstance(stance_raw, str) and stance_raw.strip():
            stance = stance_raw.strip()
        contacts = touch_feedback.get("contacts")
        if isinstance(contacts, list):
            contact_count = max(contact_count, len(contacts))

    if isinstance(touch_world, dict) and touch_world:
        sources.append("touch_world")
        grounded_signals.append(bool(touch_world.get("grounded")))
        surface = touch_world.get("ground_surface")
        if isinstance(surface, dict):
            support_scores.append(_clamp01(surface.get("solidity"), default=0.0))
        contacts = touch_world.get("contacts")
        if isinstance(contacts, list):
            contact_count = max(contact_count, len(contacts))

    if isinstance(motor_feedback, dict) and motor_feedback:
        sources.append("motor_feedback")
        grounded_signals.append(bool(motor_feedback.get("grounded")))
        gravity_load = _clamp01(motor_feedback.get("gravity_load"), default=0.0)
        support_scores.append(gravity_load)

    grounded = any(grounded_signals)
    support_confidence = max(support_scores) if support_scores else (1.0 if grounded else 0.0)

    return {
        "grounded": grounded,
        "surface": "solid" if grounded else "unknown",
        "support_confidence": round(_clamp01(support_confidence, default=0.0), 3),
        "stance": stance,
        "contact_count": int(contact_count),
        "sources": sources,
    }


def _update_motor_control_status(
    *,
    now: float,
    decision: str,
    reason: str,
    world_connected: bool,
    dreaming: bool,
    urge_level: float,
    threshold: float,
    cooldown: float,
    heartbeat_age: Optional[float],
    last_intent_age: Optional[float],
    last_feedback_age: Optional[float],
    intent_issued: bool = False,
) -> None:
    heartbeat_fresh = heartbeat_age is None or heartbeat_age <= 30.0
    standing_by_choice = decision == "stand_still_by_choice"
    payload = {
        "timestamp": datetime.fromtimestamp(now, timezone.utc).isoformat(),
        "source": "model_manager",
        "movement_logic_ready": True,
        "world_connected": bool(world_connected),
        "dreaming": bool(dreaming),
        "intent_pipeline_ready": bool(world_connected and not dreaming and heartbeat_fresh),
        "decision": decision,
        "reason": reason,
        "standing_by_choice": standing_by_choice,
        "intent_issued": bool(intent_issued),
        "urge_to_move": {
            "level": round(_clamp01(urge_level, default=0.0), 3),
            "threshold": round(max(0.0, float(threshold)), 3),
        },
        "cooldown_seconds": round(max(0.0, float(cooldown)), 3),
        "ages_seconds": {
            "world_heartbeat": round(heartbeat_age, 3) if heartbeat_age is not None else None,
            "last_motor_intent": round(last_intent_age, 3) if last_intent_age is not None else None,
            "last_motor_feedback": round(last_feedback_age, 3) if last_feedback_age is not None else None,
        },
        "ground_sense": _motor_ground_sense_snapshot(),
    }
    update_inastate("motor_control_status", payload)


def _walk_to_marker_policy(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    defaults: Dict[str, Any] = {
        "enabled": False,
        "marker_name": "marker_1m",
        "marker": {},
        "distance_m": 1.0,
        "arrival_radius": 0.2,
        "max_distance": 4.0,
        "step_duration": 0.9,
        "step_strength": 0.5,
        "run": False,
        "min_urge_level": 0.45,
        "cooldown_seconds": 45.0,
        "min_feedback_age": 0.6,
        "evaluation_grace_seconds": 1.2,
        "min_movement": 0.12,
        "min_progress": 0.05,
    }
    if cfg is None:
        cfg = load_config()
    raw = cfg.get("walk_to_marker_experiment", {}) if isinstance(cfg, dict) else {}
    raw = raw if isinstance(raw, dict) else {}
    policy = defaults.copy()
    policy["enabled"] = _coerce_bool(raw.get("enabled", policy["enabled"]), policy["enabled"])
    policy["marker_name"] = str(raw.get("marker_name") or policy["marker_name"])
    marker = raw.get("marker")
    policy["marker"] = marker if isinstance(marker, dict) else {}
    policy["distance_m"] = max(0.25, min(_coerce_float(raw.get("distance_m", policy["distance_m"]), policy["distance_m"]), 3.0))
    policy["arrival_radius"] = max(0.05, _coerce_float(raw.get("arrival_radius", policy["arrival_radius"]), policy["arrival_radius"]))
    policy["max_distance"] = max(policy["arrival_radius"] + 0.05, _coerce_float(raw.get("max_distance", policy["max_distance"]), policy["max_distance"]))
    policy["step_duration"] = max(0.2, _coerce_float(raw.get("step_duration", policy["step_duration"]), policy["step_duration"]))
    policy["step_strength"] = max(0.1, min(_coerce_float(raw.get("step_strength", policy["step_strength"]), policy["step_strength"]), 1.0))
    policy["run"] = _coerce_bool(raw.get("run", policy["run"]), policy["run"])
    policy["min_urge_level"] = max(0.0, min(_coerce_float(raw.get("min_urge_level", policy["min_urge_level"]), policy["min_urge_level"]), 1.0))
    policy["cooldown_seconds"] = max(0.0, _coerce_float(raw.get("cooldown_seconds", policy["cooldown_seconds"]), policy["cooldown_seconds"]))
    policy["min_feedback_age"] = max(0.0, _coerce_float(raw.get("min_feedback_age", policy["min_feedback_age"]), policy["min_feedback_age"]))
    policy["evaluation_grace_seconds"] = max(
        0.2, _coerce_float(raw.get("evaluation_grace_seconds", policy["evaluation_grace_seconds"]), policy["evaluation_grace_seconds"])
    )
    policy["min_movement"] = max(0.02, _coerce_float(raw.get("min_movement", policy["min_movement"]), policy["min_movement"]))
    policy["min_progress"] = max(0.01, _coerce_float(raw.get("min_progress", policy["min_progress"]), policy["min_progress"]))
    return policy


def _world_pose_snapshot() -> Optional[Dict[str, float]]:
    pose = get_inastate("world_pose")
    if not isinstance(pose, dict):
        return None
    pos = pose.get("position")
    if not isinstance(pos, (list, tuple)) or len(pos) < 3:
        return None
    try:
        x = float(pos[0])
        y = float(pos[1])
        z = float(pos[2])
    except Exception:
        return None
    if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
        return None
    try:
        yaw_deg = float(pose.get("yaw_deg")) if pose.get("yaw_deg") is not None else 0.0
    except Exception:
        yaw_deg = 0.0
    return {"x": x, "y": y, "z": z, "yaw_deg": yaw_deg}


def _distance_xy(ax: float, ay: float, bx: float, by: float) -> float:
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def _resolve_marker_position(policy: Dict[str, Any], pose: Dict[str, float]) -> Tuple[float, float, float]:
    marker = policy.get("marker")
    if isinstance(marker, dict) and marker:
        try:
            mx = float(marker.get("x"))
            my = float(marker.get("y"))
            mz = float(marker.get("z", pose["z"]))
            if math.isfinite(mx) and math.isfinite(my) and math.isfinite(mz):
                return mx, my, mz
        except Exception:
            pass
    yaw_rad = math.radians(float(pose.get("yaw_deg", 0.0)))
    distance_m = float(policy.get("distance_m", 1.0))
    # world-space forward for yaw in this simulation coordinate system
    fwd_x = -math.sin(yaw_rad)
    fwd_y = math.cos(yaw_rad)
    return (
        float(pose["x"] + (fwd_x * distance_m)),
        float(pose["y"] + (fwd_y * distance_m)),
        float(pose["z"]),
    )


def _set_walk_to_marker_status(status: str, **extra: Any) -> None:
    payload = {
        "status": status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": "model_manager",
    }
    payload.update(extra)
    update_inastate("walk_to_marker_status", payload)


def _finalize_walk_to_marker_attempt(now: float) -> None:
    global _walk_to_marker_attempt, _last_walk_to_marker_ts
    attempt = _walk_to_marker_attempt
    if not isinstance(attempt, dict):
        _walk_to_marker_attempt = None
        return
    _walk_to_marker_attempt = None
    _last_walk_to_marker_ts = now

    pose = _world_pose_snapshot()
    if not pose:
        _set_walk_to_marker_status(
            "step_failed",
            experiment_id=attempt.get("id"),
            success=False,
            failure_bucket="command_integration_or_execution",
            reason="no_world_pose_feedback",
            marker=attempt.get("marker"),
        )
        return

    start_pos = attempt.get("start_position") or [pose["x"], pose["y"], pose["z"]]
    marker = attempt.get("marker") or {}
    marker_x = _coerce_float(marker.get("x"), pose["x"])
    marker_y = _coerce_float(marker.get("y"), pose["y"])
    start_x = _coerce_float(start_pos[0], pose["x"]) if isinstance(start_pos, (list, tuple)) and len(start_pos) >= 1 else pose["x"]
    start_y = _coerce_float(start_pos[1], pose["y"]) if isinstance(start_pos, (list, tuple)) and len(start_pos) >= 2 else pose["y"]

    movement_m = _distance_xy(start_x, start_y, pose["x"], pose["y"])
    start_distance = _distance_xy(start_x, start_y, marker_x, marker_y)
    end_distance = _distance_xy(pose["x"], pose["y"], marker_x, marker_y)
    progress_m = max(0.0, start_distance - end_distance)

    min_movement = max(0.02, _coerce_float(attempt.get("min_movement"), 0.12))
    min_progress = max(0.01, _coerce_float(attempt.get("min_progress"), 0.05))
    success = movement_m >= min_movement and progress_m >= min_progress and end_distance <= start_distance
    result = "stepped_toward_marker" if success else "step_failed"
    failure_bucket = None

    diagnostics = {
        "movement_m": round(movement_m, 4),
        "progress_m": round(progress_m, 4),
        "start_distance_m": round(start_distance, 4),
        "end_distance_m": round(end_distance, 4),
        "motor_status": get_inastate("motor_control_status") or {},
        "touch_world": get_inastate("touch_world") or {},
        "ground_fault": get_inastate("ground_sense_fault") or {},
    }

    if not success:
        touch_world = diagnostics["touch_world"] if isinstance(diagnostics["touch_world"], dict) else {}
        contacts = touch_world.get("contacts") if isinstance(touch_world.get("contacts"), list) else []
        bounded = any(
            isinstance(entry, dict)
            and str(entry.get("surface", "")).startswith("bounds_")
            and _coerce_float(entry.get("pressure"), 0.0) >= 0.35
            for entry in contacts
        )
        ground_fault = diagnostics["ground_fault"] if isinstance(diagnostics["ground_fault"], dict) else {}
        motor_status = diagnostics["motor_status"] if isinstance(diagnostics["motor_status"], dict) else {}
        motor_decision = str(motor_status.get("decision") or "")

        if movement_m < min_movement:
            if _coerce_bool(ground_fault.get("active", False), False):
                failure_bucket = "grounding_fault"
            elif bounded:
                failure_bucket = "collision_or_bounds"
            elif motor_decision in {"stand_still_by_choice", "walk_to_marker_waiting", "walk_to_marker_cooldown"}:
                failure_bucket = "policy_or_instinct_hold"
            else:
                failure_bucket = "command_integration_or_execution"
        else:
            failure_bucket = "path_or_heading_mismatch"

    _set_walk_to_marker_status(
        result,
        experiment_id=attempt.get("id"),
        success=success,
        marker=attempt.get("marker"),
        start_position=[round(start_x, 4), round(start_y, 4), _coerce_float(start_pos[2], pose["z"]) if isinstance(start_pos, (list, tuple)) and len(start_pos) >= 3 else round(pose["z"], 4)],
        end_position=[round(pose["x"], 4), round(pose["y"], 4), round(pose["z"], 4)],
        diagnostics=diagnostics,
        failure_bucket=failure_bucket,
    )
    if success:
        log_to_statusbox("[Manager] Walk-to-marker experiment: step completed toward marker.")
    else:
        log_to_statusbox(f"[Manager] Walk-to-marker experiment: step failed ({failure_bucket or 'unknown'}).")


def _maybe_run_walk_to_marker_experiment(
    *,
    now: float,
    world_connected: bool,
    dreaming: bool,
    urge_level: float,
    threshold: float,
    cooldown: float,
    heartbeat_age: Optional[float],
    last_intent_age: Optional[float],
    last_feedback_age: Optional[float],
) -> bool:
    global _walk_to_marker_attempt, _last_walk_to_marker_ts, _last_motor_intent_ts
    policy = _walk_to_marker_policy(load_config())
    if not policy.get("enabled", False):
        return False

    if isinstance(_walk_to_marker_attempt, dict):
        evaluate_after = _coerce_float(_walk_to_marker_attempt.get("evaluate_after"), now)
        if now >= evaluate_after:
            _finalize_walk_to_marker_attempt(now)
        else:
            _set_walk_to_marker_status(
                "step_in_progress",
                experiment_id=_walk_to_marker_attempt.get("id"),
                evaluate_after=datetime.fromtimestamp(evaluate_after, timezone.utc).isoformat(),
            )
            _update_motor_control_status(
                now=now,
                decision="walk_to_marker_waiting",
                reason="awaiting_step_feedback",
                world_connected=world_connected,
                dreaming=dreaming,
                urge_level=urge_level,
                threshold=threshold,
                cooldown=cooldown,
                heartbeat_age=heartbeat_age,
                last_intent_age=last_intent_age,
                last_feedback_age=last_feedback_age,
            )
        return True

    experiment_cooldown = _coerce_float(policy.get("cooldown_seconds"), 0.0)
    if experiment_cooldown > 0 and _last_walk_to_marker_ts and (now - _last_walk_to_marker_ts) < experiment_cooldown:
        remaining = max(0.0, experiment_cooldown - (now - _last_walk_to_marker_ts))
        _set_walk_to_marker_status("cooldown", cooldown_remaining_seconds=round(remaining, 2))
        _update_motor_control_status(
            now=now,
            decision="walk_to_marker_cooldown",
            reason="experiment_cooldown",
            world_connected=world_connected,
            dreaming=dreaming,
            urge_level=urge_level,
            threshold=threshold,
            cooldown=cooldown,
            heartbeat_age=heartbeat_age,
            last_intent_age=last_intent_age,
            last_feedback_age=last_feedback_age,
        )
        return True

    if not world_connected:
        _set_walk_to_marker_status("blocked", reason="world_not_connected")
        _update_motor_control_status(
            now=now,
            decision="hold_world_disconnected",
            reason="walk_to_marker_world_not_connected",
            world_connected=world_connected,
            dreaming=dreaming,
            urge_level=urge_level,
            threshold=threshold,
            cooldown=cooldown,
            heartbeat_age=heartbeat_age,
            last_intent_age=last_intent_age,
            last_feedback_age=last_feedback_age,
        )
        return True

    if dreaming:
        _set_walk_to_marker_status("blocked", reason="dreaming_active")
        _update_motor_control_status(
            now=now,
            decision="hold_dreaming",
            reason="walk_to_marker_dreaming",
            world_connected=world_connected,
            dreaming=dreaming,
            urge_level=urge_level,
            threshold=threshold,
            cooldown=cooldown,
            heartbeat_age=heartbeat_age,
            last_intent_age=last_intent_age,
            last_feedback_age=last_feedback_age,
        )
        return True

    if heartbeat_age is not None and heartbeat_age > 30.0:
        _set_walk_to_marker_status("blocked", reason="world_heartbeat_stale")
        _update_motor_control_status(
            now=now,
            decision="hold_world_sync",
            reason="walk_to_marker_world_heartbeat_stale",
            world_connected=world_connected,
            dreaming=dreaming,
            urge_level=urge_level,
            threshold=threshold,
            cooldown=cooldown,
            heartbeat_age=heartbeat_age,
            last_intent_age=last_intent_age,
            last_feedback_age=last_feedback_age,
        )
        return True

    pose = _world_pose_snapshot()
    if pose is None:
        _set_walk_to_marker_status("blocked", reason="no_world_pose")
        _update_motor_control_status(
            now=now,
            decision="walk_to_marker_waiting",
            reason="missing_world_pose",
            world_connected=world_connected,
            dreaming=dreaming,
            urge_level=urge_level,
            threshold=threshold,
            cooldown=cooldown,
            heartbeat_age=heartbeat_age,
            last_intent_age=last_intent_age,
            last_feedback_age=last_feedback_age,
        )
        return True

    marker_x, marker_y, marker_z = _resolve_marker_position(policy, pose)
    marker_name = str(policy.get("marker_name") or "marker")
    marker_payload = {"name": marker_name, "x": round(marker_x, 4), "y": round(marker_y, 4), "z": round(marker_z, 4)}
    distance_to_marker = _distance_xy(pose["x"], pose["y"], marker_x, marker_y)
    arrival_radius = _coerce_float(policy.get("arrival_radius"), 0.2)
    max_distance = _coerce_float(policy.get("max_distance"), 4.0)

    if distance_to_marker <= arrival_radius:
        _last_walk_to_marker_ts = now
        _set_walk_to_marker_status("already_at_marker", marker=marker_payload, distance_m=round(distance_to_marker, 4), success=True)
        _update_motor_control_status(
            now=now,
            decision="stand_still_by_choice",
            reason="walk_to_marker_already_reached",
            world_connected=world_connected,
            dreaming=dreaming,
            urge_level=urge_level,
            threshold=threshold,
            cooldown=cooldown,
            heartbeat_age=heartbeat_age,
            last_intent_age=last_intent_age,
            last_feedback_age=last_feedback_age,
        )
        return True

    if distance_to_marker > max_distance:
        _last_walk_to_marker_ts = now
        _set_walk_to_marker_status("blocked", reason="marker_out_of_range", marker=marker_payload, distance_m=round(distance_to_marker, 4))
        _update_motor_control_status(
            now=now,
            decision="walk_to_marker_waiting",
            reason="marker_out_of_range",
            world_connected=world_connected,
            dreaming=dreaming,
            urge_level=urge_level,
            threshold=threshold,
            cooldown=cooldown,
            heartbeat_age=heartbeat_age,
            last_intent_age=last_intent_age,
            last_feedback_age=last_feedback_age,
        )
        return True

    min_urge = _coerce_float(policy.get("min_urge_level"), 0.45)
    if urge_level < min_urge:
        _last_walk_to_marker_ts = now
        _set_walk_to_marker_status(
            "refused_volitional",
            reason="urge_below_marker_threshold",
            marker=marker_payload,
            distance_m=round(distance_to_marker, 4),
            urge_level=round(urge_level, 3),
            min_urge_level=round(min_urge, 3),
        )
        _update_motor_control_status(
            now=now,
            decision="stand_still_by_choice",
            reason="walk_to_marker_low_urge",
            world_connected=world_connected,
            dreaming=dreaming,
            urge_level=urge_level,
            threshold=threshold,
            cooldown=cooldown,
            heartbeat_age=heartbeat_age,
            last_intent_age=last_intent_age,
            last_feedback_age=last_feedback_age,
        )
        return True

    min_feedback_age = _coerce_float(policy.get("min_feedback_age"), 0.6)
    if last_feedback_age is not None and last_feedback_age < min_feedback_age:
        _set_walk_to_marker_status("waiting", reason="recent_motor_feedback_settling")
        _update_motor_control_status(
            now=now,
            decision="stand_still_by_choice",
            reason="walk_to_marker_wait_feedback_settle",
            world_connected=world_connected,
            dreaming=dreaming,
            urge_level=urge_level,
            threshold=threshold,
            cooldown=cooldown,
            heartbeat_age=heartbeat_age,
            last_intent_age=last_intent_age,
            last_feedback_age=last_feedback_age,
        )
        return True

    dx = marker_x - pose["x"]
    dy = marker_y - pose["y"]
    distance = max(distance_to_marker, 1e-6)
    ux = dx / distance
    uy = dy / distance
    yaw_rad = math.radians(pose.get("yaw_deg", 0.0))
    cos_y = math.cos(yaw_rad)
    sin_y = math.sin(yaw_rad)
    strafe = (ux * cos_y) + (uy * sin_y)
    forward = (-ux * sin_y) + (uy * cos_y)
    norm = math.sqrt((forward * forward) + (strafe * strafe))
    if norm > 1.0:
        forward /= norm
        strafe /= norm
    step_strength = _coerce_float(policy.get("step_strength"), 0.5)
    forward *= step_strength
    strafe *= step_strength

    duration = _coerce_float(policy.get("step_duration"), 0.9)
    run = _coerce_bool(policy.get("run"), False)
    experiment_id = f"walk_marker_{int(now * 1000)}"
    marker_intent = {
        "id": experiment_id,
        "name": "walk_to_marker",
        "marker": marker_payload,
        "start_distance_m": round(distance_to_marker, 4),
        "step_only_if_wanted": True,
    }
    update_inastate(
        "motor_intent",
        {
            "forward": round(forward, 3),
            "strafe": round(strafe, 3),
            "turn": 0.0,
            "up": 0.0,
            "run": run,
            "duration": round(duration, 2),
            "seq": int(now * 1000),
            "timestamp": datetime.fromtimestamp(now, timezone.utc).isoformat(),
            "source": "model_manager",
            "reason": "walk_to_marker_experiment",
            "urge_level": round(urge_level, 3),
            "experiment": marker_intent,
        },
    )
    _last_motor_intent_ts = now
    _last_walk_to_marker_ts = now
    eval_after = now + max(0.5, duration) + _coerce_float(policy.get("evaluation_grace_seconds"), 1.2)
    _walk_to_marker_attempt = {
        "id": experiment_id,
        "issued_at": now,
        "evaluate_after": eval_after,
        "start_position": [pose["x"], pose["y"], pose["z"]],
        "marker": marker_payload,
        "min_movement": _coerce_float(policy.get("min_movement"), 0.12),
        "min_progress": _coerce_float(policy.get("min_progress"), 0.05),
    }
    _set_walk_to_marker_status(
        "step_issued",
        experiment_id=experiment_id,
        marker=marker_payload,
        start_position=[round(pose["x"], 4), round(pose["y"], 4), round(pose["z"], 4)],
        start_distance_m=round(distance_to_marker, 4),
        intent={"forward": round(forward, 3), "strafe": round(strafe, 3), "duration": round(duration, 2), "run": run},
        evaluate_after=datetime.fromtimestamp(eval_after, timezone.utc).isoformat(),
    )
    _update_motor_control_status(
        now=now,
        decision="move_intent_issued",
        reason="walk_to_marker_experiment",
        world_connected=world_connected,
        dreaming=dreaming,
        urge_level=urge_level,
        threshold=threshold,
        cooldown=cooldown,
        heartbeat_age=heartbeat_age,
        last_intent_age=0.0,
        last_feedback_age=last_feedback_age,
        intent_issued=True,
    )
    log_to_statusbox(
        "[Manager] Walk-to-marker experiment issued one voluntary step "
        f"(dist={distance_to_marker:.2f}m, urge={urge_level:.2f})."
    )
    return True


def _ground_fault_policy(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    defaults: Dict[str, Any] = {
        "enabled": True,
        "activate_after": 2,
        "clear_after": 3,
        "fall_speed_threshold": -0.35,
        "ground_distance_threshold": 0.1,
        "pressure_min_threshold": 0.06,
        "speed_tolerance": 0.2,
    }
    if cfg is None:
        cfg = load_config()
    raw = cfg.get("ground_sense_fault_guard", {}) if isinstance(cfg, dict) else {}
    raw = raw if isinstance(raw, dict) else {}
    policy = defaults.copy()
    policy["enabled"] = _coerce_bool(raw.get("enabled", policy["enabled"]), policy["enabled"])
    policy["activate_after"] = max(1, _coerce_int(raw.get("activate_after", policy["activate_after"]), int(policy["activate_after"])))
    policy["clear_after"] = max(1, _coerce_int(raw.get("clear_after", policy["clear_after"]), int(policy["clear_after"])))
    policy["fall_speed_threshold"] = _coerce_float(raw.get("fall_speed_threshold", policy["fall_speed_threshold"]), policy["fall_speed_threshold"])
    policy["ground_distance_threshold"] = max(
        0.01, _coerce_float(raw.get("ground_distance_threshold", policy["ground_distance_threshold"]), policy["ground_distance_threshold"])
    )
    policy["pressure_min_threshold"] = max(
        0.0, _coerce_float(raw.get("pressure_min_threshold", policy["pressure_min_threshold"]), policy["pressure_min_threshold"])
    )
    policy["speed_tolerance"] = max(0.01, _coerce_float(raw.get("speed_tolerance", policy["speed_tolerance"]), policy["speed_tolerance"]))
    return policy


def _update_ground_sense_fault_state() -> bool:
    global _ground_fault_streak, _ground_fault_clear_streak, _ground_fault_active
    global _ground_fault_window_id, _ground_fault_window_started_at, _last_ground_fault_log

    policy = _ground_fault_policy(load_config())
    now = time.time()
    now_iso = datetime.fromtimestamp(now, timezone.utc).isoformat()

    touch_feedback = get_inastate("touch_feedback") or {}
    touch_world = get_inastate("touch_world") or {}
    motor_feedback = get_inastate("motor_feedback") or {}

    tf_grounded = bool(touch_feedback.get("grounded")) if isinstance(touch_feedback, dict) else False
    tw_grounded = bool(touch_world.get("grounded")) if isinstance(touch_world, dict) else False
    mf_grounded = bool(motor_feedback.get("grounded")) if isinstance(motor_feedback, dict) else False

    vertical_speed = _coerce_float(motor_feedback.get("vertical_speed"), 0.0) if isinstance(motor_feedback, dict) else 0.0
    foot_pressure = _coerce_float(touch_feedback.get("foot_pressure"), 0.0) if isinstance(touch_feedback, dict) else 0.0
    ground_distance = _coerce_float(touch_world.get("ground_distance"), 0.0) if isinstance(touch_world, dict) else 0.0
    speed_tolerance = _coerce_float(policy.get("speed_tolerance"), 0.2)

    contradictions: List[str] = []
    if tf_grounded and vertical_speed <= _coerce_float(policy.get("fall_speed_threshold"), -0.35):
        contradictions.append("grounded_but_falling")
    if tf_grounded and ground_distance > _coerce_float(policy.get("ground_distance_threshold"), 0.1):
        contradictions.append("grounded_but_above_ground_plane")
    if abs(vertical_speed) <= speed_tolerance and (mf_grounded != tw_grounded):
        contradictions.append("motor_touch_ground_disagree")
    if tw_grounded and abs(vertical_speed) <= speed_tolerance and foot_pressure < _coerce_float(policy.get("pressure_min_threshold"), 0.06):
        contradictions.append("grounded_without_foot_pressure")

    if not _coerce_bool(policy.get("enabled"), True):
        _ground_fault_streak = 0
        _ground_fault_clear_streak = 0
        _ground_fault_active = False
        _ground_fault_window_id = None
        _ground_fault_window_started_at = None
        update_inastate(
            "ground_sense_fault",
            {
                "active": False,
                "enabled": False,
                "timestamp": now_iso,
                "contradictions": [],
                "streak": 0,
            },
        )
        update_inastate(
            "map_training_guard",
            {
                "timestamp": now_iso,
                "aggressive_training_allowed": True,
                "reason": "ground_fault_guard_disabled",
            },
        )
        return False

    previous_active = _ground_fault_active
    if contradictions:
        _ground_fault_streak += 1
        _ground_fault_clear_streak = 0
    else:
        _ground_fault_streak = 0
        if _ground_fault_active:
            _ground_fault_clear_streak += 1
        else:
            _ground_fault_clear_streak = 0

    activate_after = max(1, _coerce_int(policy.get("activate_after"), 2))
    clear_after = max(1, _coerce_int(policy.get("clear_after"), 3))
    if contradictions and _ground_fault_streak >= activate_after:
        _ground_fault_active = True
        if _ground_fault_window_id is None:
            _ground_fault_window_id = f"gsf_{uuid.uuid4().hex[:10]}"
        if _ground_fault_window_started_at is None:
            _ground_fault_window_started_at = now_iso
    elif _ground_fault_active and not contradictions and _ground_fault_clear_streak >= clear_after:
        _ground_fault_active = False
        _ground_fault_window_id = None
        _ground_fault_window_started_at = None

    payload = {
        "active": bool(_ground_fault_active),
        "enabled": True,
        "timestamp": now_iso,
        "contradictions": contradictions,
        "streak": int(_ground_fault_streak),
        "clear_streak": int(_ground_fault_clear_streak),
        "window_id": _ground_fault_window_id,
        "window_started_at": _ground_fault_window_started_at,
        "evidence": {
            "touch_feedback_grounded": tf_grounded,
            "touch_world_grounded": tw_grounded,
            "motor_feedback_grounded": mf_grounded,
            "vertical_speed": round(vertical_speed, 4),
            "foot_pressure": round(foot_pressure, 4),
            "ground_distance": round(ground_distance, 4),
        },
        "tag_fragments_with": ["sensor_incoherent"] if _ground_fault_active else [],
        "aggressive_map_training_allowed": not _ground_fault_active,
    }
    update_inastate("ground_sense_fault", payload)
    update_inastate(
        "map_training_guard",
        {
            "timestamp": now_iso,
            "aggressive_training_allowed": not _ground_fault_active,
            "reason": "ground_sense_fault_active" if _ground_fault_active else "ground_sense_consistent",
            "fault_window_id": _ground_fault_window_id,
        },
    )

    should_log = False
    if previous_active != _ground_fault_active:
        should_log = True
    elif contradictions and (_last_ground_fault_log == 0.0 or (now - _last_ground_fault_log) >= _GROUND_FAULT_LOG_COOLDOWN):
        should_log = True
    if should_log:
        if _ground_fault_active:
            log_to_statusbox(f"[Manager] Ground-sense fault active: {', '.join(contradictions)}.")
        elif previous_active and not _ground_fault_active:
            log_to_statusbox("[Manager] Ground-sense fault cleared; aggressive map training re-enabled.")
        _last_ground_fault_log = now

    return bool(_ground_fault_active)


def _maybe_emit_motor_intent() -> None:
    global _last_motor_intent_ts
    now = time.time()
    world_connected = bool(get_inastate("world_connected", False))
    dreaming = bool(get_inastate("dreaming", False))

    move_urge = get_inastate("urge_to_move") or {}
    if not isinstance(move_urge, dict):
        move_urge = {}
    urge_level = _resolve_meta_adjusted_level(move_urge, default=0.0)

    config = load_config()
    try:
        threshold = float(config.get("motor_urge_threshold", _MOTOR_URGE_THRESHOLD))
    except Exception:
        threshold = _MOTOR_URGE_THRESHOLD

    try:
        cooldown = float(config.get("motor_intent_cooldown", _MOTOR_INTENT_COOLDOWN))
    except Exception:
        cooldown = _MOTOR_INTENT_COOLDOWN
    cooldown = max(0.0, cooldown)

    heartbeat_age = _age_seconds(get_inastate("last_world_heartbeat"), now=now)
    last_intent_age = _age_seconds(get_inastate("last_motor_intent"), now=now)
    last_feedback_age = _age_seconds(get_inastate("last_motor_update"), now=now)

    if _maybe_run_walk_to_marker_experiment(
        now=now,
        world_connected=world_connected,
        dreaming=dreaming,
        urge_level=urge_level,
        threshold=threshold,
        cooldown=cooldown,
        heartbeat_age=heartbeat_age,
        last_intent_age=last_intent_age,
        last_feedback_age=last_feedback_age,
    ):
        return

    if not world_connected:
        _update_motor_control_status(
            now=now,
            decision="hold_world_disconnected",
            reason="world_not_connected",
            world_connected=world_connected,
            dreaming=dreaming,
            urge_level=urge_level,
            threshold=threshold,
            cooldown=cooldown,
            heartbeat_age=heartbeat_age,
            last_intent_age=last_intent_age,
            last_feedback_age=last_feedback_age,
        )
        return

    if dreaming:
        _update_motor_control_status(
            now=now,
            decision="hold_dreaming",
            reason="dreaming_active",
            world_connected=world_connected,
            dreaming=dreaming,
            urge_level=urge_level,
            threshold=threshold,
            cooldown=cooldown,
            heartbeat_age=heartbeat_age,
            last_intent_age=last_intent_age,
            last_feedback_age=last_feedback_age,
        )
        return

    if urge_level < threshold:
        _update_motor_control_status(
            now=now,
            decision="stand_still_by_choice",
            reason="urge_below_threshold",
            world_connected=world_connected,
            dreaming=dreaming,
            urge_level=urge_level,
            threshold=threshold,
            cooldown=cooldown,
            heartbeat_age=heartbeat_age,
            last_intent_age=last_intent_age,
            last_feedback_age=last_feedback_age,
        )
        return

    if cooldown > 0 and _last_motor_intent_ts and (now - _last_motor_intent_ts) < cooldown:
        _update_motor_control_status(
            now=now,
            decision="stand_still_by_choice",
            reason="intent_cooldown_active",
            world_connected=world_connected,
            dreaming=dreaming,
            urge_level=urge_level,
            threshold=threshold,
            cooldown=cooldown,
            heartbeat_age=heartbeat_age,
            last_intent_age=last_intent_age,
            last_feedback_age=last_feedback_age,
        )
        return

    if heartbeat_age is not None and heartbeat_age > 30.0:
        _update_motor_control_status(
            now=now,
            decision="hold_world_sync",
            reason="world_heartbeat_stale",
            world_connected=world_connected,
            dreaming=dreaming,
            urge_level=urge_level,
            threshold=threshold,
            cooldown=cooldown,
            heartbeat_age=heartbeat_age,
            last_intent_age=last_intent_age,
            last_feedback_age=last_feedback_age,
        )
        return

    if last_intent_age is not None and last_intent_age < max(2.0, cooldown * 0.5):
        _update_motor_control_status(
            now=now,
            decision="stand_still_by_choice",
            reason="recent_intent_still_active",
            world_connected=world_connected,
            dreaming=dreaming,
            urge_level=urge_level,
            threshold=threshold,
            cooldown=cooldown,
            heartbeat_age=heartbeat_age,
            last_intent_age=last_intent_age,
            last_feedback_age=last_feedback_age,
        )
        return

    if last_feedback_age is not None and last_feedback_age < 2.5:
        _update_motor_control_status(
            now=now,
            decision="stand_still_by_choice",
            reason="recent_feedback_settling",
            world_connected=world_connected,
            dreaming=dreaming,
            urge_level=urge_level,
            threshold=threshold,
            cooldown=cooldown,
            heartbeat_age=heartbeat_age,
            last_intent_age=last_intent_age,
            last_feedback_age=last_feedback_age,
        )
        return

    scale = max(0.2, min(1.0, urge_level))
    forward = min(1.0, 0.35 + (0.45 * scale))
    strafe = random.uniform(-0.4, 0.4) * scale
    turn = random.uniform(-0.6, 0.6) * scale
    duration = min(1.8, 0.6 + (0.9 * scale))
    run = scale > 0.75

    update_inastate(
        "motor_intent",
        {
            "forward": round(forward, 3),
            "strafe": round(strafe, 3),
            "turn": round(turn, 3),
            "up": 0.0,
            "run": run,
            "duration": round(duration, 2),
            "seq": int(now * 1000),
            "timestamp": datetime.fromtimestamp(now, timezone.utc).isoformat(),
            "source": "model_manager",
            "reason": "urge_to_move",
            "urge_level": round(urge_level, 3),
        },
    )
    _last_motor_intent_ts = now
    _update_motor_control_status(
        now=now,
        decision="move_intent_issued",
        reason="urge_to_move",
        world_connected=world_connected,
        dreaming=dreaming,
        urge_level=urge_level,
        threshold=threshold,
        cooldown=cooldown,
        heartbeat_age=heartbeat_age,
        last_intent_age=0.0,
        last_feedback_age=last_feedback_age,
        intent_issued=True,
    )
    log_to_statusbox(f"[Manager] Motor intent issued (urge {urge_level:.2f}).")


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


def _update_humor_bridge():
    """
    Refresh the humor expression invite so downstream comms can see whether
    Ina actually feels amused enough to riff.
    """
    global _last_humor_bridge_log
    try:
        from humor_engine import maybe_prepare_expression_invite
    except Exception:
        return

    try:
        invite = maybe_prepare_expression_invite()
    except Exception as exc:
        log_to_statusbox(f"[Manager] Humor bridge update failed: {exc}")
        return

    if not invite or not invite.get("ready"):
        return

    now = time.time()
    if (now - _last_humor_bridge_log) < _HUMOR_BRIDGE_LOG_COOLDOWN:
        return

    level = float(invite.get("level", 0.0) or 0.0)
    log_to_statusbox(
        f"[Manager] Ina is amused (playfulness {level:.2f}); humour stays invitation-only."
    )
    _last_humor_bridge_log = now


def _maybe_run_trauma_processor():
    """
    Invoke the trauma processor on a slow cadence so cooling support
    is available when fragments spike. The processor performs its own
    energy/stress gating; this function just rate-limits launches.
    """
    global _last_trauma_run
    if _memory_guard_state.get("level") in {"soft", "hard"}:
        return
    now = time.time()
    if (now - _last_trauma_run) < _TRAUMA_COOLDOWN:
        return
    _last_trauma_run = now
    safe_run(["python", "trauma_processor.py"])

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
    if not isinstance(emo, dict):
        emo = {}
    if not emo and isinstance(snapshot, dict):
        emo = snapshot if isinstance(snapshot, dict) else {}
    if not isinstance(emo, dict):
        emo = {}

    identity_hint = {}
    last_reflection = get_inastate("last_reflection_event") or {}
    if isinstance(last_reflection, dict):
        identity_hint = last_reflection.get("identity_hint") or {}

    energy = _clamp01(get_inastate("current_energy") or 0.5, default=0.5)
    sleep_pressure = _clamp01(get_inastate("sleep_pressure") or 0.0, default=0.0)
    urge_voice = get_inastate("urge_to_voice") or get_inastate("urge_to_communicate") or {}
    if not isinstance(urge_voice, dict):
        urge_voice = {}
    urge_type = get_inastate("urge_to_type") or {}
    if not isinstance(urge_type, dict):
        urge_type = {}
    urge_voice_level = _resolve_meta_adjusted_level(urge_voice, default=0.0)
    urge_type_level = _resolve_meta_adjusted_level(urge_type, default=0.0)

    prediction = get_inastate("current_prediction") or {}
    if not isinstance(prediction, dict):
        prediction = {}
    pred_vec = prediction.get("predicted_vector") or {}
    if not isinstance(pred_vec, dict):
        pred_vec = {}
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
    resource_context = _extract_resource_context()
    resource_trend_pressure = _clamp01(resource_context.get("trend_pressure"), default=0.0)

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

    if resource_context.get("available"):
        resource_weight = 0.85
        total_weight += resource_weight
        total_contrib += resource_trend_pressure * resource_weight
        reasons.append(
            {
                "axis": "resource_trend",
                "value": round(1.0 - resource_trend_pressure, 3),
                "pressure": round(resource_trend_pressure, 3),
                "weight": round(resource_weight, 3),
                "direction": "risk" if resource_trend_pressure >= 0.55 else "watch",
                "reason": resource_context.get("trend_summary") or resource_context.get("summary") or "resource telemetry available",
                "hint": resource_context.get("optimization_hint"),
                "focus_module": resource_context.get("largest_module"),
            }
        )

    reasons = sorted(reasons, key=lambda r: r["pressure"] * r["weight"], reverse=True)
    importance_score = _clamp01(total_contrib / max(total_weight, 1.0), default=0.0)

    machine_semantics = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "scaffold_version": scaffold.get("version", 1),
        "axes": axes_out,
        "resource_context": {
            "available": resource_context.get("available", False),
            "pressure_level": resource_context.get("pressure_level"),
            "trend_pressure": round(resource_trend_pressure, 3),
            "trend_summary": resource_context.get("trend_summary"),
            "optimization_hint": resource_context.get("optimization_hint"),
            "top_modules": resource_context.get("top_modules"),
        },
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


def _maybe_repair_corrupt_fragments():
    """
    Optionally repair or quarantine corrupted fragments using Ina's choice
    and conservative safety rails.
    """
    global _last_fragment_repair_run

    policy = _fragment_repair_policy()
    if not policy.get("enabled", False):
        return
    if _memory_guard_state.get("level") in {"soft", "hard"}:
        return

    now = time.time()
    cooldown = float(policy.get("cooldown_seconds") or 0.0)
    if cooldown > 0 and _last_fragment_repair_run and (now - _last_fragment_repair_run) < cooldown:
        return

    intent = get_inastate("fragment_repair_intent")
    intent_allowed = False
    intent_mode = None
    if isinstance(intent, dict):
        intent_allowed = _coerce_bool(intent.get("allow", False), False) or _coerce_bool(intent.get("enabled", False), False)
        intent_mode = intent.get("mode")
    else:
        intent_allowed = _coerce_bool(intent, False)

    if policy.get("require_intent", True) and not intent_allowed:
        return

    if intent_mode:
        mode = str(intent_mode).lower()
        if mode in {"quarantine", "delete", "repair", "inspect"}:
            policy["mode"] = mode

    queue = get_inastate("corrupt_fragments") or []
    if not isinstance(queue, list) or not queue:
        return

    remaining, summary = process_corrupt_queue(CHILD, queue, policy)
    update_inastate("corrupt_fragments", remaining)
    update_inastate("fragment_repair_last_run", summary)
    _last_fragment_repair_run = now

    if policy.get("consume_intent", True) and intent_allowed:
        update_inastate("fragment_repair_intent", False)

    counts = summary.get("counts", {}) if isinstance(summary, dict) else {}
    log_to_statusbox(
        "[Fragments] Repair pass "
        f"({policy.get('mode')}): "
        f"repaired {counts.get('repaired', 0)}, "
        f"quarantined {counts.get('quarantined', 0)}, "
        f"deleted {counts.get('deleted', 0)}, "
        f"remaining {summary.get('remaining', 0)}."
    )

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
    _maybe_update_runtime_heartbeat()
    _ensure_continuity_thread()
    _apply_world_sense_override()
    _maybe_ensure_ina_client()
    guard_limits = _memory_guard_limits()
    _consume_operator_memory_signal(guard_limits)
    memory_guard = _refresh_memory_guard_state()
    memory_level = memory_guard.get("level")
    _maybe_enqueue_memory_pressure_event(memory_guard, guard_limits)
    _process_memory_pressure_queue(guard_limits)
    _maybe_shed_memory_pressure(str(memory_level or ""), guard_limits)
    defer_optional = memory_level in {"soft", "hard"}
    defer_spawns = memory_level == "hard"
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



    world_connected = bool(get_inastate("world_connected", False))
    ground_fault_active = _update_ground_sense_fault_state()

    if not defer_optional and get_inastate("emotion_snapshot", {}).get("focus", 0.0) > 0.5:
        safe_popen(["python", "meditation_state.py"])

    if not defer_optional:
        if not world_connected and get_inastate("emotion_snapshot", {}).get("fuzz_level", 0.0) > 0.7:
            safe_popen(["python", "dreamstate.py"])
        elif not world_connected and not get_inastate("dreaming", False):
            safe_popen(["python", "dreamstate.py"])

    if not defer_spawns:
        safe_run(["python", "emotion_engine.py"])
        safe_run(["python", "instinct_engine.py"])

    if not defer_optional:
        safe_popen(["python", "early_comm.py"])

    if not defer_optional and not feedback_inhibition():
        safe_popen(["python", "predictive_layer.py"])
        safe_popen(["python", "logic_engine.py"])

    if not defer_optional:
        boredom_check()
        paint_check()
        _maybe_self_read()
        if not ground_fault_active:
            rebuild_maps_if_needed()
    _check_self_adjustment()
    _update_contact_urges()
    _update_stable_pattern_urge()
    _update_self_read_exploration_opportunities()
    _update_meta_arbitration_signal(memory_guard=memory_guard)
    _maybe_emit_motor_intent()
    _update_humor_bridge()
    _maybe_run_trauma_processor()
    _run_passive_reflection()
    if not defer_optional:
        _step_deep_recall()
    _update_machine_semantics()
    if not defer_optional:
        _scan_fragment_health()
        _maybe_repair_corrupt_fragments()
        _run_prediction_meta_analysis()
        _maybe_bundle_memory(defer_optional)
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
            log_to_statusbox(traceback.format_exc())
            time.sleep(5)

if __name__ == "__main__":  
    launch_background_loops()
    schedule_runtime()
