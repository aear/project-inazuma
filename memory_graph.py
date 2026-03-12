# === memory_graph.py (Logging Enhanced) ===

import os
import json
import math
import random
import heapq
import time
import hashlib
import gc
import importlib
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Iterable, List, Optional, Set, Tuple, TYPE_CHECKING
from gui_hook import log_to_statusbox
from body_schema import get_region_anchors

if TYPE_CHECKING:  # pragma: no cover
    from transformers.fractal_multidimensional_transformers import FractalTransformer

try:
    from cold_storage import compact_fragment_file, policy_from_config as cold_policy_from_config
except Exception:  # pragma: no cover
    compact_fragment_file = None
    cold_policy_from_config = None

MEMORY_TIERS = ["short", "working", "long", "cold"]

NEURAL_MAP_BURST_DEFAULT = 60
EXPERIENCE_GRAPH_BURST_DEFAULT = 200

DEFAULT_INCREMENTAL_POLICY = {
    "mode": "incremental",
    "fragment_batch": None,
    "batch_size": 24,
    "build_budget_ms": 350.0,
    "position_blend": 0.25,
    "merge_slack": 0.03,
    "max_new_neurons": 120,
    "max_edges_updated": 20000,
    "max_synapse_pairs": 120000,
    "dirty_index_enabled": True,
    "full_mode_incremental": True,
    "local_rebuild_max_fragments": 2000,
    "emit_sparse_snapshot": True,
    "synapse_refresh_on_idle": True,
    "max_fragments_per_neuron": 128,
    "max_tags_per_neuron": 32,
    "vector_round_digits": 6,
    "position_round_digits": 4,
    "edge_direction_enabled": True,
    "gc_every_batches": 4,
    "max_neurons_total": 0,
    "max_edges_per_neuron": 0,
}

DEFAULT_TIER_POLICY = {
    "short": {"max_age_hours": 18.0, "target_count": 5000},
    "working": {"max_age_hours": 72.0, "target_count": 12000},
    "long": {"max_age_hours": 24.0 * 30.0, "target_count": 40000},
    "cold": {},
}
DEFAULT_BOOT_POLICY = {
    "boot_mode": "full",  # full | fast | auto
    "rebalance_on_boot": True,
    "prune_missing_on_boot": False,
    "shutdown_intent_ttl_hours": 6.0,
    "heartbeat_stale_seconds": 120.0,
}
DEFAULT_BUNDLE_PRUNE_POLICY = {
    "enabled": False,
    "allow_apply": False,
    "verify_sha": True,
    "require_bundle_ready": True,
    "max_prune": 0,
}
DEFAULT_CUSTOM_TRANSFORMER_RUNTIME = {
    "enabled": True,
    "cooldown_seconds": 300.0,
    "sample_limit": 24,
    "run_hindsight": True,
    "soul_drift_steps": 1,
}


def _load_config():
    path = Path("config.json")
    if path.exists():
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _neural_settings():
    cfg = _load_config()
    cluster = float(cfg.get("neural_cluster_threshold", 0.88))
    synapse = float(cfg.get("neural_synapse_threshold", 0.84))
    tag_weight = float(cfg.get("neural_tag_weight", 0.25))
    tag_weight = max(0.0, min(1.0, tag_weight))
    return cluster, synapse, tag_weight


def _neural_policy(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if cfg is None:
        cfg = _load_config()
    raw = {}
    if isinstance(cfg, dict):
        raw = cfg.get("neural_map_policy", {}) or {}
    if not isinstance(raw, dict):
        raw = {}
    policy = DEFAULT_INCREMENTAL_POLICY.copy()
    policy.update({k: raw.get(k, policy[k]) for k in policy.keys() if k in raw})
    mode = str(policy.get("mode", "incremental")).lower().strip()
    full_mode_incremental = bool(policy.get("full_mode_incremental", True))
    incremental = mode not in {"rebuild", "overwrite", "legacy", "full"}
    if mode == "full" and full_mode_incremental:
        incremental = True
    fragment_batch = policy.get("fragment_batch")
    try:
        fragment_batch = int(fragment_batch) if fragment_batch is not None else None
    except (TypeError, ValueError):
        fragment_batch = None
    try:
        batch_size = int(policy.get("batch_size", 24))
    except (TypeError, ValueError):
        batch_size = 24
    if batch_size <= 0:
        batch_size = 1
    def _clamp(val: float, lo: float, hi: float) -> float:
        try:
            return max(lo, min(float(val), hi))
        except (TypeError, ValueError):
            return lo
    position_blend = _clamp(policy.get("position_blend", 0.25), 0.0, 1.0)
    merge_slack = _clamp(policy.get("merge_slack", 0.03), 0.0, 0.25)
    try:
        max_new = int(policy.get("max_new_neurons", 120))
        if max_new < 0:
            max_new = 0
    except (TypeError, ValueError):
        max_new = 0
    try:
        max_edges_updated = int(policy.get("max_edges_updated", 20000))
    except (TypeError, ValueError):
        max_edges_updated = 20000
    if max_edges_updated < 0:
        max_edges_updated = 0
    try:
        max_synapse_pairs = int(policy.get("max_synapse_pairs", 120000))
    except (TypeError, ValueError):
        max_synapse_pairs = 120000
    if max_synapse_pairs < 0:
        max_synapse_pairs = 0
    try:
        build_budget_ms = float(policy.get("build_budget_ms", 350.0))
    except (TypeError, ValueError):
        build_budget_ms = 350.0
    if build_budget_ms < 0:
        build_budget_ms = 0.0
    dirty_index_enabled = bool(policy.get("dirty_index_enabled", True))
    try:
        local_rebuild_max = int(policy.get("local_rebuild_max_fragments", 2000))
    except (TypeError, ValueError):
        local_rebuild_max = 2000
    if local_rebuild_max < 0:
        local_rebuild_max = 0
    emit_sparse_snapshot = bool(policy.get("emit_sparse_snapshot", True))
    synapse_refresh = bool(policy.get("synapse_refresh_on_idle", True))
    try:
        max_fragments_per_neuron = int(policy.get("max_fragments_per_neuron", 128))
    except (TypeError, ValueError):
        max_fragments_per_neuron = 128
    max_fragments_per_neuron = max(1, max_fragments_per_neuron)
    try:
        max_tags_per_neuron = int(policy.get("max_tags_per_neuron", 32))
    except (TypeError, ValueError):
        max_tags_per_neuron = 32
    max_tags_per_neuron = max(1, max_tags_per_neuron)
    try:
        vector_round_digits = int(policy.get("vector_round_digits", 6))
    except (TypeError, ValueError):
        vector_round_digits = 6
    vector_round_digits = max(2, min(vector_round_digits, 8))
    try:
        position_round_digits = int(policy.get("position_round_digits", 4))
    except (TypeError, ValueError):
        position_round_digits = 4
    position_round_digits = max(1, min(position_round_digits, 6))
    edge_direction_enabled = bool(policy.get("edge_direction_enabled", True))
    try:
        gc_every_batches = int(policy.get("gc_every_batches", 4))
    except (TypeError, ValueError):
        gc_every_batches = 4
    if gc_every_batches < 0:
        gc_every_batches = 0
    try:
        max_neurons_total = int(policy.get("max_neurons_total", 0))
    except (TypeError, ValueError):
        max_neurons_total = 0
    if max_neurons_total < 0:
        max_neurons_total = 0
    try:
        max_edges_per_neuron = int(policy.get("max_edges_per_neuron", 0))
    except (TypeError, ValueError):
        max_edges_per_neuron = 0
    if max_edges_per_neuron < 0:
        max_edges_per_neuron = 0
    return {
        "mode": mode,
        "incremental": incremental,
        "fragment_batch": fragment_batch,
        "batch_size": batch_size,
        "build_budget_ms": build_budget_ms,
        "position_blend": position_blend,
        "merge_slack": merge_slack,
        "max_new_neurons": max_new,
        "max_edges_updated": max_edges_updated,
        "max_synapse_pairs": max_synapse_pairs,
        "dirty_index_enabled": dirty_index_enabled,
        "full_mode_incremental": full_mode_incremental,
        "local_rebuild_max_fragments": local_rebuild_max,
        "emit_sparse_snapshot": emit_sparse_snapshot,
        "synapse_refresh_on_idle": synapse_refresh,
        "max_fragments_per_neuron": max_fragments_per_neuron,
        "max_tags_per_neuron": max_tags_per_neuron,
        "vector_round_digits": vector_round_digits,
        "position_round_digits": position_round_digits,
        "edge_direction_enabled": edge_direction_enabled,
        "gc_every_batches": gc_every_batches,
        "max_neurons_total": max_neurons_total,
        "max_edges_per_neuron": max_edges_per_neuron,
    }


def _custom_transformer_runtime_policy(cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    config = DEFAULT_CUSTOM_TRANSFORMER_RUNTIME.copy()
    raw: Dict[str, Any] = {}
    if isinstance(cfg, dict):
        direct = cfg.get("custom_transformer_runtime")
        nested_parent = cfg.get("neural_map_policy") or {}
        nested = nested_parent.get("custom_transformers") if isinstance(nested_parent, dict) else {}
        if isinstance(direct, dict):
            raw.update(direct)
        if isinstance(nested, dict):
            raw.update(nested)
    for key in config.keys():
        if key in raw:
            config[key] = raw[key]
    config["enabled"] = bool(config.get("enabled", True))
    config["run_hindsight"] = bool(config.get("run_hindsight", True))
    config["cooldown_seconds"] = max(0.0, _safe_float(config.get("cooldown_seconds"), 300.0))
    try:
        config["sample_limit"] = max(1, int(config.get("sample_limit", 24)))
    except (TypeError, ValueError):
        config["sample_limit"] = 24
    try:
        config["soul_drift_steps"] = max(1, min(int(config.get("soul_drift_steps", 1)), 8))
    except (TypeError, ValueError):
        config["soul_drift_steps"] = 1
    return config


def _memory_policy():
    """
    Pull tier policy (age caps + target counts) from config.json when present.
    Falls back to defaults tuned for keeping short-term lean.
    """
    cfg = _load_config()
    user_policy = cfg.get("memory_policy", {}) if isinstance(cfg, dict) else {}
    policy = {}

    def _coerce_positive_int(value: Any) -> Optional[int]:
        try:
            ivalue = int(value)
        except (TypeError, ValueError):
            return None
        return ivalue if ivalue > 0 else None

    def _coerce_positive_float(value: Any) -> Optional[float]:
        try:
            fvalue = float(value)
        except (TypeError, ValueError):
            return None
        return fvalue if fvalue > 0 else None

    for tier in MEMORY_TIERS:
        tier_defaults = DEFAULT_TIER_POLICY.get(tier, {}).copy()
        overrides = user_policy.get(tier, {}) if isinstance(user_policy, dict) else {}
        if isinstance(overrides, dict):
            age_override = _coerce_positive_float(overrides.get("max_age_hours"))
            if age_override is not None:
                tier_defaults["max_age_hours"] = age_override
            target_override = _coerce_positive_int(overrides.get("target_count"))
            if target_override is not None:
                tier_defaults["target_count"] = target_override
        policy[tier] = tier_defaults
    return policy


def _cold_storage_policy() -> Dict[str, Any]:
    cfg = _load_config()
    if cold_policy_from_config is None:
        return {"enabled": False, "auto_compact": False}
    try:
        return cold_policy_from_config(cfg)
    except Exception:
        return {"enabled": False, "auto_compact": False}


EMOTION_SLIDERS: Tuple[str, ...] = (
    "intensity",
    "attention",
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
)

DEFAULT_NEURAL_SELECTOR = {
    "enabled": True,
    "candidate_pool_max": 420,
    "prefilter_enabled": True,
    "prefilter_multiplier": 3.0,
    "prefilter_min": 48,
    "prefilter_max": 180,
    "cooldown_seconds": 900.0,
    "cooldown_max": 120,
    "blocked_tags": ["exception", "privacy", "high-risk", "sensor_incoherent"],
    "cost_max_bytes": 5_000_000,
    "dream_cost_max_bytes": 1_000_000,
    "risk_max": 0.85,
    "dream_risk_max": 0.55,
    "recency_half_life_sec": 21600.0,
    "weights": {
        "emotion": 0.35,
        "symbol": 0.2,
        "recency": 0.1,
        "novelty": 0.15,
        "unresolved": 0.1,
        "cost": 0.2,
        "risk": 0.2,
    },
    "temperature": 0.85,
    "temperature_min": 0.2,
    "temperature_max": 1.8,
    "lane_weights": {
        "guided": 0.7,
        "novelty": 0.2,
        "wild": 0.1,
    },
    "energy_low": 0.35,
    "stress_high": 0.5,
    "clarity_low": -0.35,
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _parse_iso_timestamp(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    raw = str(value)
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


def _age_seconds(value: Optional[str]) -> Optional[float]:
    ts = _parse_iso_timestamp(value)
    if ts is None:
        return None
    return max(0.0, time.time() - ts)


def _boot_policy(cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    policy = DEFAULT_BOOT_POLICY.copy()
    if not isinstance(cfg, dict):
        return policy
    mode = cfg.get("memory_graph_boot_mode", policy["boot_mode"])
    policy["boot_mode"] = str(mode or policy["boot_mode"]).lower().strip()
    policy["rebalance_on_boot"] = bool(cfg.get("memory_rebalance_on_boot", policy["rebalance_on_boot"]))
    policy["prune_missing_on_boot"] = bool(cfg.get("memory_graph_prune_missing", policy["prune_missing_on_boot"]))
    try:
        ttl = float(cfg.get("shutdown_intent_ttl_hours", policy["shutdown_intent_ttl_hours"]))
        policy["shutdown_intent_ttl_hours"] = max(0.0, ttl)
    except (TypeError, ValueError):
        pass
    try:
        stale = float(cfg.get("runtime_heartbeat_stale_seconds", policy["heartbeat_stale_seconds"]))
        policy["heartbeat_stale_seconds"] = max(0.0, stale)
    except (TypeError, ValueError):
        pass
    return policy


def _bundle_prune_policy(cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    policy = DEFAULT_BUNDLE_PRUNE_POLICY.copy()
    if not isinstance(cfg, dict):
        return policy
    raw = cfg.get("bundle_prune_policy") or {}
    if not isinstance(raw, dict):
        return policy
    if "enabled" in raw:
        policy["enabled"] = bool(raw.get("enabled"))
    if "allow_apply" in raw:
        policy["allow_apply"] = bool(raw.get("allow_apply"))
    if "verify_sha" in raw:
        policy["verify_sha"] = bool(raw.get("verify_sha"))
    if "require_bundle_ready" in raw:
        policy["require_bundle_ready"] = bool(raw.get("require_bundle_ready"))
    if "max_prune" in raw:
        try:
            policy["max_prune"] = int(raw.get("max_prune") or 0)
        except (TypeError, ValueError):
            policy["max_prune"] = 0
    return policy


def _load_inastate(child: str) -> Dict[str, Any]:
    path = Path("AI_Children") / child / "memory" / "inastate.json"
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _write_inastate_value(child: str, key: str, value: Any) -> None:
    path = Path("AI_Children") / child / "memory" / "inastate.json"
    data: Dict[str, Any] = {}
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh) or {}
        except Exception:
            data = {}
    if not isinstance(data, dict):
        data = {}
    data[key] = value
    try:
        with path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
    except Exception:
        return


def _extract_timestamp(raw: Any) -> Optional[float]:
    if raw is None:
        return None
    if isinstance(raw, dict):
        raw = raw.get("timestamp")
    if isinstance(raw, (int, float)):
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None
    if isinstance(raw, str):
        return _parse_iso_timestamp(raw)
    return None


def _shutdown_context(child: str, policy: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg_policy = policy or DEFAULT_BOOT_POLICY
    state = _load_inastate(child)
    last_shutdown = state.get("last_shutdown") if isinstance(state.get("last_shutdown"), dict) else {}
    shutdown_intent = state.get("shutdown_intent") if isinstance(state.get("shutdown_intent"), dict) else {}
    runtime_disruption = bool(state.get("runtime_disruption"))

    heartbeat_raw = state.get("runtime_heartbeat")
    heartbeat_ts = _extract_timestamp(heartbeat_raw)
    heartbeat_age = None
    if heartbeat_ts is not None:
        heartbeat_age = max(0.0, time.time() - heartbeat_ts)

    shutdown_ts = _extract_timestamp(last_shutdown)
    intent_ts = _extract_timestamp(shutdown_intent)

    if heartbeat_ts is not None and shutdown_ts is not None and shutdown_ts < heartbeat_ts:
        last_shutdown = {}
        shutdown_ts = None
    if heartbeat_ts is not None and intent_ts is not None and intent_ts < heartbeat_ts:
        shutdown_intent = {}
        intent_ts = None

    ttl_hours = cfg_policy.get("shutdown_intent_ttl_hours", DEFAULT_BOOT_POLICY["shutdown_intent_ttl_hours"])
    if intent_ts is not None and ttl_hours > 0:
        if (time.time() - intent_ts) > (ttl_hours * 3600.0):
            shutdown_intent = {}
            intent_ts = None

    return {
        "last_shutdown": last_shutdown,
        "shutdown_intent": shutdown_intent,
        "runtime_disruption": runtime_disruption,
        "heartbeat_age": heartbeat_age,
        "heartbeat_ts": heartbeat_ts,
        "shutdown_ts": shutdown_ts,
        "intent_ts": intent_ts,
    }


def _resolve_boot_mode(policy: Dict[str, Any], shutdown: Dict[str, Any]) -> Tuple[str, str]:
    mode = str(policy.get("boot_mode") or "full").lower().strip()
    if mode in {"incremental", "fast"}:
        return "fast", "config"
    if mode == "full":
        return "full", "config"

    last_shutdown = shutdown.get("last_shutdown") or {}
    shutdown_intent = shutdown.get("shutdown_intent") or {}
    shutdown_source = (last_shutdown.get("source") or shutdown_intent.get("source") or "").lower()
    shutdown_clean = bool(last_shutdown.get("clean"))

    if shutdown_source == "gui":
        return "fast", "gui_shutdown"
    if shutdown_clean:
        return "fast", "clean_shutdown"
    if shutdown.get("runtime_disruption"):
        return "full", "runtime_disruption"

    heartbeat_age = shutdown.get("heartbeat_age")
    stale_limit = float(policy.get("heartbeat_stale_seconds", DEFAULT_BOOT_POLICY["heartbeat_stale_seconds"]))
    if heartbeat_age is not None and stale_limit > 0 and heartbeat_age > stale_limit:
        return "full", "stale_heartbeat"

    return "fast", "auto_default"


def _selector_state_path(child: str) -> Path:
    return Path("AI_Children") / child / "memory" / "neural_selector_state.json"


def _load_selector_state(child: str) -> Dict[str, Any]:
    path = _selector_state_path(child)
    if not path.exists():
        return {"recent": []}
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return {"recent": []}
    return data if isinstance(data, dict) else {"recent": []}


def _save_selector_state(child: str, state: Dict[str, Any]) -> None:
    path = _selector_state_path(child)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=2)
    except Exception:
        return


def _selector_config(cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    config = DEFAULT_NEURAL_SELECTOR.copy()
    raw = {}
    if isinstance(cfg, dict):
        raw = cfg.get("neural_selector", {}) or {}
    if isinstance(raw, dict):
        for key in config.keys():
            if key in {"weights", "lane_weights"}:
                continue
            if key in raw:
                config[key] = raw.get(key)
        weights = config["weights"].copy()
        raw_weights = raw.get("weights") if isinstance(raw, dict) else None
        if isinstance(raw_weights, dict):
            for key in weights:
                if key in raw_weights:
                    weights[key] = _safe_float(raw_weights.get(key), weights[key])
        config["weights"] = weights
        lane_weights = config["lane_weights"].copy()
        raw_lanes = raw.get("lane_weights") if isinstance(raw, dict) else None
        if isinstance(raw_lanes, dict):
            for key in lane_weights:
                if key in raw_lanes:
                    lane_weights[key] = _safe_float(raw_lanes.get(key), lane_weights[key])
        config["lane_weights"] = lane_weights
    config["candidate_pool_max"] = max(0, int(_safe_float(config.get("candidate_pool_max"), 0)))
    config["prefilter_enabled"] = bool(config.get("prefilter_enabled", True))
    config["prefilter_multiplier"] = max(1.0, _safe_float(config.get("prefilter_multiplier"), 3.0))
    config["prefilter_min"] = max(0, int(_safe_float(config.get("prefilter_min"), 48)))
    config["prefilter_max"] = max(0, int(_safe_float(config.get("prefilter_max"), 180)))
    config["cooldown_seconds"] = max(0.0, _safe_float(config.get("cooldown_seconds"), 0.0))
    config["cooldown_max"] = max(0, int(_safe_float(config.get("cooldown_max"), 0)))
    config["cost_max_bytes"] = int(_safe_float(config.get("cost_max_bytes"), 0.0)) or None
    config["dream_cost_max_bytes"] = int(_safe_float(config.get("dream_cost_max_bytes"), 0.0)) or None
    config["risk_max"] = _clamp(_safe_float(config.get("risk_max"), 1.0), 0.0, 1.0)
    config["dream_risk_max"] = _clamp(_safe_float(config.get("dream_risk_max"), 1.0), 0.0, 1.0)
    config["recency_half_life_sec"] = max(1.0, _safe_float(config.get("recency_half_life_sec"), 21600.0))
    config["temperature"] = _safe_float(config.get("temperature"), 0.85)
    config["temperature_min"] = _safe_float(config.get("temperature_min"), 0.2)
    config["temperature_max"] = _safe_float(config.get("temperature_max"), 1.8)
    config["energy_low"] = _safe_float(config.get("energy_low"), 0.35)
    config["stress_high"] = _safe_float(config.get("stress_high"), 0.5)
    config["clarity_low"] = _safe_float(config.get("clarity_low"), -0.35)
    blocked = config.get("blocked_tags") or []
    if isinstance(blocked, (list, tuple)):
        config["blocked_tags"] = [str(tag).lower() for tag in blocked if tag]
    else:
        config["blocked_tags"] = []
    return config


def _selector_prefilter_target(limit: int, config: Dict[str, Any]) -> int:
    if not config.get("prefilter_enabled", True):
        return 0
    target = int(math.ceil(max(1, limit) * config.get("prefilter_multiplier", 3.0)))
    target = max(target, int(config.get("prefilter_min", 0) or 0))
    prefilter_max = int(config.get("prefilter_max", 0) or 0)
    if prefilter_max > 0:
        target = min(target, prefilter_max)
    return max(target, 0)


def _collect_recent_symbol_ids(state: Dict[str, Any]) -> List[str]:
    symbols: List[str] = []
    metrics = state.get("tone_voice_metrics") or {}
    recent = metrics.get("recent_symbols") or []
    if isinstance(recent, list):
        symbols.extend(str(s) for s in recent if s)
    history = state.get("tone_voice_history") or []
    if isinstance(history, list):
        symbols.extend(str(entry.get("symbol")) for entry in history[-8:] if isinstance(entry, dict) and entry.get("symbol"))
    for key in ("last_spoken_symbol", "last_symbol_word_id"):
        if state.get(key):
            symbols.append(str(state.get(key)))
    prediction = state.get("current_prediction") or {}
    if isinstance(prediction, dict):
        pred_word = prediction.get("predicted_symbol_word") or {}
        if isinstance(pred_word, dict) and pred_word.get("symbol_word_id"):
            symbols.append(str(pred_word.get("symbol_word_id")))
    symbol_matches = state.get("emotion_symbol_matches") or []
    if isinstance(symbol_matches, list):
        for entry in symbol_matches:
            if isinstance(entry, dict) and entry.get("symbol_word_id"):
                symbols.append(str(entry.get("symbol_word_id")))
    return list({s for s in symbols if s})


def _mode_from_state(state: Dict[str, Any]) -> str:
    if state.get("dreaming"):
        return "dream"
    if state.get("meditating"):
        return "meditation"
    mode = state.get("mode") or (state.get("emotion_snapshot") or {}).get("mode")
    boredom = _safe_float(state.get("emotion_boredom"), 0.0)
    if boredom > 0.4:
        return "boredom"
    return str(mode or "awake").lower()


def _emotion_vector_from_state(state: Dict[str, Any]) -> List[float]:
    snapshot = state.get("emotion_snapshot") or {}
    values = snapshot.get("values") if isinstance(snapshot, dict) else snapshot
    if not isinstance(values, dict):
        return [0.0] * len(EMOTION_SLIDERS)
    vector = [_clamp(_safe_float(values.get(axis), 0.0), -1.0, 1.0) for axis in EMOTION_SLIDERS]
    return vector


def _emotion_vector_from_fragment(fragment: Dict[str, Any]) -> List[float]:
    emotions = fragment.get("emotions") or {}
    if not isinstance(emotions, dict):
        return [0.0] * len(EMOTION_SLIDERS)
    sliders = emotions.get("sliders") if isinstance(emotions.get("sliders"), dict) else emotions
    if not isinstance(sliders, dict):
        return [0.0] * len(EMOTION_SLIDERS)
    return [_clamp(_safe_float(sliders.get(axis), 0.0), -1.0, 1.0) for axis in EMOTION_SLIDERS]


def _fragment_symbols(fragment: Dict[str, Any]) -> List[str]:
    symbols: List[str] = []
    for key in ("symbols", "symbols_spoken", "attempted_symbols"):
        value = fragment.get(key)
        if isinstance(value, list):
            symbols.extend(str(s) for s in value if s)
        elif isinstance(value, str):
            symbols.append(value)
    context = fragment.get("context") or {}
    if isinstance(context, dict):
        ctx_symbols = context.get("symbols")
        if isinstance(ctx_symbols, list):
            symbols.extend(str(s) for s in ctx_symbols if s)
    return list({s for s in symbols if s})


def _cluster_key(fragment: Dict[str, Any]) -> str:
    for key in ("cluster_id", "cluster", "cluster_key"):
        if fragment.get(key):
            return str(fragment.get(key))
    tags = fragment.get("tags") or []
    if isinstance(tags, list) and tags:
        return str(tags[0])
    return str(fragment.get("type") or "unknown")


def _overlap_score(a: Iterable[str], b: Iterable[str]) -> float:
    set_a = {str(x).lower() for x in a if x}
    set_b = {str(x).lower() for x in b if x}
    if not set_a or not set_b:
        return 0.0
    union = set_a | set_b
    return len(set_a & set_b) / max(len(union), 1)


def _resolve_index_path(child: str, frag_id: str, meta: Dict[str, Any]) -> Optional[Path]:
    base = Path("AI_Children") / child / "memory" / "fragments"
    filename = meta.get("filename") or f"frag_{frag_id}.json"
    tier = meta.get("tier")
    if tier:
        candidate = base / tier / filename
        if candidate.exists():
            return candidate
    candidate = base / filename
    if candidate.exists():
        return candidate
    return None


def _load_fragment_from_path(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return None
    if "tier" not in data:
        parent_tier = path.parent.name
        if parent_tier in MEMORY_TIERS:
            data["tier"] = parent_tier
    return data


def _neighbor_ids(fragment: Dict[str, Any]) -> List[str]:
    neighbors: List[str] = []
    for key in ("prev_id", "next_id", "prev", "next"):
        value = fragment.get(key)
        if isinstance(value, str) and value:
            neighbors.append(value)
        elif isinstance(value, list):
            neighbors.extend(str(item) for item in value if item)
    context = fragment.get("context")
    if isinstance(context, dict):
        ctx_neighbors = context.get("neighbors") or context.get("neighbor_ids")
        if isinstance(ctx_neighbors, list):
            neighbors.extend(str(item) for item in ctx_neighbors if item)
    return list(dict.fromkeys(neighbors))


def _is_compacted_fragment(fragment: Dict[str, Any]) -> bool:
    if isinstance(fragment.get("cold_core"), dict):
        return True
    if fragment.get("cold_compacted_at"):
        return True
    return False


def _candidate_pool_from_index(
    child: str,
    index: Dict[str, Any],
    *,
    pool_max: int,
    blocked_tags: Iterable[str],
    recent_ids: Set[str],
    known_fragments: Optional[Set[str]],
    cost_max_bytes: Optional[int],
) -> List[str]:
    blocked = {str(tag).lower() for tag in blocked_tags if tag}
    entries: List[Tuple[float, float, str]] = []
    for frag_id, meta in index.items():
        if known_fragments and frag_id in known_fragments:
            continue
        tags = meta.get("tags") or []
        if isinstance(tags, list) and blocked:
            lowered = {str(tag).lower() for tag in tags if tag}
            if lowered & blocked:
                continue
        if frag_id in recent_ids:
            continue
        path = _resolve_index_path(child, frag_id, meta)
        if path is None:
            continue
        if cost_max_bytes is not None:
            try:
                if path.stat().st_size > cost_max_bytes:
                    continue
            except OSError:
                continue
        ts = _parse_iso_timestamp(meta.get("last_seen") or meta.get("timestamp"))
        if ts is None:
            try:
                ts = path.stat().st_mtime
            except OSError:
                ts = 0.0
        importance = _safe_float(meta.get("importance"), 0.0)
        entries.append((ts or 0.0, importance, frag_id))
    entries.sort(reverse=True)
    return [entry[2] for entry in entries[:pool_max]]


def _softmax(scores: List[float], temperature: float) -> List[float]:
    if not scores:
        return []
    temp = max(temperature, 1e-6)
    scaled = [score / temp for score in scores]
    peak = max(scaled)
    weights = [math.exp(val - peak) for val in scaled]
    total = sum(weights) or 1e-6
    return [w / total for w in weights]


def _weighted_sample(items: List[Dict[str, Any]], weights: List[float], k: int) -> List[Dict[str, Any]]:
    if k <= 0 or not items:
        return []
    selected: List[Dict[str, Any]] = []
    pool = list(items)
    pool_weights = list(weights)
    for _ in range(min(k, len(pool))):
        total = sum(pool_weights)
        if total <= 0:
            choice = random.choice(pool)
        else:
            pick = random.random() * total
            upto = 0.0
            choice = pool[-1]
            for item, weight in zip(pool, pool_weights):
                upto += weight
                if upto >= pick:
                    choice = item
                    break
        idx = pool.index(choice)
        selected.append(choice)
        pool.pop(idx)
        pool_weights.pop(idx)
    return selected


def _unresolved_intensity(fragment: Dict[str, Any]) -> float:
    tags = fragment.get("tags") or []
    tagset = {str(tag).lower() for tag in tags if tag}
    unresolved = 0.0
    if tagset & {"unresolved", "suppressed", "high_conflict", "trauma", "shadow"}:
        unresolved = 0.35
    emotions = fragment.get("emotions") or {}
    summary = emotions.get("summary") if isinstance(emotions, dict) else None
    if isinstance(summary, dict):
        cooled = summary.get("cooled_intensity")
        if cooled is not None:
            unresolved = max(unresolved, min(abs(_safe_float(cooled)), 1.0))
    intensity = None
    if isinstance(emotions, dict):
        intensity = emotions.get("intensity")
        if intensity is None and isinstance(emotions.get("sliders"), dict):
            intensity = emotions.get("sliders", {}).get("intensity")
    if intensity is not None:
        unresolved = max(unresolved, min(abs(_safe_float(intensity)), 1.0))
    return _clamp(unresolved, 0.0, 1.0)


def _risk_level(fragment: Dict[str, Any]) -> float:
    emotions = fragment.get("emotions") or {}
    risk = None
    if isinstance(emotions, dict):
        risk = emotions.get("risk")
        if risk is None and isinstance(emotions.get("sliders"), dict):
            risk = emotions.get("sliders", {}).get("risk")
    return _clamp(max(0.0, _safe_float(risk, 0.0)), 0.0, 1.0)


def _score_candidates(
    candidates: List[Dict[str, Any]],
    *,
    emotion_vector: List[float],
    recent_symbols: List[str],
    now_ts: float,
    config: Dict[str, Any],
    mode: str,
) -> List[Dict[str, Any]]:
    cluster_counts: Dict[str, int] = {}
    for entry in candidates:
        cluster_key = _cluster_key(entry["fragment"])
        entry["cluster_key"] = cluster_key
        cluster_counts[cluster_key] = cluster_counts.get(cluster_key, 0) + 1

    total = max(1, len(candidates))
    weights = config["weights"]
    recency_half_life = config["recency_half_life_sec"]
    cost_scale = config.get("cost_max_bytes") or 5_000_000

    scored: List[Dict[str, Any]] = []
    for entry in candidates:
        frag = entry["fragment"]
        frag_symbols = _fragment_symbols(frag)
        emo_vec = _emotion_vector_from_fragment(frag)
        emo_sim = cosine_similarity(emotion_vector, emo_vec)
        symbol_overlap = _overlap_score(recent_symbols, frag_symbols)

        frag_ts = entry.get("timestamp")
        if frag_ts is None:
            frag_ts = now_ts
        age = max(0.0, now_ts - frag_ts)
        recency = math.exp(-age / recency_half_life) if recency_half_life else 0.0

        cluster_freq = cluster_counts.get(entry["cluster_key"], 1) / total
        novelty = 1.0 - cluster_freq
        unresolved = _unresolved_intensity(frag)
        risk = entry.get("risk", 0.0)

        bytes_cost = entry.get("bytes", 0)
        cost_penalty = min(bytes_cost / max(cost_scale, 1), 1.0) if bytes_cost else 0.0

        if mode == "dream" and risk > config["dream_risk_max"]:
            continue
        if mode == "dream" and config.get("dream_cost_max_bytes") and bytes_cost > config["dream_cost_max_bytes"]:
            continue
        if risk > config["risk_max"]:
            continue

        contrib = {
            "emotion_sim": weights["emotion"] * emo_sim,
            "symbol_overlap": weights["symbol"] * symbol_overlap,
            "recency": weights["recency"] * recency,
            "novelty": weights["novelty"] * novelty,
            "unresolved": weights["unresolved"] * unresolved,
            "cost": -weights["cost"] * cost_penalty,
            "risk": -weights["risk"] * risk,
        }
        score = sum(contrib.values())
        score = _clamp(score, -5.0, 5.0)
        scored.append({
            **entry,
            "score": score,
            "features": {
                "emotion_sim": emo_sim,
                "symbol_overlap": symbol_overlap,
                "recency": recency,
                "novelty": novelty,
                "unresolved": unresolved,
                "cost_penalty": cost_penalty,
                "risk": risk,
            },
            "contrib": contrib,
        })
    return scored


def _select_with_lanes(
    scored: List[Dict[str, Any]],
    *,
    limit: int,
    temperature: float,
    config: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    if not scored or limit <= 0:
        return [], {"guided": 0, "novelty": 0, "wild": 0}
    if limit >= len(scored):
        return scored[:limit], {"guided": len(scored), "novelty": 0, "wild": 0}

    lane_weights = config["lane_weights"]
    guided_target = int(round(limit * lane_weights.get("guided", 0.7)))
    novelty_target = int(round(limit * lane_weights.get("novelty", 0.2)))
    wild_target = max(0, limit - guided_target - novelty_target)

    guided_probs = _softmax([item["score"] for item in scored], temperature)
    guided = _weighted_sample(scored, guided_probs, guided_target)

    remaining = [item for item in scored if item not in guided]
    novelty_weights = [item["features"]["novelty"] for item in remaining]
    novelty = _weighted_sample(remaining, novelty_weights, novelty_target)

    remaining = [item for item in remaining if item not in novelty]
    wild = random.sample(remaining, min(wild_target, len(remaining))) if remaining else []

    selected = guided + novelty + wild
    if len(selected) < limit and remaining:
        refill = [item for item in remaining if item not in wild]
        selected.extend(refill[: max(0, limit - len(selected))])
    lane_counts = {"guided": len(guided), "novelty": len(novelty), "wild": len(wild)}
    return selected[:limit], lane_counts


def _log_selector_audit(child: str, payload: Dict[str, Any]) -> None:
    path = Path("AI_Children") / child / "memory" / "neural_selector_log.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        return


def select_fragments_for_neural_map(
    child: str,
    limit: int,
    *,
    cfg: Optional[Dict[str, Any]] = None,
    known_fragments: Optional[Set[str]] = None,
) -> Tuple[List[Dict[str, Any]], int, Dict[str, Any]]:
    config = _selector_config(cfg)
    if not config.get("enabled", True):
        fragments, total = load_fragments(child, limit=limit)
        return fragments, total, {"mode": "disabled"}

    index_path = Path("AI_Children") / child / "memory" / "memory_map.json"
    if not index_path.exists():
        fragments, total = load_fragments(child, limit=limit)
        return fragments, total, {"mode": "fallback"}
    try:
        with index_path.open("r", encoding="utf-8") as fh:
            index = json.load(fh)
    except Exception:
        fragments, total = load_fragments(child, limit=limit)
        return fragments, total, {"mode": "fallback"}
    if not isinstance(index, dict) or not index:
        fragments, total = load_fragments(child, limit=limit)
        return fragments, total, {"mode": "fallback"}

    inastate = _load_inastate(child)
    mode = _mode_from_state(inastate)
    emotion_vector = _emotion_vector_from_state(inastate)
    recent_symbols = _collect_recent_symbol_ids(inastate)

    selector_state = _load_selector_state(child)
    now_ts = time.time()
    cooldown_seconds = config["cooldown_seconds"]
    recent_entries = selector_state.get("recent") or []
    recent_ids: Set[str] = set()
    pruned_recent = []
    for entry in recent_entries:
        if not isinstance(entry, dict):
            continue
        frag_id = entry.get("id")
        ts = _safe_float(entry.get("ts"), 0.0)
        if not frag_id:
            continue
        if cooldown_seconds and (now_ts - ts) > cooldown_seconds:
            continue
        recent_ids.add(str(frag_id))
        pruned_recent.append({"id": str(frag_id), "ts": ts})

    pool_max = max(config["candidate_pool_max"], max(1, limit) * 6)
    cost_max = config.get("cost_max_bytes")
    if mode == "dream" and config.get("dream_cost_max_bytes"):
        cost_max = min(cost_max or config["dream_cost_max_bytes"], config["dream_cost_max_bytes"])

    candidate_ids = _candidate_pool_from_index(
        child,
        index,
        pool_max=pool_max,
        blocked_tags=config.get("blocked_tags", []),
        recent_ids=recent_ids,
        known_fragments=known_fragments,
        cost_max_bytes=cost_max,
    )
    if not candidate_ids:
        fragments, total = load_fragments(child, limit=limit)
        return fragments, total, {"mode": "fallback"}
    raw_pool = len(candidate_ids)
    prefilter_target = _selector_prefilter_target(limit, config)
    if prefilter_target and len(candidate_ids) > prefilter_target:
        candidate_ids = candidate_ids[:prefilter_target]

    candidates: List[Dict[str, Any]] = []
    for frag_id in candidate_ids:
        meta = index.get(frag_id, {}) if isinstance(index, dict) else {}
        path = _resolve_index_path(child, frag_id, meta)
        if path is None:
            continue
        fragment = _load_fragment_from_path(path)
        if not fragment:
            continue
        frag_ts = _parse_iso_timestamp(fragment.get("timestamp"))
        if frag_ts is None:
            try:
                frag_ts = path.stat().st_mtime
            except OSError:
                frag_ts = now_ts
        candidates.append({
            "fragment": fragment,
            "fragment_id": frag_id,
            "timestamp": frag_ts,
            "bytes": _safe_float(path.stat().st_size, 0.0) if path.exists() else 0.0,
            "risk": _risk_level(fragment),
            "importance": _safe_float(meta.get("importance"), 0.0),
        })

    scored = _score_candidates(
        candidates,
        emotion_vector=emotion_vector,
        recent_symbols=recent_symbols,
        now_ts=now_ts,
        config=config,
        mode=mode,
    )
    if not scored:
        fragments, total = load_fragments(child, limit=limit)
        return fragments, total, {"mode": "fallback"}

    stress = _safe_float((inastate.get("emotion_snapshot") or {}).get("values", {}).get("stress"), 0.0)
    clarity = _safe_float((inastate.get("emotion_snapshot") or {}).get("values", {}).get("clarity"), 0.0)
    energy = _safe_float(inastate.get("current_energy"), 0.5)

    temperature = _safe_float(config["temperature"], 0.85)
    if energy < config["energy_low"]:
        temperature *= 0.7
    if stress > config["stress_high"]:
        temperature *= 0.7
    if clarity < config["clarity_low"]:
        temperature *= 0.85
    if mode == "boredom":
        temperature *= 1.2
    if mode == "dream":
        temperature *= 1.6
    if mode == "meditation":
        temperature *= 0.85
    temperature = _clamp(temperature, config["temperature_min"], config["temperature_max"])

    selected, lane_counts = _select_with_lanes(
        scored,
        limit=limit,
        temperature=temperature,
        config=config,
    )

    selected_ids = [item["fragment_id"] for item in selected]
    if selected_ids:
        for frag_id in selected_ids:
            pruned_recent.append({"id": frag_id, "ts": now_ts})
        if config["cooldown_max"]:
            pruned_recent = pruned_recent[-config["cooldown_max"] :]
        selector_state["recent"] = pruned_recent
        selector_state["updated_at"] = datetime.now(timezone.utc).isoformat()
        _save_selector_state(child, selector_state)

    audit = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "temperature": round(temperature, 4),
        "limit": limit,
        "pool": len(candidate_ids),
        "pool_raw": raw_pool,
        "pool_prefilter": len(candidate_ids),
        "scored": len(scored),
        "lane_counts": lane_counts,
        "selected_ids": selected_ids[: min(10, len(selected_ids))],
        "energy": round(energy, 4),
        "stress": round(stress, 4),
        "clarity": round(clarity, 4),
    }
    top_contrib = []
    for item in selected[: min(5, len(selected))]:
        contrib = item.get("contrib", {})
        sorted_terms = sorted(contrib.items(), key=lambda kv: abs(kv[1]), reverse=True)
        top_contrib.append({
            "fragment_id": item.get("fragment_id"),
            "top_terms": [(term, round(val, 4)) for term, val in sorted_terms[:3]],
        })
    audit["top_contributors"] = top_contrib
    _log_selector_audit(child, audit)

    slim_fragments = []
    for item in selected:
        slim = _slim_fragment(item["fragment"])
        if slim:
            slim_fragments.append(slim)

    return slim_fragments, len(index), audit


# === Spatial helpers (body schema → neural positions) ===
def _load_body_anchors() -> Dict[str, Dict[str, float]]:
    anchors = get_region_anchors()
    if not anchors:
        return {}
    return anchors


def _guess_region_from_tags(tags: List[str], anchors: Dict[str, Dict[str, float]]) -> str:
    tagset = {str(t).lower() for t in (tags or [])}

    if {"audio", "sound", "voice", "hearing"} & tagset:
        return "head" if "head" in anchors else next(iter(anchors.keys()), "head")
    if {"vision", "image", "video", "sight"} & tagset:
        return "head" if "head" in anchors else next(iter(anchors.keys()), "head")
    if {"emotion", "feeling", "heart"} & tagset:
        return "chest" if "chest" in anchors else next(iter(anchors.keys()), "chest")
    if {"core", "energy", "stomach", "gut"} & tagset:
        return "core" if "core" in anchors else next(iter(anchors.keys()), "core")
    if "left_arm" in tagset and "left_arm" in anchors:
        return "left_arm"
    if "right_arm" in tagset and "right_arm" in anchors:
        return "right_arm"
    if "left_leg" in tagset and "left_leg" in anchors:
        return "left_leg"
    if "right_leg" in tagset and "right_leg" in anchors:
        return "right_leg"

    # Default to head or first available anchor
    if "head" in anchors:
        return "head"
    return next(iter(anchors.keys()), "head")


def _project_vector_to_anchor(vector: List[float], anchor: Dict[str, float], seed: str) -> List[float]:
    """
    Map a latent vector into body space using the region's anchor.
    Keeps placement stable via a hash-based RNG when vectors are missing.
    """
    center = anchor.get("center", [0.0, 0.0, 0.0])
    radius = float(anchor.get("radius", 1.0) or 1.0)

    rng = random.Random(hash(seed) & 0xFFFFFFFF)
    if vector and len(vector) >= 3:
        base = [float(v) for v in vector[:3]]
        norm = math.sqrt(sum(v * v for v in base)) or 1e-6
        unit = [v / norm for v in base]
    else:
        theta = rng.uniform(0, 2 * math.pi)
        phi = rng.uniform(0, math.pi)
        unit = [
            math.sin(phi) * math.cos(theta),
            math.sin(phi) * math.sin(theta),
            math.cos(phi),
        ]

    r = radius * (0.35 + 0.6 * rng.random())
    return [center[i] + unit[i] * r for i in range(3)]


# === Experience Graph Utilities ===
def _experience_base(child: str, base_path: Optional[Path] = None) -> Path:
    root = Path(base_path) if base_path else Path("AI_Children")
    return root / child / "memory" / "experiences"


def load_experience_events(child: str, base_path: Optional[Path] = None, limit: Optional[int] = None) -> Tuple[List[Dict[str, Any]], int]:
    """Load structured events previously logged by the experience logger (optionally limited)."""

    events_dir = _experience_base(child, base_path) / "events"
    if not events_dir.exists():
        return [], 0

    limit_val = 0
    if limit is not None:
        try:
            limit_val = max(0, int(limit))
        except (TypeError, ValueError):
            limit_val = 0

    def _ts_value(payload: Dict[str, Any], path: Path) -> float:
        raw = payload.get("timestamp") or ""
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
        except Exception:
            try:
                return path.stat().st_mtime
            except Exception:
                return 0.0

    events: List[Dict[str, Any]] = []
    total = 0

    if limit_val > 0:
        heap: List[tuple] = []
        for path in sorted(events_dir.glob("evt_*.json")):
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
            except Exception:
                continue
            if "id" not in data:
                continue
            total += 1
            entry = (_ts_value(data, path), path.name, data)
            if len(heap) < limit_val:
                heapq.heappush(heap, entry)
            else:
                if entry[0] > heap[0][0]:
                    heapq.heapreplace(heap, entry)
        heap.sort()
        events = [item[2] for item in heap]
    else:
        for path in sorted(events_dir.glob("evt_*.json")):
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
            except Exception:
                continue
            if "id" not in data:
                continue
            events.append(data)
        total = len(events)

    return events, total


def build_experience_graph(child: str, base_path: Optional[Path] = None) -> Dict[str, Any]:
    """Construct a graph over events grounded in shared entities and words."""

    cfg = _load_config()
    burst_limit = cfg.get("experience_graph_burst")
    try:
        burst_limit = int(burst_limit)
    except (TypeError, ValueError):
        burst_limit = EXPERIENCE_GRAPH_BURST_DEFAULT
    if burst_limit <= 0:
        burst_limit = EXPERIENCE_GRAPH_BURST_DEFAULT

    events, total_events = load_experience_events(
        child,
        base_path=base_path,
        limit=burst_limit,
    )
    if not events:
        return {
            "events": [],
            "edges": [],
            "words_index": {},
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    if total_events > len(events):
        log_to_statusbox(
            f"[ExperienceGraph] Limiting to {len(events)} most recent events (burst={burst_limit}, total={total_events})."
        )

    nodes: List[Dict[str, Any]] = []
    words_index: Dict[str, Set[str]] = {}

    for raw in events:
        entity_labels = {
            entity.get("name") or entity.get("label")
            for entity in raw.get("perceived_entities", [])
            if entity.get("name") or entity.get("label")
        }
        node = {
            "id": raw.get("id"),
            "timestamp": raw.get("timestamp"),
            "situation_tags": raw.get("situation_tags", []),
            "entities": sorted(entity_labels),
            "episode_id": raw.get("episode_id"),
            "narrative": raw.get("narrative", ""),
            "word_usage": raw.get("word_usage", []),
        }
        nodes.append(node)
        for usage in node["word_usage"]:
            for word in usage.get("words", []):
                if not word:
                    continue
                words_index.setdefault(word.lower(), set()).add(node["id"])

    edges: List[Dict[str, Any]] = []
    for i, left in enumerate(nodes):
        left_tags = set(left.get("situation_tags", []))
        left_entities = set(left.get("entities", []))
        for right in nodes[i + 1 :]:
            shared_tags = left_tags.intersection(right.get("situation_tags", []))
            shared_entities = left_entities.intersection(right.get("entities", []))
            if not shared_tags and not shared_entities:
                continue
            edges.append(
                {
                    "source": left["id"],
                    "target": right["id"],
                    "shared_tags": sorted(shared_tags),
                    "shared_entities": sorted(shared_entities),
                }
            )

    graph = {
        "events": nodes,
        "edges": edges,
        "words_index": {word: sorted(list(event_ids)) for word, event_ids in words_index.items()},
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    out_path = _experience_base(child, base_path) / "experience_graph.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(graph, fh, indent=2)

    log_to_statusbox(
        f"[ExperienceGraph] {len(nodes)} events | {len(edges)} edges | {len(words_index)} grounded words."
    )
    return graph

# === Core Utilities ===
def cosine_similarity(v1, v2):
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    return dot / (norm1 * norm2 + 1e-8)

def vector_average(vectors):
    if not vectors:
        return []
    length = len(vectors[0])
    avg = [0.0] * length
    for vec in vectors:
        for i in range(length):
            avg[i] += vec[i]
    return [round(x / len(vectors), 6) for x in avg]


def tag_similarity(tags_a, tags_b):
    a = set(tags_a or [])
    b = set(tags_b or [])
    union = a.union(b)
    if not union:
        return 0.0
    return len(a.intersection(b)) / len(union)


def _slim_fragment(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Keep only the fields needed for encoding/clustering to reduce memory pressure.
    """
    if not data or "id" not in data:
        return None
    tags = data.get("tags")
    if isinstance(tags, list):
        lowered = {str(tag).lower() for tag in tags if tag}
        if "sensor_incoherent" in lowered:
            return None
    keep_keys = {
        "id",
        "tags",
        "emotions",
        "summary",
        "tier",
        "modality",
        "audio_features",
        "image_features",
        "video_features",
        "timestamp",
        "source",
    }
    slim = {k: data.get(k) for k in keep_keys if k in data}
    # Preserve modality hints embedded in tags (e.g., "audio" / "vision") for clustering.
    return slim


def _iter_fragment_files(base: Path):
    for path in base.glob("frag_*.json"):
        yield path
    for tier in MEMORY_TIERS:
        tier_path = base / tier
        if tier_path.exists():
            for path in tier_path.glob("frag_*.json"):
                yield path


def load_fragments(child, limit: Optional[int] = None) -> Tuple[List[Dict[str, Any]], int]:
    base = Path("AI_Children") / child / "memory" / "fragments"
    limit_val = 0
    if limit is not None:
        try:
            limit_val = max(0, int(limit))
        except (TypeError, ValueError):
            limit_val = 0

    selected_paths: List[Path] = []
    total = 0
    if limit_val > 0:
        heap: List[tuple] = []
        for path in _iter_fragment_files(base):
            total += 1
            try:
                mtime = path.stat().st_mtime
            except Exception:
                continue
            entry = (mtime, path)
            if len(heap) < limit_val:
                heapq.heappush(heap, entry)
            else:
                if entry[0] > heap[0][0]:
                    heapq.heapreplace(heap, entry)
        heap.sort()
        selected_paths = [item[1] for item in heap]
    else:
        selected_paths = list(_iter_fragment_files(base))
        total = len(selected_paths)

    seen: Set[str] = set()

    def load(f: Path):
        try:
            with open(f, "r", encoding="utf-8") as file:
                data = json.load(file)
                if "tier" not in data:
                    parent_tier = f.parent.name
                    if parent_tier in MEMORY_TIERS:
                        data["tier"] = parent_tier
                slim = _slim_fragment(data)
                if slim and "emotions" in data:
                    frag_id = slim.get("id")
                    if frag_id and frag_id not in seen:
                        seen.add(frag_id)
                        return slim
        except Exception:
            return None
        return None

    with ThreadPoolExecutor(max_workers=8) as pool:
        fragments = [frag for frag in pool.map(load, selected_paths) if frag]
    return fragments, total

def cluster_fragments(fragments, cache, threshold=0.92, tag_weight=0.25):
    clusters = []
    tag_weight = max(0.0, min(1.0, tag_weight))
    for frag in fragments:
        frag_id = frag.get("id")
        if not frag_id:
            continue
        vec = cache.get(frag_id)
        if vec is None:
            continue

        frag_tags = set(frag.get("tags", []))
        best = None
        best_score = 0.0

        for node in clusters:
            node_vec = [
                val / node["count"] for val in node["vector_sum"]
            ]
            vec_score = cosine_similarity(vec, node_vec)
            tag_score = tag_similarity(frag_tags, node["tags"])
            score = ((1 - tag_weight) * vec_score) + (tag_weight * tag_score)
            if score >= threshold and score > best_score:
                best_score = score
                best = node

        if best:
            best["fragments"].append(frag_id)
            best["tags"].update(frag_tags)
            best["count"] += 1
            best["vector_sum"] = [
                a + b for a, b in zip(best["vector_sum"], vec)
            ]
        else:
            clusters.append({
                "fragments": [frag_id],
                "tags": set(frag_tags),
                "vector_sum": list(vec),
                "count": 1
            })
    return clusters

def build_synaptic_links(
    neurons,
    threshold=0.91,
    *,
    max_edges: Optional[int] = None,
    max_pairs: Optional[int] = None,
    max_edges_per_neuron: Optional[int] = None,
    include_direction: bool = True,
    return_stats: bool = False,
):
    synapses = []
    pairs_evaluated = 0
    truncated = False
    edge_cap = max_edges if (isinstance(max_edges, int) and max_edges > 0) else None
    pair_cap = max_pairs if (isinstance(max_pairs, int) and max_pairs > 0) else None
    per_node_cap = max_edges_per_neuron if (isinstance(max_edges_per_neuron, int) and max_edges_per_neuron > 0) else None

    for i, source in enumerate(neurons):
        source_edges = 0
        for j, target in enumerate(neurons):
            if j <= i:
                continue
            if per_node_cap is not None and source_edges >= per_node_cap:
                break
            if pair_cap is not None and pairs_evaluated >= pair_cap:
                truncated = True
                break
            if edge_cap is not None and len(synapses) >= edge_cap:
                truncated = True
                break
            pairs_evaluated += 1
            vec_a = source.get("vector")
            vec_b = target.get("vector")
            if vec_a and vec_b:
                sim = cosine_similarity(vec_a, vec_b)
                if sim >= threshold:
                    payload = {
                        "source": source["id"],
                        "target": target["id"],
                        "weight": round(sim, 4),
                        "network_type": "memory_graph",
                    }
                    if include_direction:
                        direction = None
                        pos_a = source.get("position")
                        pos_b = target.get("position")
                        if pos_a and pos_b:
                            delta = [pos_b[k] - pos_a[k] for k in range(3)]
                            norm = math.sqrt(sum(d * d for d in delta))
                            if norm > 1e-6:
                                direction = [round(d / norm, 5) for d in delta]
                        payload["direction"] = direction
                    synapses.append(payload)
                    source_edges += 1
        if truncated:
            break
    if return_stats:
        return synapses, {
            "pairs_evaluated": pairs_evaluated,
            "edge_count": len(synapses),
            "truncated": truncated,
        }
    return synapses


def _neural_map_path(child: str) -> Path:
    return Path("AI_Children") / child / "memory" / "neural" / "neural_memory_map.json"


def _load_neural_map(child: str) -> Dict[str, Any]:
    path = _neural_map_path(child)
    if not path.exists():
        return {"neurons": [], "synapses": [], "converted_from_legacy": False, "updated_at": None}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return {"neurons": [], "synapses": [], "converted_from_legacy": False, "updated_at": None}
    if "neurons" not in data or "synapses" not in data:
        data.setdefault("neurons", [])
        data.setdefault("synapses", [])
    return data


def _save_neural_map(child: str, payload: Dict[str, Any]) -> None:
    path = _neural_map_path(child)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=4)


def _neural_build_state_path(child: str) -> Path:
    return Path("AI_Children") / child / "memory" / "neural" / "neural_build_state.json"


def _neural_dirty_index_path(child: str) -> Path:
    return Path("AI_Children") / child / "memory" / "neural" / "neural_dirty_index.json"


def _neural_snapshot_path(child: str) -> Path:
    return Path("AI_Children") / child / "memory" / "neural" / "neural_memory_snapshot_csr.json"


def _custom_transformer_usage_path(child: str) -> Path:
    return Path("AI_Children") / child / "memory" / "neural" / "custom_transformer_usage.json"


def _load_json_dict(path: Path, default: Dict[str, Any]) -> Dict[str, Any]:
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except Exception:
        return default
    return payload if isinstance(payload, dict) else default


def _save_json_dict(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
    except Exception:
        return


class _NeuralCustomTransformerRuntime:
    def __init__(self, child: str, cfg: Optional[Dict[str, Any]]) -> None:
        self.child = child
        self.cfg = cfg if isinstance(cfg, dict) else {}
        self.policy = _custom_transformer_runtime_policy(self.cfg)
        self.enabled = bool(self.policy.get("enabled", True))
        self.sample_limit = int(self.policy.get("sample_limit", 24))
        self.cooldown_seconds = float(self.policy.get("cooldown_seconds", 300.0))
        self.soul_drift_steps = int(self.policy.get("soul_drift_steps", 1))
        self.run_hindsight = bool(self.policy.get("run_hindsight", True))
        self.sampled_fragments: List[Dict[str, Any]] = []
        self.sampled_tags: List[str] = []
        self.sampled_symbols: List[str] = []
        self.usage_path = _custom_transformer_usage_path(child)

    def observe(self, fragments: Iterable[Dict[str, Any]]) -> None:
        if not self.enabled:
            return
        for fragment in fragments:
            if len(self.sampled_fragments) >= self.sample_limit:
                break
            slim = _slim_fragment(fragment) or {}
            if not slim:
                continue
            self.sampled_fragments.append(slim)
            tags = slim.get("tags") or []
            if isinstance(tags, list):
                for tag in tags:
                    tag_text = str(tag).strip()
                    if tag_text:
                        self.sampled_tags.append(tag_text)
            for key in ("symbols", "symbols_spoken", "attempted_symbols"):
                value = fragment.get(key)
                if isinstance(value, list):
                    for item in value:
                        text = str(item).strip()
                        if text:
                            self.sampled_symbols.append(text)
                elif isinstance(value, str):
                    text = value.strip()
                    if text:
                        self.sampled_symbols.append(text)

    @staticmethod
    def _call_transformer(fn) -> Dict[str, Any]:
        try:
            result = fn()
            payload: Dict[str, Any] = {"status": "ok"}
            if isinstance(result, dict):
                payload.update(result)
            return payload
        except ModuleNotFoundError as exc:
            return {"status": "unavailable", "error": str(exc)}
        except ImportError as exc:
            return {"status": "unavailable", "error": str(exc)}
        except Exception as exc:
            return {"status": "error", "error": str(exc)}

    def _emotion_average(self) -> Dict[str, float]:
        aggregate: Dict[str, List[float]] = {}
        for fragment in self.sampled_fragments:
            emotions = fragment.get("emotions") or {}
            sliders = emotions.get("sliders") if isinstance(emotions.get("sliders"), dict) else emotions
            if not isinstance(sliders, dict):
                continue
            for key, value in sliders.items():
                if not isinstance(value, (int, float)):
                    continue
                aggregate.setdefault(str(key), []).append(float(value))
        averaged: Dict[str, float] = {}
        for key, values in aggregate.items():
            if values:
                averaged[key] = round(sum(values) / len(values), 4)
        return averaged

    def _run_seedling(self) -> Dict[str, Any]:
        module = importlib.import_module("transformers.seedling_transformer")
        cls = getattr(module, "SeedlingTransformer")
        symbols = self.sampled_symbols or self.sampled_tags
        if not symbols:
            return {"status": "skipped", "reason": "no_symbols"}
        seeded = cls(seed=0).germinate(symbols[:64])
        clusters = seeded.get("clusters", {}) if isinstance(seeded, dict) else {}
        seeds = seeded.get("seeds", {}) if isinstance(seeded, dict) else {}
        return {"clusters": len(clusters), "seeds": len(seeds)}

    def _run_mycelial(self, emotional_vector: Dict[str, float]) -> Dict[str, Any]:
        module = importlib.import_module("transformers.mycelial_transformer")
        cls = getattr(module, "MycelialTransformer")
        summaries: List[str] = []
        for fragment in self.sampled_fragments:
            summary = fragment.get("summary")
            if summary:
                summaries.append(str(summary))
        payload = {
            "tags": self.sampled_tags[:16],
            "fragments": [str(fragment.get("id")) for fragment in self.sampled_fragments if fragment.get("id")],
            "text": summaries[:12],
        }
        result = cls(max_links=2).weave(payload, emotional_vector=emotional_vector)
        pathways = result.get("pathways", []) if isinstance(result, dict) else []
        return {"pathways": len(pathways)}

    def _run_mirror(self, emotional_vector: Dict[str, float]) -> Dict[str, Any]:
        module = importlib.import_module("transformers.heuristic_mirror_transformer")
        cls = getattr(module, "HeuristicMirrorTransformer")
        transformer = cls(child=self.child)
        result = transformer.mirror({"tags": self.sampled_tags[:12]}, emotional_vector, perceived_audience="self")
        mirrored = result.get("mirrored_symbols", []) if isinstance(result, dict) else []
        return {"mirrored_symbols": len(mirrored)}

    def _run_bridge(self, emotional_vector: Dict[str, float]) -> Dict[str, Any]:
        module = importlib.import_module("transformers.bridge_transformer")
        cls = getattr(module, "BridgeTransformer")
        tags = [tag for tag in self.sampled_tags if tag]
        if len(tags) < 2:
            return {"status": "skipped", "reason": "insufficient_tags"}
        result = cls().bridge(tags[0], tags[1], emotional_vector)
        question = result.get("question") if isinstance(result, dict) else ""
        return {"question": str(question)[:120]}

    def _run_quantum(self, emotional_vector: Dict[str, float]) -> Dict[str, Any]:
        module = importlib.import_module("transformers.QTransformer")
        cls = getattr(module, "QTransformer")
        symbol_pool = self.sampled_symbols or self.sampled_tags or ["self"]
        symbol = str(symbol_pool[0])
        emotion_values = list(emotional_vector.values())[:24]
        if len(emotion_values) < 24:
            emotion_values.extend([0.0] * (24 - len(emotion_values)))
        result = cls().dream(symbol, emotion_values)
        tags = result.get("tags", []) if isinstance(result, dict) else []
        return {"tags": len(tags), "raw_bits": str(result.get("raw_bits", ""))}

    def _run_shadow(self) -> Dict[str, Any]:
        module = importlib.import_module("transformers.shadow_transformer")
        cls = getattr(module, "ShadowTransformer")
        transformer = cls()
        shadow_tags = {"suppressed", "unresolved", "high_conflict"}
        processed = 0
        for fragment in self.sampled_fragments:
            tags = {str(tag).lower() for tag in (fragment.get("tags") or []) if tag}
            if tags & shadow_tags:
                transformer.process_fragment(fragment)
                processed += 1
        return {"processed": processed, "index_size": len(getattr(transformer, "index", {}))}

    def _run_hindsight(self) -> Dict[str, Any]:
        if not self.run_hindsight:
            return {"status": "skipped", "reason": "disabled"}
        pred_path = Path("AI_Children") / self.child / "memory" / "prediction_log.json"
        predictions = []
        if pred_path.exists():
            try:
                with pred_path.open("r", encoding="utf-8") as fh:
                    payload = json.load(fh)
                    if isinstance(payload, list):
                        predictions = payload
            except Exception:
                predictions = []
        if len(predictions) < 2:
            return {"status": "skipped", "reason": "insufficient_predictions", "count": len(predictions)}
        module = importlib.import_module("transformers.hindsight_transformer")
        cls = getattr(module, "HindsightTransformer")
        transformer = cls()
        transformer.run()
        telemetry = transformer.get_intent_telemetry()
        return {
            "insights": int(telemetry.get("insights", 0) or 0),
            "trust": _safe_float(telemetry.get("trust"), 0.0),
        }

    def _run_soul_drift(self) -> Dict[str, Any]:
        module = importlib.import_module("transformers.soul_drift")
        drift_cfg_cls = getattr(module, "DriftConfig")
        drift_state_cls = getattr(module, "DriftState")
        transformer_cls = getattr(module, "SoulDriftTransformer")
        symbols = list(dict.fromkeys(self.sampled_symbols + self.sampled_tags))
        if not symbols:
            symbols = ["self", "memory"]
        symbols = symbols[:16]
        seed_weight = round(1.0 / max(1, len(symbols)), 6)
        weights = {symbol: seed_weight for symbol in symbols}
        links = {symbol: {} for symbol in symbols}
        try:
            import numpy as np  # local optional dependency
        except Exception as exc:
            return {"status": "unavailable", "error": str(exc)}
        cfg = drift_cfg_cls(log_history=False, max_history=64, rng_seed=0)
        state = drift_state_cls(
            step=0,
            symbol_weights=weights,
            symbol_links=links,
            emotion_vector=np.zeros(2, dtype=float),
            fuzz_level=0.2,
            entropy_score=0.0,
            tags_active=("neural_map",),
        )
        transformer = transformer_cls(cfg, state)
        transformer.run_session(self.soul_drift_steps, silence=True)
        telemetry = transformer.intent_telemetry()
        return {
            "entropy_bump": _safe_float(telemetry.get("entropy_bump"), 0.0),
            "fuzz_level": _safe_float(telemetry.get("fuzz_level"), 0.0),
        }

    def run(self) -> Dict[str, Any]:
        if not self.enabled:
            return {"status": "disabled"}
        if not self.sampled_fragments:
            return {"status": "skipped", "reason": "no_samples"}

        prior = _load_json_dict(self.usage_path, {})
        now_ts = time.time()
        last_run_ts = _parse_iso_timestamp(prior.get("last_run"))
        if last_run_ts is not None and self.cooldown_seconds > 0:
            elapsed = now_ts - last_run_ts
            if elapsed < self.cooldown_seconds:
                return {
                    "status": "cooldown",
                    "seconds_remaining": round(self.cooldown_seconds - elapsed, 2),
                }

        emotional_vector = self._emotion_average()
        report = {
            "seedling": self._call_transformer(lambda: self._run_seedling()),
            "mycelial": self._call_transformer(lambda: self._run_mycelial(emotional_vector)),
            "mirror": self._call_transformer(lambda: self._run_mirror(emotional_vector)),
            "bridge": self._call_transformer(lambda: self._run_bridge(emotional_vector)),
            "quantum": self._call_transformer(lambda: self._run_quantum(emotional_vector)),
            "shadow": self._call_transformer(lambda: self._run_shadow()),
            "hindsight": self._call_transformer(lambda: self._run_hindsight()),
            "soul_drift": self._call_transformer(lambda: self._run_soul_drift()),
        }

        ok_count = sum(1 for value in report.values() if value.get("status") == "ok")
        unavailable = [name for name, value in report.items() if value.get("status") == "unavailable"]
        errored = [name for name, value in report.items() if value.get("status") == "error"]
        skipped = [name for name, value in report.items() if value.get("status") == "skipped"]

        payload = {
            "last_run": datetime.now(timezone.utc).isoformat(),
            "sample_count": len(self.sampled_fragments),
            "tag_count": len(self.sampled_tags),
            "symbol_count": len(self.sampled_symbols),
            "ok_count": ok_count,
            "unavailable": unavailable,
            "errored": errored,
            "skipped": skipped,
            "report": report,
        }
        _save_json_dict(self.usage_path, payload)
        _write_inastate_value(
            self.child,
            "custom_transformers_usage",
            {
                "last_run": payload["last_run"],
                "ok_count": ok_count,
                "unavailable": unavailable,
                "errored": errored,
                "skipped": skipped,
            },
        )
        if unavailable or errored:
            log_to_statusbox(
                f"[NeuralMap] Custom transformers: ok={ok_count} unavailable={len(unavailable)} error={len(errored)} skipped={len(skipped)}."
            )
        else:
            log_to_statusbox(f"[NeuralMap] Custom transformers active ({ok_count}/{len(report)}).")
        return payload


def _load_neural_build_state(child: str) -> Dict[str, Any]:
    data = _load_json_dict(_neural_build_state_path(child), {"pending": [], "updated_at": None})
    pending_raw = data.get("pending")
    if isinstance(pending_raw, list):
        data["pending"] = [str(fid) for fid in pending_raw if fid]
    else:
        data["pending"] = []
    return data


def _save_neural_build_state(child: str, state: Dict[str, Any]) -> None:
    payload = dict(state)
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    _save_json_dict(_neural_build_state_path(child), payload)


def _load_neural_dirty_index(child: str) -> Dict[str, str]:
    data = _load_json_dict(_neural_dirty_index_path(child), {"fragments": {}})
    fragments = data.get("fragments") if isinstance(data.get("fragments"), dict) else {}
    return {str(fid): str(sig) for fid, sig in fragments.items() if fid and sig}


def _save_neural_dirty_index(child: str, signatures: Dict[str, str]) -> None:
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "fragments": signatures,
    }
    _save_json_dict(_neural_dirty_index_path(child), payload)


def _load_memory_index(child: str) -> Dict[str, Any]:
    path = Path("AI_Children") / child / "memory" / "memory_map.json"
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _fragment_signature(child: str, frag_id: str, meta: Dict[str, Any]) -> Optional[str]:
    path = _resolve_index_path(child, frag_id, meta)
    if path is None:
        return None
    try:
        st = path.stat()
    except OSError:
        return None
    tags = meta.get("tags") if isinstance(meta.get("tags"), list) else []
    tag_text = ",".join(sorted(str(tag).lower() for tag in tags if tag))
    digest = hashlib.sha1(tag_text.encode("utf-8")).hexdigest()[:12]
    return (
        f"{meta.get('tier') or ''}:{meta.get('filename') or ''}:"
        f"{int(st.st_mtime_ns)}:{int(st.st_size)}:"
        f"{_safe_float(meta.get('importance'), 0.0):.6f}:{digest}"
    )


def _collect_dirty_fragment_ids(
    child: str,
    index: Dict[str, Any],
    previous_signatures: Dict[str, str],
) -> Tuple[List[str], Dict[str, str], Set[str]]:
    dirty: List[str] = []
    current: Dict[str, str] = {}
    for frag_id, meta in index.items():
        if not frag_id or not isinstance(meta, dict):
            continue
        signature = _fragment_signature(child, frag_id, meta)
        if not signature:
            continue
        current[frag_id] = signature
        if previous_signatures.get(frag_id) != signature:
            dirty.append(frag_id)
    removed = {fid for fid in previous_signatures.keys() if fid not in current}
    return dirty, current, removed


def _prune_neurons_to_valid_fragments(
    neurons: List[Dict[str, Any]],
    valid_ids: Set[str],
) -> Tuple[int, int]:
    pruned_refs = 0
    pruned_neurons = 0
    kept: List[Dict[str, Any]] = []
    for neuron in neurons:
        original = [str(fid) for fid in neuron.get("fragments", []) if fid]
        current = [fid for fid in original if fid in valid_ids]
        pruned_refs += max(0, len(original) - len(current))
        if not current:
            pruned_neurons += 1
            continue
        neuron["fragments"] = current
        kept.append(neuron)
    if pruned_neurons:
        neurons[:] = kept
    return pruned_refs, pruned_neurons


def _apply_neuron_caps(neurons: List[Dict[str, Any]], policy: Dict[str, Any]) -> int:
    if not neurons:
        return 0
    frag_limit = int(policy.get("max_fragments_per_neuron", 128))
    tag_limit = int(policy.get("max_tags_per_neuron", 32))
    vector_digits = int(policy.get("vector_round_digits", 6))
    position_digits = int(policy.get("position_round_digits", 4))
    changes = 0
    for neuron in neurons:
        original_fragments = list(neuron.get("fragments", []))
        clipped_fragments = _clip_sequence(original_fragments, frag_limit)
        if clipped_fragments != original_fragments:
            neuron["fragments"] = clipped_fragments
            changes += 1

        original_tags = [str(tag) for tag in neuron.get("tags", []) if tag]
        unique_tags = sorted(set(original_tags))
        clipped_tags = unique_tags[-tag_limit:] if tag_limit > 0 else unique_tags
        if clipped_tags != original_tags:
            neuron["tags"] = clipped_tags
            changes += 1

        rounded_vec = _round_vector(neuron.get("vector"), vector_digits)
        if rounded_vec and rounded_vec != neuron.get("vector"):
            neuron["vector"] = rounded_vec
            changes += 1

        rounded_pos = _round_vector(neuron.get("position"), position_digits)
        if rounded_pos and rounded_pos != neuron.get("position"):
            neuron["position"] = rounded_pos
            changes += 1
    return changes


def _enforce_neuron_budget(neurons: List[Dict[str, Any]], max_neurons_total: int) -> int:
    if max_neurons_total <= 0:
        return 0
    if len(neurons) <= max_neurons_total:
        return 0
    overflow = len(neurons) - max_neurons_total
    scored: List[Tuple[int, float, int]] = []
    for idx, neuron in enumerate(neurons):
        fragment_count = len(neuron.get("fragments", []) or [])
        last_used_ts = _parse_iso_timestamp(neuron.get("last_used")) or 0.0
        scored.append((fragment_count, last_used_ts, idx))
    scored.sort(key=lambda item: (item[0], item[1], item[2]))
    drop_indices = {idx for _, _, idx in scored[:overflow]}
    neurons[:] = [neuron for idx, neuron in enumerate(neurons) if idx not in drop_indices]
    return overflow


def _detach_neurons_for_dirty_rebuild(
    neurons: List[Dict[str, Any]],
    dirty_ids: Set[str],
    *,
    max_fragments: int,
) -> Tuple[Set[str], int, bool]:
    if not dirty_ids:
        return set(), 0, False
    affected_indices: List[int] = []
    rebuild_ids: Set[str] = set()
    for idx, neuron in enumerate(neurons):
        frag_ids = [str(fid) for fid in neuron.get("fragments", []) if fid]
        if not frag_ids:
            continue
        if any(fid in dirty_ids for fid in frag_ids):
            affected_indices.append(idx)
            rebuild_ids.update(frag_ids)
            if max_fragments > 0 and len(rebuild_ids) > max_fragments:
                return set(), 0, True
    if not affected_indices:
        return set(), 0, False
    drop_set = set(affected_indices)
    kept = [neuron for idx, neuron in enumerate(neurons) if idx not in drop_set]
    neurons[:] = kept
    return rebuild_ids, len(affected_indices), False


def _load_fragments_by_id(
    child: str,
    fragment_ids: List[str],
    index: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    loaded: List[Dict[str, Any]] = []
    missing: List[str] = []
    for frag_id in fragment_ids:
        meta = index.get(frag_id)
        if not isinstance(meta, dict):
            missing.append(frag_id)
            continue
        path = _resolve_index_path(child, frag_id, meta)
        if path is None:
            missing.append(frag_id)
            continue
        fragment = _load_fragment_from_path(path)
        if not fragment:
            missing.append(frag_id)
            continue
        slim = _slim_fragment(fragment)
        if slim:
            loaded.append(slim)
        else:
            missing.append(frag_id)
    return loaded, missing


def _save_sparse_snapshot(child: str, neurons: List[Dict[str, Any]], synapses: List[Dict[str, Any]]) -> None:
    node_ids = [str(neuron.get("id")) for neuron in neurons if neuron.get("id")]
    node_to_index = {node_id: idx for idx, node_id in enumerate(node_ids)}
    rows: Dict[int, List[Tuple[int, float]]] = {}
    for synapse in synapses:
        source = node_to_index.get(str(synapse.get("source")))
        target = node_to_index.get(str(synapse.get("target")))
        if source is None or target is None:
            continue
        weight = _safe_float(synapse.get("weight"), 0.0)
        rows.setdefault(source, []).append((target, round(weight, 4)))
    indptr: List[int] = [0]
    indices: List[int] = []
    weights: List[float] = []
    for row_idx in range(len(node_ids)):
        edges = rows.get(row_idx, [])
        edges.sort(key=lambda item: item[0])
        for target_idx, weight in edges:
            indices.append(int(target_idx))
            weights.append(weight)
        indptr.append(len(indices))
    payload = {
        "format": "csr_v1",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "node_ids": node_ids,
        "indptr": indptr,
        "indices": indices,
        "weights": weights,
        "edge_count": len(indices),
    }
    _save_json_dict(_neural_snapshot_path(child), payload)


def _existing_fragment_ids(neurons: List[Dict[str, Any]]) -> Set[str]:
    known: Set[str] = set()
    for neuron in neurons:
        for frag_id in neuron.get("fragments", []):
            known.add(frag_id)
    return known


def _node_id_allocator(neurons: List[Dict[str, Any]]):
    prefix = "node_"
    max_idx = -1
    for neuron in neurons:
        node_id = str(neuron.get("id") or "")
        if node_id.startswith(prefix):
            try:
                idx = int(node_id[len(prefix):])
                max_idx = max(max_idx, idx)
            except ValueError:
                continue
    counter = max_idx + 1
    while True:
        yield f"{prefix}{counter:04}"
        counter += 1


def _blend_position(old: Optional[List[float]], new: Optional[List[float]], blend: float) -> Optional[List[float]]:
    if not new and not old:
        return None
    if not old:
        return list(new)
    if not new:
        return list(old)
    blend = max(0.0, min(1.0, blend))
    return [
        old[i] + (new[i] - old[i]) * blend for i in range(min(len(old), len(new)))
    ]


def _round_vector(values: Optional[Iterable[Any]], digits: int) -> List[float]:
    if values is None:
        return []
    rounded: List[float] = []
    for value in values:
        try:
            rounded.append(round(float(value), digits))
        except (TypeError, ValueError):
            rounded.append(0.0)
    return rounded


def _clip_sequence(values: Iterable[Any], limit: int) -> List[str]:
    clipped = [str(value) for value in values if value]
    if limit > 0 and len(clipped) > limit:
        return clipped[-limit:]
    return clipped


def _merge_vectors(
    base: List[float],
    base_count: int,
    new_vec: List[float],
    new_count: int,
    *,
    round_digits: int = 6,
) -> List[float]:
    if not base_count:
        return [round(v, round_digits) for v in new_vec]
    if not new_count:
        return [round(v, round_digits) for v in base]
    length = min(len(base), len(new_vec))
    merged = []
    total = base_count + new_count
    for i in range(length):
        merged.append(round((base[i] * base_count + new_vec[i] * new_count) / total, round_digits))
    return merged


def _materialize_candidate(node_id: str, candidate: Dict[str, Any], policy: Dict[str, Any]) -> Dict[str, Any]:
    now_iso = datetime.now(timezone.utc).isoformat()
    frag_limit = int(policy.get("max_fragments_per_neuron", 128))
    tag_limit = int(policy.get("max_tags_per_neuron", 32))
    vector_digits = int(policy.get("vector_round_digits", 6))
    pos_digits = int(policy.get("position_round_digits", 4))
    return {
        "id": node_id,
        "fragments": _clip_sequence(candidate["fragments"], frag_limit),
        "vector": _round_vector(candidate.get("vector"), vector_digits),
        "position": _round_vector(candidate.get("position"), pos_digits),
        "region": candidate["region"],
        "network_type": "memory_graph",
        "symbolic_density": 0.0,
        "tags": _clip_sequence(candidate["tags"], tag_limit),
        "activation_history": [],
        "last_used": now_iso,
    }


def _update_neuron_from_candidate(neuron: Dict[str, Any], candidate: Dict[str, Any], policy: Dict[str, Any]) -> None:
    existing_frags = neuron.setdefault("fragments", [])
    new_ids = [fid for fid in candidate["fragments"] if fid not in existing_frags]
    if new_ids:
        existing_frags.extend(new_ids)
    max_fragments_per_neuron = int(policy.get("max_fragments_per_neuron", 128))
    if max_fragments_per_neuron > 0 and len(existing_frags) > max_fragments_per_neuron:
        neuron["fragments"] = _clip_sequence(existing_frags, max_fragments_per_neuron)
        existing_frags = neuron["fragments"]
    retained_new = len([fid for fid in new_ids if fid in existing_frags])
    base_vec = neuron.get("vector")
    base_count = (max(0, len(existing_frags) - retained_new) if base_vec else 0)
    base_vec = base_vec or candidate["vector"]
    new_count = retained_new
    combined_vec = _merge_vectors(
        base_vec,
        base_count,
        candidate["vector"],
        new_count,
        round_digits=int(policy.get("vector_round_digits", 6)),
    )
    neuron["vector"] = combined_vec
    position = _blend_position(neuron.get("position"), candidate["position"], policy["position_blend"]) or candidate["position"]
    neuron["position"] = _round_vector(position, int(policy.get("position_round_digits", 4)))
    tag_union = set(neuron.get("tags", []))
    tag_union.update(candidate["tags"])
    sorted_tags = sorted(tag_union)
    tag_limit = int(policy.get("max_tags_per_neuron", 32))
    neuron["tags"] = sorted_tags[-tag_limit:] if tag_limit > 0 else sorted_tags
    if not neuron.get("region"):
        neuron["region"] = candidate["region"]
    neuron["last_used"] = datetime.now(timezone.utc).isoformat()


def _score_candidate_match(neuron: Dict[str, Any], candidate: Dict[str, Any], tag_weight: float) -> float:
    vec_a = neuron.get("vector")
    vec_b = candidate["vector"]
    if not vec_a or not vec_b:
        return 0.0
    vec_score = cosine_similarity(vec_a, vec_b)
    tag_score = tag_similarity(neuron.get("tags", []), candidate["tags"])
    tag_weight = max(0.0, min(1.0, tag_weight))
    return ((1 - tag_weight) * vec_score) + (tag_weight * tag_score)


def _prepare_candidates(clusters: List[Dict[str, Any]], cache: Dict[str, List[float]], anchors: Dict[str, Dict[str, float]], fallback_anchor: Dict[str, Any]):
    candidates: List[Dict[str, Any]] = []
    for group in clusters:
        fragment_ids = group["fragments"]
        if not fragment_ids:
            continue
        vector_sum = group.get("vector_sum")
        count = group.get("count", len(fragment_ids))
        if vector_sum and count:
            avg_vec = [round(v / count, 6) for v in vector_sum]
        else:
            avg_vec = vector_average([cache[fid] for fid in fragment_ids if fid in cache])
        if not avg_vec:
            continue
        tags = sorted(group["tags"]) if isinstance(group.get("tags"), set) else list(group.get("tags", []))
        region = _guess_region_from_tags(tags, anchors)
        anchor = anchors.get(region, fallback_anchor)
        position = _project_vector_to_anchor(avg_vec, anchor, seed=fragment_ids[0])
        candidates.append({
            "fragments": fragment_ids,
            "tags": tags,
            "vector": avg_vec,
            "region": region,
            "position": position,
        })
    return candidates


def _merge_candidates_into_neurons(
    neurons: List[Dict[str, Any]],
    candidates: List[Dict[str, Any]],
    cluster_threshold: float,
    tag_weight: float,
    policy: Dict[str, Any],
) -> Tuple[int, int, int]:
    if not candidates:
        return 0, 0, 0
    merge_threshold = max(0.0, min(1.0, cluster_threshold - policy.get("merge_slack", 0.0)))
    merged = 0
    created = 0
    skipped = 0
    id_allocator = _node_id_allocator(neurons)
    for candidate in candidates:
        best = None
        best_score = 0.0
        for neuron in neurons:
            score = _score_candidate_match(neuron, candidate, tag_weight)
            if score > best_score:
                best_score = score
                best = neuron
        if best and best_score >= merge_threshold:
            _update_neuron_from_candidate(best, candidate, policy)
            merged += 1
            continue
        if created < policy.get("max_new_neurons", 0):
            node_id = next(id_allocator)
            neurons.append(_materialize_candidate(node_id, candidate, policy))
            created += 1
            continue
        skipped += 1
    return merged, created, skipped


def _integrate_fragment_batch(
    fragments: List[Dict[str, Any]],
    transformer: "FractalTransformer",
    neurons: List[Dict[str, Any]],
    *,
    incremental: bool,
    cluster_threshold: float,
    tag_weight: float,
    policy: Dict[str, Any],
    anchors: Dict[str, Dict[str, float]],
    fallback_anchor: Dict[str, Any],
) -> Dict[str, int]:
    if not fragments:
        return {"input": 0, "encoded": 0, "clusters": 0, "merged": 0, "created": 0, "skipped": 0}
    encoded = transformer.encode_many(fragments)
    if not encoded:
        return {"input": len(fragments), "encoded": 0, "clusters": 0, "merged": 0, "created": 0, "skipped": 0}
    vector_digits = int(policy.get("vector_round_digits", 6))
    cache: Dict[str, List[float]] = {}
    for item in encoded:
        frag_id = item.get("id")
        vector = item.get("vector")
        if frag_id and vector:
            cache[str(frag_id)] = _round_vector(vector, vector_digits)
    encoded_count = len(cache)
    del encoded
    if not cache:
        return {"input": len(fragments), "encoded": 0, "clusters": 0, "merged": 0, "created": 0, "skipped": 0}
    clusters = cluster_fragments(
        fragments,
        cache,
        threshold=cluster_threshold,
        tag_weight=tag_weight,
    )
    candidates = _prepare_candidates(clusters, cache, anchors, fallback_anchor)
    if incremental:
        merged, created, skipped = _merge_candidates_into_neurons(
            neurons,
            candidates,
            cluster_threshold,
            tag_weight,
            policy,
        )
    else:
        merged = 0
        created = 0
        skipped = 0
        id_allocator = _node_id_allocator(neurons)
        for candidate in candidates:
            neurons.append(_materialize_candidate(next(id_allocator), candidate, policy))
            created += 1
        merged = created
    return {
        "input": len(fragments),
        "encoded": encoded_count,
        "clusters": len(clusters),
        "merged": merged,
        "created": created,
        "skipped": skipped,
    }


def build_fractal_memory(child):
    start_time = datetime.now()
    start_perf = time.perf_counter()
    from transformers.fractal_multidimensional_transformers import FractalTransformer

    cluster_threshold, synapse_threshold, tag_weight = _neural_settings()
    cfg = _load_config()
    policy = _neural_policy(cfg)
    custom_runtime = _NeuralCustomTransformerRuntime(child, cfg)
    burst_limit = policy.get("fragment_batch")
    if burst_limit is None:
        burst_limit = cfg.get("neural_map_burst")
    try:
        burst_limit = int(burst_limit) if burst_limit is not None else NEURAL_MAP_BURST_DEFAULT
    except (TypeError, ValueError):
        burst_limit = NEURAL_MAP_BURST_DEFAULT
    if burst_limit <= 0:
        burst_limit = NEURAL_MAP_BURST_DEFAULT
    batch_size = max(1, min(int(policy.get("batch_size", 24)), burst_limit))
    build_budget_ms = max(0.0, _safe_float(policy.get("build_budget_ms"), 0.0))

    transformer = FractalTransformer()
    incremental = policy.get("incremental", True)
    existing_map = _load_neural_map(child) if incremental else {"neurons": [], "synapses": [], "converted_from_legacy": False}
    neurons = existing_map.get("neurons", []) if incremental else []
    if not isinstance(neurons, list):
        neurons = []
    neuron_cap_changes = _apply_neuron_caps(neurons, policy) if incremental else 0
    neuron_budget_pruned = _enforce_neuron_budget(neurons, int(policy.get("max_neurons_total", 0))) if incremental else 0
    known_fragments = _existing_fragment_ids(neurons) if incremental else set()

    index = _load_memory_index(child)
    valid_ids = set(index.keys()) if isinstance(index, dict) else set()

    pruned_refs = 0
    pruned_neurons = 0
    if incremental and valid_ids:
        pruned_refs, pruned_neurons = _prune_neurons_to_valid_fragments(neurons, valid_ids)
        if pruned_refs or pruned_neurons or neuron_cap_changes or neuron_budget_pruned:
            known_fragments = _existing_fragment_ids(neurons)
            log_to_statusbox(
                f"[NeuralMap] Pruned {pruned_refs} stale refs, dropped {pruned_neurons} empty neuron(s), budget-pruned {neuron_budget_pruned}."
            )
    elif neuron_cap_changes or neuron_budget_pruned:
        log_to_statusbox(
            f"[NeuralMap] Applied payload caps ({neuron_cap_changes} updates) and budget-pruned {neuron_budget_pruned} neuron(s)."
        )

    anchors = _load_body_anchors()
    fallback_anchor = anchors.get("head", {"center": [0.0, 0.0, 0.0], "radius": 2.0})

    dirty_mode = bool(policy.get("dirty_index_enabled", True) and incremental and index)
    queue_state = _load_neural_build_state(child) if dirty_mode else {"pending": []}
    pending_ids = [fid for fid in queue_state.get("pending", []) if fid in valid_ids] if dirty_mode else []
    dirty_signatures = _load_neural_dirty_index(child) if dirty_mode else {}
    current_signatures: Dict[str, str] = {}
    removed_signatures: Set[str] = set()
    processed_ids: List[str] = []
    detached_neurons = 0
    detached_overflow = False
    detected_dirty = 0
    source = "selector"
    selector_meta: Dict[str, Any] = {}
    selector_total = 0
    selector_loaded = 0
    selector_targets: List[Dict[str, Any]] = []

    if dirty_mode:
        source = "dirty_index"
        dirty_ids, current_signatures, removed_signatures = _collect_dirty_fragment_ids(child, index, dirty_signatures)
        detected_dirty = len(dirty_ids)
        dirty_set = set(dirty_ids)
        if dirty_set:
            known_dirty = dirty_set & known_fragments
            if known_dirty:
                local_rebuild_ids, detached_neurons, detached_overflow = _detach_neurons_for_dirty_rebuild(
                    neurons,
                    known_dirty,
                    max_fragments=int(policy.get("local_rebuild_max_fragments", 0)),
                )
                if detached_overflow:
                    log_to_statusbox(
                        "[NeuralMap] Dirty-set touched too many existing neurons; keeping current map and deferring local rebuild."
                    )
                else:
                    dirty_set.update(local_rebuild_ids)
                    known_fragments = _existing_fragment_ids(neurons)
            pending_set = set(pending_ids)
            for frag_id in dirty_set:
                if frag_id in valid_ids and frag_id not in pending_set:
                    pending_ids.append(frag_id)
                    pending_set.add(frag_id)
        if pending_ids:
            def _queue_sort_key(frag_id: str) -> Tuple[float, float]:
                meta = index.get(frag_id, {}) if isinstance(index, dict) else {}
                ts = _parse_iso_timestamp(meta.get("last_seen") or meta.get("timestamp")) or 0.0
                importance = _safe_float(meta.get("importance"), 0.0)
                return ts, importance
            pending_ids = sorted(dict.fromkeys(pending_ids), key=_queue_sort_key, reverse=True)
        queue_state["last_source"] = source
    else:
        fragments, selector_total, selector_meta = select_fragments_for_neural_map(
            child,
            burst_limit,
            cfg=cfg,
            known_fragments=known_fragments if incremental else None,
        )
        selector_loaded = len(fragments)
        if selector_total > selector_loaded:
            log_to_statusbox(
                f"[NeuralMap] Limiting to {selector_loaded} selected fragments (burst={burst_limit}, total={selector_total})."
            )
        elif selector_loaded:
            log_to_statusbox(f"[NeuralMap] Loaded {selector_loaded} fragments.")
        if selector_meta.get("mode") not in {"fallback", "disabled"}:
            log_to_statusbox(
                f"[NeuralMap] Selector lanes={selector_meta.get('lane_counts')} temp={selector_meta.get('temperature')} mode={selector_meta.get('mode')}."
            )
        selector_targets = [frag for frag in fragments if not incremental or frag.get("id") not in known_fragments]

    totals = {"input": 0, "encoded": 0, "clusters": 0, "merged": 0, "created": 0, "skipped": 0}
    batches_run = 0
    budget_hit = False

    if source == "dirty_index":
        while pending_ids and totals["input"] < burst_limit:
            elapsed_ms = (time.perf_counter() - start_perf) * 1000.0
            if build_budget_ms > 0 and elapsed_ms >= build_budget_ms and totals["input"] > 0:
                budget_hit = True
                break
            step = min(batch_size, burst_limit - totals["input"], len(pending_ids))
            batch_ids = pending_ids[:step]
            pending_ids = pending_ids[step:]
            batch_fragments, missing_ids = _load_fragments_by_id(child, batch_ids, index)
            if missing_ids:
                for frag_id in missing_ids:
                    current_signatures.pop(frag_id, None)
                    dirty_signatures.pop(frag_id, None)
            if not batch_fragments:
                continue
            stats = _integrate_fragment_batch(
                batch_fragments,
                transformer,
                neurons,
                incremental=incremental,
                cluster_threshold=cluster_threshold,
                tag_weight=tag_weight,
                policy=policy,
                anchors=anchors,
                fallback_anchor=fallback_anchor,
            )
            for key in totals:
                totals[key] += stats.get(key, 0)
            processed_ids.extend([str(frag.get("id")) for frag in batch_fragments if frag.get("id")])
            custom_runtime.observe(batch_fragments)
            dropped = _enforce_neuron_budget(neurons, int(policy.get("max_neurons_total", 0)))
            if dropped:
                neuron_budget_pruned += dropped
                known_fragments = _existing_fragment_ids(neurons)
            batches_run += 1
            gc_every_batches = int(policy.get("gc_every_batches", 0))
            if gc_every_batches > 0 and (batches_run % gc_every_batches) == 0:
                gc.collect()
    else:
        cursor = 0
        while cursor < len(selector_targets) and totals["input"] < burst_limit:
            elapsed_ms = (time.perf_counter() - start_perf) * 1000.0
            if build_budget_ms > 0 and elapsed_ms >= build_budget_ms and totals["input"] > 0:
                budget_hit = True
                break
            step = min(batch_size, burst_limit - totals["input"], len(selector_targets) - cursor)
            batch_fragments = selector_targets[cursor : cursor + step]
            cursor += step
            if not batch_fragments:
                continue
            stats = _integrate_fragment_batch(
                batch_fragments,
                transformer,
                neurons,
                incremental=incremental,
                cluster_threshold=cluster_threshold,
                tag_weight=tag_weight,
                policy=policy,
                anchors=anchors,
                fallback_anchor=fallback_anchor,
            )
            for key in totals:
                totals[key] += stats.get(key, 0)
            custom_runtime.observe(batch_fragments)
            dropped = _enforce_neuron_budget(neurons, int(policy.get("max_neurons_total", 0)))
            if dropped:
                neuron_budget_pruned += dropped
                known_fragments = _existing_fragment_ids(neurons)
            batches_run += 1
            gc_every_batches = int(policy.get("gc_every_batches", 0))
            if gc_every_batches > 0 and (batches_run % gc_every_batches) == 0:
                gc.collect()

    custom_usage = custom_runtime.run()

    if dirty_mode:
        for frag_id in removed_signatures:
            dirty_signatures.pop(frag_id, None)
        for frag_id in processed_ids:
            sig = current_signatures.get(frag_id)
            if sig:
                dirty_signatures[frag_id] = sig
        queue_state["pending"] = pending_ids
        queue_state["dirty_detected"] = detected_dirty
        queue_state["processed"] = len(processed_ids)
        queue_state["detached_neurons"] = detached_neurons
        _save_neural_build_state(child, queue_state)
        _save_neural_dirty_index(child, dirty_signatures)
        log_to_statusbox(
            f"[NeuralMap] Dirty queue: detected {detected_dirty}, processed {len(processed_ids)}, remaining {len(pending_ids)}."
        )

    map_changed = bool(
        pruned_refs
        or pruned_neurons
        or neuron_cap_changes
        or neuron_budget_pruned
        or detached_neurons
        or totals["merged"]
        or totals["created"]
    )
    queue_remaining = len(pending_ids) if dirty_mode else 0
    needs_synapse_refresh = False
    if not incremental:
        needs_synapse_refresh = True
    elif queue_remaining == 0 and map_changed:
        needs_synapse_refresh = True
    elif queue_remaining == 0 and totals["input"] == 0 and policy.get("synapse_refresh_on_idle", True):
        needs_synapse_refresh = True

    prior_synapses = existing_map.get("synapses", []) if incremental else []
    if not isinstance(prior_synapses, list):
        prior_synapses = []
    synapses = prior_synapses
    synapse_stats = {"pairs_evaluated": 0, "edge_count": len(synapses), "truncated": False}
    if needs_synapse_refresh:
        synapses, synapse_stats = build_synaptic_links(
            neurons,
            threshold=synapse_threshold,
            max_edges=int(policy.get("max_edges_updated", 0) or 0),
            max_pairs=int(policy.get("max_synapse_pairs", 0) or 0),
            max_edges_per_neuron=int(policy.get("max_edges_per_neuron", 0) or 0),
            include_direction=bool(policy.get("edge_direction_enabled", True)),
            return_stats=True,
        )
        if synapse_stats.get("truncated"):
            log_to_statusbox(
                f"[NeuralMap] Synapse refresh hit budget ({synapse_stats.get('edge_count')} edges, {synapse_stats.get('pairs_evaluated')} pairs)."
            )

    should_save_map = bool(not incremental or map_changed or needs_synapse_refresh)
    if should_save_map:
        result = existing_map if incremental else {}
        result.update({
            "neurons": neurons,
            "synapses": synapses,
            "converted_from_legacy": existing_map.get("converted_from_legacy", False) if incremental else False,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })
        _save_neural_map(child, result)
        if policy.get("emit_sparse_snapshot", True):
            _save_sparse_snapshot(child, neurons, synapses)

    if source == "selector" and selector_loaded and not selector_targets and incremental:
        log_to_statusbox("[NeuralMap] No new fragments detected after selector filtering.")
    if totals["clusters"] > 0:
        log_to_statusbox(
            f"[NeuralMap] Clustered {totals['encoded']} fragment(s) into {totals['clusters']} cluster(s) "
            f"(threshold={cluster_threshold:.2f}, tag_weight={tag_weight:.2f})."
        )
    if incremental and (totals["merged"] or totals["created"] or totals["skipped"]):
        log_to_statusbox(
            f"[NeuralMap] Incremental update — merged {totals['merged']}, added {totals['created']}, "
            f"skipped {totals['skipped']} (max_new={policy.get('max_new_neurons')})."
        )
    if budget_hit:
        log_to_statusbox(
            f"[NeuralMap] Build budget reached after {totals['input']} fragment(s) and {batches_run} batch(es); resuming next cycle."
        )
    if neuron_budget_pruned:
        log_to_statusbox(
            f"[NeuralMap] Enforced global neuron budget; pruned {neuron_budget_pruned} node(s) this cycle."
        )
    if custom_usage.get("status") == "cooldown":
        log_to_statusbox(
            f"[NeuralMap] Custom transformer runtime cooling down ({custom_usage.get('seconds_remaining')}s remaining)."
        )
    if totals["input"] == 0 and not map_changed and not needs_synapse_refresh:
        if source == "dirty_index":
            log_to_statusbox("[NeuralMap] No dirty fragments to process.")
        else:
            log_to_statusbox("[NeuralMap] No fragments available for neural map build.")

    duration = datetime.now() - start_time
    log_to_statusbox(
        f"[NeuralMap] {len(neurons)} neurons | {len(synapses)} synapses | "
        f"Policy={policy.get('mode')} | Source={source} | Batches={batches_run}."
    )
    log_to_statusbox(f"[NeuralMap] Mapping time: {duration}.")

# === MemoryManager Class ===
class MemoryManager:
    def __init__(self, child="Inazuma_Yagami", tier_policy: Optional[Dict[str, Any]] = None):
        self.child = child
        self.base_path = Path("AI_Children") / child / "memory" / "fragments"
        self.index_path = Path("AI_Children") / child / "memory" / "memory_map.json"
        self.memory_map = {}
        self.policy = tier_policy or _memory_policy()
        self.cold_storage_policy = _cold_storage_policy()
        self.load_map()

    @staticmethod
    def _stat_payload(path: Path) -> Dict[str, Any]:
        try:
            st = path.stat()
        except OSError:
            return {}
        return {"mtime_ns": int(st.st_mtime_ns), "size_bytes": int(st.st_size)}

    def ensure_tier_directories(self):
        for tier in MEMORY_TIERS:
            (self.base_path / tier).mkdir(parents=True, exist_ok=True)

    def index_legacy_root(self):
        """
        Legacy fragments lived directly under memory/fragments without tier folders.
        Scan them so counts are accurate even if tiers are unused.
        """
        root_files = list(self.base_path.glob("frag_*.json"))
        for frag in root_files:
            try:
                with open(frag, "r") as f:
                    data = json.load(f)
                existing = self.memory_map.get(data["id"], {})
                self.memory_map[data["id"]] = {
                    "tier": existing.get("tier", "short"),
                    "tags": data.get("tags", []),
                    "importance": data.get("importance", 0),
                    "last_seen": existing.get(
                        "last_seen",
                        data.get("timestamp", datetime.now(timezone.utc).isoformat())
                    ),
                    "filename": frag.name,
                    **self._stat_payload(frag),
                }
            except Exception:
                continue

    def load_map(self):
        if self.index_path.exists():
            try:
                with open(self.index_path, "r") as f:
                    self.memory_map = json.load(f)
            except:
                self.memory_map = {}
        else:
            self.memory_map = {}

    def save_map(self):
        with open(self.index_path, "w") as f:
            json.dump(self.memory_map, f, indent=2)

    def _resolve_fragment_path(self, frag_id: str, meta: Dict[str, Any]) -> Optional[Path]:
        """
        Resolve the on-disk path for a fragment given its metadata entry.
        """
        filename = meta.get("filename", f"frag_{frag_id}.json")
        tier = meta.get("tier")
        candidates = []
        if tier:
            candidates.append(self.base_path / tier / filename)
        candidates.append(self.base_path / filename)
        for path in candidates:
            if path.exists():
                return path
        return None

    def _compact_cold_fragment(
        self,
        frag_id: str,
        meta: Dict[str, Any],
        path: Path,
        cold_policy: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        try:
            import cold_storage  # type: ignore
        except Exception:
            return None

        if not cold_policy.get("enabled", True):
            return None

        if cold_policy.get("require_index_entry", True):
            if frag_id not in self.memory_map:
                return {"fragment_id": frag_id, "status": "skipped", "reason": "index_missing"}
            resolved = self._resolve_fragment_path(frag_id, meta)
            if resolved is None or resolved != path:
                return {"fragment_id": frag_id, "status": "skipped", "reason": "index_unqueryable"}

        fragment = _load_fragment_from_path(path)
        if not fragment:
            return {"fragment_id": frag_id, "status": "skipped", "reason": "unreadable"}
        if _is_compacted_fragment(fragment):
            return {"fragment_id": frag_id, "status": "skipped", "reason": "already_compacted"}

        if cold_policy.get("require_neighbor_links", True):
            neighbors = _neighbor_ids(fragment)
            missing = [nid for nid in neighbors if nid not in self.memory_map]
            if missing:
                return {"fragment_id": frag_id, "status": "skipped", "reason": "missing_neighbors"}

        return cold_storage.compact_fragment_file(path, child=self.child, policy=cold_policy)

    @staticmethod
    def _parse_timestamp(value: Optional[str]) -> Optional[datetime]:
        if not value:
            return None
        try:
            ts = datetime.fromisoformat(value)
        except Exception:
            try:
                ts = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except Exception:
                return None
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts

    def _tier_for_age(self, timestamp: Optional[datetime], now: datetime) -> str:
        """
        Decide target tier based on age cutoffs. Defaults to 'short' when unknown.
        """
        policy = self.policy or DEFAULT_TIER_POLICY
        age_hours: Optional[float] = None
        if timestamp:
            age_hours = (now - timestamp).total_seconds() / 3600.0

        short_cap = policy.get("short", {}).get("max_age_hours")
        working_cap = policy.get("working", {}).get("max_age_hours")
        long_cap = policy.get("long", {}).get("max_age_hours")

        if age_hours is None or short_cap is None:
            return "short"
        if age_hours <= short_cap:
            return "short"
        if working_cap is not None and age_hours <= working_cap:
            return "working"
        if long_cap is not None and age_hours <= long_cap:
            return "long"
        return "cold"

    @staticmethod
    def _timestamp_sort_key(ts: Optional[datetime]) -> datetime:
        """
        Provide a stable sort key, pushing unknown timestamps to the oldest end.
        """
        if ts is None:
            return datetime.fromtimestamp(0, tz=timezone.utc)
        return ts

    def index_tier(self, tier="short"):
        tier_path = self.base_path / tier
        if not tier_path.exists():
            return
        for frag in tier_path.glob("frag_*.json"):
            try:
                with open(frag, "r") as f:
                    data = json.load(f)
                existing = self.memory_map.get(data["id"], {})
                self.memory_map[data["id"]] = {
                    "tier": tier,
                    "tags": data.get("tags", []),
                    "importance": data.get("importance", 0),
                    "last_seen": existing.get(
                        "last_seen",
                        data.get("timestamp", datetime.now(timezone.utc).isoformat())
                    ),
                    "filename": frag.name,
                    **self._stat_payload(frag),
                }
            except:
                continue

    def reindex_all(self, rebalance: bool = True):
        """
        Full reindex of all fragments. Optionally rebalance tiers afterward.
        """
        self.ensure_tier_directories()
        self.index_legacy_root()
        for tier in MEMORY_TIERS:
            self.index_tier(tier)
        self.save_map()
        log_to_statusbox(f"[Memory] Reindexed all fragments across {len(MEMORY_TIERS)} tiers.")
        log_to_statusbox(f"[Memory] Fragment count: {len(self.memory_map)}")
        if rebalance:
            self.rebalance_tiers()

    def reindex(self, new_only=True):
        added = 0
        self.ensure_tier_directories()
        self.index_legacy_root()
        for tier in MEMORY_TIERS:
            tier_path = self.base_path / tier
            if not tier_path.exists():
                continue
            for frag in tier_path.glob("frag_*.json"):
                try:
                    with open(frag, "r") as f:
                        data = json.load(f)
                    if not new_only or data["id"] not in self.memory_map:
                        self.memory_map[data["id"]] = {
                            "tier": tier,
                            "tags": data.get("tags", []),
                            "importance": data.get("importance", 0),
                            "last_seen": data.get(
                                "timestamp",
                                datetime.now(timezone.utc).isoformat()
                            ),
                            "filename": frag.name,
                            **self._stat_payload(frag),
                        }
                        added += 1
                    else:
                        existing = self.memory_map.get(data["id"], {})
                        existing.update({
                            "tier": tier,
                            "tags": data.get("tags", []),
                            "importance": data.get("importance", 0),
                            "filename": frag.name,
                            **self._stat_payload(frag),
                        })
                        self.memory_map[data["id"]] = existing
                except:
                    continue
        self.save_map()
        log_to_statusbox(f"[Memory] Reindexed {added} new fragments across tiers. Current total: {len(self.memory_map)}")

    def fast_reindex(self, rebalance: bool = False, prune_missing: bool = False):
        """
        Fast reindex that avoids JSON parsing for unchanged files by using size+mtime.
        """
        self.ensure_tier_directories()
        existing_by_path: Dict[str, str] = {}
        for frag_id, meta in self.memory_map.items():
            filename = meta.get("filename", f"frag_{frag_id}.json")
            tier = meta.get("tier")
            if tier in MEMORY_TIERS:
                existing_by_path[f"{tier}/{filename}"] = frag_id
            existing_by_path[filename] = frag_id

        scanned = 0
        added = 0
        updated = 0
        unchanged = 0
        skipped = 0
        seen_ids: Set[str] = set()

        for path in _iter_fragment_files(self.base_path):
            scanned += 1
            try:
                st = path.stat()
            except OSError:
                skipped += 1
                continue
            rel = path.relative_to(self.base_path).as_posix()
            tier = path.parent.name if path.parent.name in MEMORY_TIERS else "short"

            frag_id = existing_by_path.get(rel)
            if frag_id:
                meta = self.memory_map.get(frag_id, {})
                cached_mtime = meta.get("mtime_ns")
                cached_size = meta.get("size_bytes")
                if cached_mtime == st.st_mtime_ns and cached_size == st.st_size:
                    changed = False
                    if meta.get("filename") != path.name:
                        meta["filename"] = path.name
                        changed = True
                    if meta.get("tier") != tier:
                        meta["tier"] = tier
                        changed = True
                    if cached_mtime is None or cached_size is None:
                        meta["mtime_ns"] = int(st.st_mtime_ns)
                        meta["size_bytes"] = int(st.st_size)
                        changed = True
                    if changed:
                        self.memory_map[frag_id] = meta
                        updated += 1
                    else:
                        unchanged += 1
                    seen_ids.add(frag_id)
                    continue

            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                skipped += 1
                continue
            frag_id = data.get("id")
            if not frag_id:
                skipped += 1
                continue

            existing = self.memory_map.get(frag_id, {})
            entry = {
                "tier": tier,
                "tags": data.get("tags", existing.get("tags", [])),
                "importance": data.get("importance", existing.get("importance", 0)),
                "last_seen": existing.get(
                    "last_seen",
                    data.get("timestamp", datetime.now(timezone.utc).isoformat()),
                ),
                "filename": path.name,
                "mtime_ns": int(st.st_mtime_ns),
                "size_bytes": int(st.st_size),
            }

            if frag_id in seen_ids:
                prior = self.memory_map.get(frag_id, {})
                prior_mtime = prior.get("mtime_ns") or 0
                if prior_mtime >= entry["mtime_ns"]:
                    continue

            if frag_id in self.memory_map:
                updated += 1
            else:
                added += 1
            self.memory_map[frag_id] = entry
            seen_ids.add(frag_id)

        self.save_map()
        log_to_statusbox(
            f"[Memory] Fast reindex: scanned {scanned}, added {added}, updated {updated}, "
            f"unchanged {unchanged}, skipped {skipped}."
        )
        if prune_missing:
            self.prune_missing()
        if rebalance:
            self.rebalance_tiers()

    def ingest_fragment_file(self, fragment_path, to_tier):
        try:
            with open(fragment_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return False

        frag_id = data.get("id")
        if not frag_id:
            return False

        destination = self.base_path / to_tier
        destination.mkdir(parents=True, exist_ok=True)
        target_path = destination / fragment_path.name

        try:
            fragment_path.rename(target_path)
        except FileNotFoundError:
            return False

        self.memory_map[frag_id] = {
            "tier": to_tier,
            "tags": data.get("tags", []),
            "importance": data.get("importance", 0),
            "last_seen": data.get("timestamp", datetime.now(timezone.utc).isoformat()),
            "filename": target_path.name,
            **self._stat_payload(target_path),
        }
        self.save_map()
        return True

    def prune_missing(self):
        removed = []
        for frag_id, meta in list(self.memory_map.items()):
            tier = meta.get("tier")
            filename = meta.get("filename", f"frag_{frag_id}.json")
            candidate_paths = []
            if tier:
                candidate_paths.append(self.base_path / tier / filename)
            candidate_paths.append(self.base_path / filename)
            if not any(path.exists() for path in candidate_paths):
                removed.append(frag_id)
                self.memory_map.pop(frag_id, None)
        if removed:
            self.save_map()
            log_to_statusbox(
                f"[Memory] Pruned {len(removed)} missing fragment entries from index."
            )
        return len(removed)

    def rebalance_tiers(self, now: Optional[datetime] = None):
        """
        Move fragments out of short-term when they age out or exceed target counts.
        """
        if not self.memory_map:
            return {"moved": 0, "missing": 0, "transitions": {}, "counts": {}}

        self.ensure_tier_directories()
        now = now or datetime.now(timezone.utc)
        cold_policy = None
        auto_compact = False
        purge_pending = False
        try:
            import cold_storage  # type: ignore

            cold_policy = cold_storage.policy_from_config(_load_config())
            auto_compact = bool(cold_policy.get("enabled")) and bool(cold_policy.get("auto_compact"))
            purge_pending = auto_compact and bool(cold_policy.get("purge_pending_delete", False))
        except Exception:
            cold_policy = None
        buckets = {tier: [] for tier in MEMORY_TIERS}
        transitions: Dict[str, int] = {}
        missing = 0

        for frag_id, meta in self.memory_map.items():
            ts = self._parse_timestamp(meta.get("last_seen")) or self._parse_timestamp(meta.get("timestamp"))
            target_tier = self._tier_for_age(ts, now)
            buckets[target_tier].append(
                {
                    "id": frag_id,
                    "current_tier": meta.get("tier") or "short",
                    "timestamp": ts,
                    "path": self._resolve_fragment_path(frag_id, meta),
                    "last_seen": meta.get("last_seen"),
                }
            )

        # Apply target count caps so short/working stay lean.
        for tier, next_tier in [("short", "working"), ("working", "long"), ("long", "cold")]:
            try:
                cap = int(self.policy.get(tier, {}).get("target_count", 0))
            except (TypeError, ValueError):
                cap = 0
            if cap <= 0:
                continue
            bucket = buckets[tier]
            bucket.sort(key=lambda r: self._timestamp_sort_key(r["timestamp"]))
            if len(bucket) > cap:
                overflow = bucket[:-cap]
                buckets[tier] = bucket[-cap:]
                buckets[next_tier].extend(overflow)

        moved = 0
        compacted = 0
        compaction_skipped = 0
        for target_tier, records in buckets.items():
            for record in records:
                frag_id = record["id"]
                current_tier = record["current_tier"]
                path = record["path"]
                if path is None or not path.exists():
                    missing += 1
                    continue

                dest_dir = self.base_path / target_tier
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest_path = dest_dir / path.name

                if current_tier == target_tier and path.parent == dest_dir:
                    continue

                try:
                    path.rename(dest_path)
                except Exception:
                    missing += 1
                    continue

                moved += 1
                transition_key = f"{current_tier}->{target_tier}"
                transitions[transition_key] = transitions.get(transition_key, 0) + 1

                meta = self.memory_map.get(frag_id, {})
                meta["tier"] = target_tier
                meta["filename"] = dest_path.name
                if record["last_seen"]:
                    meta["last_seen"] = record["last_seen"]
                self.memory_map[frag_id] = meta

                if auto_compact and target_tier == "cold" and cold_policy:
                    result = self._compact_cold_fragment(frag_id, meta, dest_path, cold_policy)
                    if isinstance(result, dict):
                        status = result.get("status")
                        if status in {"failed", "compacted", "retained"}:
                            log_to_statusbox(
                                f"[ColdStorage] {status} fragment {frag_id}."
                            )

                if (
                    target_tier == "cold"
                    and compact_fragment_file
                    and self.cold_storage_policy.get("auto_compact", False)
                    and self.cold_storage_policy.get("enabled", True)
                ):
                    try:
                        if compact_fragment_file(dest_path, child=self.child, policy=self.cold_storage_policy):
                            compacted += 1
                    except Exception:
                        compaction_skipped += 1

        if moved or missing:
            self.save_map()

        if purge_pending and cold_policy:
            try:
                import cold_storage  # type: ignore

                purge_stats = cold_storage.purge_pending_delete(self.child, cold_policy)
                deleted = purge_stats.get("deleted", 0)
                kept = purge_stats.get("kept", 0)
                if deleted or kept:
                    log_to_statusbox(
                        f"[ColdStorage] Pending-delete purge: deleted {deleted}, kept {kept}."
                    )
            except Exception:
                pass

        transition_summary = ", ".join(f"{k}:{v}" for k, v in sorted(transitions.items()))
        if not transition_summary:
            transition_summary = "none"
        log_to_statusbox(
            f"[Memory] Rebalanced tiers: moved {moved} fragment(s); transitions {transition_summary}."
        )
        if missing:
            log_to_statusbox(f"[Memory] Rebalance skipped {missing} missing fragment files.")
        if compacted:
            log_to_statusbox(f"[Memory] Cold compaction updated {compacted} fragment(s).")
        if compaction_skipped:
            log_to_statusbox(f"[Memory] Cold compaction skipped {compaction_skipped} fragment(s).")

        counts = self.stats()
        return {"moved": moved, "missing": missing, "transitions": transitions, "counts": counts}

    def promote(self, frag_id, to_tier, *, touch=True):
        if frag_id not in self.memory_map:
            return False
        old_tier = self.memory_map[frag_id]["tier"]
        if old_tier == to_tier:
            return True
        filename = self.memory_map[frag_id].get("filename", f"frag_{frag_id}.json")
        src = self.base_path / old_tier / filename
        dst = self.base_path / to_tier / filename
        if not src.exists():
            return False
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.rename(dst)
        self.memory_map[frag_id]["tier"] = to_tier
        self.memory_map[frag_id]["filename"] = dst.name
        if touch:
            self.memory_map[frag_id]["last_seen"] = datetime.now(timezone.utc).isoformat()
        self.save_map()
        return True

    def get_by_tag(self, tag, tier=None):
        results = []
        for fid, meta in self.memory_map.items():
            if tag in meta.get("tags", []) and (tier is None or meta.get("tier") == tier):
                results.append(fid)
        return results

    def stats(self):
        counts = {tier: 0 for tier in MEMORY_TIERS}
        for meta in self.memory_map.values():
            t = meta.get("tier")
            if t in counts:
                counts[t] += 1
        log_to_statusbox(f"[Memory] Stats: {json.dumps(counts)}")
        return counts


def _resolve_bundle_paths(child: str, cfg: Optional[Dict[str, Any]], inastate: Dict[str, Any]) -> Tuple[Path, Path]:
    status = inastate.get("bundle_status") if isinstance(inastate.get("bundle_status"), dict) else {}
    root_value = status.get("root")
    bundle_dir_value = status.get("bundle_dir")

    root = Path(root_value) if root_value else Path("AI_Children") / child / "memory"
    root = root.expanduser()

    if not bundle_dir_value:
        policy = cfg.get("bundle_policy", {}) if isinstance(cfg, dict) else {}
        bundle_dir_value = policy.get("bundle_dir") or "bundles"
    bundle_dir = Path(str(bundle_dir_value)).expanduser()
    if not bundle_dir.is_absolute():
        bundle_dir = root / bundle_dir
    return root, bundle_dir


def _iter_bundle_index_entries(bundle_dir: Path) -> Iterable[Dict[str, Any]]:
    if not bundle_dir.exists():
        return
    for index_path in sorted(bundle_dir.glob("*.index.jsonl")):
        try:
            with index_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(payload, dict):
                        yield payload
        except OSError:
            continue


def _hash_file(path: Path) -> Optional[str]:
    hasher = hashlib.sha256()
    try:
        with path.open("rb") as fh:
            while True:
                chunk = fh.read(64 * 1024)
                if not chunk:
                    break
                hasher.update(chunk)
    except OSError:
        return None
    return hasher.hexdigest()


def _plan_or_apply_bundle_prune(
    manager: MemoryManager,
    root: Path,
    bundle_dir: Path,
    *,
    apply: bool,
    verify_sha: bool,
    max_prune: int,
) -> Dict[str, Any]:
    fragment_root = root / "fragments"
    rel_to_id: Dict[str, str] = {}
    for frag_id, meta in manager.memory_map.items():
        filename = meta.get("filename", f"frag_{frag_id}.json")
        tier = meta.get("tier")
        if tier in MEMORY_TIERS:
            rel_path = f"fragments/{tier}/{filename}"
        else:
            rel_path = f"fragments/{filename}"
        rel_to_id[rel_path] = frag_id

    candidates = 0
    pruned = 0
    bytes_total = 0
    verified = 0
    skipped = 0

    for entry in _iter_bundle_index_entries(bundle_dir):
        rel_path = entry.get("rel_path")
        if not rel_path or rel_path not in rel_to_id:
            continue
        frag_id = rel_to_id.get(rel_path)
        if not frag_id:
            continue
        abs_path = root / rel_path
        if not abs_path.exists():
            skipped += 1
            continue
        if verify_sha:
            expected = entry.get("sha256")
            actual = _hash_file(abs_path)
            if not expected or actual != expected:
                skipped += 1
                continue
            verified += 1

        size_hint = entry.get("size")
        try:
            size_value = int(size_hint) if size_hint is not None else abs_path.stat().st_size
        except (TypeError, ValueError, OSError):
            size_value = 0

        candidates += 1
        bytes_total += size_value

        if apply:
            try:
                abs_path.unlink()
            except OSError:
                skipped += 1
                continue
            manager.memory_map.pop(frag_id, None)
            pruned += 1
            if max_prune > 0 and pruned >= max_prune:
                break

    if apply and pruned:
        manager.save_map()

    return {
        "status": "applied" if apply else "planned",
        "candidates": candidates,
        "pruned": pruned,
        "bytes": bytes_total,
        "verified": verified,
        "skipped": skipped,
        "root": str(root),
        "bundle_dir": str(bundle_dir),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def handle_bundle_prune(child: str, cfg: Optional[Dict[str, Any]], manager: MemoryManager) -> None:
    state = _load_inastate(child)
    request_raw = state.get("bundle_prune_request")
    if not request_raw:
        return
    request = str(request_raw).lower().strip()
    if request not in {"plan", "apply"}:
        log_to_statusbox(f"[BundlePrune] Ignoring unknown request: {request_raw}")
        _write_inastate_value(child, "bundle_prune_request", None)
        return

    policy = _bundle_prune_policy(cfg)
    if not policy.get("enabled", False):
        log_to_statusbox("[BundlePrune] Prune request ignored; bundle_prune_policy.enabled is false.")
        _write_inastate_value(child, "bundle_prune_request", None)
        return

    if request == "apply" and not policy.get("allow_apply", False):
        log_to_statusbox("[BundlePrune] Apply requested but bundle_prune_policy.allow_apply is false.")
        _write_inastate_value(child, "bundle_prune_request", None)
        return

    if policy.get("require_bundle_ready", True):
        ready = state.get("bundle_prune_ready") if isinstance(state.get("bundle_prune_ready"), dict) else {}
        if not ready.get("ready", False):
            log_to_statusbox("[BundlePrune] Bundle prune request ignored; bundle_prune_ready is false.")
            _write_inastate_value(child, "bundle_prune_request", None)
            return

    root, bundle_dir = _resolve_bundle_paths(child, cfg, state)
    if not bundle_dir.exists():
        log_to_statusbox(f"[BundlePrune] Bundle directory missing: {bundle_dir}")
        _write_inastate_value(child, "bundle_prune_request", None)
        return

    report = _plan_or_apply_bundle_prune(
        manager,
        root,
        bundle_dir,
        apply=request == "apply",
        verify_sha=policy.get("verify_sha", True),
        max_prune=int(policy.get("max_prune") or 0),
    )
    report_path = Path("AI_Children") / child / "memory" / "bundle_prune_report.json"
    try:
        with report_path.open("w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
    except Exception:
        pass
    _write_inastate_value(child, "bundle_prune_report", report)
    _write_inastate_value(child, "bundle_prune_request", None)
    log_to_statusbox(
        f"[BundlePrune] {report['status']} {report['pruned']} of {report['candidates']} candidate fragment(s)."
    )

if __name__ == "__main__":
    cfg = _load_config()
    child = cfg.get("current_child", "Inazuma_Yagami")
    mgr = MemoryManager(child)

    boot_policy = _boot_policy(cfg)
    shutdown_ctx = _shutdown_context(child, boot_policy)
    boot_mode, reason = _resolve_boot_mode(boot_policy, shutdown_ctx)
    log_to_statusbox(f"[Memory] Boot mode: {boot_mode} (reason={reason}).")

    if boot_mode == "full":
        mgr.reindex_all(rebalance=boot_policy.get("rebalance_on_boot", True))
        if boot_policy.get("prune_missing_on_boot", False):
            mgr.prune_missing()
    else:
        mgr.fast_reindex(
            rebalance=boot_policy.get("rebalance_on_boot", False),
            prune_missing=boot_policy.get("prune_missing_on_boot", False),
        )

    mgr.stats()
    handle_bundle_prune(child, cfg, mgr)
    build_fractal_memory(child)
    build_experience_graph(child)
