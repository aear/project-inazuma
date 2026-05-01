# === memory_graph.py (Logging Enhanced) ===

import argparse
import os
import json
import math
import random
import heapq
import time
import tempfile
import shutil
from array import array
from collections import deque
import hashlib
import gc
import importlib
import sqlite3
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Iterable, List, Optional, Set, Tuple, TYPE_CHECKING
from gui_hook import log_to_statusbox
from body_schema import get_region_anchors
from experience_storage import iter_event_paths
from io_utils import atomic_write_json, file_lock, load_json_dict

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
    "fragment_anchor_count": 12,
    "tag_anchor_count": 6,
    "vector_round_digits": 6,
    "position_round_digits": 4,
    "edge_direction_enabled": True,
    "max_connection_degree": 0,
    "min_synapse_weight": 0.0,
    "gc_every_batches": 4,
    "max_neurons_total": 0,
    "max_edges_per_neuron": 0,
    "max_synapses_total": 0,
    "max_pending_dirty_fragments": 0,
    "compact_save_enabled": True,
    "synapse_spool_enabled": True,
    "spill_to_disk_enabled": False,
    "max_hot_neurons": 0,
    "spill_after_batches": 0,
    "spill_precision_mode": "adaptive",
    "spill_high_precision_threshold": 0.68,
    "spill_medium_precision_threshold": 0.35,
}

DEFAULT_TIER_POLICY = {
    "short": {"max_age_hours": 18.0, "target_count": 5000},
    "working": {"max_age_hours": 72.0, "target_count": 12000},
    "long": {"max_age_hours": 24.0 * 30.0, "target_count": 40000},
    "cold": {},
}
DEFAULT_RETENTION_POLICY = {
    "protect_cold_importance": 0.72,
    "compact_low_importance_threshold": 0.18,
    "compact_low_importance_age_hours": 24.0 * 7.0,
    "pre_compact_enabled": False,
    "pre_compact_limit": 24,
    "pre_compact_min_size_bytes": 65536,
    "human_prune_limit": 750,
    "human_prune_large_index_threshold": 1_000_000,
    "human_prune_large_index_limit": 5_000,
    "human_prune_candidate_multiplier": 6,
    "human_prune_min_age_hours": 24.0 * 7.0,
    "human_prune_max_importance": 0.18,
    "human_prune_min_size_bytes": 2048,
    "human_prune_cooldown_seconds": 3600.0,
    "human_prune_review_required": False,
    "human_prune_report_dir": "prune_reports",
    "human_prune_experience_enabled": True,
    "human_prune_experience_limit": 250,
    "human_prune_experience_min_age_hours": 24.0 * 30.0,
    "human_prune_experience_max_importance": 0.18,
    "human_prune_experience_min_size_bytes": 2048,
    "human_prune_experience_cold_root": None,
    "human_prune_experience_retain_local_stub": True,
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
        fragment_anchor_count = int(policy.get("fragment_anchor_count", 0))
    except (TypeError, ValueError):
        fragment_anchor_count = 12
    fragment_anchor_count = max(0, fragment_anchor_count)
    try:
        tag_anchor_count = int(policy.get("tag_anchor_count", 0))
    except (TypeError, ValueError):
        tag_anchor_count = 6
    tag_anchor_count = max(0, tag_anchor_count)
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
        max_connection_degree = int(policy.get("max_connection_degree", 0))
    except (TypeError, ValueError):
        max_connection_degree = 0
    if max_connection_degree < 0:
        max_connection_degree = 0
    try:
        min_synapse_weight = float(policy.get("min_synapse_weight", 0.0))
    except (TypeError, ValueError):
        min_synapse_weight = 0.0
    min_synapse_weight = _clamp(min_synapse_weight, 0.0, 1.0)
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
    try:
        max_synapses_total = int(policy.get("max_synapses_total", 0))
    except (TypeError, ValueError):
        max_synapses_total = 0
    if max_synapses_total < 0:
        max_synapses_total = 0
    try:
        max_pending_dirty_fragments = int(policy.get("max_pending_dirty_fragments", 0))
    except (TypeError, ValueError):
        max_pending_dirty_fragments = 0
    if max_pending_dirty_fragments < 0:
        max_pending_dirty_fragments = 0
    compact_save_enabled = bool(policy.get("compact_save_enabled", True))
    synapse_spool_enabled = bool(policy.get("synapse_spool_enabled", True))
    spill_to_disk_enabled = bool(policy.get("spill_to_disk_enabled", False))
    try:
        max_hot_neurons = int(policy.get("max_hot_neurons", 0))
    except (TypeError, ValueError):
        max_hot_neurons = 0
    if max_hot_neurons < 0:
        max_hot_neurons = 0
    try:
        spill_after_batches = int(policy.get("spill_after_batches", 0))
    except (TypeError, ValueError):
        spill_after_batches = 0
    if spill_after_batches < 0:
        spill_after_batches = 0
    spill_precision_mode = str(policy.get("spill_precision_mode", "adaptive") or "adaptive").strip().lower()
    if spill_precision_mode not in {"adaptive", "float", "int16", "int8"}:
        spill_precision_mode = "adaptive"
    try:
        spill_high_precision_threshold = float(policy.get("spill_high_precision_threshold", 0.68))
    except (TypeError, ValueError):
        spill_high_precision_threshold = 0.68
    try:
        spill_medium_precision_threshold = float(policy.get("spill_medium_precision_threshold", 0.35))
    except (TypeError, ValueError):
        spill_medium_precision_threshold = 0.35
    spill_high_precision_threshold = _clamp(spill_high_precision_threshold, 0.0, 1.0)
    spill_medium_precision_threshold = _clamp(spill_medium_precision_threshold, 0.0, spill_high_precision_threshold)
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
        "fragment_anchor_count": fragment_anchor_count,
        "tag_anchor_count": tag_anchor_count,
        "vector_round_digits": vector_round_digits,
        "position_round_digits": position_round_digits,
        "edge_direction_enabled": edge_direction_enabled,
        "max_connection_degree": max_connection_degree,
        "min_synapse_weight": min_synapse_weight,
        "gc_every_batches": gc_every_batches,
        "max_neurons_total": max_neurons_total,
        "max_edges_per_neuron": max_edges_per_neuron,
        "max_synapses_total": max_synapses_total,
        "max_pending_dirty_fragments": max_pending_dirty_fragments,
        "compact_save_enabled": compact_save_enabled,
        "synapse_spool_enabled": synapse_spool_enabled,
        "spill_to_disk_enabled": spill_to_disk_enabled,
        "max_hot_neurons": max_hot_neurons,
        "spill_after_batches": spill_after_batches,
        "spill_precision_mode": spill_precision_mode,
        "spill_high_precision_threshold": spill_high_precision_threshold,
        "spill_medium_precision_threshold": spill_medium_precision_threshold,
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

    def _coerce_float(value: Any) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _coerce_positive_float(value: Any) -> Optional[float]:
        fvalue = _coerce_float(value)
        return fvalue if fvalue is not None and fvalue > 0 else None

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

    retention = DEFAULT_RETENTION_POLICY.copy()
    raw_retention = user_policy.get("retention", {}) if isinstance(user_policy, dict) else {}
    if isinstance(raw_retention, dict):
        protect = _coerce_float(raw_retention.get("protect_cold_importance"))
        if protect is not None:
            retention["protect_cold_importance"] = _clamp(protect, 0.0, 1.0)
        low_importance = _coerce_float(raw_retention.get("compact_low_importance_threshold"))
        if low_importance is not None:
            retention["compact_low_importance_threshold"] = _clamp(low_importance, 0.0, 1.0)
        low_age = _coerce_positive_float(raw_retention.get("compact_low_importance_age_hours"))
        if low_age is not None:
            retention["compact_low_importance_age_hours"] = low_age
        retention["pre_compact_enabled"] = bool(raw_retention.get("pre_compact_enabled", retention["pre_compact_enabled"]))
        pre_limit = _coerce_positive_int(raw_retention.get("pre_compact_limit"))
        if pre_limit is not None:
            retention["pre_compact_limit"] = pre_limit
        min_size = _coerce_positive_int(raw_retention.get("pre_compact_min_size_bytes"))
        if min_size is not None:
            retention["pre_compact_min_size_bytes"] = min_size
        human_limit = _coerce_positive_int(raw_retention.get("human_prune_limit"))
        if human_limit is not None:
            retention["human_prune_limit"] = human_limit
        large_threshold = _coerce_positive_int(raw_retention.get("human_prune_large_index_threshold"))
        if large_threshold is not None:
            retention["human_prune_large_index_threshold"] = large_threshold
        large_limit = _coerce_positive_int(raw_retention.get("human_prune_large_index_limit"))
        if large_limit is not None:
            retention["human_prune_large_index_limit"] = large_limit
        candidate_multiplier = _coerce_positive_int(raw_retention.get("human_prune_candidate_multiplier"))
        if candidate_multiplier is not None:
            retention["human_prune_candidate_multiplier"] = candidate_multiplier
        human_age = _coerce_positive_float(raw_retention.get("human_prune_min_age_hours"))
        if human_age is not None:
            retention["human_prune_min_age_hours"] = human_age
        human_importance = _coerce_float(raw_retention.get("human_prune_max_importance"))
        if human_importance is not None:
            retention["human_prune_max_importance"] = _clamp(human_importance, 0.0, 1.0)
        human_min_size = _coerce_positive_int(raw_retention.get("human_prune_min_size_bytes"))
        if human_min_size is not None:
            retention["human_prune_min_size_bytes"] = human_min_size
        human_cooldown = _coerce_positive_float(raw_retention.get("human_prune_cooldown_seconds"))
        if human_cooldown is not None:
            retention["human_prune_cooldown_seconds"] = human_cooldown
        if "human_prune_review_required" in raw_retention:
            retention["human_prune_review_required"] = bool(raw_retention.get("human_prune_review_required"))
        report_dir = raw_retention.get("human_prune_report_dir")
        if isinstance(report_dir, str) and report_dir.strip():
            retention["human_prune_report_dir"] = report_dir.strip()
        if "human_prune_experience_enabled" in raw_retention:
            retention["human_prune_experience_enabled"] = bool(raw_retention.get("human_prune_experience_enabled"))
        experience_limit = _coerce_positive_int(raw_retention.get("human_prune_experience_limit"))
        if experience_limit is not None:
            retention["human_prune_experience_limit"] = experience_limit
        experience_age = _coerce_positive_float(raw_retention.get("human_prune_experience_min_age_hours"))
        if experience_age is not None:
            retention["human_prune_experience_min_age_hours"] = experience_age
        experience_importance = _coerce_float(raw_retention.get("human_prune_experience_max_importance"))
        if experience_importance is not None:
            retention["human_prune_experience_max_importance"] = _clamp(experience_importance, 0.0, 1.0)
        experience_min_size = _coerce_positive_int(raw_retention.get("human_prune_experience_min_size_bytes"))
        if experience_min_size is not None:
            retention["human_prune_experience_min_size_bytes"] = experience_min_size
        experience_root = raw_retention.get("human_prune_experience_cold_root")
        if isinstance(experience_root, str) and experience_root.strip():
            retention["human_prune_experience_cold_root"] = experience_root.strip()
        if "human_prune_experience_retain_local_stub" in raw_retention:
            retention["human_prune_experience_retain_local_stub"] = bool(
                raw_retention.get("human_prune_experience_retain_local_stub")
            )
    layout = cfg.get("storage_layout") if isinstance(cfg, dict) else {}
    if isinstance(layout, dict) and not retention.get("human_prune_experience_cold_root"):
        cold_experience_root = layout.get("cold_experience_root")
        if isinstance(cold_experience_root, str) and cold_experience_root.strip():
            retention["human_prune_experience_cold_root"] = cold_experience_root.strip()
    policy["retention"] = retention
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
    return load_json_dict(path)


def _write_inastate_value(child: str, key: str, value: Any) -> None:
    path = Path("AI_Children") / child / "memory" / "inastate.json"
    lock_path = path.with_name("inastate.lock")
    try:
        with file_lock(lock_path):
            data = load_json_dict(path)
            data[key] = value
            atomic_write_json(path, data)
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


def _memory_graph_phase_args(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="Memory graph maintenance phases.")
    parser.add_argument("--phase", choices=("full", "boot", "neural"), default="full")
    return parser.parse_args(argv)


def _set_memory_graph_deferred_build(child: str, status: str, **extra: Any) -> Dict[str, Any]:
    state = _load_inastate(child)
    payload = state.get("memory_graph_deferred_build") if isinstance(state.get("memory_graph_deferred_build"), dict) else {}
    payload = dict(payload)
    payload.update(extra)
    payload["status"] = str(status or "").strip().lower() or "queued"
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    if payload["status"] != "running":
        payload.pop("pid", None)
    _write_inastate_value(child, "memory_graph_deferred_build", payload)
    return payload


def _run_memory_index_verification(child: str, cfg: Dict[str, Any], mgr: "MemoryManager") -> Dict[str, Any]:
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
    if hasattr(mgr, "human_memory_prune_pass"):
        mgr.human_memory_prune_pass(force=True)
    handle_bundle_prune(child, cfg, mgr)
    verification = {
        "boot_mode": boot_mode,
        "reason": reason,
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "fragment_count": len(mgr.memory_map),
    }
    _write_inastate_value(child, "memory_graph_last_verification", verification)
    return verification


def _run_memory_neural_phase(child: str, mgr: "MemoryManager", *, launch_source: str = "manual") -> None:
    launched_at = datetime.now(timezone.utc).isoformat()
    _set_memory_graph_deferred_build(
        child,
        "running",
        launch_source=launch_source,
        launched_at=launched_at,
        pid=os.getpid(),
    )
    log_to_statusbox("[Memory] Starting neural graph build phase.")
    build_summary: Dict[str, Any] = {}
    experience_summary: Dict[str, Any] = {}
    try:
        mgr.fast_reindex(rebalance=False, prune_missing=False)
        if hasattr(mgr, "human_memory_prune_pass"):
            mgr.human_memory_prune_pass()
        build_summary = build_fractal_memory(child) or {}
        if build_summary.get("needs_resume"):
            _set_memory_graph_deferred_build(
                child,
                "queued",
                launch_source=launch_source,
                queued_at=datetime.now(timezone.utc).isoformat(),
                fragment_count=len(mgr.memory_map),
                last_cycle=build_summary,
                last_error=None,
            )
            log_to_statusbox("[Memory] Neural graph burst paused with work remaining; re-queued for the scheduler.")
            return
        experience_summary = build_experience_graph(child) or {}
    except Exception as exc:
        _set_memory_graph_deferred_build(
            child,
            "failed",
            launch_source=launch_source,
            failed_at=datetime.now(timezone.utc).isoformat(),
            last_cycle=build_summary,
            last_error=str(exc),
        )
        raise

    _set_memory_graph_deferred_build(
        child,
        "completed",
        launch_source=launch_source,
        completed_at=datetime.now(timezone.utc).isoformat(),
        fragment_count=len(mgr.memory_map),
        last_cycle=build_summary,
        experience_cycle=experience_summary,
        last_error=None,
    )
    log_to_statusbox("[Memory] Neural graph build phase completed.")


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


_PROTECTED_FRAGMENT_TAGS = {
    "anchor", "anchors", "core", "identity", "bond", "attachment", "pinned", "protected",
    "self", "self_model", "relationship", "love", "grief", "family", "memory_anchor",
}


def _record_tags(record: Dict[str, Any]) -> Set[str]:
    tags = record.get("tags") if isinstance(record.get("tags"), list) else []
    return {str(tag).lower() for tag in tags if tag}


def _is_anchor_fragment_record(record: Dict[str, Any], retention: Optional[Dict[str, Any]] = None) -> bool:
    retention = retention or DEFAULT_RETENTION_POLICY
    if _record_tags(record) & _PROTECTED_FRAGMENT_TAGS:
        return True
    protect = _clamp(_safe_float(retention.get("protect_cold_importance"), DEFAULT_RETENTION_POLICY["protect_cold_importance"]), 0.0, 1.0)
    importance = _clamp(_safe_float(record.get("importance"), 0.0), 0.0, 1.0)
    return importance >= protect


def _select_pregraph_compaction_candidates(child: str, index: Any, retention: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any], Path]]:
    limit = max(0, int(retention.get("pre_compact_limit", 0) or 0))
    if limit <= 0:
        return []
    min_age_hours = max(0.0, _safe_float(retention.get("compact_low_importance_age_hours"), DEFAULT_RETENTION_POLICY["compact_low_importance_age_hours"]))
    low_importance = _clamp(_safe_float(retention.get("compact_low_importance_threshold"), DEFAULT_RETENTION_POLICY["compact_low_importance_threshold"]), 0.0, 1.0)
    min_size_bytes = max(0, int(retention.get("pre_compact_min_size_bytes", 0) or 0))
    now = datetime.now(timezone.utc)
    heap: List[Tuple[Tuple[int, float, str], str, Dict[str, Any], Path]] = []
    for frag_id, meta in index.items():
        if not frag_id or not isinstance(meta, dict):
            continue
        if _is_anchor_fragment_record(meta, retention):
            continue
        tier = str(meta.get("tier") or "short")
        if tier not in {"long", "cold"}:
            continue
        importance = _clamp(_safe_float(meta.get("importance"), 0.0), 0.0, 1.0)
        if importance > low_importance:
            continue
        ts = _parse_iso_timestamp(meta.get("last_seen") or meta.get("timestamp"))
        if ts is None:
            continue
        age_hours = (now - datetime.fromtimestamp(ts, timezone.utc)).total_seconds() / 3600.0
        if age_hours < min_age_hours:
            continue
        size_bytes = int(_safe_float(meta.get("size_bytes"), 0.0))
        if size_bytes < min_size_bytes:
            continue
        path = _resolve_index_path(child, frag_id, meta)
        if path is None:
            continue
        score = (size_bytes, age_hours, str(frag_id))
        entry = (score, str(frag_id), meta, path)
        if len(heap) < limit:
            heapq.heappush(heap, entry)
        elif entry[0] > heap[0][0]:
            heapq.heapreplace(heap, entry)
    return [(frag_id, meta, path) for _, frag_id, meta, path in sorted(heap, key=lambda item: item[0], reverse=True)]


def _human_memory_prune_settings(retention: Optional[Dict[str, Any]], index_count: int = 0) -> Dict[str, Any]:
    retention = retention or DEFAULT_RETENTION_POLICY
    base_limit = max(1, int(_safe_float(retention.get("human_prune_limit"), DEFAULT_RETENTION_POLICY["human_prune_limit"])))
    large_threshold = max(1, int(_safe_float(
        retention.get("human_prune_large_index_threshold"),
        DEFAULT_RETENTION_POLICY["human_prune_large_index_threshold"],
    )))
    large_limit = max(base_limit, int(_safe_float(
        retention.get("human_prune_large_index_limit"),
        DEFAULT_RETENTION_POLICY["human_prune_large_index_limit"],
    )))
    limit = large_limit if index_count >= large_threshold else base_limit
    min_age_default = _safe_float(
        retention.get("compact_low_importance_age_hours"),
        DEFAULT_RETENTION_POLICY["compact_low_importance_age_hours"],
    )
    max_importance_default = _safe_float(
        retention.get("compact_low_importance_threshold"),
        DEFAULT_RETENTION_POLICY["compact_low_importance_threshold"],
    )
    return {
        "limit": limit,
        "candidate_multiplier": max(1, int(_safe_float(
            retention.get("human_prune_candidate_multiplier"),
            DEFAULT_RETENTION_POLICY["human_prune_candidate_multiplier"],
        ))),
        "min_age_hours": max(0.0, _safe_float(
            retention.get("human_prune_min_age_hours"),
            min_age_default,
        )),
        "max_importance": _clamp(_safe_float(
            retention.get("human_prune_max_importance"),
            max_importance_default,
        ), 0.0, 1.0),
        "min_size_bytes": max(0, int(_safe_float(
            retention.get("human_prune_min_size_bytes"),
            DEFAULT_RETENTION_POLICY["human_prune_min_size_bytes"],
        ))),
        "cooldown_seconds": max(0.0, _safe_float(
            retention.get("human_prune_cooldown_seconds"),
            DEFAULT_RETENTION_POLICY["human_prune_cooldown_seconds"],
        )),
    }


def _human_prune_candidate_from_meta(
    child: str,
    frag_id: str,
    meta: Dict[str, Any],
    retention: Dict[str, Any],
    settings: Dict[str, Any],
    now_ts: float,
) -> Optional[Tuple[str, Dict[str, Any], Path, Tuple[int, float, float, str]]]:
    if not frag_id or not isinstance(meta, dict):
        return None
    if _is_anchor_fragment_record(meta, retention):
        return None
    tier = str(meta.get("tier") or "short")
    if tier == "working":
        return None
    if tier not in set(MEMORY_TIERS):
        tier = "short"
    importance = _clamp(_safe_float(meta.get("importance"), 0.0), 0.0, 1.0)
    if importance > settings["max_importance"]:
        return None
    ts = _parse_iso_timestamp(meta.get("last_seen") or meta.get("timestamp"))
    if ts is None:
        return None
    age_hours = max(0.0, (now_ts - ts) / 3600.0)
    if age_hours < settings["min_age_hours"]:
        return None
    path = _resolve_index_path(child, frag_id, meta)
    if path is None:
        return None
    size_bytes = int(_safe_float(meta.get("size_bytes"), 0.0))
    if size_bytes <= 0:
        try:
            size_bytes = int(path.stat().st_size)
        except OSError:
            return None
    if size_bytes < settings["min_size_bytes"]:
        return None
    score = (size_bytes, age_hours, -importance, str(frag_id))
    return str(frag_id), meta, path, score


def _select_human_memory_prune_candidates(
    child: str,
    index: Any,
    retention: Optional[Dict[str, Any]],
    *,
    index_count: int = 0,
    now: Optional[datetime] = None,
) -> List[Tuple[str, Dict[str, Any], Path]]:
    retention = retention or DEFAULT_RETENTION_POLICY
    settings = _human_memory_prune_settings(retention, index_count=index_count)
    limit = int(settings["limit"])
    if limit <= 0 or not index:
        return []
    now_dt = now or datetime.now(timezone.utc)
    if now_dt.tzinfo is None:
        now_dt = now_dt.replace(tzinfo=timezone.utc)
    now_ts = now_dt.timestamp()

    if hasattr(index, "iter_human_prune_candidates"):
        return list(index.iter_human_prune_candidates(retention, settings, now_ts))[:limit]

    heap: List[Tuple[Tuple[int, float, float, str], str, Dict[str, Any], Path]] = []
    for frag_id, meta in index.items():
        candidate = _human_prune_candidate_from_meta(child, str(frag_id), meta, retention, settings, now_ts)
        if candidate is None:
            continue
        candidate_id, candidate_meta, candidate_path, score = candidate
        entry = (score, candidate_id, candidate_meta, candidate_path)
        if len(heap) < limit:
            heapq.heappush(heap, entry)
        elif entry[0] > heap[0][0]:
            heapq.heapreplace(heap, entry)
    return [(frag_id, meta, path) for _, frag_id, meta, path in sorted(heap, key=lambda item: item[0], reverse=True)]


def _format_child_path(value: str, child: str) -> str:
    return value.replace("{child}", child)


def _memory_root(child: str) -> Path:
    return Path("AI_Children") / child / "memory"


def _human_prune_report_dir(child: str, retention: Dict[str, Any]) -> Path:
    raw = retention.get("human_prune_report_dir") or DEFAULT_RETENTION_POLICY["human_prune_report_dir"]
    report_dir = Path(_format_child_path(str(raw), child)).expanduser()
    if not report_dir.is_absolute():
        report_dir = _memory_root(child) / report_dir
    return report_dir


def _experience_cold_root(child: str, retention: Dict[str, Any]) -> Path:
    raw = retention.get("human_prune_experience_cold_root")
    if isinstance(raw, str) and raw.strip():
        return Path(_format_child_path(raw.strip(), child)).expanduser()
    return _memory_root(child) / "cold_storage" / "experiences"


def _safe_load_json_file(path: Path) -> Optional[Dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _experience_tags(payload: Dict[str, Any]) -> Set[str]:
    tags: Set[str] = set()
    for key in ("tags", "situation_tags"):
        value = payload.get(key)
        if isinstance(value, list):
            tags.update(str(tag).lower() for tag in value if tag)
    for key in ("outcome", "result", "internal_state"):
        value = payload.get(key)
        if isinstance(value, dict):
            nested = value.get("tags") or value.get("flags")
            if isinstance(nested, list):
                tags.update(str(tag).lower() for tag in nested if tag)
    return tags


def _experience_importance(payload: Dict[str, Any]) -> float:
    scores: List[float] = []
    for key in ("importance", "salience", "priority", "novelty"):
        if key in payload:
            scores.append(abs(_safe_float(payload.get(key), 0.0)))
    for container_key in ("internal_state", "outcome", "result"):
        container = payload.get(container_key)
        if not isinstance(container, dict):
            continue
        for key in ("importance", "salience", "priority", "novelty", "risk", "stress"):
            if key in container:
                scores.append(abs(_safe_float(container.get(key), 0.0)))
        emotions = container.get("emotions")
        if isinstance(emotions, dict):
            for key in ("intensity", "risk", "stress", "care", "trust", "novelty"):
                if key in emotions:
                    scores.append(abs(_safe_float(emotions.get(key), 0.0)))
    if payload.get("word_usage") or payload.get("feedback_hooks"):
        scores.append(0.5)
    return _clamp(max(scores) if scores else 0.0, 0.0, 1.0)


def _experience_timestamp(payload: Dict[str, Any], path: Path) -> Optional[float]:
    for key in ("timestamp", "start_time", "end_time", "created_at"):
        ts = _parse_iso_timestamp(payload.get(key))
        if ts is not None:
            return ts
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def _is_protected_experience(payload: Dict[str, Any], retention: Dict[str, Any]) -> bool:
    if _experience_tags(payload) & _PROTECTED_FRAGMENT_TAGS:
        return True
    if payload.get("word_usage") or payload.get("feedback_hooks"):
        return True
    protect = _clamp(
        _safe_float(retention.get("protect_cold_importance"), DEFAULT_RETENTION_POLICY["protect_cold_importance"]),
        0.0,
        1.0,
    )
    return _experience_importance(payload) >= protect


def _experience_summary(payload: Dict[str, Any]) -> str:
    for key in ("summary", "narrative", "intent"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()[:280]
    tags = sorted(_experience_tags(payload))
    if tags:
        return "Experience tagged " + ", ".join(tags[:8])
    return "Low-signal experience record with no short narrative."


def _candidate_reason(
    *,
    kind: str,
    age_hours: float,
    importance: float,
    size_bytes: int,
    tags: Iterable[str],
    action: str,
) -> str:
    age_days = age_hours / 24.0
    tag_text = ", ".join(sorted(str(tag) for tag in tags if tag)[:6]) or "no protected tags"
    size_kib = size_bytes / 1024.0
    return (
        f"{kind} is about {age_days:.1f} days old, uses {size_kib:.1f} KiB, "
        f"has importance {importance:.2f}, and has {tag_text}. Recommended action: {action}."
    )


def _experience_candidate_from_path(
    child: str,
    kind: str,
    path: Path,
    retention: Dict[str, Any],
    settings: Dict[str, Any],
    now_ts: float,
) -> Optional[Tuple[Dict[str, Any], Tuple[int, float, float, str]]]:
    payload = _safe_load_json_file(path)
    if not payload:
        return None
    if payload.get("cold_experience") or payload.get("experience_compacted_at"):
        return None
    if _is_protected_experience(payload, retention):
        return None
    try:
        size_bytes = int(path.stat().st_size)
    except OSError:
        return None
    if size_bytes < int(settings.get("min_size_bytes") or 0):
        return None
    ts = _experience_timestamp(payload, path)
    if ts is None:
        return None
    age_hours = max(0.0, (now_ts - ts) / 3600.0)
    if age_hours < float(settings.get("min_age_hours") or 0.0):
        return None
    importance = _experience_importance(payload)
    if importance > float(settings.get("max_importance") or 0.0):
        return None
    tags = sorted(_experience_tags(payload))
    action = "copy full record to HDD cold storage and leave a local recall stub"
    item = {
        "kind": kind,
        "id": str(payload.get("id") or path.stem),
        "path": str(path),
        "size_bytes": size_bytes,
        "age_hours": round(age_hours, 3),
        "importance": round(importance, 4),
        "tags": tags,
        "summary": _experience_summary(payload),
        "recommended_action": "compact_experience_to_cold_stub",
        "reason": _candidate_reason(
            kind=kind,
            age_hours=age_hours,
            importance=importance,
            size_bytes=size_bytes,
            tags=tags,
            action=action,
        ),
    }
    score = (size_bytes, age_hours, -importance, str(path))
    return item, score


def _select_experience_prune_candidates(
    child: str,
    retention: Optional[Dict[str, Any]],
    *,
    now: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    retention = retention or DEFAULT_RETENTION_POLICY
    if not bool(retention.get("human_prune_experience_enabled", True)):
        return []
    limit = max(0, int(_safe_float(
        retention.get("human_prune_experience_limit"),
        DEFAULT_RETENTION_POLICY["human_prune_experience_limit"],
    )))
    if limit <= 0:
        return []
    settings = {
        "min_age_hours": max(0.0, _safe_float(
            retention.get("human_prune_experience_min_age_hours"),
            DEFAULT_RETENTION_POLICY["human_prune_experience_min_age_hours"],
        )),
        "max_importance": _clamp(_safe_float(
            retention.get("human_prune_experience_max_importance"),
            DEFAULT_RETENTION_POLICY["human_prune_experience_max_importance"],
        ), 0.0, 1.0),
        "min_size_bytes": max(0, int(_safe_float(
            retention.get("human_prune_experience_min_size_bytes"),
            DEFAULT_RETENTION_POLICY["human_prune_experience_min_size_bytes"],
        ))),
    }
    now_dt = now or datetime.now(timezone.utc)
    if now_dt.tzinfo is None:
        now_dt = now_dt.replace(tzinfo=timezone.utc)
    now_ts = now_dt.timestamp()
    base = _memory_root(child) / "experiences"
    paths = [
        ("experience_event", base / "events"),
        ("experience_episode", base / "episodes"),
    ]
    heap: List[Tuple[Tuple[int, float, float, str], Dict[str, Any]]] = []
    for kind, root in paths:
        if not root.exists():
            continue
        path_iter = iter_event_paths(root) if kind == "experience_event" else root.glob("*.json")
        for path in path_iter:
            if not path.is_file():
                continue
            candidate = _experience_candidate_from_path(child, kind, path, retention, settings, now_ts)
            if candidate is None:
                continue
            item, score = candidate
            entry = (score, item)
            if len(heap) < limit:
                heapq.heappush(heap, entry)
            elif entry[0] > heap[0][0]:
                heapq.heapreplace(heap, entry)
    return [item for _, item in sorted(heap, key=lambda entry: entry[0], reverse=True)]


def _fragment_prune_report_items(
    candidates: List[Tuple[str, Dict[str, Any], Path]],
    now_dt: datetime,
) -> List[Dict[str, Any]]:
    now_ts = now_dt.timestamp()
    items: List[Dict[str, Any]] = []
    for frag_id, meta, path in candidates:
        ts = _parse_iso_timestamp(meta.get("last_seen") or meta.get("timestamp"))
        age_hours = max(0.0, (now_ts - ts) / 3600.0) if ts is not None else 0.0
        size_bytes = int(_safe_float(meta.get("size_bytes"), 0.0))
        if size_bytes <= 0:
            try:
                size_bytes = int(path.stat().st_size)
            except OSError:
                size_bytes = 0
        importance = _clamp(_safe_float(meta.get("importance"), 0.0), 0.0, 1.0)
        tags = [str(tag).lower() for tag in meta.get("tags", []) if tag] if isinstance(meta.get("tags"), list) else []
        action = "move to cold tier, compact to anchors/shards, and quarantine the full detail before purge"
        items.append({
            "kind": "fragment",
            "id": str(frag_id),
            "path": str(path),
            "tier": str(meta.get("tier") or "short"),
            "filename": str(meta.get("filename") or path.name),
            "size_bytes": size_bytes,
            "age_hours": round(age_hours, 3),
            "importance": round(importance, 4),
            "tags": tags,
            "recommended_action": "compact_fragment_to_cold_stub",
            "reason": _candidate_reason(
                kind="fragment",
                age_hours=age_hours,
                importance=importance,
                size_bytes=size_bytes,
                tags=tags,
                action=action,
            ),
        })
    return items


def _write_human_prune_review_report(
    child: str,
    retention: Dict[str, Any],
    fragment_items: List[Dict[str, Any]],
    experience_items: List[Dict[str, Any]],
    settings: Dict[str, Any],
    now_dt: datetime,
) -> Dict[str, Any]:
    report_id = f"prune_{now_dt.strftime('%Y%m%dT%H%M%SZ')}"
    report_dir = _human_prune_report_dir(child, retention)
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / f"{report_id}.json"
    md_path = report_dir / f"{report_id}.md"
    candidates = fragment_items + experience_items
    total_bytes = sum(int(item.get("size_bytes") or 0) for item in candidates)
    report = {
        "report_id": report_id,
        "child": child,
        "status": "review_required",
        "timestamp": now_dt.isoformat(),
        "summary": {
            "fragments": len(fragment_items),
            "experiences": len(experience_items),
            "candidates": len(candidates),
            "candidate_bytes": total_bytes,
        },
        "settings": settings,
        "approval": {
            "inastate_key": "human_memory_prune_apply_report",
            "inastate_value": report_id,
            "note": "Set this only after reviewing the report. The next prune pass will apply this exact report and clear the key.",
        },
        "candidates": candidates,
    }
    try:
        from exchange_review import evaluate_prune_report_payload

        report["exchange_review"] = evaluate_prune_report_payload(report)
    except Exception:
        report["exchange_review"] = {
            "status": "unavailable",
            "reason": "exchange_review_failed",
        }
    report["paths"] = {"json": str(json_path), "markdown": str(md_path)}
    atomic_write_json(json_path, report, indent=2, ensure_ascii=True)

    exchange = report.get("exchange_review") if isinstance(report.get("exchange_review"), dict) else {}
    lines = [
        f"# Memory prune report: {report_id}",
        "",
        f"Child: {child}",
        f"Generated: {now_dt.isoformat()}",
        "",
        "## Summary",
        "",
        f"- Fragment candidates: {len(fragment_items)}",
        f"- Experience candidates: {len(experience_items)}",
        f"- Candidate bytes: {total_bytes}",
        "",
        "## Law of Exchange",
        "",
        f"- Recommendation: {exchange.get('recommendation', 'unavailable')}",
        f"- Balance: {exchange.get('law_of_exchange', {}).get('plain_english', 'Exchange review unavailable.') if isinstance(exchange.get('law_of_exchange'), dict) else 'Exchange review unavailable.'}",
        "",
        "## Approval",
        "",
        "Review this report before applying. To approve the exact report, set ",
        "`human_memory_prune_apply_report` in inastate to this report id.",
        "",
        "## Reasons",
        "",
    ]
    preview = candidates[:120]
    for item in preview:
        lines.append(
            f"- {item.get('kind')} `{item.get('id')}`: {item.get('reason')} Path: `{item.get('path')}`"
        )
    if len(candidates) > len(preview):
        lines.append("")
        lines.append(f"JSON report contains {len(candidates) - len(preview)} additional candidate(s).")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def _load_human_prune_report(child: str, retention: Dict[str, Any], report_id: str) -> Optional[Dict[str, Any]]:
    raw = str(report_id or "").strip()
    if not raw:
        return None
    candidate = Path(_format_child_path(raw, child)).expanduser()
    if not candidate.is_absolute():
        if candidate.suffix != ".json":
            candidate = _human_prune_report_dir(child, retention) / f"{raw}.json"
        else:
            candidate = _human_prune_report_dir(child, retention) / candidate
    return _safe_load_json_file(candidate)


def _compact_experience_file(
    child: str,
    item: Dict[str, Any],
    retention: Dict[str, Any],
    now_dt: datetime,
) -> Dict[str, Any]:
    path = Path(str(item.get("path") or ""))
    if not path.exists() or not path.is_file():
        return {"status": "missing", "path": str(path)}
    payload = _safe_load_json_file(path)
    if not payload:
        return {"status": "unreadable", "path": str(path)}
    if payload.get("cold_experience") or payload.get("experience_compacted_at"):
        return {"status": "already_compacted", "path": str(path)}

    kind = str(item.get("kind") or "experience")
    cold_root = _experience_cold_root(child, retention)
    full_dir = cold_root / "full" / kind
    summary_dir = cold_root / "summaries" / kind
    full_path = full_dir / path.name
    summary_path = summary_dir / path.name
    source_hash = _hash_file(path)
    if not source_hash:
        return {"status": "unreadable", "path": str(path)}

    full_path.parent.mkdir(parents=True, exist_ok=True)
    if full_path.exists():
        full_path = full_path.with_name(f"{full_path.stem}__{now_dt.strftime('%Y%m%dT%H%M%SZ')}{full_path.suffix}")
    try:
        shutil.copy2(path, full_path)
    except Exception as exc:
        return {"status": "failed", "reason": f"copy_failed: {exc}", "path": str(path)}
    if _hash_file(full_path) != source_hash:
        try:
            full_path.unlink()
        except OSError:
            pass
        return {"status": "failed", "reason": "copy_verification_failed", "path": str(path)}

    summary = {
        "version": 1,
        "id": payload.get("id") or path.stem,
        "kind": kind,
        "timestamp": payload.get("timestamp") or payload.get("start_time") or payload.get("end_time"),
        "summary": _experience_summary(payload),
        "tags": sorted(_experience_tags(payload)),
        "importance": _experience_importance(payload),
        "source_path": str(path),
        "cold_full_path": str(full_path),
        "source_sha256": source_hash,
        "compacted_at": now_dt.isoformat(),
        "reason": item.get("reason"),
    }
    atomic_write_json(summary_path, summary, indent=2, ensure_ascii=True)

    if bool(retention.get("human_prune_experience_retain_local_stub", True)):
        stub = {
            "id": summary["id"],
            "type": payload.get("type") or kind,
            "timestamp": summary["timestamp"],
            "summary": summary["summary"],
            "tags": summary["tags"],
            "importance": summary["importance"],
            "cold_experience": {
                "full_path": str(full_path),
                "summary_path": str(summary_path),
                "source_sha256": source_hash,
            },
            "experience_compacted_at": now_dt.isoformat(),
            "reconstructed": False,
        }
        atomic_write_json(path, stub, indent=2, ensure_ascii=True)

    return {
        "status": "compacted",
        "path": str(path),
        "cold_full_path": str(full_path),
        "summary_path": str(summary_path),
    }


def _pregraph_fragment_compaction_pass(child: str, index: Any, retention: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    retention = retention or DEFAULT_RETENTION_POLICY
    if not bool(retention.get("pre_compact_enabled", False)):
        return {"status": "disabled", "candidates": 0, "compacted": 0, "skipped": 0}
    if compact_fragment_file is None:
        return {"status": "unavailable", "candidates": 0, "compacted": 0, "skipped": 0}
    cold_policy = _cold_storage_policy()
    if not cold_policy.get("enabled", True):
        return {"status": "disabled", "candidates": 0, "compacted": 0, "skipped": 0}
    selected = _select_pregraph_compaction_candidates(child, index, retention)
    compacted = 0
    skipped = 0
    for frag_id, meta, path in selected:
        fragment = _load_fragment_from_path(path)
        if not fragment or _is_compacted_fragment(fragment) or _is_anchor_fragment_record(fragment, retention):
            skipped += 1
            continue
        try:
            result = compact_fragment_file(path, child=child, policy=cold_policy)
        except Exception:
            result = None
        if isinstance(result, dict) and result.get("status") in {"compacted", "retained"}:
            compacted += 1
        else:
            skipped += 1
    return {"status": "ok", "candidates": len(selected), "compacted": compacted, "skipped": skipped}


def _candidate_pool_from_index(
    child: str,
    index: Any,
    *,
    pool_max: int,
    blocked_tags: Iterable[str],
    recent_ids: Set[str],
    known_fragments: Optional[Set[str]],
    cost_max_bytes: Optional[int],
) -> List[str]:
    if pool_max <= 0:
        return []
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
        entry = (ts or 0.0, importance, frag_id)
        if len(entries) < pool_max:
            heapq.heappush(entries, entry)
        elif entry > entries[0]:
            heapq.heapreplace(entries, entry)
    entries.sort(reverse=True)
    return [entry[2] for entry in entries]


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
    index: Optional[Any] = None,
) -> Tuple[List[Dict[str, Any]], int, Dict[str, Any]]:
    config = _selector_config(cfg)
    if not config.get("enabled", True):
        fragments, total = load_fragments(child, limit=limit)
        return fragments, total, {"mode": "disabled"}

    owned_index = None
    if index is None:
        owned_index = _load_memory_index(child)
        index = owned_index
    if not index:
        fragments, total = load_fragments(child, limit=limit)
        if owned_index is not None and hasattr(owned_index, "close"):
            owned_index.close()
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
        meta = index.get(frag_id, {})
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

    if owned_index is not None and hasattr(owned_index, "close"):
        owned_index.close()
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
        for path in iter_event_paths(events_dir):
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
            except Exception:
                continue
            if "id" not in data:
                continue
            total += 1
            entry = (_ts_value(data, path), str(path), data)
            if len(heap) < limit_val:
                heapq.heappush(heap, entry)
            else:
                if entry[0] > heap[0][0]:
                    heapq.heapreplace(heap, entry)
        heap.sort()
        events = [item[2] for item in heap]
    else:
        for path in sorted(iter_event_paths(events_dir)):
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
            vec_score = cosine_similarity(vec, node["centroid"])
            tag_score = tag_similarity(frag_tags, node["tags"])
            score = ((1 - tag_weight) * vec_score) + (tag_weight * tag_score)
            if score >= threshold and score > best_score:
                best_score = score
                best = node

        if best:
            count = int(best["count"])
            best["fragments"].append(frag_id)
            best["tags"].update(frag_tags)
            best["centroid"] = _merge_vectors(best["centroid"], count, vec, 1, round_digits=6)
            best["count"] = count + 1
        else:
            clusters.append({
                "fragments": [frag_id],
                "tags": set(frag_tags),
                "centroid": list(vec),
                "count": 1
            })
    return clusters

@dataclass(slots=True)
class _SynapseRecord:
    source: str
    target: str
    weight: float
    direction: Optional[Tuple[float, float, float]] = None
    relation: Optional[str] = None


def _synapse_get(synapse: Any, key: str, default: Any = None) -> Any:
    if isinstance(synapse, dict):
        return synapse.get(key, default)
    return getattr(synapse, key, default)


def _synapse_sort_key(synapse: Any) -> Tuple[float, str, str]:
    return (
        _safe_float(_synapse_get(synapse, "weight"), 0.0),
        str(_synapse_get(synapse, "source") or ""),
        str(_synapse_get(synapse, "target") or ""),
    )


def _synapse_to_payload(synapse: Any) -> Dict[str, Any]:
    payload = {
        "source": _synapse_get(synapse, "source"),
        "target": _synapse_get(synapse, "target"),
        "weight": round(_safe_float(_synapse_get(synapse, "weight"), 0.0), 4),
        "network_type": str(_synapse_get(synapse, "network_type", "memory_graph") or "memory_graph"),
    }
    direction = _synapse_get(synapse, "direction")
    if direction is not None:
        payload["direction"] = [round(float(value), 5) for value in direction[:3]]
    relation = _synapse_get(synapse, "relation")
    if relation:
        payload["relation"] = str(relation)
    return payload


class _BoundedSynapseCollector:
    def __init__(self, limit: Optional[int] = None) -> None:
        self.limit = limit if isinstance(limit, int) and limit > 0 else 0
        self._items: List[Any] = []
        self._heap: List[Tuple[Tuple[float, str, str], int, Any]] = []
        self._seen = 0
        self._counter = 0

    def add(self, synapse: Any) -> None:
        if not synapse:
            return
        self._seen += 1
        if self.limit <= 0:
            self._items.append(synapse)
            return
        key = _synapse_sort_key(synapse)
        entry = (key, self._counter, synapse)
        self._counter += 1
        if len(self._heap) < self.limit:
            heapq.heappush(self._heap, entry)
            return
        if entry[0] > self._heap[0][0]:
            heapq.heapreplace(self._heap, entry)

    def extend(self, synapses: Iterable[Any]) -> None:
        for synapse in synapses:
            self.add(synapse)

    @property
    def pruned_count(self) -> int:
        return max(0, self._seen - self.count)

    @property
    def count(self) -> int:
        return len(self._heap) if self.limit > 0 else len(self._items)

    def finalize(self) -> List[Any]:
        if self.limit <= 0:
            return list(self._items)
        return [entry[2] for entry in sorted(self._heap, key=lambda item: item[0], reverse=True)]


class _BoundedSynapseSpool:
    def __init__(self, path: Path, limit: Optional[int] = None) -> None:
        self.path = Path(path)
        self.limit = limit if isinstance(limit, int) and limit > 0 else 0
        self._heap: List[Tuple[Tuple[float, str, str], int, int]] = []
        self._seen = 0
        self._counter = 0
        self._row_count = 0
        self.conn: Optional[sqlite3.Connection] = None
        self._cleanup_files()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.execute("PRAGMA journal_mode=OFF")
        self.conn.execute("PRAGMA synchronous=OFF")
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS synapses ("
            "row_id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "sort_weight REAL NOT NULL, "
            "sort_source TEXT NOT NULL, "
            "sort_target TEXT NOT NULL, "
            "payload TEXT NOT NULL)"
        )

    def _cleanup_files(self) -> None:
        for suffix in ("", "-journal", "-wal", "-shm"):
            target = Path(str(self.path) + suffix)
            try:
                if target.exists():
                    target.unlink()
            except OSError:
                continue

    def _insert(self, synapse: Any, key: Tuple[float, str, str]) -> int:
        if self.conn is None:
            return 0
        payload = json.dumps(_synapse_to_payload(synapse), ensure_ascii=False)
        cursor = self.conn.execute(
            "INSERT INTO synapses(sort_weight, sort_source, sort_target, payload) VALUES (?, ?, ?, ?)",
            (float(key[0]), str(key[1]), str(key[2]), payload),
        )
        self._row_count += 1
        return int(cursor.lastrowid or 0)

    def _delete_row(self, row_id: int) -> None:
        if self.conn is None or row_id <= 0:
            return
        self.conn.execute("DELETE FROM synapses WHERE row_id = ?", (int(row_id),))
        self._row_count = max(0, self._row_count - 1)

    def add(self, synapse: Any) -> None:
        if not synapse:
            return
        self._seen += 1
        key = _synapse_sort_key(synapse)
        if self.limit <= 0:
            self._insert(synapse, key)
            return
        if len(self._heap) < self.limit:
            row_id = self._insert(synapse, key)
            heapq.heappush(self._heap, (key, self._counter, row_id))
            self._counter += 1
            return
        if key <= self._heap[0][0]:
            self._counter += 1
            return
        row_id = self._insert(synapse, key)
        _, _, dropped_row_id = heapq.heapreplace(self._heap, (key, self._counter, row_id))
        self._counter += 1
        self._delete_row(dropped_row_id)

    def extend(self, synapses: Iterable[Any]) -> None:
        for synapse in synapses:
            self.add(synapse)

    @property
    def pruned_count(self) -> int:
        return max(0, self._seen - self.count)

    @property
    def count(self) -> int:
        return len(self._heap) if self.limit > 0 else self._row_count

    def __len__(self) -> int:
        return self.count

    def __iter__(self):
        return self.iter_sorted()

    def iter_sorted(self) -> Iterable[Dict[str, Any]]:
        if self.conn is None:
            return
        rows = self.conn.execute(
            "SELECT payload FROM synapses ORDER BY sort_weight DESC, sort_source DESC, sort_target DESC, row_id DESC"
        )
        for row in rows:
            try:
                payload = json.loads(row[0]) if row and row[0] else None
            except Exception:
                payload = None
            if isinstance(payload, dict):
                yield payload

    def finalize(self):
        if self.conn is not None:
            self.conn.commit()
        return self

    def cleanup(self) -> None:
        if self.conn is not None:
            try:
                self.conn.close()
            except Exception:
                pass
            self.conn = None
        self._heap.clear()
        self._row_count = 0
        self._cleanup_files()


def build_synaptic_links(
    neurons,
    threshold=0.91,
    *,
    max_edges: Optional[int] = None,
    max_pairs: Optional[int] = None,
    max_edges_per_neuron: Optional[int] = None,
    include_direction: bool = True,
    batch_size: Optional[int] = None,
    compact_records: bool = False,
    spool_path: Optional[Path] = None,
    return_stats: bool = False,
):
    pairs_evaluated = 0
    truncated = False
    edge_cap = max_edges if (isinstance(max_edges, int) and max_edges > 0) else None
    pair_cap = max_pairs if (isinstance(max_pairs, int) and max_pairs > 0) else None
    per_node_cap = max_edges_per_neuron if (isinstance(max_edges_per_neuron, int) and max_edges_per_neuron > 0) else None
    emit_batch_size = batch_size if (isinstance(batch_size, int) and batch_size > 0) else 256
    collector: Any
    if spool_path is not None:
        collector = _BoundedSynapseSpool(spool_path, edge_cap)
    else:
        collector = _BoundedSynapseCollector(edge_cap)
    edge_batch: List[Any] = []

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
            pairs_evaluated += 1
            vec_a = source.get("vector")
            vec_b = target.get("vector")
            if vec_a and vec_b:
                sim = cosine_similarity(vec_a, vec_b)
                if sim >= threshold:
                    direction = None
                    if include_direction:
                        pos_a = source.get("position")
                        pos_b = target.get("position")
                        if pos_a and pos_b:
                            delta = [pos_b[k] - pos_a[k] for k in range(3)]
                            norm = math.sqrt(sum(d * d for d in delta))
                            if norm > 1e-6:
                                direction = tuple(round(d / norm, 5) for d in delta)
                    if compact_records:
                        payload = _SynapseRecord(
                            source=str(source["id"]),
                            target=str(target["id"]),
                            weight=round(sim, 4),
                            direction=direction,
                        )
                    else:
                        payload = {
                            "source": source["id"],
                            "target": target["id"],
                            "weight": round(sim, 4),
                            "network_type": "memory_graph",
                        }
                        if include_direction:
                            payload["direction"] = list(direction) if direction is not None else None
                    edge_batch.append(payload)
                    source_edges += 1
                    if len(edge_batch) >= emit_batch_size:
                        collector.extend(edge_batch)
                        edge_batch.clear()
        if truncated:
            break
    if edge_batch:
        collector.extend(edge_batch)
    synapses = collector.finalize()
    if edge_cap is not None and collector.pruned_count > 0:
        truncated = True
    if return_stats:
        return synapses, {
            "pairs_evaluated": pairs_evaluated,
            "edge_count": len(synapses),
            "truncated": truncated,
            "budget_pruned": collector.pruned_count,
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


def _atomic_write_json_stream(path: Path, write_fn) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            write_fn(fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        raise


def _write_json_array_stream(handle, items: Iterable[Any]) -> None:
    handle.write("[")
    first = True
    for item in items:
        if not first:
            handle.write(",")
        json.dump(item, handle, ensure_ascii=False)
        first = False
    handle.write("]")


def _neural_build_state_path(child: str) -> Path:
    return Path("AI_Children") / child / "memory" / "neural" / "neural_build_state.json"


def _neural_dirty_index_path(child: str) -> Path:
    return Path("AI_Children") / child / "memory" / "neural" / "neural_dirty_index.json"


def _neural_snapshot_path(child: str) -> Path:
    return Path("AI_Children") / child / "memory" / "neural" / "neural_memory_snapshot_csr.json"


def _neural_spill_path(child: str) -> Path:
    return Path("AI_Children") / child / "memory" / "neural" / "neural_memory_spill.sqlite"


def _neural_synapse_spool_path(child: str) -> Path:
    return Path("AI_Children") / child / "memory" / "neural" / "neural_synapse_spool.sqlite"


def _custom_transformer_usage_path(child: str) -> Path:
    return Path("AI_Children") / child / "memory" / "neural" / "custom_transformer_usage.json"


def _memory_index_db_path(child: str) -> Path:
    return Path("AI_Children") / child / "memory" / "memory_map.sqlite"


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


def _quantize_sequence(values: Optional[Iterable[Any]], bits: int) -> Optional[Dict[str, Any]]:
    if values is None:
        return None
    raw = []
    for value in values:
        try:
            raw.append(float(value))
        except (TypeError, ValueError):
            raw.append(0.0)
    if not raw:
        return None
    qmax = (2 ** (bits - 1)) - 1
    max_abs = max(abs(value) for value in raw)
    if max_abs <= 1e-12:
        return {"bits": bits, "scale": 1.0, "values": [0 for _ in raw]}
    scale = max_abs / float(qmax)
    quantized = [int(max(-qmax, min(qmax, round(value / scale)))) for value in raw]
    return {"bits": bits, "scale": round(scale, 8), "values": quantized}


def _dequantize_sequence(payload: Any) -> List[float]:
    if isinstance(payload, list):
        return [float(value) for value in payload]
    if not isinstance(payload, dict):
        return []
    values = payload.get("values") if isinstance(payload.get("values"), list) else []
    scale = _safe_float(payload.get("scale"), 1.0)
    if abs(scale) <= 1e-12:
        scale = 1.0
    return [round(_safe_float(value, 0.0) * scale, 6) for value in values]


_SPILL_HIGH_IMPORTANCE_TAGS = {
    "care", "emotion", "feeling", "heart", "trauma", "shadow", "unresolved", "high_conflict",
    "bond", "attachment", "love", "grief", "memory", "identity", "core",
}


_SPILL_MEDIUM_IMPORTANCE_TAGS = {
    "intensity", "stress", "risk", "fear", "hope", "joy", "pain", "voice", "vision", "meaning",
}


def _spill_importance_score(neuron: Dict[str, Any]) -> float:
    tags = {str(tag).lower() for tag in (neuron.get("tags") or []) if tag}
    tag_score = 0.0
    if tags & _SPILL_HIGH_IMPORTANCE_TAGS:
        tag_score = 1.0
    elif tags & _SPILL_MEDIUM_IMPORTANCE_TAGS:
        tag_score = 0.6
    symbolic_score = _clamp(abs(_safe_float(neuron.get("symbolic_density"), 0.0)), 0.0, 1.0)
    activation_history = [abs(_safe_float(value, 0.0)) for value in (neuron.get("activation_history") or []) if isinstance(value, (int, float))]
    activation_score = _clamp(max(activation_history) if activation_history else 0.0, 0.0, 1.0)
    fragment_score = _clamp(len(neuron.get("fragments", []) or []) / 64.0, 0.0, 1.0)
    reuse_score = _clamp(_safe_float(neuron.get("spill_reuse_count"), 0.0) / 3.0, 0.0, 1.0)
    return _clamp(
        (0.35 * tag_score)
        + (0.2 * symbolic_score)
        + (0.15 * activation_score)
        + (0.15 * fragment_score)
        + (0.15 * reuse_score),
        0.0,
        1.0,
    )


def _spill_precision_level(neuron: Dict[str, Any], policy: Dict[str, Any]) -> str:
    mode = str(policy.get("spill_precision_mode", "adaptive") or "adaptive").strip().lower()
    if mode in {"float", "int16", "int8"}:
        return mode
    score = _spill_importance_score(neuron)
    high = _safe_float(policy.get("spill_high_precision_threshold"), 0.68)
    medium = _safe_float(policy.get("spill_medium_precision_threshold"), 0.35)
    if score >= high:
        return "float"
    if score >= medium:
        return "int16"
    return "int8"


def _encode_spilled_neuron(neuron: Dict[str, Any], policy: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(neuron)
    precision = _spill_precision_level(payload, policy)
    payload["spill_precision"] = precision
    vector_digits = int(policy.get("vector_round_digits", 6))
    pos_digits = int(policy.get("position_round_digits", 4))
    if precision == "float":
        payload["vector"] = _round_vector(payload.get("vector"), vector_digits)
        payload["position"] = _round_vector(payload.get("position"), pos_digits)
    else:
        bits = 16 if precision == "int16" else 8
        vector_q = _quantize_sequence(payload.pop("vector", None), bits)
        position_q = _quantize_sequence(payload.pop("position", None), bits)
        if vector_q is not None:
            payload["vector_q"] = vector_q
        if position_q is not None:
            payload["position_q"] = position_q
    return payload


def _decode_spilled_neuron(payload: Dict[str, Any]) -> Dict[str, Any]:
    neuron = dict(payload)
    neuron.pop("spill_precision", None)
    vector_q = neuron.pop("vector_q", None)
    position_q = neuron.pop("position_q", None)
    if vector_q is not None:
        neuron["vector"] = _dequantize_sequence(vector_q)
    if position_q is not None:
        neuron["position"] = _dequantize_sequence(position_q)
    return neuron


def _lightweight_neuron_view(neuron: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": neuron.get("id"),
        "vector": list(neuron.get("vector") or []),
        "position": list(neuron.get("position") or []),
    }


def _dense_float_array(values: Optional[Iterable[Any]]) -> array:
    dense = array("f")
    if values is None:
        return dense
    for value in values:
        dense.append(float(_safe_float(value, 0.0)))
    return dense


class _NeuralSpillStore:
    def __init__(self, child: str, policy: Dict[str, Any]) -> None:
        self.child = child
        self.policy = policy if isinstance(policy, dict) else {}
        self.enabled = bool(self.policy.get("spill_to_disk_enabled", False))
        self.max_hot_neurons = max(0, int(self.policy.get("max_hot_neurons", 0) or 0))
        self.spill_after_batches = max(0, int(self.policy.get("spill_after_batches", 0) or 0))
        self.path = _neural_spill_path(child)
        self.conn: Optional[sqlite3.Connection] = None
        self.meta: Dict[str, Dict[str, Any]] = {}
        if not self.enabled or self.max_hot_neurons <= 0:
            self.enabled = False
            return
        self._cleanup_files()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS spill_neurons ("
            "node_id TEXT PRIMARY KEY, "
            "payload TEXT NOT NULL, "
            "last_used REAL DEFAULT 0.0, "
            "fragment_count INTEGER DEFAULT 0)"
        )
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS spill_fragments ("
            "fragment_id TEXT PRIMARY KEY, "
            "node_id TEXT NOT NULL)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_spill_fragments_node_id ON spill_fragments(node_id)"
        )
        self.conn.commit()

    def _cleanup_files(self) -> None:
        for suffix in ("", "-journal", "-wal", "-shm"):
            target = Path(str(self.path) + suffix)
            try:
                if target.exists():
                    target.unlink()
            except OSError:
                continue

    @staticmethod
    def _chunks(values: Iterable[str], size: int = 250) -> Iterable[List[str]]:
        batch: List[str] = []
        for value in values:
            if not value:
                continue
            batch.append(str(value))
            if len(batch) >= size:
                yield batch
                batch = []
        if batch:
            yield batch

    @staticmethod
    def _spill_sort_key(neuron: Dict[str, Any]) -> Tuple[float, int, str]:
        last_used_ts = _parse_iso_timestamp(neuron.get("last_used")) or 0.0
        fragment_count = len(neuron.get("fragments", []) or [])
        return (last_used_ts, -fragment_count, str(neuron.get("id") or ""))

    @staticmethod
    def _meta_from_neuron(neuron: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": str(neuron.get("id") or ""),
            "vector": _dense_float_array(neuron.get("vector")),
            "position": _dense_float_array(neuron.get("position")),
            "tags": tuple(str(tag) for tag in (neuron.get("tags") or []) if tag),
            "region": str(neuron.get("region") or ""),
            "last_used_ts": _parse_iso_timestamp(neuron.get("last_used")) or 0.0,
            "fragment_count": len(neuron.get("fragments", []) or []),
            "reuse_count": int(_safe_float(neuron.get("spill_reuse_count"), 0.0)),
        }

    def count(self) -> int:
        return len(self.meta)

    def has_spills(self) -> bool:
        return bool(self.meta)

    def node_ids(self) -> List[str]:
        return list(self.meta.keys())

    def scored_entries(self) -> List[Tuple[str, int, float]]:
        return [
            (
                node_id,
                int(meta.get("fragment_count", 0) or 0),
                float(meta.get("last_used_ts", 0.0) or 0.0),
            )
            for node_id, meta in self.meta.items()
        ]

    def _store_neurons(self, neurons: List[Dict[str, Any]]) -> int:
        if not self.enabled or self.conn is None or not neurons:
            return 0
        payload_rows: List[Tuple[str, str, float, int]] = []
        fragment_rows: List[Tuple[str, str]] = []
        node_ids: List[str] = []
        for neuron in neurons:
            node_id = str(neuron.get("id") or "").strip()
            if not node_id:
                continue
            meta = self._meta_from_neuron(neuron)
            encoded = _encode_spilled_neuron(neuron, self.policy)
            payload_rows.append(
                (
                    node_id,
                    json.dumps(encoded, ensure_ascii=False),
                    float(meta.get("last_used_ts", 0.0) or 0.0),
                    int(meta.get("fragment_count", 0) or 0),
                )
            )
            node_ids.append(node_id)
            self.meta[node_id] = meta
            for fragment_id in neuron.get("fragments", []) or []:
                if fragment_id:
                    fragment_rows.append((str(fragment_id), node_id))
        if not payload_rows:
            return 0
        with self.conn:
            self.conn.executemany("DELETE FROM spill_fragments WHERE node_id = ?", [(node_id,) for node_id in node_ids])
            self.conn.executemany(
                "INSERT OR REPLACE INTO spill_neurons(node_id, payload, last_used, fragment_count) VALUES (?, ?, ?, ?)",
                payload_rows,
            )
            if fragment_rows:
                self.conn.executemany(
                    "INSERT OR REPLACE INTO spill_fragments(fragment_id, node_id) VALUES (?, ?)",
                    fragment_rows,
                )
        return len(payload_rows)

    def spill_overflow(self, neurons: List[Dict[str, Any]]) -> int:
        if not self.enabled or self.conn is None or self.max_hot_neurons <= 0:
            return 0
        if len(neurons) <= self.max_hot_neurons:
            return 0
        overflow = len(neurons) - self.max_hot_neurons
        scored = [
            (self._spill_sort_key(neuron), idx, neuron)
            for idx, neuron in enumerate(neurons)
            if neuron.get("id")
        ]
        if not scored:
            return 0
        scored.sort(key=lambda item: item[0])
        selected = scored[:overflow]
        drop_indices = {idx for _, idx, _ in selected}
        spill_list = [neuron for _, _, neuron in selected]
        spilled = self._store_neurons(spill_list)
        if spilled:
            neurons[:] = [neuron for idx, neuron in enumerate(neurons) if idx not in drop_indices]
        return spilled

    def maybe_flush(self, neurons: List[Dict[str, Any]], batches_run: int) -> int:
        if not self.enabled or self.spill_after_batches <= 0:
            return 0
        if batches_run <= 0 or (batches_run % self.spill_after_batches) != 0:
            return 0
        return self.spill_overflow(neurons)

    def best_match(self, candidate: Dict[str, Any], tag_weight: float) -> Tuple[Optional[str], float]:
        if not self.enabled or not self.meta:
            return None, 0.0
        best_id: Optional[str] = None
        best_score = 0.0
        for node_id, meta in self.meta.items():
            score = _score_candidate_match(meta, candidate, tag_weight)
            if score > best_score:
                best_id = node_id
                best_score = score
        return best_id, best_score

    def _remove_node_ids(self, node_ids: List[str]) -> Set[str]:
        removed_fragments: Set[str] = set()
        if not self.enabled or self.conn is None or not node_ids:
            return removed_fragments
        unique_ids = [node_id for node_id in dict.fromkeys(node_ids) if node_id]
        if not unique_ids:
            return removed_fragments
        for chunk in self._chunks(unique_ids):
            placeholders = ",".join("?" for _ in chunk)
            rows = self.conn.execute(
                f"SELECT fragment_id FROM spill_fragments WHERE node_id IN ({placeholders})",
                chunk,
            ).fetchall()
            removed_fragments.update(str(row[0]) for row in rows if row and row[0])
            with self.conn:
                self.conn.execute(f"DELETE FROM spill_neurons WHERE node_id IN ({placeholders})", chunk)
                self.conn.execute(f"DELETE FROM spill_fragments WHERE node_id IN ({placeholders})", chunk)
        for node_id in unique_ids:
            self.meta.pop(node_id, None)
        return removed_fragments

    def drop_nodes(self, node_ids: List[str]) -> Set[str]:
        return self._remove_node_ids(node_ids)

    def activate_node(self, node_id: str, neurons: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not self.enabled or self.conn is None or not node_id:
            return None
        meta = self.meta.get(str(node_id), {})
        row = self.conn.execute("SELECT payload FROM spill_neurons WHERE node_id = ?", (str(node_id),)).fetchone()
        if not row or not row[0]:
            self.meta.pop(str(node_id), None)
            return None
        try:
            payload = json.loads(row[0])
        except Exception:
            payload = None
        self._remove_node_ids([str(node_id)])
        if isinstance(payload, dict):
            neuron = _decode_spilled_neuron(payload)
            neuron["spill_reuse_count"] = int(_safe_float(meta.get("reuse_count"), _safe_float(neuron.get("spill_reuse_count"), 0.0))) + 1
            neurons.append(neuron)
            return neuron
        return None

    def activate_for_fragments(self, fragment_ids: Set[str], neurons: List[Dict[str, Any]]) -> int:
        if not self.enabled or self.conn is None or not fragment_ids:
            return 0
        node_ids: List[str] = []
        seen: Set[str] = set()
        for chunk in self._chunks(fragment_ids):
            placeholders = ",".join("?" for _ in chunk)
            rows = self.conn.execute(
                f"SELECT DISTINCT node_id FROM spill_fragments WHERE fragment_id IN ({placeholders})",
                chunk,
            ).fetchall()
            for row in rows:
                node_id = str(row[0]) if row and row[0] else ""
                if node_id and node_id not in seen:
                    seen.add(node_id)
                    node_ids.append(node_id)
        activated = 0
        for node_id in node_ids:
            if self.activate_node(node_id, neurons):
                activated += 1
        return activated

    def iter_storage_neurons(self, hot_neurons: List[Dict[str, Any]], policy: Dict[str, Any], *, compact: bool) -> Iterable[Dict[str, Any]]:
        for neuron in hot_neurons:
            yield _compact_neuron_for_storage(neuron, policy) if compact else neuron
        if not self.enabled or self.conn is None or not self.meta:
            return
        rows = self.conn.execute("SELECT payload FROM spill_neurons ORDER BY node_id")
        for row in rows:
            try:
                payload = json.loads(row[0]) if row and row[0] else None
            except Exception:
                payload = None
            if isinstance(payload, dict):
                neuron = _decode_spilled_neuron(payload)
                yield _compact_neuron_for_storage(neuron, policy) if compact else neuron

    def iter_link_neurons(self) -> Iterable[Dict[str, Any]]:
        if not self.enabled or not self.meta:
            return
        for node_id in sorted(self.meta.keys()):
            meta = self.meta.get(node_id) or {}
            yield {
                "id": node_id,
                "vector": meta.get("vector") or array("f"),
                "position": meta.get("position") or array("f"),
            }

    def materialize_all(self, neurons: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not self.enabled or self.conn is None or not self.meta:
            return list(neurons)
        materialized = list(neurons)
        rows = self.conn.execute("SELECT payload FROM spill_neurons ORDER BY node_id")
        for row in rows:
            try:
                payload = json.loads(row[0]) if row and row[0] else None
            except Exception:
                payload = None
            if isinstance(payload, dict):
                materialized.append(_decode_spilled_neuron(payload))
        return materialized

    def cleanup(self) -> None:
        if self.conn is not None:
            try:
                self.conn.close()
            except Exception:
                pass
            self.conn = None
        self.meta.clear()
        self._cleanup_files()


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


def _persist_memory_index_db(child: str, index_data: Dict[str, Any], *, source_mtime_ns: Optional[int] = None) -> None:
    db_path = _memory_index_db_path(child)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("CREATE TABLE IF NOT EXISTS fragments (frag_id TEXT PRIMARY KEY, tier TEXT, filename TEXT, last_seen TEXT, timestamp TEXT, importance REAL, mtime_ns INTEGER, size_bytes INTEGER, tags_json TEXT)")
        conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
        rows = []
        for frag_id, meta in index_data.items():
            if not frag_id or not isinstance(meta, dict):
                continue
            tags = meta.get("tags") if isinstance(meta.get("tags"), list) else []
            rows.append((
                str(frag_id),
                str(meta.get("tier") or ""),
                str(meta.get("filename") or ""),
                str(meta.get("last_seen") or ""),
                str(meta.get("timestamp") or ""),
                float(_safe_float(meta.get("importance"), 0.0)),
                int(_safe_float(meta.get("mtime_ns"), 0.0)),
                int(_safe_float(meta.get("size_bytes"), 0.0)),
                json.dumps([str(tag) for tag in tags if tag], ensure_ascii=False),
            ))
        with conn:
            conn.execute("DELETE FROM fragments")
            if rows:
                conn.executemany(
                    "INSERT OR REPLACE INTO fragments(frag_id, tier, filename, last_seen, timestamp, importance, mtime_ns, size_bytes, tags_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    rows,
                )
            conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES ('source_mtime_ns', ?)", (str(int(source_mtime_ns or 0)),))
    finally:
        conn.close()


class _MemoryIndexStore:
    def __init__(self, child: str) -> None:
        self.child = child
        self.json_path = Path("AI_Children") / child / "memory" / "memory_map.json"
        self.db_path = _memory_index_db_path(child)
        self.conn: Optional[sqlite3.Connection] = None
        self._count = 0
        if not self.json_path.exists() and not self.db_path.exists():
            return
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("CREATE TABLE IF NOT EXISTS fragments (frag_id TEXT PRIMARY KEY, tier TEXT, filename TEXT, last_seen TEXT, timestamp TEXT, importance REAL, mtime_ns INTEGER, size_bytes INTEGER, tags_json TEXT)")
        self.conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
        self._ensure_current()
        row = self.conn.execute("SELECT COUNT(*) FROM fragments").fetchone()
        self._count = int(row[0] or 0) if row else 0

    def _meta_value(self, key: str) -> Optional[str]:
        if self.conn is None:
            return None
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (str(key),)).fetchone()
        return str(row[0]) if row and row[0] is not None else None

    def _ensure_current(self) -> None:
        if self.conn is None or not self.json_path.exists():
            return
        try:
            source_mtime_ns = int(self.json_path.stat().st_mtime_ns)
        except OSError:
            return
        current = self._meta_value("source_mtime_ns")
        row = self.conn.execute("SELECT 1 FROM fragments LIMIT 1").fetchone()
        if current == str(source_mtime_ns) and row is not None:
            return
        try:
            with self.json_path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except Exception:
            return
        if isinstance(payload, dict):
            _persist_memory_index_db(self.child, payload, source_mtime_ns=source_mtime_ns)
            self.close()
            self.conn = sqlite3.connect(str(self.db_path))
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA synchronous=NORMAL")

    def __len__(self) -> int:
        return int(self._count)

    def __bool__(self) -> bool:
        return self._count > 0

    @staticmethod
    def _row_to_meta(row: Any) -> Dict[str, Any]:
        if not row:
            return {}
        tags = []
        try:
            payload = json.loads(row[8]) if row[8] else []
            if isinstance(payload, list):
                tags = [str(tag) for tag in payload if tag]
        except Exception:
            tags = []
        meta = {
            "tier": str(row[1] or "") or None,
            "filename": str(row[2] or "") or None,
            "last_seen": str(row[3] or "") or None,
            "timestamp": str(row[4] or "") or None,
            "importance": float(_safe_float(row[5], 0.0)),
            "mtime_ns": int(_safe_float(row[6], 0.0)),
            "size_bytes": int(_safe_float(row[7], 0.0)),
            "tags": tags,
        }
        return {key: value for key, value in meta.items() if value not in (None, "", [])}

    def get(self, frag_id: str, default: Any = None) -> Any:
        if self.conn is None or not frag_id:
            return default
        row = self.conn.execute(
            "SELECT frag_id, tier, filename, last_seen, timestamp, importance, mtime_ns, size_bytes, tags_json FROM fragments WHERE frag_id = ?",
            (str(frag_id),),
        ).fetchone()
        if not row:
            return default
        return self._row_to_meta(row)

    def __contains__(self, frag_id: object) -> bool:
        if self.conn is None or not frag_id:
            return False
        row = self.conn.execute("SELECT 1 FROM fragments WHERE frag_id = ? LIMIT 1", (str(frag_id),)).fetchone()
        return bool(row)

    def items(self) -> Iterable[Tuple[str, Dict[str, Any]]]:
        if self.conn is None:
            return []
        cursor = self.conn.execute(
            "SELECT frag_id, tier, filename, last_seen, timestamp, importance, mtime_ns, size_bytes, tags_json FROM fragments"
        )
        for row in cursor:
            yield str(row[0]), self._row_to_meta(row)

    def keys(self) -> Iterable[str]:
        if self.conn is None:
            return []
        cursor = self.conn.execute("SELECT frag_id FROM fragments")
        for row in cursor:
            if row and row[0]:
                yield str(row[0])

    def iter_human_prune_candidates(
        self,
        retention: Dict[str, Any],
        settings: Dict[str, Any],
        now_ts: float,
    ) -> Iterable[Tuple[str, Dict[str, Any], Path]]:
        if self.conn is None:
            return []
        limit = int(settings.get("limit") or 0)
        if limit <= 0:
            return []
        fetch_limit = max(limit, limit * int(settings.get("candidate_multiplier") or 1))
        cutoff_iso = datetime.fromtimestamp(
            now_ts - (float(settings.get("min_age_hours") or 0.0) * 3600.0),
            timezone.utc,
        ).isoformat()
        cursor = self.conn.execute(
            """
            SELECT frag_id, tier, filename, last_seen, timestamp, importance, mtime_ns, size_bytes, tags_json
            FROM fragments
            WHERE (tier IS NULL OR tier = '' OR tier IN ('short', 'long', 'cold'))
              AND importance <= ?
              AND size_bytes >= ?
              AND (
                    (last_seen IS NOT NULL AND last_seen != '' AND last_seen <= ?)
                 OR (timestamp IS NOT NULL AND timestamp != '' AND timestamp <= ?)
              )
            ORDER BY size_bytes DESC, importance ASC, COALESCE(NULLIF(last_seen, ''), NULLIF(timestamp, ''), '') ASC
            LIMIT ?
            """,
            (
                float(settings.get("max_importance") or 0.0),
                int(settings.get("min_size_bytes") or 0),
                cutoff_iso,
                cutoff_iso,
                fetch_limit,
            ),
        )
        selected: List[Tuple[str, Dict[str, Any], Path]] = []
        for row in cursor:
            frag_id = str(row[0])
            meta = self._row_to_meta(row)
            candidate = _human_prune_candidate_from_meta(self.child, frag_id, meta, retention, settings, now_ts)
            if candidate is None:
                continue
            candidate_id, candidate_meta, candidate_path, _ = candidate
            selected.append((candidate_id, candidate_meta, candidate_path))
            if len(selected) >= limit:
                break
        return selected

    def close(self) -> None:
        if self.conn is not None:
            try:
                self.conn.close()
            except Exception:
                pass
            self.conn = None


def _load_memory_index(child: str) -> Any:
    store = _MemoryIndexStore(child)
    if store:
        return store
    if store.conn is not None:
        store.close()
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
    index: Any,
    previous_signatures: Dict[str, str],
) -> Tuple[List[str], Dict[str, str], Set[str]]:
    dirty: List[str] = []
    changed: Dict[str, str] = {}
    seen: Set[str] = set()
    for frag_id, meta in index.items():
        if not frag_id or not isinstance(meta, dict):
            continue
        seen.add(str(frag_id))
        signature = _fragment_signature(child, frag_id, meta)
        if not signature:
            continue
        if previous_signatures.get(frag_id) != signature:
            dirty.append(frag_id)
            changed[frag_id] = signature
    removed = {fid for fid in previous_signatures.keys() if fid not in seen}
    return dirty, changed, removed


def _prune_neurons_to_valid_fragments(
    neurons: List[Dict[str, Any]],
    valid_ids: Any,
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
        clipped_fragments = _clip_sequence_with_anchors(
            original_fragments,
            frag_limit,
            int(policy.get("fragment_anchor_count", 0)),
        )
        if clipped_fragments != original_fragments:
            neuron["fragments"] = clipped_fragments
            changes += 1

        original_tags = [str(tag) for tag in neuron.get("tags", []) if tag]
        unique_tags = list(dict.fromkeys(original_tags))
        clipped_tags = _clip_sequence_with_anchors(unique_tags, tag_limit, int(policy.get("tag_anchor_count", 0)))
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


def _enforce_total_neuron_budget(
    neurons: List[Dict[str, Any]],
    max_neurons_total: int,
    spill_store: Optional[_NeuralSpillStore] = None,
) -> Tuple[int, Set[str]]:
    if max_neurons_total <= 0:
        return 0, set()
    spill_entries = spill_store.scored_entries() if spill_store is not None else []
    total_neurons = len(neurons) + len(spill_entries)
    if total_neurons <= max_neurons_total:
        return 0, set()
    overflow = total_neurons - max_neurons_total
    scored: List[Tuple[int, float, str, Any]] = []
    for idx, neuron in enumerate(neurons):
        fragment_ids = [str(fid) for fid in (neuron.get("fragments", []) or []) if fid]
        fragment_count = len(fragment_ids)
        last_used_ts = _parse_iso_timestamp(neuron.get("last_used")) or 0.0
        scored.append((fragment_count, last_used_ts, "hot", (idx, fragment_ids)))
    for node_id, fragment_count, last_used_ts in spill_entries:
        scored.append((fragment_count, last_used_ts, "spill", str(node_id)))
    scored.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    hot_drop_indices: Set[int] = set()
    spill_drop_ids: List[str] = []
    removed_fragments: Set[str] = set()
    for _, _, kind, payload in scored[:overflow]:
        if kind == "hot":
            idx, fragment_ids = payload
            hot_drop_indices.add(idx)
            removed_fragments.update(fragment_ids)
        else:
            spill_drop_ids.append(str(payload))
    if hot_drop_indices:
        neurons[:] = [neuron for idx, neuron in enumerate(neurons) if idx not in hot_drop_indices]
    if spill_drop_ids and spill_store is not None:
        removed_fragments.update(spill_store.drop_nodes(spill_drop_ids))
    return overflow, removed_fragments


def _enforce_synapse_budget(synapses: Any, max_synapses_total: int) -> int:
    if max_synapses_total <= 0:
        return 0
    if isinstance(synapses, _BoundedSynapseSpool):
        return 0
    if len(synapses) <= max_synapses_total:
        return 0
    synapses.sort(
        key=lambda synapse: (
            _safe_float(_synapse_get(synapse, "weight"), 0.0),
            str(_synapse_get(synapse, "source") or ""),
            str(_synapse_get(synapse, "target") or ""),
        ),
        reverse=True,
    )
    overflow = len(synapses) - max_synapses_total
    del synapses[max_synapses_total:]
    return overflow


def _apply_synapse_connection_caps(synapses: Any, policy: Dict[str, Any]) -> Tuple[Any, Dict[str, int]]:
    max_degree = int(policy.get("max_connection_degree", 0) or 0)
    min_weight = _safe_float(policy.get("min_synapse_weight"), 0.0)
    if max_degree <= 0 and min_weight <= 0.0:
        return synapses, {"weight_pruned": 0, "degree_pruned": 0}

    if isinstance(synapses, _BoundedSynapseSpool):
        ordered = synapses.iter_sorted()
    else:
        synapses.sort(
            key=lambda synapse: (
                _safe_float(_synapse_get(synapse, "weight"), 0.0),
                str(_synapse_get(synapse, "source") or ""),
                str(_synapse_get(synapse, "target") or ""),
            ),
            reverse=True,
        )
        ordered = synapses

    kept: List[Any] = []
    degree: Dict[str, int] = {}
    weight_pruned = 0
    degree_pruned = 0
    for synapse in ordered:
        source = str(_synapse_get(synapse, "source") or "")
        target = str(_synapse_get(synapse, "target") or "")
        if not source or not target:
            continue
        weight = _safe_float(_synapse_get(synapse, "weight"), 0.0)
        if min_weight > 0.0 and weight < min_weight:
            weight_pruned += 1
            continue
        if max_degree > 0 and (degree.get(source, 0) >= max_degree or degree.get(target, 0) >= max_degree):
            degree_pruned += 1
            continue
        kept.append(synapse)
        degree[source] = degree.get(source, 0) + 1
        degree[target] = degree.get(target, 0) + 1

    if isinstance(synapses, _BoundedSynapseSpool):
        synapses.cleanup()
    return kept, {"weight_pruned": weight_pruned, "degree_pruned": degree_pruned}


def _pending_fragment_sort_key(fragment_id: str, index: Any) -> Tuple[float, float]:
    meta = index.get(fragment_id, {})
    ts = _parse_iso_timestamp(meta.get("last_seen") or meta.get("timestamp")) or 0.0
    importance = _safe_float(meta.get("importance"), 0.0)
    return ts, importance


def _trim_pending_fragment_ids(
    fragment_ids: Iterable[str],
    index: Any,
    max_pending_dirty_fragments: int,
) -> Tuple[List[str], int]:
    deduped = [str(fragment_id) for fragment_id in dict.fromkeys(fragment_ids) if fragment_id]
    if max_pending_dirty_fragments <= 0 or len(deduped) <= max_pending_dirty_fragments:
        ordered = sorted(deduped, key=lambda fragment_id: _pending_fragment_sort_key(fragment_id, index), reverse=True)
        return ordered, 0
    kept = heapq.nlargest(
        max_pending_dirty_fragments,
        deduped,
        key=lambda fragment_id: _pending_fragment_sort_key(fragment_id, index),
    )
    ordered = sorted(kept, key=lambda fragment_id: _pending_fragment_sort_key(fragment_id, index), reverse=True)
    return ordered, len(deduped) - len(ordered)


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
    index: Any,
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


def _snapshot_synapse_sort_key(synapse: Any, node_to_index: Dict[str, int]) -> Tuple[int, int, int]:
    source = node_to_index.get(str(_synapse_get(synapse, "source")))
    target = node_to_index.get(str(_synapse_get(synapse, "target")))
    if source is None or target is None:
        return (1, 0, 0)
    return (0, source, target)


def _save_sparse_snapshot_by_ids(child: str, node_ids: List[str], synapses: Any) -> None:
    path = _neural_snapshot_path(child)
    node_to_index = {node_id: idx for idx, node_id in enumerate(node_ids)}
    row_counts = [0 for _ in node_ids]
    edge_count = 0

    snapshot_edges: Optional[List[Tuple[int, int, float]]] = None
    if isinstance(synapses, _BoundedSynapseSpool):
        snapshot_edges = []
        for synapse in synapses.iter_sorted():
            source = node_to_index.get(str(_synapse_get(synapse, "source")))
            target = node_to_index.get(str(_synapse_get(synapse, "target")))
            if source is None or target is None:
                continue
            snapshot_edges.append((int(source), int(target), round(_safe_float(_synapse_get(synapse, "weight"), 0.0), 4)))
        snapshot_edges.sort()
        edge_count = len(snapshot_edges)
        for source, _, _ in snapshot_edges:
            row_counts[source] += 1
    elif synapses:
        synapses.sort(key=lambda synapse: _snapshot_synapse_sort_key(synapse, node_to_index))
        for synapse in synapses:
            source = node_to_index.get(str(_synapse_get(synapse, "source")))
            target = node_to_index.get(str(_synapse_get(synapse, "target")))
            if source is None or target is None:
                continue
            row_counts[source] += 1
            edge_count += 1

    def _iter_indptr() -> Iterable[int]:
        running = 0
        yield 0
        for count in row_counts:
            running += int(count)
            yield running

    def _iter_indices() -> Iterable[int]:
        if snapshot_edges is not None:
            for _, target, _ in snapshot_edges:
                yield int(target)
            return
        for synapse in synapses:
            source = node_to_index.get(str(_synapse_get(synapse, "source")))
            target = node_to_index.get(str(_synapse_get(synapse, "target")))
            if source is None or target is None:
                continue
            yield int(target)

    def _iter_weights() -> Iterable[float]:
        if snapshot_edges is not None:
            for _, _, weight in snapshot_edges:
                yield float(weight)
            return
        for synapse in synapses:
            source = node_to_index.get(str(_synapse_get(synapse, "source")))
            target = node_to_index.get(str(_synapse_get(synapse, "target")))
            if source is None or target is None:
                continue
            yield round(_safe_float(_synapse_get(synapse, "weight"), 0.0), 4)

    updated_at = datetime.now(timezone.utc).isoformat()

    def _write(handle) -> None:
        handle.write("{")
        handle.write(json.dumps("format"))
        handle.write(":")
        json.dump("csr_v1", handle, ensure_ascii=False)
        handle.write(",")
        handle.write(json.dumps("updated_at"))
        handle.write(":")
        json.dump(updated_at, handle, ensure_ascii=False)
        handle.write(",")
        handle.write(json.dumps("node_ids"))
        handle.write(":")
        _write_json_array_stream(handle, node_ids)
        handle.write(",")
        handle.write(json.dumps("indptr"))
        handle.write(":")
        _write_json_array_stream(handle, _iter_indptr())
        handle.write(",")
        handle.write(json.dumps("indices"))
        handle.write(":")
        _write_json_array_stream(handle, _iter_indices())
        handle.write(",")
        handle.write(json.dumps("weights"))
        handle.write(":")
        _write_json_array_stream(handle, _iter_weights())
        handle.write(",")
        handle.write(json.dumps("edge_count"))
        handle.write(":")
        json.dump(edge_count, handle, ensure_ascii=False)
        handle.write("}")

    _atomic_write_json_stream(path, _write)


def _save_sparse_snapshot(child: str, neurons: List[Dict[str, Any]], synapses: Any) -> None:
    node_ids = [str(neuron.get("id")) for neuron in neurons if neuron.get("id")]
    _save_sparse_snapshot_by_ids(child, node_ids, synapses)


def _save_neural_map_streaming(
    child: str,
    payload_meta: Dict[str, Any],
    hot_neurons: List[Dict[str, Any]],
    synapses: Any,
    policy: Dict[str, Any],
    spill_store: Optional[_NeuralSpillStore],
) -> None:
    path = _neural_map_path(child)
    compact = bool(policy.get("compact_save_enabled", True))
    meta = {key: value for key, value in payload_meta.items() if key not in {"neurons", "synapses"}}

    def _write(handle) -> None:
        handle.write("{")
        items = list(meta.items())
        for idx, (key, value) in enumerate(items):
            if idx:
                handle.write(",")
            handle.write(json.dumps(str(key)))
            handle.write(":")
            json.dump(value, handle, ensure_ascii=False)
        if items:
            handle.write(",")
        handle.write(json.dumps("neurons"))
        handle.write(":")
        neuron_iter = spill_store.iter_storage_neurons(hot_neurons, policy, compact=compact) if spill_store is not None else (
            _compact_neuron_for_storage(neuron, policy) if compact else neuron for neuron in hot_neurons if neuron.get("id")
        )
        _write_json_array_stream(handle, neuron_iter)
        handle.write(",")
        handle.write(json.dumps("synapses"))
        handle.write(":")
        source_synapses = synapses.iter_sorted() if isinstance(synapses, _BoundedSynapseSpool) else synapses
        synapse_iter = (
            _compact_synapse_for_storage(synapse) if compact else _synapse_to_payload(synapse)
            for synapse in source_synapses
            if _synapse_get(synapse, "source") and _synapse_get(synapse, "target")
        )
        _write_json_array_stream(handle, synapse_iter)
        handle.write("}")

    _atomic_write_json_stream(path, _write)


def _existing_fragment_ids(neurons: List[Dict[str, Any]]) -> Set[str]:
    known: Set[str] = set()
    for neuron in neurons:
        for frag_id in neuron.get("fragments", []):
            known.add(frag_id)
    return known


def _node_id_allocator(neurons: List[Dict[str, Any]], extra_ids: Optional[Iterable[str]] = None):
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
    if extra_ids is not None:
        for node_id in extra_ids:
            node_text = str(node_id or "")
            if not node_text.startswith(prefix):
                continue
            try:
                idx = int(node_text[len(prefix):])
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


def _clip_sequence_with_anchors(values: Iterable[Any], limit: int, anchor_count: int = 0) -> List[str]:
    clipped = [str(value) for value in values if value]
    if limit <= 0 or len(clipped) <= limit:
        return clipped
    anchor_count = max(0, min(int(anchor_count or 0), limit))
    if anchor_count <= 0:
        return clipped[-limit:]
    tail_count = max(0, limit - anchor_count)
    head = clipped[:anchor_count]
    tail = clipped[-tail_count:] if tail_count > 0 else []
    selected: List[str] = []
    seen = set()
    for value in head + tail:
        if value not in seen:
            selected.append(value)
            seen.add(value)
    if len(selected) < limit:
        extras: List[str] = []
        for value in reversed(clipped):
            if value in seen:
                continue
            extras.append(value)
            seen.add(value)
            if len(selected) + len(extras) >= limit:
                break
        selected[anchor_count:anchor_count] = list(reversed(extras))
    return selected[:limit]


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


def _compact_neuron_for_storage(neuron: Dict[str, Any], policy: Dict[str, Any]) -> Dict[str, Any]:
    compact = {
        "id": neuron.get("id"),
        "fragments": _clip_sequence_with_anchors(
            neuron.get("fragments", []),
            int(policy.get("max_fragments_per_neuron", 128)),
            int(policy.get("fragment_anchor_count", 0)),
        ),
        "vector": _round_vector(neuron.get("vector"), int(policy.get("vector_round_digits", 6))),
        "position": _round_vector(neuron.get("position"), int(policy.get("position_round_digits", 4))),
        "region": neuron.get("region"),
        "tags": _clip_sequence_with_anchors(
            neuron.get("tags", []),
            int(policy.get("max_tags_per_neuron", 32)),
            int(policy.get("tag_anchor_count", 0)),
        ),
        "last_used": neuron.get("last_used"),
    }
    network_type = str(neuron.get("network_type") or "memory_graph")
    if network_type != "memory_graph":
        compact["network_type"] = network_type
    symbolic_density = _safe_float(neuron.get("symbolic_density"), 0.0)
    if abs(symbolic_density) > 1e-6:
        compact["symbolic_density"] = round(symbolic_density, 4)
    activation_history = [
        round(float(value), 4)
        for value in (neuron.get("activation_history") or [])
        if isinstance(value, (int, float))
    ]
    if activation_history:
        compact["activation_history"] = activation_history[-32:]
    return {key: value for key, value in compact.items() if value not in (None, [], {}, "")}


def _compact_synapse_for_storage(synapse: Any) -> Dict[str, Any]:
    payload = _synapse_to_payload(synapse)
    compact = {
        "source": payload.get("source"),
        "target": payload.get("target"),
        "weight": round(_safe_float(payload.get("weight"), 0.0), 4),
    }
    direction = payload.get("direction")
    if isinstance(direction, list) and direction:
        compact["direction"] = [round(float(value), 5) for value in direction[:3]]
    relation = payload.get("relation")
    if relation:
        compact["relation"] = str(relation)
    network_type = str(payload.get("network_type") or "memory_graph")
    if network_type != "memory_graph":
        compact["network_type"] = network_type
    return {key: value for key, value in compact.items() if value not in (None, [], {}, "")}


def _compact_graph_for_storage(
    neurons: List[Dict[str, Any]],
    synapses: List[Dict[str, Any]],
    policy: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    return (
        [_compact_neuron_for_storage(neuron, policy) for neuron in neurons if neuron.get("id")],
        [_compact_synapse_for_storage(synapse) for synapse in synapses if _synapse_get(synapse, "source") and _synapse_get(synapse, "target")],
    )


def _materialize_candidate(node_id: str, candidate: Dict[str, Any], policy: Dict[str, Any]) -> Dict[str, Any]:
    now_iso = datetime.now(timezone.utc).isoformat()
    frag_limit = int(policy.get("max_fragments_per_neuron", 128))
    tag_limit = int(policy.get("max_tags_per_neuron", 32))
    vector_digits = int(policy.get("vector_round_digits", 6))
    pos_digits = int(policy.get("position_round_digits", 4))
    return {
        "id": node_id,
        "fragments": _clip_sequence_with_anchors(candidate["fragments"], frag_limit, int(policy.get("fragment_anchor_count", 0))),
        "vector": _round_vector(candidate.get("vector"), vector_digits),
        "position": _round_vector(candidate.get("position"), pos_digits),
        "region": candidate["region"],
        "network_type": "memory_graph",
        "symbolic_density": 0.0,
        "tags": _clip_sequence_with_anchors(candidate["tags"], tag_limit, int(policy.get("tag_anchor_count", 0))),
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
        neuron["fragments"] = _clip_sequence_with_anchors(
            existing_frags,
            max_fragments_per_neuron,
            int(policy.get("fragment_anchor_count", 0)),
        )
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
    tag_list = [str(tag) for tag in (neuron.get("tags", []) or []) if tag]
    for tag in candidate["tags"]:
        tag_text = str(tag)
        if tag_text and tag_text not in tag_list:
            tag_list.append(tag_text)
    tag_limit = int(policy.get("max_tags_per_neuron", 32))
    neuron["tags"] = _clip_sequence_with_anchors(tag_list, tag_limit, int(policy.get("tag_anchor_count", 0)))
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
        centroid = group.get("centroid")
        vector_sum = group.get("vector_sum")
        count = group.get("count", len(fragment_ids))
        if isinstance(centroid, list) and centroid:
            avg_vec = [round(float(v), 6) for v in centroid]
        elif vector_sum and count:
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
    spill_store: Optional[_NeuralSpillStore] = None,
) -> Tuple[int, int, int]:
    if not candidates:
        return 0, 0, 0
    merge_threshold = max(0.0, min(1.0, cluster_threshold - policy.get("merge_slack", 0.0)))
    merged = 0
    created = 0
    skipped = 0
    id_allocator = _node_id_allocator(neurons, spill_store.node_ids() if spill_store is not None else None)
    for candidate in candidates:
        best = None
        best_score = 0.0
        for neuron in neurons:
            score = _score_candidate_match(neuron, candidate, tag_weight)
            if score > best_score:
                best_score = score
                best = neuron
        if spill_store is not None and spill_store.has_spills():
            spill_id, spill_score = spill_store.best_match(candidate, tag_weight)
            if spill_id and spill_score > best_score:
                activated = spill_store.activate_node(spill_id, neurons)
                if activated is not None:
                    best = activated
                    best_score = spill_score
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
    spill_store: Optional[_NeuralSpillStore] = None,
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
            spill_store=spill_store,
        )
    else:
        merged = 0
        created = 0
        skipped = 0
        id_allocator = _node_id_allocator(neurons, spill_store.node_ids() if spill_store is not None else None)
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
    spill_store = _NeuralSpillStore(child, policy)
    custom_runtime = _NeuralCustomTransformerRuntime(child, cfg)
    spill_flushes = 0
    spilled_neurons = 0
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
    neuron_budget_pruned, _ = _enforce_total_neuron_budget(
        neurons,
        int(policy.get("max_neurons_total", 0)),
        spill_store=spill_store if incremental else None,
    ) if incremental else (0, set())
    known_fragments = _existing_fragment_ids(neurons) if incremental else set()

    index = _load_memory_index(child)
    valid_ids = index
    retention_policy = _memory_policy().get("retention", DEFAULT_RETENTION_POLICY)
    prep_stats = _pregraph_fragment_compaction_pass(child, index, retention_policy)
    if prep_stats.get("status") == "ok" and prep_stats.get("compacted"):
        log_to_statusbox(
            f"[NeuralMap] Pre-graph compaction compacted {prep_stats['compacted']} fragment(s) and skipped {prep_stats['skipped']} candidate(s)."
        )

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

    initial_spilled = spill_store.spill_overflow(neurons)
    if initial_spilled:
        spilled_neurons += initial_spilled
        spill_flushes += 1
        log_to_statusbox(
            f"[NeuralMap] Spilled {initial_spilled} cold neuron(s) to the temporary disk store to cap hot RAM."
        )

    anchors = _load_body_anchors()
    fallback_anchor = anchors.get("head", {"center": [0.0, 0.0, 0.0], "radius": 2.0})

    dirty_mode = bool(policy.get("dirty_index_enabled", True) and incremental and index)
    queue_state = _load_neural_build_state(child) if dirty_mode else {"pending": []}
    pending_ids = [fid for fid in queue_state.get("pending", []) if fid in valid_ids] if dirty_mode else []
    dirty_signatures = _load_neural_dirty_index(child) if dirty_mode else {}
    current_signatures: Dict[str, str] = {}
    removed_signatures: Set[str] = set()
    processed_count = 0
    detached_neurons = 0
    detached_overflow = False
    detected_dirty = 0
    source = "selector"
    selector_meta: Dict[str, Any] = {}
    selector_total = 0
    selector_loaded = 0
    selector_targets: List[Dict[str, Any]] = []
    selector_target_count = 0

    if dirty_mode:
        source = "dirty_index"
        dirty_ids, current_signatures, removed_signatures = _collect_dirty_fragment_ids(child, index, dirty_signatures)
        detected_dirty = len(dirty_ids)
        dirty_set = set(dirty_ids)
        if dirty_set:
            known_dirty = dirty_set & known_fragments
            if known_dirty:
                reactivated = spill_store.activate_for_fragments(known_dirty, neurons)
                if reactivated:
                    log_to_statusbox(
                        f"[NeuralMap] Reloaded {reactivated} spilled neuron(s) touched by the dirty set."
                    )
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
                    known_fragments.difference_update(local_rebuild_ids)
            pending_set = set(pending_ids)
            for frag_id in dirty_set:
                if frag_id in valid_ids and frag_id not in pending_set:
                    pending_ids.append(frag_id)
                    pending_set.add(frag_id)
        pending_ids, pending_trimmed = _trim_pending_fragment_ids(
            pending_ids,
            index,
            int(policy.get("max_pending_dirty_fragments", 0) or 0),
        )
        if pending_trimmed:
            log_to_statusbox(
                f"[NeuralMap] Trimmed dirty queue by {pending_trimmed} fragment(s) to stay within the pending hard cap."
            )
        queue_state["last_source"] = source
    else:
        fragments, selector_total, selector_meta = select_fragments_for_neural_map(
            child,
            burst_limit,
            cfg=cfg,
            known_fragments=known_fragments if incremental else None,
            index=index,
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
        selector_target_count = len(selector_targets)

    totals = {"input": 0, "encoded": 0, "clusters": 0, "merged": 0, "created": 0, "skipped": 0}
    batches_run = 0
    budget_hit = False

    if source == "dirty_index":
        pending_queue = deque(pending_ids)
        while pending_queue and totals["input"] < burst_limit:
            elapsed_ms = (time.perf_counter() - start_perf) * 1000.0
            if build_budget_ms > 0 and elapsed_ms >= build_budget_ms and totals["input"] > 0:
                budget_hit = True
                break
            step = min(batch_size, burst_limit - totals["input"], len(pending_queue))
            batch_ids = [pending_queue.popleft() for _ in range(step)]
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
                spill_store=spill_store,
            )
            for key in totals:
                totals[key] += stats.get(key, 0)
            batch_processed_ids = [str(frag.get("id")) for frag in batch_fragments if frag.get("id")]
            processed_count += len(batch_processed_ids)
            known_fragments.update(batch_processed_ids)
            for frag_id in batch_processed_ids:
                sig = current_signatures.get(frag_id)
                if sig:
                    dirty_signatures[frag_id] = sig
            custom_runtime.observe(batch_fragments)
            dropped, removed_fragments = _enforce_total_neuron_budget(
                neurons,
                int(policy.get("max_neurons_total", 0)),
                spill_store=spill_store,
            )
            if dropped:
                neuron_budget_pruned += dropped
                known_fragments.difference_update(removed_fragments)
            batches_run += 1
            spilled_now = spill_store.maybe_flush(neurons, batches_run)
            if spilled_now:
                spilled_neurons += spilled_now
                spill_flushes += 1
            gc_every_batches = int(policy.get("gc_every_batches", 0))
            if gc_every_batches > 0 and (batches_run % gc_every_batches) == 0:
                gc.collect()
            del batch_fragments
            del batch_processed_ids
        pending_ids = list(pending_queue)
    else:
        selector_queue = deque(selector_targets)
        while selector_queue and totals["input"] < burst_limit:
            elapsed_ms = (time.perf_counter() - start_perf) * 1000.0
            if build_budget_ms > 0 and elapsed_ms >= build_budget_ms and totals["input"] > 0:
                budget_hit = True
                break
            step = min(batch_size, burst_limit - totals["input"], len(selector_queue))
            batch_fragments = [selector_queue.popleft() for _ in range(step)]
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
                spill_store=spill_store,
            )
            for key in totals:
                totals[key] += stats.get(key, 0)
            batch_processed_ids = [str(frag.get("id")) for frag in batch_fragments if frag.get("id")]
            known_fragments.update(batch_processed_ids)
            custom_runtime.observe(batch_fragments)
            dropped, removed_fragments = _enforce_total_neuron_budget(
                neurons,
                int(policy.get("max_neurons_total", 0)),
                spill_store=spill_store,
            )
            if dropped:
                neuron_budget_pruned += dropped
                known_fragments.difference_update(removed_fragments)
            batches_run += 1
            spilled_now = spill_store.maybe_flush(neurons, batches_run)
            if spilled_now:
                spilled_neurons += spilled_now
                spill_flushes += 1
            gc_every_batches = int(policy.get("gc_every_batches", 0))
            if gc_every_batches > 0 and (batches_run % gc_every_batches) == 0:
                gc.collect()
            del batch_fragments
            del batch_processed_ids
        selector_targets = list(selector_queue)

    custom_usage = custom_runtime.run()

    if dirty_mode:
        for frag_id in removed_signatures:
            dirty_signatures.pop(frag_id, None)
        queue_state["pending"] = pending_ids
        queue_state["dirty_detected"] = detected_dirty
        queue_state["processed"] = processed_count
        queue_state["detached_neurons"] = detached_neurons
        _save_neural_build_state(child, queue_state)
        _save_neural_dirty_index(child, dirty_signatures)
        log_to_statusbox(
            f"[NeuralMap] Dirty queue: detected {detected_dirty}, processed {processed_count}, remaining {len(pending_ids)}."
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
    total_neuron_count = len(neurons) + spill_store.count()

    max_synapses_total = int(policy.get("max_synapses_total", 0) or 0)
    synapse_budget_pruned = 0
    synapse_stats = {"pairs_evaluated": 0, "edge_count": len(synapses), "truncated": False}
    if needs_synapse_refresh:
        max_edges_updated = int(policy.get("max_edges_updated", 0) or 0)
        if max_synapses_total > 0 and (max_edges_updated <= 0 or max_synapses_total < max_edges_updated):
            max_edges_updated = max_synapses_total
        link_neurons = [_lightweight_neuron_view(neuron) for neuron in neurons]
        if spill_store.has_spills():
            link_neurons.extend(spill_store.iter_link_neurons())
        synapse_spool_path = None
        if policy.get("synapse_spool_enabled", True) and max_edges_updated > 0 and int(policy.get("max_connection_degree", 0) or 0) <= 0:
            synapse_spool_path = _neural_synapse_spool_path(child)
        synapses, synapse_stats = build_synaptic_links(
            link_neurons,
            threshold=synapse_threshold,
            max_edges=max_edges_updated,
            max_pairs=int(policy.get("max_synapse_pairs", 0) or 0),
            max_edges_per_neuron=int(policy.get("max_edges_per_neuron", 0) or 0),
            include_direction=bool(policy.get("edge_direction_enabled", True)),
            compact_records=True,
            spool_path=synapse_spool_path,
            return_stats=True,
        )
        if synapse_stats.get("truncated"):
            log_to_statusbox(
                f"[NeuralMap] Synapse refresh hit budget ({synapse_stats.get('edge_count')} edges, {synapse_stats.get('pairs_evaluated')} pairs)."
            )
    synapse_budget_pruned = _enforce_synapse_budget(synapses, max_synapses_total)
    if synapse_budget_pruned:
        log_to_statusbox(f"[NeuralMap] Applied hard synapse cap; pruned {synapse_budget_pruned} edge(s).")
    synapses, connection_cap_stats = _apply_synapse_connection_caps(synapses, policy)
    if connection_cap_stats.get("weight_pruned"):
        log_to_statusbox(f"[NeuralMap] Dropped {connection_cap_stats['weight_pruned']} weak connection(s) below the persistence threshold.")
    if connection_cap_stats.get("degree_pruned"):
        log_to_statusbox(f"[NeuralMap] Applied per-neuron connection cap; pruned {connection_cap_stats['degree_pruned']} edge(s).")

    should_save_map = bool(
        not incremental
        or map_changed
        or needs_synapse_refresh
        or synapse_budget_pruned
        or connection_cap_stats.get("weight_pruned")
        or connection_cap_stats.get("degree_pruned")
    )
    if should_save_map:
        result = existing_map if incremental else {}
        result.update({
            "converted_from_legacy": existing_map.get("converted_from_legacy", False) if incremental else False,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })
        active_spill_store = spill_store if spill_store.has_spills() else None
        _save_neural_map_streaming(child, result, neurons, synapses, policy, active_spill_store)
        if policy.get("emit_sparse_snapshot", True):
            node_ids = [str(neuron.get("id")) for neuron in neurons if neuron.get("id")]
            if active_spill_store is not None:
                node_ids.extend(str(node.get("id")) for node in active_spill_store.iter_link_neurons() if node.get("id"))
            _save_sparse_snapshot_by_ids(child, node_ids, synapses)

    if source == "selector" and selector_loaded and selector_target_count == 0 and incremental:
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
    if spilled_neurons:
        log_to_statusbox(
            f"[NeuralMap] Spilled {spilled_neurons} cold neuron(s) across {spill_flushes} flush(es) to bound RAM during the build."
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
    final_neuron_count = total_neuron_count
    log_to_statusbox(
        f"[NeuralMap] {final_neuron_count} neurons | {len(synapses)} synapses | "
        f"Policy={policy.get('mode')} | Source={source} | Batches={batches_run}."
    )
    log_to_statusbox(f"[NeuralMap] Mapping time: {duration}.")
    selector_remaining = len(selector_targets) if isinstance(selector_targets, list) else 0
    if source == "dirty_index":
        needs_resume = bool(queue_remaining or budget_hit)
    else:
        needs_resume = bool((selector_total > selector_loaded) or selector_remaining or budget_hit)

    summary = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "input_processed": totals["input"],
        "encoded": totals["encoded"],
        "clusters": totals["clusters"],
        "merged": totals["merged"],
        "created": totals["created"],
        "skipped": totals["skipped"],
        "batches_run": batches_run,
        "budget_hit": bool(budget_hit),
        "queue_remaining": queue_remaining,
        "selector_total": selector_total,
        "selector_loaded": selector_loaded,
        "selector_remaining": selector_remaining,
        "map_changed": bool(map_changed),
        "needs_resume": bool(needs_resume),
        "neurons": final_neuron_count,
        "synapses": len(synapses),
    }

    if isinstance(synapses, _BoundedSynapseSpool):
        synapses.cleanup()
    if hasattr(index, "close"):
        index.close()
    spill_store.cleanup()
    return summary

# === MemoryManager Class ===
class MemoryManager:
    def __init__(
        self,
        child="Inazuma_Yagami",
        tier_policy: Optional[Dict[str, Any]] = None,
        *,
        autoload: bool = True,
    ):
        self.child = child
        self.base_path = Path("AI_Children") / child / "memory" / "fragments"
        self.index_path = Path("AI_Children") / child / "memory" / "memory_map.json"
        self.memory_map = {}
        self._map_loaded = False
        self.policy = tier_policy or _memory_policy()
        self.cold_storage_policy = _cold_storage_policy()
        if autoload:
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

    def load_map(self, force: bool = False):
        if self._map_loaded and not force:
            return
        if self.index_path.exists():
            try:
                with open(self.index_path, "r") as f:
                    self.memory_map = json.load(f)
            except:
                self.memory_map = {}
        else:
            self.memory_map = {}
        self._map_loaded = True

    def save_map(self):
        with open(self.index_path, "w") as f:
            json.dump(self.memory_map, f, indent=2)
        self._map_loaded = True
        try:
            source_mtime_ns = int(self.index_path.stat().st_mtime_ns)
        except OSError:
            source_mtime_ns = None
        try:
            _persist_memory_index_db(self.child, self.memory_map, source_mtime_ns=source_mtime_ns)
        except Exception:
            pass

    def unload_map(self):
        self.memory_map = {}
        self._map_loaded = False

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
    def _retention_adjusted_tier(target_tier: str, importance: float, age_hours: Optional[float], retention: Optional[Dict[str, Any]] = None) -> str:
        retention = retention or DEFAULT_RETENTION_POLICY
        protect = _clamp(_safe_float(retention.get("protect_cold_importance"), DEFAULT_RETENTION_POLICY["protect_cold_importance"]), 0.0, 1.0)
        low_importance = _clamp(_safe_float(retention.get("compact_low_importance_threshold"), DEFAULT_RETENTION_POLICY["compact_low_importance_threshold"]), 0.0, 1.0)
        low_age = max(0.0, _safe_float(retention.get("compact_low_importance_age_hours"), DEFAULT_RETENTION_POLICY["compact_low_importance_age_hours"]))
        if target_tier == "cold" and importance >= protect:
            return "long"
        if target_tier in {"long", "cold"} and age_hours is not None and age_hours >= low_age and importance <= low_importance:
            return "cold"
        return target_tier

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

    def human_memory_prune_pass(self, *, force: bool = False, now: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Mandatory forgetting pass: preserve the gist of old cold memories while
        shedding low-importance detail payloads that human memory would blur.
        """
        if not self._map_loaded:
            self.load_map()
        retention = (self.policy or {}).get("retention", DEFAULT_RETENTION_POLICY)
        index_count = len(self.memory_map)
        settings = _human_memory_prune_settings(retention, index_count=index_count)
        now_dt = now or datetime.now(timezone.utc)
        if now_dt.tzinfo is None:
            now_dt = now_dt.replace(tzinfo=timezone.utc)

        review_required = bool(retention.get("human_prune_review_required", False))
        state = _load_inastate(self.child)
        apply_report_id = None
        apply_report = None
        if review_required and isinstance(state, dict):
            raw_apply = state.get("human_memory_prune_apply_report")
            if isinstance(raw_apply, str) and raw_apply.strip():
                apply_report_id = raw_apply.strip()
                apply_report = _load_human_prune_report(self.child, retention, apply_report_id)
                if not apply_report:
                    stats = {
                        "status": "failed",
                        "reason": "approved_report_not_found",
                        "approved_report": apply_report_id,
                        "timestamp": now_dt.isoformat(),
                    }
                    _write_inastate_value(self.child, "human_memory_prune_last_run", stats)
                    return stats

        if not force and not apply_report:
            last_run = state.get("human_memory_prune_last_run") if isinstance(state, dict) else None
            last_ts = None
            if isinstance(last_run, dict):
                last_ts = _parse_iso_timestamp(last_run.get("timestamp"))
            cooldown = float(settings.get("cooldown_seconds") or 0.0)
            if last_ts is not None and cooldown > 0 and (now_dt.timestamp() - last_ts) < cooldown:
                return {
                    "status": "cooldown",
                    "limit": settings["limit"],
                    "index_count": index_count,
                    "timestamp": now_dt.isoformat(),
                }

        selected: List[Tuple[str, Dict[str, Any], Path]] = []
        experience_items: List[Dict[str, Any]] = []

        if apply_report:
            report_candidates = apply_report.get("candidates") if isinstance(apply_report, dict) else []
            if isinstance(report_candidates, list):
                for item in report_candidates:
                    if not isinstance(item, dict):
                        continue
                    kind = str(item.get("kind") or "")
                    if kind == "fragment":
                        frag_id = str(item.get("id") or "")
                        live_meta = self.memory_map.get(frag_id)
                        if not frag_id or not isinstance(live_meta, dict):
                            continue
                        report_path = Path(str(item.get("path") or ""))
                        path = self._resolve_fragment_path(frag_id, live_meta) or report_path
                        selected.append((frag_id, live_meta, path))
                    elif kind.startswith("experience_"):
                        experience_items.append(item)
        else:
            index = _load_memory_index(self.child)
            try:
                try:
                    index_count = len(index)
                except Exception:
                    index_count = len(self.memory_map)
                selected = _select_human_memory_prune_candidates(
                    self.child,
                    index,
                    retention,
                    index_count=index_count,
                    now=now_dt,
                )
            finally:
                if hasattr(index, "close"):
                    index.close()
            experience_items = _select_experience_prune_candidates(self.child, retention, now=now_dt)

        if review_required and not apply_report:
            fragment_items = _fragment_prune_report_items(selected, now_dt)
            report = _write_human_prune_review_report(
                self.child,
                retention,
                fragment_items,
                experience_items,
                settings,
                now_dt,
            )
            stats = {
                "status": "review_required",
                "selected": len(fragment_items),
                "experience_selected": len(experience_items),
                "report_id": report.get("report_id"),
                "report_paths": report.get("paths", {}),
                "limit": settings["limit"],
                "index_count": index_count,
                "timestamp": now_dt.isoformat(),
            }
            _write_inastate_value(self.child, "human_memory_prune_last_run", stats)
            _write_inastate_value(
                self.child,
                "human_memory_prune_report",
                {
                    "report_id": report.get("report_id"),
                    "status": report.get("status"),
                    "timestamp": report.get("timestamp"),
                    "summary": report.get("summary", {}),
                    "paths": report.get("paths", {}),
                    "approval": report.get("approval", {}),
                },
            )
            log_to_statusbox(
                f"[Memory] Human-memory prune report ready: {len(fragment_items)} fragment(s), "
                f"{len(experience_items)} experience(s)."
            )
            return stats

        if compact_fragment_file is None and selected:
            stats = {
                "status": "unavailable",
                "reason": "cold_storage_unavailable",
                "limit": settings["limit"],
                "index_count": index_count,
                "timestamp": now_dt.isoformat(),
            }
            _write_inastate_value(self.child, "human_memory_prune_last_run", stats)
            return stats

        cold_policy = dict(self.cold_storage_policy or _cold_storage_policy())
        cold_policy["enabled"] = True
        cold_policy["auto_compact"] = True
        cold_policy["retain_full_fragment"] = False
        cold_policy["purge_pending_delete"] = True

        compacted = 0
        retained = 0
        already_compacted = 0
        moved_to_cold = 0
        protected = 0
        failed = 0
        missing = 0
        skipped = 0
        experience_compacted = 0
        experience_failed = 0
        experience_skipped = 0
        changed = False

        for frag_id, meta, path in selected:
            live_meta = self.memory_map.get(frag_id)
            if not isinstance(live_meta, dict):
                skipped += 1
                continue
            path = self._resolve_fragment_path(frag_id, live_meta) or path
            if path is None or not path.exists():
                missing += 1
                continue
            if _is_anchor_fragment_record(live_meta, retention):
                protected += 1
                continue
            fragment = _load_fragment_from_path(path)
            if not fragment:
                skipped += 1
                continue
            if _is_anchor_fragment_record(fragment, retention):
                protected += 1
                continue
            if _is_compacted_fragment(fragment):
                already_compacted += 1
                live_meta.update(self._stat_payload(path))
                self.memory_map[frag_id] = live_meta
                changed = True
                continue

            cold_dir = self.base_path / "cold"
            if path.parent != cold_dir:
                cold_dir.mkdir(parents=True, exist_ok=True)
                dest_path = cold_dir / path.name
                if dest_path.exists() and dest_path != path:
                    skipped += 1
                    continue
                try:
                    path.rename(dest_path)
                except OSError:
                    skipped += 1
                    continue
                path = dest_path
                live_meta["tier"] = "cold"
                live_meta["filename"] = path.name
                self.memory_map[frag_id] = live_meta
                moved_to_cold += 1
                changed = True

            result = self._compact_cold_fragment(frag_id, live_meta, path, cold_policy)
            status = result.get("status") if isinstance(result, dict) else None
            if status in {"compacted", "retained"}:
                if status == "compacted":
                    compacted += 1
                else:
                    retained += 1
                live_meta["tier"] = "cold"
                live_meta["filename"] = path.name
                live_meta.update(self._stat_payload(path))
                if not live_meta.get("last_seen"):
                    live_meta["last_seen"] = fragment.get("timestamp", now_dt.isoformat())
                self.memory_map[frag_id] = live_meta
                changed = True
            elif status == "skipped":
                skipped += 1
            else:
                failed += 1

        for item in experience_items:
            result = _compact_experience_file(self.child, item, retention, now_dt)
            status = result.get("status") if isinstance(result, dict) else None
            if status == "compacted":
                experience_compacted += 1
            elif status in {"already_compacted", "missing"}:
                experience_skipped += 1
            else:
                experience_failed += 1

        if changed:
            self.save_map()

        purge_stats = {"deleted": 0, "kept": 0}
        try:
            import cold_storage  # type: ignore

            purge_stats = cold_storage.purge_pending_delete(self.child, cold_policy)
        except Exception:
            pass

        stats = {
            "status": "ok",
            "selected": len(selected),
            "compacted": compacted,
            "retained": retained,
            "already_compacted": already_compacted,
            "moved_to_cold": moved_to_cold,
            "protected": protected,
            "failed": failed,
            "missing": missing,
            "skipped": skipped,
            "experience_selected": len(experience_items),
            "experience_compacted": experience_compacted,
            "experience_skipped": experience_skipped,
            "experience_failed": experience_failed,
            "purged_pending": int(purge_stats.get("deleted", 0) or 0),
            "pending_kept": int(purge_stats.get("kept", 0) or 0),
            "limit": settings["limit"],
            "index_count": index_count,
            "timestamp": now_dt.isoformat(),
        }
        if apply_report_id:
            stats["applied_report"] = apply_report_id
        _write_inastate_value(self.child, "human_memory_prune_last_run", stats)
        if apply_report_id:
            _write_inastate_value(self.child, "human_memory_prune_apply_report", None)
            _write_inastate_value(self.child, "human_memory_prune_last_applied_report", stats)
        if compacted or stats["purged_pending"] or failed or experience_compacted or experience_failed:
            log_to_statusbox(
                f"[Memory] Human-memory prune: compacted {compacted}, moved {moved_to_cold}, purged {stats['purged_pending']}, "
                f"experiences {experience_compacted}, skipped {skipped + protected + already_compacted + experience_skipped}, "
                f"failed {failed + experience_failed}."
            )
        return stats

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

        retention = self.policy.get("retention", DEFAULT_RETENTION_POLICY)
        for frag_id, meta in self.memory_map.items():
            ts = self._parse_timestamp(meta.get("last_seen")) or self._parse_timestamp(meta.get("timestamp"))
            age_hours = (now - ts).total_seconds() / 3600.0 if ts else None
            importance = _clamp(_safe_float(meta.get("importance"), 0.0), 0.0, 1.0)
            target_tier = self._tier_for_age(ts, now)
            target_tier = self._retention_adjusted_tier(target_tier, importance, age_hours, retention)
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

def main(argv: Optional[List[str]] = None) -> int:
    args = _memory_graph_phase_args(argv)
    cfg = _load_config()
    child = cfg.get("current_child", "Inazuma_Yagami")
    mgr = MemoryManager(child)

    if args.phase == "boot":
        verification = _run_memory_index_verification(child, cfg, mgr)
        _set_memory_graph_deferred_build(
            child,
            "queued",
            requested_at=datetime.now(timezone.utc).isoformat(),
            requested_by="boot",
            boot_verification=verification,
        )
        log_to_statusbox("[Memory] Boot verification complete. Deferred neural build queued.")
        return 0

    if args.phase == "neural":
        _run_memory_neural_phase(child, mgr, launch_source="deferred")
        return 0

    verification = _run_memory_index_verification(child, cfg, mgr)
    _set_memory_graph_deferred_build(
        child,
        "queued",
        requested_by="full",
        requested_at=datetime.now(timezone.utc).isoformat(),
        boot_verification=verification,
    )
    _run_memory_neural_phase(child, mgr, launch_source="full")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
