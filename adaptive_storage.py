"""Conservative online placement learning for rebuildable storage artifacts.

Durable memory is deliberately outside this policy. The learner may choose only
between the configured fast runtime location and the existing rebuildable
fallback. Small, infrequent probes update exponentially weighted device
telemetry; hysteresis prevents placement churn.
"""
from __future__ import annotations

import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


ARTIFACT_CLASSES = ("index", "neural", "runtime")
DEFAULT_POLICY = {
    "enabled": True,
    "apply_recommendations": True,
    "probe_interval_seconds": 3600.0,
    "probe_bytes": 262144,
    "ewma_alpha": 0.25,
    "min_samples_before_switch": 3,
    "switch_margin": 0.18,
    "min_fast_free_ratio": 0.08,
    "state_path": "AI_Children/{child}/memory/adaptive_storage_state.json",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def policy_from_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = dict(DEFAULT_POLICY)
    raw = config.get("adaptive_storage_policy") if isinstance(config, dict) else None
    if isinstance(raw, dict):
        cfg.update(raw)
    return cfg


def _state_path(child: str, policy: Dict[str, Any]) -> Path:
    raw = str(policy.get("state_path") or DEFAULT_POLICY["state_path"])
    return Path(raw.format(child=child))


def _empty_state(child: str) -> Dict[str, Any]:
    return {
        "version": 1,
        "child": child,
        "updated_at": None,
        "last_probe_at": None,
        "devices": {},
        "decisions": {name: {"tier": "fast", "reason": "safe_default_rebuildable_fast"} for name in ARTIFACT_CLASSES},
    }


def load_state(child: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    policy = policy_from_config(config)
    path = _state_path(child, policy)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return _empty_state(child)
    if not isinstance(payload, dict):
        return _empty_state(child)
    state = _empty_state(child)
    state.update(payload)
    if not isinstance(state.get("devices"), dict):
        state["devices"] = {}
    if not isinstance(state.get("decisions"), dict):
        state["decisions"] = {}
    return state


def save_state(child: str, state: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Path:
    policy = policy_from_config(config)
    path = _state_path(child, policy)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)
    return path


def _ewma(previous: Any, observed: float, alpha: float) -> float:
    try:
        old = float(previous)
    except (TypeError, ValueError):
        return float(observed)
    return (alpha * float(observed)) + ((1.0 - alpha) * old)


def record_observation(
    state: Dict[str, Any],
    tier: str,
    *,
    success: bool,
    latency_seconds: Optional[float] = None,
    throughput_bytes_per_second: Optional[float] = None,
    free_ratio: Optional[float] = None,
    alpha: float = 0.25,
) -> Dict[str, Any]:
    devices = state.setdefault("devices", {})
    sample = devices.setdefault(tier, {"samples": 0, "failures": 0})
    sample["samples"] = int(sample.get("samples") or 0) + 1
    if not success:
        sample["failures"] = int(sample.get("failures") or 0) + 1
    sample["success_ewma"] = _ewma(sample.get("success_ewma"), 1.0 if success else 0.0, alpha)
    if success and latency_seconds is not None:
        sample["latency_seconds_ewma"] = _ewma(sample.get("latency_seconds_ewma"), max(0.0, latency_seconds), alpha)
    if success and throughput_bytes_per_second is not None:
        sample["throughput_bps_ewma"] = _ewma(sample.get("throughput_bps_ewma"), max(0.0, throughput_bytes_per_second), alpha)
    if free_ratio is not None:
        sample["free_ratio"] = max(0.0, min(1.0, float(free_ratio)))
    sample["last_observed_at"] = _now_iso()
    return sample


def _probe(path: Path, probe_bytes: int) -> Dict[str, Any]:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    probe = target / ".ina_adaptive_storage_probe"
    payload = b"I" * max(4096, int(probe_bytes))
    started = time.perf_counter()
    try:
        # Buffered write/read is intentional: a storage learner must never stall
        # Ina on a filesystem-wide journal commit merely to collect telemetry.
        with probe.open("wb") as handle:
            handle.write(payload)
            handle.flush()
        with probe.open("rb") as handle:
            read_back = handle.read()
        elapsed = max(time.perf_counter() - started, 1e-9)
        if read_back != payload:
            raise OSError("probe verification mismatch")
        stat = os.statvfs(target)
        total = int(stat.f_blocks) * int(stat.f_frsize or stat.f_bsize or 1)
        available = int(stat.f_bavail) * int(stat.f_frsize or stat.f_bsize or 1)
        return {
            "success": True,
            "latency_seconds": elapsed,
            "throughput_bytes_per_second": (len(payload) * 2) / elapsed,
            "free_ratio": (available / total) if total else None,
        }
    except Exception as exc:
        return {"success": False, "error": str(exc)}
    finally:
        try:
            probe.unlink()
        except OSError:
            pass


def _device_score(sample: Dict[str, Any], *, prefer_fast: bool, min_fast_free_ratio: float) -> float:
    if not isinstance(sample, dict) or int(sample.get("samples") or 0) <= 0:
        return 0.35 if prefer_fast else 0.25
    success = float(sample.get("success_ewma", 0.0) or 0.0)
    latency = max(float(sample.get("latency_seconds_ewma", 1.0) or 1.0), 1e-6)
    throughput = max(float(sample.get("throughput_bps_ewma", 0.0) or 0.0), 0.0)
    free_ratio = float(sample.get("free_ratio", 1.0) or 0.0)
    if prefer_fast and free_ratio < min_fast_free_ratio:
        return -1.0
    return (success * 2.0) + (0.25 / (1.0 + latency * 1000.0)) + (0.08 * math.log1p(throughput / 1_000_000.0))


def _update_decisions(state: Dict[str, Any], policy: Dict[str, Any]) -> None:
    devices = state.get("devices") if isinstance(state.get("devices"), dict) else {}
    fast = devices.get("fast") if isinstance(devices.get("fast"), dict) else {}
    durable = devices.get("durable") if isinstance(devices.get("durable"), dict) else {}
    minimum = max(1, int(policy.get("min_samples_before_switch", 3)))
    margin = max(0.0, float(policy.get("switch_margin", 0.18)))
    enough = min(int(fast.get("samples") or 0), int(durable.get("samples") or 0)) >= minimum
    fast_score = _device_score(fast, prefer_fast=True, min_fast_free_ratio=float(policy.get("min_fast_free_ratio", 0.08)))
    durable_score = _device_score(durable, prefer_fast=False, min_fast_free_ratio=0.0)
    decisions = state.setdefault("decisions", {})
    for artifact in ARTIFACT_CLASSES:
        current = decisions.get(artifact) if isinstance(decisions.get(artifact), dict) else {}
        current_tier = str(current.get("tier") or "fast")
        bias = {"index": 0.30, "neural": 0.24, "runtime": 0.12}[artifact]
        adjusted_fast = fast_score + bias
        desired = "fast" if adjusted_fast >= durable_score else "durable"
        if desired != current_tier and (not enough or abs(adjusted_fast - durable_score) < margin):
            desired = current_tier
            reason = "hysteresis_or_insufficient_samples"
        else:
            reason = "online_ewma_device_score"
        decisions[artifact] = {
            "tier": desired,
            "reason": reason,
            "fast_score": round(adjusted_fast, 6),
            "durable_score": round(durable_score, 6),
            "samples_ready": enough,
            "updated_at": _now_iso(),
        }


def update_from_storage_vitals(
    child: str,
    config: Dict[str, Any],
    storage_vitals: Dict[str, Any],
    *,
    force: bool = False,
) -> Dict[str, Any]:
    policy = policy_from_config(config)
    state = load_state(child, config)
    if not bool(policy.get("enabled", True)):
        return state
    now = time.time()
    try:
        last = datetime.fromisoformat(str(state.get("last_probe_at") or "").replace("Z", "+00:00")).timestamp()
    except Exception:
        last = 0.0
    interval = max(60.0, float(policy.get("probe_interval_seconds", 3600.0)))
    if force or now - last >= interval:
        roles = storage_vitals.get("roles") if isinstance(storage_vitals, dict) else {}
        for tier, role in (("fast", "fast_runtime"), ("durable", "project")):
            sample = roles.get(role) if isinstance(roles, dict) and isinstance(roles.get(role), dict) else {}
            raw_path = sample.get("path")
            result = _probe(Path(raw_path), int(policy.get("probe_bytes", 262144))) if raw_path else {"success": False, "error": "path_unavailable"}
            record_observation(
                state, tier, success=bool(result.get("success")),
                latency_seconds=result.get("latency_seconds"),
                throughput_bytes_per_second=result.get("throughput_bytes_per_second"),
                free_ratio=result.get("free_ratio"),
                alpha=float(policy.get("ewma_alpha", 0.25)),
            )
        state["last_probe_at"] = _now_iso()
    _update_decisions(state, policy)
    state["updated_at"] = _now_iso()
    save_state(child, state, config)
    return state


def recommend_rebuildable_tier(
    child: str,
    artifact_class: str,
    config: Optional[Dict[str, Any]],
    *,
    fast_available: bool,
    durable_available: bool = True,
) -> str:
    policy = policy_from_config(config)
    if not durable_available:
        return "fast" if fast_available else "unavailable"
    if not fast_available or not bool(policy.get("enabled", True)):
        return "durable"
    if not bool(policy.get("apply_recommendations", True)):
        return "fast"
    state = load_state(child, config)
    decisions = state.get("decisions") if isinstance(state.get("decisions"), dict) else {}
    decision = decisions.get(artifact_class) if isinstance(decisions.get(artifact_class), dict) else {}
    tier = str(decision.get("tier") or "fast")
    return tier if tier in {"fast", "durable"} else "fast"


__all__ = [
    "ARTIFACT_CLASSES", "DEFAULT_POLICY", "load_state", "policy_from_config",
    "recommend_rebuildable_tier", "record_observation", "save_state",
    "update_from_storage_vitals",
]
