"""Cooperative I/O pressure signals for latency-sensitive frontends."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

DEFAULT_POLICY = {"enabled": True, "soft_lag_seconds": 0.35, "hard_lag_seconds": 1.5, "ttl_seconds": 20.0}

def classify_latency(lag_seconds: float, policy: Optional[Dict[str, Any]] = None) -> str:
    cfg = dict(DEFAULT_POLICY)
    if isinstance(policy, dict): cfg.update(policy)
    if not bool(cfg.get("enabled", True)): return "clear"
    lag = max(0.0, float(lag_seconds))
    if lag >= max(0.0, float(cfg["hard_lag_seconds"])): return "hard"
    if lag >= max(0.0, float(cfg["soft_lag_seconds"])): return "soft"
    return "clear"

def pressure_signal(client: str, lag_seconds: float, *, policy: Optional[Dict[str, Any]] = None, observed_at: Optional[str] = None) -> Dict[str, Any]:
    level = classify_latency(lag_seconds, policy)
    return {"client": str(client), "level": level, "lag_seconds": round(max(0.0, float(lag_seconds)), 4), "observed_at": observed_at or datetime.now(timezone.utc).isoformat(), "reason": "latency_sensitive_client_blocked" if level != "clear" else "responsive"}

def active_pressure(signal: Any, *, now: Optional[datetime] = None, policy: Optional[Dict[str, Any]] = None) -> str:
    if not isinstance(signal, dict): return "clear"
    cfg = dict(DEFAULT_POLICY)
    if isinstance(policy, dict): cfg.update(policy)
    if not bool(cfg.get("enabled", True)): return "clear"
    try:
        stamp = datetime.fromisoformat(str(signal.get("observed_at") or "").replace("Z", "+00:00"))
        if stamp.tzinfo is None: stamp = stamp.replace(tzinfo=timezone.utc)
        if ((now or datetime.now(timezone.utc)) - stamp.astimezone(timezone.utc)).total_seconds() > float(cfg["ttl_seconds"]): return "clear"
    except Exception: return "clear"
    level = str(signal.get("level") or "clear").lower()
    return level if level in {"soft", "hard"} else "clear"

__all__ = ["DEFAULT_POLICY", "active_pressure", "classify_latency", "pressure_signal"]
