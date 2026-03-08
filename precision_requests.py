from __future__ import annotations

import json
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from io_utils import atomic_write_json
except Exception:  # pragma: no cover - optional dependency
    atomic_write_json = None


_MAIN_CONFIG = Path("config.json")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_main_config() -> Dict[str, Any]:
    if _MAIN_CONFIG.exists():
        try:
            with _MAIN_CONFIG.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
                return data if isinstance(data, dict) else {}
        except Exception:
            return {}
    return {}


def _resolve_child(child: Optional[str]) -> str:
    if child:
        return str(child)
    cfg = _load_main_config()
    return str(cfg.get("current_child") or "Inazuma_Yagami")


def _request_path(child: str) -> Path:
    return Path("AI_Children") / child / "memory" / "precision_request.json"


def request_precision(
    task: str,
    *,
    child: Optional[str] = None,
    requested_precision: Optional[int] = None,
    ttl_sec: float = 2.0,
    reason: Optional[str] = None,
    integrity_threat: bool = False,
    stakes: Optional[float] = None,
    source: Optional[str] = None,
) -> Path:
    child = _resolve_child(child)
    ttl = max(0.1, float(ttl_sec or 0.0))
    expires_at = datetime.now(timezone.utc).timestamp() + ttl
    payload: Dict[str, Any] = {
        "task": str(task),
        "requested_at": _now_iso(),
        "expires_at": datetime.fromtimestamp(expires_at, timezone.utc).isoformat(),
        "ttl_sec": ttl,
        "integrity_threat": bool(integrity_threat),
    }
    if requested_precision is not None:
        try:
            payload["requested_precision"] = int(requested_precision)
        except Exception:
            pass
    if reason:
        payload["reason"] = str(reason)
    if stakes is not None:
        try:
            payload["stakes"] = float(stakes)
        except Exception:
            pass
    if source:
        payload["source"] = str(source)

    path = _request_path(child)
    path.parent.mkdir(parents=True, exist_ok=True)
    if atomic_write_json is not None:
        atomic_write_json(path, payload, indent=2, ensure_ascii=True)
    else:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=True)
    return path


def clear_precision_request(*, child: Optional[str] = None) -> None:
    child = _resolve_child(child)
    path = _request_path(child)
    try:
        path.unlink()
    except FileNotFoundError:
        return
    except Exception:
        return


@contextmanager
def precision_request(
    task: str,
    *,
    child: Optional[str] = None,
    requested_precision: Optional[int] = None,
    ttl_sec: float = 2.0,
    reason: Optional[str] = None,
    integrity_threat: bool = False,
    stakes: Optional[float] = None,
    source: Optional[str] = None,
):
    request_precision(
        task,
        child=child,
        requested_precision=requested_precision,
        ttl_sec=ttl_sec,
        reason=reason,
        integrity_threat=integrity_threat,
        stakes=stakes,
        source=source,
    )
    try:
        yield
    finally:
        clear_precision_request(child=child)
