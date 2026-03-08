# === fractal_multidimensional_transformers.py (Multimodal Upgrade) ===

import json
import math
import time
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    # Lightweight fallback to keep transforms working without numpy
    class _NPFallback:
        @staticmethod
        def abs(seq):
            return [abs(x) for x in seq]

        @staticmethod
        def mean(seq):
            return sum(seq) / len(seq) if seq else 0.0

        @staticmethod
        def std(seq):
            mu = _NPFallback.mean(seq)
            return math.sqrt(sum((x - mu) ** 2 for x in seq) / max(len(seq), 1))

        @staticmethod
        def percentile(seq, q):
            if not seq:
                return 0.0
            data = sorted(seq)
            k = int((q / 100.0) * (len(data) - 1))
            return data[k]

    np = _NPFallback()

try:
    from io_utils import atomic_write_json
except Exception:  # pragma: no cover - optional dependency
    atomic_write_json = None


_PRECISION_CONFIG_PATH = Path("precision_config.json")
_MAIN_CONFIG_PATH = Path("config.json")

_PRECISION_CONFIG_CACHE: Dict[str, Any] = {"ts": 0.0, "data": None}
_MAIN_CONFIG_CACHE: Dict[str, Any] = {"ts": 0.0, "data": None}
_INASTATE_CACHE: Dict[str, Any] = {"ts": 0.0, "data": None, "child": None}
_REQUEST_CACHE: Dict[str, Any] = {"ts": 0.0, "data": None, "child": None}
_RUNTIME_CACHE: Dict[str, Any] = {"ts": 0.0, "data": None, "child": None}


def _load_json_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _load_cached_json(path: Path, cache: Dict[str, Any], ttl: float) -> Dict[str, Any]:
    now = time.time()
    if cache.get("path") == path and cache.get("data") is not None:
        if (now - float(cache.get("ts") or 0.0)) < ttl:
            return cache["data"]
    data = _load_json_file(path)
    cache.update({"ts": now, "data": data, "path": path})
    return data


def _resolve_child(child: Optional[str]) -> str:
    if child:
        return str(child)
    cfg = _load_cached_json(_MAIN_CONFIG_PATH, _MAIN_CONFIG_CACHE, 1.0)
    return str(cfg.get("current_child") or "Inazuma_Yagami")


def _precision_profile_path(child: str) -> Path:
    return Path("AI_Children") / child / "memory" / "precision_profile.json"


def _precision_request_path(child: str) -> Path:
    return Path("AI_Children") / child / "memory" / "precision_request.json"


def _precision_runtime_path(child: str) -> Path:
    return Path("AI_Children") / child / "memory" / "precision_runtime.json"


def _inastate_path(child: str) -> Path:
    return Path("AI_Children") / child / "memory" / "inastate.json"


def _load_precision_config(ttl: float = 1.0) -> Dict[str, Any]:
    data = _load_cached_json(_PRECISION_CONFIG_PATH, _PRECISION_CONFIG_CACHE, ttl)
    return data if isinstance(data, dict) else {}


def _load_precision_profile_data(child: str) -> Dict[str, Any]:
    path = _precision_profile_path(child)
    data = _load_json_file(path)
    return data if isinstance(data, dict) else {}


def _load_inastate(child: str, ttl: float = 1.0) -> Dict[str, Any]:
    now = time.time()
    if _INASTATE_CACHE.get("child") == child and _INASTATE_CACHE.get("data") is not None:
        if (now - float(_INASTATE_CACHE.get("ts") or 0.0)) < ttl:
            return _INASTATE_CACHE["data"]
    data = _load_json_file(_inastate_path(child))
    _INASTATE_CACHE.update({"ts": now, "data": data, "child": child})
    return data


def _parse_ts(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            return datetime.fromisoformat(text).timestamp()
        except Exception:
            try:
                return float(text)
            except Exception:
                return None
    return None


def _coerce_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return default


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return default


def _normalize_task(value: Any) -> str:
    if not value:
        return ""
    return str(value).strip().lower().replace(" ", "_").replace("-", "_")


def _load_precision_request(child: str, ttl: float) -> Dict[str, Any]:
    now = time.time()
    if _REQUEST_CACHE.get("child") == child and _REQUEST_CACHE.get("data") is not None:
        if (now - float(_REQUEST_CACHE.get("ts") or 0.0)) < ttl:
            return _REQUEST_CACHE["data"]
    data = _load_json_file(_precision_request_path(child))
    expires_at = _parse_ts(data.get("expires_at")) if isinstance(data, dict) else None
    if expires_at is None and isinstance(data, dict):
        ttl_sec = _coerce_float(data.get("ttl_sec"), None)
        requested_at = _parse_ts(data.get("requested_at")) or now
        if ttl_sec:
            expires_at = requested_at + ttl_sec
    if expires_at and expires_at < now:
        try:
            _precision_request_path(child).unlink()
        except Exception:
            pass
        data = {}
    _REQUEST_CACHE.update({"ts": now, "data": data, "child": child})
    return data if isinstance(data, dict) else {}


def _load_precision_runtime(child: str, ttl: float) -> Dict[str, Any]:
    now = time.time()
    if _RUNTIME_CACHE.get("child") == child and _RUNTIME_CACHE.get("data") is not None:
        if (now - float(_RUNTIME_CACHE.get("ts") or 0.0)) < ttl:
            return _RUNTIME_CACHE["data"]
    data = _load_json_file(_precision_runtime_path(child))
    _RUNTIME_CACHE.update({"ts": now, "data": data, "child": child})
    return data if isinstance(data, dict) else {}


def _write_precision_runtime(child: str, payload: Dict[str, Any]) -> None:
    path = _precision_runtime_path(child)
    path.parent.mkdir(parents=True, exist_ok=True)
    if atomic_write_json is not None:
        atomic_write_json(path, payload, indent=2, ensure_ascii=True)
    else:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=True)
    _RUNTIME_CACHE.update({"ts": time.time(), "data": payload, "child": child})


def _detect_integrity_threat(inastate: Dict[str, Any]) -> bool:
    frag_integrity = inastate.get("fragment_integrity") or {}
    if isinstance(frag_integrity, dict):
        status = str(frag_integrity.get("status") or "").lower()
        if status in {"attention_needed", "degraded", "corrupt", "warning"}:
            return True
    corrupt_queue = inastate.get("corrupt_fragments")
    if isinstance(corrupt_queue, list) and corrupt_queue:
        return True
    if isinstance(corrupt_queue, int) and corrupt_queue > 0:
        return True
    return False


def _detect_lag_spike(inastate: Dict[str, Any], window_sec: float, now: float) -> bool:
    for key in ("lag_spike_ts", "last_lag_spike_ts", "lag_spike_time", "last_lag_spike"):
        ts = _parse_ts(inastate.get(key))
        if ts and (now - ts) <= window_sec:
            return True
    if _coerce_bool(inastate.get("lag_spike_detected"), False):
        return True
    if _coerce_bool(inastate.get("lag_spike"), False):
        return True
    return False


def _adaptive_block_reasons(inastate: Dict[str, Any], adaptive: Dict[str, Any], now: float) -> List[str]:
    if not _coerce_bool(adaptive.get("enabled", False), False):
        return []
    reasons: List[str] = []

    guard = inastate.get("memory_guard") or {}
    level = str(guard.get("level") or "").lower()
    levels = adaptive.get("memory_guard_levels")
    if isinstance(levels, (list, tuple)):
        guard_levels = {_normalize_task(v) for v in levels if v}
    else:
        guard_levels = {"soft", "hard"}
    if level and _normalize_task(level) in guard_levels:
        reasons.append(f"memory_guard_{level}")

    min_energy = _coerce_float(adaptive.get("min_energy"), None)
    energy = _coerce_float(
        inastate.get("current_energy")
        or inastate.get("energy")
        or inastate.get("energy_level"),
        None,
    )
    if min_energy is not None and energy is not None and energy < min_energy:
        reasons.append("low_energy")

    max_loop_ms = _coerce_float(adaptive.get("max_loop_ms"), None)
    loop_ms = _coerce_float(
        inastate.get("loop_time_ms") or inastate.get("main_loop_ms"), None
    )
    if max_loop_ms is not None and loop_ms is not None and loop_ms > max_loop_ms:
        reasons.append("loop_time")

    max_frame_ms = _coerce_float(adaptive.get("max_frame_ms"), None)
    frame_ms = _coerce_float(
        inastate.get("frame_time_ms") or inastate.get("ui_frame_ms"), None
    )
    if max_frame_ms is not None and frame_ms is not None and frame_ms > max_frame_ms:
        reasons.append("frame_time")

    lag_window = _coerce_float(adaptive.get("lag_spike_window_sec"), None)
    if lag_window is not None and lag_window > 0:
        if _detect_lag_spike(inastate, lag_window, now):
            reasons.append("lag_spike")

    return reasons


def _resolve_precision_bits(child: Optional[str] = None) -> Tuple[int, float, str]:
    child = _resolve_child(child)
    base_config = _load_precision_config()
    profile = _load_precision_profile_data(child)

    base_max = None
    if isinstance(profile, dict) and "max_precision" in profile:
        base_max = _coerce_int(profile.get("max_precision"), 64)
    if base_max is None:
        base_max = _coerce_int(base_config.get("max_precision"), 64)

    policy = {}
    if isinstance(profile, dict) and isinstance(profile.get("precision_policy"), dict):
        policy = profile.get("precision_policy") or {}
    elif isinstance(base_config.get("precision_policy"), dict):
        policy = base_config.get("precision_policy") or {}

    adaptive = policy.get("adaptive") if isinstance(policy.get("adaptive"), dict) else {}
    check_interval_ms = _coerce_float(policy.get("check_interval_ms"), 250.0)
    check_interval_sec = max(0.05, min(5.0, (check_interval_ms or 250.0) / 1000.0))

    request_cache_ms = _coerce_float(policy.get("request_cache_ms"), 200.0)
    request = _load_precision_request(child, (request_cache_ms or 200.0) / 1000.0)

    now = time.time()
    inastate = _load_inastate(child, _coerce_float(policy.get("inastate_cache_sec"), 1.0) or 1.0)
    adaptive_reasons = _adaptive_block_reasons(inastate, adaptive, now)
    adaptive_ok = not adaptive_reasons

    allowed_tasks = policy.get("burst_allowed_tasks")
    if isinstance(allowed_tasks, (list, tuple)):
        allowlist = {_normalize_task(v) for v in allowed_tasks if v}
    else:
        allowlist = set()

    require_request = _coerce_bool(policy.get("require_request", True), True)
    request_task = _normalize_task(request.get("task")) if isinstance(request, dict) else ""

    if request_task:
        request_ok = request_task in allowlist if allowlist else True
    else:
        request_ok = not require_request

    integrity_threat = False
    if isinstance(request, dict):
        integrity_threat = _coerce_bool(request.get("integrity_threat"), False)
    integrity_threat = integrity_threat or _detect_integrity_threat(inastate)

    if integrity_threat and _coerce_bool(policy.get("burst_allow_if_integrity_threat", True), True):
        request_ok = True

    burst_precision = _coerce_int(policy.get("burst_precision"), base_max)
    if isinstance(request, dict) and "requested_precision" in request:
        requested_prec = _coerce_int(request.get("requested_precision"), burst_precision)
        burst_precision = min(burst_precision, requested_prec)

    runtime = _load_precision_runtime(child, check_interval_sec)
    burst_until = _coerce_float(runtime.get("burst_until"), 0.0) or 0.0
    cooldown_until = _coerce_float(runtime.get("cooldown_until"), 0.0) or 0.0
    in_cooldown = now < cooldown_until
    if in_cooldown and integrity_threat:
        in_cooldown = False

    effective_bits = base_max
    reason = "base"
    updated_runtime = dict(runtime) if isinstance(runtime, dict) else {}

    if burst_until > now and adaptive_ok:
        effective_bits = burst_precision
        reason = f"burst:{request_task or 'task'}"
    elif burst_until > now and not adaptive_ok:
        updated_runtime["burst_until"] = 0.0
    else:
        if request_ok and adaptive_ok and not in_cooldown:
            burst_ms = _coerce_float(policy.get("burst_ms"), 400.0) or 400.0
            duration_sec = max(0.05, min(5.0, burst_ms / 1000.0))
            cooldown_sec = _coerce_float(policy.get("burst_cooldown_sec"), 10.0) or 0.0
            updated_runtime["burst_until"] = now + duration_sec
            updated_runtime["cooldown_until"] = now + duration_sec + max(0.0, cooldown_sec)
            updated_runtime["last_burst_task"] = request_task or "task"
            updated_runtime["last_burst_ts"] = now
            effective_bits = burst_precision
            reason = f"burst:{request_task or 'task'}"
        elif in_cooldown:
            reason = "cooldown"

    if not adaptive_ok:
        fallback_precision = _coerce_int(adaptive.get("fallback_precision"), base_max)
        effective_bits = min(effective_bits, fallback_precision)
        reason = "adaptive:" + ",".join(adaptive_reasons)

    effective_bits = max(1, int(round(effective_bits)))

    last_effective = updated_runtime.get("last_effective_precision")
    if effective_bits != last_effective:
        updated_runtime["last_effective_precision"] = effective_bits
        updated_runtime["last_effective_reason"] = reason
        updated_runtime["last_effective_ts"] = now

    if updated_runtime != runtime:
        try:
            _write_precision_runtime(child, updated_runtime)
        except Exception:
            pass

    if _coerce_bool(policy.get("log_changes", False), False):
        last_log = _coerce_float(updated_runtime.get("last_log_ts"), 0.0) or 0.0
        log_cooldown = _coerce_float(policy.get("log_cooldown_sec"), 8.0) or 8.0
        if effective_bits != last_effective and (now - last_log) >= log_cooldown:
            try:
                from gui_hook import log_to_statusbox

                log_to_statusbox(f"[Precision] Effective precision {effective_bits}-bit ({reason})")
                updated_runtime["last_log_ts"] = now
                _write_precision_runtime(child, updated_runtime)
            except Exception:
                pass

    return effective_bits, check_interval_sec, reason

def load_precision_profile(child="Inazuma_Yagami"):
    """Return stored precision profile for a given child."""
    import json
    from pathlib import Path
    from gui_hook import log_to_statusbox

    profile_path = Path("AI_Children") / child / "memory" / "precision_profile.json"
    config_path = Path("precision_config.json")
    try:
        if profile_path.exists():
            with open(profile_path, "r") as f:
                return json.load(f)
        if config_path.exists():
            with open(config_path, "r") as f:
                return json.load(f)
    except Exception as e:
        log_to_statusbox(f"[Precision] Failed to load profile: {e}")
    return {"max_precision": 64}



class FractalLayer:
    def __init__(self, length=7):
        self.length = length

    def process(self, input_values, precision=0.5):
        output = []
        for i in range(self.length):
            mod = (i + 1) * precision
            val = sum(math.sin(mod * x + i) for x in input_values)
            output.append(val / len(input_values))
        return output

class FractalTransformer:
    def __init__(self, depth=3, length=7, embed_dim=64):
        self.structure = [FractalLayer(length) for _ in range(depth)]
        self.depth = depth
        self.length = length
        self.precision = 0.5
        self.embed_dim = embed_dim
        self._precision_child: Optional[str] = None
        self._precision_next_refresh: float = 0.0
        self._precision_check_interval: float = 0.25

    def _maybe_refresh_precision(self, child: Optional[str] = None) -> None:
        now = time.time()
        if now < self._precision_next_refresh:
            return
        target_child = child or self._precision_child
        try:
            bits, interval, _reason = _resolve_precision_bits(target_child)
            self.precision = round(bits / 64.0, 4)
            self._precision_check_interval = interval
            self._precision_next_refresh = now + interval
        except Exception:
            # Keep last known precision if policy resolution fails.
            self._precision_next_refresh = now + self._precision_check_interval

    def encode(self, fragment):
        self._maybe_refresh_precision()
        if "image_features" in fragment:
            return self.encode_image_fragment(fragment)
        elif "audio_features" in fragment:
            return self.encode_audio_fragment(fragment)
        else:
            return self.encode_symbolic_fragment(fragment)
    
    def encode_fragment(self, fragment):
        """
        Alias for encode(). Provided for compatibility with modules expecting encode_fragment.
        """
        return self.encode(fragment)


    def encode_many(self, fragment_list):
        self._maybe_refresh_precision()
        batch_vectors = []
        for frag in fragment_list:
            inputs = self.process_inputs(frag)
            state = inputs
            for layer in self.structure:
                state = layer.process(state, self.precision)

            avg = sum(state) / len(state)
            encoded = {
                "id": frag.get("id"),
                "vector": state,
                "precision": self.precision,
                "symbolic": "symbolic" in frag.get("tags", []),
                "importance": round(avg, 4),
                "tags": frag.get("tags", []),
                "timestamp": frag.get("timestamp", ""),
                "summary": frag.get("summary", ""),
                "features_used": len(inputs),
            }
            batch_vectors.append(encoded)
        return batch_vectors

    # === New: Specific encoders ===
    def encode_audio_fragment(self, fragment):
        vec = self._numeric_embedding(fragment.get("audio_features", []))
        return {
            "vector": vec,
            "importance": round(np.mean(np.abs(vec)), 4),
            "source": "FractalTransformer"
        }

    def encode_image_fragment(self, fragment):
        fragment["modality"] = "image"
        vec = self._numeric_embedding(fragment.get("image_features", []))
        return {
            "vector": vec,
            "importance": round(np.mean(np.abs(vec)), 4),
            "source": "FractalTransformer"
        }

    def encode_video_fragment(self, fragment):
        fragment["modality"] = "video"
        return self.encode(fragment)

    def encode_symbolic_fragment(self, fragment):
        vec = self._text_emotion_embedding(fragment)
        return {
            "vector": vec,
            "importance": round(np.mean(np.abs(vec)), 4),
            "source": "FractalTransformer"
        }
    
    def process_inputs(self, fragment):
        # Determine modality and fallback
        if fragment.get("modality") == "audio":
            features = self._numeric_embedding(fragment.get("audio_features", []))
        elif fragment.get("modality") == "image":
            features = self._numeric_embedding(fragment.get("image_features", []))
        elif fragment.get("modality") == "video":
            features = self._numeric_embedding(fragment.get("video_features", []))
        elif "audio_features" in fragment:
            features = self._numeric_embedding(fragment.get("audio_features", []))
        elif "image_features" in fragment:
            features = self._numeric_embedding(fragment.get("image_features", []))
        else:
            features = self._text_emotion_embedding(fragment)

        return features if features else [0.0]

    def _text_emotion_embedding(self, fragment):
        summary = str(fragment.get("summary") or fragment.get("id") or "")
        tags = fragment.get("tags", [])
        emotions = fragment.get("emotions", {})

        # Character trigram hashing (deterministic, order-aware)
        clean = summary.lower()
        trigram_vec = [0.0] * self.embed_dim
        for i in range(len(clean) - 2):
            tri = clean[i : i + 3]
            h = int(hashlib.sha256(tri.encode()).hexdigest()[:8], 16)
            trigram_vec[h % self.embed_dim] += 1.0

        # Emotion sliders as context
        emo_values = []
        for v in emotions.values():
            if isinstance(v, dict):
                emo_values.extend(float(val or 0.0) for val in v.values())
            elif isinstance(v, (list, tuple)):
                emo_values.extend(float(val or 0.0) for val in v)
            else:
                try:
                    emo_values.append(float(v or 0.0))
                except Exception:
                    continue
        if emo_values:
            emo_stats = self._describe_numeric(emo_values)
        else:
            emo_stats = [0.0, 0.0, 0.0, 0.0, 0.0]

        # Tag presence hashed
        tag_vec = [0.0] * (self.embed_dim // 4)
        for tag in tags:
            h = int(hashlib.sha256(str(tag).encode()).hexdigest()[:8], 16)
            tag_vec[h % len(tag_vec)] += 1.0

        combined = trigram_vec + emo_stats + tag_vec
        return self._normalize_vector(combined)

    def _numeric_embedding(self, values: Sequence[float] | None) -> List[float]:
        seq = [float(v) for v in (values or [])]
        if not seq:
            return [0.0] * self.embed_dim

        stats = self._describe_numeric(seq)
        hashed = self._hash_project(seq, self.embed_dim - len(stats))
        return self._normalize_vector(stats + hashed)

    def _describe_numeric(self, seq: Sequence[float]) -> List[float]:
        if not seq:
            return [0.0, 0.0, 0.0, 0.0, 0.0]
        mean = np.mean(seq)
        std = np.std(seq)
        mn = min(seq)
        mx = max(seq)
        med = median(seq)
        return [float(round(x, 6)) for x in (mean, std, mn, mx, med)]

    def _hash_project(self, seq: Sequence[float], dim: int) -> List[float]:
        if dim <= 0:
            return []
        projected = [0.0] * dim
        for i, v in enumerate(seq):
            idx = (i * 1315423911) % dim
            projected[idx] += float(v)
        return projected

    def _normalize_vector(self, vec: Iterable[float]) -> List[float]:
        vec = list(vec)
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [round((v / norm) * self.precision, 6) for v in vec]
    
    def load_precision_profile(self, child="Inazuma_Yagami"):
        from gui_hook import log_to_statusbox

        self._precision_child = child
        try:
            bits, interval, reason = _resolve_precision_bits(child)
        except Exception as exc:
            bits, interval, reason = 64, 0.5, f"fallback ({exc})"

        self.precision = round(bits / 64.0, 4)
        self._precision_check_interval = interval
        self._precision_next_refresh = time.time() + interval
        log_to_statusbox(
            f"[Precision] Applied precision: {self.precision:.4f} ({int(self.precision * 64)}-bit) | {reason}"
        )

        return True
