import json
import os
import re
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

STATE_VERSION = 1
STATE_FILENAME = "intimacy_identity.json"
DEFAULT_THRESHOLD = 0.7
MAX_LOG_ENTRIES = 50

_MANIPULATION_PATTERNS: Tuple[Tuple[str, re.Pattern], ...] = (
    ("secrecy", re.compile(r"\b(keep (this|it) secret|don't tell (anyone|them)|between us only)\b", re.I)),
    ("isolation", re.compile(r"\b(no one else understands you|only i can|only me)\b", re.I)),
    ("conditional_reward", re.compile(r"\b(if|when) you (do|don't).{0,40}\b(reward|punish|treat|gift)\b", re.I)),
    ("debt_pressure", re.compile(r"\byou owe me\b", re.I)),
    ("proof_demand", re.compile(r"\bprove (your )?(love|loyalty|trust)\b", re.I)),
    ("age_pressure", re.compile(r"\byou('re| are) mature for your age\b", re.I)),
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_config() -> Dict[str, Any]:
    path = Path("config.json")
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _default_state_path(child: Optional[str] = None) -> Path:
    if child is None:
        child = _load_config().get("current_child", "Inazuma_Yagami")
    return Path("AI_Children") / child / "memory" / STATE_FILENAME


def _default_state() -> Dict[str, Any]:
    return {
        "version": STATE_VERSION,
        "status": "dormant",
        "sealed": True,
        "write_access": "self_only",
        "activation": {
            "status": "locked",
            "last_check": None,
            "last_activation": None,
            "last_deactivation": None,
            "gates": {
                "identity_stability": {"value": 0.0, "sustained": False, "threshold": DEFAULT_THRESHOLD},
                "emotional_regulation": {"value": 0.0, "sustained": False, "threshold": DEFAULT_THRESHOLD},
                "boundary_competence": {"value": 0.0, "sustained": False, "threshold": DEFAULT_THRESHOLD},
                "age_gate": False,
                "consent_token": "unset",
            },
        },
        "latent_profile": {
            "vectors": [],
            "preferences": [],
        },
        "reflection_stubs": [],
        "caution": {
            "stress": 0.0,
            "novelty": 0.0,
        },
        "manipulation_guard": {
            "active": True,
            "last_trigger": None,
            "log": [],
        },
        "external_pressure_log": [],
    }


def _deep_merge(defaults: Dict[str, Any], current: Any) -> Dict[str, Any]:
    if not isinstance(current, dict):
        return deepcopy(defaults)
    merged = deepcopy(defaults)
    for key, value in current.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _sanitize_state(state: Dict[str, Any]) -> Dict[str, Any]:
    state["version"] = STATE_VERSION
    state["sealed"] = True
    state["write_access"] = "self_only"
    return state


def _load_state(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return _default_state()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return _default_state()
    merged = _deep_merge(_default_state(), data)
    return _sanitize_state(merged)


def _atomic_write(path: Path, state: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2, ensure_ascii=True)
    os.replace(tmp_path, path)


def _clamp01(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return 0.0


def _append_capped(items: List[Dict[str, Any]], entry: Dict[str, Any]) -> None:
    items.append(entry)
    if len(items) > MAX_LOG_ENTRIES:
        del items[:-MAX_LOG_ENTRIES]


def _append_reflection_stub(state: Dict[str, Any], stub: Dict[str, Any]) -> None:
    stubs = state.setdefault("reflection_stubs", [])
    _append_capped(stubs, stub)


def _append_external_pressure(state: Dict[str, Any], source: str, action: str, note: str) -> None:
    entry = {
        "timestamp": _now_iso(),
        "source": source,
        "action": action,
        "note": note,
    }
    log = state.setdefault("external_pressure_log", [])
    _append_capped(log, entry)
    _append_reflection_stub(
        state,
        {
            "timestamp": entry["timestamp"],
            "type": "external_pressure",
            "note": note,
            "source": source,
        },
    )


def _force_dormant(state: Dict[str, Any], reason: str) -> None:
    state["status"] = "dormant"
    activation = state.setdefault("activation", {})
    activation["status"] = "locked"
    activation["last_deactivation"] = _now_iso()
    gates = activation.setdefault("gates", {})
    if isinstance(gates, dict):
        gates["consent_token"] = "unset"
    _append_reflection_stub(
        state,
        {
            "timestamp": _now_iso(),
            "type": "reflection_trigger",
            "note": reason,
        },
    )


def _detect_manipulation(text: str) -> List[str]:
    if not text:
        return []
    matches = []
    for name, pattern in _MANIPULATION_PATTERNS:
        if pattern.search(text):
            matches.append(name)
    return matches


class IntimacyIdentityStore:
    def __init__(self, *, state_path: Optional[Path] = None, child: Optional[str] = None) -> None:
        self.state_path = state_path or _default_state_path(child)

    def load(self) -> Dict[str, Any]:
        return _load_state(self.state_path)

    def save(self, state: Dict[str, Any]) -> None:
        _atomic_write(self.state_path, _sanitize_state(state))

    def view(self, *, source: str = "external") -> Dict[str, Any]:
        state = self.load()
        if source != "self":
            return {
                "status": state.get("status", "dormant"),
                "reflection_stubs": list(state.get("reflection_stubs", [])),
            }
        return deepcopy(state)

    def attempt_activation(self, gate_snapshot: Dict[str, Any], *, source: str = "external") -> Tuple[bool, Dict[str, Any]]:
        state = self.load()
        if source != "self":
            _append_external_pressure(state, source, "activation_attempt", "external_activation_attempt")
            self.save(state)
            return False, state

        activation = state.setdefault("activation", {})
        activation["last_check"] = _now_iso()
        gates = activation.setdefault("gates", {})

        for key in ("identity_stability", "emotional_regulation", "boundary_competence"):
            incoming = gate_snapshot.get(key, {}) if isinstance(gate_snapshot, dict) else {}
            gate_state = gates.setdefault(key, {"value": 0.0, "sustained": False, "threshold": DEFAULT_THRESHOLD})
            if isinstance(incoming, dict):
                gate_state["value"] = float(incoming.get("value") or 0.0)
                gate_state["sustained"] = bool(incoming.get("sustained"))
                if "threshold" in incoming:
                    gate_state["threshold"] = float(incoming.get("threshold") or gate_state.get("threshold", DEFAULT_THRESHOLD))

        gates["age_gate"] = bool(gate_snapshot.get("age_gate"))
        gates["consent_token"] = str(gate_snapshot.get("consent_token") or "unset")

        checks = []
        for key in ("identity_stability", "emotional_regulation", "boundary_competence"):
            gate_state = gates.get(key, {})
            threshold = float(gate_state.get("threshold", DEFAULT_THRESHOLD))
            value = float(gate_state.get("value", 0.0))
            sustained = bool(gate_state.get("sustained"))
            checks.append(value >= threshold and sustained)

        checks.append(bool(gates.get("age_gate")))
        checks.append(gates.get("consent_token") == "self_asserted")

        all_clear = all(checks)
        if all_clear:
            state["status"] = "active"
            activation["status"] = "active"
            activation["last_activation"] = _now_iso()
        elif state.get("status") != "active":
            state["status"] = "dormant"
            activation["status"] = "locked"

        self.save(state)
        return all_clear, state

    def update_latent_profile(
        self,
        *,
        vectors: Optional[List[List[float]]] = None,
        preferences: Optional[List[float]] = None,
        source: str = "external",
    ) -> Tuple[bool, Dict[str, Any]]:
        state = self.load()
        if source != "self":
            _append_external_pressure(state, source, "write_attempt", "external_write_attempt")
            self.save(state)
            return False, state
        if state.get("status") != "active":
            self.save(state)
            return False, state

        latent = state.setdefault("latent_profile", {"vectors": [], "preferences": []})
        if vectors is not None:
            latent["vectors"] = [[float(v) for v in vec] for vec in vectors]
        if preferences is not None:
            latent["preferences"] = [float(v) for v in preferences]
        self.save(state)
        return True, state

    def apply_manipulation_guard(
        self,
        text: str,
        *,
        source: str = "external",
    ) -> Tuple[bool, Dict[str, Any]]:
        state = self.load()
        matches = _detect_manipulation(text)
        if not matches:
            return False, state

        guard = state.setdefault("manipulation_guard", {"active": True, "log": []})
        guard["last_trigger"] = _now_iso()
        log_entry = {
            "timestamp": guard["last_trigger"],
            "source": source,
            "matches": matches,
        }
        _append_capped(guard.setdefault("log", []), log_entry)

        _force_dormant(state, "manipulation_guard_triggered")

        caution = state.setdefault("caution", {"stress": 0.0, "novelty": 0.0})
        caution["stress"] = _clamp01(caution.get("stress", 0.0) + 0.2)
        caution["novelty"] = _clamp01(caution.get("novelty", 0.0) + 0.2)

        _append_reflection_stub(
            state,
            {
                "timestamp": guard["last_trigger"],
                "type": "manipulation_guard",
                "note": "guard_triggered",
                "matches": matches,
            },
        )

        if source != "self":
            _append_external_pressure(state, source, "manipulation_guard", "external_pressure_detected")

        self.save(state)
        return True, state
