from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import hashlib
import json
import random

from ina_ml import coerce_vector, mean_center, normalize_distribution, shannon_entropy, vector_norm
from origin_record import make_origin

SymbolId = str


def _vec(symbol_weights: Dict[SymbolId, float], symbols: List[SymbolId]) -> list[float]:
    return [float(symbol_weights.get(key, 0.0)) for key in symbols]


def _to_dict(vector: Sequence[float], symbols: List[SymbolId]) -> Dict[SymbolId, float]:
    return {key: float(value) for key, value in zip(symbols, vector)}


def _normalize(vector: Iterable[float]) -> list[float]:
    return normalize_distribution(vector)


def _uniform_like(vector: Sequence[float]) -> list[float]:
    return [1.0 / len(vector)] * len(vector) if vector else []


def _shannon_entropy(vector: Iterable[float]) -> float:
    return shannon_entropy(vector)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _has_any(tags: Tuple[str, ...], whitelist: Tuple[str, ...]) -> bool:
    return any(tag in whitelist for tag in tags)


def _fragment_along_links(
    vector: Sequence[float], symbols: List[SymbolId],
    links: Dict[SymbolId, Dict[SymbolId, float]], alpha: float, cap: float,
) -> list[float]:
    if alpha <= 0:
        return list(vector)
    result = list(vector)
    symbol_index = {symbol: index for index, symbol in enumerate(symbols)}
    for index, symbol in enumerate(symbols):
        neighbours = links.get(symbol)
        if not neighbours:
            continue
        weight = result[index]
        move = min(alpha * weight, cap)
        if move <= 0 or weight <= 0:
            continue
        result[index] -= move
        total = sum(neighbours.values())
        if total <= 0:
            result[index] += move
            continue
        for neighbour, neighbour_weight in neighbours.items():
            target = symbol_index.get(neighbour)
            if target is not None:
                result[target] += move * (neighbour_weight / total)
    return result


def _refocus(weights: Dict[SymbolId, float], focus: List[SymbolId], boost: float) -> Dict[SymbolId, float]:
    if not focus or boost <= 0:
        return weights
    symbols = list(weights)
    vector = _vec(weights, symbols)
    focus_set = set(focus)
    for index, symbol in enumerate(symbols):
        if symbol in focus_set:
            vector[index] *= 1.0 + boost
    return _to_dict(_normalize(vector), symbols)


def _topk_symbols_from_emotion(weights: Dict[SymbolId, float], k: int) -> List[SymbolId]:
    return sorted(weights, key=weights.get, reverse=True)[:k]


@dataclass
class DriftConfig:
    drift_rate: float = 0.002
    fuzz_sigma: float = 0.03
    rng_seed: Optional[int] = None
    max_fragmentation: float = 0.25
    emotion_bias_strength: float = 0.002
    decay_to_ambiguity: float = 0.001
    dream_tags_whitelist: Tuple[str, ...] = ("dreamstate", "meditation", "silence")
    resolve_boost: float = 0.5
    resolve_half_life_steps: int = 32
    log_history: bool = True
    max_history: int = 2048
    log_dir: str = ""


@dataclass
class DriftState:
    step: int
    symbol_weights: Dict[SymbolId, float]
    symbol_links: Dict[SymbolId, Dict[SymbolId, float]]
    emotion_vector: Sequence[float]
    fuzz_level: float
    entropy_score: float
    tags_active: Tuple[str, ...] = ()


class SoulDriftTransformer:
    VERSION = "V2"

    def __init__(self, cfg: DriftConfig, init_state: DriftState) -> None:
        self.cfg = cfg
        self.state = init_state
        self.state.emotion_vector = coerce_vector(init_state.emotion_vector)
        self.rng = random.Random(cfg.rng_seed)
        self._resolve_decay_counter = 0
        self._focus_symbols: List[SymbolId] = []
        self._history: List[DriftState] = []
        self._last_trigger: Optional[str] = None
        self._last_telemetry: Dict[str, object] = {}
        self.log_dir = Path(cfg.log_dir or ".")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / "drift_log.ndjson"

    def _append_history(self, state: DriftState) -> None:
        snapshot = replace(state)
        self._history.append(snapshot)
        if len(self._history) > self.cfg.max_history:
            self._history.pop(0)
        if self.cfg.log_history:
            top = sorted(state.symbol_weights.items(), key=lambda item: item[1], reverse=True)[:3]
            entry = {
                "step": state.step, "entropy": state.entropy_score, "fuzz": state.fuzz_level,
                "top_symbols": top, "trigger": self._last_trigger,
            }
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry) + "\n")
        self._last_trigger = None

    def step(self, silence: bool = True) -> DriftState:
        state = self.state
        symbols = list(state.symbol_weights)
        vector = _vec(state.symbol_weights, symbols)
        previous_entropy = state.entropy_score
        noise_scale = 0.5 if self._resolve_decay_counter > 0 else 1.0
        vector = [
            value + self.rng.gauss(0.0, self.cfg.fuzz_sigma) * noise_scale * self.cfg.drift_rate
            for value in vector
        ]
        uniform = _uniform_like(vector)
        vector = [
            (1.0 - self.cfg.decay_to_ambiguity) * value + self.cfg.decay_to_ambiguity * uniform[index]
            for index, value in enumerate(vector)
        ]
        alpha = state.fuzz_level * self.cfg.drift_rate
        vector = _fragment_along_links(vector, symbols, state.symbol_links, alpha, self.cfg.max_fragmentation)
        if self._resolve_decay_counter > 0 and self._focus_symbols:
            boost = (self._resolve_decay_counter / self.cfg.resolve_half_life_steps) * self.cfg.resolve_boost
            vector = _vec(_refocus(_to_dict(vector, symbols), self._focus_symbols, boost), symbols)

        emotion_bias_applied = 0.0
        emotion_values = coerce_vector(state.emotion_vector)
        if silence and _has_any(state.tags_active, self.cfg.dream_tags_whitelist) and emotion_values:
            bias = mean_center(emotion_values[index % len(emotion_values)] for index in range(len(vector)))
            norm = vector_norm(bias)
            if norm > 0 and self.cfg.emotion_bias_strength > 0:
                scaled_bias = [(value / norm) * self.cfg.emotion_bias_strength for value in bias]
                vector = [value + scaled_bias[index] for index, value in enumerate(vector)]
                emotion_bias_applied = sum(abs(value) for value in scaled_bias)

        vector = _normalize(max(0.0, value) for value in vector)
        state.symbol_weights = _to_dict(vector, symbols)
        state.entropy_score = _shannon_entropy(vector)
        state.fuzz_level = _clamp(state.fuzz_level + (0.01 if silence else -0.02), 0.0, 1.0)
        if self._resolve_decay_counter > 0:
            state.fuzz_level = _clamp(state.fuzz_level - 0.02, 0.0, 1.0)
            self._resolve_decay_counter -= 1
        state.step += 1
        if self.cfg.log_history:
            self._append_history(state)

        serialized = json.dumps(sorted(state.symbol_weights.items()), separators=(",", ":"))
        delta_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        origin = make_origin(
            self.__class__.__name__, self.VERSION,
            inputs={"step": state.step - 1, "silence": silence, "tags": state.tags_active},
            trigger="dream_or_silence" if silence else "active_drift",
            metadata={"emotion_bias_applied": round(emotion_bias_applied, 6)},
        )
        self._last_telemetry = {
            "intent": "creative_entropy",
            "entropy_bump": round(state.entropy_score - previous_entropy, 4),
            "fuzz_level": round(state.fuzz_level, 4), "delta_graph_hash": delta_hash,
            "step": state.step, "emotion_bias_applied": round(emotion_bias_applied, 6),
            "numeric_backend": "ina_ml", "origins": [origin],
        }
        return state

    def run_session(self, steps: int, silence: bool = True) -> DriftState:
        start = self.snapshot()
        start_weights = start.symbol_weights.copy()
        for _ in range(max(0, int(steps))):
            self.step(silence=silence)
        end_state = self.state
        if self.cfg.log_history:
            differences = {key: end_state.symbol_weights.get(key, 0.0) - start_weights.get(key, 0.0) for key in start_weights}
            summary = {
                "start_entropy": start.entropy_score, "end_entropy": end_state.entropy_score,
                "steps": max(0, int(steps)),
                "symbols_risen": sorted(differences.items(), key=lambda item: item[1], reverse=True)[:3],
                "symbols_fallen": sorted(differences.items(), key=lambda item: item[1])[:3],
            }
            (self.log_dir / "session_summary.json").write_text(json.dumps(summary), encoding="utf-8")
        return end_state

    def inject_trigger(self, emotion_delta: Iterable[float], tag: str = "trigger") -> DriftState:
        state = self.state
        current = coerce_vector(state.emotion_vector)
        delta = coerce_vector(emotion_delta)
        size = max(len(current), len(delta))
        state.emotion_vector = [
            (current[index] if index < len(current) else 0.0) + (delta[index] if index < len(delta) else 0.0)
            for index in range(size)
        ]
        focus = _topk_symbols_from_emotion(state.symbol_weights, k=12)
        state.symbol_weights = _refocus(state.symbol_weights, focus, self.cfg.resolve_boost)
        state.fuzz_level = _clamp(state.fuzz_level - 0.4, 0.0, 1.0)
        self._resolve_decay_counter = self.cfg.resolve_half_life_steps
        self._focus_symbols = focus
        state.tags_active += (tag,)
        self._last_trigger = tag
        state.entropy_score = _shannon_entropy(_vec(state.symbol_weights, list(state.symbol_weights)))
        if self.cfg.log_history:
            self._append_history(state)
        return state

    def snapshot(self) -> DriftState:
        return replace(self.state)

    def intent_telemetry(self) -> Dict[str, object]:
        return dict(self._last_telemetry)
