from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple
import json
import hashlib
import numpy as np
from pathlib import Path

SymbolId = str


def _vec(symbol_weights: Dict[SymbolId, float], symbols: List[SymbolId]) -> np.ndarray:
    return np.array([symbol_weights.get(k, 0.0) for k in symbols], dtype=float)


def _to_dict(vec: np.ndarray, symbols: List[SymbolId]) -> Dict[SymbolId, float]:
    return {k: float(v) for k, v in zip(symbols, vec)}


def _normalize(vec: np.ndarray) -> np.ndarray:
    total = vec.sum()
    if total <= 0:
        n = len(vec)
        return np.full(n, 1.0 / n)
    return vec / total


def _uniform_like(vec: np.ndarray) -> np.ndarray:
    n = len(vec)
    return np.full(n, 1.0 / n)


def _shannon_entropy(vec: np.ndarray) -> float:
    vec = vec[vec > 0]
    return -float(np.sum(vec * np.log(vec)))


def _clamp(v: float, a: float, b: float) -> float:
    return max(a, min(b, v))


def _has_any(tags: Tuple[str, ...], whitelist: Tuple[str, ...]) -> bool:
    return any(t in whitelist for t in tags)


def _fragment_along_links(
    vec: np.ndarray,
    symbols: List[SymbolId],
    links: Dict[SymbolId, Dict[SymbolId, float]],
    alpha: float,
    cap: float,
) -> np.ndarray:
    if alpha <= 0:
        return vec
    vec = vec.copy()
    for i, sym in enumerate(symbols):
        neigh = links.get(sym)
        if not neigh:
            continue
        weight = vec[i]
        move = min(alpha * weight, cap)
        if move <= 0 or weight <= 0:
            continue
        vec[i] -= move
        total = sum(neigh.values())
        if total <= 0:
            vec[i] += move
            continue
        for j_sym, w in neigh.items():
            if j_sym not in symbols:
                continue
            j = symbols.index(j_sym)
            vec[j] += move * (w / total)
    return vec


def _refocus(
    weights: Dict[SymbolId, float],
    focus: List[SymbolId],
    boost: float,
) -> Dict[SymbolId, float]:
    if not focus or boost <= 0:
        return weights
    symbols = list(weights.keys())
    vec = _vec(weights, symbols)
    for i, sym in enumerate(symbols):
        if sym in focus:
            vec[i] *= (1.0 + boost)
    vec = _normalize(vec)
    return _to_dict(vec, symbols)


def _topk_symbols_from_emotion(weights: Dict[SymbolId, float], k: int) -> List[SymbolId]:
    return sorted(weights, key=weights.get, reverse=True)[:k]


@dataclass
class DriftConfig:
    drift_rate: float = 0.002
    fuzz_sigma: float = 0.03
    rng_seed: Optional[int] = None
    max_fragmentation: float = 0.25
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
    emotion_vector: np.ndarray
    fuzz_level: float
    entropy_score: float
    tags_active: Tuple[str, ...] = ()


class SoulDriftTransformer:
    def __init__(self, cfg: DriftConfig, init_state: DriftState):
        self.cfg = cfg
        self.state = init_state
        self.rng = np.random.default_rng(cfg.rng_seed)
        self._resolve_decay_counter = 0
        self._focus_symbols: List[SymbolId] = []
        self._history: List[DriftState] = []
        self._last_trigger: Optional[str] = None
        self._last_telemetry: Dict[str, object] = {}
        self.log_dir = Path(cfg.log_dir or ".")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / "drift_log.ndjson"

    # internal ---------------------------------------------------------------
    def _append_history(self, state: DriftState) -> None:
        snap = replace(state)
        self._history.append(snap)
        if len(self._history) > self.cfg.max_history:
            self._history.pop(0)
        if self.cfg.log_history:
            top = sorted(state.symbol_weights.items(), key=lambda kv: kv[1], reverse=True)[:3]
            entry = {
                "step": state.step,
                "entropy": state.entropy_score,
                "fuzz": state.fuzz_level,
                "top_symbols": top,
                "trigger": self._last_trigger,
            }
            with self.log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")
        self._last_trigger = None

    # public ----------------------------------------------------------------
    def step(self, silence: bool = True) -> DriftState:
        s = self.state
        symbols = list(s.symbol_weights.keys())
        vec = _vec(s.symbol_weights, symbols)
        prev_entropy = s.entropy_score

        noise = self.rng.normal(0, self.cfg.fuzz_sigma, size=len(vec))
        if self._resolve_decay_counter > 0:
            noise *= 0.5
        vec = vec + noise * self.cfg.drift_rate

        vec = (1 - self.cfg.decay_to_ambiguity) * vec + self.cfg.decay_to_ambiguity * _uniform_like(vec)

        alpha = s.fuzz_level * self.cfg.drift_rate
        vec = _fragment_along_links(vec, symbols, s.symbol_links, alpha, self.cfg.max_fragmentation)

        if self._resolve_decay_counter > 0 and self._focus_symbols:
            boost = (self._resolve_decay_counter / self.cfg.resolve_half_life_steps) * self.cfg.resolve_boost
            vec = _vec(_refocus(_to_dict(vec, symbols), self._focus_symbols, boost), symbols)

        if silence and _has_any(s.tags_active, self.cfg.dream_tags_whitelist):
            pass  # placeholder for emotion bias

        vec = np.clip(vec, 0.0, None)
        vec = _normalize(vec)
        s.symbol_weights = _to_dict(vec, symbols)
        s.entropy_score = _shannon_entropy(vec)
        if silence:
            s.fuzz_level = _clamp(s.fuzz_level + 0.01, 0.0, 1.0)
        else:
            s.fuzz_level = _clamp(s.fuzz_level - 0.02, 0.0, 1.0)
        if self._resolve_decay_counter > 0:
            s.fuzz_level = _clamp(s.fuzz_level - 0.02, 0.0, 1.0)
            self._resolve_decay_counter -= 1
        s.step += 1
        if self.cfg.log_history:
            self._append_history(s)

        try:
            serialized = json.dumps(sorted(s.symbol_weights.items()), separators=(",", ":"))
            delta_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        except Exception:
            delta_hash = "unknown"
        entropy_bump = round(s.entropy_score - prev_entropy, 4)
        self._last_telemetry = {
            "intent": "creative_entropy",
            "entropy_bump": entropy_bump,
            "fuzz_level": round(s.fuzz_level, 4),
            "delta_graph_hash": delta_hash,
            "step": s.step,
        }
        return s

    def run_session(self, steps: int, silence: bool = True) -> DriftState:
        start_vec = self.snapshot()
        start_entropy = start_vec.entropy_score
        start_weights = start_vec.symbol_weights.copy()
        for _ in range(steps):
            self.step(silence=silence)
        end_state = self.state
        if self.cfg.log_history:
            end_weights = end_state.symbol_weights
            diff = {k: end_weights.get(k, 0) - start_weights.get(k, 0) for k in start_weights}
            risen = sorted(diff.items(), key=lambda kv: kv[1], reverse=True)[:3]
            fallen = sorted(diff.items(), key=lambda kv: kv[1])[:3]
            summary = {
                "start_entropy": start_entropy,
                "end_entropy": end_state.entropy_score,
                "steps": steps,
                "symbols_risen": risen,
                "symbols_fallen": fallen,
            }
            path = self.log_dir / "session_summary.json"
            with path.open("w", encoding="utf-8") as fh:
                json.dump(summary, fh)
        return end_state

    def inject_trigger(self, emotion_delta: np.ndarray, tag: str = "trigger") -> DriftState:
        s = self.state
        s.emotion_vector = s.emotion_vector + emotion_delta
        focus = _topk_symbols_from_emotion(s.symbol_weights, k=12)
        s.symbol_weights = _refocus(s.symbol_weights, focus, self.cfg.resolve_boost)
        s.fuzz_level = _clamp(s.fuzz_level - 0.4, 0.0, 1.0)
        self._resolve_decay_counter = self.cfg.resolve_half_life_steps
        self._focus_symbols = focus
        s.tags_active += (tag,)
        self._last_trigger = tag
        s.entropy_score = _shannon_entropy(_vec(s.symbol_weights, list(s.symbol_weights.keys())))
        if self.cfg.log_history:
            self._append_history(s)
        return s

    def snapshot(self) -> DriftState:
        return replace(self.state)

    def intent_telemetry(self) -> Dict[str, object]:
        return dict(self._last_telemetry)
