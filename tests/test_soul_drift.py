import numpy as np
import pytest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ina.transformers.soul_drift import DriftConfig, DriftState, SoulDriftTransformer


def _basic_state(num_symbols=3, non_uniform=False):
    if non_uniform:
        total = sum(range(1, num_symbols + 1))
        symbols = {f"s{i}": (i + 1) / total for i in range(num_symbols)}
    else:
        symbols = {f"s{i}": 1.0 / num_symbols for i in range(num_symbols)}
    links = {k: {j: 1.0 for j in symbols if j != k} for k in symbols}
    emotion = np.zeros(2)
    entropy = -np.sum([w * np.log(w) for w in symbols.values()])
    return DriftState(
        step=0,
        symbol_weights=symbols,
        symbol_links=links,
        emotion_vector=emotion,
        fuzz_level=0.0,
        entropy_score=float(entropy),
        tags_active=("dreamstate",),
    )


def test_entropy_increases_during_silence(tmp_path):
    cfg = DriftConfig(rng_seed=0, log_history=False)
    state = _basic_state(non_uniform=True)
    transformer = SoulDriftTransformer(cfg, state)
    start_entropy = state.entropy_score
    transformer.run_session(100, silence=True)
    assert transformer.state.entropy_score > start_entropy


def test_trigger_resolves_weights_and_fuzz(tmp_path):
    cfg = DriftConfig(rng_seed=1, log_history=False)
    state = _basic_state(num_symbols=15)
    state.fuzz_level = 0.5
    transformer = SoulDriftTransformer(cfg, state)
    prev_fuzz = transformer.state.fuzz_level
    prev_weights = transformer.state.symbol_weights.copy()
    focus = sorted(prev_weights, key=prev_weights.get, reverse=True)[:12]
    prev_sum = sum(prev_weights[s] for s in focus)
    transformer.inject_trigger(np.array([1.0, 0.0]))
    new_sum = sum(transformer.state.symbol_weights[s] for s in focus)
    assert new_sum > prev_sum
    assert transformer.state.fuzz_level < prev_fuzz


def test_fragmentation_capped(tmp_path):
    cfg = DriftConfig(
        drift_rate=1.0,
        fuzz_sigma=0.0,
        decay_to_ambiguity=0.0,
        max_fragmentation=0.25,
        rng_seed=2,
        log_history=False,
    )
    symbols = {"A": 0.8, "B": 0.2}
    links = {"A": {"B": 1.0}, "B": {"A": 1.0}}
    emotion = np.zeros(1)
    entropy = -np.sum([w * np.log(w) for w in symbols.values()])
    state = DriftState(
        step=0,
        symbol_weights=symbols,
        symbol_links=links,
        emotion_vector=emotion,
        fuzz_level=1.0,
        entropy_score=float(entropy),
        tags_active=("dreamstate",),
    )
    transformer = SoulDriftTransformer(cfg, state)
    before = state.symbol_weights.copy()
    transformer.step()
    after = transformer.state.symbol_weights
    for k in before:
        assert before[k] - after[k] <= cfg.max_fragmentation + 1e-6


def test_history_bounds(tmp_path):
    cfg = DriftConfig(rng_seed=0, log_history=True, max_history=10, log_dir=str(tmp_path))
    state = _basic_state()
    transformer = SoulDriftTransformer(cfg, state)
    transformer.run_session(20)
    assert len(transformer._history) <= cfg.max_history
