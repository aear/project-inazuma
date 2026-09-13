"""A small conventional decoder Transformer for controlled comparisons.

This module deliberately contains the model, not a training pipeline or a live
cognition integration.  The default parameters are deterministic random
weights: useful for tests and resource measurements, but not evidence of a
learned capability.  Evaluators may inject a complete state dictionary.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class ConventionalTransformerConfig:
    vocab_size: int = 257  # BOS plus every byte value
    model_width: int = 32
    heads: int = 4
    layers: int = 2
    feed_forward_width: int = 64
    max_sequence: int = 128
    layer_norm_epsilon: float = 1e-5

    def __post_init__(self) -> None:
        for name in ("vocab_size", "model_width", "heads", "layers",
                     "feed_forward_width", "max_sequence"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.model_width % self.heads:
            raise ValueError("model_width must be divisible by heads")
        if self.vocab_size < 257:
            raise ValueError("the byte tokenizer requires at least 257 tokens")
        if self.layer_norm_epsilon <= 0:
            raise ValueError("layer_norm_epsilon must be positive")


def _matrix(rng: random.Random, rows: int, columns: int, scale: float) -> list[list[float]]:
    return [[rng.gauss(0.0, scale) for _ in range(columns)] for _ in range(rows)]


def _linear(vector: Sequence[float], weight: Sequence[Sequence[float]]) -> list[float]:
    return [sum(value * row[index] for index, value in enumerate(vector)) for row in weight]


def _layer_norm(vector: Sequence[float], epsilon: float) -> list[float]:
    mean = sum(vector) / len(vector)
    variance = sum((value - mean) ** 2 for value in vector) / len(vector)
    scale = 1.0 / math.sqrt(variance + epsilon)
    return [(value - mean) * scale for value in vector]


def _softmax(values: Sequence[float]) -> list[float]:
    peak = max(values)
    exponentials = [math.exp(value - peak) for value in values]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def _gelu(value: float) -> float:
    return 0.5 * value * (1.0 + math.tanh(
        math.sqrt(2.0 / math.pi) * (value + 0.044715 * value ** 3)
    ))


class ConventionalTransformer:
    """Pre-normalized, causal multi-head self-attention decoder.

    It implements the cognitive benchmark ``ChoiceScorer`` protocol directly.
    No method writes memory or registers the model with Ina's live council.
    """

    module_version = "V1"
    deployment_status = "benchmark_only"
    trained = False

    def __init__(
        self,
        config: ConventionalTransformerConfig | None = None,
        *,
        seed: int = 0x1A,
        state: Mapping[str, Any] | None = None,
        name: str = "ina-conventional-transformer-untrained",
    ) -> None:
        self.config = config or ConventionalTransformerConfig()
        self.name = str(name)
        self._seed = int(seed)
        self._initialize()
        if state is not None:
            self.load_state_dict(state)

    def _initialize(self) -> None:
        cfg = self.config
        rng = random.Random(self._seed)
        scale = 1.0 / math.sqrt(cfg.model_width)
        self.token_embeddings = _matrix(rng, cfg.vocab_size, cfg.model_width, scale)
        self.position_embeddings = _matrix(rng, cfg.max_sequence, cfg.model_width, scale)
        self.blocks = []
        for _ in range(cfg.layers):
            self.blocks.append({
                "query": _matrix(rng, cfg.model_width, cfg.model_width, scale),
                "key": _matrix(rng, cfg.model_width, cfg.model_width, scale),
                "value": _matrix(rng, cfg.model_width, cfg.model_width, scale),
                "attention_output": _matrix(rng, cfg.model_width, cfg.model_width, scale),
                "feed_forward_in": _matrix(rng, cfg.feed_forward_width, cfg.model_width, scale),
                "feed_forward_out": _matrix(
                    rng, cfg.model_width, cfg.feed_forward_width,
                    1.0 / math.sqrt(cfg.feed_forward_width),
                ),
            })
        self.output_projection = _matrix(rng, cfg.vocab_size, cfg.model_width, scale)

    @staticmethod
    def encode_text(text: str) -> list[int]:
        return [0, *(value + 1 for value in str(text).encode("utf-8"))]

    def _attention(
        self, states: list[list[float]], block: Mapping[str, Any]
    ) -> tuple[list[list[float]], list[list[list[float]]]]:
        cfg = self.config
        width = cfg.model_width // cfg.heads
        queries = [_linear(row, block["query"]) for row in states]
        keys = [_linear(row, block["key"]) for row in states]
        values = [_linear(row, block["value"]) for row in states]
        outputs = [[0.0] * cfg.model_width for _ in states]
        trace: list[list[list[float]]] = [[] for _ in range(cfg.heads)]
        divisor = math.sqrt(width)
        for position in range(len(states)):
            for head in range(cfg.heads):
                start = head * width
                stop = start + width
                scores = [
                    sum(queries[position][i] * keys[source][i] for i in range(start, stop))
                    / divisor
                    for source in range(position + 1)
                ]
                weights = _softmax(scores)
                trace[head].append(weights)
                for offset, index in enumerate(range(start, stop)):
                    outputs[position][index] = sum(
                        weights[source] * values[source][index]
                        for source in range(position + 1)
                    )
        return ([_linear(row, block["attention_output"]) for row in outputs], trace)

    def forward(self, token_ids: Sequence[int], *, return_attention: bool = False) -> dict[str, Any]:
        ids = [int(value) for value in token_ids]
        if not ids:
            raise ValueError("token_ids must not be empty")
        if len(ids) > self.config.max_sequence:
            raise ValueError("token sequence exceeds max_sequence")
        if any(value < 0 or value >= self.config.vocab_size for value in ids):
            raise ValueError("token id outside vocabulary")
        states = [
            [token + position for token, position in zip(
                self.token_embeddings[token_id], self.position_embeddings[index]
            )]
            for index, token_id in enumerate(ids)
        ]
        attention_trace = []
        for block in self.blocks:
            normalized = [_layer_norm(row, self.config.layer_norm_epsilon) for row in states]
            attended, trace = self._attention(normalized, block)
            states = [[left + right for left, right in zip(row, update)]
                      for row, update in zip(states, attended)]
            normalized = [_layer_norm(row, self.config.layer_norm_epsilon) for row in states]
            hidden = [[_gelu(value) for value in _linear(row, block["feed_forward_in"])]
                      for row in normalized]
            projected = [_linear(row, block["feed_forward_out"]) for row in hidden]
            states = [[left + right for left, right in zip(row, update)]
                      for row, update in zip(states, projected)]
            if return_attention:
                attention_trace.append(trace)
        normalized = [_layer_norm(row, self.config.layer_norm_epsilon) for row in states]
        result: dict[str, Any] = {
            "logits": [_linear(row, self.output_projection) for row in normalized],
            "hidden_states": normalized,
            "model_status": self.deployment_status,
            "trained": self.trained,
        }
        if return_attention:
            result["attention"] = attention_trace
        return result

    def score_choices(self, prompt: str, choices: Sequence[str]) -> list[float]:
        """Return mean continuation log-likelihood under the current weights."""
        scores = []
        prompt_bytes = str(prompt).encode("utf-8")
        for choice in choices:
            choice_bytes = str(choice).encode("utf-8")
            if not choice_bytes:
                scores.append(float("-inf"))
                continue
            combined = prompt_bytes + choice_bytes
            keep = self.config.max_sequence - 1
            dropped = max(0, len(combined) - keep)
            retained = combined[dropped:]
            choice_start = max(0, len(prompt_bytes) - dropped)
            if choice_start >= len(retained):
                raise ValueError("choice is outside the configured sequence window")
            ids = [0, *(value + 1 for value in retained)]
            logits = self.forward(ids)["logits"]
            log_probabilities = []
            for byte_index in range(choice_start, len(retained)):
                row = logits[byte_index]
                target = retained[byte_index] + 1
                peak = max(row)
                log_denominator = peak + math.log(sum(math.exp(value - peak) for value in row))
                log_probabilities.append(row[target] - log_denominator)
            scores.append(sum(log_probabilities) / len(log_probabilities))
        return scores

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": "ina.conventional_transformer/V1",
            "config": asdict(self.config),
            "trained": bool(self.trained),
            "token_embeddings": self.token_embeddings,
            "position_embeddings": self.position_embeddings,
            "blocks": self.blocks,
            "output_projection": self.output_projection,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("schema") != "ina.conventional_transformer/V1":
            raise ValueError("unsupported conventional Transformer state schema")
        if dict(state.get("config") or {}) != asdict(self.config):
            raise ValueError("state configuration does not match model configuration")
        # Validate exact tensor shapes before replacing any parameter.
        matrices = [
            (state.get("token_embeddings"), self.config.vocab_size, self.config.model_width),
            (state.get("position_embeddings"), self.config.max_sequence, self.config.model_width),
            (state.get("output_projection"), self.config.vocab_size, self.config.model_width),
        ]
        blocks = state.get("blocks")
        if not isinstance(blocks, list) or len(blocks) != self.config.layers:
            raise ValueError("state has the wrong number of Transformer blocks")
        expected = {
            "query": (self.config.model_width, self.config.model_width),
            "key": (self.config.model_width, self.config.model_width),
            "value": (self.config.model_width, self.config.model_width),
            "attention_output": (self.config.model_width, self.config.model_width),
            "feed_forward_in": (self.config.feed_forward_width, self.config.model_width),
            "feed_forward_out": (self.config.model_width, self.config.feed_forward_width),
        }
        for block in blocks:
            if not isinstance(block, Mapping):
                raise ValueError("invalid Transformer block")
            matrices.extend((block.get(name), *shape) for name, shape in expected.items())
        for matrix, rows, columns in matrices:
            if (not isinstance(matrix, list) or len(matrix) != rows
                    or any(not isinstance(row, list) or len(row) != columns for row in matrix)):
                raise ValueError("state tensor shape does not match configuration")
        self.token_embeddings = state["token_embeddings"]
        self.position_embeddings = state["position_embeddings"]
        self.blocks = blocks
        self.output_projection = state["output_projection"]
        self.trained = bool(state.get("trained", False))

    @classmethod
    def from_json(cls, path: str | Path, *, name: str | None = None) -> "ConventionalTransformer":
        state = json.loads(Path(path).read_text(encoding="utf-8"))
        config = ConventionalTransformerConfig(**state["config"])
        return cls(config, state=state, name=name or "ina-conventional-transformer")

    def promotion_evidence(self) -> dict[str, Any]:
        """Describe status without making or automating a council decision."""
        return {
            "module": type(self).__name__,
            "module_version": self.module_version,
            "deployment_status": self.deployment_status,
            "trained_weights": bool(self.trained),
            "council_member": False,
            "promotion_state": "not_evaluated",
            "required_evidence": (
                "repeated held-out capability advantage",
                "independent corroborating benchmark signals",
                "bounded resource and interference measurements",
                "explicit human review",
            ),
        }
