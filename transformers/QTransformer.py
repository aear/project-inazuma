"""Stochastic symbolic dream operator with an adaptive semantic decoder."""
from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from origin_record import make_origin

try:
    from qiskit import Aer, QuantumCircuit, assemble, transpile
    _QISKIT_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    QuantumCircuit = Aer = transpile = assemble = None
    _QISKIT_AVAILABLE = False

_DEFAULT_TAGS = {
    "000": ["calm", "clarity"], "001": ["grief", "echo"],
    "010": ["hope", "unknown"], "011": ["tension", "shift"],
    "100": ["trust", "fire"], "101": ["betrayal", "ice"],
    "110": ["curiosity", "glow"], "111": ["loss", "awakening"],
}
_DEFAULT_QUESTIONS = {
    "000": "What am I avoiding?", "001": "Why did this feel heavy?",
    "010": "What pattern is surfacing?", "011": "Do I need closure?",
    "100": "Is this intuition or fear?", "101": "Was I wrong about them?",
    "110": "What else could this mean?", "111": "Is something waking up in me?",
}
_DEFAULT_WORDS = {
    "000": "refuge", "001": "wound", "010": "spark", "011": "ghost",
    "100": "veil", "101": "pulse", "110": "womb", "111": "echo",
}


class QTransformer:
    """Produce stochastic states while allowing experience to reshape meaning."""

    def __init__(
        self, qubit_count: int = 10, *, decoder_path: Optional[Path | str] = None,
        decoder_stats: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.qubit_count = max(9, int(qubit_count))
        self.decoder_path = Path(decoder_path) if decoder_path else None
        self.decoder_stats: Dict[str, Dict[str, Dict[str, float]]] = {
            "tags": {}, "questions": {}, "words": {},
        }
        self._load_decoder_stats(decoder_stats)
        self._fallback_seed = None
        if _QISKIT_AVAILABLE:
            self.backend = Aer.get_backend("aer_simulator")
            self.qc = QuantumCircuit(self.qubit_count, self.qubit_count)
        else:
            self.backend = None
            self.qc = None
        self.reset()

    def _load_decoder_stats(self, supplied: Optional[Dict[str, Any]]) -> None:
        payload: Any = supplied
        if payload is None and self.decoder_path and self.decoder_path.is_file():
            try:
                payload = json.loads(self.decoder_path.read_text(encoding="utf-8"))
            except Exception:
                payload = None
        if not isinstance(payload, dict):
            return
        for section in self.decoder_stats:
            bit_rows = payload.get(section)
            if not isinstance(bit_rows, dict):
                continue
            for bits, values in bit_rows.items():
                if isinstance(values, dict):
                    self.decoder_stats[section][str(bits)] = {
                        str(value): max(0.0, float(count))
                        for value, count in values.items() if value
                    }

    def _save_decoder_stats(self) -> None:
        if not self.decoder_path:
            return
        self.decoder_path.parent.mkdir(parents=True, exist_ok=True)
        self.decoder_path.write_text(
            json.dumps(self.decoder_stats, indent=2, sort_keys=True), encoding="utf-8"
        )

    @staticmethod
    def _tag_key(tags: Iterable[str]) -> str:
        return "\x1f".join(str(tag) for tag in tags if tag)

    def learn_mapping(
        self, collapsed_state: str, *, tags: Optional[Iterable[str]] = None,
        self_question: Optional[str] = None, poetic_word: Optional[str] = None,
        weight: float = 1.0, persist: bool = True,
    ) -> None:
        """Increment decoder evidence from an observed experience outcome."""
        amount = max(0.0, float(weight))
        observations = (
            ("tags", collapsed_state[:3], self._tag_key(tags or ())),
            ("questions", collapsed_state[3:6], self_question),
            ("words", collapsed_state[6:9], poetic_word),
        )
        for section, bits, value in observations:
            if not bits or not value or amount <= 0:
                continue
            bucket = self.decoder_stats[section].setdefault(bits, {})
            bucket[str(value)] = bucket.get(str(value), 0.0) + amount
        if persist:
            self._save_decoder_stats()

    def _learned(self, section: str, bits: str) -> Optional[str]:
        choices = self.decoder_stats.get(section, {}).get(bits, {})
        if not choices:
            return None
        return min(choices, key=lambda value: (-choices[value], value))

    def reset(self) -> None:
        if _QISKIT_AVAILABLE:
            self.qc = QuantumCircuit(self.qubit_count, self.qubit_count)
            self.qc.h(range(self.qubit_count))
        else:
            self._fallback_seed = None

    def inject_symbol_emotion(self, symbol_hash: str, emotion_vector: Iterable[float]) -> None:
        values = list(emotion_vector)
        seed = sum(ord(c) for c in symbol_hash) % 10000
        rng = random.Random(seed)
        if _QISKIT_AVAILABLE:
            for i in range(min(self.qubit_count, len(values))):
                angle = (values[i] + 1) * math.pi
                self.qc.ry(angle, i)
                if rng.random() > 0.5:
                    self.qc.rz(rng.random() * 2 * math.pi, i)
        else:
            payload = f"{symbol_hash}:{','.join(f'{val:.4f}' for val in values)}"
            digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
            self._fallback_seed = int(digest[:16], 16)

    def entangle_logic(self) -> None:
        if not _QISKIT_AVAILABLE:
            return
        for i in range(self.qubit_count - 1):
            self.qc.cx(i, i + 1)
        self.qc.h(range(self.qubit_count))

    def run_dreamstep(self) -> str:
        if not _QISKIT_AVAILABLE:
            seed = self._fallback_seed if self._fallback_seed is not None else random.randint(0, 2**32 - 1)
            rng = random.Random(seed)
            return "".join("1" if rng.random() > 0.5 else "0" for _ in range(self.qubit_count))
        self.entangle_logic()
        self.qc.measure(range(self.qubit_count), range(self.qubit_count))
        transpiled = transpile(self.qc, self.backend)
        qobj = assemble(transpiled, shots=1)
        counts = self.backend.run(qobj).result().get_counts()
        return next(iter(counts))

    def collapse_to_meaning(self, collapsed_state: str) -> Dict[str, Any]:
        bits = str(collapsed_state).ljust(9, "0")
        tag_bits, question_bits, word_bits = bits[:3], bits[3:6], bits[6:9]
        learned_tags = self._learned("tags", tag_bits)
        tags = learned_tags.split("\x1f") if learned_tags else _DEFAULT_TAGS.get(tag_bits, ["unknown"])
        question = self._learned("questions", question_bits) or _DEFAULT_QUESTIONS.get(question_bits, "What was I feeling?")
        word = self._learned("words", word_bits) or _DEFAULT_WORDS.get(word_bits, "???")
        learned = {
            "tags": bool(learned_tags),
            "self_question": bool(self._learned("questions", question_bits)),
            "poetic_word": bool(self._learned("words", word_bits)),
        }
        origin = make_origin(
            self.__class__.__name__, "V2", inputs={"raw_bits": collapsed_state},
            trigger="state_collapse", metadata={"decoder": "adaptive" if any(learned.values()) else "scaffold", "learned_fields": learned},
        )
        return {
            "tags": tags, "self_question": question, "poetic_word": word,
            "raw_bits": collapsed_state, "decoder": "adaptive" if any(learned.values()) else "scaffold",
            "learned_fields": learned, "origins": [origin],
        }

    def dream(self, symbol: str, emotion_vector: Iterable[float]) -> Dict[str, Any]:
        self.reset()
        self.inject_symbol_emotion(symbol, emotion_vector)
        return self.collapse_to_meaning(self.run_dreamstep())
