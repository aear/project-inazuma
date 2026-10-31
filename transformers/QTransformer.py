"""Stochastic symbolic dream operator with an adaptive semantic decoder."""
from __future__ import annotations

import hashlib
import json
import math
import random
import threading
from collections import OrderedDict
from itertools import islice
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

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
        self._learning_lock = threading.RLock()
        self._learning_events = OrderedDict()
        self.last_learning_update = {"status": "not_observed", "replay_scope": "process_local_512_events"}
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
        from io_utils import atomic_write_json
        self.decoder_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(self.decoder_path, self.decoder_stats, indent=2)

    @staticmethod
    def _tag_key(tags: Iterable[str]) -> str:
        return "\x1f".join(str(tag) for tag in tags if tag)

    def learn_mapping(
        self, collapsed_state: str, *, tags: Optional[Iterable[str]] = None,
        self_question: Optional[str] = None, poetic_word: Optional[str] = None,
        weight: float = 1.0, persist: bool = True, source_event: Optional[str] = None,
    ) -> None:
        """Admit one observed decoder example; event IDs optionally suppress replay.

        Replay identity is caller supplied and retained for 512 events in this
        object only. The existing on-disk statistics format is unchanged.
        """
        if not isinstance(collapsed_state, str) or not 9 <= len(collapsed_state) <= 256 or set(collapsed_state) - {"0", "1"}:
            raise ValueError("a 9..256 bit observation is required")
        amount = float(weight)
        if not math.isfinite(amount) or amount < 0:
            raise ValueError("learning weight must be finite and nonnegative")
        if source_event is not None and (not isinstance(source_event, str) or not 1 <= len(source_event.strip()) <= 160):
            raise ValueError("source_event must be a bounded nonblank reference")
        source_event = source_event.strip() if source_event is not None else None
        if isinstance(tags, str):raise ValueError("tags must be a collection of labels")
        tag_rows = list(islice(tags or (), 17))
        if len(tag_rows) > 16 or any(not isinstance(tag, str) or not 1 <= len(tag) <= 80 or "\x1f" in tag for tag in tag_rows):
            raise ValueError("select at most 16 bounded tag labels")
        for value, limit in ((self_question, 512), (poetic_word, 128)):
            if value is not None and (not isinstance(value, str) or not 1 <= len(value) <= limit):
                raise ValueError("bounded decoder labels required")
        observations = (("tags", collapsed_state[:3], self._tag_key(tag_rows)),
                        ("questions", collapsed_state[3:6], self_question),
                        ("words", collapsed_state[6:9], poetic_word))
        identity = (collapsed_state, observations, amount)
        with self._learning_lock:
            if source_event in self._learning_events:
                same = self._learning_events[source_event] == identity
                self.last_learning_update = {"status": "duplicate_event" if same else "conflicting_replay",
                    "source_event": source_event, "updated": False, "replay_scope": "process_local_512_events"}
                return
            staged = []
            for section, bits, value in observations:
                if not value or not amount:
                    continue
                bucket = self.decoder_stats[section].get(bits, {})
                if len(bucket) > 64 or (value not in bucket and len(bucket) >= 64):
                    self.last_learning_update = {"status": "capacity_reached", "updated": False}
                    return
                count = float(bucket.get(value, 0.0)) + amount
                if not math.isfinite(count):
                    raise ValueError("decoder evidence total must remain finite")
                staged.append((section, bits, value, count))
            if not staged:
                self.last_learning_update = {"status": "no_observation", "updated": False}
                return
            # Persistence failure restores this object's prior statistics too.
            before = [(section, bits, value, value in self.decoder_stats[section].get(bits, {}),
                       self.decoder_stats[section].get(bits, {}).get(value), bits in self.decoder_stats[section])
                      for section, bits, value, _ in staged]
            try:
                for section, bits, value, count in staged:
                    self.decoder_stats[section].setdefault(bits, {})[value] = count
                if persist:
                    self._save_decoder_stats()
            except Exception:
                for section, bits, value, present, previous, had_bucket in before:
                    bucket = self.decoder_stats[section].get(bits, {})
                    if present:bucket[value] = previous
                    else:bucket.pop(value, None)
                    if not had_bucket and not bucket:self.decoder_stats[section].pop(bits, None)
                raise
            if source_event is not None:
                self._learning_events[source_event] = identity
                while len(self._learning_events) > 512:
                    self._learning_events.popitem(last=False)
            self.last_learning_update = {"status": "updated", "updated": True,
                "source_event": source_event, "replay_scope": "process_local_512_events",
                "source_identity_authenticated": False}

    def _mapping_assessment(self, section: str, bits: str) -> Dict[str, Any]:
        with self._learning_lock:
            choices = self.decoder_stats.get(section, {}).get(bits, {})
            if len(choices) > 64:
                return {"status": "capacity_unresolved", "selected": None, "confidence_is_calibrated": False}
            entries = tuple(choices.items())
        ranked = []
        for value, count in entries:
            try:
                amount = float(count)
            except (TypeError, ValueError, OverflowError):
                continue
            if math.isfinite(amount) and amount > 0:
                ranked.append((amount, value))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        tied = len(ranked) > 1 and math.isclose(ranked[0][0], ranked[1][0], rel_tol=1e-9, abs_tol=0.0)
        return {"status": "ambiguous" if tied else "observed_preference" if ranked else "not_observed",
                "selected": ranked[0][1] if ranked and not tied else None,
                "candidates": [{"value": value, "weight": amount} for amount, value in ranked[:4]],
                "confidence_is_calibrated": False, "truth_certified": False}

    def _learned(self, section: str, bits: str) -> Optional[str]:
        return self._mapping_assessment(section, bits)["selected"]

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
        assessments = {section: self._mapping_assessment(section, part)
                       for section, part in (("tags", tag_bits), ("questions", question_bits), ("words", word_bits))}
        learned_tags = assessments["tags"]["selected"]
        unresolved = {section: item["status"] in {"ambiguous", "capacity_unresolved"}
                      for section, item in assessments.items()}
        tags = learned_tags.split("\x1f") if learned_tags else (["unknown"] if unresolved["tags"] else _DEFAULT_TAGS.get(tag_bits, ["unknown"]))
        question = assessments["questions"]["selected"] or ("What meaning fits this state?" if unresolved["questions"] else _DEFAULT_QUESTIONS.get(question_bits, "What was I feeling?"))
        word = assessments["words"]["selected"] or ("???" if unresolved["words"] else _DEFAULT_WORDS.get(word_bits, "???"))
        learned = {
            "tags": bool(learned_tags),
            "self_question": bool(assessments["questions"]["selected"]),
            "poetic_word": bool(assessments["words"]["selected"]),
        }
        origin = make_origin(
            self.__class__.__name__, "V3", inputs={"raw_bits": collapsed_state},
            trigger="state_collapse", metadata={"decoder": "adaptive" if any(learned.values()) else "scaffold", "learned_fields": learned},
        )
        return {
            "tags": tags, "self_question": question, "poetic_word": word,
            "raw_bits": collapsed_state, "decoder": "adaptive" if any(learned.values()) else "scaffold",
            "learned_fields": learned, "origins": [origin],
            "mapping_assessments": assessments,
        }

    def dream(self, symbol: str, emotion_vector: Iterable[float]) -> Dict[str, Any]:
        self.reset()
        self.inject_symbol_emotion(symbol, emotion_vector)
        return self.collapse_to_meaning(self.run_dreamstep())

    def collapse_candidates(
        self, candidates: Iterable[Mapping[str, Any]], *, context: str = "",
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Collapse a bounded representational superposition of viable choices.

        Callers remain responsible for grounding, capability, consent, and
        safety.  This operation is only an ambiguity resolver among candidates
        that have already passed those gates.  The returned trace deliberately
        exposes the choice distribution, not simulator/circuit internals.
        """
        states = []
        for raw in list(candidates)[:16]:
            identifier = str(raw.get("id") or "").strip()
            if not identifier or any(item["id"] == identifier for item in states):
                continue
            try:
                activation = max(0.0, float(raw.get("activation", 0.0)))
            except (TypeError, ValueError):
                activation = 0.0
            states.append({"id": identifier[:160], "activation": activation})
        total = sum(item["activation"] for item in states)
        if not states or total <= 0.0:
            raise ValueError("candidate collapse requires positive activation")
        distribution = [{
            "id": item["id"],
            "probability": round(item["activation"] / total, 9),
        } for item in states]
        if seed is None:
            entropy = f"{context}:{random.SystemRandom().getrandbits(128)}"
            resolved_seed = int(hashlib.sha256(entropy.encode("utf-8")).hexdigest()[:16], 16)
        else:
            resolved_seed = int(seed)
        point = random.Random(resolved_seed).random()
        cumulative = 0.0
        selected = distribution[-1]["id"]
        for item in distribution:
            cumulative += item["probability"]
            if point <= cumulative:
                selected = item["id"]
                break
        origin = make_origin(
            self.__class__.__name__, "V3",
            inputs={"candidate_ids": [item["id"] for item in distribution]},
            trigger="expression_ambiguity_collapse",
            metadata={"candidate_count": len(distribution), "representation": "weighted_superposition"},
        )
        return {
            "selected_id": selected, "distribution": distribution,
            "representation": "weighted_superposition", "origins": [origin],
        }
