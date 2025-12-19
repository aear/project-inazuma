from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from gui_hook import log_to_statusbox
from model_manager import get_inastate, load_config, update_inastate


@dataclass
class FragmentEntry:
    fragment: Dict[str, Any]
    path: Path
    timestamp: datetime
    shard: str


def _now_ts() -> float:
    return time.time()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_ts(raw: Optional[str], fallback: Optional[datetime] = None) -> datetime:
    if not raw:
        return fallback or datetime.now(timezone.utc)
    try:
        return datetime.fromisoformat(raw)
    except Exception:
        return fallback or datetime.now(timezone.utc)


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


class TraumaProcessor:
    """
    Windowed trauma processor.

    - Scans recent fragments for high-intensity signatures.
    - Builds a context → causality → meaning chain.
    - Applies a cooling plan only when energy/stress gates allow it.
    - Tracks per-fragment processing counts to avoid rumination.
    - Emits confidence-tagged summaries back into the fragment file
      and inastate for downstream awareness.
    """

    def __init__(
        self,
        context_window: int = 2,
        max_per_run: int = 3,
        intensity_threshold: float = 0.65,
        energy_floor: float = 0.35,
        stress_ceiling: float = 0.72,
    ) -> None:
        config = load_config()
        self.child = config.get("current_child", "Inazuma_Yagami")
        self.memory_dir = Path("AI_Children") / self.child / "memory"
        self.fragments_dir = self.memory_dir / "fragments"
        self.state_path = self.memory_dir / "trauma_processor_state.json"
        self.log_path = self.memory_dir / "trauma_processor_log.jsonl"
        self.context_window = max(1, context_window)
        self.max_per_run = max(1, max_per_run)
        self.intensity_threshold = abs(float(intensity_threshold))
        self.energy_floor = float(energy_floor)
        self.stress_ceiling = float(stress_ceiling)
        self.max_files_per_shard = 80
        self.shards = ["short", "working", "pending"]
        self.rumination_cooldown = 6 * 3600  # seconds
        self.rumination_cap = 4
        self.cooling_floor = 0.25
        self.cool_strength = 0.4

    # ------------------------------------------------------------------ #
    # State helpers
    # ------------------------------------------------------------------ #

    def _log(self, msg: str) -> None:
        log_to_statusbox(f"[Trauma] {msg}")

    def _load_state(self) -> Dict[str, Any]:
        if not self.state_path.exists():
            return {"processed": {}}
        try:
            with self.state_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if not isinstance(data, dict):
                return {"processed": {}}
            data.setdefault("processed", {})
            return data
        except Exception:
            return {"processed": {}}

    def _save_state(self, state: Dict[str, Any]) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with self.state_path.open("w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------ #
    # Gate checks
    # ------------------------------------------------------------------ #

    def _gate_ready(self) -> Tuple[bool, str, Dict[str, float]]:
        energy = float(get_inastate("current_energy") or 0.5)
        snapshot = get_inastate("emotion_snapshot") or {}
        values = snapshot.get("values") if isinstance(snapshot, dict) else snapshot
        if not isinstance(values, dict):
            values = {}
        stress = float(values.get("stress", 0.0) or 0.0)
        if energy < self.energy_floor:
            return False, f"energy {energy:.2f} below floor {self.energy_floor:.2f}", {"energy": energy, "stress": stress}
        if stress > self.stress_ceiling:
            return False, f"stress {stress:.2f} above ceiling {self.stress_ceiling:.2f}", {"energy": energy, "stress": stress}
        return True, "ready", {"energy": energy, "stress": stress}

    # ------------------------------------------------------------------ #
    # Fragment loading
    # ------------------------------------------------------------------ #

    def _load_fragment(self, path: Path, shard: str) -> Optional[FragmentEntry]:
        try:
            with path.open("r", encoding="utf-8") as handle:
                fragment = json.load(handle)
        except Exception as exc:
            self._log(f"Failed to read {path.name}: {exc}")
            return None
        raw_ts = fragment.get("timestamp")
        timestamp = _parse_ts(raw_ts, fallback=datetime.fromtimestamp(path.stat().st_mtime, timezone.utc))
        return FragmentEntry(fragment=fragment, path=path, timestamp=timestamp, shard=shard)

    def _gather_recent_fragments(self) -> List[FragmentEntry]:
        entries: List[FragmentEntry] = []
        for shard in self.shards:
            shard_dir = self.fragments_dir / shard
            if not shard_dir.exists():
                continue
            try:
                files = sorted(
                    shard_dir.glob("*.json"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )[: self.max_files_per_shard]
            except Exception:
                continue
            for path in files:
                entry = self._load_fragment(path, shard)
                if entry:
                    entries.append(entry)
        entries.sort(key=lambda item: item.timestamp)
        return entries

    # ------------------------------------------------------------------ #
    # Emotion helpers
    # ------------------------------------------------------------------ #

    def _fragment_id(self, fragment: Dict[str, Any]) -> Optional[str]:
        return fragment.get("fragment_id") or fragment.get("id")

    def _emotion_value(self, fragment: Dict[str, Any], key: str) -> Optional[float]:
        emotions = fragment.get("emotions")
        if not isinstance(emotions, dict):
            return None
        if key in emotions and isinstance(emotions.get(key), (int, float)):
            return float(emotions[key])
        sliders = emotions.get("sliders")
        if isinstance(sliders, dict) and key in sliders:
            try:
                return float(sliders[key])
            except Exception:
                return None
        summary = emotions.get("summary")
        if isinstance(summary, dict) and key in summary:
            try:
                return float(summary[key])
            except Exception:
                return None
        return None

    # ------------------------------------------------------------------ #
    # Context + causality
    # ------------------------------------------------------------------ #

    def _summarise_entry(self, entry: FragmentEntry) -> Dict[str, Any]:
        fragment = entry.fragment
        frag_id = self._fragment_id(fragment)
        return {
            "id": frag_id,
            "type": fragment.get("type"),
            "timestamp": entry.timestamp.isoformat(),
            "tags": fragment.get("tags") or [],
            "intensity": self._emotion_value(fragment, "intensity"),
            "stress": self._emotion_value(fragment, "stress"),
            "summary": fragment.get("summary"),
            "importance": fragment.get("importance"),
            "shard": entry.shard,
        }

    def _context_window(self, entries: List[FragmentEntry], index: int) -> Dict[str, List[Dict[str, Any]]]:
        before: List[Dict[str, Any]] = []
        after: List[Dict[str, Any]] = []
        for i in range(1, self.context_window + 1):
            if index - i >= 0:
                before.append(self._summarise_entry(entries[index - i]))
            if index + i < len(entries):
                after.append(self._summarise_entry(entries[index + i]))
        return {"before": before, "after": after}

    def _score_causal_link(
        self,
        target_tags: Sequence[str],
        target_intensity: float,
        context: Dict[str, Any],
    ) -> float:
        overlap = len(set(tag.lower() for tag in target_tags) & set(tag.lower() for tag in context.get("tags", [])))
        intensity = float(context.get("intensity") or 0.0)
        stress = float(context.get("stress") or 0.0)
        stability = 1.0 - min(1.0, abs(intensity - target_intensity))
        return (overlap * 0.25) + (max(0.0, stress) * 0.2) + (stability * 0.3)

    def _infer_causality(
        self,
        fragment: Dict[str, Any],
        context: Dict[str, List[Dict[str, Any]]],
        intensity: float,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        tags = fragment.get("tags") or []
        candidates = context["before"] + context["after"]
        if not candidates:
            return None, None
        scored = sorted(
            (
                (self._score_causal_link(tags, intensity, ctx), ctx)
                for ctx in candidates
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        best_score, best_ctx = scored[0]
        if best_score <= 0:
            return None, None
        direction = "preceded" if best_ctx in context["before"] else "followed"
        cause_text = (
            f"{best_ctx.get('type')} {direction} the event "
            f"with overlapping cues {best_ctx.get('tags')[:3]} "
            f"and intensity {best_ctx.get('intensity')!s}"
        )
        return best_ctx, cause_text

    def _meaning_note(
        self,
        fragment: Dict[str, Any],
        context: Dict[str, List[Dict[str, Any]]],
        cause_text: Optional[str],
    ) -> str:
        window_hint = ", ".join(
            filter(
                None,
                [
                    ctx.get("type")
                    for ctx in (context["before"][:1] + context["after"][:1])
                ],
            )
        )
        summary = fragment.get("summary") or ""
        lines = []
        if summary:
            lines.append(f"Core memory: {summary[:140]}")
        if window_hint:
            lines.append(f"Window references: {window_hint}")
        if cause_text:
            lines.append(f"Likely driver: {cause_text}")
        lines.append("Meaning integration: treat this pattern as information, not command; redirect to current safety.")
        return " | ".join(lines)

    # ------------------------------------------------------------------ #
    # Rumination tracking
    # ------------------------------------------------------------------ #

    def _can_process_fragment(self, frag_id: Optional[str], state: Dict[str, Any]) -> bool:
        if not frag_id:
            return False
        meta = state.get("processed", {}).get(frag_id)
        if not isinstance(meta, dict):
            return True
        count = int(meta.get("count", 0))
        last_ts = float(meta.get("last_ts") or 0.0)
        if count >= self.rumination_cap and (_now_ts() - last_ts) < self.rumination_cooldown:
            return False
        if (_now_ts() - last_ts) < self.rumination_cooldown / 4:
            return False
        return True

    def _mark_processed(self, frag_id: str, state: Dict[str, Any]) -> None:
        processed = state.setdefault("processed", {})
        meta = processed.setdefault(frag_id, {})
        meta["count"] = int(meta.get("count", 0)) + 1
        meta["last_ts"] = _now_ts()

    # ------------------------------------------------------------------ #
    # Cooling + persistence
    # ------------------------------------------------------------------ #

    def _apply_cooling(
        self,
        entry: FragmentEntry,
        intensity: float,
        context: Dict[str, List[Dict[str, Any]]],
        cause_text: Optional[str],
        meaning_text: str,
        confidence: float,
    ) -> Dict[str, Any]:
        original = intensity
        confidence = _clamp(confidence, 0.1, 0.99)
        delta = self.cool_strength * confidence
        target_abs = max(self.cooling_floor, max(0.0, abs(original) - delta))
        cooled_intensity = math.copysign(target_abs, original)
        cooldown = abs(original) - abs(cooled_intensity)

        trauma_record = {
            "processed_at": _now_iso(),
            "confidence": round(confidence, 3),
            "original_intensity": round(original, 4),
            "cooled_intensity": round(cooled_intensity, 4),
            "cooling_delta": round(cooldown, 4),
            "context_window": context,
            "cause_hypothesis": cause_text,
            "meaning_note": meaning_text,
        }

        fragment = entry.fragment
        processing = fragment.setdefault("processing", {})
        trauma_history = processing.setdefault("trauma", [])
        trauma_history.append(trauma_record)
        if len(trauma_history) > 5:
            del trauma_history[:-5]

        emotions = fragment.setdefault("emotions", {})
        summary = emotions.get("summary")
        if not isinstance(summary, dict):
            summary = {}
            emotions["summary"] = summary
        summary["cooled_intensity"] = trauma_record["cooled_intensity"]
        summary["cooling_confidence"] = trauma_record["confidence"]

        entry.path.parent.mkdir(parents=True, exist_ok=True)
        with entry.path.open("w", encoding="utf-8") as handle:
            json.dump(fragment, handle, indent=2, ensure_ascii=False)

        log_entry = {
            "child": self.child,
            "fragment_id": self._fragment_id(fragment),
            "path": str(entry.path),
            "timestamp": trauma_record["processed_at"],
            "cooling_delta": trauma_record["cooling_delta"],
            "confidence": trauma_record["confidence"],
        }
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        return trauma_record

    # ------------------------------------------------------------------ #
    # Public entrypoint
    # ------------------------------------------------------------------ #

    def run(self) -> None:
        ready, note, gate_state = self._gate_ready()
        if not ready:
            self._log(f"Skipping trauma processing ({note}).")
            update_inastate(
                "trauma_processor_last",
                {
                    "timestamp": _now_iso(),
                    "status": "skipped",
                    "reason": note,
                    "energy": gate_state.get("energy"),
                    "stress": gate_state.get("stress"),
                },
            )
            return

        state = self._load_state()
        entries = self._gather_recent_fragments()
        if not entries:
            self._log("No fragments available for trauma processing.")
            return

        processed_payloads: List[Dict[str, Any]] = []
        total_processed = 0

        for idx, entry in enumerate(entries):
            if total_processed >= self.max_per_run:
                break

            fragment = entry.fragment
            frag_id = self._fragment_id(fragment)
            if not self._can_process_fragment(frag_id, state):
                continue

            intensity = self._emotion_value(fragment, "intensity")
            if intensity is None or abs(intensity) < self.intensity_threshold:
                continue

            context = self._context_window(entries, idx)
            cause_ctx, cause_text = self._infer_causality(fragment, context, intensity)
            meaning_text = self._meaning_note(fragment, context, cause_text)
            context_summary = {
                "before": context["before"],
                "after": context["after"],
            }

            coverage = len(context["before"]) + len(context["after"])
            overlap_bonus = 0.0
            if cause_ctx:
                overlap = len(set(fragment.get("tags") or []) & set(cause_ctx.get("tags") or []))
                overlap_bonus = min(0.25, overlap * 0.1)
            confidence = 0.5 + (0.08 * coverage) + overlap_bonus
            confidence -= max(0.0, gate_state.get("stress", 0.0)) * 0.1
            confidence = _clamp(confidence, 0.25, 0.95)

            trauma_record = self._apply_cooling(
                entry=entry,
                intensity=intensity,
                context=context_summary,
                cause_text=cause_text,
                meaning_text=meaning_text,
                confidence=confidence,
            )

            self._mark_processed(frag_id, state)
            total_processed += 1
            processed_payloads.append(
                {
                    "fragment_id": frag_id,
                    "confidence": trauma_record["confidence"],
                    "cooling_delta": trauma_record["cooling_delta"],
                    "context_size": coverage,
                    "cause": cause_text,
                }
            )
            self._log(
                f"Cooled {frag_id} (Δ{trauma_record['cooling_delta']:.2f}, "
                f"conf {trauma_record['confidence']:.2f})"
            )

        state["last_run"] = _now_iso()
        self._save_state(state)

        update_inastate(
            "trauma_processor_last",
            {
                "timestamp": _now_iso(),
                "status": "processed" if processed_payloads else "idle",
                "energy": gate_state.get("energy"),
                "stress": gate_state.get("stress"),
                "processed": processed_payloads,
            },
        )

        if not processed_payloads:
            self._log("No qualifying fragments exceeded the intensity threshold.")


def run():
    processor = TraumaProcessor()
    processor.run()


if __name__ == "__main__":
    run()
