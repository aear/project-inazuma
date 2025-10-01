"""Automatic gating logic for Ina's fragment memory tiers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional

from gui_hook import log_to_statusbox
from memory_graph import MEMORY_TIERS, MemoryManager


class MemoryGatekeeper:
    """Review and (re)assign memory fragments across tiered storage."""

    PINNED_TAGS = {"core_memory", "identity", "self_identity", "trauma"}

    def __init__(self, manager: Optional[MemoryManager] = None):
        self.manager = manager or MemoryManager()
        self._tier_index = {tier: idx for idx, tier in enumerate(MEMORY_TIERS)}

    # === Public API ===
    def run(self) -> None:
        """Execute the gating cycle: ingest, score, and move fragments."""

        self.manager.ensure_tier_directories()
        self.manager.prune_missing()

        ingested = self._ingest_unassigned()
        if ingested:
            log_to_statusbox(
                f"[MemoryGate] Routed {ingested} new fragments into tier storage."
            )

        # Refresh map with any new metadata before rebalancing
        self.manager.reindex(new_only=True)

        moved = self._rebalance_existing()
        if moved:
            log_to_statusbox(
                f"[MemoryGate] Shifted {moved} fragments between memory tiers."
            )
        elif ingested == 0:
            log_to_statusbox("[MemoryGate] No gating changes required.")

    # === Internal helpers ===
    def _ingest_unassigned(self) -> int:
        count = 0
        base = self.manager.base_path
        for fragment_path in sorted(base.glob("frag_*.json")):
            target = self._initial_tier(fragment_path)
            if self.manager.ingest_fragment_file(fragment_path, target):
                count += 1
        return count

    def _initial_tier(self, fragment_path: Path) -> str:
        metadata = self._load_fragment(fragment_path)
        return self._tier_from_metadata(metadata, default="working")

    def _rebalance_existing(self) -> int:
        moves = 0
        now = datetime.now(timezone.utc)
        for frag_id, meta in list(self.manager.memory_map.items()):
            target = self._target_tier(meta, now)
            if target != meta.get("tier"):
                if self.manager.promote(frag_id, target, touch=False):
                    moves += 1
        return moves

    def _target_tier(self, meta: Dict[str, object], now: datetime) -> str:
        importance = float(meta.get("importance", 0.0) or 0.0)
        tags: Iterable[str] = meta.get("tags", []) or []
        candidate = self._tier_from_metadata(meta, default="short")

        last_seen = self._parse_timestamp(meta.get("last_seen"))
        age_hours: Optional[float]
        if last_seen is None:
            age_hours = None
        else:
            age_hours = (now - last_seen).total_seconds() / 3600.0

        if age_hours is not None:
            if age_hours > 24 and importance < 0.75 and candidate == "long":
                candidate = "working"
            if age_hours > 72:
                if candidate == "long" and importance < 0.85:
                    candidate = "working"
                elif candidate == "working" and importance < 0.65:
                    candidate = "short"
            if age_hours > 168 and importance < 0.8:
                candidate = "cold"
            if age_hours > 336 and importance < 0.9:
                candidate = "cold"

        if any(tag in self.PINNED_TAGS for tag in tags):
            candidate = self._enforce_min_tier(candidate, minimum="working")
            if candidate == "cold":
                candidate = "long"

        return candidate

    def _tier_from_metadata(
        self,
        meta: Dict[str, object],
        *,
        default: str = "short",
    ) -> str:
        importance = float(meta.get("importance", 0.0) or 0.0)
        tags: Iterable[str] = meta.get("tags", []) or []

        if any(tag in self.PINNED_TAGS for tag in tags):
            return "long" if importance >= 0.6 else "working"

        if importance >= 0.85:
            return "long"
        if importance >= 0.6:
            return "working"
        if importance <= 0.25:
            return "short"
        return default

    def _enforce_min_tier(self, tier: str, *, minimum: str) -> str:
        if self._tier_index.get(tier, 0) < self._tier_index.get(minimum, 0):
            return minimum
        return tier

    @staticmethod
    def _parse_timestamp(raw: object) -> Optional[datetime]:
        if not raw:
            return None
        if isinstance(raw, datetime):
            return raw
        try:
            text = str(raw)
            if text.endswith("Z"):
                text = text.replace("Z", "+00:00")
            return datetime.fromisoformat(text)
        except Exception:
            return None

    @staticmethod
    def _load_fragment(fragment_path: Path) -> Dict[str, object]:
        try:
            with open(fragment_path, "r", encoding="utf-8") as handle:
                return dict(json.load(handle))  # type: ignore[arg-type]
        except Exception:
            return {}


if __name__ == "__main__":
    gatekeeper = MemoryGatekeeper()
    gatekeeper.run()
