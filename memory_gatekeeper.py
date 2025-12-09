# memory_gatekeeper.py
"""
Memory gatekeeper for Ina's fragments.

This module decides what to do with newly created fragments:
  - drop them
  - store them in short_term / working / long_term / cold shards

It sits between:
  - fragmentation_engine (upstream, creates fragments)
  - fragment_archiver (downstream, persists fragments)
  - memory_graph (later, for structural reasoning)

For now, routing is rule-based and uses:
  - modality        (audio / vision)
  - metadata.flags  (e.g. "self_voice", "dreamstate", "high_emotion")
  - attention_state (focused / peripheral / ignored)
  - optional extra fields in metadata.extra

This is intentionally simple and explainable so we can evolve it
as Ina's emotion + meaning systems mature.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from fragment_schema import Fragment, Modality, AttentionState
from fragment_archiver import FragmentArchiver, FragmentArchiverConfig


# --------------------------------------------------------------------------- #
# Gate decisions
# --------------------------------------------------------------------------- #


class GateAction(str, Enum):
    DROP = "drop"
    STORE = "store"  # store in a specific shard


@dataclass
class GateDecision:
    """
    Result of routing a fragment.

    Attributes:
        action: GateAction.DROP or GateAction.STORE
        target_shard: valid shard name if action == STORE, else None
        reason: human-readable note for logging / introspection
    """

    action: GateAction
    target_shard: Optional[str]
    reason: str


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass
class MemoryGatekeeperConfig:
    """
    Config for MemoryGatekeeper.

    Attributes:
        archiver_config: config passed through to FragmentArchiver.
        short_term_shard: shard name for "recent, maybe important" fragments.
        working_shard: shard name for "actively being thought about / used".
        long_term_shard: shard name for "kept, important, likely to recur".
        cold_shard: shard name for "rarely used, archival".
        drop_ignored: if True, fragments with attention_state=IGNORED are dropped
                      unless explicitly forced by flags.
    """

    archiver_config: FragmentArchiverConfig = FragmentArchiverConfig()

    short_term_shard: str = "short_term"
    working_shard: str = "working"
    long_term_shard: str = "long_term"
    cold_shard: str = "cold"

    drop_ignored: bool = True


# --------------------------------------------------------------------------- #
# MemoryGatekeeper
# --------------------------------------------------------------------------- #


class MemoryGatekeeper:
    """
    Central gatekeeper responsible for routing fragments into Ina's memory tiers.

    Typical usage:

        gk_cfg = MemoryGatekeeperConfig()
        gatekeeper = MemoryGatekeeper(gk_cfg)

        # from fragmentation_engine:
        decision, frag_id = gatekeeper.handle_new_fragment(fragment)

    If you just want the decision without writing to disk:

        decision = gatekeeper.route_fragment(fragment)
    """

    def __init__(self, config: MemoryGatekeeperConfig) -> None:
        self.config = config
        self.archiver = FragmentArchiver(config.archiver_config)

        # Convenience: set of known shard names for validation
        self._valid_shards = set(self.config.archiver_config.shard_names)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def handle_new_fragment(self, fragment: Fragment) -> Tuple[GateDecision, Optional[str]]:
        """
        Route the fragment and, if stored, archive it.

        Returns:
            (GateDecision, fragment_id or None)
        """
        decision = self.route_fragment(fragment)

        if decision.action == GateAction.DROP:
            # For debugging, you might log the reason here.
            return decision, None

        shard = decision.target_shard
        if shard is None:
            # Misconfiguration: STORE action but no shard.
            # Fallback: short_term.
            shard = self.config.short_term_shard

        if shard not in self._valid_shards:
            # Unknown shard name – again fallback to short_term.
            shard = self.config.short_term_shard

        frag_id = self.archiver.save_fragment(fragment, target_shard=shard)
        return decision, frag_id

    def route_fragment(self, fragment: Fragment) -> GateDecision:
        """
        Decide what to do with a fragment without writing it to disk.

        The current rule set is deliberately simple and based on:
          - attention_state
          - metadata.flags
          - lightweight intent tags (action/communication vs thought/logic)
          - modality
          - optional heuristic fields in metadata.extra
        """
        meta = fragment.get("metadata", {})
        flags: List[str] = meta.get("flags", [])
        tags: List[str] = meta.get("tags", []) or fragment.get("tags", [])
        modality_str: str = meta.get("modality", "")
        attention_state_str: str = meta.get("attention_state", AttentionState.PERIPHERAL.value)

        # Convert to enums where possible
        try:
            modality = Modality(modality_str)
        except ValueError:
            modality = None  # type: ignore

        try:
            attention_state = AttentionState(attention_state_str)
        except ValueError:
            attention_state = AttentionState.PERIPHERAL

        # Lightweight intent classification
        is_action = self._has_any_flag(flags, ["action", "communication", "speech", "comm"]) or any(
            t in {"action", "communication", "comm", "speech"} for t in tags
        )
        is_thought = self._has_any_flag(flags, ["logic", "thought"]) or any(
            t in {"logic", "thought", "reasoning"} for t in tags
        )

        # ------------------------------------------------------------------
        # 1. Hard drop rules
        # ------------------------------------------------------------------
        if self._has_flag(flags, "drop"):
            return GateDecision(
                action=GateAction.DROP,
                target_shard=None,
                reason="Explicit drop flag set on fragment",
            )

        if self.config.drop_ignored and attention_state == AttentionState.IGNORED:
            # Unless explicitly protected by some flag, we discard ignored noise.
            if not self._has_any_flag(
                flags, ["high_emotion", "system_event", "self_voice"]
            ) and not (is_action or is_thought):
                return GateDecision(
                    action=GateAction.DROP,
                    target_shard=None,
                    reason="Ignored input (no override flags)",
                )

        # ------------------------------------------------------------------
        # 2. High-salience routing (emotionally or semantically important)
        # ------------------------------------------------------------------
        # NOTE: meta["extra"] is a good place to stash numbers like
        # "emotion_intensity", "novelty_score", etc. when you have them.
        extra: Dict[str, Any] = meta.get("extra", {}) or {}
        emotion_intensity = float(extra.get("emotion_intensity", 0.0))  # [-1, 1] if you adopt that convention
        novelty_score = float(extra.get("novelty_score", 0.0))          # [0, 1] heuristic
        category = str(extra.get("category", "")).lower()

        if category in {"action", "communication"} and not is_action:
            is_action = True
        if category in {"thought", "logic"} and not is_thought:
            is_thought = True

        # Example heuristic thresholds – adjust later:
        HIGH_EMOTION = 0.6
        HIGH_NOVELTY = 0.7

        if self._has_flag(flags, "system_event"):
            # Things like wake/sleep, crash/reboot, etc.
            return GateDecision(
                action=GateAction.STORE,
                target_shard=self.config.long_term_shard,
                reason="System event fragment",
            )

        if self._has_flag(flags, "self_reflection"):
            return GateDecision(
                action=GateAction.STORE,
                target_shard=self.config.long_term_shard,
                reason="Self-reflection fragment",
            )

        # Explicit thought/logic gets long-term priority
        if is_thought:
            return GateDecision(
                action=GateAction.STORE,
                target_shard=self.config.long_term_shard,
                reason="Logic/thought fragment",
            )

        # Explicit communication/action goes to working tier
        if is_action:
            return GateDecision(
                action=GateAction.STORE,
                target_shard=self.config.working_shard,
                reason="Communication/action fragment",
            )

        if emotion_intensity >= HIGH_EMOTION or self._has_flag(flags, "high_emotion"):
            return GateDecision(
                action=GateAction.STORE,
                target_shard=self.config.long_term_shard,
                reason="High emotional intensity",
            )

        if novelty_score >= HIGH_NOVELTY or self._has_flag(flags, "novel"):
            # Strongly novel experience, but maybe not emotionally intense yet.
            return GateDecision(
                action=GateAction.STORE,
                target_shard=self.config.working_shard,
                reason="High novelty",
            )

        # ------------------------------------------------------------------
        # 3. Modality-specific heuristics
        # ------------------------------------------------------------------
        if modality == Modality.AUDIO:
            # Examples:
            # - Ina's own speech ("self_voice") might go to working/long_term.
            # - Background ambient might be short_term.
            if self._has_flag(flags, "self_voice"):
                return GateDecision(
                    action=GateAction.STORE,
                    target_shard=self.config.working_shard,
                    reason="Self-voice audio fragment",
                )
            if self._has_flag(flags, "music"):
                return GateDecision(
                    action=GateAction.STORE,
                    target_shard=self.config.short_term_shard,
                    reason="Music audio fragment",
                )

        if modality == Modality.VISION:
            if self._has_flag(flags, "face_detected"):
                return GateDecision(
                    action=GateAction.STORE,
                    target_shard=self.config.working_shard,
                    reason="Vision fragment with face detected",
                )
            if self._has_flag(flags, "ui_like") or self._has_flag(flags, "text_like"):
                # Screens, interfaces, etc. – probably short term unless reused.
                return GateDecision(
                    action=GateAction.STORE,
                    target_shard=self.config.short_term_shard,
                    reason="UI/text-like vision fragment",
                )

        # ------------------------------------------------------------------
        # 4. Default routing based on attention
        # ------------------------------------------------------------------
        if attention_state == AttentionState.FOCUSED:
            return GateDecision(
                action=GateAction.STORE,
                target_shard=self.config.short_term_shard,
                reason="Focused input (default short_term)",
            )

        if attention_state == AttentionState.PERIPHERAL:
            # Peripheral but not ignored: store in short_term or cold depending on flags.
            if self._has_flag(flags, "dreamstate"):
                return GateDecision(
                    action=GateAction.STORE,
                    target_shard=self.config.cold_shard,
                    reason="Dreamstate fragment (cold tier)",
                )
            return GateDecision(
                action=GateAction.STORE,
                target_shard=self.config.short_term_shard,
                reason="Peripheral input (default short_term)",
            )

        # Fallback – shouldn't normally hit if attention_state is valid.
        return GateDecision(
            action=GateAction.STORE,
            target_shard=self.config.short_term_shard,
            reason="Fallback routing (short_term)",
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _has_flag(flags: List[str], flag: str) -> bool:
        return flag in flags

    @staticmethod
    def _has_any_flag(flags: List[str], candidates: List[str]) -> bool:
        return any(f in flags for f in candidates)
