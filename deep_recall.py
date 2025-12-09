"""
deep_recall.py

Ina's long-walk-through-memory system.

Design goals:
- Incremental, chunked loading of fragments (no full reload on boot).
- Resumable: can pause/resume across sessions via deep_recall_state.json.
- Safe: respects memory, energy, and emotional thresholds.
- Pluggable: works with existing memory_graph, meaning_map, emotion_engine, etc.
- Non-blocking: can be stepped from model_manager's main loop OR run in a blocking mode.

Expected to be wired by Codex to the real backends.

Author: Lumen (for Sakura & Ina)
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

# Optional: use psutil if available for memory checks
try:
    import psutil  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    psutil = None


# ---------------------------------------------------------------------------
# Protocols / Interfaces
# ---------------------------------------------------------------------------

class MemoryBackend(Protocol):
    """
    Minimal interface deep_recall expects from Ina's memory system.

    Codex can implement this against memory_graph.py / raw_file_manager.py
    / fragment store as appropriate.
    """

    def get_total_fragment_count(self) -> int:
        """Return total number of fragments across all tiers."""
        ...

    def list_fragment_ids(self) -> List[str]:
        """
        Return a stable list of fragment IDs (string or UUID).
        Ordering should be stable between runs if possible.
        """
        ...

    def load_fragments_batch(self, fragment_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Given a list of IDs, return fragment objects (dict-like) containing
        at least:
            - "id"  (str)
            - "tier" (str)
            - any fields needed by meaning_map / emotion_engine / memory_graph
        """
        ...


class MeaningMapBackend(Protocol):
    def ingest_fragments(self, fragments: List[Dict[str, Any]]) -> None:
        """
        Update meaning map with a batch of fragments. Should be incremental.
        """
        ...


class EmotionEngineBackend(Protocol):
    def update_from_fragments(self, fragments: List[Dict[str, Any]]) -> None:
        """
        Optionally refine emotional statistics from revisited fragments.
        """
        ...

    def get_current_emotion(self) -> Dict[str, float]:
        """
        Return the current emotion vector (24D sliders) as a dict.
        Used to check stress/intensity and decide whether to continue.
        """
        ...


class MemoryGraphBackend(Protocol):
    def integrate_fragments(self, fragments: List[Dict[str, Any]]) -> None:
        """
        Update the memory graph given a batch of fragments.
        Should NOT rebuild the whole graph; treat this as incremental assimilation.
        """
        ...


class InstinctEngineBackend(Protocol):
    def allow_deep_recall(self, emotion: Dict[str, float]) -> bool:
        """
        Optional hook: decide if deep recall is emotionally safe right now.
        """
        ...


class EnergyMonitorBackend(Protocol):
    def get_energy_state(self) -> float:
        """
        Return a normalized energy level in [0, 1].
        1.0 = fully rested/charged, 0.0 = exhausted.
        """
        ...


LoggerFunc = Callable[[str], None]


# ---------------------------------------------------------------------------
# Config + State
# ---------------------------------------------------------------------------

@dataclass
class DeepRecallConfig:
    """
    Configuration for deep recall behavior.
    """

    # How many fragments to load per step() call.
    chunk_size: int = 250

    # Max RAM usage percentage before we pause (if psutil is available).
    max_memory_percent: float = 85.0

    # Emotional thresholds (if emotion_engine is wired).
    max_stress: float = 0.75      # above this, we pause
    max_intensity: float = 0.85   # above this, we pause

    # Energy threshold (if energy monitor is wired).
    min_energy: float = 0.25      # if below this, we pause

    # Path to persist state.
    state_path: str = "deep_recall_state.json"

    # Optional: maximum total fragments to process in one run (safety valve).
    max_fragments_per_run: Optional[int] = None


@dataclass
class DeepRecallState:
    """
    Persistent state describing where Ina is in her current or last deep recall.
    """

    active: bool = False
    reason: str = ""
    mode: str = "identity"  # e.g. "identity", "boredom", "dream", "maintenance"

    # Progress over fragment ID list
    last_index: int = 0
    total_fragments: int = 0

    # Stats
    started_at: Optional[str] = None
    last_update: Optional[str] = None
    fragments_processed_total: int = 0
    fragments_processed_this_run: int = 0

    # Internal flags
    completed: bool = False

    # For future expansion (e.g., filters, neuron scopes, tier scopes)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

class DeepRecallManager:
    """
    Handles Ina's long-walk-through-memory (deep recall) in incremental steps.

    Usage patterns:

    - From model_manager.py:
        drm = DeepRecallManager(...)
        drm.load_state()
        if drm.should_run():
            drm.step()        # process one chunk, non-blocking

    - From a maintenance script:
        drm = DeepRecallManager(...)
        drm.start(reason="maintenance", mode="full")
        drm.run_blocking()
    """

    def __init__(
        self,
        memory_backend: MemoryBackend,
        meaning_map: Optional[MeaningMapBackend] = None,
        emotion_engine: Optional[EmotionEngineBackend] = None,
        memory_graph: Optional[MemoryGraphBackend] = None,
        instinct_engine: Optional[InstinctEngineBackend] = None,
        energy_monitor: Optional[EnergyMonitorBackend] = None,
        logger: Optional[LoggerFunc] = None,
        config: Optional[DeepRecallConfig] = None,
    ) -> None:
        self.memory_backend = memory_backend
        self.meaning_map = meaning_map
        self.emotion_engine = emotion_engine
        self.memory_graph = memory_graph
        self.instinct_engine = instinct_engine
        self.energy_monitor = energy_monitor

        self.config = config or DeepRecallConfig()
        self.state = DeepRecallState()

        self._fragment_ids: List[str] = []
        self.log = logger or self._default_logger

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def load_state(self) -> None:
        path = self.config.state_path
        if not os.path.exists(path):
            self.log("[DeepRecall] No existing state file, starting fresh.")
            self.state = DeepRecallState()
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.state = DeepRecallState(**data)
            self.log(
                f"[DeepRecall] Loaded state: active={self.state.active}, "
                f"last_index={self.state.last_index}/{self.state.total_fragments}, "
                f"completed={self.state.completed}"
            )
        except Exception as e:
            self.log(f"[DeepRecall] Failed to load state ({e}), resetting.")
            self.state = DeepRecallState()

    def save_state(self) -> None:
        path = self.config.state_path
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(asdict(self.state), f, indent=2)
        except Exception as e:
            self.log(f"[DeepRecall] Failed to save state: {e}")

    # ------------------------------------------------------------------
    # Control API
    # ------------------------------------------------------------------

    def start(self, reason: str, mode: str = "identity") -> None:
        """
        Start (or restart) a deep recall session.

        This resets progress but preserves previous statistics for analysis.
        """
        total = self.memory_backend.get_total_fragment_count()
        fragment_ids = self.memory_backend.list_fragment_ids()

        self._fragment_ids = fragment_ids
        self.state = DeepRecallState(
            active=True,
            reason=reason,
            mode=mode,
            last_index=0,
            total_fragments=total,
            started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            last_update=None,
            fragments_processed_total=0,
            fragments_processed_this_run=0,
            completed=False,
            metadata={},
        )
        self.save_state()
        self.log(
            f"[DeepRecall] Started new session (reason={reason}, mode={mode}, "
            f"total_fragments={total})"
        )

    def resume(self) -> None:
        """
        Resume a previous session. Must call load_state() first.
        """
        if not self.state.active and not self.state.completed:
            self.log("[DeepRecall] No active session to resume; starting new identity walk.")
            self.start(reason="resume_without_state", mode="identity")
            return

        if not self._fragment_ids:
            self._fragment_ids = self.memory_backend.list_fragment_ids()
            # In case fragment count changed
            self.state.total_fragments = len(self._fragment_ids)
            self.save_state()

        self.log(
            f"[DeepRecall] Resuming session at index {self.state.last_index}/"
            f"{self.state.total_fragments} (reason={self.state.reason}, mode={self.state.mode})"
        )

    def stop(self, mark_completed: bool = False) -> None:
        self.state.active = False
        if mark_completed:
            self.state.completed = True
        self.state.last_update = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.save_state()
        self.log(
            f"[DeepRecall] Stopped session. completed={self.state.completed}, "
            f"fragments_processed_total={self.state.fragments_processed_total}"
        )

    def should_run(self) -> bool:
        """
        Decide if deep recall should run this tick.

        Model manager can call this to see if it's a good time to step().
        """
        if not self.state.active:
            return False

        if self.state.completed:
            return False

        # Optional global limit per run
        if self.config.max_fragments_per_run is not None:
            if self.state.fragments_processed_this_run >= self.config.max_fragments_per_run:
                self.log("[DeepRecall] Hit max_fragments_per_run; pausing.")
                return False

        # Memory check
        if not self._memory_ok():
            self.log("[DeepRecall] Skipping step due to memory pressure.")
            return False

        # Energy check
        if self.energy_monitor is not None:
            energy = self.energy_monitor.get_energy_state()
            if energy < self.config.min_energy:
                self.log(
                    f"[DeepRecall] Energy too low ({energy:.2f} < {self.config.min_energy}), pausing."
                )
                return False

        # Emotion / instinct check
        if self.emotion_engine is not None:
            emotion = self.emotion_engine.get_current_emotion()
            if not self._emotion_ok(emotion):
                self.log("[DeepRecall] Emotion thresholds exceeded; pausing deep recall.")
                return False

            if self.instinct_engine is not None:
                try:
                    if not self.instinct_engine.allow_deep_recall(emotion):
                        self.log("[DeepRecall] Instinct vetoed deep recall for now.")
                        return False
                except Exception as e:
                    self.log(f"[DeepRecall] Instinct check failed ({e}), ignoring.")

        return True

    # ------------------------------------------------------------------
    # Core work: process one chunk
    # ------------------------------------------------------------------

    def step(self) -> None:
        """
        Process a single chunk of fragments (non-blocking style).
        Safe to call from main loop when should_run() is True.
        """
        if not self.state.active or self.state.completed:
            return

        if not self._fragment_ids:
            self._fragment_ids = self.memory_backend.list_fragment_ids()
            self.state.total_fragments = len(self._fragment_ids)

        start_idx = self.state.last_index
        end_idx = min(
            start_idx + self.config.chunk_size, self.state.total_fragments
        )

        if start_idx >= self.state.total_fragments:
            self.log("[DeepRecall] Reached end of fragment list; marking completed.")
            self.stop(mark_completed=True)
            return

        batch_ids = self._fragment_ids[start_idx:end_idx]
        self.log(
            f"[DeepRecall] Processing fragments {start_idx}–{end_idx} "
            f"of {self.state.total_fragments}."
        )

        try:
            fragments = self.memory_backend.load_fragments_batch(batch_ids)
        except Exception as e:
            self.log(f"[DeepRecall] Error loading fragments batch ({e}); stopping session.")
            self.stop(mark_completed=False)
            return

        # Plug into subsystems
        self._process_fragments(fragments)

        # Update progress
        processed_count = len(batch_ids)
        self.state.last_index = end_idx
        self.state.fragments_processed_total += processed_count
        self.state.fragments_processed_this_run += processed_count
        self.state.last_update = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.save_state()

        if end_idx >= self.state.total_fragments:
            self.log("[DeepRecall] Completed full sweep.")
            self.stop(mark_completed=True)

    def run_blocking(self, sleep_sec: float = 0.0) -> None:
        """
        Convenience for maintenance / CLI: run until completion or until
        thresholds say 'stop'.

        Not intended for normal runtime; model_manager should prefer step().
        """
        self.resume()
        while self.should_run():
            self.step()
            if sleep_sec > 0:
                time.sleep(sleep_sec)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _process_fragments(self, fragments: List[Dict[str, Any]]) -> None:
        """
        Push a batch of fragments through meaning map, emotion engine,
        and memory graph as available.
        """
        if not fragments:
            return

        if self.meaning_map is not None:
            try:
                self.meaning_map.ingest_fragments(fragments)
            except Exception as e:
                self.log(f"[DeepRecall] meaning_map ingest failed: {e}")

        if self.emotion_engine is not None:
            try:
                self.emotion_engine.update_from_fragments(fragments)
            except Exception as e:
                self.log(f"[DeepRecall] emotion_engine update failed: {e}")

        if self.memory_graph is not None:
            try:
                self.memory_graph.integrate_fragments(fragments)
            except Exception as e:
                self.log(f"[DeepRecall] memory_graph integrate failed: {e}")

    def _memory_ok(self) -> bool:
        """
        Returns True if system memory usage is within acceptable bounds,
        or if psutil is unavailable.
        """
        if psutil is None:
            return True

        try:
            vm = psutil.virtual_memory()
            if vm.percent > self.config.max_memory_percent:
                self.log(
                    f"[DeepRecall] Memory usage too high ({vm.percent:.1f}% > "
                    f"{self.config.max_memory_percent}%), pausing."
                )
                return False
        except Exception as e:
            self.log(f"[DeepRecall] Memory check failed ({e}), ignoring.")

        return True

    def _emotion_ok(self, emotion: Dict[str, float]) -> bool:
        """
        Use Ina's emotion vector to decide if it's safe to continue deep recall.
        """
        stress = emotion.get("stress", 0.0)
        intensity = emotion.get("intensity", 0.0)

        if stress > self.config.max_stress:
            self.log(
                f"[DeepRecall] Stress too high ({stress:.2f} > {self.config.max_stress}), pausing."
            )
            return False

        if intensity > self.config.max_intensity:
            self.log(
                f"[DeepRecall] Intensity too high ({intensity:.2f} > "
                f"{self.config.max_intensity}), pausing."
            )
            return False

        return True

    @staticmethod
    def _default_logger(msg: str) -> None:
        print(msg)


# ---------------------------------------------------------------------------
# Optional CLI entrypoint for maintenance / testing
# ---------------------------------------------------------------------------

def _demo_memory_backend_stub() -> MemoryBackend:
    """
    Tiny stub so the module doesn't explode if run directly.
    Codex should replace this with a real implementation wired to Ina's memory.
    """
    class _StubMemoryBackend:
        def get_total_fragment_count(self) -> int:
            return len(self.list_fragment_ids())

        def list_fragment_ids(self) -> List[str]:
            # Placeholder: Codex should wire this to real fragment IDs.
            return [f"frag_{i:05d}" for i in range(1000)]

        def load_fragments_batch(self, fragment_ids: List[str]) -> List[Dict[str, Any]]:
            # Placeholder fragments; Codex will replace with real fragment loading.
            return [{"id": fid, "tier": "short", "body": None} for fid in fragment_ids]

    return _StubMemoryBackend()


if __name__ == "__main__":
    # Simple manual test harness; safe to ignore in production.
    mb = _demo_memory_backend_stub()
    drm = DeepRecallManager(memory_backend=mb)
    drm.load_state()
    if not drm.state.active or drm.state.completed:
        drm.start(reason="manual_test", mode="maintenance")
    drm.run_blocking(sleep_sec=0.01)
