# fragment_archiver.py
"""
Disk-backed archiver for Ina's memory fragments.

This module sits between:
  - fragmentation_engine + memory_gatekeeper (upstream)
  - memory_graph, training code, etc. (downstream)

Responsibilities:
  - Append validated fragments to shard-specific storage.
  - Assign and preserve `fragment_id` if missing.
  - Rotate physical files when they reach a size threshold.
  - Provide iteration and simple lookup over fragments per shard.
  - Support basic migration between shards (short_term → long_term, etc.).

Shards are logical memory tiers:
  - "short_term"
  - "working"
  - "long_term"
  - "cold"
You can configure whatever names you need in FragmentArchiverConfig.

Storage format:
  - Each shard is a directory under base_dir, e.g. ./fragments/short_term
  - Inside each shard, fragments are stored as JSON Lines:
        shard_000001.jsonl
        shard_000002.jsonl
        ...
  - Each line is one JSON object:
        {
          "fragment_id": "...",
          "metadata": {...},
          "payload": {...}  # e.g., audio_features, raw_audio_path, image_focus, etc.
        }

Validation is handled by fragment_validator.py – this module assumes
it receives valid fragments or that the caller handles validation.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple

from fragment_schema import Fragment


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass
class FragmentArchiverConfig:
    """
    Configuration for the fragment archiver.

    Attributes:
        base_dir: Root directory where all shard subdirectories live.
        shard_names: List of valid shard names (e.g. ["short_term", "working", "long_term", "cold"]).
        max_file_size_bytes: When a shard's current file reaches this size, roll to a new one.
        gzip_enabled: Placeholder for future compressed shard support.
    """

    base_dir: str = "./fragments"
    shard_names: List[str] = field(
        default_factory=lambda: ["short_term", "working", "long_term", "cold"]
    )
    max_file_size_bytes: int = 64 * 1024 * 1024  # 64MB by default
    gzip_enabled: bool = False  # currently unused; future hook


# --------------------------------------------------------------------------- #
# Internal: per-shard store (append-only)
# --------------------------------------------------------------------------- #


class _ShardStore:
    """
    Internal append-only store for a single shard.

    Not thread-safe by itself; wrap with locks if you write from
    multiple threads/processes.
    """

    def __init__(self, shard_dir: str, max_file_size_bytes: int) -> None:
        self.shard_dir = shard_dir
        self.max_file_size_bytes = max_file_size_bytes

        os.makedirs(self.shard_dir, exist_ok=True)

        self._current_index: int = 0
        self._current_file_path: Optional[str] = None
        self._current_file_handle = None  # type: ignore

        self._init_current_file()

    # --- public-like methods (used by FragmentArchiver) ------------------- #

    def append_fragment(self, fragment: Fragment) -> str:
        """
        Append a fragment to this shard and return its fragment_id.

        If the fragment lacks 'fragment_id', one is generated.
        """
        if "fragment_id" not in fragment:
            fragment_id = self._generate_fragment_id()
            fragment["fragment_id"] = fragment_id
        else:
            fragment_id = str(fragment["fragment_id"])

        self._roll_file_if_needed()

        line = json.dumps(fragment, ensure_ascii=False)
        self._current_file_handle.write(line + "\n")
        self._current_file_handle.flush()

        return fragment_id

    def iter_fragments(self) -> Generator[Fragment, None, None]:
        """
        Iterate over all fragments in this shard, oldest first.

        Ensures the current file is closed before reading to avoid
        partial writes being read mid-line.
        """
        self._ensure_closed_file()

        files = sorted(
            f
            for f in os.listdir(self.shard_dir)
            if f.startswith("shard_") and f.endswith(".jsonl")
        )
        for fname in files:
            path = os.path.join(self.shard_dir, fname)
            with open(path, "rt", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        frag = json.loads(line)
                        yield frag
                    except json.JSONDecodeError:
                        # Corrupted line; skip. Hook up real logging if needed.
                        continue

    def find_fragment(self, fragment_id: str) -> Optional[Fragment]:
        """
        Linear search for a fragment by id within this shard.

        This is O(N); if you need fast lookup later, build an index
        in memory_graph or a dedicated index file.
        """
        for frag in self.iter_fragments():
            if frag.get("fragment_id") == fragment_id:
                return frag
        return None

    # --- internal helpers ------------------------------------------------- #

    def _init_current_file(self) -> None:
        """Determine the next shard file name to write to and open it."""
        existing_indices: List[int] = []
        for fname in os.listdir(self.shard_dir):
            if fname.startswith("shard_") and fname.endswith(".jsonl"):
                try:
                    idx_str = fname.split("_")[1].split(".")[0]
                    existing_indices.append(int(idx_str))
                except (IndexError, ValueError):
                    continue

        if existing_indices:
            self._current_index = max(existing_indices)
        else:
            self._current_index = 1

        self._open_current_file()

    def _open_current_file(self) -> None:
        """Open (or reopen) the current shard file for appending."""
        fname = f"shard_{self._current_index:06d}.jsonl"
        self._current_file_path = os.path.join(self.shard_dir, fname)
        self._current_file_handle = open(self._current_file_path, "a", encoding="utf-8")

    def _roll_file_if_needed(self) -> None:
        """Rotate to a new file if the current file exceeds the size threshold."""
        if self._current_file_handle is None or self._current_file_path is None:
            self._open_current_file()
            return

        try:
            size = os.path.getsize(self._current_file_path)
        except OSError:
            size = 0

        if size >= self.max_file_size_bytes:
            self._current_file_handle.close()
            self._current_index += 1
            self._open_current_file()

    def _ensure_closed_file(self) -> None:
        """Close the current file if it's open (for safe iteration)."""
        if self._current_file_handle is not None:
            try:
                self._current_file_handle.close()
            except Exception:
                pass
            self._current_file_handle = None
            self._current_file_path = None

    @staticmethod
    def _generate_fragment_id() -> str:
        """Generate a globally unique fragment ID."""
        return str(uuid.uuid4())


# --------------------------------------------------------------------------- #
# FragmentArchiver – public API
# --------------------------------------------------------------------------- #


class FragmentArchiver:
    """
    High-level archiver that manages multiple shard stores.

    This is the component that other modules should talk to.

    Typical usage:

        cfg = FragmentArchiverConfig(base_dir="./fragments")
        archiver = FragmentArchiver(cfg)

        # From memory_gatekeeper:
        frag_id = archiver.save_fragment(fragment, target_shard="short_term")

        # For memory_graph:
        for frag in archiver.iter_shard("long_term"):
            ...

        # Promote:
        archiver.migrate_fragment(frag_id, from_shard="short_term", to_shard="long_term")
    """

    def __init__(self, config: FragmentArchiverConfig) -> None:
        self.config = config
        self._shards: Dict[str, _ShardStore] = {}

        # Initialize all shard directories & stores
        for name in self.config.shard_names:
            shard_dir = os.path.join(self.config.base_dir, name)
            self._shards[name] = _ShardStore(
                shard_dir=shard_dir,
                max_file_size_bytes=self.config.max_file_size_bytes,
            )

    # ------------------------------------------------------------------ #
    # Public methods
    # ------------------------------------------------------------------ #

    def save_fragment(self, fragment: Fragment, target_shard: str) -> str:
        """
        Save a fragment into the given shard and return its fragment_id.

        Assumes the fragment is already validated; if you want validation,
        call fragment_validator.validate_fragment(...) before this.
        """
        store = self._get_shard_store(target_shard)
        fragment_id = store.append_fragment(fragment)
        return fragment_id

    def iter_shard(self, shard_name: str) -> Generator[Fragment, None, None]:
        """
        Iterate over all fragments in the given shard.
        """
        store = self._get_shard_store(shard_name)
        yield from store.iter_fragments()

    def iter_shard_limited(
        self, shard_name: str, limit: int
    ) -> Iterable[Fragment]:
        """
        Iterate over at most `limit` fragments from a given shard.
        """
        count = 0
        for frag in self.iter_shard(shard_name):
            yield frag
            count += 1
            if count >= limit:
                break

    def iter_all_shards(self) -> Generator[Tuple[str, Fragment], None, None]:
        """
        Iterate over all shards, yielding (shard_name, fragment).
        """
        for name in self.config.shard_names:
            for frag in self.iter_shard(name):
                yield name, frag

    def find_fragment(
        self, fragment_id: str, shard_hint: Optional[str] = None
    ) -> Optional[Tuple[str, Fragment]]:
        """
        Best-effort lookup of a fragment by id.

        If shard_hint is provided, only that shard is searched.
        Otherwise, all shards are scanned linearly.

        Returns:
            (shard_name, fragment) if found, else None.
        """
        if shard_hint is not None:
            if shard_hint not in self._shards:
                return None
            frag = self._shards[shard_hint].find_fragment(fragment_id)
            if frag is not None:
                return shard_hint, frag
            return None

        # no hint: scan all shards
        for name in self.config.shard_names:
            frag = self._shards[name].find_fragment(fragment_id)
            if frag is not None:
                return name, frag
        return None

    def migrate_fragment(
        self,
        fragment_id: str,
        from_shard: str,
        to_shard: str,
        delete_from_source: bool = False,
    ) -> Optional[str]:
        """
        Migrate a fragment from one shard to another.

        This is implemented as:
          - find fragment in from_shard
          - append it to to_shard (possibly with same fragment_id)
          - optionally leave the original as historical or mark it for GC

        Returns:
            new_fragment_id in the destination shard, or None if not found.

        NOTE:
            This is a simple append-only implementation; it does not
            remove or rewrite the fragment in the source file. Actual
            GC/cleanup can be done in a separate maintenance pass.
        """
        if from_shard == to_shard:
            # No-op migration
            return fragment_id

        if from_shard not in self._shards or to_shard not in self._shards:
            return None

        src_store = self._shards[from_shard]
        dst_store = self._shards[to_shard]

        frag = src_store.find_fragment(fragment_id)
        if frag is None:
            return None

        # Optionally mark original as migrated in-place (soft delete)
        if delete_from_source:
            # You can choose to set a flag on the fragment here;
            # this requires rewriting the source file, so skipping for now.
            # Instead, rely on a "migrated" or "archived" flag if you store that.
            pass

        # Ensure fragment_id is preserved (or regenerated if you prefer).
        new_id = frag.get("fragment_id", fragment_id)
        frag["fragment_id"] = new_id
        dst_store.append_fragment(frag)

        return str(new_id)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _get_shard_store(self, shard_name: str) -> _ShardStore:
        if shard_name not in self._shards:
            # Auto-create shard if needed, or raise. For now, raise to catch bugs.
            raise KeyError(f"Unknown shard name: {shard_name!r}")
        return self._shards[shard_name]
