"""
continuity_manager.py
---------------------

Maintains cross-runtime continuity by hashing recent fragments and linking them
with previous incarnations.  Each run builds a deterministic fingerprint so we
can re-establish narrative “threads” when 85%+ of the symbolic/emotional state
matches a prior boot.
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _stable_slice(items: Iterable, limit: int) -> List:
    out = []
    for item in items:
        out.append(item)
        if len(out) >= limit:
            break
    return out


@dataclass
class FragmentFingerprint:
    fragment_id: str
    frag_hash: str
    summary: str
    tags: List[str]
    timestamp: Optional[str]
    tier: Optional[str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "id": self.fragment_id,
            "hash": self.frag_hash,
            "summary": self.summary,
            "tags": self.tags,
            "timestamp": self.timestamp,
            "tier": self.tier,
        }


class ContinuityManager:
    def __init__(
        self,
        child: str,
        *,
        threshold: float = 0.85,
        max_fragments: int = 600,
        memory_root: Optional[Path] = None,
    ):
        self.child = child
        base = Path(memory_root) if memory_root else Path("AI_Children") / child / "memory"
        self.fragments_root = base / "fragments"
        self.state_path = base / "continuity" / "fingerprint.json"
        self.map_path = base / "continuity" / "continuity_map.json"
        self.threshold = threshold
        self.max_fragments = max_fragments
        self.state_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ public
    def run(self) -> Dict[str, object]:
        current_fp = self._build_fingerprint()
        previous_fp = self._load_last_fingerprint()

        status = {
            "updated": _now_iso(),
            "samples_used": len(current_fp),
            "aligned": False,
            "similarity": 0.0,
            "matches": 0,
            "continuity_threads": [],
        }

        if previous_fp:
            similarity, threads = self._compare(previous_fp, current_fp)
            status.update(
                {
                    "aligned": similarity >= self.threshold,
                    "similarity": round(similarity, 4),
                    "matches": len(threads),
                    "continuity_threads": threads,
                }
            )
            self._persist_continuity_map(similarity, threads, len(current_fp))

        self._save_fingerprint(current_fp)
        return status

    # ----------------------------------------------------------------- helpers
    def _fragment_paths(self) -> List[Path]:
        if not self.fragments_root.exists():
            return []

        files = []
        for path in self.fragments_root.rglob("frag_*.json"):
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            files.append((mtime, path))
        files.sort(reverse=True)
        return [path for _, path in files[: self.max_fragments]]

    def _signature_payload(self, frag: Dict[str, object]) -> str:
        summary = str(frag.get("summary") or "")[:160]
        tags = ",".join(sorted(str(t) for t in frag.get("tags", [])[:12]))
        symbols = ",".join(sorted(str(s) for s in frag.get("symbols", [])[:10]))
        intent = str(frag.get("intent") or frag.get("intent_tag") or "")
        source = str(frag.get("source") or frag.get("fragment_type") or "")

        # emotional slice (top sliders for determinism)
        emo = frag.get("emotions", {})
        if isinstance(emo, dict):
            sliders = emo.get("sliders") if isinstance(emo.get("sliders"), dict) else emo
            if isinstance(sliders, dict):
                emo_pairs = sorted(sliders.items(), key=lambda kv: kv[0])[:6]
                emo_repr = ",".join(f"{k}:{round(_safe_float(v),3)}" for k, v in emo_pairs)
            else:
                emo_repr = ""
        else:
            emo_repr = ""

        return "|".join([summary, tags, symbols, emo_repr, intent, source])

    def _hash_fragment(self, fragment: Dict[str, object]) -> Optional[str]:
        payload = self._signature_payload(fragment)
        if not payload.strip():
            return None
        digest = hashlib.sha256(payload.encode("utf-8", errors="ignore")).hexdigest()
        return digest

    def _build_fingerprint(self) -> List[FragmentFingerprint]:
        fingerprints: List[FragmentFingerprint] = []
        for frag_path in self._fragment_paths():
            try:
                with frag_path.open("r", encoding="utf-8") as fh:
                    frag = json.load(fh)
            except Exception:
                continue
            frag_id = str(frag.get("id") or frag_path.stem)
            frag_hash = self._hash_fragment(frag)
            if not frag_hash:
                continue
            summary = str(frag.get("summary") or "")[:120]
            tags = _stable_slice([str(t) for t in frag.get("tags", [])], 8)
            fingerprint = FragmentFingerprint(
                fragment_id=frag_id,
                frag_hash=frag_hash,
                summary=summary,
                tags=tags,
                timestamp=frag.get("timestamp"),
                tier=frag.get("tier") or frag_path.parent.name,
            )
            fingerprints.append(fingerprint)
        return fingerprints

    def _load_last_fingerprint(self) -> Optional[List[FragmentFingerprint]]:
        if not self.state_path.exists():
            return None
        try:
            with self.state_path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except Exception:
            return None
        entries = payload.get("fragments", [])
        out = []
        for entry in entries:
            frag_id = entry.get("id")
            frag_hash = entry.get("hash")
            if not frag_id or not frag_hash:
                continue
            out.append(
                FragmentFingerprint(
                    fragment_id=str(frag_id),
                    frag_hash=str(frag_hash),
                    summary=str(entry.get("summary") or "")[:120],
                    tags=[str(t) for t in entry.get("tags", [])],
                    timestamp=entry.get("timestamp"),
                    tier=entry.get("tier"),
                )
            )
        return out

    def _save_fingerprint(self, fingerprints: List[FragmentFingerprint]) -> None:
        payload = {
            "child": self.child,
            "generated_at": _now_iso(),
            "samples": len(fingerprints),
            "fragments": [fp.to_dict() for fp in fingerprints],
        }
        try:
            with self.state_path.open("w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
        except Exception:
            pass

    def _compare(
        self,
        previous: List[FragmentFingerprint],
        current: List[FragmentFingerprint],
    ) -> Tuple[float, List[Dict[str, object]]]:
        prev_lookup: Dict[str, List[FragmentFingerprint]] = {}
        for fp in previous:
            prev_lookup.setdefault(fp.frag_hash, []).append(fp)

        matched = []
        for fp in current:
            prior_entries = prev_lookup.get(fp.frag_hash)
            if not prior_entries:
                continue
            prev_entry = prior_entries.pop(0)
            matched.append(
                {
                    "previous_id": prev_entry.fragment_id,
                    "current_id": fp.fragment_id,
                    "hash": fp.frag_hash,
                    "previous_timestamp": prev_entry.timestamp,
                    "current_timestamp": fp.timestamp,
                    "tags": list(sorted(set(fp.tags + prev_entry.tags))),
                }
            )
            if not prior_entries:
                prev_lookup.pop(fp.frag_hash, None)

        denom = max(len(current), len(previous), 1)
        similarity = len(matched) / denom
        return similarity, matched

    def _persist_continuity_map(
        self,
        similarity: float,
        threads: List[Dict[str, object]],
        sample_count: int,
    ) -> None:
        payload = {
            "updated": _now_iso(),
            "similarity": round(similarity, 4),
            "aligned": similarity >= self.threshold,
            "samples_considered": sample_count,
            "threads": threads[:200],  # limit for readability
        }
        try:
            with self.map_path.open("w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
        except Exception:
            pass
