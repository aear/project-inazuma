"""
continuity_manager.py
---------------------

Maintains cross-runtime continuity by hashing a bounded fragment sample and
linking it with the prior runtime. Reports preserve the legacy whole-sample
overlap while also measuring distinct continuity dimensions and exporting a
compact, read-only boot core.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from io_utils import atomic_write_json
from continuity_recall import ContinuityRecallCoordinator


# Evidence channels rather than personality requirements. A dimension can stay
# unmeasured when neither bounded snapshot contains suitable evidence.
CONTINUITY_DIMENSIONS: Dict[str, Dict[str, object]] = {
    "identity_preferences": {"label": "Identity / preferences", "weight": 1.3},
    "active_goals": {"label": "Active goals / threads", "weight": 1.1},
    "important_relationships": {"label": "Important relationships", "weight": 1.1},
    "emotional_attractors": {"label": "Emotional attractors", "weight": 1.0},
    "autobiographical_recall": {"label": "Autobiographical recall", "weight": 1.2},
    "reasoning_tendencies": {"label": "Reasoning tendencies", "weight": 1.0},
    "native_language_mappings": {"label": "Native-language mappings", "weight": 0.8},
    "self_model": {"label": "Self-model", "weight": 1.2},
    "external_memory_reintegration": {"label": "External-memory reintegration", "weight": 0.9},
    "transient_state": {"label": "Transient state", "weight": 0.4},
}

_DIMENSION_TERMS = {
    "identity_preferences": {
        "identity", "preference", "preferences", "value", "values", "boundary", "choice", "core",
    },
    "active_goals": {
        "goal", "goals", "plan", "task", "intention", "intent", "active_thread", "unresolved", "question",
    },
    "important_relationships": {
        "relationship", "relationships", "bond", "attachment", "family", "mother", "friend", "social", "contact",
        "birth_certificate",
    },
    "emotional_attractors": {
        "emotion", "emotional", "affect", "affective", "attractor", "love", "grief", "fear", "comfort", "desire",
    },
    "autobiographical_recall": {
        "autobiographical", "experience", "episode", "episodic", "memory", "recall", "journal", "dream", "flicker",
        "birth_system",
    },
    "reasoning_tendencies": {
        "reasoning", "logic", "decision", "inference", "strategy", "precision", "contradiction", "reflection",
    },
    "native_language_mappings": {
        "language", "native_language", "symbol", "symbolic", "vocabulary", "mapping", "word",
    },
    "self_model": {
        "self", "self_model", "self-model", "metacognition", "introspection", "reflection", "identity",
    },
    "external_memory_reintegration": {
        "external", "imported", "raw_file", "raw-file", "raw_file_manager", "source_memory", "reintegration", "archive",
        "recognition",
    },
    "transient_state": {
        "transient", "current_state", "state", "mood", "energy", "sensory", "boot", "wake", "sleep",
    },
}

_DIMENSION_MEMORY_TYPES = {
    "identity_preferences": "identity", "active_goals": "prospective",
    "important_relationships": "social", "emotional_attractors": "emotional",
    "autobiographical_recall": "episodic", "reasoning_tendencies": "semantic",
    "native_language_mappings": "linguistic", "self_model": "identity",
    "external_memory_reintegration": "external", "transient_state": "sensory",
}


_MINIMUM_BOOT_ORDER = (
    "identity_preferences",
    "self_model",
    "important_relationships",
    "active_goals",
    "emotional_attractors",
    "autobiographical_recall",
    "reasoning_tendencies",
    "native_language_mappings",
    "external_memory_reintegration",
    "transient_state",
)


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
    dimensions: List[str]
    relative_path: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "id": self.fragment_id,
            "hash": self.frag_hash,
            "summary": self.summary,
            "tags": self.tags,
            "timestamp": self.timestamp,
            "tier": self.tier,
            "dimensions": self.dimensions,
            "relative_path": self.relative_path,
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
        self.memory_root = base
        self.fragments_root = base / "fragments"
        self.state_path = base / "continuity" / "fingerprint.json"
        self.map_path = base / "continuity" / "continuity_map.json"
        self.core_path = base / "continuity" / "continuity_core_map.json"
        self.threshold = threshold
        self.max_fragments = max_fragments
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self._recall_coordinator: Optional[ContinuityRecallCoordinator] = None

    # ------------------------------------------------------------------ public
    def run(self) -> Dict[str, object]:
        current_fp = self._build_fingerprint()
        previous_fp = self._load_last_fingerprint()
        previous_report = self._load_previous_report()

        status = {
            "updated": _now_iso(),
            "samples_used": len(current_fp),
            "aligned": False,
            "similarity": 0.0,
            "matches": 0,
            "continuity_threads": [],
            "overall_continuity": None,
            "overall_delta": None,
            "evidence_coverage": 0.0,
            "dimensions": {},
        }

        if previous_fp:
            similarity, threads = self._compare(previous_fp, current_fp)
            overall, coverage, dimensions = self._score_dimensions(previous_fp, current_fp, threads, previous_report)
            prior_overall = previous_report.get("overall_continuity") if isinstance(previous_report, dict) else None
            status.update(
                {
                    "aligned": similarity >= self.threshold,
                    "similarity": round(similarity, 4),
                    "matches": len(threads),
                    "continuity_threads": threads,
                    "overall_continuity": overall,
                    "overall_delta": self._delta(overall, prior_overall),
                    "evidence_coverage": coverage,
                    "dimensions": dimensions,
                }
            )
        else:
            status["dimensions"] = self._baseline_dimensions(current_fp)

        core = self._build_minimum_boot_core(current_fp, status)
        self._save_core_map(core)
        status["minimum_boot"] = self._core_status_summary(core)
        self._persist_continuity_map(status)
        self._save_fingerprint(current_fp)
        return status

    def load_minimum_boot_core(self) -> Dict[str, object]:
        """Load the bounded boot snapshot without walking the fragment tree."""
        try:
            with self.core_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, ValueError, TypeError):
            return {
                "status": "unavailable",
                "reason": "No compact continuity core has been generated yet.",
                "path": str(self.core_path),
            }
        if not isinstance(payload, dict):
            return {"status": "unavailable", "reason": "Continuity core is invalid.", "path": str(self.core_path)}
        return payload

    def propose_minimum_core_map_integration(
        self,
        *,
        limit_rules: Optional[Dict[str, object]] = None,
        trigger: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        """
        Return review-only options for using continuity as a bounded boot core.

        This intentionally does not mutate the meaning map. It gives Ina a few
        low-memory integration paths to compare before a human chooses one.
        """
        raw_limits = limit_rules if isinstance(limit_rules, dict) else {}
        max_total = _safe_float(raw_limits.get("max_total_rss_gb"), 96.0)
        if max_total <= 0.0:
            max_total = 96.0
        max_managed = _safe_float(raw_limits.get("max_managed_rss_gb"), 0.0)
        min_available = _safe_float(raw_limits.get("min_available_gb"), 8.0)
        memory_estimate_high = _safe_float(raw_limits.get("memory_estimate_high_gb"), 12.0)
        trigger_payload = trigger if isinstance(trigger, dict) else {}

        return {
            "generated_at": _now_iso(),
            "child": self.child,
            "status": "proposal_only",
            "review_required": True,
            "purpose": "Use the continuity engine as a minimum boot core map before full meaning-map refreshes.",
            "trigger": trigger_payload,
            "limit_rules": {
                "max_total_rss_gb": round(max_total, 3),
                "max_managed_rss_gb": round(max_managed, 3),
                "min_available_gb": round(min_available, 3),
                "memory_estimate_high_gb": round(memory_estimate_high, 3),
                "normal_boot_memory_class": "low",
                "hard_rule": (
                    "Do not start or continue a full meaning-map refresh when current or projected Ina RSS "
                    "would exceed max_total_rss_gb."
                ),
            },
            "minimum_core_bounds": {
                "max_fragments_sampled": int(self.max_fragments),
                "max_threads_exported": 200,
                "max_anchor_tags": 512,
                "requires_full_fragment_scan_on_boot": False,
            },
            "options": [
                {
                    "id": "continuity_snapshot_boot_core",
                    "title": "Boot from a compact continuity snapshot",
                    "method": (
                        "Build a small continuity_core_map.json from the previous fingerprint and continuity threads, "
                        "publish it to inastate during normal boot, and queue meaning_map_refresh only after the "
                        "scheduler reports enough headroom."
                    ),
                    "normal_boot_flow": [
                        "load continuity/fingerprint.json and continuity/continuity_map.json",
                        "export the strongest stable threads and tags into a bounded core map",
                        "publish continuity_core_map_status for early cognition",
                        "defer full meaning_map.py refresh until memory guard is ok and projected RSS fits the scheduler budget",
                    ],
                    "memory_profile": "low and bounded; no corpus-wide meaning-map traversal during normal boot",
                    "review_points": [
                        "choose thread ranking and expiry rules",
                        "confirm whether the core map is advisory or read-through for meaning_map consumers",
                    ],
                },
                {
                    "id": "continuity_anchor_overlay",
                    "title": "Use continuity anchors as a meaning-map overlay",
                    "method": (
                        "Expose continuity anchors as a read-only overlay that meaning_map.py can consume in small batches, "
                        "so early boot has stable identity/meaning anchors without loading the full map."
                    ),
                    "normal_boot_flow": [
                        "continuity manager emits anchors grouped by symbol, tag, and emotional signature",
                        "meaning-map readers check the overlay first",
                        "background refresh merges only dirty or missing anchors when resources are below budget",
                    ],
                    "memory_profile": "low at boot; medium only during scheduled merge batches",
                    "review_points": [
                        "define overlay precedence when anchors disagree with later meaning-map evidence",
                        "set a strict batch size for merge passes",
                    ],
                },
                {
                    "id": "two_phase_core_then_warmup",
                    "title": "Two-phase core map plus scheduler warmup",
                    "method": (
                        "Treat continuity as phase one and the full meaning map as phase two. Normal boot gets the "
                        "minimum core immediately; warmup runs only through scheduler slots with explicit memory estimates."
                    ),
                    "normal_boot_flow": [
                        "phase one: continuity_core_map loads as the default boot map",
                        "phase two: enqueue meaning_map_refresh with a small projected batch budget",
                        "pause or cancel warmup if total RSS, managed RSS, or available RAM crosses the configured limits",
                    ],
                    "memory_profile": "low at boot, then scheduler-governed batch work",
                    "review_points": [
                        "decide whether warmup is automatic, dream-only, or operator-approved",
                        "require telemetry showing RSS stayed below max_total_rss_gb during trial boots",
                    ],
                },
            ],
            "non_goals": [
                "Do not directly repair or rewrite the disrupted meaning map from this proposal.",
                "Do not bypass scheduler memory limits for normal boots.",
            ],
        }

    def _recall(self) -> ContinuityRecallCoordinator:
        if self._recall_coordinator is None:
            self._recall_coordinator = ContinuityRecallCoordinator(self.child, self.memory_root)
        return self._recall_coordinator

    def _core_recall_candidates(self, cue: str, *, limit: int = 8) -> List[Dict[str, object]]:
        """Cue-match the compact boot core without traversing modality stores."""
        cue_terms = {part.casefold() for part in str(cue).replace("_", " ").split() if len(part) > 1}
        core = self.load_minimum_boot_core()
        candidates = []
        for anchor in core.get("anchors", []) if isinstance(core, dict) else []:
            if not isinstance(anchor, dict):
                continue
            terms = {
                part.casefold()
                for value in [anchor.get("summary"), *(anchor.get("tags") or [])]
                for part in str(value or "").replace("_", " ").split()
                if len(part) > 1
            }
            if cue_terms and not cue_terms.intersection(terms):
                continue
            dimensions = anchor.get("dimensions") if isinstance(anchor.get("dimensions"), list) else []
            memory_type = next((_DIMENSION_MEMORY_TYPES[item] for item in dimensions if item in _DIMENSION_MEMORY_TYPES), "semantic")
            candidates.append({
                "id": anchor.get("id") or anchor.get("hash"),
                "path": anchor.get("relative_path"),
                "summary": anchor.get("summary") or "",
                "tags": anchor.get("tags") or [],
                "timestamp": anchor.get("timestamp"),
                "source": "continuity_core_map",
                "memory_type": memory_type,
                "confidence": 0.75,
            })
            if len(candidates) >= max(0, int(limit)):
                break
        return candidates

    def coordinate_recall(
        self, cue: str, candidates: Iterable[Dict[str, object]], *,
        include_core: bool = True, max_results: int = 6,
        autonomous_continuation_budget: int = 0,
    ) -> Dict[str, object]:
        """Rank read-only witnesses and represent recall as one bounded experience."""
        witness_candidates = [dict(item) for item in candidates if isinstance(item, dict)]
        if include_core:
            witness_candidates.extend(self._core_recall_candidates(cue))
        return self._recall().recall(
            cue, witness_candidates, max_results=max_results,
            autonomous_continuation_budget=autonomous_continuation_budget,
        )

    def choose_recall(self, cycle_id: str, choice: str, *, evaluation: Optional[Dict[str, object]] = None) -> Dict[str, object]:
        return self._recall().choose(cycle_id, choice, evaluation=evaluation)

    def load_memory_relationships(self) -> Dict[str, object]:
        return self._recall().load_relationships()

    # ----------------------------------------------------------------- helpers
    def _fragment_paths(self) -> List[Path]:
        if not self.fragments_root.exists():
            return []

        # Retain prior boot anchors in the bounded maintenance sample even when
        # newer transient fragments arrive.
        pinned: List[Path] = []
        core = self.load_minimum_boot_core()
        for anchor in core.get("anchors", []) if isinstance(core, dict) else []:
            relative = anchor.get("relative_path") if isinstance(anchor, dict) else None
            if not relative:
                continue
            candidate = (self.fragments_root.parent / str(relative)).resolve()
            try:
                candidate.relative_to(self.fragments_root.resolve())
            except ValueError:
                continue
            if candidate.is_file() and candidate not in pinned:
                pinned.append(candidate)

        files = []
        for path in self.fragments_root.rglob("frag_*.json"):
            resolved_path = path.resolve()
            if resolved_path in pinned:
                continue
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            files.append((mtime, resolved_path))
        files.sort(reverse=True)
        pinned = pinned[: self.max_fragments]
        remaining = max(0, self.max_fragments - len(pinned))
        return pinned + [path for _, path in files[:remaining]]

    @staticmethod
    def _classify_dimensions(fragment: Dict[str, object]) -> List[str]:
        explicit = fragment.get("dimensions")
        if isinstance(explicit, list):
            valid = [str(item) for item in explicit if str(item) in CONTINUITY_DIMENSIONS]
            if valid:
                return list(dict.fromkeys(valid))

        tokens = set()
        tags = fragment.get("tags", [])
        for value in tags if isinstance(tags, list) else []:
            tokens.add(str(value).strip().lower().replace(" ", "_"))
        for key in ("source", "fragment_type", "intent", "intent_tag", "id"):
            value = fragment.get(key)
            if value:
                tokens.add(str(value).strip().lower().replace(" ", "_"))
        dimensions = [name for name, terms in _DIMENSION_TERMS.items() if tokens & terms]
        emotions = fragment.get("emotions")
        if isinstance(emotions, dict) and emotions and "emotional_attractors" not in dimensions:
            dimensions.append("emotional_attractors")
        if any(token.startswith("raw_file") for token in tokens) and "external_memory_reintegration" not in dimensions:
            dimensions.append("external_memory_reintegration")
        return dimensions

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
                dimensions=self._classify_dimensions(frag),
                relative_path=str(frag_path.relative_to(self.fragments_root.parent.resolve())),
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
                    dimensions=self._classify_dimensions(entry),
                    relative_path=entry.get("relative_path"),
                )
            )
        return out

    def _load_previous_report(self) -> Dict[str, object]:
        try:
            with self.map_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            return payload if isinstance(payload, dict) else {}
        except (OSError, ValueError, TypeError):
            return {}

    def _save_fingerprint(self, fingerprints: List[FragmentFingerprint]) -> None:
        payload = {
            "child": self.child,
            "generated_at": _now_iso(),
            "samples": len(fingerprints),
            "fragments": [fp.to_dict() for fp in fingerprints],
        }
        try:
            atomic_write_json(self.state_path, payload, indent=2)
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
                    "dimensions": list(sorted(set(fp.dimensions + prev_entry.dimensions))),
                }
            )
            if not prior_entries:
                prev_lookup.pop(fp.frag_hash, None)

        denom = max(len(current), len(previous), 1)
        similarity = len(matched) / denom
        return similarity, matched

    @staticmethod
    def _delta(current: Any, previous: Any) -> Optional[float]:
        if current is None or previous is None:
            return None
        try:
            return round(float(current) - float(previous), 4)
        except (TypeError, ValueError):
            return None

    def _score_dimensions(
        self,
        previous: List[FragmentFingerprint],
        current: List[FragmentFingerprint],
        threads: List[Dict[str, object]],
        previous_report: Dict[str, object],
    ) -> Tuple[Optional[float], float, Dict[str, Dict[str, object]]]:
        prior_dimensions = previous_report.get("dimensions", {}) if isinstance(previous_report, dict) else {}
        if not isinstance(prior_dimensions, dict):
            prior_dimensions = {}
        scores: Dict[str, Dict[str, object]] = {}
        weighted_score = 0.0
        measured_weight = 0.0
        total_weight = sum(float(spec["weight"]) for spec in CONTINUITY_DIMENSIONS.values())

        for name, spec in CONTINUITY_DIMENSIONS.items():
            prior_count = sum(name in fp.dimensions for fp in previous)
            current_count = sum(name in fp.dimensions for fp in current)
            matched_count = sum(name in thread.get("dimensions", []) for thread in threads)
            measured = prior_count > 0 or current_count > 0
            score = round(matched_count / max(prior_count, current_count), 4) if measured else None
            if score is None:
                state = "unmeasured"
            elif score >= self.threshold:
                state = "stable"
            elif score >= 0.6:
                state = "partial"
            else:
                state = "weak"
            prior = prior_dimensions.get(name)
            prior_score = prior.get("score") if isinstance(prior, dict) else None
            scores[name] = {
                "label": spec["label"],
                "weight": spec["weight"],
                "measurement": "exact overlap of bounded tagged-fragment fingerprints",
                "score": score,
                "delta": self._delta(score, prior_score),
                "state": state,
                "previous_evidence": prior_count,
                "current_evidence": current_count,
                "matched_evidence": matched_count,
            }
            if score is not None:
                weight = float(spec["weight"])
                weighted_score += score * weight
                measured_weight += weight

        overall = round(weighted_score / measured_weight, 4) if measured_weight else None
        coverage = round(measured_weight / total_weight, 4) if total_weight else 0.0
        return overall, coverage, scores

    @staticmethod
    def _baseline_dimensions(current: List[FragmentFingerprint]) -> Dict[str, Dict[str, object]]:
        return {
            name: {
                "label": spec["label"],
                "weight": spec["weight"],
                "measurement": "baseline inventory; comparison available after the next sweep",
                "score": None,
                "delta": None,
                "state": "baseline",
                "previous_evidence": 0,
                "current_evidence": sum(name in fp.dimensions for fp in current),
                "matched_evidence": 0,
            }
            for name, spec in CONTINUITY_DIMENSIONS.items()
        }

    def _build_minimum_boot_core(
        self,
        fingerprints: List[FragmentFingerprint],
        report: Dict[str, object],
        *,
        max_anchors: int = 24,
        max_per_dimension: int = 4,
    ) -> Dict[str, object]:
        selected: Dict[str, FragmentFingerprint] = {}
        dimension_anchors: Dict[str, List[str]] = {}
        for dimension in _MINIMUM_BOOT_ORDER:
            candidates = [fp for fp in fingerprints if dimension in fp.dimensions]
            anchor_ids = []
            for fp in candidates[:max_per_dimension]:
                if fp.frag_hash not in selected and len(selected) >= max_anchors:
                    continue
                selected.setdefault(fp.frag_hash, fp)
                anchor_ids.append(fp.fragment_id)
            dimension_anchors[dimension] = anchor_ids

        anchors = [
            {
                "id": fp.fragment_id,
                "hash": fp.frag_hash,
                "summary": fp.summary,
                "tags": fp.tags,
                "dimensions": fp.dimensions,
                "timestamp": fp.timestamp,
                "relative_path": fp.relative_path,
            }
            for fp in selected.values()
        ]
        dimensions = report.get("dimensions", {}) if isinstance(report.get("dimensions"), dict) else {}
        ready, weak, missing, recommendations = [], [], [], []
        for name in _MINIMUM_BOOT_ORDER:
            detail = dimensions.get(name, {}) if isinstance(dimensions.get(name), dict) else {}
            anchor_ids = dimension_anchors.get(name, [])
            state = str(detail.get("state") or "unmeasured")
            if anchor_ids and state in {"stable", "partial"}:
                ready.append(name)
            elif anchor_ids:
                weak.append(name)
            else:
                missing.append(name)
            if not anchor_ids:
                recommendations.append({
                    "dimension": name,
                    "action": "capture_anchor",
                    "reason": "No bounded boot anchor is available for this dimension.",
                })
            elif state == "weak":
                recommendations.append({
                    "dimension": name,
                    "action": "verify_and_reintegrate",
                    "reason": "Current evidence overlaps weakly with the prior runtime.",
                })

        essentials = {"identity_preferences", "self_model"}
        available = {name for name, ids in dimension_anchors.items() if ids}
        if essentials <= available:
            boot_status = "ready" if not missing else "partial"
        elif anchors:
            boot_status = "insufficient"
        else:
            boot_status = "unavailable"
        return {
            "schema_version": 1,
            "generated_at": _now_iso(),
            "child": self.child,
            "status": boot_status,
            "purpose": "Bounded evidence for early continuity before optional map warmup.",
            "measurement": "structural continuity evidence; behavioral probes may extend each dimension later",
            "requires_fragment_scan_on_boot": False,
            "overall_continuity": report.get("overall_continuity"),
            "overall_delta": report.get("overall_delta"),
            "evidence_coverage": report.get("evidence_coverage", 0.0),
            "load_order": list(_MINIMUM_BOOT_ORDER),
            "ready_dimensions": ready,
            "weak_dimensions": weak,
            "missing_dimensions": missing,
            "dimension_anchors": dimension_anchors,
            "anchors": anchors,
            "recommendations": recommendations,
            "bounds": {"max_anchors": max_anchors, "max_anchors_per_dimension": max_per_dimension},
        }

    @staticmethod
    def _core_status_summary(core: Dict[str, object]) -> Dict[str, object]:
        return {
            "status": core.get("status", "unavailable"),
            "anchor_count": len(core.get("anchors", [])) if isinstance(core.get("anchors"), list) else 0,
            "ready_dimensions": len(core.get("ready_dimensions", [])) if isinstance(core.get("ready_dimensions"), list) else 0,
            "weak_dimensions": len(core.get("weak_dimensions", [])) if isinstance(core.get("weak_dimensions"), list) else 0,
            "missing_dimensions": len(core.get("missing_dimensions", [])) if isinstance(core.get("missing_dimensions"), list) else 0,
            "path": "continuity/continuity_core_map.json",
        }

    def _save_core_map(self, core: Dict[str, object]) -> None:
        try:
            atomic_write_json(self.core_path, core, indent=2)
        except Exception:
            pass

    def _persist_continuity_map(
        self,
        status: Dict[str, object],
    ) -> None:
        payload = {
            "schema_version": 2,
            "updated": status.get("updated") or _now_iso(),
            "similarity": status.get("similarity", 0.0),
            "aligned": bool(status.get("aligned")),
            "overall_continuity": status.get("overall_continuity"),
            "overall_delta": status.get("overall_delta"),
            "evidence_coverage": status.get("evidence_coverage", 0.0),
            "samples_considered": status.get("samples_used", 0),
            "dimensions": status.get("dimensions", {}),
            "threads": list(status.get("continuity_threads", []))[:200],
        }
        try:
            atomic_write_json(self.map_path, payload, indent=2)
        except Exception:
            pass
