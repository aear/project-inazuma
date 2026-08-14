"""Cross-memory recall arbitration for Continuity Engine.

Continuity coordinates a federation of read-only modality witnesses. It stores
links and rankings, never rewrites the traces owned by those witnesses.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping
import uuid

from experience_engine import CHOICES, ExperienceCycleEngine
from io_utils import atomic_write_json, load_json_dict


SCHEMA = "ina.continuity_recall/V2"
MEMORY_TYPES = (
    "episodic", "semantic", "procedural", "emotional", "sensory",
    "social", "identity", "linguistic", "prospective", "external",
)
TYPE_RELATIONSHIPS = {
    frozenset(("episodic", "semantic")): ("consolidates", 0.16),
    frozenset(("episodic", "emotional")): ("colours", 0.14),
    frozenset(("episodic", "identity")): ("situates_self", 0.16),
    frozenset(("episodic", "social")): ("situates_relationship", 0.14),
    frozenset(("semantic", "linguistic")): ("expresses", 0.13),
    frozenset(("procedural", "sensory")): ("grounds_action", 0.15),
    frozenset(("prospective", "identity")): ("continues_intent", 0.13),
    frozenset(("external", "semantic")): ("reintegrates", 0.10),
}
_TOKEN_RE = re.compile(r"[\w'-]+", re.UNICODE)
_STOPWORDS = {"a", "an", "and", "are", "be", "for", "i", "in", "is", "it", "of", "the", "to", "was"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tokens(value: Any) -> set[str]:
    return {token.casefold() for token in _TOKEN_RE.findall(str(value or "")) if token.casefold() not in _STOPWORDS}


def _safe_unit(value: Any, default: float = 0.5) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        number = default
    return max(0.0, min(1.0, number))


def _distribution(rows: Iterable[Mapping[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        label = str(row.get(key) or "unknown")[:160]
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items()))


def _diversity(counts: Mapping[str, Any]) -> dict[str, Any]:
    values = [max(0, int(value or 0)) for value in counts.values()]
    total = sum(values)
    if not total:
        return {"score": None, "dominance": None, "dominant": None, "count": 0}
    shares = [value / total for value in values]
    score = 1.0 - sum(share * share for share in shares)
    dominant = max(counts, key=lambda label: int(counts[label] or 0))
    return {"score": round(score, 4), "dominance": round(max(shares), 4),
            "dominant": str(dominant), "count": total}


def _selection_skew(candidate: Mapping[str, int], selected: Mapping[str, int]) -> dict[str, Any]:
    candidate_total = max(1, sum(candidate.values()))
    selected_total = max(1, sum(selected.values()))
    differences = {
        label: round(selected.get(label, 0) / selected_total - candidate.get(label, 0) / candidate_total, 4)
        for label in sorted(set(candidate) | set(selected))
    }
    strongest = max(differences, key=lambda label: abs(differences[label])) if differences else None
    return {"strength": round(abs(differences.get(strongest, 0.0)), 4) if strongest else 0.0,
            "strongest_dimension": strongest, "share_deltas": differences}


def infer_memory_type(candidate: Mapping[str, Any]) -> str:
    explicit = str(candidate.get("memory_type") or "").strip().lower()
    if explicit in MEMORY_TYPES:
        return explicit
    text = " ".join(
        [str(candidate.get("source") or ""), str(candidate.get("summary") or "")]
        + [str(tag) for tag in candidate.get("tags", []) if tag is not None]
    ).casefold()
    rules = (
        ("identity", ("identity", "self_model", "preference")),
        ("prospective", ("goal", "plan", "intent", "question")),
        ("social", ("relationship", "social", "friend", "family")),
        ("emotional", ("emotion", "affect", "feeling", "mood")),
        ("procedural", ("procedural", "skill", "motor", "how_to")),
        ("sensory", ("sensory", "vision", "audio", "touch", "proprio")),
        ("linguistic", ("language", "word", "symbol", "utterance")),
        ("external", ("external", "archive", "raw_file", "self_read")),
        ("episodic", ("experience", "episode", "event", "autobiographical", "dream")),
    )
    for memory_type, markers in rules:
        if any(marker in text for marker in markers):
            return memory_type
    return "semantic"


def _reference(candidate: Mapping[str, Any]) -> dict[str, str]:
    identifier = candidate.get("id") or candidate.get("event_id") or candidate.get("fragment_id")
    path = candidate.get("path") or candidate.get("reference_path")
    if not identifier and not path:
        raise ValueError("each recall witness needs an id or path")
    result = {"id": str(identifier)} if identifier else {}
    if path:
        result["path"] = str(path)
    result["kind"] = infer_memory_type(candidate)
    return result


class ContinuityRecallCoordinator:
    def __init__(
        self, child: str, memory_root: Path | str, *,
        experience_engine: ExperienceCycleEngine | None = None,
    ) -> None:
        self.child = str(child)
        self.memory_root = Path(memory_root)
        self.continuity_root = self.memory_root / "continuity"
        self.relationship_path = self.continuity_root / "memory_relationships.json"
        self.actions_root = self.continuity_root / "recall_actions"
        self.continuity_root.mkdir(parents=True, exist_ok=True)
        self.actions_root.mkdir(parents=True, exist_ok=True)
        self.experience_engine = experience_engine or ExperienceCycleEngine(
            self.child, root_path=self.memory_root / "experiences" / "cycles",
            enable_hot=False,
        )

    def _normalise(self, candidate: Mapping[str, Any], cue_terms: set[str]) -> dict[str, Any]:
        reference = _reference(candidate)
        summary = " ".join(str(candidate.get("summary") or candidate.get("narrative") or "").split())[:420]
        tags = [str(tag)[:80] for tag in candidate.get("tags", []) if tag is not None][:12]
        terms = _tokens(summary) | {term for tag in tags for term in _tokens(tag)}
        overlap = len(terms & cue_terms) / max(1, len(cue_terms))
        confidence = _safe_unit(candidate.get("confidence", candidate.get("score", 0.5)) or 0.5)
        return {
            "reference": reference,
            "source": str(candidate.get("source") or "unknown")[:160],
            "memory_type": reference["kind"],
            "summary": summary,
            "tags": tags,
            "terms": terms,
            "source_confidence": round(confidence, 4),
            "recency": str(candidate.get("timestamp") or candidate.get("updated_at") or "")[:64],
            "causal_references": [str(item) for item in candidate.get("causal_references", []) if item][:8],
            "cue_overlap": overlap,
            "original": dict(candidate),
        }

    def _relationship_support(self, row: Mapping[str, Any], rows: list[dict[str, Any]]) -> tuple[float, list[dict[str, Any]]]:
        support = 0.0
        links = []
        for other in rows:
            if other is row or other["memory_type"] == row["memory_type"]:
                continue
            relation = TYPE_RELATIONSHIPS.get(frozenset((row["memory_type"], other["memory_type"])))
            if relation is None:
                continue
            shared = sorted(row["terms"] & other["terms"])
            causal = bool(set(row["causal_references"]) & set(other["causal_references"]))
            if not shared and not causal:
                continue
            name, weight = relation
            support += weight
            links.append({
                "from": row["reference"].get("id") or row["reference"].get("path"),
                "to": other["reference"].get("id") or other["reference"].get("path"),
                "relation": name, "weight": weight, "shared_terms": shared[:6],
            })
        return min(0.35, support), links[:12]

    def arbitrate(self, cue: str, candidates: Iterable[Mapping[str, Any]], *, max_results: int = 6) -> dict[str, Any]:
        cue_terms = _tokens(cue)
        rows = [self._normalise(candidate, cue_terms) for candidate in candidates if isinstance(candidate, Mapping)][:64]
        all_links = []
        for row in rows:
            relationship_support, links = self._relationship_support(row, rows)
            all_links.extend(links)
            row["relationship_support"] = round(relationship_support, 4)
            row["recall_score"] = round(min(1.0, 0.68 * row["cue_overlap"] + 0.17 * row["source_confidence"] + relationship_support), 4)
        rows.sort(key=lambda item: (-item["recall_score"], item["memory_type"], item["source"], str(item["reference"])))

        limit = max(1, min(12, int(max_results)))
        selected: list[dict[str, Any]] = []
        seen_types: set[str] = set()
        seen_sources: dict[str, int] = {}
        # First pass gives independent witness types a voice; second fills by score.
        for diversity_pass in (True, False):
            for row in rows:
                if row in selected or len(selected) >= limit:
                    continue
                if diversity_pass and row["memory_type"] in seen_types:
                    continue
                if seen_sources.get(row["source"], 0) >= 2:
                    continue
                selected.append(row)
                seen_types.add(row["memory_type"])
                seen_sources[row["source"]] = seen_sources.get(row["source"], 0) + 1
            if len(selected) >= limit:
                break

        now = _now()
        prior = load_json_dict(self.relationship_path)
        link_index: dict[tuple[str, str, str], dict[str, Any]] = {}
        for link in prior.get("links", []) if isinstance(prior.get("links"), list) else []:
            if isinstance(link, dict):
                link_index[(str(link.get("from")), str(link.get("to")), str(link.get("relation")))] = dict(link)
        for link in all_links:
            key = (str(link.get("from")), str(link.get("to")), str(link.get("relation")))
            previous = link_index.get(key, {})
            merged = dict(link)
            merged.update(first_seen=previous.get("first_seen") or now, last_seen=now,
                          observation_count=int(previous.get("observation_count", 0) or 0) + 1)
            link_index[key] = merged

        witnesses = prior.get("witnesses", {}) if isinstance(prior.get("witnesses"), dict) else {}
        for row in rows:
            witness_id = str(row["reference"].get("id") or row["reference"].get("path"))
            previous = witnesses.get(witness_id, {}) if isinstance(witnesses.get(witness_id), dict) else {}
            witnesses[witness_id] = {
                "reference": row["reference"], "source": row["source"],
                "memory_type": row["memory_type"], "confidence": row["source_confidence"],
                "recency": row["recency"], "causal_references": row["causal_references"],
                "first_seen": previous.get("first_seen") or now, "last_seen": now,
                "observation_count": int(previous.get("observation_count", 0) or 0) + 1,
            }

        candidate_types = _distribution(rows, "memory_type")
        selected_types = _distribution(selected, "memory_type")
        candidate_sources = _distribution(rows, "source")
        selected_sources = _distribution(selected, "source")
        arbitration_summary = {
            "timestamp": now, "candidate_count": len(rows), "selected_count": len(selected),
            "candidate_memory_types": candidate_types, "selected_memory_types": selected_types,
            "candidate_sources": candidate_sources, "selected_sources": selected_sources,
            "candidate_type_diversity": _diversity(candidate_types),
            "selected_type_diversity": _diversity(selected_types),
            "candidate_source_diversity": _diversity(candidate_sources),
            "selected_source_diversity": _diversity(selected_sources),
            "memory_type_selection_skew": _selection_skew(candidate_types, selected_types),
            "source_selection_skew": _selection_skew(candidate_sources, selected_sources),
        }
        history = prior.get("recall_history", []) if isinstance(prior.get("recall_history"), list) else []
        relationship_payload = {
            "schema": SCHEMA, "updated_at": now, "witness_model": "federation_of_witnesses",
            "modality_store_mutation_allowed": False,
            "types_present": sorted({row["memory_type"] for row in rows}),
            "links": sorted(link_index.values(), key=lambda link: str(link.get("last_seen") or ""), reverse=True)[:512],
            "witnesses": dict(list(sorted(witnesses.items(), key=lambda item: str(item[1].get("last_seen") or ""), reverse=True))[:512]),
            "latest_arbitration": arbitration_summary, "recall_history": (history + [arbitration_summary])[-64:],
            "bounds": {"candidates_per_recall": 64, "links": 512, "witnesses": 512, "recall_history": 64},
        }
        atomic_write_json(self.relationship_path, relationship_payload, indent=2, ensure_ascii=False)
        return {"cue": str(cue)[:1000], "candidate_count": len(rows), "selected": selected, "relationships": relationship_payload}

    def _persist_action(self, plan_id: str, cycle_id: str, plan: Mapping[str, Any]) -> Path:
        path = self.actions_root / f"{plan_id}.json"
        selected = []
        for row in plan.get("selected", []):
            selected.append({key: row[key] for key in (
                "reference", "source", "memory_type", "summary", "tags", "source_confidence",
                "recency", "causal_references", "cue_overlap", "relationship_support", "recall_score",
            )})
        atomic_write_json(path, {
            "schema": SCHEMA, "plan_id": plan_id, "cycle_id": cycle_id, "cue": plan.get("cue"),
            "candidate_count": plan.get("candidate_count", 0), "selected": selected,
            "created_at": _now(), "source_traces_mutated": False,
        }, indent=2, ensure_ascii=False)
        return path

    def recall(
        self, cue: str, candidates: Iterable[Mapping[str, Any]], *, max_results: int = 6,
        autonomous_continuation_budget: int = 0,
    ) -> dict[str, Any]:
        plan = self.arbitrate(cue, candidates, max_results=max_results)
        plan_id = f"recall_plan_{uuid.uuid4().hex}"
        action_path = self.actions_root / f"{plan_id}.json"
        cycle = self.experience_engine.start_cycle(
            f"Recall: {str(cue)[:800]}", domain="continuity_recall",
            payload_references=[{"id": plan_id, "path": str(action_path), "kind": "recall_plan"}],
            autonomous_continuation_budget=autonomous_continuation_budget,
        )
        action_path = self._persist_action(plan_id, cycle["cycle_id"], plan)
        attempt = self.experience_engine.complete_attempt(
            cycle["cycle_id"], attempt_reference={"id": cycle["cycle_id"], "path": str(action_path), "kind": "recall_ranking"},
            observation_references=[row["reference"] for row in plan["selected"]],
            evaluation={
                "candidate_count": plan["candidate_count"], "surfaced_count": len(plan["selected"]),
                "memory_types": sorted({row["memory_type"] for row in plan["selected"]}),
                "sources": sorted({row["source"] for row in plan["selected"]})[:16],
                "source_traces_mutated": False,
            }, choice=None,
        )
        returned = []
        for row in plan["selected"]:
            candidate = dict(row["original"])
            candidate["memory_type"] = row["memory_type"]
            candidate["continuity_recall"] = {
                "cycle_id": cycle["cycle_id"], "score": row["recall_score"],
                "relationship_support": row["relationship_support"],
                "source_confidence": row["source_confidence"],
            }
            returned.append(candidate)
        return {
            "schema": SCHEMA, "cycle_id": cycle["cycle_id"], "attempt_id": attempt["attempt_id"],
            "stage": "evaluation", "selected": returned, "next_choices": list(CHOICES),
            "relationship_path": str(self.relationship_path), "action_path": str(action_path),
            "source_traces_mutated": False,
        }

    def choose(self, cycle_id: str, choice: str, *, evaluation: Mapping[str, Any] | None = None) -> dict[str, Any]:
        return self.experience_engine.record_choice(cycle_id, choice, evaluation=evaluation)

    def load_relationships(self) -> dict[str, Any]:
        return load_json_dict(self.relationship_path)


__all__ = ["SCHEMA", "MEMORY_TYPES", "TYPE_RELATIONSHIPS", "ContinuityRecallCoordinator", "infer_memory_type"]
