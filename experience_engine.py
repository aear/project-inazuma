"""Optional, domain-neutral Experience Cycles layered over experience storage.

Domain data stays with its owning tool. The cycle stores only stable IDs/paths;
Hindsight is the designated owner of later lesson extraction.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping
import uuid

from io_utils import atomic_write_json, load_json_dict
from experience_cycle_storage import CycleTierPolicy
from experience_cycle_index import ExperienceCycleIndex


SCHEMA = "ina.experience_cycle/V2"
CHOICES = ("keep", "revise", "revisit", "stop")
MAX_AUTONOMOUS_CONTINUATIONS = 32


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _identifier(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def normalize_references(references: Iterable[str | Mapping[str, Any]] | None) -> list[dict[str, str]]:
    result = []
    for reference in references or ():
        if isinstance(reference, str):
            item = {"id": reference}
        elif isinstance(reference, Mapping):
            item = {key: str(reference[key]) for key in ("id", "path", "kind") if reference.get(key) is not None}
        else:
            raise TypeError("payload references must be IDs, paths, or reference mappings")
        if not item.get("id") and not item.get("path"):
            raise ValueError("each payload reference needs an id or path")
        result.append(item)
    return result[:64]


def new_cycle(
    intent: str, *, domain: str, payload_references: Iterable[str | Mapping[str, Any]] | None = None,
    parent_cycle_id: str | None = None, autonomous_continuation_budget: int = 0,
) -> dict[str, Any]:
    budget = int(autonomous_continuation_budget)
    if budget < 0 or budget > MAX_AUTONOMOUS_CONTINUATIONS:
        raise ValueError(f"autonomous continuation budget must be 0..{MAX_AUTONOMOUS_CONTINUATIONS}")
    stamp = _now()
    return {
        "schema": SCHEMA, "cycle_id": _identifier("cycle"), "parent_cycle_id": parent_cycle_id,
        "domain": str(domain)[:80], "intent": str(intent)[:1000],
        "payload_references": normalize_references(payload_references),
        "stage": "intent", "attempt_ids": [], "autonomous_continuation_budget": budget,
        "autonomous_continuations_used": 0, "next_choices": list(CHOICES),
        "may_pause": True, "may_stop": True, "lesson_owner": "HindsightTransformer",
        "created_at": stamp, "updated_at": stamp,
    }


def _bounded_metadata(value: Mapping[str, Any] | None, max_bytes: int = 16 * 1024) -> dict[str, Any]:
    payload = dict(value or {})
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(encoded) > max_bytes:
        raise ValueError(f"cycle metadata exceeds {max_bytes} bytes; store domain payload externally and reference it")
    return payload


def new_attempt(
    cycle_id: str, *, attempt_reference: str | Mapping[str, Any],
    observation_references: Iterable[str | Mapping[str, Any]] | None = None,
    evaluation: Mapping[str, Any] | None = None, choice: str | None = None,
) -> dict[str, Any]:
    selected = None if choice is None else str(choice).lower()
    if selected is not None and selected not in CHOICES:
        raise ValueError(f"choice must be one of: {', '.join(CHOICES)}")
    return {
        "schema": SCHEMA, "attempt_id": _identifier("attempt"), "cycle_id": str(cycle_id),
        "attempt_reference": normalize_references([attempt_reference])[0],
        "observation_references": normalize_references(observation_references),
        "evaluation": _bounded_metadata(evaluation), "choice": selected, "created_at": _now(),
    }


class ExperienceCycleEngine:
    """Persist immutable attempts with a quota-bounded NVMe hot tier."""

    def __init__(
        self, child: str = "Inazuma_Yagami", base_path: Path | str = "AI_Children", *,
        root_path: Path | str | None = None, enable_hot: bool | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        self.child = str(child)
        base = Path(base_path)
        self.root = (
            Path(root_path)
            if root_path is not None
            else base / self.child / "memory" / "experiences" / "cycles"
        )
        if enable_hot is None:
            enable_hot = root_path is None and base == Path("AI_Children")
        self.storage = CycleTierPolicy(self.child, self.root, config=config, enable_hot=bool(enable_hot))
        self.index = ExperienceCycleIndex(
            self.child, self.root, config=self.storage.config, enable_fast=bool(enable_hot),
        )
        self.cycles = self.root / "manifests"
        self.attempts = self.root / "attempts"
        self.decisions = self.root / "decisions"
        for directory in (self.cycles, self.attempts, self.decisions):
            directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _directory(root: Path, kind: str) -> Path:
        path = root / kind
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _manifest_path(self, cycle_id: str) -> Path:
        for root in self.storage.roots_for_read():
            path = root / "manifests" / f"{cycle_id}.json"
            if path.is_file():
                return path
        raise FileNotFoundError(cycle_id)

    def _write_json(self, root: Path, kind: str, identifier: str, payload: Mapping[str, Any], *, immutable: bool = False) -> Path:
        directory = self._directory(root, kind)
        path = directory / f"{identifier}.json"
        if immutable:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(dict(payload), handle, ensure_ascii=False, indent=2)
                handle.write("\n")
        else:
            atomic_write_json(path, dict(payload), indent=2, ensure_ascii=False)
        self.storage.record_write(root, path.stat().st_size)
        return path

    def _rewrite_manifest(self, cycle: Mapping[str, Any]) -> Path:
        path = self._manifest_path(str(cycle["cycle_id"]))
        atomic_write_json(path, dict(cycle), indent=2, ensure_ascii=False)
        self.storage.record_write(path.parent.parent, path.stat().st_size)
        self.index.upsert(cycle, path)
        return path

    def start_cycle(self, intent: str, **kwargs: Any) -> dict[str, Any]:
        cycle = new_cycle(intent, **kwargs)
        root = self.storage.choose_write_root()
        manifest = self._write_json(root, "manifests", cycle["cycle_id"], cycle)
        self.index.upsert(cycle, manifest)
        return cycle

    def load_cycle(self, cycle_id: str) -> dict[str, Any]:
        cycle = load_json_dict(self._manifest_path(cycle_id))
        if not cycle:
            raise FileNotFoundError(cycle_id)
        return cycle

    def complete_attempt(self, cycle_id: str, **kwargs: Any) -> dict[str, Any]:
        cycle = self.load_cycle(cycle_id)
        if cycle.get("attempt_ids"):
            raise RuntimeError("a cycle contains exactly one attempt; start a linked revision")
        attempt = new_attempt(cycle_id, **kwargs)
        root = self.storage.choose_write_root()
        self._write_json(root, "attempts", attempt["attempt_id"], attempt, immutable=True)
        cycle["attempt_ids"] = [attempt["attempt_id"]]
        cycle["stage"] = attempt["choice"] or "evaluation"
        cycle["last_choice"] = attempt["choice"]
        cycle["updated_at"] = _now()
        self._rewrite_manifest(cycle)
        return attempt

    def record_choice(self, cycle_id: str, choice: str, *, evaluation: Mapping[str, Any] | None = None) -> dict[str, Any]:
        cycle = self.load_cycle(cycle_id)
        selected = str(choice).lower()
        if selected not in CHOICES:
            raise ValueError(f"choice must be one of: {', '.join(CHOICES)}")
        if not cycle.get("attempt_ids"):
            raise RuntimeError("an attempt must be observed and evaluated before choosing")
        if cycle.get("decision_id"):
            raise RuntimeError("this cycle already has an immutable decision")
        decision = {
            "schema": SCHEMA, "decision_id": _identifier("decision"),
            "cycle_id": str(cycle_id), "choice": selected,
            "evaluation": _bounded_metadata(evaluation), "created_at": _now(),
        }
        root = self.storage.choose_write_root()
        self._write_json(root, "decisions", decision["decision_id"], decision, immutable=True)
        cycle.update({"decision_id": decision["decision_id"], "last_choice": selected, "stage": selected, "updated_at": _now()})
        self._rewrite_manifest(cycle)
        return decision

    def continue_cycle(
        self, parent_cycle_id: str, *, choice: str, intent: str,
        autonomous: bool = False, payload_references: Iterable[str | Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        selected = str(choice).lower()
        if selected not in {"revise", "revisit"}:
            raise ValueError("only revise or revisit creates a child cycle")
        parent = self.load_cycle(parent_cycle_id)
        if parent.get("last_choice") != selected:
            raise PermissionError("parent cycle must explicitly choose this continuation")
        budget = int(parent.get("autonomous_continuation_budget", 0))
        used = int(parent.get("autonomous_continuations_used", 0))
        if autonomous and used >= budget:
            raise PermissionError("autonomous continuation needs a remaining explicit budget")
        if autonomous:
            parent["autonomous_continuations_used"] = used + 1
            parent["updated_at"] = _now()
            self._rewrite_manifest(parent)
        return self.start_cycle(
            intent, domain=parent.get("domain", "unknown"), payload_references=payload_references,
            parent_cycle_id=parent_cycle_id,
            autonomous_continuation_budget=max(0, budget - (1 if autonomous else 0)),
        )

    def drain_hot_tier(self, *, max_files: int = 256, max_bytes: int = 16 * 1024 * 1024) -> dict[str, Any]:
        result = self.storage.drain_to_durable(max_files=max_files, max_bytes=max_bytes)
        for relative in result.get("moved_paths", []):
            path = self.root / relative
            if path.parent.name != "manifests" or not path.is_file():
                continue
            cycle = load_json_dict(path)
            if cycle:
                self.index.upsert(cycle, path)
        return result

    def recent_cycles(self, *, limit: int = 50, domain: str | None = None) -> list[dict[str, Any]]:
        return self.index.recent(limit=limit, domain=domain)


__all__ = ["SCHEMA", "CHOICES", "MAX_AUTONOMOUS_CONTINUATIONS", "ExperienceCycleEngine", "new_cycle", "new_attempt", "normalize_references"]
