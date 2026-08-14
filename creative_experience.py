"""Bounded, event-driven experiment records for Ina's creative tools."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
import uuid

from io_utils import atomic_write_json, load_json_dict
from experience_engine import CHOICES, SCHEMA, new_attempt, new_cycle


STAGES = ("intent", "attempt", "observation", "evaluation", "keep", "revise", "revisit", "stop")
NEXT_CHOICES = CHOICES


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def experience_path(child: str, root: Path | str = ".") -> Path:
    return Path(root) / "AI_Children" / str(child) / "memory" / "creative_experiences.json"


def begin_experience(tool: str, intention: str, *, hypothesis: str = "", source: str = "chosen") -> dict[str, Any]:
    tool_name = str(tool).strip().lower()
    if tool_name not in {"daw", "drawing", "motor"}:
        raise ValueError("tool must be daw, drawing, or motor")
    cycle = new_cycle(
        intention, domain=tool_name,
        payload_references=[{"id": f"{tool_name}_intent", "kind": str(source)[:80]}],
    )
    cycle.update({
        "session_id": cycle["cycle_id"], "tool": tool_name,
        "intention": cycle["intent"],
        "hypothesis": str(hypothesis)[:500], "source": str(source)[:80],
        "started_at": cycle["created_at"], "experiment_count": 0, "experiments": [],
    })
    return cycle

def record_experiment(session: Mapping[str, Any], variation: Mapping[str, Any] | str, *, observation: str = "") -> dict[str, Any]:
    result = dict(session)
    reference = variation if isinstance(variation, str) else {key: variation[key] for key in ("id", "path", "kind") if variation.get(key) is not None}
    if not reference:
        raise ValueError("creative variation must be referenced by id or path")
    attempt = new_attempt(
        str(result.get("cycle_id") or result.get("session_id")),
        attempt_reference=reference,
        observation_references=([{"id": str(observation), "kind": "pending_observation"}] if observation else None),
        evaluation={"status": "awaiting_inspection"}, choice=None,
    )
    experiments = [dict(item) for item in result.get("experiments", []) if isinstance(item, Mapping)][-31:]
    experiments.append(attempt)
    result.update({"stage": "observation", "experiments": experiments, "attempt_ids": [attempt["attempt_id"]], "experiment_count": len(experiments), "updated_at": _now()})
    return result

def choose_next(session: Mapping[str, Any], choice: str, *, reflection: str = "") -> dict[str, Any]:
    selected = str(choice).strip().lower()
    if selected not in NEXT_CHOICES:
        raise ValueError(f"choice must be one of: {', '.join(NEXT_CHOICES)}")
    result = dict(session)
    result.update({"stage": selected, "last_choice": selected, "reflection": str(reflection)[:2000], "updated_at": _now()})
    return result


def save_experience(child: str, session: Mapping[str, Any], *, root: Path | str = ".", history_limit: int = 64) -> Path:
    path = experience_path(child, root)
    ledger = load_json_dict(path)
    history = [item for item in ledger.get("sessions", []) if isinstance(item, dict)]
    session_id = str(session.get("session_id") or "")
    history = [item for item in history if item.get("session_id") != session_id]
    history.append(dict(session))
    atomic_write_json(path, {"schema": SCHEMA, "sessions": history[-max(1, int(history_limit)):], "updated_at": _now()}, indent=2, ensure_ascii=False)
    return path


def experience_command_fields(session: Mapping[str, Any]) -> dict[str, Any]:
    """Small metadata copied into one creative command; it never schedules follow-up work."""
    return {
        "creative_experience": {
            "schema": SCHEMA,
            "session_id": session.get("session_id"),
            "stage": session.get("stage"),
            "intention": session.get("intention"),
            "hypothesis": session.get("hypothesis"),
            "next_choices": list(NEXT_CHOICES),
            "may_pause": True,
            "may_stop": True,
        }
    }


__all__ = ["SCHEMA", "STAGES", "NEXT_CHOICES", "begin_experience", "record_experiment", "choose_next", "save_experience", "experience_path", "experience_command_fields"]
