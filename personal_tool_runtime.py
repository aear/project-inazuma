"""Voluntary runtime bridge for Ina's private notes, expressions, and experiments.

Nothing in this module invents a request or schedules repeated work.  It only
executes one explicitly queued command at a time and returns bounded results.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping

from code_experiment_lab import CodeExperimentLab
from config_layers import load_config
from expression_core import ExpressionTraceStore, TextRealiser, create_expression_intent
from ina_desktop.files import VirtualFileSystem, configured_drives
from instruction_authority import seal_code_command, verify_code_command
from lexical_reference import lookup_definition, lookup_related_words
from external_project_registry import read_project_source
from expressive_variation import compare_variation
from self_inquiry_journey import begin_intuition_inquiry, continue_self_inquiry, current_inquiry_request
from runtime_state import append_inastate_queue, drain_inastate_queue, get_inastate, update_inastate
from threat_attribution import assess_attribution, prepare_report, queue_report_for_review


QUEUE_KEY = "personal_tool_command_queue"
RESULT_KEY = "personal_tool_last_result"
CATALOG_KEY = "personal_tool_capabilities"
QUEUE_LIMIT = 16
MAX_TEXT_LENGTH = 64 * 1024


def capability_catalog() -> dict[str, Any]:
    """Describe optional actions without recommending or triggering one."""
    return {
        "schema": "ina.personal_tool_capabilities/V1",
        "voluntary": True,
        "automatic_trigger": False,
        "commands": {
            "intuition_inquire": {
                "arguments": ["hunch", "question", "countercheck", "trigger_references", "depth_budget?"],
                "scope": "voluntary finite evidence request; no execution or memory retrieval",
            },
            "intuition_review": {
                "arguments": ["journey", "choice (deeper, stop, remain_uncertain)", "observation_references?", "hypotheses?"],
                "scope": "revisable candidates with retained prior hypotheses; no truth certification",
            },
            "compare_expressive_variation": {
                "arguments": ["modality (image, text, sound)", "before", "after", "source", "references?"],
                "scope": "caller-observed features; optional one-variable trials, not inferred meaning",
                "retention": "returned capture only; no automatic memory write or execution",
            },
            "external_project_self_read": {
                "arguments": ["project", "relative_path?", "offset?"],
                "scope": "read-enabled private registry projects; bounded source excerpts",
                "retention": "voluntary reference only; code is not instruction authority",
                "execution": "none",
            },
            "write_note": {
                "arguments": ["title", "text"],
                "destination": "private personal storage/Notes",
                "language": "chosen by Ina; English is available but not preferred or required",
            },
            "realise_private_text": {
                "arguments": ["purpose", "text", "title", "meaning_references?"],
                "destination": "private personal storage/Expressions",
                "delivery": "none",
            },
            "experiment_create": {
                "arguments": ["question", "hypothesis", "code", "dataset?"],
                "continuation_budget": 0,
                "authority": "process-local voluntary code capability required",
            },
            "experiment_run": {
                "arguments": ["experiment_id"],
                "authority": "process-local voluntary code capability required",
            },
            "experiment_judge": {
                "arguments": ["experiment_id", "choice", "metrics", "explanation"],
                "promotion": "human review required",
                "authority": "process-local voluntary code capability required",
            },
            "cyber_attribution_assess": {
                "arguments": ["evidence", "proposed_subject?"],
                "scope": "passive evidence only; infrastructure is not identity",
            },
            "cyber_report_prepare": {
                "arguments": ["incident", "indicators", "evidence", "proposed_subject?", "suspected_crime?", "personal_data_involved?", "jurisdiction?"],
                "destination": "private personal storage/Security Reports/authority_review.jsonl",
                "delivery": "none; human review required",
            },
            "english_dictionary_lookup": {
                "arguments": ["term", "provider? (wiktionary or oxford)"], "source": "English Wiktionary or configured Oxford Languages",
                "retention": "reference result only; no automatic memory write",
            },
            "english_thesaurus_lookup": {
                "arguments": ["term", "relation?", "limit?"], "source": "Datamuse",
                "retention": "reference result only; no automatic memory write",
            },
        },
    }


def request_personal_tool(
    command: Mapping[str, Any], *, child: str | None = None, code_authority: object | None = None,
) -> dict[str, Any]:
    """Queue one chosen command; callers must supply the action and content."""
    payload = dict(command)
    payload.setdefault("id", f"personal_tool_{datetime.now(timezone.utc).timestamp():.6f}")
    payload = seal_code_command(payload, code_authority)
    return append_inastate_queue(QUEUE_KEY, payload, queue_limit=QUEUE_LIMIT, child=child)


def _paths(child: str, project_root: Path, config: Mapping[str, Any]) -> tuple[VirtualFileSystem, Path]:
    fs = VirtualFileSystem(configured_drives(config, child, project_root=project_root))
    fs.ensure_writable_roots()
    personal = fs.drives["ina_hdd"].root
    return fs, personal


def execute_personal_tool_command(
    command: Mapping[str, Any], *, child: str, project_root: Path | str = ".",
    config: Mapping[str, Any] | None = None, lab: CodeExperimentLab | None = None,
) -> dict[str, Any]:
    root = Path(project_root)
    cfg = dict(config or load_config(root / "config.json"))
    action = str(command.get("action") or "").strip().lower()
    verify_code_command(command)
    command_id = str(command.get("id") or "")[:160]
    fs, personal = _paths(child, root, cfg)
    if action == "intuition_inquire":
        journey = begin_intuition_inquiry(command.get('hunch'), question=command.get('question'),
            countercheck=command.get('countercheck'), trigger_references=command.get('trigger_references'),
            depth_budget=command.get('depth_budget', 1))
        value = {'journey': journey, 'evidence_request': current_inquiry_request(journey)}
    elif action == "intuition_review":
        supplied = command.get('journey')
        if not isinstance(supplied, Mapping) or not supplied.get('intuition'):
            raise ValueError('an intuition inquiry is required')
        journey = continue_self_inquiry(supplied, choice=command.get('choice'),
            observation_references=command.get('observation_references'), hypotheses=command.get('hypotheses'))
        value = {'journey': journey, 'evidence_request': current_inquiry_request(journey)}
    elif action == "compare_expressive_variation":
        value = compare_variation(modality=command.get('modality'),
                                  before=command.get('before'), after=command.get('after'),
                                  source=command.get('source'), references=command.get('references', ()))
    elif action == "external_project_self_read":
        value = read_project_source(str(command.get("project") or ''), inazuma_root=root,
                                    relative_path=str(command.get('relative_path') or '.'),
                                    offset=command.get('offset', 0))
    elif action == "write_note":
        text = str(command.get("text") or "")
        if not text or len(text.encode("utf-8")) > MAX_TEXT_LENGTH:
            raise ValueError("note text must be 1..65536 UTF-8 bytes")
        path = fs.write_note(str(command.get("title") or "note"), text)
        value: dict[str, Any] = {"path": str(path), "private": True}
    elif action == "realise_private_text":
        text = str(command.get("text") or "")
        if not text or len(text.encode("utf-8")) > MAX_TEXT_LENGTH:
            raise ValueError("expression text must be 1..65536 UTF-8 bytes")
        intent = create_expression_intent(
            str(command.get("purpose") or "Private written expression"),
            meaning_references=command.get("meaning_references") or (),
            allowed_media=("text",), provenance=("chosen_private_expression",),
        )
        realised = TextRealiser().realise(
            intent, content={"text": text},
            provenance=("ina_supplied_wording", "private_personal_storage"),
        )
        title = str(command.get("title") or intent["intent_id"])
        path = fs.write_note(title, text, folder="Expressions")
        traces = ExpressionTraceStore(personal / "Expressions" / "expression_trace.jsonl")
        traces.append(intent)
        traces.append(realised)
        value = {
            "path": str(path), "private": True, "delivered": False,
            "intent_id": intent["intent_id"], "realisation_id": realised["realisation_id"],
        }
    elif action == "experiment_create":
        experiment_lab = lab or CodeExperimentLab(personal / "Code Experiments")
        value = experiment_lab.create(
            question=str(command.get("question") or ""),
            hypothesis=str(command.get("hypothesis") or ""),
            code=str(command.get("code") or ""), dataset=command.get("dataset"),
            autonomous_continuation_budget=0,
        )
    elif action == "experiment_run":
        experiment_lab = lab or CodeExperimentLab(personal / "Code Experiments")
        value = experiment_lab.run(str(command.get("experiment_id") or ""))
    elif action == "experiment_judge":
        experiment_lab = lab or CodeExperimentLab(personal / "Code Experiments")
        metrics = command.get("metrics")
        if not isinstance(metrics, Mapping):
            raise ValueError("experiment judgement metrics must be an object")
        value = experiment_lab.judge(
            str(command.get("experiment_id") or ""),
            choice=str(command.get("choice") or ""), metrics=metrics,
            explanation=str(command.get("explanation") or ""),
        )
    elif action == "cyber_attribution_assess":
        evidence = command.get("evidence")
        if not isinstance(evidence, list) or len(evidence) > 200:
            raise ValueError("attribution evidence must be a list of at most 200 records")
        value = assess_attribution(evidence, proposed_subject=str(command.get("proposed_subject") or ""))
    elif action == "cyber_report_prepare":
        incident, indicators, evidence = command.get("incident"), command.get("indicators"), command.get("evidence")
        if not isinstance(incident, Mapping):
            raise ValueError("cyber incident must be an object")
        if not isinstance(indicators, list) or len(indicators) > 200:
            raise ValueError("cyber indicators must be a list of at most 200 records")
        if not isinstance(evidence, list) or len(evidence) > 200:
            raise ValueError("attribution evidence must be a list of at most 200 records")
        attribution = assess_attribution(
            evidence, proposed_subject=str(command.get("proposed_subject") or ""),
        )
        report = prepare_report(
            incident, indicators, attribution,
            suspected_crime=bool(command.get("suspected_crime", True)),
            personal_data_involved=bool(command.get("personal_data_involved", False)),
            jurisdiction=str(command.get("jurisdiction") or "uk"),
        )
        value = queue_report_for_review(
            report, personal / "Security Reports" / "authority_review.jsonl",
        )
    elif action == "english_dictionary_lookup":
        value = lookup_definition(str(command.get("term") or ""), provider=str(command.get("provider") or "wiktionary"))
    elif action == "english_thesaurus_lookup":
        value = lookup_related_words(
            str(command.get("term") or ""), relation=str(command.get("relation") or "synonym"),
            limit=int(command.get("limit") or 20),
        )
    else:
        raise ValueError(f"unknown personal tool action: {action or 'missing'}")
    return {
        "schema": "ina.personal_tool_result/V1", "id": command_id,
        "action": action, "status": "ok", "value": value,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }


def process_personal_tool_queue(
    *, child: str, project_root: Path | str = ".", config: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Publish discoverability and consume at most one explicitly queued action."""
    catalog = capability_catalog()
    if get_inastate(CATALOG_KEY, child=child) != catalog:
        update_inastate(CATALOG_KEY, catalog, child=child)
    claimed = drain_inastate_queue(
        QUEUE_KEY, batch_limit=1, queue_limit=QUEUE_LIMIT, child=child,
    )
    if not claimed["batch"]:
        return None
    command = claimed["batch"][0]
    try:
        if not isinstance(command, Mapping):
            raise ValueError("personal tool command must be an object")
        result = execute_personal_tool_command(
            command, child=child, project_root=project_root, config=config,
        )
    except Exception as exc:
        result = {
            "schema": "ina.personal_tool_result/V1",
            "id": str(command.get("id") or "")[:160] if isinstance(command, Mapping) else "",
            "action": str(command.get("action") or "")[:80] if isinstance(command, Mapping) else "",
            "status": "failed", "error": str(exc)[:1000],
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
    update_inastate(RESULT_KEY, result, child=child)
    return result


__all__ = [
    "CATALOG_KEY", "QUEUE_KEY", "RESULT_KEY", "capability_catalog",
    "execute_personal_tool_command", "process_personal_tool_queue", "request_personal_tool",
]
