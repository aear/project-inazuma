"""Voluntary runtime bridge for Ina's private notes, expressions, and experiments.

Nothing in this module invents a request or schedules repeated work.  It only
executes one explicitly queued command at a time and returns bounded results.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import base64
from pathlib import Path
from typing import Any, Mapping

from code_experiment_lab import CodeExperimentLab
from memory_maintenance_tools import inspect_indexes, trial_configured_index, change_index
from config_layers import load_config
from expression_core import ExpressionTraceStore, TextRealiser, create_expression_intent
from ina_desktop.files import VirtualFileSystem, configured_drives
from instruction_authority import seal_code_command, verify_code_command
from lexical_reference import lookup_definition, lookup_related_words
from external_project_registry import read_project_source
from expressive_variation import compare_variation
from memetic_processor import process_meme
from predictive_landscape import revise_landscape
from self_inquiry_journey import begin_intuition_inquiry, continue_self_inquiry, current_inquiry_request
from runtime_state import append_inastate_queue, drain_inastate_queue, get_inastate, update_inastate
from threat_attribution import assess_attribution, prepare_report, queue_report_for_review, verify_evidence_bytes


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
            "english_compose_selected": {
                "arguments": ["native_intent", "expression_intent", "choice?", "cognition_plan?"],
                "scope": "one private English candidate from selected structured meaning; no thought selection, delivery or automatic learning",
                "automatic_trigger": False,
            },
            "english_learning_options": {
                "arguments": [],
                "scope": "optional comprehension, expression and held-out assessment goals; does not run benchmarks or adopt a goal",
                "automatic_trigger": False,
            },
            "english_social_options": {
                "arguments": ["topic_reference?", "audience_reference?"],
                "scope": "owner-invited English in ordinary conversation or chosen short practice; mixed/native/clarify/silence remain choices",
                "automatic_trigger": False,
            },
            "english_social_review": {
                "arguments": ["draft?", "incoming_text?", "reference_draft?", "mode? (conversation or rephrase)", "choice? (english, mixed, native_only, clarify, abstain, silence)", "discourse?"],
                "scope": "one private structural review of supplied words; no invented reply, correction, delivery, reward or automatic learning",
                "automatic_trigger": False,
            },
            "speech_render": {
                "arguments": ["text (1..512 UTF-8 bytes)", "voice? (en-gb, en-us, de, fr, es)"],
                "scope": "local eSpeak NG TTS; private WAV; no playback, publication, translation or learning",
                "automatic_trigger": False,
            },
            "radio_status": {"arguments": [], "scope": "local station clip catalog; no listener data"},
            "radio_prepare_clip": {"arguments": ["relative_path (PCM WAV)", "source_drive? (ina_hdd or music_renders)", "title", "kind (song or speech)", "publication=ina_authored_public"],
                "scope": "explicit public Opus export, isolated decoder; original untouched; not scheduled automatically"},
            "radio_render_announcement": {"arguments": ["symbols (1..8 chosen native voice symbols)"],
                "scope": "existing Ina voice, private WAV only; no microphone relay or external voice imitation"},
            "radio_publish": {"arguments": ["clips (1..32 chosen IDs in order)", "start_at? (UTC epoch seconds)"],
                "scope": "finite public programme up to two hours; no repeat or listener commands"},
            "radio_stop": {"arguments": [], "scope": "stop programme; downloaded public copies cannot be recalled"},
            "radio_remove_clip": {"arguments": ["clip_id"], "scope": "stop programme then remove one exported clip; original preserved"},
            "work_report_write": {
                "arguments": ["report (kind, title, observations, uncertainties, evidence?, measurements?, interpretations?, next_steps?, reproduction_steps?, expected?, actual?, previous_report?)"],
                "kinds": ["drawing", "music", "experiment", "capability", "bug"],
                "scope": "new private report per revision; preserves supplied wording and references; no publication or promotion",
            },
            "work_report_read": {
                "arguments": ["report_id"], "scope": "read one chosen private report; no directory scan",
            },
            "bug_source_inspect": {
                "arguments": ["code", "source_reference?"],
                "scope": "64 KiB supplied Python source; static candidates, no import or execution",
            },
            "bug_file_inspect": {
                "arguments": ["relative_path"],
                "scope": "one non-private Python file on read-only code drive; no scan or execution",
                "follow_up": "chosen experiment_create/run/judge for bounded reproduction; work_report_write for findings",
            },
            "memory_index_inspect": {
                "arguments": [], "scope": "configured index locations and schema only; no fragment payloads",
                "authority": "process-local voluntary action capability", "automatic_trigger": False,
            },
            "memory_index_trial": {
                "arguments": ["location? (temporary or configured index location)"], "scope": "one disposable synthetic trial on a chosen device and a private receipt; no live memory",
                "authority": "process-local voluntary action capability", "automatic_trigger": False,
            },
            "memory_index_change": {
                "arguments": ["location", "confirmation (inspection identity)", "choice (upgrade, rollback, defer, decline)", "max_seconds? (0.1..10)"],
                "scope": "one transactional derived index change; no memory movement, traces or payload rewrites",
                "authority": "process-local voluntary action capability", "automatic_trigger": False,
                "rollback": "fresh inspection then choice=rollback; retained receipt records both locations and schemas",
            },
            "cyber_evidence_verify": {
                "arguments": ["evidence", "content_base64"],
                "scope": "verify at most 64 KiB of explicitly supplied bytes; no file reads or source-identity claim",
                "retention": "process-local sealed byte-check receipt, not proof of historical custody",
            },
            "predictive_landscape": {
                "arguments": ["specification", "previous_model?"],
                "scope": "3D scenario coordinates, conditional paths and optional bounded intuition inquiry",
                "execution": "private versioned map snapshots only; no forecasts certified, scheduling or publication",
            },
            "memetic_process": {
                "arguments": ["operation (interpret, compose, review)", "operation-specific observations or segments"],
                "scope": "bounded context hypotheses, private composition recipes and reaction evidence",
                "execution": "no publication, artifact loading, code execution or automatic learning",
            },
            "intuition_inquire": {
                "arguments": ["hunch", "question", "countercheck", "trigger_references", "depth_budget?"],
                "scope": "voluntary finite evidence request; no execution or memory retrieval",
            },
            "perspective_begin": {
                "arguments": ["event", "max_steps? (1..4)"],
                "scope": "chosen private review; provisional, finite, no public delivery",
            },
            "perspective_step": {
                "arguments": ["state", "choice (continue, stop, defer, ask_evidence, remain_uncertain)", "observation_references?", "countercheck?", "hypotheses?"],
                "scope": "one bounded step, yields control; no memory retrieval or automatic adoption",
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
    if action == "english_compose_selected":
        from english_composition import compose_selected_english
        value = compose_selected_english(command.get("native_intent"), command.get("expression_intent"),
            choice=command.get("choice", "english"), cognition_plan=command.get("cognition_plan"))
    elif action == "english_learning_options":
        from english_learning_profile import build_english_learning_profile
        value = build_english_learning_profile()
    elif action == "english_social_options":
        from social_english import social_english_options
        value = social_english_options(topic_reference=command.get("topic_reference", ""),
                                      audience_reference=command.get("audience_reference", ""))
    elif action == "english_social_review":
        from social_english import review_social_english
        value = review_social_english(command.get("draft", ""), incoming_text=command.get("incoming_text", ""),
            reference_draft=command.get("reference_draft", ""), mode=command.get("mode", "conversation"),
            choice=command.get("choice", "english"), discourse=command.get("discourse"))
    elif action == "speech_render":
        from speech_bridge import render_private_speech
        value = render_private_speech(fs, text=command.get("text"), voice=command.get("voice", "en-gb"))
    elif action.startswith("radio_"):
        import radio_station as radio
        from ina_desktop.files import VirtualDrive
        fs=VirtualFileSystem(tuple(fs.drives.values()) + (VirtualDrive(
            "music_renders", "Ina's own music renders", root/"AI_Children"/child/"memory"/"music_studio"/"renders",
            False,"ina_authored_music"),))
        if action == "radio_status":value=radio.status(cfg)
        elif action == "radio_prepare_clip":
            value=radio.prepare_clip(fs,cfg,relative_path=command.get("relative_path"),title=command.get("title"),
                kind=command.get("kind"),publication=command.get("publication"),source_drive=command.get("source_drive") or "ina_hdd")
        elif action == "radio_render_announcement":
            value=radio.render_announcement(fs,symbols=command.get("symbols"),child=child)
        elif action == "radio_publish":value=radio.publish_program(cfg,clips=command.get("clips"),start_at=command.get("start_at"))
        elif action == "radio_stop":value=radio.stop_program(cfg)
        elif action == "radio_remove_clip":value=radio.remove_clip(cfg,clip_id=command.get("clip_id"))
        else:raise ValueError("unknown radio tool")
    elif action in {"work_report_write", "work_report_read", "bug_source_inspect", "bug_file_inspect"}:
        from ina_work_reports import save_work_report, read_work_report, inspect_python_source, inspect_code_file
        if action == "work_report_write":
            value = save_work_report(fs, command.get("report") or {})
        elif action == "work_report_read":
            value = read_work_report(fs, command.get("report_id"))
        elif action == "bug_file_inspect":
            value = inspect_code_file(fs, command.get("relative_path"))
        else:
            value = inspect_python_source(command.get("code"), source_reference=command.get("source_reference") or "")
    elif action == "memory_index_inspect":
        value = inspect_indexes(child, root.resolve(), cfg)
    elif action == "memory_index_trial":
        value = trial_configured_index(child, root.resolve(), cfg, personal, location=str(command.get("location") or "temporary"))
    elif action == "memory_index_change":
        value = change_index(child, root.resolve(), cfg, personal,
            location=str(command.get("location") or ""), confirmation=str(command.get("confirmation") or ""),
            choice=str(command.get("choice") or "defer"), max_seconds=command.get("max_seconds", 5))
    elif action == "predictive_landscape":
        model = revise_landscape(command.get('specification') or {}, command.get('previous_model'))
        path = fs.write_note(model['model_id'], json.dumps(model, ensure_ascii=False), folder='Predictive Landscapes')
        value = {'model': model, 'path': str(path), 'private': True, 'scheduled_work': False}
    elif action == "memetic_process":
        value = process_meme(dict(command))
    elif action == "perspective_begin":
        from perspective_review import begin_review
        value = begin_review(command.get("event") or {}, variant="private_tool", chosen=True,
                             max_steps=command.get("max_steps", 1))
    elif action == "perspective_step":
        from perspective_review import review_step
        value = review_step(command.get("state") or {}, choice=command.get("choice"),
            observation_references=command.get("observation_references") or (),
            countercheck=command.get("countercheck") or "", hypotheses=command.get("hypotheses"))
    elif action == "intuition_inquire":
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
    elif action == "cyber_evidence_verify":
        encoded = command.get('content_base64')
        if not isinstance(encoded, str) or len(encoded) > 87384:
            raise ValueError('encoded evidence exceeds 64 KiB byte budget')
        content = base64.b64decode(encoded, validate=True)
        if len(content) > 65536:
            raise ValueError('evidence exceeds 64 KiB byte budget')
        value = verify_evidence_bytes(command.get('evidence') or {}, content)
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
