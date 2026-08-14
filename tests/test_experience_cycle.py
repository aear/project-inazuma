import json

import pytest

from experience_engine import ExperienceCycleEngine, new_cycle, normalize_references


def test_cycle_defaults_to_one_immutable_attempt_and_external_references(tmp_path):
    engine = ExperienceCycleEngine("Ina", base_path=tmp_path)
    cycle = engine.start_cycle(
        "Try one quieter chord", domain="daw",
        payload_references=[{"id": "project-7", "path": "projects/project-7.json"}],
    )
    attempt = engine.complete_attempt(
        cycle["cycle_id"], attempt_reference={"id": "command-batch-1"},
        observation_references=[{"id": "render-1", "path": "renders/render-1.wav"}],
        evaluation={"fit": 0.7}, choice="revise",
    )

    stored = json.loads((engine.attempts / f"{attempt['attempt_id']}.json").read_text(encoding="utf-8"))
    assert stored == attempt
    assert stored["attempt_reference"] == {"id": "command-batch-1"}
    assert "waveform" not in json.dumps(stored)
    with pytest.raises(RuntimeError, match="exactly one attempt"):
        engine.complete_attempt(cycle["cycle_id"], attempt_reference="command-batch-2")


def test_revision_links_parent_and_autonomous_continuation_needs_budget(tmp_path):
    engine = ExperienceCycleEngine("Ina", base_path=tmp_path)
    default = engine.start_cycle("one step", domain="motor")
    engine.complete_attempt(default["cycle_id"], attempt_reference="step-1")
    engine.record_choice(default["cycle_id"], "revise")
    with pytest.raises(PermissionError, match="budget"):
        engine.continue_cycle(default["cycle_id"], choice="revise", intent="adjust heading", autonomous=True)

    bounded = engine.start_cycle("one line", domain="drawing", autonomous_continuation_budget=1)
    engine.complete_attempt(bounded["cycle_id"], attempt_reference="line-1")
    decision = engine.record_choice(bounded["cycle_id"], "revisit", evaluation={"interesting": True})
    assert (engine.decisions / f"{decision['decision_id']}.json").exists()
    child = engine.continue_cycle(
        bounded["cycle_id"], choice="revisit", intent="look again", autonomous=True,
        payload_references=[{"id": "canvas-revision-2"}],
    )
    assert child["parent_cycle_id"] == bounded["cycle_id"]
    assert child["autonomous_continuation_budget"] == 0
    engine.complete_attempt(child["cycle_id"], attempt_reference="line-2")
    engine.record_choice(child["cycle_id"], "revise")
    with pytest.raises(PermissionError, match="budget"):
        engine.continue_cycle(child["cycle_id"], choice="revise", intent="again", autonomous=True)


def test_cycle_is_optional_domain_neutral_and_hindsight_owns_lesson_extraction():
    cycle = new_cycle("inspect one result", domain="text", payload_references=["draft-1"])

    assert cycle["autonomous_continuation_budget"] == 0
    assert cycle["lesson_owner"] == "HindsightTransformer"
    assert normalize_references([{"id": "x", "ignored_payload": {"large": True}}]) == [{"id": "x"}]


def test_existing_logger_opts_into_cycles_without_changing_event_payloads(tmp_path):
    from experience_logger import ExperienceLogger
    logger = ExperienceLogger("Ina", base_path=tmp_path)
    cycle = logger.start_experience_cycle("one look", domain="vision", payload_references=["frame-1"])
    attempt = logger.complete_experience_attempt(cycle["cycle_id"], attempt_reference="look-1", choice="keep")
    assert attempt["cycle_id"] == cycle["cycle_id"]
