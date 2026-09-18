import json

from personal_tool_runtime import capability_catalog, execute_personal_tool_command


class FakeLab:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(("create", kwargs))
        return {"experiment_id": "experiment_test", "cycle_id": "cycle_test"}

    def run(self, experiment_id):
        self.calls.append(("run", experiment_id))
        return {"return_code": 0, "network": "isolated", "workspace_scope": "experiment-only"}

    def judge(self, experiment_id, **kwargs):
        self.calls.append(("judge", experiment_id, kwargs))
        return {"choice": kwargs["choice"]}


def _config(tmp_path):
    return {"ina_hdd_writable_path": str(tmp_path / "personal")}


def test_catalog_is_discoverable_but_never_an_automatic_trigger():
    catalog = capability_catalog()
    assert catalog["voluntary"] is True
    assert catalog["automatic_trigger"] is False
    assert set(catalog["commands"]) >= {
        "write_note", "realise_private_text", "experiment_create",
        "experiment_run", "experiment_judge",
    }


def test_note_and_private_text_expression_reach_personal_storage(tmp_path):
    note = execute_personal_tool_command(
        {"id": "n1", "action": "write_note", "title": "idea", "text": "Maybe this shape returns."},
        child="Ina", project_root=tmp_path, config=_config(tmp_path), lab=FakeLab(),
    )
    expression = execute_personal_tool_command(
        {"id": "e1", "action": "realise_private_text", "title": "words.md",
         "purpose": "Put this into words", "text": "I remember the opening pulse."},
        child="Ina", project_root=tmp_path, config=_config(tmp_path), lab=FakeLab(),
    )

    assert open(note["value"]["path"], encoding="utf-8").read() == "Maybe this shape returns."
    assert open(expression["value"]["path"], encoding="utf-8").read() == "I remember the opening pulse."
    assert expression["value"]["delivered"] is False
    traces = (tmp_path / "personal" / "Expressions" / "expression_trace.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["schema"] for line in traces] == [
        "ina.expression_intent/V1", "ina.expression_realisation/V1",
    ]


def test_experiment_commands_reach_create_run_and_judge_without_continuation(tmp_path):
    lab = FakeLab()
    common = {"child": "Ina", "project_root": tmp_path, "config": _config(tmp_path), "lab": lab}
    created = execute_personal_tool_command(
        {"action": "experiment_create", "question": "Q?", "hypothesis": "H.", "code": "print(1)"},
        **common,
    )
    execute_personal_tool_command({"action": "experiment_run", "experiment_id": "experiment_test"}, **common)
    judged = execute_personal_tool_command(
        {"action": "experiment_judge", "experiment_id": "experiment_test", "choice": "stop",
         "metrics": {"honesty": {"complete": True}}, "explanation": "Enough."}, **common,
    )

    assert created["value"]["experiment_id"] == "experiment_test"
    assert lab.calls[0][1]["autonomous_continuation_budget"] == 0
    assert [call[0] for call in lab.calls] == ["create", "run", "judge"]
    assert judged["value"]["choice"] == "stop"
