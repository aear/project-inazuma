import json

import pytest

from code_experiment_lab import CodeExperimentLab, PythonScratchRoom, RoomLimits, SandboxUnavailable


class FakeRoom:
    name = "fake-python"
    version = "V1"

    def run(self, experiment_dir, source_name):
        dataset = json.loads((experiment_dir / "input.json").read_text(encoding="utf-8"))
        return {
            "room": self.name, "room_version": self.version, "return_code": 0,
            "timed_out": False, "elapsed_seconds": 0.01,
            "stdout": str(sum(dataset or [])), "stderr": "", "stdout_truncated": False,
            "stderr_truncated": False, "network": "isolated",
            "workspace_scope": "experiment-only", "limits": {},
        }


def test_question_to_judgement_is_reproducible_and_separate_from_promotion(tmp_path):
    lab = CodeExperimentLab(tmp_path, rooms={"fake-python": FakeRoom()})
    experiment = lab.create(
        question="Does this ranking score improve?", hypothesis="The weighted sum is larger.",
        code="print('bounded')", dataset=[2, 3], room="fake-python",
    )
    result = lab.run(experiment["experiment_id"])
    decision = lab.judge(
        experiment["experiment_id"], choice="keep", metrics={"score": 5},
        explanation="The bounded comparison matched the prediction.",
    )
    proposal = lab.proposal_summary(experiment["experiment_id"])

    assert result["stdout"] == "5"
    assert decision["choice"] == "keep"
    assert proposal["promotion_state"] == "review-required"
    assert proposal["production_tree_modified"] is False
    assert proposal["source_sha256"] and proposal["dataset_sha256"]


def test_attempt_is_immutable_and_judgement_requires_a_run(tmp_path):
    lab = CodeExperimentLab(tmp_path, rooms={"fake-python": FakeRoom()})
    experiment = lab.create(question="Q?", hypothesis="H.", code="pass", room="fake-python")
    with pytest.raises(RuntimeError, match="completed run"):
        lab.judge(experiment["experiment_id"], choice="stop", metrics={}, explanation="No run.")
    lab.run(experiment["experiment_id"])
    with pytest.raises(RuntimeError, match="only once"):
        lab.run(experiment["experiment_id"])


def test_python_room_fails_closed_and_command_has_no_project_mount(tmp_path):
    (tmp_path / "main.py").write_text("pass", encoding="utf-8")
    room = PythonScratchRoom(limits=RoomLimits(), python="/usr/bin/python3", bwrap="/missing/bwrap")
    command = room._command(tmp_path, "main.py")
    assert "--unshare-all" in command
    assert "--clearenv" in command
    assert str((tmp_path / "main.py").resolve()) in command
    assert str(tmp_path.resolve()) not in command
    assert "--remount-ro" in command
    assert "/workspace/main.py" in command
    with pytest.raises(SandboxUnavailable):
        room.run(tmp_path, "main.py")


def test_support_modules_are_hashed_bounded_and_mounted_read_only(tmp_path):
    lab = CodeExperimentLab(tmp_path / "lab", rooms={"fake-python": FakeRoom()})
    experiment = lab.create(
        question="Q?", hypothesis="H.", code="import helper", room="fake-python",
        support_files={"helper.py": "VALUE = 3\n"},
    )
    support = experiment["support_files"][0]
    directory = tmp_path / "lab" / "artifacts" / experiment["experiment_id"]
    assert support["name"] == "helper.py" and support["sha256"]
    assert (directory / "helper.py").read_text(encoding="utf-8") == "VALUE = 3\n"

    command = PythonScratchRoom(python="/usr/bin/python3", bwrap="/usr/bin/bwrap")._command(
        directory, "main.py", marker="READY\n",
    )
    assert str((directory / "helper.py").resolve()) in command
    assert "/workspace/helper.py" in command
    assert "sys.path.insert(0,'/workspace')" in command[-1]
    with pytest.raises(ValueError):
        lab.create(question="Q?", hypothesis="H.", code="pass", room="fake-python",
                   support_files={"../escape.py": "pass"})


def test_process_limit_is_relative_to_existing_host_tasks():
    room = PythonScratchRoom(limits=RoomLimits(processes=8))
    baseline = room._user_task_count()
    assert baseline >= 1
    assert baseline + room.limits.processes > baseline


def test_strong_storage_evidence_can_become_review_issue_with_code(tmp_path):
    captured = {}

    def reporter(child, title, summary, **kwargs):
        captured.update(child=child, title=title, summary=summary, kwargs=kwargs)
        return {"queued": True, "entry_id": "review-1", "delivery_choice": kwargs["delivery_choice"]}

    lab = CodeExperimentLab(
        tmp_path / "lab", rooms={"fake-python": FakeRoom()}, finding_reporter=reporter,
    )
    experiment = lab.create_storage_optimization_goal(
        evidence_report={
            "operation": "memory_lookup", "artifact_class": "index", "snapshot_id": "snapshot-1",
            "summary": {"strong": True, "samples": 5, "mean_storage_attribution": 0.95},
        },
        hypothesis="Batching the reads reduces latency.", code="print('candidate')",
        dataset=[2, 3], room="fake-python",
    )
    lab.run(experiment["experiment_id"])
    lab.judge(
        experiment["experiment_id"], choice="keep", metrics={"latency_ratio": 0.5},
        explanation="The candidate reduced measured latency.",
    )
    queued = lab.queue_review_issue(
        experiment["experiment_id"], child="Ina", config={},
        touched_files=["memory_lookup.py"], delivery_choice="hold",
    )
    assert queued["queued"] is True
    assert "print('candidate')" in captured["summary"]
    assert "production tree was not modified" in captured["summary"]
    assert captured["kwargs"]["metadata"]["production_tree_modified"] is False
    assert captured["kwargs"]["delivery_choice"] == "hold"


def test_storage_experiment_goal_rejects_weak_evidence(tmp_path):
    lab = CodeExperimentLab(tmp_path / "lab", rooms={"fake-python": FakeRoom()})
    with pytest.raises(ValueError, match="strong attributed"):
        lab.create_storage_optimization_goal(
            evidence_report={"operation": "lookup", "artifact_class": "index", "summary": {"strong": False}},
            hypothesis="Maybe faster.", code="pass", room="fake-python",
        )
