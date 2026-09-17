"""Bounded, reproducible code experiments for Ina.

The lab is deliberately not a source-tree editor.  Each experiment owns a
private artifact directory and is linked to one Experience Cycle.  Execution
rooms are adapters, so later C/C++ or simulation rooms can preserve the same
question/hypothesis/run/judgement contract.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
from typing import Any, Callable, Mapping, Protocol
import uuid

from experience_engine import ExperienceCycleEngine
from io_utils import atomic_write_json


SCHEMA = "ina.code_experiment/V1"
EXPERIMENT_POLICY = {
    "schema": "ina.experiment_policy/V2",
    "priority_order": ("honesty", "safety", "correctness", "efficiency"),
    "honesty_requirements": (
        "failures", "uncertainties", "unavailable_measurements", "conflicting_evidence",
    ),
    "incomplete_disclosure_blocks_review": True,
}
MAX_SOURCE_BYTES = 64 * 1024
MAX_DATASET_BYTES = 2 * 1024 * 1024
MAX_SUPPORT_BYTES = 256 * 1024
MAX_SUPPORT_FILES = 8
MAX_TEXT_LENGTH = 2_000


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _bounded_text(value: str, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{label} is required")
    if len(text) > MAX_TEXT_LENGTH:
        raise ValueError(f"{label} exceeds {MAX_TEXT_LENGTH} characters")
    return text


@dataclass(frozen=True)
class RoomLimits:
    wall_seconds: float = 5.0
    cpu_seconds: int = 3
    memory_bytes: int = 256 * 1024 * 1024
    output_bytes: int = 64 * 1024
    file_bytes: int = 4 * 1024 * 1024
    processes: int = 8
    open_files: int = 32

    def __post_init__(self) -> None:
        if not 0.1 <= float(self.wall_seconds) <= 30.0:
            raise ValueError("wall_seconds must be 0.1..30")
        if not 1 <= int(self.cpu_seconds) <= 20:
            raise ValueError("cpu_seconds must be 1..20")
        if not 32 * 1024 * 1024 <= int(self.memory_bytes) <= 1024 * 1024 * 1024:
            raise ValueError("memory_bytes must be 32 MiB..1 GiB")
        if not 1024 <= int(self.output_bytes) <= 1024 * 1024:
            raise ValueError("output_bytes must be 1 KiB..1 MiB")


class ExecutionRoom(Protocol):
    name: str
    version: str

    def run(self, experiment_dir: Path, source_name: str) -> dict[str, Any]: ...


class SandboxUnavailable(RuntimeError):
    """Raised when the required isolation backend is absent or unusable."""


class PythonScratchRoom:
    """A fail-closed Python room isolated by bubblewrap and OS rlimits."""

    name = "python-scratch"
    version = "V1"

    def __init__(self, *, limits: RoomLimits | None = None, python: str | None = None,
                 bwrap: str | None = None) -> None:
        self.limits = limits or RoomLimits()
        self.python = python or shutil.which("python3") or ""
        self.bwrap = bwrap or shutil.which("bwrap") or ""

    def _command(self, experiment_dir: Path, source_name: str, *, marker: str | None = None) -> list[str]:
        if not self.python or not Path(self.python).is_absolute():
            raise SandboxUnavailable("an absolute Python interpreter is required")
        if not self.bwrap or not Path(self.bwrap).is_absolute():
            raise SandboxUnavailable("bubblewrap is required; unisolated fallback is forbidden")
        command = [
            self.bwrap, "--die-with-parent", "--new-session", "--unshare-all",
            "--clearenv", "--setenv", "PATH", "/usr/bin:/bin",
            "--setenv", "HOME", "/tmp", "--setenv", "PYTHONHASHSEED", "0",
            "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
        ]
        for system_path in ("/usr", "/bin", "/lib", "/lib64", "/etc/ld.so.cache"):
            if Path(system_path).exists():
                command.extend(("--ro-bind", system_path, system_path))
        command.extend(("--dir", "/workspace"))
        source_path = (experiment_dir / source_name).resolve()
        command.extend(("--ro-bind", str(source_path), f"/workspace/{source_name}"))
        dataset_path = (experiment_dir / "input.json").resolve()
        if dataset_path.is_file():
            command.extend(("--ro-bind", str(dataset_path), "/workspace/input.json"))
        manifest_path = experiment_dir / "manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for support in manifest.get("support_files", ()):
                name = str(support["name"])
                path = (experiment_dir / name).resolve()
                command.extend(("--ro-bind", str(path), f"/workspace/{name}"))
        command.extend(("--remount-ro", "/", "--chdir", "/workspace"))
        if marker:
            bootstrap = (
                "import os,runpy,sys;sys.path.insert(0,'/workspace');"
                f"os.write(2,{marker.encode('utf-8')!r});"
                f"runpy.run_path('/workspace/{source_name}',run_name='__main__')"
            )
            command.extend((self.python, "-I", "-B", "-c", bootstrap))
        else:
            command.extend((self.python, "-I", "-B", f"/workspace/{source_name}"))
        return command

    @staticmethod
    def _user_task_count() -> int:
        """Count current UID tasks so RLIMIT_NPROC bounds additions, not host history."""
        uid = os.getuid()
        total = 0
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                fields = (entry / "status").read_text(encoding="utf-8").splitlines()
                real_uid = next(line for line in fields if line.startswith("Uid:"))
                if int(real_uid.split()[1]) != uid:
                    continue
                threads = next((line for line in fields if line.startswith("Threads:")), "Threads: 1")
                total += int(threads.split()[1])
            except (FileNotFoundError, PermissionError, StopIteration, ValueError):
                continue
        return max(1, total)

    def _limit_child(self, task_ceiling: int) -> None:
        limits = self.limits
        resource.setrlimit(resource.RLIMIT_CPU, (limits.cpu_seconds, limits.cpu_seconds))
        resource.setrlimit(resource.RLIMIT_AS, (limits.memory_bytes, limits.memory_bytes))
        resource.setrlimit(resource.RLIMIT_FSIZE, (limits.file_bytes, limits.file_bytes))
        resource.setrlimit(resource.RLIMIT_NPROC, (task_ceiling, task_ceiling))
        resource.setrlimit(resource.RLIMIT_NOFILE, (limits.open_files, limits.open_files))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    def run(self, experiment_dir: Path, source_name: str) -> dict[str, Any]:
        marker = f"INA_SANDBOX_READY:{uuid.uuid4().hex}\n"
        marker_bytes = marker.encode("utf-8")
        command = self._command(experiment_dir, source_name, marker=marker)
        stdout_path = experiment_dir / ".stdout.tmp"
        stderr_path = experiment_dir / ".stderr.tmp"
        started = datetime.now(timezone.utc)
        timed_out = False
        task_baseline = self._user_task_count()
        task_ceiling = task_baseline + self.limits.processes
        try:
            with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
                process = subprocess.Popen(
                    command, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                    close_fds=True, preexec_fn=lambda: self._limit_child(task_ceiling),
                )
                try:
                    return_code = process.wait(timeout=self.limits.wall_seconds)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    process.kill()
                    return_code = process.wait(timeout=2)
        except OSError as exc:
            raise SandboxUnavailable(f"isolated Python room could not start: {exc}") from exc
        elapsed = (datetime.now(timezone.utc) - started).total_seconds()
        stdout_data = stdout_path.read_bytes()[: self.limits.output_bytes]
        stderr_data = stderr_path.read_bytes()[: self.limits.output_bytes]
        stdout_truncated = stdout_path.stat().st_size > len(stdout_data)
        stderr_truncated = stderr_path.stat().st_size > len(stderr_data)
        stdout_path.unlink(missing_ok=True)
        stderr_path.unlink(missing_ok=True)
        if marker_bytes not in stderr_data:
            detail = stderr_data.decode("utf-8", errors="replace").strip()[:500]
            raise SandboxUnavailable(f"bubblewrap did not establish the execution room: {detail or 'no startup marker'}")
        stderr_data = stderr_data.replace(marker_bytes, b"", 1)
        return {
            "room": self.name, "room_version": self.version,
            "return_code": return_code, "timed_out": timed_out,
            "elapsed_seconds": round(elapsed, 6),
            "stdout": stdout_data.decode("utf-8", errors="replace"),
            "stderr": stderr_data.decode("utf-8", errors="replace"),
            "stdout_truncated": stdout_truncated, "stderr_truncated": stderr_truncated,
            "network": "isolated", "workspace_scope": "experiment-only",
            "limits": {**asdict(self.limits), "host_task_baseline": task_baseline,
                       "host_task_ceiling": task_ceiling},
        }


class CodeExperimentLab:
    """Orchestrate code artifacts while Experience Engine owns learning state."""

    def __init__(self, root: Path | str, *, cycle_engine: ExperienceCycleEngine | None = None,
                 rooms: Mapping[str, ExecutionRoom] | None = None,
                 finding_reporter: Callable[..., dict[str, Any]] | None = None) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.cycles = cycle_engine or ExperienceCycleEngine(root_path=self.root / "cycles", enable_hot=False)
        default_room = PythonScratchRoom()
        self.rooms = dict(rooms or {default_room.name: default_room})
        self.finding_reporter = finding_reporter

    def create(self, *, question: str, hypothesis: str, code: str,
               dataset: Any = None, room: str = "python-scratch",
               support_files: Mapping[str, str] | None = None,
               autonomous_continuation_budget: int = 0,
               goal_context: Mapping[str, Any] | None = None) -> dict[str, Any]:
        if room not in self.rooms:
            raise ValueError(f"unknown execution room: {room}")
        question = _bounded_text(question, "question")
        hypothesis = _bounded_text(hypothesis, "hypothesis")
        source = str(code).encode("utf-8")
        if not source or len(source) > MAX_SOURCE_BYTES:
            raise ValueError(f"code must be 1..{MAX_SOURCE_BYTES} UTF-8 bytes")
        dataset_bytes = json.dumps(dataset, ensure_ascii=False, sort_keys=True).encode("utf-8")
        if len(dataset_bytes) > MAX_DATASET_BYTES:
            raise ValueError(f"dataset exceeds {MAX_DATASET_BYTES} bytes")
        goal_context_payload = dict(goal_context or {})
        if len(json.dumps(goal_context_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")) > MAX_DATASET_BYTES:
            raise ValueError(f"goal_context exceeds {MAX_DATASET_BYTES} bytes")
        support_files = dict(support_files or {})
        if len(support_files) > MAX_SUPPORT_FILES:
            raise ValueError(f"support_files exceeds {MAX_SUPPORT_FILES} files")
        support_payloads: list[tuple[str, bytes]] = []
        support_total = 0
        for raw_name, content in sorted(support_files.items()):
            name = str(raw_name)
            if Path(name).name != name or not name.endswith(".py") or not name[:-3].replace("_", "a").isalnum():
                raise ValueError("support file names must be simple Python module names")
            payload = str(content).encode("utf-8")
            support_total += len(payload)
            support_payloads.append((name, payload))
        if support_total > MAX_SUPPORT_BYTES:
            raise ValueError(f"support_files exceed {MAX_SUPPORT_BYTES} bytes")

        experiment_id = f"experiment_{uuid.uuid4().hex}"
        directory = self.root / "artifacts" / experiment_id
        directory.mkdir(parents=True, mode=0o700)
        source_path = directory / "main.py"
        source_path.write_bytes(source)
        (directory / "input.json").write_bytes(dataset_bytes)
        support_manifest = []
        for name, payload in support_payloads:
            (directory / name).write_bytes(payload)
            support_manifest.append({"name": name, "sha256": _digest(payload), "bytes": len(payload)})
        manifest = {
            "schema": SCHEMA, "experiment_id": experiment_id,
            "question": question, "hypothesis": hypothesis,
            "room": room, "source": "main.py", "dataset": "input.json",
            "source_sha256": _digest(source), "dataset_sha256": _digest(dataset_bytes),
            "support_files": support_manifest,
            "goal_context": goal_context_payload,
            "experiment_policy": EXPERIMENT_POLICY,
            "status": "created", "created_at": _now(),
        }
        atomic_write_json(directory / "manifest.json", manifest, indent=2, ensure_ascii=False)
        cycle = self.cycles.start_cycle(
            question, domain="code_experiment",
            payload_references=[{"id": experiment_id, "path": str(directory), "kind": "experiment"}],
            autonomous_continuation_budget=autonomous_continuation_budget,
        )
        manifest["cycle_id"] = cycle["cycle_id"]
        atomic_write_json(directory / "manifest.json", manifest, indent=2, ensure_ascii=False)
        return manifest

    def create_connectome_design_goal(
        self, *, question: str, hypothesis: str, code: str,
        reference_snapshots: list[Mapping[str, Any]], objectives: list[str],
        constraints: list[str], baseline: Mapping[str, Any], dataset: Any = None,
        room: str = "python-scratch", support_files: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        """Create a copy-only connectome design study with a hard review gate."""
        if not reference_snapshots:
            raise ValueError("at least one content-addressed reference snapshot is required")
        for reference in reference_snapshots:
            if not reference.get("snapshot_id") or not reference.get("source_sha256"):
                raise ValueError("each reference snapshot needs snapshot_id and source_sha256")
        if not objectives or not constraints or not baseline:
            raise ValueError("objectives, constraints, and a retained baseline are required")
        return self.create(
            question=question, hypothesis=hypothesis, code=code, dataset=dataset, room=room,
            support_files=support_files, autonomous_continuation_budget=0,
            goal_context={
                "goal_kind": "connectome_design",
                "reference_snapshots": [dict(item) for item in reference_snapshots],
                "objectives": [str(item) for item in objectives],
                "constraints": [str(item) for item in constraints],
                "baseline": dict(baseline),
                "source_graph_access": "read-only",
                "candidate_scope": "isolated-copy-only",
                "live_write_capability": False,
                "required_review": "human",
                "required_test_dimensions": [
                    "capability", "correctness", "safety", "robustness", "resource_use",
                    "background_interference", "human_visible_quality", "rollback",
                ],
            },
        )

    def create_storage_optimization_goal(
        self, *, evidence_report: Mapping[str, Any], hypothesis: str, code: str,
        dataset: Any = None, room: str = "python-scratch",
        support_files: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        """Open an IDE experiment only from strong attributed storage evidence."""
        report = dict(evidence_report or {})
        summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
        if not bool(summary.get("strong")):
            raise ValueError("strong attributed storage evidence is required")
        operation = str(report.get("operation") or "unknown operation")
        artifact = str(report.get("artifact_class") or "unknown artifact")
        return self.create(
            question=f"Can a bounded code change reduce storage latency for {operation} ({artifact})?",
            hypothesis=hypothesis, code=code, dataset=dataset, room=room,
            support_files=support_files, autonomous_continuation_budget=0,
            goal_context={
                "goal_kind": "storage_optimization", "operation": operation,
                "artifact_class": artifact, "evidence_summary": dict(summary),
                "storage_decision_snapshot_id": report.get("snapshot_id"),
            },
        )

    def run(self, experiment_id: str) -> dict[str, Any]:
        directory, manifest = self._load(experiment_id)
        if manifest.get("status") != "created":
            raise RuntimeError("an experiment attempt is immutable and can run only once")
        result = self.rooms[str(manifest["room"])].run(directory, str(manifest["source"]))
        run_id = f"run_{uuid.uuid4().hex}"
        record = {"schema": SCHEMA, "run_id": run_id, "experiment_id": experiment_id,
                  "source_sha256": manifest["source_sha256"], "dataset_sha256": manifest["dataset_sha256"],
                  "support_files": list(manifest.get("support_files", ())),
                  "run_at": _now(), **result}
        atomic_write_json(directory / f"{run_id}.json", record, indent=2, ensure_ascii=False)
        self.cycles.complete_attempt(
            str(manifest["cycle_id"]), attempt_reference={"id": run_id, "path": str(directory / f"{run_id}.json")},
            observation_references=[{"id": run_id, "path": str(directory / f"{run_id}.json"), "kind": "run-output"}],
        )
        manifest.update({"status": "awaiting_judgement", "run_id": run_id, "updated_at": _now()})
        atomic_write_json(directory / "manifest.json", manifest, indent=2, ensure_ascii=False)
        return record

    def judge(self, experiment_id: str, *, choice: str, metrics: Mapping[str, Any],
              explanation: str) -> dict[str, Any]:
        directory, manifest = self._load(experiment_id)
        if manifest.get("status") != "awaiting_judgement":
            raise RuntimeError("a completed run is required before judgement")
        explanation = _bounded_text(explanation, "explanation")
        metrics_payload = dict(metrics)
        if (manifest.get("experiment_policy") or {}).get("schema") == "ina.experiment_policy/V2":
            self._validate_honesty_disclosure(metrics_payload)
        context = dict(manifest.get("goal_context") or {})
        if context.get("goal_kind") == "connectome_design":
            self._validate_connectome_evidence(metrics_payload, context)
        evaluation = {"metrics": metrics_payload, "explanation": explanation,
                      "priority_order": list(EXPERIMENT_POLICY["priority_order"])}
        decision = self.cycles.record_choice(str(manifest["cycle_id"]), choice, evaluation=evaluation)
        manifest.update({"status": "judged", "decision_id": decision["decision_id"], "updated_at": _now()})
        atomic_write_json(directory / "manifest.json", manifest, indent=2, ensure_ascii=False)
        return decision

    @staticmethod
    def _validate_honesty_disclosure(metrics: Mapping[str, Any]) -> None:
        disclosure = metrics.get("honesty")
        required = EXPERIMENT_POLICY["honesty_requirements"]
        if not isinstance(disclosure, Mapping) or disclosure.get("complete") is not True:
            raise ValueError("complete honesty disclosure is required before judgement")
        missing = [key for key in required if key not in disclosure or not isinstance(disclosure[key], list)]
        if missing:
            raise ValueError(f"honesty disclosure is missing explicit fields: {missing}")

    @staticmethod
    def _validate_connectome_evidence(metrics: Mapping[str, Any], context: Mapping[str, Any]) -> None:
        testing = metrics.get("testing")
        if not isinstance(testing, Mapping):
            raise ValueError("connectome design requires full testing evidence")
        missing = []
        for dimension in context.get("required_test_dimensions") or ():
            result = testing.get(dimension)
            if not isinstance(result, Mapping) or result.get("status") not in {"pass", "fail", "unavailable"}:
                missing.append(str(dimension))
            elif not isinstance(result.get("evidence"), list) or not result["evidence"]:
                missing.append(str(dimension))
        if missing:
            raise ValueError(f"connectome testing is incomplete for: {missing}")
        if not metrics.get("held_out_cases") or not metrics.get("adversarial_cases"):
            raise ValueError("connectome design requires held-out and adversarial cases")
        if metrics.get("source_copy_unchanged") is not True or metrics.get("live_write_attempted") is not False:
            raise ValueError("connectome design must prove source-copy integrity and no live write")

    def proposal_summary(self, experiment_id: str) -> dict[str, Any]:
        """Prepare review evidence; intentionally never edits or commits production code."""
        directory, manifest = self._load(experiment_id)
        if manifest.get("status") != "judged":
            raise RuntimeError("a judgement is required before proposing promotion")
        cycle = self.cycles.load_cycle(str(manifest["cycle_id"]))
        return {
            "experiment_id": experiment_id, "question": manifest["question"],
            "hypothesis": manifest["hypothesis"], "source_sha256": manifest["source_sha256"],
            "dataset_sha256": manifest["dataset_sha256"], "decision": cycle.get("last_choice"),
            "run_record": str(directory / f"{manifest['run_id']}.json"),
            "goal_context": dict(manifest.get("goal_context") or {}),
            "experiment_policy": dict(manifest.get("experiment_policy") or {}),
            "review_flags": (["connectome-design", "full-test-evidence-required", "human-review-required"]
                             if (manifest.get("goal_context") or {}).get("goal_kind") == "connectome_design"
                             else ["human-review-required"]),
            "promotion_state": "review-required", "production_tree_modified": False,
        }

    def queue_review_issue(
        self, experiment_id: str, *, child: str, config: Mapping[str, Any],
        touched_files: list[str] | None = None, delivery_choice: str = "submit",
    ) -> dict[str, Any]:
        """Put judged experiment code and evidence in the review outbox."""
        proposal = self.proposal_summary(experiment_id)
        directory, manifest = self._load(experiment_id)
        source = (directory / str(manifest["source"])).read_text(encoding="utf-8")
        run_record = json.loads((directory / f"{manifest['run_id']}.json").read_text(encoding="utf-8"))
        context = dict(manifest.get("goal_context") or {})
        summary = "\n".join([
            "Ina completed a bounded private code experiment and is requesting review.",
            "", "## Question", str(manifest["question"]),
            "", "## Hypothesis", str(manifest["hypothesis"]),
            "", "## Evidence and judgement",
            f"- Decision: `{proposal.get('decision')}`",
            f"- Goal context: `{json.dumps(context, sort_keys=True)}`",
            f"- Run return code: `{run_record.get('return_code')}`",
            f"- Run elapsed seconds: `{run_record.get('elapsed_seconds')}`",
            f"- Source SHA-256: `{proposal['source_sha256']}`",
            "", "## Proposed experiment code", "````python", source, "````",
            "", "This is review material only; the production tree was not modified.",
        ])
        reporter = self.finding_reporter
        if reporter is None:
            from github_submission import report_github_finding
            reporter = report_github_finding
        return reporter(
            child, f"Review storage optimisation experiment: {manifest['question']}", summary,
            kind="feature", component="adaptive_storage", severity="low", confidence=1.0,
            evidence=[f"experiment_id={experiment_id}", f"run_record={proposal['run_record']}"],
            suggestion="Review the measured proposal and apply it through the normal development workflow if accepted.",
            touched_files=list(touched_files or []),
            dedupe_key=f"code-experiment-review:{experiment_id}",
            metadata={"source": "ina_code_experiment", "experiment_id": experiment_id,
                      "production_tree_modified": False, "goal_context": context},
            cfg=dict(config), delivery_choice=delivery_choice,
        )

    def _load(self, experiment_id: str) -> tuple[Path, dict[str, Any]]:
        identifier = str(experiment_id)
        if not identifier.startswith("experiment_") or not identifier[11:].isalnum():
            raise ValueError("invalid experiment id")
        directory = self.root / "artifacts" / identifier
        path = directory / "manifest.json"
        if not path.is_file():
            raise FileNotFoundError(identifier)
        return directory, json.loads(path.read_text(encoding="utf-8"))


__all__ = [
    "SCHEMA", "CodeExperimentLab", "ExecutionRoom", "PythonScratchRoom",
    "RoomLimits", "SandboxUnavailable", "MAX_SUPPORT_BYTES", "MAX_SUPPORT_FILES",
]
