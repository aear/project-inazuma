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
from typing import Any, Mapping, Protocol
import uuid

from experience_engine import ExperienceCycleEngine
from io_utils import atomic_write_json


SCHEMA = "ina.code_experiment/V1"
MAX_SOURCE_BYTES = 64 * 1024
MAX_DATASET_BYTES = 2 * 1024 * 1024
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
        command.extend(("--remount-ro", "/", "--chdir", "/workspace"))
        if marker:
            bootstrap = (
                "import os,runpy;"
                f"os.write(2,{marker.encode('utf-8')!r});"
                f"runpy.run_path('/workspace/{source_name}',run_name='__main__')"
            )
            command.extend((self.python, "-I", "-B", "-c", bootstrap))
        else:
            command.extend((self.python, "-I", "-B", f"/workspace/{source_name}"))
        return command

    def _limit_child(self) -> None:
        limits = self.limits
        resource.setrlimit(resource.RLIMIT_CPU, (limits.cpu_seconds, limits.cpu_seconds))
        resource.setrlimit(resource.RLIMIT_AS, (limits.memory_bytes, limits.memory_bytes))
        resource.setrlimit(resource.RLIMIT_FSIZE, (limits.file_bytes, limits.file_bytes))
        resource.setrlimit(resource.RLIMIT_NPROC, (limits.processes, limits.processes))
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
        try:
            with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
                process = subprocess.Popen(
                    command, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                    close_fds=True, preexec_fn=self._limit_child,
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
            "limits": asdict(self.limits),
        }


class CodeExperimentLab:
    """Orchestrate code artifacts while Experience Engine owns learning state."""

    def __init__(self, root: Path | str, *, cycle_engine: ExperienceCycleEngine | None = None,
                 rooms: Mapping[str, ExecutionRoom] | None = None) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.cycles = cycle_engine or ExperienceCycleEngine(root_path=self.root / "cycles", enable_hot=False)
        default_room = PythonScratchRoom()
        self.rooms = dict(rooms or {default_room.name: default_room})

    def create(self, *, question: str, hypothesis: str, code: str,
               dataset: Any = None, room: str = "python-scratch",
               autonomous_continuation_budget: int = 0) -> dict[str, Any]:
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

        experiment_id = f"experiment_{uuid.uuid4().hex}"
        directory = self.root / "artifacts" / experiment_id
        directory.mkdir(parents=True, mode=0o700)
        source_path = directory / "main.py"
        source_path.write_bytes(source)
        (directory / "input.json").write_bytes(dataset_bytes)
        manifest = {
            "schema": SCHEMA, "experiment_id": experiment_id,
            "question": question, "hypothesis": hypothesis,
            "room": room, "source": "main.py", "dataset": "input.json",
            "source_sha256": _digest(source), "dataset_sha256": _digest(dataset_bytes),
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

    def run(self, experiment_id: str) -> dict[str, Any]:
        directory, manifest = self._load(experiment_id)
        if manifest.get("status") != "created":
            raise RuntimeError("an experiment attempt is immutable and can run only once")
        result = self.rooms[str(manifest["room"])].run(directory, str(manifest["source"]))
        run_id = f"run_{uuid.uuid4().hex}"
        record = {"schema": SCHEMA, "run_id": run_id, "experiment_id": experiment_id,
                  "source_sha256": manifest["source_sha256"], "dataset_sha256": manifest["dataset_sha256"],
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
        evaluation = {"metrics": dict(metrics), "explanation": explanation}
        decision = self.cycles.record_choice(str(manifest["cycle_id"]), choice, evaluation=evaluation)
        manifest.update({"status": "judged", "decision_id": decision["decision_id"], "updated_at": _now()})
        atomic_write_json(directory / "manifest.json", manifest, indent=2, ensure_ascii=False)
        return decision

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
            "promotion_state": "review-required", "production_tree_modified": False,
        }

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
    "RoomLimits", "SandboxUnavailable",
]
