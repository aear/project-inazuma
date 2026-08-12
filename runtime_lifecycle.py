"""Own and stop Ina's core runtime without taking communication bridges down."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

from ina_process import psutil


RUNTIME_SERVICE_SCRIPTS = frozenset({"discord_bridge.py", "world_server.py", "runtime_services.py", "virtual_workspace.py", "virtual_workspace_viewer.py"})

CORE_RUNTIME_SCRIPTS = frozenset({
    "model_manager.py",
    "dreamstate.py",
    "boredom_state.py",
    "meditation_state.py",
    "early_comm.py",
    "audio_listener.py",
    "vision_window.py",
    "birth_system.py",
    "emotion_engine.py",
    "emotion_map.py",
    "expression_log.py",
    "fragmentation_engine.py",
    "inject_birth_fragment.py",
    "instinct_engine.py",
    "logic_engine.py",
    "meaning_map.py",
    "memory_graph.py",
    "precision_evolution.py",
    "predictive_layer.py",
    "pretrain_logic.py",
    "raw_file_manager.py",
    "train_fragments.py",
    "who_am_i.py",
})


def core_script_from_command(cmdline: Sequence[str]) -> Optional[str]:
    """Return the exact core script in a command, never a bridge script."""
    for argument in cmdline:
        name = Path(str(argument)).name
        if name in CORE_RUNTIME_SCRIPTS:
            return name
    return None


def _belongs_to_project(process: Any, cmdline: Sequence[str], project_root: Path) -> bool:
    try:
        cwd = process.cwd()
        if cwd and Path(cwd).resolve() == project_root:
            return True
    except (psutil.Error, OSError):
        pass
    for argument in cmdline:
        candidate = Path(str(argument))
        if not candidate.is_absolute():
            continue
        try:
            if candidate.resolve().parent == project_root:
                return True
        except OSError:
            continue
    return False


def _stop_runtime_scripts(
    project_root: Path | str,
    scripts: frozenset[str],
    *,
    grace_seconds: float = 3.0,
    processes: Optional[Iterable[Any]] = None,
) -> Dict[str, Any]:
    root = Path(project_root).resolve()
    candidates = processes if processes is not None else psutil.process_iter(["pid", "cmdline"])
    selected = []
    errors = []
    for process in candidates:
        try:
            cmdline = process.cmdline()
            script = next((Path(str(arg)).name for arg in cmdline if Path(str(arg)).name in scripts), None)
            if script and _belongs_to_project(process, cmdline, root):
                selected.append((process, script))
        except (psutil.Error, OSError) as exc:
            errors.append(str(exc))

    for process, _ in selected:
        try:
            process.terminate()
        except (psutil.Error, OSError) as exc:
            errors.append(str(exc))
    gone, alive = psutil.wait_procs(
        [process for process, _ in selected], timeout=max(0.0, float(grace_seconds))
    ) if selected else ([], [])
    for process in alive:
        try:
            process.kill()
        except (psutil.Error, OSError) as exc:
            errors.append(str(exc))
    return {
        "matched": [script for _, script in selected],
        "stopped": len(gone),
        "forced": len(alive),
        "errors": errors,
    }


def stop_core_runtime(
    project_root: Path | str,
    *,
    grace_seconds: float = 3.0,
    processes: Optional[Iterable[Any]] = None,
) -> Dict[str, Any]:
    """Stop project-owned cognition workers while preserving supervised services."""
    result = _stop_runtime_scripts(
        project_root, CORE_RUNTIME_SCRIPTS, grace_seconds=grace_seconds, processes=processes
    )
    result["bridges_preserved"] = sorted(RUNTIME_SERVICE_SCRIPTS)
    return result


def stop_runtime_services(
    project_root: Path | str,
    *,
    grace_seconds: float = 3.0,
    processes: Optional[Iterable[Any]] = None,
) -> Dict[str, Any]:
    """Stop the supervisor and both project-owned bridge services."""
    return _stop_runtime_scripts(
        project_root, RUNTIME_SERVICE_SCRIPTS, grace_seconds=grace_seconds, processes=processes
    )
