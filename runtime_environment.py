"""Explicit startup checks; never install packages or start Ina as a probe."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

DISCORD_MODULES = ("discord", "discord.sinks", "numpy", "aiohttp", "audioop", "nacl", "davey")
GUI_MODULES = ("numpy", "tkinter", "PIL", "cv2", "PyQt5.QtWidgets", "pydub", "audioop")
_PROBE = '''import importlib,json,sys
errors=[]
for name in json.loads(sys.argv[1]):
 try: importlib.import_module(name)
 except Exception as exc: errors.append({"module":name,"error":type(exc).__name__})
print(json.dumps({"ok":not errors,"errors":errors,"python":sys.version.split()[0]}))
'''


def project_python(root: Path, *, service: str | None = None) -> str:
    """Select the validated main environment; retain Pycord's supported Python.

    The ready marker is published only after an explicit compatibility trial.
    A selected but missing interpreter fails visibly instead of silently changing
    Python versions. Paths stay inside the project; no interpreter from chat.
    """
    relative = "Scripts/python.exe" if os.name == "nt" else "bin/python"
    legacy = root / "venv" / relative
    if service == "discord_bridge":
        return str(legacy)
    core = root / ".runtime" / "core315"
    if (core / ".ina-ready").is_file():
        return str(core / relative)
    return str(legacy) if legacy.is_file() else sys.executable


def python_environment(executable: str, base=None) -> dict[str, str]:
    """Keep legacy `python` child commands in their parent's selected environment."""
    environment = dict(os.environ if base is None else base)
    directory = str(Path(executable).absolute().parent)
    inherited = environment.get("PATH", os.defpath)
    environment["PATH"] = directory + os.pathsep + inherited
    return environment


def check_environment(executable: str, modules=DISCORD_MODULES, *, timeout=30.0) -> dict:
    """Import only external dependencies in a finite, isolated Python process."""
    probe_env = python_environment(executable)
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        probe_env[name] = "1"
    try:
        result = subprocess.run(
            [executable, "-I", "-c", _PROBE, json.dumps(list(modules))],
            cwd=Path(executable).absolute().parent, capture_output=True, text=True,
            timeout=timeout, env=probe_env,
        )
        if result.returncode:
            return {"ok": False, "reason": "interpreter_probe_failed", "returncode": result.returncode}
        report = json.loads(result.stdout)
        if not isinstance(report, dict) or not isinstance(report.get("ok"), bool):
            raise ValueError("invalid probe report")
        return report
    except subprocess.TimeoutExpired:
        return {"ok": False, "reason": "dependency_probe_timeout"}
    except (OSError, ValueError):
        return {"ok": False, "reason": "interpreter_probe_unavailable"}


def environment_message(report: dict) -> str:
    failures = ", ".join(str(item["module"]) for item in report.get("errors", ()) if isinstance(item, dict) and "module" in item)
    return ("Python environment unavailable: " + (failures or str(report.get("reason", "unknown")))
            + ". Repair the project venv with a compatible Python and requirements.txt; then restart the service.")


def prepare_gui(root: Path) -> None:
    """Select the project environment before GUI imports or runtime writes."""
    target = project_python(root)
    # Do not resolve symlinks: the venv and base Python may share one binary.
    if Path(sys.executable).absolute() != Path(target).absolute():
        os.execv(target, [target, str(root / "GUI.py"), *sys.argv[1:]])
    report = check_environment(target, GUI_MODULES)
    if not report.get("ok"):
        raise SystemExit(environment_message(report))
    os.environ.update(python_environment(target))

