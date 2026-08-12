from __future__ import annotations

import subprocess
from typing import Any


OUTPUT_SINK = "ina_workspace_output"
INPUT_SINK = "ina_workspace_input"


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, check=False)


def _existing_sinks() -> set[str]:
    result = _run(["pactl", "list", "sinks", "short"])
    if result.returncode != 0:
        return set()
    return {
        fields[1]
        for line in result.stdout.splitlines()
        if len(fields := line.split()) >= 2
    }


def ensure_audio_buses() -> dict[str, Any]:
    """Create silent output and injectable-input buses through PipeWire Pulse."""
    existing = _existing_sinks()
    modules: list[int] = []
    errors: list[str] = []
    for name, description in (
        (OUTPUT_SINK, "Ina Workspace Output"),
        (INPUT_SINK, "Ina Workspace Input Bus"),
    ):
        if name in existing:
            continue
        result = _run([
            "pactl", "load-module", "module-null-sink",
            f"sink_name={name}", "rate=48000", "channels=2",
            f"sink_properties=device.description={description.replace(' ', '_')}",
        ])
        if result.returncode == 0:
            try:
                modules.append(int(result.stdout.strip()))
            except ValueError:
                pass
        else:
            errors.append(result.stderr.strip() or f"failed to create {name}")
    ready = {OUTPUT_SINK, INPUT_SINK}.issubset(_existing_sinks())
    return {
        "ready": ready,
        "output_sink": OUTPUT_SINK,
        "output_monitor": f"{OUTPUT_SINK}.monitor",
        "input_sink": INPUT_SINK,
        "input_source": f"{INPUT_SINK}.monitor",
        "module_ids": modules,
        "errors": errors,
    }


def unload_audio_buses(module_ids: list[int]) -> None:
    for module_id in module_ids:
        _run(["pactl", "unload-module", str(int(module_id))])
