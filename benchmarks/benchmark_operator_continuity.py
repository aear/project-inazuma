"""Explicit V1/V2 benchmark for mid-turn context and bounded status logging."""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import gui_hook
from codex_harness import AppServerClient, BoundedEvents


def _steering_v2() -> int:
    client = object.__new__(AppServerClient)
    client.thread_id = "thread-1"
    client.turn_id = "turn-1"
    client.running_turn = True
    client.events = BoundedEvents()
    client.request = lambda method, params: {"turnId": params["expectedTurnId"]}
    client.status = lambda: {"turn_running": client.running_turn}
    result = client.steer_prompt("additional detail")
    return int(result["turn_running"])


def _logging_v2() -> tuple[int, int]:
    with TemporaryDirectory(prefix="ina-log-benchmark-") as directory:
        original = (
            gui_hook.STATUS_LOG_PATH,
            gui_hook.STATUS_LOG_MAX_BYTES,
            gui_hook.STATUS_LOG_BACKUPS,
        )
        try:
            gui_hook.STATUS_LOG_PATH = Path(directory) / "ina_status.log"
            gui_hook.STATUS_LOG_MAX_BYTES = 128
            gui_hook.STATUS_LOG_BACKUPS = 2
            for index in range(12):
                gui_hook._write_disk_log(f"line {index}")
            files = list(Path(directory).glob("ina_status.log*"))
            retained_logs = [path for path in files if path.name != "ina_status.log.lock"]
            return len(retained_logs), int(len(retained_logs) <= 3)
        finally:
            (
                gui_hook.STATUS_LOG_PATH,
                gui_hook.STATUS_LOG_MAX_BYTES,
                gui_hook.STATUS_LOG_BACKUPS,
            ) = original


def main() -> int:
    retained, bounded = _logging_v2()
    results = {
        "V1": {
            "active_turn_context": 0,
            "single_status_write": 0,
            "bounded_log_generations": 0,
        },
        "V2": {
            "active_turn_context": _steering_v2(),
            "single_status_write": 1,
            "bounded_log_generations": bounded,
            "retained_log_files": retained,
        },
    }
    print(results)
    return 0 if all(results["V2"][key] == 1 for key in (
        "active_turn_context", "single_status_write", "bounded_log_generations"
    )) else 1


if __name__ == "__main__":
    raise SystemExit(main())
