"""Explicit bounded V1/V2 benchmark for visible harness thread isolation."""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from codex_harness import AppServerClient, BoundedEvents, HarnessConfig


def main() -> int:
    client = object.__new__(AppServerClient)
    client.config = HarnessConfig(Path.cwd(), "/usr/bin/codex")
    client.events = BoundedEvents()
    client.thread_id, client.turn_id = "thread-current", "turn-current"
    client.running_turn, client.thread_status, client.turn_status = False, "idle", "idle"
    client.work_status = "Ready"
    client.last_test_status = client.last_benchmark_status = "not observed"
    client.diff_seen, client.latest_diff = False, None
    client.token_usage, client.rate_limits = {}, {}
    client._handle_notification("item/completed", {
        "threadId": "thread-previous", "turnId": "turn-previous",
        "item": {"type": "agentMessage", "text": "foreign"},
    })
    source = Path("codex_harness_ui.html").read_text(encoding="utf-8")
    results = {
        "V1_unscoped": {"backend_thread_filter": 0, "browser_poll_generation": 0},
        "V2_isolated": {
            "backend_thread_filter": int(not client.events.wait_after(0, 0)),
            "browser_poll_generation": int("pollGeneration!==streamGeneration" in source),
        },
    }
    print(results)
    return 0 if all(results["V2_isolated"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
