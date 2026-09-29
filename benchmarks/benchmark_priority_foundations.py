"""Bounded V1/V2 evidence for the language, security, code, and handover foundations."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from authorized_security_lab import create_engagement
from external_project_registry import project_status
from kernel_lab import comparison_record
from text_memory import _trim_vocab


def _vocab_candidate(limit: int) -> dict:
    sample = {
        f"word-{number}": {"count": number % 31, "last_seen": f"2026-09-{number % 28 + 1:02d}"}
        for number in range(50_000)
    }
    started = time.perf_counter()
    retained = _trim_vocab(sample, limit)
    elapsed = time.perf_counter() - started
    encoded_bytes = len(json.dumps(retained, separators=(",", ":")).encode("utf-8"))
    return {"version": "V1" if limit == 25_000 else "V2", "limit": limit,
            "retained": len(retained), "seconds": elapsed, "encoded_bytes_proxy": encoded_bytes}


def run() -> dict:
    with tempfile.TemporaryDirectory(prefix="ina_foundation_benchmark_") as directory:
        root = Path(directory)
        registry_status = project_status("Project Mercury", inazuma_root=root)
    engagement = create_engagement(
        platform="hack_the_box_labs", target="10.10.10.10",
        consent_text="bounded fixture consent for this benchmark only",
        rules_url="https://www.hackthebox.com/aup", expires_at="2099-01-01T00:00:00+00:00",
        allowed_actions=["service_enumeration"], ai_policy="ai_native",
    )
    return {
        "benchmark": "priority_foundations",
        "vocab_capacity": [_vocab_candidate(25_000), _vocab_candidate(50_000)],
        "security": {"external_lab_recorded": engagement["schema"], "one_attempt_default": True},
        "kernel": comparison_record({"name": "candidate"}, [{"name": "historical"}]),
        "handover": {"redacted_unconfigured_status": registry_status},
        "claims": {"live_ina_language_quality": "unavailable", "live_vm_boot": "unavailable"},
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
