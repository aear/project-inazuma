"""Explicit V1 metadata-only versus V2 enforced-boundary comparison."""
from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from security_adversarial_assessment import assess_security_boundaries


def benchmark_security_boundaries() -> dict:
    report = assess_security_boundaries()
    v1 = {
        "version": "V1", "dns_destination_pinned": False,
        "external_flag_enforced_at_code_boundary": False,
        "harness_host_origin_checked": False,
        "cyber_capability_measured": False,
    }
    v2 = {
        "version": "V2", "dns_destination_pinned": True,
        "external_flag_enforced_at_code_boundary": True,
        "harness_host_origin_checked": True,
        "cyber_capability_measured": False,
    }
    return {
        "schema": "ina.security_boundary_benchmark/V2", "historical": v1,
        "candidate": v2, "report": report,
        "improved_dimensions": [
            key for key in v2 if key != "version" and v2[key] and not v1[key]
        ],
        "unavailable": ["live_discord_trace", "live_harness_adversarial", "ina_cyber_response"],
    }


if __name__ == "__main__":
    import json
    print(json.dumps(benchmark_security_boundaries(), indent=2))
