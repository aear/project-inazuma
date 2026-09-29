"""V1 infrastructure-only guessing versus V2 governed attribution evidence."""
from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from threat_attribution import assess_attribution, hash_evidence


def benchmark_threat_attribution() -> dict:
    evidence = [{
        "evidence_id": "owned-log", "evidence_type": "owned_system_log",
        "sha256": hash_evidence(b"bounded-fixture"), "origin": "owned-system",
        "independence_group": "owned", "supports_subject": "candidate",
        "chain_of_custody_recorded": True,
    }]
    candidate = assess_attribution(evidence, proposed_subject="candidate")
    return {
        "schema": "ina.threat_attribution_benchmark/V2",
        "historical": {"version": "V1", "ip_treated_as_identity": True, "independence_required": False, "human_review_required": False},
        "candidate": {
            "version": "V2", "ip_treated_as_identity": False,
            "independence_required": True, "human_review_required": True,
            "single_source_identity_blocked": not candidate["identity_claim_authorized"],
        },
        "not_run_reason": None,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(benchmark_threat_attribution(), indent=2))
