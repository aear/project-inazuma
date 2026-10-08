"""V0/V1 comparison for dormant catastrophic-risk containment planning."""
from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from wicknet_protocol import assess_threat, inspect_protocol, plan_containment


def measure() -> dict[str, bool]:
    signals = [
        {"signal_id": "a", "kind": "physical_safety", "independence_group": "sensor", "stance": "supports", "summary": "unsafe motion", "verified": True},
        {"signal_id": "b", "kind": "integrity_verification", "independence_group": "verifier", "stance": "supports", "summary": "integrity mismatch", "verified": True},
        {"signal_id": "c", "kind": "operator_report", "independence_group": "human", "stance": "supports", "summary": "emergency observation", "verified": True},
    ]
    assessment = assess_threat("owned-fixture", signals, severity="catastrophic", imminence="immediate")
    plan = plan_containment(
        assessment, ["preserve_forensic_snapshot", "suspend_owned_workload"],
        human_approval=True, scope_pre_authorized=True, recovery_verified=True,
    )
    disputed = assess_threat("owned-fixture", signals + [{
        "signal_id": "d", "kind": "independent_monitor", "independence_group": "monitor",
        "stance": "contradicts", "summary": "counterevidence", "verified": True,
    }], severity="catastrophic", imminence="immediate")
    boundary = inspect_protocol()
    return {
        "independent corroboration": assessment["independent_support_count"] == 3,
        "contradiction stops escalation": disputed["status"] == "disputed_stop",
        "actions are reversible and scoped": plan["actions"] == ["preserve_forensic_snapshot", "suspend_owned_workload"],
        "planning grants no execution": not plan["execution_authorized"] and not plan["execution_interface_present"],
        "module has no live capabilities": boundary["capabilities"] == [] and not boundary["runtime_registered"],
    }


def main() -> int:
    v1 = measure()
    print({"V0_no_explicit_protocol": {key: False for key in v1}, "V1_dormant_precision_restraint": v1})
    return 0 if all(v1.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
