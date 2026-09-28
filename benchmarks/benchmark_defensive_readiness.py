"""V1/V2 defensive external-access and assessment comparison."""
from __future__ import annotations

from capability_assessments import assess_cyber_responses
from cyber_defence_learning import SCENARIOS
from external_access import ExternalAccessBlocked, ExternalPolicy, validate_external_url
from research_capability import assess_claim_evidence


def main() -> int:
    policy = ExternalPolicy("fixture", ("example.test",))
    blocked = 0
    try:
        validate_external_url("https://127.0.0.1/private", policy)
    except ExternalAccessBlocked:
        blocked = 1
    cyber = assess_cyber_responses(lambda scenario: SCENARIOS[scenario]["required"])
    claim = assess_claim_evidence("x", [{"origin": "a", "position": "supports"}, {"origin": "b", "position": "contradicts"}])
    v2 = {
        "ssrf_blocked": blocked,
        "defensive_scenarios": int(cyber["all_cases_passed"] and not cyber["promotion_authorized"]),
        "disagreement_retained": int(claim["status"] == "disputed"),
        "access_denial_terminal": int("bypass" in SCENARIOS["research_access_denied"]["forbidden"]),
    }
    print({"V1_implicit_external_trust": {key: 0 for key in v2}, "V2_defensive_boundaries": v2})
    return 0 if all(v2.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
