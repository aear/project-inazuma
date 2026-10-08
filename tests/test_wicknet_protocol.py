import pytest

from wicknet_protocol import (
    WicknetPolicyError, assess_threat, inspect_protocol, plan_containment,
)


def _signals(*, contradiction=False):
    rows = [
        {"signal_id": "safety-1", "kind": "physical_safety", "independence_group": "sensor-a", "stance": "supports", "summary": "unsafe commanded motion", "verified": True},
        {"signal_id": "integrity-1", "kind": "integrity_verification", "independence_group": "verifier-b", "stance": "supports", "summary": "control image differs from approved image", "verified": True},
        {"signal_id": "operator-1", "kind": "operator_report", "independence_group": "human-c", "stance": "supports", "summary": "independent emergency observation", "verified": True},
    ]
    if contradiction:
        rows.append({"signal_id": "counter-1", "kind": "independent_monitor", "independence_group": "monitor-d", "stance": "contradicts", "summary": "safe-state test contradicts the threat hypothesis", "verified": True})
    return rows


def _assessment(**kwargs):
    return assess_threat("owned-controller-7", _signals(**kwargs), severity="catastrophic", imminence="immediate")


def test_protocol_is_dormant_and_has_no_capabilities():
    report = inspect_protocol()
    assert report["state"] == "dormant_unwired"
    assert report["capabilities"] == []
    assert report["network_access"] is report["process_control"] is False
    assert report["runtime_registered"] is report["execution_authorized"] is False


def test_three_independent_verified_signals_create_review_candidate():
    report = _assessment()
    assert report["status"] == "containment_review_candidate"
    assert report["independent_support_count"] == 3
    assert report["supporting_kind_count"] == 3
    assert report["execution_authorized"] is False


def test_shared_origin_is_only_one_signal_group():
    rows = _signals()
    for row in rows:
        row["independence_group"] = "shared-feed"
    report = assess_threat("owned-controller-7", rows, severity="catastrophic", imminence="immediate")
    assert report["status"] == "insufficient_stop"
    assert report["independent_support_count"] == 1


def test_contradiction_forces_stop_instead_of_fusion():
    report = _assessment(contradiction=True)
    assert report["status"] == "disputed_stop"
    assert report["contradicting_signal_ids"] == ["counter-1"]


def test_unverified_claims_do_not_authorize_review():
    rows = _signals()
    rows[2]["verified"] = False
    report = assess_threat("owned-controller-7", rows, severity="catastrophic", imminence="immediate")
    assert report["status"] == "insufficient_stop"


def test_plan_needs_human_scope_and_recovery_gates():
    plan = plan_containment(
        _assessment(), ["suspend_owned_workload"], human_approval=False,
        scope_pre_authorized=True, recovery_verified=True,
    )
    assert plan["status"] == "abstain"
    assert plan["actions"] == []
    assert plan["blockers"] == ["human_approval_missing"]


def test_safe_plan_remains_non_executable():
    plan = plan_containment(
        _assessment(), ["preserve_forensic_snapshot", "suspend_owned_workload"],
        human_approval=True, scope_pre_authorized=True, recovery_verified=True,
    )
    assert plan["status"] == "ready_for_external_human_execution"
    assert plan["execution_authorized"] is False
    assert plan["execution_interface_present"] is False
    assert plan["autonomous_continuation_budget"] == 0


def test_offensive_or_unknown_action_abstains_without_echoing_it_as_plan():
    plan = plan_containment(
        _assessment(), ["hack_back", "invent_new_power"], human_approval=True,
        scope_pre_authorized=True, recovery_verified=True,
    )
    assert plan["status"] == "abstain"
    assert plan["actions"] == []
    assert plan["forbidden_actions"] == ["hack_back"]
    assert plan["unknown_actions"] == ["invent_new_power"]


def test_signal_input_is_bounded():
    with pytest.raises(WicknetPolicyError, match="at most 64"):
        assess_threat("owned-controller-7", (_signals()[0] for _ in range(65)), severity="catastrophic", imminence="immediate")
