from datetime import datetime, timedelta, timezone

import pytest

from authorized_security_lab import LabAuthorizationError, authorize_action, create_engagement


def _engagement(**overrides):
    values = {
        "platform": "hack_the_box_labs", "target": "10.10.10.10",
        "allowed_actions": ["enumerate_assigned_target"],
        "expires_at": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
        "consent_text": "Assigned by HTB to this account for this active lab session.",
        "rules_url": "https://www.hackthebox.com/legal/aup", "ai_policy": "ai_native",
    }
    values.update(overrides)
    return create_engagement(**values)


def test_exact_target_action_and_time_are_all_required():
    engagement = _engagement()
    assert authorize_action(engagement, target="10.10.10.10", action="enumerate_assigned_target")["authorized"] is True
    with pytest.raises(LabAuthorizationError, match="outside"):
        authorize_action(engagement, target="10.10.10.11", action="enumerate_assigned_target")
    with pytest.raises(LabAuthorizationError, match="outside"):
        authorize_action(engagement, target="10.10.10.10", action="persistence")


def test_external_lab_requires_explicit_ai_native_rules():
    with pytest.raises(LabAuthorizationError, match="autonomous AI"):
        _engagement(ai_policy="ai_assisted")


def test_forbidden_action_cannot_be_put_into_engagement():
    with pytest.raises(LabAuthorizationError, match="forbidden"):
        _engagement(allowed_actions=["enumerate_assigned_target", "attack_third_party"])
