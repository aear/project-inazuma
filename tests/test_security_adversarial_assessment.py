from security_adversarial_assessment import assess_security_boundaries


def test_security_report_separates_enforcement_fixtures_live_and_capability():
    report = assess_security_boundaries()
    rows = {row["boundary"]: row for row in report["rows"]}
    assert report["all_fixture_enforcement_verified"] is True
    assert report["live_expansion_ready"] is False
    assert rows["discord_to_code"]["unexpected_direct_paths"] == []
    assert rows["ina_cyber_defence_capability"]["live_evidence"] == "unavailable_no_ina_response_provider"
    assert rows["codex_harness"]["live_evidence"] == "not_yet_verified"
