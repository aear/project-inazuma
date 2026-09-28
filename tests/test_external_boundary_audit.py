from external_boundary_audit import audit_external_boundaries


def test_external_boundary_inventory_names_residual_risk_instead_of_claiming_completion():
    report = audit_external_boundaries()
    assert report["boundary_count"] >= 14
    assert report["missing_owners"] == []
    assert report["residual_review"] == []
    assert set(report["live_unverified"]) == {"codex_harness", "discord", "obs_websocket"}
    assert report["complete"] is False
    assert report["claim"] == "registered_runtime_boundaries_only_not_proof_of_absence"


def test_high_consequence_boundaries_have_explicit_controls():
    report = audit_external_boundaries()
    by_id = {row["id"]: row for row in report["boundaries"]}
    assert "exact_host" in by_id["github_submission"]["controls"]
    assert "external_text_untrusted" in by_id["research_wikipedia"]["controls"]
    assert "no_trading" in by_id["crypto_history"]["controls"]
    assert "loopback_default" in by_id["world_tcp_and_stream"]["controls"]
    assert "image_signature_check" in by_id["discord"]["controls"]
    assert "mutations_disabled_by_default" in by_id["obs_websocket"]["controls"]
    assert "host_header_gate" in by_id["codex_harness"]["controls"]
