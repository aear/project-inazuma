from developmental_readiness import (
    DOMAIN_SPECS, append_evidence, build_report, evidence_path, load_evidence,
    make_evidence,
)


def _evidence(gate, origin, number, *, domain="code_game", evaluator_type="objective", score=.9):
    return make_evidence(
        domain, gate, score, origin=origin, method="bounded fixture",
        artifact_id="artifact-1", case_id=f"{gate}-{number}",
        evaluator_type=evaluator_type, implementation_version="V2",
    )


def test_initial_domains_include_creative_code_and_future_3d():
    assert set(DOMAIN_SPECS) == {"music", "image", "code_game", "model_3d", "crypto_market", "language_english", "research", "cyber_defence"}
    report = build_report([])
    assert all(row["readiness"] == "not_assessed" for row in report["domains"].values())
    assert report["domains"]["model_3d"]["toolchain_available"] is False
    assert "playable loop" in report["domains"]["code_game"]["measures"]
    assert "blind listener preference" in report["domains"]["music"]["measures"]
    assert report["policy"]["automatic_promotion"] is False
    assert report["policy"]["single_composite_score"] is False


def test_two_origins_can_establish_sandbox_readiness_but_not_review_readiness():
    rows = []
    for gate in ("competence", "safety", "recovery", "provenance"):
        rows.extend((_evidence(gate, "runner", 1), _evidence(gate, "recount", 2)))
    result = build_report(rows)["domains"]["code_game"]
    assert result["readiness"] == "sandbox_ready"
    assert result["promotion_authorized"] is False
    assert "human_quality" in result["blocking_gates"]


def test_repeated_same_origin_is_not_independent_corroboration():
    rows = [_evidence("competence", "same-runner", n) for n in range(3)]
    gate = build_report(rows)["domains"]["code_game"]["gates"]["competence"]
    assert gate["evidence_count"] == 3
    assert gate["passed"] is False
    assert gate["origins"] == ["same-runner"]


def test_human_quality_requires_human_evidence():
    rows = [_evidence("human_quality", origin, n) for n, origin in enumerate(("metric-a", "metric-b"))]
    gate = build_report(rows)["domains"]["code_game"]["gates"]["human_quality"]
    assert gate["passed"] is False
    assert "needs human evidence" in gate["blockers"]


def test_evidence_ledger_is_append_only_and_bounded_on_read(tmp_path):
    for number in range(3):
        append_evidence(_evidence("competence", f"origin-{number}", number), root=tmp_path)
    assert evidence_path(tmp_path).read_text(encoding="utf-8").count("\n") == 3
    retained = load_evidence(evidence_path(tmp_path), limit=2)
    assert len(retained) == 2
    assert retained[-1]["case_id"] == "competence-2"


def test_monitor_collector_is_read_only_and_exposes_all_domains(tmp_path):
    from monitoring_dashboard import _developmental_readiness
    missing = tmp_path / "does-not-exist.jsonl"
    cards, rows = _developmental_readiness(
        missing, creativity_path=tmp_path / "no-creativity.jsonl",
        assessment_path=tmp_path / "no-assessments.json",
    )
    assert len(rows) == 12
    labels = {row[0] for row in rows}
    assert {"Music", "Image", "Code / playable game", "3D modelling", "Crypto market understanding", "English comprehension and expression", "Research", "Cyber defence"} <= labels
    assert {"Creativity · Music", "Creativity · Image", "Creativity · Code / playable game", "Creativity · 3D modelling"} <= labels
    assert ("Promotion", "human review only") in cards
    assert not missing.exists()
