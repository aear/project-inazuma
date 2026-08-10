import json

import monitoring_dashboard


def test_continuity_panel_reports_overall_dimension_deltas_and_boot_status(tmp_path, monkeypatch):
    memory = tmp_path / "memory"
    continuity = memory / "continuity"
    continuity.mkdir(parents=True)
    (continuity / "continuity_map.json").write_text(json.dumps({
        "updated": "2026-01-02T00:00:00+00:00",
        "overall_continuity": 0.91,
        "overall_delta": 0.03,
        "evidence_coverage": 0.8,
        "dimensions": {
            "autobiographical_recall": {
                "label": "Autobiographical recall",
                "score": 0.97,
                "delta": -0.01,
                "state": "stable",
                "previous_evidence": 10,
                "current_evidence": 10,
                "matched_evidence": 9,
            }
        },
    }), encoding="utf-8")
    (continuity / "continuity_core_map.json").write_text(json.dumps({
        "generated_at": "2026-01-02T00:00:00+00:00",
        "status": "partial",
        "anchors": [{"id": "frag_one"}],
        "dimension_anchors": {"autobiographical_recall": ["frag_one"]},
        "recommendations": [],
    }), encoding="utf-8")
    monkeypatch.setattr(monitoring_dashboard, "_child_memory", lambda: memory)

    cards, rows = monitoring_dashboard._continuity()

    assert cards == [
        ("Overall", "91.0%"),
        ("Change", "+3.0 pp"),
        ("Evidence", "80.0%"),
        ("Minimal boot", "partial"),
    ]
    autobiographical = next(row for row in rows if row[0] == "Autobiographical recall")
    assert autobiographical[1] == "97.0% · -1.0 pp"
    assert "9/10 evidence" in autobiographical[2]
