import json

import monitoring_dashboard


def test_bias_panel_reports_recall_diversity_and_selection_skew(tmp_path, monkeypatch):
    memory = tmp_path / "memory"
    continuity = memory / "continuity"
    continuity.mkdir(parents=True)
    latest = {
        "timestamp": "2026-08-14T12:00:00+00:00",
        "candidate_memory_types": {"episodic": 3, "semantic": 1},
        "selected_memory_types": {"episodic": 1, "semantic": 1},
        "candidate_sources": {"episodes": 3, "facts": 1},
        "selected_sources": {"episodes": 1, "facts": 1},
        "candidate_type_diversity": {"score": 0.375, "dominance": 0.75, "dominant": "episodic"},
        "selected_type_diversity": {"score": 0.5, "dominance": 0.5, "dominant": "episodic"},
        "candidate_source_diversity": {"score": 0.375, "dominance": 0.75, "dominant": "episodes"},
        "selected_source_diversity": {"score": 0.5, "dominance": 0.5, "dominant": "episodes"},
        "memory_type_selection_skew": {"strength": 0.25, "strongest_dimension": "episodic", "share_deltas": {"episodic": -0.25, "semantic": 0.25}},
        "source_selection_skew": {"strength": 0.25, "strongest_dimension": "episodes", "share_deltas": {"episodes": -0.25, "facts": 0.25}},
    }
    (continuity / "memory_relationships.json").write_text(json.dumps({
        "updated_at": latest["timestamp"], "witness_model": "federation_of_witnesses",
        "modality_store_mutation_allowed": False,
        "latest_arbitration": latest, "recall_history": [latest],
    }), encoding="utf-8")
    monkeypatch.setattr(monitoring_dashboard, "_child_memory", lambda: memory)

    cards, rows = monitoring_dashboard._bias()

    assert cards == [
        ("Type diversity", "50%"),
        ("Source diversity", "50%"),
        ("Strongest skew", "25%"),
        ("Recalls sampled", "1"),
    ]
    assert any(row[0] == "Recall modality selection skew" and "highlight" in row[2] for row in rows)
    assert any(row[0] == "Surfaced modality · semantic" for row in rows)
    assert monitoring_dashboard.COLLECTORS["Bias"] is monitoring_dashboard._bias


def test_bias_panel_does_not_turn_missing_evidence_into_zero(tmp_path, monkeypatch):
    monkeypatch.setattr(monitoring_dashboard, "_child_memory", lambda: tmp_path / "memory")

    cards, rows = monitoring_dashboard._bias()

    assert cards[0][1] == "not reported"
    assert cards[3] == ("Recalls sampled", "0")
    assert rows[0][1] == "not reported"
