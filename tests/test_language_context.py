import json
from datetime import datetime, timedelta, timezone

import language_context as lc
import language_processing as lp


def _state_reader(values):
    def read(key, default=None, *, child=None):
        return values.get(key, default)
    return read


def test_snapshot_is_descriptive_bounded_and_ignores_stale_prediction():
    now = datetime(2026, 8, 10, 12, 0, tzinfo=timezone.utc)
    stale = now - timedelta(hours=2)
    state = {
        "current_prediction": {
            "timestamp": stale.isoformat(),
            "predicted_vector": {"confidence": 0.99, "clarity": 0.9},
            "predicted_symbol_word": {"symbol": "sym_stale", "confidence": 0.99},
        },
        "emotion_snapshot": {"timestamp": now.isoformat(), "values": {"care": 0.8}, "label": "not-copied"},
        "machine_semantics": {
            "updated_at": now.isoformat(),
            "axes": {"attention_value": {"value": 0.7}, "unused": 1.0},
        },
    }
    snapshot = lc.build_language_context_snapshot(
        {
            "source_text": "Which bank did you mean?",
            "conversation_scene": {
                "turns": [{"speaker": "human", "text": "river bank", "reply_to_id": "m1"}],
                "topic_terms": ["river", "bank"],
                "participants": ["human"],
                "signals": {"continuity_terms": ["bank"]},
                "memory_references": [{"event_id": "e1", "cue": "river", "summary": "By the water"}],
            },
        },
        child="TestChild",
        state_reader=_state_reader(state),
        logic_reader=False,
        now=now,
    )

    assert snapshot["prediction"]["fresh"] is False
    assert snapshot["prediction"]["eligible_for_shadow_score"] is False
    assert snapshot["candidate_referents"] == ["river", "bank"]
    assert snapshot["reply_ancestry"] == ["m1"]
    assert snapshot["affective_state"] == {"care": 0.8}
    assert snapshot["machine_semantics"] == {"attention_value": 0.7}
    assert snapshot["freshness"]["affect_age_seconds"] == 0.0
    assert snapshot["freshness"]["machine_semantics_age_seconds"] == 0.0


def test_prediction_requires_freshness_confidence_and_clarity():
    now = datetime(2026, 8, 10, 12, 0, tzinfo=timezone.utc)
    base = {"timestamp": now.isoformat(), "predicted_symbol_word": {"symbol": "sym_river"}}
    for confidence, clarity, expected in ((0.9, 0.4, True), (0.4, 0.4, False), (0.9, 0.01, False)):
        prediction = {**base, "predicted_vector": {"confidence": confidence, "clarity": clarity}}
        snapshot = lc.build_language_context_snapshot(
            {}, child="TestChild",
            state_reader=_state_reader({"current_prediction": prediction}),
            logic_reader=False, now=now,
        )
        assert snapshot["prediction"]["eligible_for_shadow_score"] is expected


def test_supplied_event_state_avoids_reloading_runtime_state():
    def fail_reader(*args, **kwargs):
        raise AssertionError("event state should be reused")

    snapshot = lc.build_language_context_snapshot(
        {
            "language_state_signals": {
                "current_prediction": {}, "machine_semantics": {}, "emotion_snapshot": {}
            }
        },
        child="TestChild", state_reader=fail_reader, logic_reader=False,
    )
    assert snapshot["prediction"]["present"] is False


def test_logic_snapshot_read_and_result_are_bounded():
    calls = []

    def read_logic(child, limit, *, config=None):
        calls.append((child, limit))
        return [{"description": f"trace {index}"} for index in range(20)]

    snapshot = lc.build_language_context_snapshot(
        {
            "language_state_signals": {
                "current_prediction": {}, "machine_semantics": {}, "emotion_snapshot": {}
            }
        },
        child="TestChild", logic_reader=read_logic,
    )
    assert calls == [("TestChild", 3)]
    assert len(snapshot["logic_signals"]) == 3


def test_counterfactual_audit_can_reject_locally_attractive_repetition():
    snapshot = {"active_memory_references": [], "candidate_referents": [], "topic_continuity": {}}
    current = [
        {"symbol": "sym_a", "context_score": 0.0, "candidate": {}},
        {"symbol": "sym_b", "context_score": 0.0, "candidate": {}},
    ]
    counterfactual = [
        {"symbol": "sym_x", "context_score": 0.8, "candidate": {}},
        {"symbol": "sym_x", "context_score": 0.8, "candidate": {}},
    ]
    audit = lc.audit_counterfactual_expression(
        ["one", "two"], current, counterfactual, snapshot
    )
    assert audit["counterfactual"]["local_context"] > audit["current"]["local_context"]
    assert audit["counterfactual_more_coherent"] is False
    assert audit["counterfactual"]["repetition_penalty"] == 2.0


def test_language_evidence_sources_remain_distinct():
    records = [
        lc.new_evidence_record(kind, {"mapping": "sym_a"})
        for kind in (
            "human_feedback", "contextual_success", "self_consistency", "future_recall_match"
        )
    ]
    assert [record["kind"] for record in records] == [
        "human_feedback", "contextual_success", "self_consistency", "future_recall_match"
    ]


def test_language_evidence_store_keeps_channels_separate(tmp_path):
    lc.record_language_evidence(
        "TestChild", "human_feedback", {"mapping": "sym_a"}, base_path=tmp_path
    )
    lc.record_language_evidence(
        "TestChild", "self_consistency", {"mapping": "sym_a"}, base_path=tmp_path
    )
    path = tmp_path / "TestChild" / "memory" / "language_evidence.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [record["kind"] for record in records] == ["human_feedback", "self_consistency"]


def test_shadow_reranker_reports_alternative_but_preserves_selected_symbol(monkeypatch):
    links = {
        "links": [
            {"word": "bank", "symbol": "sym_finance", "confidence": 0.8, "tags": ["money"]},
            {"word": "bank", "symbol": "sym_river", "confidence": 0.8, "tags": ["river"]},
        ]
    }
    snapshot = {
        "enabled": True,
        "topic_continuity": {"topic_terms": ["river"], "continuity_terms": ["river"]},
        "candidate_referents": ["river"],
        "active_memory_references": [],
        "prediction": {},
    }
    index = lp._build_text_vocab_word_symbol_index(links)
    audit = lp._shadow_audit_text_vocab_mappings(
        ["bank"], links, index, child="TestChild",
        context={"language_context_snapshot": snapshot},
    )

    assert index["bank"] == "sym_finance"
    assert audit["candidate_audits"][0]["shadow_symbol"] == "sym_river"
    assert audit["selected_output_unchanged"] is True
    assert audit["counterfactual_audit"]["changed_tokens"] == ["bank"]
