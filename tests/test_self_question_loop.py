import runtime_state
from module_benchmarks import benchmark_module
from self_question_loop import (
    classify_question, formulate_help_request, question_key,
    semantic_text_candidate,
)


def _with_store(tmp_path):
    store = tmp_path / "self_questions.json"
    prior = runtime_state._self_questions_path
    runtime_state._self_questions_path = lambda child=None: store
    return prior


def test_routes_major_question_needs_and_canonicalises_symmetric_relations():
    assert classify_question("What experience grounds the word 'farm'?") == "semantic_grounding"
    assert classify_question("Do I understand what 'calm' means?") == "semantic_validation"
    assert classify_question("Which of these symbols feels most like me?") == "self_reflection"
    assert question_key("What links text and vision?") == question_key("What links vision and text?")


def test_resolved_question_stays_resolved_until_materially_new_evidence(tmp_path):
    prior = _with_store(tmp_path)
    try:
        runtime_state.seed_self_question("What experience grounds 'farm'?", child="tester", evidence={"event": 1})
        runtime_state.mark_self_question_resolved("What experience grounds 'farm'?", child="tester", evidence={"event": 1})
        runtime_state.seed_self_question("What experience grounds 'farm'?", child="tester")
        row = runtime_state._load_self_question_entries("tester")[0]
        assert row["resolved_at"]
        assert row["trigger_count"] == 2
        assert row["ask_count"] == 0
        runtime_state.seed_self_question("What experience grounds 'farm'?", child="tester", evidence={"event": 2})
        assert "resolved_at" not in runtime_state._load_self_question_entries("tester")[0]
    finally:
        runtime_state._self_questions_path = prior


def test_help_ask_count_and_evidence_evaluation_are_separate(tmp_path):
    prior = _with_store(tmp_path)
    question = "Do I understand what 'calm' means?"
    try:
        runtime_state.seed_self_question(question, child="tester")
        request = runtime_state.create_self_question_help_request(question, child="tester")
        assert "example where it would be wrong" in request
        assert runtime_state.record_self_question_evidence(
            question, {"label": "calm"}, uncertainty_changed=False,
            resolved=True, child="tester",
        )
        row = runtime_state._load_self_question_entries("tester")[0]
        assert row["ask_count"] == 1
        assert "resolved_at" not in row
    finally:
        runtime_state._self_questions_path = prior


def test_opaque_sound_ids_are_not_lexical_evidence_and_candidates_are_retained(tmp_path):
    assert semantic_text_candidate("pair:sym_snd_deadbeef") is None
    assert semantic_text_candidate("didn't") == "did not"
    prior = _with_store(tmp_path)
    try:
        runtime_state.seed_self_question(
            "Which of these symbols feels most like me?", child="tester",
            candidate_symbols=[{"symbol_word_id": "one"}, {"symbol_word_id": "two"}],
        )
        assert len(runtime_state._load_self_question_entries("tester")[0]["candidate_symbols"]) == 2
    finally:
        runtime_state._self_questions_path = prior


def test_self_question_resolution_benchmark_compares_retained_versions():
    v1, v2 = benchmark_module("self_question_resolution")
    assert (v1.version, v1.correct, v1.total) == ("V1", 0, 5)
    assert (v2.version, v2.correct, v2.total) == ("V2", 5, 5)
