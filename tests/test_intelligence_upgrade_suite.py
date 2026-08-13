from discourse_context import build_discourse_context, resolution_for, role_alignment
from module_benchmarks import benchmark_module, list_benchmark_modules
from neural_taxonomy import count_node_types, normalize_node_type
from self_questions_format import format_questions


def test_discourse_resolves_speaker_addressee_possession_and_referents():
    context = build_discourse_context(
        "I trust your handling of this, but they remember that.",
        speaker={"id": "sakura", "name": "Sakura"},
        addressee={"id": "ina", "name": "Ina", "is_self": True},
        self_identity={"id": "ina", "name": "Ina", "is_self": True},
        current_subject="memory", prior_referent="garden", mentioned_entities=("Rowan",),
    )
    assert resolution_for(context, "i")["referents"][0]["id"] == "sakura"
    assert resolution_for(context, "your")["possessive"] is True
    assert resolution_for(context, "your")["referents"][0]["id"] == "self"
    assert resolution_for(context, "this")["referents"][0]["id"] == "memory"
    assert resolution_for(context, "that")["referents"][0]["id"] == "garden"
    assert resolution_for(context, "they")["referents"][0]["id"] == "rowan"


def test_historical_i_remains_bound_to_episode_speaker():
    current = build_discourse_context("I remember", speaker="Sakura", addressee="Ina")
    recalled = build_discourse_context("I remember", speaker="Rowan", addressee="Ina")
    alignment = role_alignment(current, recalled, "i")
    assert alignment["available"] is True
    assert alignment["matched"] is False
    assert alignment["present_referents"] == ["sakura"]
    assert alignment["recalled_referents"] == ["rowan"]


def test_module_benchmark_compares_retained_versions_deterministically():
    assert [spec.version for spec in list_benchmark_modules()["discourse"]] == ["V1", "V2"]
    first = benchmark_module("discourse")
    second = benchmark_module("discourse")
    assert [(row.version, row.accuracy, row.correct, row.total) for row in first] == [
        ("V1", 0.0, 0, 8), ("V2", 1.0, 8, 8),
    ]
    assert [(row.version, row.accuracy) for row in second] == [("V1", 0.0), ("V2", 1.0)]


def test_neural_taxonomy_exposes_typed_logic_and_memory_nodes():
    nodes = [
        {"id": "a", "type": "sound"},
        {"id": "b", "node_type": "word"},
        {"id": "c", "network_type": "logic"},
        {"id": "d"},
    ]
    assert normalize_node_type(nodes[2], "logic") == "logic"
    assert count_node_types(nodes) == {"sound": 1, "word": 1, "logic": 1, "memory": 1}


def test_self_question_copy_format_contains_selected_details():
    copied = format_questions([
        {"question": "Why is clarity high?", "first_asked": "t1", "last_updated": "t2", "count": 3},
        {"question": "What changed?", "first_asked": "t3", "count": 1, "resolved_at": "t4", "resolved_reason": "observed"},
    ])
    assert "Why is clarity high?" in copied
    assert "Asked: 3 time(s)" in copied
    assert "Resolved: t4" in copied
    assert "Reason: observed" in copied
