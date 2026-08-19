from discourse_context import build_discourse_context, render_referent_gloss, resolution_for, retrieval_routes, role_alignment
from semantic_event import build_native_intent, build_semantic_event
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


def test_deictic_routes_and_ambiguous_glosses_preserve_uncertainty():
    context = build_discourse_context(
        "They remember your garden.", speaker="Sakura",
        addressee={"id": "ina", "name": "Ina", "is_self": True},
        mentioned_entities=("Rowan", "Mira"),
    )
    routes = retrieval_routes(context)
    your = next(route for route in routes if route["surface"] == "your")
    they = resolution_for(context, "they")
    rendered, ambiguity = render_referent_gloss("they", they)
    assert your["retrieval_terms"] == ["ina"]
    assert your["status"] == "resolved"
    assert rendered == "they[?=Rowan/Mira]"
    assert ambiguity["confidence"] == 0.45


def test_semantic_event_precedes_rendering_and_keeps_constructions():
    discourse = build_discourse_context(
        "I did not give you the key.", speaker="Sakura",
        addressee={"id": "ina", "name": "Ina", "is_self": True},
    )
    event = build_semantic_event("I did not give you the key.", discourse)
    assert event["events"][0]["agent"]["id"] == "sakura"
    assert event["events"][0]["predicate"] == "give"
    assert event["events"][0]["negated"] is True
    assert event["events"][0]["arguments"]["recipient"]["referent"]["id"] == "self"
    assert {"agent", "arguments", "negation"} <= set(event["construction_features"])
    intent = build_native_intent(event)
    assert intent["lexical_realizations"][:3] == ["i", "give", "you"]
    assert {item["construction"] for item in intent["grammar"]} == {"tense", "negation"}


def test_module_benchmark_compares_retained_versions_deterministically():
    assert [spec.version for spec in list_benchmark_modules()["discourse"]] == ["V1", "V2", "V3"]
    first = benchmark_module("discourse")
    second = benchmark_module("discourse")
    assert [(row.version, row.accuracy, row.correct, row.total) for row in first] == [
        ("V1", 0.0, 0, 8), ("V2", 1.0, 8, 8), ("V3", 1.0, 5, 5),
    ]
    assert [(row.version, row.accuracy) for row in second] == [("V1", 0.0), ("V2", 1.0), ("V3", 1.0)]


def test_semantic_topology_benchmark_compares_scalar_and_contextual_versions():
    v1, v2 = benchmark_module("semantic_topology")
    assert (v1.version, v2.version) == ("V1", "V2")
    assert v1.source_revision != "working-tree"
    assert v2.accuracy > v1.accuracy
    assert v2.accuracy == 1.0
    assert {
        "composition", "morphology", "constructions", "pragmatics", "discourse",
        "uncertainty", "whole_utterance", "reading_span", "topology", "capacity",
    } <= set(v2.component_scores)


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
