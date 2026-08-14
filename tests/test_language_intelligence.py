from language_intelligence import DiscourseEntityMemory, analyze_utterance, morphology, reading_span_metadata
from module_benchmarks import benchmark_module


def test_role_and_negation_minimal_pairs_keep_distinct_structure():
    left = analyze_utterance("I told you.")
    right = analyze_utterance("You told me.")
    assert left["clauses"][0]["subject"] == "i"
    assert left["clauses"][0]["arguments"]["addressee"] == "you"
    assert right["clauses"][0]["subject"] == "you"
    assert right["clauses"][0]["arguments"]["addressee"] == "me"

    outer = analyze_utterance("I didn't say she stole it.")
    inner = analyze_utterance("I said she didn't steal it.")
    assert [clause["negated"] for clause in outer["clauses"]] == [True, False]
    assert [clause["negated"] for clause in inner["clauses"]] == [False, True]
    assert [token["normalized"] for token in morphology("didn't")] == ["did", "not"]


def test_possessive_and_pragmatic_interpretations_are_factorized():
    ambiguous = analyze_utterance("John gave Peter his coat.")
    explicit = analyze_utterance("John gave Peter Peter's coat.")
    assert ambiguous["referents"][0]["resolved"] is None
    assert {row["meaning"]["possessor"] for row in ambiguous["whole_utterance_interpretations"]} == {"john", "peter"}
    assert explicit["referents"][0]["resolved"] == "peter"
    assert ambiguous["uncertainty"]["referents"]["confidence"] < ambiguous["uncertainty"]["morphology"]["confidence"]

    sincere = analyze_utterance("That's great.", context={"tone": "sincere"})
    sarcastic = analyze_utterance("That's great.", context={"tone": "sarcastic"})
    assert sincere["speech_act"]["interpretation"] == "sincere_positive_evaluation"
    assert sarcastic["speech_act"]["interpretation"] == "sarcastic_negative_evaluation"


def test_discourse_entities_and_reading_hierarchy_persist():
    memory = DiscourseEntityMemory()
    analyze_utterance("John opened the book.", discourse=memory, turn=1)
    later = analyze_utterance("He remembered it.", discourse=memory, turn=2)
    assert any(entity["id"] == "john" for entity in later["discourse_state"]["entities"])
    assert later["referents"][0]["resolved"] == "john"
    span = reading_span_metadata("novel.epub", 3, 20, "Passage")
    assert span["hierarchy"] == ["document", "section", "passage"]
    assert span["parent_ids"] == [span["document_id"], span["section_id"]]


def test_language_benchmark_scores_components_separately():
    v1, v2 = benchmark_module("language_components")
    assert v2.accuracy > v1.accuracy
    assert set(v2.component_scores) == {"composition", "morphology", "constructions", "pragmatics", "discourse", "uncertainty", "counterfactuals", "reading_spans"}
    assert all(score["correct"] == score["total"] for score in v2.component_scores.values())
