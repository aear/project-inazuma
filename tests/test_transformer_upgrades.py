import json
import sqlite3
from pathlib import Path

from historical_source import historical_text, resolve_revision
from module_benchmarks import (
    TRANSFORMER_V1_REVISION, benchmark_module, list_benchmark_modules,
)
from origin_record import make_origin, normalize_origins
from self_questions_format import format_question
import runtime_state
from transformers.QTransformer import QTransformer
from transformers.heuristic_mirror_transformer import HeuristicMirrorTransformer
from transformers.hindsight_transformer import HindsightTransformer
from transformers.mycelial_transformer import MycelialTransformer
from transformers.seedling_transformer import SeedlingTransformer
from transformers.shadow_transformer import ShadowTransformer


def test_composable_origin_normalizes_legacy_and_formats_chain():
    current = make_origin(
        "BridgeTransformer", "V2", inputs={"symbol": "violence", "logic_tag": "love"},
        references=["frag-7"], trigger="contradiction", event_id="event-2",
    )
    legacy = {"transformer": "SeedlingTransformer", "symbol": "alpha", "timestamp": "t1"}
    rows = normalize_origins([legacy, current])
    rendered = format_question({"question": "Why?", "origins": rows})
    assert rows[0]["module_version"] == "legacy"
    assert rows[1]["schema"] == "ina.origin/V1"
    assert "BridgeTransformer@V2" in rendered
    assert "frag-7" in rendered


def test_self_question_store_composes_and_bounds_origin_history(tmp_path):
    store = tmp_path / "self_questions.json"
    prior = runtime_state._self_questions_path
    runtime_state._self_questions_path = lambda child=None: store
    try:
        for index in range(20):
            runtime_state.seed_self_question(
                "How can violence be love?", child="tester",
                origin=make_origin("BridgeTransformer", "V2", event_id=f"event-{index}"),
            )
        entries = runtime_state._load_self_question_entries("tester")
    finally:
        runtime_state._self_questions_path = prior
    assert entries[0]["count"] == 20
    assert len(entries[0]["origins"]) == 16
    assert entries[0]["origins"][-1]["event_id"] == "event-19"


def test_q_decoder_learns_from_experience_statistics():
    transformer = QTransformer()
    transformer.learn_mapping(
        "000000000", tags=["rest", "repair"],
        self_question="What restored me?", poetic_word="harbour", persist=False,
    )
    result = transformer.collapse_to_meaning("000000000")
    assert result["tags"] == ["rest", "repair"]
    assert result["self_question"] == "What restored me?"
    assert result["poetic_word"] == "harbour"
    assert result["decoder"] == "adaptive"
    assert result["origins"][0]["module"] == "QTransformer"


def test_mirror_learns_separate_audience_models(tmp_path):
    transformer = HeuristicMirrorTransformer(child="tester", root_path=tmp_path)
    for _ in range(8):
        transformer.observe_reaction("Sakura", {"trust": 0.5}, {"trust": 0.9})
        transformer.observe_reaction("Knell", {"trust": 0.5}, {"trust": 0.1})
    sakura = transformer.mirror({}, {"trust": 0.5}, "Sakura")
    knell = transformer.mirror({}, {"trust": 0.5}, "Knell")
    assert sakura["predicted_emotions"]["trust"] > knell["predicted_emotions"]["trust"]
    assert sakura["audience_model_observations"] == 8


def test_hindsight_evaluates_all_claimed_dimensions_and_confidence():
    results = HindsightTransformer().evaluate_claims(
        {"predicted_vector": {"clarity": 0.8, "stress": 0.2, "confidence": 0.9}},
        {"observed_vector": {"clarity": 0.6, "stress": 0.5}},
    )
    assert set(results) == {"clarity", "stress"}
    assert results["clarity"]["error"] == -0.2
    assert results["stress"]["confidence"] == 0.9
    assert results["stress"]["calibration_loss"] > 0


def test_mycelial_ranks_usefulness_before_retaining_links():
    transformer = MycelialTransformer(max_links=1)
    result = transformer.weave(
        {"tags": ["forest"], "text": ["unused", "healing"]},
        {"care": 0.8}, {"forest->unused": 0.05, "forest->healing": 1.0},
    )
    assert result["pathways"][0]["to"] == "text:healing"
    assert result["pathways"][0]["factors"]["historical_usefulness"] == 1.0


def test_seedling_uses_profile_geometry_not_first_character():
    profiles = {"alpha": {"vector": [1, 0]}, "atom": {"vector": [0, 1]}, "beta": {"vector": [0.99, 0.01]}}
    result = SeedlingTransformer(seed=1, similarity_threshold=0.8).germinate(profiles, symbol_profiles=profiles)
    mapping = result["symbol_clusters"]
    assert mapping["alpha"] == mapping["beta"]
    assert mapping["alpha"] != mapping["atom"]


def test_shadow_uses_sqlite_tag_candidates(tmp_path):
    memory = tmp_path / "tester" / "memory"
    fragments = memory / "fragments"
    fragments.mkdir(parents=True)
    (fragments / "candidate.json").write_text(json.dumps({"id": "candidate", "tags": ["unresolved"]}))
    database = memory / "memory_map.sqlite"
    with sqlite3.connect(str(database)) as connection:
        connection.execute("CREATE TABLE fragments(frag_id TEXT, tier TEXT, filename TEXT, tags_json TEXT)")
        connection.execute("CREATE TABLE fragment_tags(tag TEXT, frag_id TEXT, PRIMARY KEY(tag, frag_id))")
        connection.execute("CREATE INDEX idx_fragment_tags_tag ON fragment_tags(tag)")
        connection.execute("INSERT INTO fragments VALUES (?, ?, ?, ?)", ("candidate", "", "candidate.json", '["unresolved"]'))
        connection.execute("INSERT INTO fragment_tags VALUES (?, ?)", ("unresolved", "candidate"))
    transformer = ShadowTransformer(child="tester", root_path=tmp_path, index_db_path=database)
    assert [row["id"] for row in transformer.find_shadow_candidates()] == ["candidate"]
    assert transformer._candidate_source == "index"
    assert transformer._tag_index_used is True


def test_benchmark_v1_comes_from_git_history_and_v2_wins():
    assert "placeholder for emotion bias" in historical_text(
        "transformers/soul_drift.py", TRANSFORMER_V1_REVISION,
    )
    expected = {
        "q_decoder", "bridge_origin", "mirror_audience", "hindsight_claims",
        "mycelial_links", "seedling_clusters", "shadow_candidates", "soul_drift",
        "self_question_origins", "ina_ml_distribution",
    }
    assert expected <= set(list_benchmark_modules())
    for module in expected:
        v1, v2 = benchmark_module(module)
        assert v1.version == "V1" and v2.version == "V2"
        assert v1.source_revision == resolve_revision(TRANSFORMER_V1_REVISION)
        assert v2.source_revision == "working-tree"
        assert v2.accuracy > v1.accuracy
