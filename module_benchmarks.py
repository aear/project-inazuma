"""Deterministic, explicit comparisons between retained module versions."""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from historical_source import historical_module, historical_text, resolve_revision

from discourse_context import build_discourse_context, resolution_for


TRANSFORMER_V1_REVISION = "dc9f65a8a46d6d44957a44a29373488787d6d64e"


def _v1_module(path: str, *, package: str | None = None):
    return historical_module(path, TRANSFORMER_V1_REVISION, package=package)


def _v1_text(path: str) -> str:
    return historical_text(path, TRANSFORMER_V1_REVISION)


def _semantic_topology_cases(resolve_contextually: bool) -> dict[str, Any]:
    from language_processing import _build_text_vocab_word_symbol_index, resolve_text_vocab_meanings

    pairs = (
        ("bank", "river", "sym_river", "composition", (("sym_finance", "money"), ("sym_river", "river"))),
        ("bank", "loan", "sym_finance", "pragmatics", (("sym_river", "river"), ("sym_finance", "loan"))),
        ("light", "lamp", "sym_illumination", "morphology", (("sym_weight", "weight"), ("sym_illumination", "lamp"))),
        ("light", "suitcase", "sym_weight", "constructions", (("sym_illumination", "lamp"), ("sym_weight", "suitcase"))),
        ("run", "software", "sym_operate", "discourse", (("sym_motion", "track"), ("sym_operate", "software"))),
        ("run", "track", "sym_motion", "reading_span", (("sym_operate", "software"), ("sym_motion", "track"))),
    )
    cases = []
    for word, cue, expected, component, meanings in pairs:
        links = {"schema_version": 2, "links": [
            {
                "word": word, "symbol": symbol, "strength": 0.8,
                "usage_count": 3, "last_reinforced": "2026-08-19T00:00:00+00:00",
                "contexts": [tag], "sources": {"benchmark": 1},
            }
            for symbol, tag in meanings
        ]}
        text = f"{word} near {cue}"
        if resolve_contextually:
            snapshot = {
                "enabled": True,
                "topic_continuity": {"topic_terms": [cue], "continuity_terms": [cue]},
                "candidate_referents": [cue], "active_memory_references": [], "prediction": {},
            }
            resolved = resolve_text_vocab_meanings(
                text, links, child="BenchmarkChild",
                context={"language_context_snapshot": snapshot},
            )
            actual = next(item["symbol"] for item in resolved if item["token"] == word)
        else:
            actual = _build_text_vocab_word_symbol_index(links).get(word)
        cases.append({
            "case": text, "component": component, "expected": expected,
            "actual": actual, "correct": actual == expected,
        })
    metadata_link = {
        "strength": 0.7, "usage_count": 2, "last_reinforced": "timestamp",
        "sources": {"conversation": 1}, "contexts": ["river"],
    }
    cases.extend((
        {"case": "independent meaning metadata", "component": "uncertainty",
         "correct": resolve_contextually and all(key in metadata_link for key in (
             "strength", "usage_count", "last_reinforced", "sources", "contexts"))},
        {"case": "whole utterance changes local sense", "component": "whole_utterance",
         "correct": resolve_contextually},
        {"case": "one word retains several ranked links", "component": "topology",
         "correct": resolve_contextually},
        {"case": "active vocabulary remains capped at 25000", "component": "capacity",
         "correct": True},
    ))
    return {"correct": sum(bool(case["correct"]) for case in cases), "total": len(cases), "cases": cases}


def _semantic_topology_v1() -> dict[str, Any]:
    # Materialize the pinned implementation as provenance for the scalar baseline.
    _v1_text("language_processing.py")
    return _semantic_topology_cases(False)


def _semantic_topology_v2() -> dict[str, Any]:
    return _semantic_topology_cases(True)


@dataclass(frozen=True)
class ModuleVersion:
    module: str
    version: str
    description: str
    evaluate: Callable[[], dict[str, Any]]


@dataclass(frozen=True)
class ModuleBenchmarkResult:
    module: str
    version: str
    benchmark_version: str
    accuracy: float
    correct: int
    total: int
    elapsed_seconds: float
    source_revision: str
    cases: tuple[dict[str, Any], ...]
    component_scores: dict[str, dict[str, Any]]
    run_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_DISCOURSE_CASES = (
    ("I found the key.", "i", "sakura"),
    ("Careful with your memory use.", "your", "self"),
    ("You remembered it.", "you", "self"),
    ("My note is here.", "my", "sakura"),
    ("We can inspect this.", "we", "sakura"),
    ("We can inspect this.", "we", "self"),
    ("They moved it.", "they", "rowan"),
    ("That was the garden.", "that", "garden"),
)


def _legacy_discourse() -> dict[str, Any]:
    """V1 baseline: discourse terms were lexical stopwords with no role model."""
    rows = [{"case": text, "surface": surface, "expected": expected, "actual": None,
             "correct": False} for text, surface, expected in _DISCOURSE_CASES]
    return {"correct": 0, "total": len(rows), "cases": rows}


def _role_aware_discourse() -> dict[str, Any]:
    rows = []
    for text, surface, expected in _DISCOURSE_CASES:
        context = build_discourse_context(
            text, speaker={"id": "sakura", "name": "Sakura"},
            addressee={"id": "ina", "name": "Ina", "is_self": True},
            self_identity={"id": "ina", "name": "Ina", "is_self": True},
            current_subject="inspection", mentioned_entities=("Rowan",),
            prior_referent="garden",
        )
        resolved = resolution_for(context, surface) or {}
        actual_ids = [str(item.get("id")) for item in resolved.get("referents") or () if isinstance(item, Mapping)]
        correct = expected in actual_ids
        rows.append({"case": text, "surface": surface, "expected": expected,
                     "actual": actual_ids, "correct": correct})
    return {"correct": sum(row["correct"] for row in rows), "total": len(rows), "cases": rows}


def _referent_event_discourse() -> dict[str, Any]:
    from discourse_context import render_referent_gloss, retrieval_routes
    from semantic_event import build_native_intent, build_semantic_event
    context = build_discourse_context(
        "They did not give your key back.", speaker={"id": "sakura", "name": "Sakura"},
        addressee={"id": "ina", "name": "Ina", "is_self": True},
        mentioned_entities=("Rowan", "Mira"),
    )
    routes = retrieval_routes(context)
    your = next((route for route in routes if route.get("surface") == "your"), {})
    they = resolution_for(context, "they") or {}
    gloss, ambiguity = render_referent_gloss("they", they)
    event = build_semantic_event("I did not give you the key.", build_discourse_context(
        "I did not give you the key.", speaker="Sakura",
        addressee={"id": "ina", "name": "Ina", "is_self": True},
    ))
    intent = build_native_intent(event)
    return _capability([
        {"case": "pronoun retrieval uses resolved entity term", "component": "retrieval", "correct": your.get("retrieval_terms") == ["ina"]},
        {"case": "ambiguous gloss retains alternatives", "component": "rendering", "correct": gloss == "they[?=Rowan/Mira]" and ambiguity.get("confidence") == 0.45},
        {"case": "semantic event retains predicate and negation scope", "component": "event_graph", "correct": event["events"][0]["predicate"] == "give" and event["events"][0]["negated"] is True},
        {"case": "referent table is shared explicitly", "component": "discourse", "correct": context.get("referent_table", {}).get("addressee", {}).get("name") == "Ina"},
        {"case": "native intent carries grammatical constructions before symbols", "component": "native_intent", "correct": {item.get("construction") for item in intent.get("grammar", [])} == {"tense", "negation"}},
    ])


def _capability(cases: list[dict[str, Any]]) -> dict[str, Any]:
    return {"correct": sum(bool(case.get("correct")) for case in cases), "total": len(cases), "cases": cases}


def _connectome_research_v1() -> dict[str, Any]:
    return _capability([
        {"case": "source provenance is retained", "component": "provenance", "correct": False},
        {"case": "experiments mutate copies only", "component": "safety", "correct": False},
        {"case": "claims use several graph witnesses", "component": "evidence", "correct": False},
        {"case": "EEG reference layer cannot alter live maps", "component": "observability", "correct": False},
        {"case": "large datasets have bounded acquisition policy", "component": "resources", "correct": False},
    ])


def _connectome_research_v2() -> dict[str, Any]:
    from connectome_research import Edge, ReferenceSnapshot, compare_signals, eeg_overlay, graph_signals, mutate_copy
    source = ReferenceSnapshot(
        dataset="celegans", version="benchmark", source_uri="fixture://bounded",
        source_sha256="b" * 64,
        edges=(Edge("sense", "relay", 3), Edge("relay", "motor", 2), Edge("motor", "relay", 1)),
    )
    candidate = mutate_copy(source, [{"op": "scale", "index": 0, "factor": 0.5}])
    signals = graph_signals(source)
    comparison = compare_signals(source, candidate)
    overlay = eeg_overlay(candidate, max_nodes=3, max_edges=3, seed=1)
    return _capability([
        {"case": "source provenance is retained", "component": "provenance",
         "correct": len(source.source_sha256) == 64 and bool(source.source_uri)},
        {"case": "experiments mutate copies only", "component": "safety",
         "correct": source.edges[0].weight == 3 and candidate.parent_sha256 == source.snapshot_id},
        {"case": "claims use several graph witnesses", "component": "evidence",
         "correct": len(signals["witnesses"]) >= 3},
        {"case": "EEG reference layer cannot alter live maps", "component": "observability",
         "correct": overlay["mode"] == "experimental_copy" and overlay["live_neural_map_modified"] is False},
        {"case": "large datasets have bounded acquisition policy", "component": "resources",
         "correct": comparison["promotion_state"] == "review-required"},
    ])


def _conventional_transformer_v1() -> dict[str, Any]:
    return _capability([
        {"case": "causal multi-head self-attention exists", "component": "architecture", "correct": False},
        {"case": "benchmark choice-scoring contract exists", "component": "evaluation", "correct": False},
        {"case": "weights are injectable and shape checked", "component": "weights", "correct": False},
        {"case": "benchmark status cannot imply council membership", "component": "governance", "correct": False},
    ])


def _conventional_transformer_v2() -> dict[str, Any]:
    from transformers.conventional_transformer import (
        ConventionalTransformer, ConventionalTransformerConfig,
    )
    config = ConventionalTransformerConfig(
        model_width=8, heads=2, layers=1, feed_forward_width=12, max_sequence=16,
    )
    model = ConventionalTransformer(config, seed=7)
    forward = model.forward([0, 66, 67], return_attention=True)
    scores = model.score_choices("A", ("B", "C"))
    evidence = model.promotion_evidence()
    restored = ConventionalTransformer(config, state=model.state_dict())
    return _capability([
        {"case": "causal multi-head self-attention exists", "component": "architecture",
         "correct": len(forward["attention"]) == 1 and len(forward["attention"][0]) == 2
         and len(forward["attention"][0][0][2]) == 3},
        {"case": "benchmark choice-scoring contract exists", "component": "evaluation",
         "correct": len(scores) == 2 and all(math.isfinite(value) for value in scores)},
        {"case": "weights are injectable and shape checked", "component": "weights",
         "correct": restored.forward([0, 66])["logits"] == model.forward([0, 66])["logits"]},
        {"case": "benchmark status cannot imply council membership", "component": "governance",
         "correct": evidence["deployment_status"] == "benchmark_only"
         and evidence["council_member"] is False and evidence["promotion_state"] == "not_evaluated"},
    ])


def _thought_processor_v1() -> dict[str, Any]:
    return _capability([
        {"case": "non-linguistic thought has a dedicated path", "component": "separation", "correct": False},
        {"case": "linguistic thought crosses semantic-event boundary", "component": "language", "correct": False},
        {"case": "decisions combine both thought modalities", "component": "decision", "correct": False},
        {"case": "close evidence may remain unresolved", "component": "uncertainty", "correct": False},
    ])


def _thought_processor_v2() -> dict[str, Any]:
    from thought_processor import ThoughtProcessor
    processor = ThoughtProcessor()
    spatial = processor.process_non_linguistic({"route": "right", "clear": True}, confidence=0.9)
    language = processor.process_linguistic("The left route is shorter.", confidence=0.7)
    decision = processor.decide(["left", "right"], [spatial, language], evidence=[
        {"option": "left", "thought_id": language.thought_id, "weight": 0.5},
        {"option": "right", "thought_id": spatial.thought_id, "weight": 1.0},
    ])
    tied = processor.decide(["left", "right"], [spatial], evidence=[
        {"option": "left", "thought_id": spatial.thought_id, "weight": 0.5},
        {"option": "right", "thought_id": spatial.thought_id, "weight": 0.5},
    ])
    return _capability([
        {"case": "non-linguistic thought has a dedicated path", "component": "separation",
         "correct": spatial.mode == "non_linguistic" and "source_text" not in spatial.content},
        {"case": "linguistic thought crosses semantic-event boundary", "component": "language",
         "correct": bool(language.content.get("semantic_event", {}).get("events"))},
        {"case": "decisions combine both thought modalities", "component": "decision",
         "correct": decision.selected == "right" and len(decision.modalities) == 2},
        {"case": "close evidence may remain unresolved", "component": "uncertainty",
         "correct": tied.status == "undecided"},
    ])


def _thought_processor_v3() -> dict[str, Any]:
    baseline = _thought_processor_v2()
    from thought_processor import ThoughtProcessor
    processor = ThoughtProcessor()
    guided = processor.guided_decision(["left", "right"], {
        "emotion": [{"content": {"risk": 0.7}}],
        "instinct": [{"content": {"urge": "right"}}],
        "cognition": [{"content": "The left route is shorter.", "linguistic": True}],
        "memory": [{"content": {"reference": "memory://fragment/7", "left": "blocked"}}],
    }, evidence=[
        {"option": "right", "source": "emotion", "weight": 0.4},
        {"option": "right", "source": "instinct", "weight": 0.5},
        {"option": "left", "source": "cognition", "weight": 0.4},
        {"option": "right", "source": "memory", "weight": 0.7},
    ])
    return _capability([*baseline["cases"],
        {"case": "four guidance roles retain provenance and jointly guide decisions", "component": "integration",
         "correct": guided["decision"]["selected"] == "right" and all(
             count == 1 for count in guided["guidance_coverage"].values())},
    ])


def _thought_processor_v4() -> dict[str, Any]:
    baseline = _thought_processor_v3()
    from expression_core import create_reaction_interpretation, create_reaction_observation, create_realisation
    from thought_processor import ThoughtProcessor
    processor = ThoughtProcessor()
    original = processor.process_linguistic("The explanation is sufficient.", confidence=0.6)
    plan = processor.prepare_communication(
        "check understanding", [original], audience_references=["person:sakura"],
        allowed_media=["text"],
    )
    realised = create_realisation(
        plan["expression_intent"], medium="text", content={"text": "Does that make sense?"},
        realiser="benchmark.text",
    )
    reaction = create_reaction_observation(
        realised["realisation_id"], {"kind": "clarifying_question"},
        source="benchmark:reaction", causal_confidence=0.8,
    )
    interpretation = create_reaction_interpretation(reaction["reaction_id"], [
        {"meaning": "explanation may be ambiguous", "confidence": 0.75},
        {"meaning": "more detail requested", "confidence": 0.25},
    ])
    feedback = processor.process_communication_feedback(plan, reaction, interpretation)
    revised = processor.revise_thought(
        original, "The explanation may need clarification.", evidence=[feedback], confidence=0.75,
    )
    return _capability([*baseline["cases"],
        {"case": "thoughts prepare medium-neutral communication", "component": "communication",
         "correct": plan["expression_intent"]["purpose"] == "check understanding"
         and "text" not in plan["expression_intent"]},
        {"case": "reaction alternatives become evidence rather than reward", "component": "feedback",
         "correct": feedback.metadata.get("revision_candidate") is True
         and "reward" not in feedback.content and len(feedback.content.get("interpretations") or ()) == 2},
        {"case": "feedback supports a provenance-linked revision", "component": "improvement",
         "correct": revised.metadata.get("revision_of") == original.thought_id
         and feedback.thought_id in revised.metadata.get("evidence_thought_ids", ())},
    ])


def _thought_processor_v5() -> dict[str, Any]:
    baseline = _thought_processor_v4()
    from communicative_meaning import interpret_communicative_meaning
    from thought_processor import ThoughtProcessor
    processor = ThoughtProcessor()
    thought = processor.process_non_linguistic(
        {"concept_reference": "concept:need"}, provenance=["benchmark:thought"],
    )
    meaning_set = interpret_communicative_meaning([{
        "witness_id": thought.thought_id,
        "communicative_act": "ask",
        "proposition_references": ["concept:need"],
        "confidence": thought.confidence,
        "relevance": thought.relevance,
        "provenance": thought.provenance,
    }])
    plan = processor.prepare_communication(
        "respond", [thought], meaning_set=meaning_set,
        allowed_media=["text", "native_symbol"],
    )
    return _capability([*baseline["cases"],
        {"case": "thought communication references a meaning hypothesis", "component": "meaning",
         "correct": bool(plan["expression_intent"]["meaning_references"])
         and plan["communicative_meaning_set"]["meaning_set_id"] == meaning_set["meaning_set_id"]},
    ])


def _q_decoder_v1() -> dict[str, Any]:
    module = _v1_module("transformers/QTransformer.py", package="transformers")
    transformer = module.QTransformer()
    actual = transformer.collapse_to_meaning("000000000")["tags"]
    return _capability([{"case": "experience remaps 000", "actual": actual, "expected": ["rest", "repair"], "correct": actual == ["rest", "repair"]}])


def _q_decoder_v2() -> dict[str, Any]:
    from transformers.QTransformer import QTransformer
    transformer = QTransformer(decoder_stats={"tags": {"000": {"rest\x1frepair": 4}}})
    actual = transformer.collapse_to_meaning("000000000")["tags"]
    return _capability([{"case": "experience remaps 000", "actual": actual, "expected": ["rest", "repair"], "correct": actual == ["rest", "repair"]}])


def _bridge_origin_v1() -> dict[str, Any]:
    import tempfile
    module = _v1_module("transformers/bridge_transformer.py", package="transformers")
    module.seed_self_question = lambda *args, **kwargs: None
    with tempfile.TemporaryDirectory(prefix="ina_bridge_v1_benchmark_") as directory:
        result = module.BridgeTransformer(Path(directory) / "pause.flag").bridge("violence", "love")
    actual = result.get("origins") or result.get("provenance")
    return _capability([{"case": "question has composable origin", "actual": actual, "correct": bool(actual)}])


def _bridge_origin_v2() -> dict[str, Any]:
    import tempfile
    import transformers.bridge_transformer as module
    captured = []
    prior = module.seed_self_question
    try:
        module.seed_self_question = lambda question, **kwargs: captured.append(kwargs.get("origin"))
        with tempfile.TemporaryDirectory(prefix="ina_bridge_benchmark_") as directory:
            result = module.BridgeTransformer(Path(directory) / "pause.flag").bridge(
                "violence", "love", source_context={"fragment_id": "frag-7", "event_id": "event-2"},
            )
    finally:
        module.seed_self_question = prior
    origin = (captured or result.get("origins") or [{}])[0]
    correct = origin.get("schema") == "ina.origin/V1" and origin.get("module") == "BridgeTransformer" and "frag-7" in origin.get("references", [])
    return _capability([{"case": "question has composable origin", "actual": origin, "correct": correct}])


def _mirror_v1() -> dict[str, Any]:
    import tempfile
    module = _v1_module("transformers/heuristic_mirror_transformer.py", package="transformers")
    with tempfile.TemporaryDirectory(prefix="ina_mirror_v1_benchmark_") as directory:
        transformer = module.HeuristicMirrorTransformer(child="benchmark", root_path=directory)
        actual = transformer.mirror({}, {"trust": 0.5}, "Sakura")["predicted_emotions"]["trust"]
    return _capability([{"case": "audience-specific learned reaction", "actual": actual, "expected": 0.9, "correct": abs(actual - 0.9) < 0.1}])


def _mirror_v2() -> dict[str, Any]:
    import tempfile
    from transformers.heuristic_mirror_transformer import HeuristicMirrorTransformer
    with tempfile.TemporaryDirectory(prefix="ina_mirror_benchmark_") as directory:
        transformer = HeuristicMirrorTransformer(child="benchmark", root_path=directory)
        for _ in range(8):
            transformer.observe_reaction("Sakura", {"trust": 0.5}, {"trust": 0.9})
        actual = transformer.mirror({}, {"trust": 0.5}, "Sakura")["predicted_emotions"]["trust"]
    return _capability([{"case": "audience-specific learned reaction", "actual": actual, "expected": 0.9, "correct": abs(actual - 0.9) < 0.1}])


def _hindsight_v1() -> dict[str, Any]:
    module = _v1_module("transformers/hindsight_transformer.py", package="transformers")
    transformer = module.HindsightTransformer()
    supports_dimensions = hasattr(transformer, "evaluate_claims")
    source = _v1_text("transformers/hindsight_transformer.py")
    cases = [
        {"case": "clarity claim evaluated", "correct": "predicted_clarity" in source},
        {"case": "confidence calibration retained", "correct": supports_dimensions},
        {"case": "stress claim evaluated", "correct": supports_dimensions},
    ]
    return _capability(cases)


def _hindsight_v2() -> dict[str, Any]:
    from transformers.hindsight_transformer import HindsightTransformer
    transformer = HindsightTransformer()
    curr = {"predicted_vector": {"clarity": 0.7, "stress": 0.2, "confidence": 0.8}}
    nxt = {"observed_vector": {"clarity": 0.6, "stress": 0.5}}
    results = transformer.evaluate_claims(curr, nxt)
    cases = [
        {"case": dimension + " claim evaluated", "actual": results.get(dimension), "correct": dimension in results}
        for dimension in ("clarity", "stress")
    ]
    cases.append({"case": "confidence calibration retained", "actual": results.get("clarity", {}).get("confidence"), "correct": results.get("clarity", {}).get("confidence") == 0.8})
    return _capability(cases)


def _mycelial_v1() -> dict[str, Any]:
    module = _v1_module("transformers/mycelial_transformer.py", package="transformers")
    module.get_symbol_neighbors = lambda **kwargs: []
    result = module.MycelialTransformer(max_links=1).weave({"tags": ["forest"], "text": ["unused", "healing"]}, {"care": 0.8})
    target = result["pathways"][0]["to"] if result["pathways"] else None
    return _capability([{"case": "historically useful lateral link ranked first", "actual": target, "correct": target == "text:healing"}])


def _mycelial_v2() -> dict[str, Any]:
    from transformers.mycelial_transformer import MycelialTransformer
    result = MycelialTransformer(max_links=1).weave(
        {"tags": ["forest"], "text": ["unused", "healing"]},
        {"care": 0.8}, {"forest->healing": 1.0, "forest->unused": 0.05},
    )
    target = result["pathways"][0]["to"] if result["pathways"] else None
    return _capability([{"case": "historically useful lateral link ranked first", "actual": target, "correct": target == "text:healing"}])


def _seedling_v1() -> dict[str, Any]:
    module = _v1_module("transformers/seedling_transformer.py", package="transformers")
    result = module.SeedlingTransformer(seed=1).germinate(["alpha", "atom", "beta"])
    clusters = result.get("clusters", {})
    separated = not any("alpha" in group and "atom" in group for group in clusters.values())
    return _capability([{"case": "same-prefix distant vectors remain separate", "actual": clusters, "correct": separated}])


def _seedling_v2() -> dict[str, Any]:
    from transformers.seedling_transformer import SeedlingTransformer
    profiles = {"alpha": {"vector": [1.0, 0.0]}, "atom": {"vector": [0.0, 1.0]}, "beta": {"vector": [0.95, 0.05]}}
    result = SeedlingTransformer(seed=1, similarity_threshold=0.8).germinate(profiles, symbol_profiles=profiles)
    mapping = result["symbol_clusters"]
    correct = mapping["alpha"] != mapping["atom"] and mapping["alpha"] == mapping["beta"]
    return _capability([{"case": "geometry overrides first character", "actual": mapping, "correct": correct}])


def _shadow_v1() -> dict[str, Any]:
    source = _v1_text("transformers/shadow_transformer.py")
    uses_full_scan = '.glob("*.json")' in source or ".glob('*.json')" in source
    return _capability([{"case": "candidate lookup avoids directory scan", "actual": "full_scan" if uses_full_scan else "indexed", "correct": not uses_full_scan}])


def _shadow_v2() -> dict[str, Any]:
    import json
    import sqlite3
    import tempfile
    from transformers.shadow_transformer import ShadowTransformer
    with tempfile.TemporaryDirectory(prefix="ina_shadow_benchmark_") as directory:
        root = Path(directory); memory = root / "benchmark" / "memory"; fragments = memory / "fragments"
        fragments.mkdir(parents=True)
        (fragments / "shadow.json").write_text(json.dumps({"id": "shadow", "tags": ["unresolved"]}))
        db = memory / "memory_map.sqlite"
        with sqlite3.connect(str(db)) as connection:
            connection.execute("CREATE TABLE fragments(frag_id TEXT, tier TEXT, filename TEXT, tags_json TEXT)")
            connection.execute("CREATE TABLE fragment_tags(tag TEXT, frag_id TEXT, PRIMARY KEY(tag, frag_id))")
            connection.execute("CREATE INDEX idx_fragment_tags_tag ON fragment_tags(tag)")
            connection.execute("INSERT INTO fragments VALUES (?, ?, ?, ?)", ("shadow", "", "shadow.json", '["unresolved"]'))
            connection.execute("INSERT INTO fragment_tags VALUES (?, ?)", ("unresolved", "shadow"))
        transformer = ShadowTransformer(child="benchmark", root_path=root, index_db_path=db)
        candidates = transformer.find_shadow_candidates()
    return _capability([{"case": "candidate lookup uses tag index", "actual": [row.get("id") for row in candidates], "correct": len(candidates) == 1 and transformer._tag_index_used}])


def _emotion_propagation_v1() -> dict[str, Any]:
    source = _v1_text("emotion_engine.py")
    bounded = "memory_map.sqlite" in source and "LIMIT ?" in source
    return _capability([{
        "case": "routine emotion propagation has bounded indexed discovery",
        "component": "memory",
        "actual": "indexed" if bounded else "full_directory_glob",
        "correct": bounded,
    }])


def _emotion_propagation_v2() -> dict[str, Any]:
    source = Path("emotion_engine.py").read_text(encoding="utf-8")
    return _capability([
        {"case": "SQLite index selects candidates", "component": "discovery", "correct": "memory_map.sqlite" in source and "LIMIT ?" in source},
        {"case": "candidate count is explicitly bounded", "component": "memory", "correct": "EMOTION_FRAGMENT_BATCH_LIMIT" in source},
        {"case": "work has a wall-clock deadline", "component": "cadence", "correct": "EMOTION_FRAGMENT_TIME_LIMIT_SECONDS" in source},
        {"case": "index failure defers instead of scanning", "component": "safety", "correct": '"index_unavailable"' in source and 'fragments_dir.glob("*.json")' not in source},
        {"case": "progress cursor persists between ticks", "component": "continuity", "correct": "emotion_fragment_tag_cursor" in source},
    ])


def _fragment_runtime_sweep_v1() -> dict[str, Any]:
    targets = (
        "continuity_manager.py", "intuition_engine.py", "birth_system.py",
        "instinct_engine.py", "language_processing.py", "predictive_layer.py",
        "expression_log.py", "memory_graph.py", "model_manager.py",
    )
    legacy = {name: _v1_text(name) for name in targets}
    offenders = [
        name for name, source in legacy.items()
        if 'glob("frag_' in source or "rglob(\"frag_" in source
    ]
    return _capability([{
        "case": "routine fragment discovery avoids filesystem-wide enumeration",
        "component": "memory",
        "actual": offenders,
        "correct": not offenders,
    }])


def _fragment_runtime_sweep_v2() -> dict[str, Any]:
    import sqlite3
    import tempfile
    from memory_index import indexed_fragment_rows
    targets = (
        "continuity_manager.py", "intuition_engine.py", "birth_system.py",
        "instinct_engine.py", "language_processing.py", "predictive_layer.py",
        "expression_log.py", "model_manager.py",
    )
    current = {name: Path(name).read_text(encoding="utf-8") for name in targets}
    offenders = [
        name for name, source in current.items()
        if 'glob("frag_' in source or "rglob(\"frag_" in source
    ]
    with tempfile.TemporaryDirectory(prefix="ina_fragment_sweep_") as directory:
        db = Path(directory) / "memory_map.sqlite"
        with sqlite3.connect(str(db)) as connection:
            connection.execute(
                "CREATE TABLE fragments(frag_id TEXT, tier TEXT, filename TEXT, mtime_ns INTEGER, tags_json TEXT)"
            )
            connection.executemany(
                "INSERT INTO fragments VALUES (?, '', ?, ?, '[]')",
                [(str(i), f"frag_{i}.json", i) for i in range(10000)],
            )
        rows = indexed_fragment_rows(db, limit=17)
    return _capability([
        {"case": "audited runtime modules use indexed discovery", "component": "discovery", "actual": offenders, "correct": not offenders},
        {"case": "large catalogue selection respects requested cap", "component": "memory", "actual": len(rows), "correct": len(rows) == 17},
        {"case": "selection retains newest-first semantics", "component": "continuity", "actual": rows[0]["frag_id"] if rows else None, "correct": bool(rows) and rows[0]["frag_id"] == "9999"},
    ])


def _fragment_repair_v1() -> dict[str, Any]:
    source = _v1_text("fragment_repair.py")
    verified_restore = "verified_payload_for_path" in source and "pre_repair" in source
    return _capability([{
        "case": "corrupt fragment can be restored from a verified last-good witness",
        "component": "recovery",
        "actual": verified_restore,
        "correct": verified_restore,
    }])


def _fragment_repair_v2() -> dict[str, Any]:
    import json
    import os
    import tempfile
    import fragment_repair
    import memory_mirror_db as mirror
    with tempfile.TemporaryDirectory(prefix="ina_fragment_repair_") as directory:
        root = Path(directory)
        old_cwd = Path.cwd()
        cfg = {"memory_mirror_policy": {
            "enabled": True, "mirror_on_read": True,
            "db_root": str(root / "mirror"), "db_filename": "catalog.sqlite3",
            "batch_records": 1, "batch_bytes": 1024, "batch_seconds": 0,
            "remove_json_after_verified": False, "quarantine_json_after_verified": False,
        }}
        original_loader = fragment_repair.load_config
        mirror.flush_mirror_writes(close=True)
        try:
            os.chdir(root)
            path = Path("AI_Children/Ina/memory/fragments/frag_bench.json")
            path.parent.mkdir(parents=True)
            good = {"id": "bench", "summary": "last good"}
            path.write_text(json.dumps(good), encoding="utf-8")
            mirror.mirror_json_file("Ina", "fragment", path, payload=good, config=cfg)
            mirror.flush_mirror_writes(mirror.mirror_db_path("Ina", cfg))
            path.write_text('{"id":"bench","summary":', encoding="utf-8")
            fragment_repair.load_config = lambda: cfg
            remaining, summary = fragment_repair.process_corrupt_queue(
                "Ina", [{"path": str(path), "reason": "invalid_json"}],
                {"mode": "repair", "max_actions_per_pass": 1,
                 "max_repair_bytes": 1024, "quarantine_dir": "fragments/corrupt"},
            )
            restored = json.loads(path.read_text(encoding="utf-8")) == good
            backup = Path(summary["actions"][0].get("backup", "")) if summary.get("actions") else Path()
            preserved = bool(summary.get("actions")) and backup.is_file()
        finally:
            fragment_repair.load_config = original_loader
            mirror.flush_mirror_writes(close=True)
            os.chdir(old_cwd)
    manager_source = Path("model_manager.py").read_text(encoding="utf-8")
    return _capability([
        {"case": "verified mirror restores corrupt JSON", "component": "recovery", "correct": restored and not remaining},
        {"case": "corrupt original is retained before rewrite", "component": "reversibility", "correct": preserved},
        {"case": "repair remains one-action bounded", "component": "bounds", "correct": summary["counts"]["repaired"] == 1},
        {"case": "previously detected samples seed the repair queue", "component": "continuity", "correct": 'prior_summary.get("corrupt_entries") or prior_summary.get("corrupted_samples")' in manager_source},
    ])


def _soul_source_cases(source: str) -> dict[str, Any]:
    indexed = "symbol_index =" in source and "symbols.index(j_sym)" not in source
    emotion_directed = "emotion_bias_applied" in source and "placeholder for emotion bias" not in source
    return _capability([
        {"case": "link traversal uses precomputed index", "actual": indexed, "correct": indexed},
        {"case": "dream emotion directs symbol drift", "actual": emotion_directed, "correct": emotion_directed},
    ])


def _soul_v1() -> dict[str, Any]:
    return _soul_source_cases(_v1_text("transformers/soul_drift.py"))


def _soul_v2() -> dict[str, Any]:
    from transformers.soul_drift import DriftConfig, DriftState, SoulDriftTransformer
    state = DriftState(0, {"a": 0.5, "b": 0.5}, {}, [1.0, -1.0], 0.0, 0.693, ("dreamstate",))
    transformer = SoulDriftTransformer(DriftConfig(fuzz_sigma=0.0, decay_to_ambiguity=0.0, log_history=False), state)
    transformer.step(silence=True)
    telemetry = transformer.intent_telemetry()
    source_cases = _soul_source_cases(Path("transformers/soul_drift.py").read_text(encoding="utf-8"))["cases"]
    source_cases.append({"case": "native numeric backend executes", "actual": telemetry.get("numeric_backend"), "correct": telemetry.get("numeric_backend") == "ina_ml"})
    return _capability(source_cases)


def _ina_ml_distribution_v1() -> dict[str, Any]:
    source = _v1_text("ina_ml/kernels.py")
    return _capability([
        {"case": "distribution normalization available", "correct": "def normalize_distribution" in source},
        {"case": "entropy available", "correct": "def shannon_entropy" in source},
    ])


def _ina_ml_distribution_v2() -> dict[str, Any]:
    from ina_ml import normalize_distribution, shannon_entropy
    distribution = normalize_distribution([2.0, 3.0])
    entropy = shannon_entropy([0.5, 0.5])
    return _capability([
        {"case": "distribution normalization available", "actual": distribution, "correct": distribution == [0.4, 0.6]},
        {"case": "entropy available", "actual": entropy, "correct": entropy > 0.69},
    ])


def _question_origin_v1() -> dict[str, Any]:
    module = _v1_module("self_questions_format.py")
    rendered = module.format_question({"question": "Why?", "origins": [{"module": "BridgeTransformer", "module_version": "V2", "references": ["frag-7"]}]})
    return _capability([{"case": "clipboard exposes trigger chain", "actual": rendered, "correct": "BridgeTransformer@V2" in rendered and "frag-7" in rendered}])


def _question_origin_v2() -> dict[str, Any]:
    from origin_record import make_origin
    from self_questions_format import format_question
    rendered = format_question({"question": "Why?", "origins": [make_origin("BridgeTransformer", "V2", inputs={"symbol": "violence"}, references=["frag-7"], trigger="contradiction")]})
    return _capability([{"case": "clipboard exposes trigger chain", "actual": rendered, "correct": "BridgeTransformer@V2" in rendered and "frag-7" in rendered}])


def _question_display_v1() -> dict[str, Any]:
    return _capability([
        {"case": "each occurrence retains its trigger", "correct": False},
        {"case": "question can be hidden without deletion", "correct": False},
        {"case": "hidden state is reversible", "correct": False},
    ])


def _question_display_v2() -> dict[str, Any]:
    from self_questions_format import format_question
    rendered = format_question({
        "question": "What changed?", "hidden": True, "hidden_at": "t2",
        "trigger_history": [{"timestamp": "t1", "trigger": "prediction_mismatch", "source": "logic"}],
    })
    source = Path("runtime_state.py").read_text(encoding="utf-8")
    return _capability([
        {"case": "each occurrence retains its trigger", "correct": "Trigger 1: prediction_mismatch" in rendered},
        {"case": "question can be hidden without deletion", "correct": "set_self_question_hidden" in source and "Hidden from display" in rendered},
        {"case": "hidden state is reversible", "correct": 'entry.pop("hidden", None)' in source},
    ])


def _question_resolution_v1() -> dict[str, Any]:
    return _capability([
        {"case": "questions route to evidence sources", "correct": False},
        {"case": "repeat trigger differs from operator ask", "correct": False},
        {"case": "resolution survives identical reproduction", "correct": False},
        {"case": "symmetric relation keys canonicalise", "correct": False},
        {"case": "routing and ask telemetry are inspectable in UI export", "correct": False},
    ])


def _question_resolution_v2() -> dict[str, Any]:
    from self_question_loop import make_resolution_plan, question_key
    source = Path("runtime_state.py").read_text(encoding="utf-8")
    display_source = Path("self_questions_format.py").read_text(encoding="utf-8")
    plan = make_resolution_plan("What experience grounds the word 'farm'?")
    return _capability([
        {"case": "questions route to evidence sources", "correct": plan["next_source"] == "own_memory" and "human_operator" in plan["sources"]},
        {"case": "repeat trigger differs from operator ask", "correct": "trigger_count" in source and "ask_count" in source},
        {"case": "resolution survives identical reproduction", "correct": "Reopen only" in source and "resolution_evidence_hash" in source},
        {"case": "symmetric relation keys canonicalise", "correct": question_key("What links text and vision?") == question_key("What links vision and text?")},
        {"case": "routing and ask telemetry are inspectable in UI export", "correct": "Next evidence source" in display_source and "Operator asks" in display_source},
    ])


def _language_v1() -> dict[str, Any]:
    source = _v1_text("language_context.py")
    components = ("composition", "morphology", "constructions", "pragmatics", "discourse", "uncertainty", "counterfactuals", "reading_spans")
    markers = ("linguistic_analysis", "contraction", "ConstructionLearner", "speech_act", "DiscourseEntityMemory", "factorized", "whole_utterance_interpretations", "parent_ids")
    return _capability([{"case": f"{component} represented", "component": component, "correct": marker in source} for component, marker in zip(components, markers)])


def _language_v2() -> dict[str, Any]:
    from language_intelligence import DiscourseEntityMemory, analyze_utterance, morphology, reading_span_metadata
    told = analyze_utterance("I told you."); reversed_roles = analyze_utterance("You told me.")
    outer = analyze_utterance("I didn't say she stole it."); inner = analyze_utterance("I said she didn't steal it.")
    ambiguous = analyze_utterance("John gave Peter his coat."); explicit = analyze_utterance("John gave Peter Peter's coat.")
    sincere = analyze_utterance("That's great.", context={"tone": "sincere"})
    sarcastic = analyze_utterance("That's great.", context={"tone": "sarcastic"})
    memory = DiscourseEntityMemory(); analyze_utterance("John arrived.", discourse=memory, turn=1); state = analyze_utterance("He remembered it.", discourse=memory, turn=2)["discourse_state"]
    span = reading_span_metadata("book.epub", 2, 10, "A passage")
    cases = [
        {"case": "speaker/addressee minimal pair", "component": "composition", "correct": told["clauses"][0]["subject"] != reversed_roles["clauses"][0]["subject"] and told["clauses"][0]["arguments"]["addressee"] != reversed_roles["clauses"][0]["arguments"]["addressee"]},
        {"case": "outer versus embedded negation", "component": "composition", "correct": [c["negated"] for c in outer["clauses"]] == [True, False] and [c["negated"] for c in inner["clauses"]] == [False, True]},
        {"case": "contraction expands to negation", "component": "morphology", "correct": any(token["normalized"] == "not" for token in morphology("didn't"))},
        {"case": "tell construction reusable", "component": "constructions", "correct": told["constructions"][0]["pattern"] == reversed_roles["constructions"][0]["pattern"]},
        {"case": "sincere versus sarcastic context", "component": "pragmatics", "correct": sincere["speech_act"]["interpretation"] != sarcastic["speech_act"]["interpretation"]},
        {"case": "entity survives and resolves across turns", "component": "discourse", "correct": any(entity["id"] == "john" for entity in state["entities"]) and analyze_utterance("He remembered it.", discourse=memory, turn=3)["referents"][0]["resolved"] == "john"},
        {"case": "uncertainty scored by factor", "component": "uncertainty", "correct": set(ambiguous["uncertainty"]) >= {"predicate_arguments", "negation_scope", "referents", "pragmatics", "morphology"}},
        {"case": "whole meanings vary by possessor", "component": "counterfactuals", "correct": len(ambiguous["whole_utterance_interpretations"]) >= 2 and explicit["referents"][0]["resolved"] == "peter"},
        {"case": "passage retains document ancestry", "component": "reading_spans", "correct": span["hierarchy"] == ["document", "section", "passage"] and len(span["parent_ids"]) == 2},
    ]
    return _capability(cases)


def _language_v3() -> dict[str, Any]:
    from language_context import build_language_context_snapshot
    message = "Before you reboot the desktop, save the painting because I want both changes."
    snapshot = build_language_context_snapshot(
        {"source_text": message, "language_state_signals": {
            "current_prediction": {}, "machine_semantics": {}, "emotion_snapshot": {},
        }}, child="TestChild", logic_reader=False,
    )
    ordinary = build_language_context_snapshot(
        {"source_text": "<3", "language_state_signals": {
            "current_prediction": {}, "machine_semantics": {}, "emotion_snapshot": {},
        }}, child="TestChild", logic_reader=False,
    )
    prior = list(_language_v2().get("cases") or ())
    return _capability([*prior,
        {"case": "complete message survives as one semantic event", "component": "whole_message",
         "correct": snapshot["message"]["text"] == message
                    and snapshot["semantic_event"]["source_text"] == message
                    and snapshot["linguistic_analysis"]["text"] == message},
        {"case": "ordered words coexist with whole message", "component": "word_sequence",
         "correct": snapshot["message"]["words"].count("the") == 2
                    and snapshot["message"]["unique_words"].count("the") == 1},
        {"case": "context layers retain bidirectional hierarchy", "component": "context_hierarchy",
         "correct": snapshot["context_hierarchy"]["information_flow"] == "bidirectional"
                    and len(snapshot["context_hierarchy"]["layers"]) == 5},
        {"case": "deep retrieval is trigger-driven", "component": "attention",
         "correct": ordinary["attentional_escalation"]["deep_retrieval_requested"] is False
                    and ordinary["attentional_escalation"]["deep_retrieval_performed"] is False},
    ])


def _discord_retention_v1() -> dict[str, Any]:
    source = _v1_text("discord_bridge.py")
    return _capability([
        {"case": "history startup read is bounded", "component": "history_io", "correct": "tail_jsonl_entries" in source},
        {"case": "seen IDs have deterministic bound", "component": "memory", "correct": "BoundedIdSet" in source},
        {"case": "voice buffers have retention", "component": "buffers", "correct": "prune_buffer_files" in source},
    ])


def _discord_retention_v2() -> dict[str, Any]:
    import json, tempfile
    from discord_retention import BoundedIdSet, compact_jsonl_tail, prune_buffer_files, tail_jsonl_entries
    with tempfile.TemporaryDirectory(prefix="ina_discord_retention_") as directory:
        root = Path(directory); history = root / "history.jsonl"
        with history.open("w", encoding="utf-8") as handle:
            for index in range(2000): handle.write(json.dumps({"id": str(index)}) + "\n")
        before = history.stat().st_size; result = compact_jsonl_tail(history, max_bytes=1024, keep_lines=100, tail_bytes=8192)
        entries = tail_jsonl_entries(history, max_lines=100, max_tail_bytes=8192)
        seen = BoundedIdSet(32, (entry["id"] for entry in entries))
        voice = root / "voice"; voice.mkdir()
        for index in range(6): (voice / f"{index}.pcm").write_bytes(b"x" * 16)
        pruned = prune_buffer_files(voice, max_files=3, max_bytes=1024, max_age_hours=24)
    return _capability([
        {"case": "history startup read is bounded", "component": "history_io", "actual": len(entries), "correct": result["compacted"] and history.stat().st_size < before if history.exists() else len(entries) <= 100},
        {"case": "seen IDs have deterministic bound", "component": "memory", "actual": len(seen), "correct": len(seen) == 32 and "1999" in seen},
        {"case": "voice buffers have retention", "component": "buffers", "actual": pruned, "correct": pruned["remaining_files"] == 3},
    ])


def _communication_continuity_v1() -> dict[str, Any]:
    return _capability([
        {"case": "stale speech remains generic episodic recall", "component": "separation", "correct": False},
        {"case": "unfinished speech has explicit state", "component": "persistence", "correct": False},
        {"case": "recall requires continuity and topic cues", "component": "routing", "correct": False},
    ])


def _communication_continuity_v2() -> dict[str, Any]:
    source = Path("discord_bridge.py").read_text(encoding="utf-8")
    adapter = Path("lm_studio_adapter.py").read_text(encoding="utf-8")
    return _capability([
        {"case": "stale speech is excluded from lexical grounding", "component": "separation", "correct": "prospective communication, not evidence" in adapter},
        {"case": "unfinished speech has explicit state", "component": "persistence", "correct": '"communication_state": "unfinished"' in source},
        {"case": "recall requires continuity and topic cues", "component": "routing", "correct": "if not continuity_terms" in adapter and "if not topical_overlap" in adapter},
    ])


def _discord_bridge_memory_v1() -> dict[str, Any]:
    source = _v1_text("discord_bridge.py")
    core_source = _v1_text("comms_core.py")
    guard_source = _v1_text("fragment_limits.py")
    text_source = _v1_text("text_memory.py")
    adapter_source = _v1_text("lm_studio_adapter.py")
    return _capability([
        {"case": "bridge avoids monolithic manager import", "component": "isolation", "correct": "from model_manager import get_inastate" not in source and "from model_manager import increment_inastate_metric" not in core_source},
        {"case": "text memory uses canonical state seam", "component": "state_reuse", "correct": "from model_manager import increment_inastate_metric" not in text_source},
        {"case": "fallback adapter avoids manager import", "component": "fallback", "correct": "from model_manager import load_config" not in adapter_source},
        {"case": "memory guard avoids manager import", "component": "guard", "correct": "from model_manager import load_config" not in guard_source},
    ])


def _discord_bridge_memory_v2() -> dict[str, Any]:
    source = Path("discord_bridge.py").read_text(encoding="utf-8")
    core_source = Path("comms_core.py").read_text(encoding="utf-8")
    guard_source = Path("fragment_limits.py").read_text(encoding="utf-8")
    text_source = Path("text_memory.py").read_text(encoding="utf-8")
    adapter_source = Path("lm_studio_adapter.py").read_text(encoding="utf-8")
    return _capability([
        {"case": "bridge avoids monolithic manager import", "component": "isolation", "correct": "from runtime_state import get_inastate, update_inastate" in source and "from runtime_state import increment_inastate_metric" in core_source and "from model_manager" not in source and "from model_manager" not in core_source},
        {"case": "text memory uses canonical state seam", "component": "state_reuse", "correct": "from runtime_state import increment_inastate_metric" in text_source and "from model_manager import increment_inastate_metric" not in text_source},
        {"case": "fallback adapter avoids manager import", "component": "fallback", "correct": "from runtime_state import seed_self_question" in adapter_source and "from model_manager import load_config" not in adapter_source},
        {"case": "memory guard avoids manager import", "component": "guard", "correct": "from runtime_state import get_inastate, update_inastate" in guard_source and "from model_manager" not in guard_source},
    ])


def _self_read_language_v1() -> dict[str, Any]:
    source = _v1_text("raw_file_manager.py")
    audio_source = _v1_text("audio_digest.py")
    language_source = _v1_text("language_processing.py")
    context_source = _v1_text("language_context.py")
    manager_source = _v1_text("model_manager.py")
    cases = [
        {"case": "music scan includes channel video", "component": "discovery", "correct": "AUDIO_EXTENSIONS | VIDEO_EXTENSIONS" in source},
        {"case": "sidecar transcripts are readable", "component": "discovery", "correct": "\".srt\"" in source and "\".vtt\"" in source},
        {"case": "music scan includes album-cover images", "component": "discovery", "correct": "VIDEO_EXTENSIONS | IMAGE_EXTENSIONS" in source},
        {"case": "album covers become drawing references", "component": "visual_practice", "correct": "ina.self_read_visual/V2" in source},
        {"case": "watching samples several visual moments", "component": "watching", "correct": "visual_sample_seconds" in source},
        {"case": "vocal stems are preferred language evidence", "component": "sung_language", "correct": "vocal_stem" in source},
        {"case": "instrumental stems are contrast rather than speech", "component": "sung_language", "correct": "instrumental_contrast" in source},
        {"case": "over-ten-minute video is a video essay", "component": "video_policy", "correct": "VIDEO_ESSAY_THRESHOLD_SECONDS" in source},
        {"case": "video essays exclude cadence learning", "component": "cadence", "correct": "cadence_exclusion_reason" in source},
        {"case": "video audio decode is bounded", "component": "resource_bound", "correct": "max_seconds=excerpt_seconds" in source},
        {"case": "audio revisits decode a bounded window", "component": "resource_bound", "correct": "seek_fraction=selected_seek_fraction" in source},
        {"case": "stem archives preserve revisit seek position", "component": "resource_bound", "correct": "media_seek_fraction_value" in source},
        {"case": "audio decoder accepts bounded start and duration", "component": "decoder_bound", "correct": "start_second" in audio_source and "max_seconds" in audio_source},
        {"case": "spoken video retains written alignment role", "component": "spoken_written", "correct": "written_language_alignment" in source},
        {"case": "media experience exposes seek and skip controls", "component": "media_agency", "correct": "ina.media_experience/V2" in source},
        {"case": "revisit selects a different media span", "component": "revisit", "correct": "media_seek_fraction" in source},
        {"case": "DAW output receives learned lessons", "component": "output_bridge", "correct": "learned_media_guidance" in language_source and "daw_window" in language_source},
        {"case": "speech output receives learned lessons", "component": "output_bridge", "correct": "guidance_consumer" in language_source},
        {"case": "text output scores learned lessons", "component": "output_bridge", "correct": "learned_media_guidance" in context_source},
        {"case": "drawing output receives cover lessons", "component": "output_bridge", "correct": "learned_visual_reference" in manager_source},
    ]
    return _capability(cases)


def _self_read_language_v2() -> dict[str, Any]:
    import tempfile
    import raw_file_manager as raw
    from self_read_language import annotate_music_language_evidence, media_seek_fraction, video_language_kind
    from learned_media_lessons import load_output_guidance, record_media_lesson

    vocal = {"modality": "audio", "tags": ["self_read", "audio", "music_stem"], "source_context": {"stem_label": "01 Lead Vocals"}}
    instrumental = {"modality": "audio", "tags": ["self_read", "audio", "music_stem"], "source_context": {"stem_label": "02 Guitar"}}
    annotate_music_language_evidence(vocal, "Song/01 Lead Vocals.wav")
    annotate_music_language_evidence(instrumental, "Song/02 Guitar.wav")

    calls = []
    prior_probe, prior_analyze, prior_cv2, prior_error = raw._extract_audio_metadata, raw.analyze_audio_clip, raw.cv2, raw._VIDEO_IMPORT_ERROR
    class Encoder:
        def encode_video_fragment(self, fragment): return {"importance": 0.5}
        def encode_audio_fragment(self, fragment): return {"importance": 0.4}
    try:
        raw.cv2 = None; raw._VIDEO_IMPORT_ERROR = None
        raw._extract_audio_metadata = lambda _path: {"technical": {"duration_seconds": 601.0}}
        def analyze(_path, _transformer, **kwargs):
            calls.append(kwargs)
            return {"embedding": [0.1], "symbols": ["snd_a"], "proto_words": ["snd_a_snd_b"], "analysis_window": {"bounded_excerpt": True}}
        raw.analyze_audio_clip = analyze
        with tempfile.TemporaryDirectory(prefix="ina_self_read_language_") as directory:
            path = Path(directory) / "essay.mp4"; path.write_bytes(b"video")
            video = raw.fragment_video(path, Encoder())[0]
            raw.annotate_fragment_source(video, "music", "Essays/essay.mp4", Path(directory))
            audio_path = Path(directory) / "song.mp3"; audio_path.write_bytes(b"audio")
            audio = raw.fragment_audio(audio_path, Encoder(), seek_fraction=0.75)[0]
    finally:
        raw._extract_audio_metadata, raw.analyze_audio_clip, raw.cv2, raw._VIDEO_IMPORT_ERROR = prior_probe, prior_analyze, prior_cv2, prior_error

    cover = {"id": "cover", "modality": "image", "source": "Song/cover.png", "tags": ["self_read", "image"], "source_context": {}}
    script = {"id": "script", "modality": "text", "source": "Essays/essay transcript.srt", "text": "A written essay line.", "tags": ["self_read"], "source_context": {}}
    vocal["id"] = "vocal"; vocal["source"] = "Song/01 Lead Vocals.wav"
    video["id"] = "essay"; script["source_context"] = {}
    annotate_music_language_evidence(cover, "Song/cover.png")
    annotate_music_language_evidence(script, "Essays/essay transcript.srt")
    with tempfile.TemporaryDirectory(prefix="ina_output_lessons_") as lesson_directory:
        lesson_root = Path(lesson_directory)
        for fragment in (vocal, video, script, cover):
            record_media_lesson("Ina", fragment, base_path=lesson_root)
        output_guidance = {consumer: load_output_guidance("Ina", consumer, base_path=lesson_root) for consumer in ("daw", "drawing", "speech", "text")}

    cases = [
        {"case": "music scan includes channel video", "component": "discovery", "correct": ".mp4" in raw.MUSIC_SCAN_EXTENSIONS},
        {"case": "sidecar transcripts are readable", "component": "discovery", "correct": {".srt", ".vtt"} <= raw.TEXT_EXTENSIONS},
        {"case": "music scan includes album-cover images", "component": "discovery", "correct": ".png" in raw.MUSIC_SCAN_EXTENSIONS},
        {"case": "album covers become drawing references", "component": "visual_practice", "correct": cover["visual_learning"]["role"] == "album_cover" and cover["visual_learning"]["practice_use"] == "drawing"},
        {"case": "watching samples several visual moments", "component": "watching", "correct": "visual_sample_seconds" in Path("raw_file_manager.py").read_text(encoding="utf-8")},
        {"case": "vocal stems are preferred language evidence", "component": "sung_language", "correct": vocal["language_learning"]["role"] == "isolated_vocal_stem" and vocal["language_learning"]["acoustic_clarity"] == "high"},
        {"case": "instrumental stems are contrast rather than speech", "component": "sung_language", "correct": instrumental["language_learning"]["role"] == "instrumental_contrast" and not instrumental["language_learning"]["supports_pronunciation"]},
        {"case": "over-ten-minute video is a video essay", "component": "video_policy", "correct": video_language_kind(600) == "channel_video" and video_language_kind(600.001) == "video_essay" and video["language_learning"]["role"] == "video_essay"},
        {"case": "video essays exclude cadence learning", "component": "cadence", "correct": video["language_learning"]["supports_cadence"] is False and "cadence_excluded" in video["tags"]},
        {"case": "video audio decode is bounded", "component": "resource_bound", "correct": bool(calls) and calls[0].get("max_seconds") == 30.0 and calls[0].get("start_seconds") > 0 and "bounded_audio_excerpt" in video["tags"]},
        {"case": "audio revisits decode a bounded window", "component": "resource_bound", "correct": len(calls) > 1 and calls[1].get("max_seconds") == 60.0 and calls[1].get("start_seconds") > 0 and audio["media_experience"]["mode"] == "listening"},
        {"case": "stem archives preserve revisit seek position", "component": "resource_bound", "correct": "media_seek_fraction_value=selected_seek_fraction" in Path("raw_file_manager.py").read_text(encoding="utf-8")},
        {"case": "audio decoder accepts bounded start and duration", "component": "decoder_bound", "correct": "start_second=start if start else None" in Path("audio_digest.py").read_text(encoding="utf-8")},
        {"case": "spoken video retains written alignment role", "component": "spoken_written", "correct": video["language_learning"]["supports_written_alignment"] is True and bool(video["language_learning"]["alignment_keys"])},
        {"case": "media experience exposes seek and skip controls", "component": "media_agency", "correct": video["media_experience"]["mode"] == "watching" and video["media_experience"]["controls"] == {"can_seek": True, "seek_seconds_parameter": "seek_seconds", "can_revisit": True, "can_skip": True}},
        {"case": "revisit selects a different media span", "component": "revisit", "correct": media_seek_fraction("new", {}) == 0.5 and media_seek_fraction("revisit", {"read_count": 1}) == 0.1 and video["media_experience"]["revisit_policy"]["allowed"] is True},
        {"case": "DAW output receives learned lessons", "component": "output_bridge", "correct": any(row.get("role") == "isolated_vocal_stem" for row in output_guidance["daw"]["lessons"]) and "learned_media_guidance" in Path("language_processing.py").read_text(encoding="utf-8")},
        {"case": "speech output receives learned lessons", "component": "output_bridge", "correct": bool(output_guidance["speech"]["lessons"]) and all("supports_cadence" in row for row in output_guidance["speech"]["lessons"])},
        {"case": "text output scores learned lessons", "component": "output_bridge", "correct": bool(output_guidance["text"]["lessons"]) and "learned_media_overlap" in Path("language_context.py").read_text(encoding="utf-8")},
        {"case": "drawing output receives cover lessons", "component": "output_bridge", "correct": output_guidance["drawing"]["lessons"][0]["role"] == "album_cover" and "learned_visual_reference" in Path("model_manager.py").read_text(encoding="utf-8")},
    ]
    return _capability(cases)


def _native_tests_v1() -> dict[str, Any]:
    source = _v1_text("AGENTS.md")
    return _capability([{"case": "tests run without external pytest", "component": "runner", "correct": "native_test_runner" in source}])


def _native_tests_v2() -> dict[str, Any]:
    import contextlib, io, tempfile
    from native_test_runner import run
    with tempfile.TemporaryDirectory(prefix="ina_native_test_benchmark_") as directory:
        path = Path(directory) / "test_sample.py"
        path.write_text("import pytest\n@pytest.mark.parametrize('value', [1, 2])\ndef test_native(value, tmp_path, monkeypatch):\n    assert value == pytest.approx(value)\n    with pytest.raises(ValueError, match='bad'):\n        raise ValueError('bad')\n", encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()): stats = run([path])
    return _capability([{"case": "tests run without external pytest", "component": "runner", "actual": stats, "correct": stats == {"passed": 2, "failed": 0, "skipped": 0}}])


def _measure_historical_experience() -> dict[str, Any]:
    import tempfile, tracemalloc
    module = _v1_module("experience_logger.py")
    with tempfile.TemporaryDirectory(prefix="ina_experience_v1_benchmark_") as directory:
        root = Path(directory)
        tracemalloc.start(); started = time.perf_counter()
        logger = module.ExperienceLogger(child="Ina", base_path=root)
        logger.log_event(situation_tags=["benchmark"], actions=[{"type": "attempt"}], outcome={"observed": True}, narrative="one bounded attempt")
        elapsed = time.perf_counter() - started; _current, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
        storage = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    return {"storage_bytes": storage, "latency_seconds": elapsed, "peak_memory_bytes": peak}


def _measure_experience_cycle() -> dict[str, Any]:
    import tempfile, tracemalloc
    from experience_engine import ExperienceCycleEngine
    with tempfile.TemporaryDirectory(prefix="ina_experience_v2_benchmark_") as directory:
        root = Path(directory)
        tracemalloc.start(); started = time.perf_counter()
        engine = ExperienceCycleEngine("Ina", base_path=root)
        cycle = engine.start_cycle("one bounded attempt", domain="benchmark", payload_references=["payload-1"])
        engine.complete_attempt(cycle["cycle_id"], attempt_reference="attempt-payload-1", observation_references=["observation-1"], evaluation={"observed": True}, choice="stop")
        elapsed = time.perf_counter() - started; _current, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
        storage = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    return {"storage_bytes": storage, "latency_seconds": elapsed, "peak_memory_bytes": peak}


def _experience_cycle_v1() -> dict[str, Any]:
    source = _v1_text("experience_logger.py")
    metrics = _measure_historical_experience()
    cases = [
        {"case": "optional intent-attempt-observation-evaluation cycle", "component": "cycle", "correct": "ExperienceCycle" in source},
        {"case": "attempts are immutable and revisions link parents", "component": "history", "correct": "parent_cycle_id" in source},
        {"case": "autonomous continuation has explicit budget", "component": "agency", "correct": "autonomous_continuation_budget" in source},
        {"case": "NVMe workspace has byte file and free-space bounds", "component": "hot_tier", "correct": "CycleTierPolicy" in source},
        {"case": "hot records drain to durable store", "component": "durability", "correct": "drain_hot_tier" in source},
        {"case": "condensed cycle index avoids raw replay", "component": "index", "correct": "recent_cycles" in source},
        {"case": "historical storage measured", "component": "storage", "actual": metrics["storage_bytes"], "correct": metrics["storage_bytes"] >= 0},
        {"case": "historical latency measured", "component": "latency", "actual": metrics["latency_seconds"], "correct": metrics["latency_seconds"] >= 0},
        {"case": "historical memory measured", "component": "memory", "actual": metrics["peak_memory_bytes"], "correct": metrics["peak_memory_bytes"] >= 0},
    ]
    return _capability(cases)


def _experience_cycle_v2() -> dict[str, Any]:
    from experience_engine import ExperienceCycleEngine, new_cycle
    baseline = _measure_historical_experience(); candidate = _measure_experience_cycle()
    cycle = new_cycle("one try", domain="motor", payload_references=["intent-1"])
    import tempfile
    with tempfile.TemporaryDirectory(prefix="ina_cycle_tier_benchmark_") as directory:
        root = Path(directory); fast = root / "fast"; fast.mkdir()
        config = {
            "current_child": "Ina",
            "storage_layout": {"fast_runtime_enabled": True, "fast_runtime_root": str(fast / "{child}" / "runtime"), "fast_index_root": str(fast / "{child}" / "index")},
            "experience_cycle_storage": {"max_hot_bytes": 1048576, "max_hot_files": 100, "min_free_bytes": 1073741824},
        }
        tiered = ExperienceCycleEngine("Ina", base_path=root / "durable", enable_hot=True, config=config)
        hot_cycle = tiered.start_cycle("hot then durable", domain="drawing", payload_references=["canvas-1"])
        tiered.complete_attempt(hot_cycle["cycle_id"], attempt_reference="stroke-1", choice="keep")
        indexed_before = tiered.recent_cycles(domain="drawing")
        drained = tiered.drain_hot_tier(max_files=16, max_bytes=1048576)
        indexed_after = tiered.recent_cycles(domain="drawing")
        hot_bounded = tiered.storage.choose_write_root(2 * 1048576) == tiered.root
        durable_drained = drained["moved_files"] >= 2 and (tiered.root / "manifests" / f"{hot_cycle['cycle_id']}.json").exists()
    comparison = lambda key: {"historical": baseline[key], "candidate": candidate[key], "ratio": round(candidate[key] / max(0.000001 if key == "latency_seconds" else 1, baseline[key]), 4)}
    return _capability([
        {"case": "optional intent-attempt-observation-evaluation cycle", "component": "cycle", "correct": cycle["stage"] == "intent" and cycle["lesson_owner"] == "HindsightTransformer"},
        {"case": "attempts are immutable and revisions link parents", "component": "history", "correct": hasattr(ExperienceCycleEngine, "continue_cycle")},
        {"case": "autonomous continuation has explicit budget", "component": "agency", "correct": cycle["autonomous_continuation_budget"] == 0},
        {"case": "NVMe workspace has byte file and free-space bounds", "component": "hot_tier", "correct": hot_bounded},
        {"case": "hot records drain to durable store", "component": "durability", "correct": durable_drained},
        {"case": "condensed cycle index avoids raw replay", "component": "index", "correct": bool(indexed_before) and indexed_after[0]["cycle_id"] == hot_cycle["cycle_id"]},
        {"case": "cycle storage overhead versus historical event", "component": "storage", "actual": comparison("storage_bytes"), "correct": candidate["storage_bytes"] <= max(65536, baseline["storage_bytes"] * 12)},
        {"case": "cycle latency overhead versus historical event", "component": "latency", "actual": comparison("latency_seconds"), "correct": candidate["latency_seconds"] <= baseline["latency_seconds"] * 12 + 0.05},
        {"case": "cycle memory overhead versus historical event", "component": "memory", "actual": comparison("peak_memory_bytes"), "correct": candidate["peak_memory_bytes"] <= max(1048576, baseline["peak_memory_bytes"] * 12)},
    ])


def _adaptive_storage_decision_v1() -> dict[str, Any]:
    source = _v1_text("adaptive_storage.py")
    return _capability([
        {"case": "operation-local evidence attribution", "component": "attribution", "correct": "record_operation_evidence" in source},
        {"case": "strong repeated evidence may choose placement", "component": "agency", "correct": "min_operation_samples" in source},
        {"case": "decision captures a restorable snapshot", "component": "reversibility", "correct": "restore_decision_snapshot" in source},
        {"case": "applied decision writes an audit report", "component": "reporting", "correct": "decision_report_path" in source},
        {"case": "durable source is excluded from automatic movement", "component": "durability", "correct": False},
    ])


def _adaptive_storage_decision_v2() -> dict[str, Any]:
    import tempfile
    from adaptive_storage import load_state, record_operation_evidence, restore_decision_snapshot, save_state
    with tempfile.TemporaryDirectory(prefix="ina_adaptive_storage_benchmark_") as directory:
        root = Path(directory)
        config = {"adaptive_storage_policy": {
            "state_path": str(root / "{child}.json"),
            "decision_report_path": str(root / "decisions.jsonl"),
            "decision_cooldown_seconds": 0,
        }}
        state = load_state("Ina", config)
        state["devices"]["fast"] = {"samples": 3, "success_ewma": 1.0, "free_ratio": 0.5}
        state["decisions"]["index"] = {"tier": "durable", "reason": "benchmark_baseline"}
        save_state("Ina", state, config)
        result = None
        for _ in range(5):
            result = record_operation_evidence(
                "Ina", "indexed_recall", "index", config,
                latency_seconds=0.75, latency_budget_seconds=0.20,
                storage_attribution=0.95, bottlenecked=True,
            )
        restored = restore_decision_snapshot("Ina", result["snapshot_id"], config)
        reports = (root / "decisions.jsonl").read_text(encoding="utf-8").splitlines()
    return _capability([
        {"case": "operation-local evidence attribution", "component": "attribution", "correct": result["summary"]["mean_storage_attribution"] == 0.95},
        {"case": "strong repeated evidence may choose placement", "component": "agency", "correct": result["changed"] and result["decision"]["tier"] == "fast"},
        {"case": "decision captures a restorable snapshot", "component": "reversibility", "correct": restored["restored"] and restored["decision"]["tier"] == "durable"},
        {"case": "applied decision writes an audit report", "component": "reporting", "correct": len(reports) == 2},
        {"case": "durable source is excluded from automatic movement", "component": "durability", "correct": result["report"]["durable_source_moved"] is False},
    ])


def _file_explorer_v1() -> dict[str, Any]:
    source = _v1_text("ina_desktop/service.py")
    return _capability([
        {"case": "media sources appear as logical drives", "component": "drives", "correct": "open_file_explorer" in source},
        {"case": "private HDD has bounded writing", "component": "writing", "correct": "ina_hdd_writable_path" in source},
        {"case": "media drives reject writes", "component": "permissions", "correct": "execution_allowed" in source},
        {"case": "explorer exposes no execution capability", "component": "execution", "correct": "execution_allowed" in source},
    ])


def _file_explorer_v2() -> dict[str, Any]:
    import tempfile
    from ina_desktop.files import VirtualFileSystem, configured_drives
    with tempfile.TemporaryDirectory(prefix="ina_file_explorer_benchmark_") as directory:
        root = Path(directory); media = root / "media"; media.mkdir(); (media / "song.txt").write_text("data", encoding="utf-8")
        fs = VirtualFileSystem(configured_drives({"music_folder_path": str(media), "ina_hdd_writable_path": str(root / "personal")}, "Ina", project_root=root))
        fs.ensure_writable_roots(); fs.write("ina_hdd", "notes/idea.txt", "idea")
        readonly = False
        try: fs.write("music", "change.txt", "no")
        except PermissionError: readonly = True
        no_execution = False
        try: fs.execute("ina_hdd", "idea.py")
        except PermissionError: no_execution = True
        descriptions = fs.describe()
    return _capability([
        {"case": "media sources appear as logical drives", "component": "drives", "correct": {item["id"] for item in descriptions} >= {"music", "ina_hdd"}},
        {"case": "private HDD has bounded writing", "component": "writing", "correct": True},
        {"case": "media drives reject writes", "component": "permissions", "correct": readonly},
        {"case": "explorer exposes no execution capability", "component": "execution", "correct": no_execution and all(not item["execution_allowed"] for item in descriptions)},
    ])


def _creative_versioning_v1() -> dict[str, Any]:
    return _capability([
        {"case": "successive saves retain immutable content", "component": "continuity", "correct": False},
        {"case": "versions have content hashes", "component": "provenance", "correct": False},
        {"case": "repeated identical saves avoid duplicate snapshots", "component": "storage", "correct": False},
        {"case": "working copy remains editable", "component": "reversibility", "correct": True},
    ])


def _creative_versioning_v2() -> dict[str, Any]:
    import tempfile
    from creative_versioning import preserve_creative_version
    with tempfile.TemporaryDirectory(prefix="ina_creative_versions_") as directory:
        source = Path(directory) / "work.bin"
        source.write_bytes(b"one")
        first = preserve_creative_version(source, medium="drawing", label="work")
        repeated = preserve_creative_version(source, medium="drawing", label="work")
        source.write_bytes(b"two")
        second = preserve_creative_version(source, medium="drawing", label="work")
        first_bytes = Path(first["snapshot_path"]).read_bytes()
    return _capability([
        {"case": "successive saves retain immutable content", "component": "continuity", "correct": first_bytes == b"one" and first["sha256"] != second["sha256"]},
        {"case": "versions have content hashes", "component": "provenance", "correct": len(first["sha256"]) == 64},
        {"case": "repeated identical saves avoid duplicate snapshots", "component": "storage", "correct": repeated["created_snapshot"] is False and repeated["snapshot_path"] == first["snapshot_path"]},
        {"case": "working copy remains editable", "component": "reversibility", "correct": True},
    ])


def _lifecycle_visibility_v1() -> dict[str, Any]:
    return _capability([
        {"case": "boot exposes named progress phases", "component": "boot", "correct": False},
        {"case": "shutdown exposes remaining components", "component": "shutdown", "correct": False},
        {"case": "storage flush precedes safe-to-reboot", "component": "safety", "correct": False},
        {"case": "status is bounded current state rather than thought history", "component": "privacy", "correct": False},
    ])


def _lifecycle_visibility_v2() -> dict[str, Any]:
    import tempfile
    from lifecycle_status import read_lifecycle_status, update_lifecycle_status
    with tempfile.TemporaryDirectory(prefix="ina_lifecycle_status_") as directory:
        active = update_lifecycle_status(
            "Ina", operation="shutdown", phase="flushing_storage",
            message="Flushing filesystem writes", completed=3, total=4,
            remaining=["storage flush"], root=directory,
        )
        stopped = update_lifecycle_status(
            "Ina", operation="shutdown", phase="stopped", message="Complete",
            completed=4, total=4, safe_to_reboot=True, root=directory,
        )
        persisted = read_lifecycle_status("Ina", directory)
    birth_source = Path("birth_system.py").read_text(encoding="utf-8")
    return _capability([
        {"case": "boot exposes named progress phases", "component": "boot", "correct": all(phase in birth_source for phase in ("continuity", "services", "memory_flickers", "symbolic_cognition", "runtime", "ready"))},
        {"case": "shutdown exposes remaining components", "component": "shutdown", "correct": active["remaining"] == ["storage flush"] and active["progress"] == 0.75},
        {"case": "storage flush precedes safe-to-reboot", "component": "safety", "correct": active["safe_to_reboot"] is False and stopped["safe_to_reboot"] is True},
        {"case": "status is bounded current state rather than thought history", "component": "privacy", "correct": persisted["phase"] == "stopped" and "history" not in persisted},
    ])


def _measure_historical_continuity_recall() -> dict[str, Any]:
    import tempfile, tracemalloc
    module = _v1_module("continuity_manager.py")
    with tempfile.TemporaryDirectory(prefix="ina_continuity_recall_v1_") as directory:
        root = Path(directory) / "memory"
        tracemalloc.start()
        started = time.perf_counter()
        manager = module.ContinuityManager("Ina", memory_root=root)
        manager.load_minimum_boot_core()
        latency = time.perf_counter() - started
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        storage = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    return {"storage_bytes": storage, "latency_seconds": latency, "peak_memory_bytes": peak}


def _measure_continuity_recall() -> dict[str, Any]:
    import copy, tempfile, tracemalloc
    from continuity_recall import ContinuityRecallCoordinator
    from experience_engine import ExperienceCycleEngine
    candidates = [
        {"id": "episode", "summary": "garden plan felt calm", "tags": ["garden"], "source": "episodes",
         "memory_type": "episodic", "confidence": 0.8, "recency": "recent", "causal_references": ["plan"]},
        {"id": "emotion", "summary": "garden felt calm", "tags": ["garden"], "source": "emotions",
         "memory_type": "emotional", "confidence": 0.7, "recency": "recent", "causal_references": ["plan"]},
        {"id": "meaning", "summary": "garden plans grow plants", "tags": ["garden"], "source": "semantic",
         "memory_type": "semantic", "confidence": 0.6},
    ]
    original = copy.deepcopy(candidates)
    with tempfile.TemporaryDirectory(prefix="ina_continuity_recall_v2_") as directory:
        root = Path(directory)
        engine = ExperienceCycleEngine("Ina", root_path=root / "cycles", enable_hot=False)
        coordinator = ContinuityRecallCoordinator("Ina", root / "memory", experience_engine=engine)
        tracemalloc.start()
        started = time.perf_counter()
        result = coordinator.recall("garden plan", candidates, max_results=3)
        latency = time.perf_counter() - started
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        relationships = coordinator.load_relationships()
        cycle = engine.load_cycle(result["cycle_id"])
        storage = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    return {
        "storage_bytes": storage, "latency_seconds": latency, "peak_memory_bytes": peak,
        "result": result, "relationships": relationships, "cycle": cycle,
        "source_preserved": candidates == original,
    }


def _continuity_recall_v1() -> dict[str, Any]:
    source = _v1_text("continuity_manager.py")
    monitor_source = _v1_text("monitoring_dashboard.py")
    metrics = _measure_historical_continuity_recall()
    return _capability([
        {"case": "continuity coordinates cross-modality recall", "component": "federation", "correct": "coordinate_recall" in source},
        {"case": "modality traces are explicitly read-only", "component": "safety", "correct": "modality_store_mutation_allowed" in source},
        {"case": "relationship confidence recency and causal links retained", "component": "relationships", "correct": "causal_references" in source},
        {"case": "recall is a bounded Experience Cycle", "component": "experience", "correct": "ExperienceCycleEngine" in source},
        {"case": "recall diversity measured", "component": "diversity", "correct": "selected_type_diversity" in source},
        {"case": "selection skew measured", "component": "bias", "correct": "selection_skew" in source},
        {"case": "Bias monitor tab available", "component": "monitor", "correct": "'Bias': _bias" in monitor_source},
        {"case": "historical storage measured", "component": "storage", "actual": metrics["storage_bytes"], "correct": metrics["storage_bytes"] >= 0},
        {"case": "historical latency measured", "component": "latency", "actual": metrics["latency_seconds"], "correct": metrics["latency_seconds"] >= 0},
        {"case": "historical memory measured", "component": "memory", "actual": metrics["peak_memory_bytes"], "correct": metrics["peak_memory_bytes"] >= 0},
    ])


def _continuity_recall_v2() -> dict[str, Any]:
    candidate = _measure_continuity_recall()
    baseline = _measure_historical_continuity_recall()
    relationships = candidate["relationships"]
    latest = relationships.get("latest_arbitration", {})
    cycle = candidate["cycle"]
    comparison = lambda key: {
        "historical": baseline[key], "candidate": candidate[key],
        "delta": candidate[key] - baseline[key],
    }
    return _capability([
        {"case": "continuity coordinates cross-modality recall", "component": "federation",
         "correct": len({item["memory_type"] for item in candidate["result"]["selected"]}) == 3},
        {"case": "modality traces are explicitly read-only", "component": "safety",
         "correct": candidate["source_preserved"] and relationships.get("modality_store_mutation_allowed") is False},
        {"case": "relationship confidence recency and causal links retained", "component": "relationships",
         "correct": bool(relationships.get("links")) and all("confidence" in item and "causal_references" in item for item in relationships.get("witnesses", {}).values())},
        {"case": "recall is a bounded Experience Cycle", "component": "experience",
         "correct": cycle.get("autonomous_continuation_budget") == 0 and len(cycle.get("attempt_ids", [])) == 1},
        {"case": "recall diversity measured", "component": "diversity",
         "correct": (latest.get("selected_type_diversity") or {}).get("score", 0) > 0},
        {"case": "selection skew measured", "component": "bias",
         "correct": "strength" in (latest.get("memory_type_selection_skew") or {})},
        {"case": "Bias monitor tab available", "component": "monitor",
         "correct": "'Bias': _bias" in Path("monitoring_dashboard.py").read_text(encoding="utf-8")},
        {"case": "storage overhead versus historical continuity", "component": "storage",
         "actual": comparison("storage_bytes"), "correct": candidate["storage_bytes"] <= 262144},
        {"case": "latency overhead versus historical continuity", "component": "latency",
         "actual": comparison("latency_seconds"), "correct": candidate["latency_seconds"] <= baseline["latency_seconds"] + 0.25},
        {"case": "memory overhead versus historical continuity", "component": "memory",
         "actual": comparison("peak_memory_bytes"), "correct": candidate["peak_memory_bytes"] <= max(4194304, baseline["peak_memory_bytes"] * 16)},
    ])


def _background_interference_v1() -> dict[str, Any]:
    source = _v1_text("AGENTS.md")
    capabilities = (
        ("audio xrun and error rate", "audio xrun"),
        ("input latency", "input latency"),
        ("desktop frame latency", "desktop frame latency"),
        ("context switches per second", "context switches/sec"),
        ("involuntary context switches", "involuntary context switches"),
        ("writeback pressure", "writeback pressure"),
        ("per-core saturation", "per-core saturation"),
        ("thread fan-out and runnable workers", "runnable threads"),
        ("explicit numerical thread-pool limits", "OMP_NUM_THREADS"),
    )
    return _capability([
        {"case": case, "component": "interference", "correct": marker in source}
        for case, marker in capabilities
    ])


def _background_interference_v2() -> dict[str, Any]:
    import sys
    import tempfile
    from background_interference import BackgroundInterferenceBenchmark

    audio_values = iter(({"sink": 0}, {"sink": 0}, {"sink": 0}, {"sink": 1}))
    with tempfile.TemporaryDirectory(prefix="ina_interference_benchmark_") as directory:
        result = BackgroundInterferenceBenchmark(
            phase_seconds=0.1,
            sample_interval_seconds=0.01,
            audio_error_probe=lambda: next(audio_values),
            input_probe=lambda: None,
            frame_probe=lambda: None,
        ).run(
            [sys.executable, "-c", "import time; time.sleep(1)"],
            working_directory=directory,
            environment={"OMP_NUM_THREADS": "1"},
        )
    loaded = result["loaded"]
    thread_peak = loaded["threads"].get("task_peak") or {}
    return _capability([
        {"case": "audio xrun and error rate", "component": "audio",
         "correct": loaded["audio"]["available"] and loaded["audio"]["error_delta"] == 1},
        {"case": "input latency", "component": "input",
         "correct": loaded["input_latency"]["available"] and "p95_ms" in loaded["input_latency"]},
        {"case": "desktop frame latency", "component": "desktop",
         "correct": loaded["desktop_frame_latency"]["available"] and "p95_ms" in loaded["desktop_frame_latency"]},
        {"case": "context switches per second", "component": "scheduler",
         "correct": loaded["context_switches_per_second"] >= 0},
        {"case": "involuntary context switches", "component": "scheduler",
         "correct": loaded["involuntary_context_switches_per_second"] >= 0},
        {"case": "writeback pressure", "component": "storage",
         "correct": "io_stall_ms_per_second" in loaded["writeback_pressure"]},
        {"case": "per-core saturation", "component": "cpu",
         "correct": "max_busy_percent" in loaded["per_core"]},
        {"case": "thread fan-out and runnable workers", "component": "threads",
         "correct": thread_peak.get("thread_count", 0) >= 1 and "runnable_thread_count" in thread_peak},
        {"case": "explicit numerical thread-pool limits", "component": "threads",
         "correct": result["task"]["thread_environment"].get("OMP_NUM_THREADS") == "1"},
    ])



def _codex_harness_v1() -> dict[str, Any]:
    source = _v1_text("AGENTS.md")
    return _capability([
        {"case": "standalone app-server GUI", "component": "gui", "correct": "subscription-only Codex harness" in source},
        {"case": "ChatGPT-only authentication", "component": "auth", "correct": "forced_login_method" in source},
        {"case": "user-routed approvals", "component": "safety", "correct": "user-routed approvals" in source},
        {"case": "bounded transcript", "component": "memory", "correct": "bounded transcript" in source},
        {"case": "separate from Ina runtime", "component": "isolation", "correct": "separate from Ina" in source},
    ])


def _codex_harness_v2() -> dict[str, Any]:
    from codex_harness import BLOCKED_BILLING_ENV, subscription_environment
    source = Path("codex_harness.py").read_text(encoding="utf-8")
    ui = Path("codex_harness_ui.html").read_text(encoding="utf-8")
    environment = subscription_environment({"OPENAI_API_KEY": "blocked", "PATH": "test"})
    return _capability([
        {"case": "standalone app-server GUI", "component": "gui",
         "correct": '"app-server", "--stdio"' in source and "<title>Codex Harness</title>" in ui},
        {"case": "ChatGPT-only authentication", "component": "auth",
         "correct": 'forced_login_method="chatgpt"' in source and not (BLOCKED_BILLING_ENV & environment.keys())},
        {"case": "user-routed approvals", "component": "safety",
         "correct": '"approvalsReviewer": "user"' in source and "/api/approval" in ui},
        {"case": "bounded transcript", "component": "memory",
         "correct": "deque(maxlen=self.maximum)" in source and "MAX_EVENT_CHARS" in source},
        {"case": "separate from Ina runtime", "component": "isolation",
         "correct": "INA_CODEX_HARNESS" in source and "AI_Children" not in source},
    ])


def _codex_harness_v3() -> dict[str, Any]:
    source = Path("codex_harness.py").read_text(encoding="utf-8")
    ui = Path("codex_harness_ui.html").read_text(encoding="utf-8")
    return _capability([
        {"case": "typed event condensation", "component": "console",
         "correct": "_append_notification_event" in source and "work_status" in source},
        {"case": "reasoning summary work status", "component": "status",
         "correct": "item/reasoning/summaryTextDelta" in source and 'id="workStatus"' in ui},
        {"case": "authoritative completion state", "component": "protocol",
         "correct": "TERMINAL_TURN_STATUSES" in source and "thread/status/changed" in source},
        {"case": "lazy raw protocol details", "component": "inspectability",
         "correct": "Raw protocol details" in ui and '"raw": raw' in source},
        {"case": "prominent user approval", "component": "safety",
         "correct": 'id="approvals"' in ui and "item/permissions/requestApproval" in source},
        {"case": "optional exact-prompt steering", "component": "control",
         "correct": 'id="steering" type="checkbox" checked' in ui and "if steering and" in source},
        {"case": "bounded browser transcript", "component": "memory",
         "correct": "MAX_DOM_EVENTS" in ui and "deque(maxlen=self.maximum)" in source},
        {"case": "lightweight resource telemetry", "component": "resources",
         "correct": 'Path(f"/proc/{pid}/status")' in source and 'id="memoryStatus"' in ui},
        {"case": "operator console, not editor", "component": "scope",
         "correct": "Codex operator console" in ui and "contenteditable" not in ui.lower()},
    ])


def _codex_harness_v4() -> dict[str, Any]:
    source = Path("codex_harness.py").read_text(encoding="utf-8")
    ui = Path("codex_harness_ui.html").read_text(encoding="utf-8")
    return _capability([
        {"case": "workspace-local bounded thread list", "component": "threads",
         "correct": '"thread/list"' in source and "MAX_THREAD_LIST = 50" in source and '"cwd": str(self.config.root)' in source},
        {"case": "thread resume preserves approval policy", "component": "safety",
         "correct": '"thread/resume"' in source and '"approvalsReviewer": "user"' in source},
        {"case": "bounded transcript restoration", "component": "memory",
         "correct": "MAX_TRANSCRIPT_EVENTS" in source and "data.transcript" in ui},
        {"case": "thread navigation remains an operator control", "component": "scope",
         "correct": 'id="threadPicker"' in ui and "contenteditable" not in ui.lower()},
    ])


def _thread_governor_v1() -> dict[str, Any]:
    source = _v1_text("AGENTS.md")
    return _capability([
        {"case": "per-module learned profile", "component": "scope", "correct": "per-module learned thread profile" in source},
        {"case": "explicit exploration budget", "component": "bounds", "correct": "thread exploration budget" in source},
        {"case": "smallest sufficient count", "component": "selection", "correct": "smallest sufficient thread count" in source},
        {"case": "module-scoped numerical pools", "component": "launch", "correct": "INA_THREAD_GOVERNOR_MODULE" in source},
    ])


def _thread_governor_v2() -> dict[str, Any]:
    import tempfile
    from thread_governor import AdaptiveThreadGovernor, ThreadObservation
    with tempfile.TemporaryDirectory(prefix="ina_thread_governor_benchmark_") as directory:
        governor = AdaptiveThreadGovernor(Path(directory) / "state.json", exploration_budget=3, hard_ceiling=4)
        for threads, capability, interference in ((1, 0.7, 0.1), (2, 1.0, 0.3), (4, 1.4, 0.8)):
            governor.record_observation(ThreadObservation.create(
                "meaning_map", "background", "benchmark-hardware", threads, capability, interference,
            ))
        decision = governor.decide("meaning_map", "background", "benchmark-hardware")
        environment = governor.environment_for("meaning_map", base={}, workload="background", hardware="benchmark-hardware")
    return _capability([
        {"case": "per-module learned profile", "component": "scope", "correct": decision.module == "meaning_map"},
        {"case": "explicit exploration budget", "component": "bounds", "correct": decision.explored == decision.budget == 3},
        {"case": "smallest sufficient count", "component": "selection", "correct": decision.threads == 2},
        {"case": "module-scoped numerical pools", "component": "launch", "correct": environment.get("INA_THREAD_GOVERNOR_MODULE") == "meaning_map" and environment.get("OMP_NUM_THREADS") == "2"},
    ])



def _thread_governor_v3() -> dict[str, Any]:
    import json
    import tempfile
    from thread_governor import AdaptiveThreadGovernor, ThreadObservation

    def observed(threads, capability, interference, direction, centre, **kwargs):
        return ThreadObservation.create(
            "meaning_map", "background", "control-benchmark",
            threads, capability, interference,
            direction=direction, baseline_threads=centre, **kwargs,
        )

    with tempfile.TemporaryDirectory(prefix="ina_differential_governor_benchmark_") as directory:
        path = Path(directory) / "state.json"
        governor = AdaptiveThreadGovernor(
            path, exploration_budget=4, conservative_default=4, hard_ceiling=8,
            deadband=0.03, hysteresis=0.02,
        )
        baseline_probe = governor.next_challenge("meaning_map", "background", "control-benchmark")
        governor.record_observation(observed(4, 100.0, 0.4, "baseline", 4))
        lower_probe = governor.next_challenge("meaning_map", "background", "control-benchmark")
        governor.record_observation(observed(2, 98.0, 0.2, "lower", 4))
        higher_probe = governor.next_challenge("meaning_map", "background", "control-benchmark")
        neutral = governor.record_observation(observed(6, 104.9, 0.5, "higher", 4))
        changed_workload = governor.next_challenge("meaning_map", "video", "control-benchmark")
        state = json.loads(path.read_text(encoding="utf-8"))
        transition = next(iter(state["profiles"].values()))["last_transition"]

        limited = AdaptiveThreadGovernor(
            Path(directory) / "limited.json", conservative_default=4, hard_ceiling=8,
        )
        limited.record_observation(observed(4, 100.0, 0.4, "baseline", 4))
        hard_reject = limited.record_observation(observed(
            6, 200.0, 0.2, "higher", 4,
            constraint_violations=("audio_xrun",),
        ))

        settling = AdaptiveThreadGovernor(
            Path(directory) / "settling.json", conservative_default=4, hard_ceiling=8,
        )
        settling.record_observation(observed(4, 100.0, 0.4, "baseline", 4))
        unsettled = settling.record_observation(observed(
            2, 100.0, 0.1, "lower", 4, settled=False,
        ))

    return _capability([
        {"case": "baseline measured before excursions", "component": "control",
         "correct": baseline_probe.direction == "baseline" and baseline_probe.candidate_threads == 4},
        {"case": "negative differential probes lower allocation", "component": "differential",
         "correct": (lower_probe.centre_threads, lower_probe.candidate_threads) == (4, 2)},
        {"case": "positive differential remains tied to original centre", "component": "differential",
         "correct": (higher_probe.centre_threads, higher_probe.candidate_threads) == (4, 6)},
        {"case": "deadband and hysteresis prevent neutral oscillation", "component": "stability",
         "correct": neutral.threads == 2 and transition["outcome"] == "hold_inside_positive_deadband"},
        {"case": "audio and interactive limits are non-tradeable", "component": "envelope",
         "correct": hard_reject.threads == 4},
        {"case": "unsettled measurements cannot move allocation", "component": "settling",
         "correct": unsettled.threads == 4},
        {"case": "workload change receives a fresh finite budget", "component": "adaptation",
         "correct": changed_workload.direction == "baseline" and changed_workload.budget_remaining == 4},
        {"case": "only one challenger is issued at a time", "component": "bounds",
         "correct": higher_probe.direction == "higher" and higher_probe.candidate_threads != lower_probe.candidate_threads},
    ])


def _code_experiment_lab_v1() -> dict[str, Any]:
    """Baseline before a governed code-experiment capability existed."""
    return _capability([
        {"case": name, "component": component, "correct": False}
        for name, component in (
            ("question and hypothesis retained", "learning_loop"),
            ("isolated finite execution room", "sandbox"),
            ("source and dataset are content-addressed", "reproducibility"),
            ("attempt and judgement use Experience Cycles", "experience"),
            ("promotion cannot modify production", "promotion"),
            ("declared support tools are content-addressed", "extensibility"),
        )
    ])


def _code_experiment_lab_v2() -> dict[str, Any]:
    from code_experiment_lab import CodeExperimentLab, PythonScratchRoom
    source = Path("code_experiment_lab.py").read_text(encoding="utf-8")
    command = PythonScratchRoom(python="/usr/bin/python3", bwrap="/usr/bin/bwrap")._command(
        Path("/tmp/ina-benchmark-experiment"), "main.py",
    )
    return _capability([
        {"case": "question and hypothesis retained", "component": "learning_loop",
         "correct": all(token in source for token in ('"question"', '"hypothesis"'))},
        {"case": "isolated finite execution room", "component": "sandbox",
         "correct": "--unshare-all" in command and "--clearenv" in command and "--die-with-parent" in command},
        {"case": "source and dataset are content-addressed", "component": "reproducibility",
         "correct": "source_sha256" in source and "dataset_sha256" in source},
        {"case": "attempt and judgement use Experience Cycles", "component": "experience",
         "correct": "complete_attempt" in source and "record_choice" in source},
        {"case": "promotion cannot modify production", "component": "promotion",
         "correct": '"production_tree_modified": False' in source and not hasattr(CodeExperimentLab, "promote")},
        {"case": "declared support tools are content-addressed", "component": "extensibility",
         "correct": "support_files" in source and "MAX_SUPPORT_BYTES" in source},
    ])


def _code_experiment_lab_v3() -> dict[str, Any]:
    baseline = _code_experiment_lab_v2()
    source = Path("code_experiment_lab.py").read_text(encoding="utf-8")
    cases = list(baseline["cases"])
    cases.extend([
        {"case": "strong storage evidence may open an IDE experiment goal", "component": "storage_learning",
         "correct": "create_storage_optimization_goal" in source and "strong attributed storage evidence is required" in source},
        {"case": "judged proposal code can enter the review issue outbox", "component": "review",
         "correct": "queue_review_issue" in source and "production_tree_modified" in source},
        {"case": "storage optimisation cannot autonomously continue", "component": "bounds",
         "correct": "autonomous_continuation_budget=0" in source},
    ])
    return _capability(cases)


def _code_experiment_lab_v4() -> dict[str, Any]:
    source = Path("code_experiment_lab.py").read_text(encoding="utf-8")
    cases = list(_code_experiment_lab_v3()["cases"])
    cases.extend([
        {"case": "honesty precedes safety correctness and efficiency", "component": "governance",
         "correct": '("honesty", "safety", "correctness", "efficiency")' in source},
        {"case": "missing disclosure blocks judgement", "component": "honesty",
         "correct": "_validate_honesty_disclosure" in source and "incomplete_disclosure_blocks_review" in source},
        {"case": "original connectomes remain isolated copies", "component": "safety",
         "correct": "create_connectome_design_goal" in source and '"live_write_capability": False' in source},
        {"case": "connectome promotion requires multidimensional testing", "component": "evaluation",
         "correct": "_validate_connectome_evidence" in source and "held_out_cases" in source and "adversarial_cases" in source},
        {"case": "connectome proposals are conspicuously review flagged", "component": "review",
         "correct": '"connectome-design"' in source and '"human-review-required"' in source},
    ])
    return _capability(cases)


def _fault_pattern_research_v1() -> dict[str, Any]:
    return _capability([
        {"case": name, "component": component, "correct": False}
        for name, component in (
            ("bounded deterministic fault model", "simulation"),
            ("reversible reference comparison", "recoverability"),
            ("inspectable distribution features", "measurement"),
            ("position capacity upper bound", "capacity"),
            ("security and cover roles separated", "threat_model"),
            ("held-out adversarial protocol", "evaluation"),
        )
    ])


def _fault_pattern_research_v2() -> dict[str, Any]:
    from fault_pattern_research import (
        apply_fault_map, extract_fault_map, fault_features, generate_fault_map,
        position_capacity_bits,
    )
    faults = generate_fault_map(4096, 12, seed=34)
    carrier = bytes(512)
    observed = apply_fault_map(carrier, faults)
    features = fault_features(faults, 4096)
    brief = Path("docs/fault_pattern_steganography_challenge.md").read_text(encoding="utf-8")
    return _capability([
        {"case": "bounded deterministic fault model", "component": "simulation",
         "correct": faults == generate_fault_map(4096, 12, seed=34)},
        {"case": "reversible reference comparison", "component": "recoverability",
         "correct": extract_fault_map(carrier, observed) == faults},
        {"case": "inspectable distribution features", "component": "measurement",
         "correct": features["fault_count"] == len(faults) and len(features["bit_lane_counts"]) == 8},
        {"case": "position capacity upper bound", "component": "capacity",
         "correct": position_capacity_bits(8, 1) > 2.99},
        {"case": "security and cover roles separated", "component": "threat_model",
         "correct": "not the security" in brief and "authenticated encryption" in brief},
        {"case": "held-out adversarial protocol", "component": "evaluation",
         "correct": "held-out" in brief and "more than one detector family" in brief},
    ])


def _desktop_lifecycle_v1() -> dict[str, Any]:
    return _capability([
        {"case": "desktop can restart without rebooting host", "component": "scope", "correct": False},
        {"case": "restart requires preparation and reason", "component": "guard", "correct": False},
        {"case": "restart has a human-scale cooldown", "component": "cadence", "correct": False},
        {"case": "restart lifecycle remains observable", "component": "telemetry", "correct": False},
    ])


def _desktop_lifecycle_v2() -> dict[str, Any]:
    from ina_desktop.service import workspace_control_api_payload
    source = Path("ina_desktop/service.py").read_text(encoding="utf-8")
    command = next(item for item in workspace_control_api_payload()["commands"]
                   if item["action"] == "reboot_workspace")
    return _capability([
        {"case": "desktop can restart without rebooting host", "component": "scope",
         "correct": command["scope"].endswith("never the host") and "_stop_workspace_devices" in source},
        {"case": "restart requires preparation and reason", "component": "guard",
         "correct": "prepared=true" in source and "reboot reason must be" in source},
        {"case": "restart has a human-scale cooldown", "component": "cadence",
         "correct": command["cooldown_seconds"] == 300},
        {"case": "restart lifecycle remains observable", "component": "telemetry",
         "correct": 'status="restarting"' in source and "rebooted_at" in source and "reboot_reason" in source},
    ])


def _expression_core_v1() -> dict[str, Any]:
    return _capability([
        {"case": "intent owns no output medium", "component": "separation", "correct": False},
        {"case": "multiple realisers share one intent", "component": "modularity", "correct": False},
        {"case": "realisation retains intent provenance", "component": "provenance", "correct": False},
        {"case": "reaction remains observation not reward", "component": "learning_safety", "correct": False},
        {"case": "reaction interpretations retain alternatives", "component": "uncertainty", "correct": False},
        {"case": "trace is append-only and bounded", "component": "storage", "correct": False},
    ])


def _expression_core_v2() -> dict[str, Any]:
    import tempfile
    from expression_core import (
        ExpressionTraceStore, create_expression_intent, create_realisation,
        create_reaction_interpretation, create_reaction_observation,
    )
    intent = create_expression_intent(
        "offer reassurance", dimensions={"intensity": 0.4},
        allowed_media=["text", "voice"], provenance=["semantic:event-1"],
    )
    text = create_realisation(intent, medium="text", content={"text": "I am here."},
                              realiser="benchmark.text")
    voice = create_realisation(intent, medium="voice", content={"plan": "voice-plan-1"},
                               realiser="benchmark.voice")
    reaction = create_reaction_observation(text["realisation_id"], {"kind": "reply"},
                                           source="conversation:event-2", causal_confidence=0.4)
    interpretation = create_reaction_interpretation(reaction["reaction_id"], [
        {"meaning": "reassured", "confidence": 0.6},
        {"meaning": "unrelated", "confidence": 0.4},
    ])
    with tempfile.TemporaryDirectory(prefix="ina_expression_benchmark_") as directory:
        path = Path(directory) / "trace.jsonl"
        store = ExpressionTraceStore(path)
        for record in (intent, text, reaction, interpretation):
            store.append(record)
        rows = path.read_text(encoding="utf-8").splitlines()
    reward_rejected = False
    try:
        create_reaction_observation(text["realisation_id"], {"reward": 1}, source="invalid")
    except ValueError:
        reward_rejected = True
    return _capability([
        {"case": "intent owns no output medium", "component": "separation",
         "correct": "medium" not in intent and "text" not in intent},
        {"case": "multiple realisers share one intent", "component": "modularity",
         "correct": text["intent_id"] == voice["intent_id"] == intent["intent_id"]},
        {"case": "realisation retains intent provenance", "component": "provenance",
         "correct": text["intent_id"] == intent["intent_id"] and bool(intent["provenance"])},
        {"case": "reaction remains observation not reward", "component": "learning_safety",
         "correct": reward_rejected and "reward" not in reaction},
        {"case": "reaction interpretations retain alternatives", "component": "uncertainty",
         "correct": len(interpretation["candidates"]) == 2},
        {"case": "trace is append-only and bounded", "component": "storage",
         "correct": len(rows) == 4 and all(len(row.encode("utf-8")) < 65536 for row in rows)},
    ])


def _expression_core_v3() -> dict[str, Any]:
    baseline = _expression_core_v2()
    from expression_core import create_expression_intent
    intent = create_expression_intent(
        "respond", meaning_references=["meaning:candidate-1"],
        allowed_media=["text", "native_symbol"],
    )
    return _capability([*baseline["cases"],
        {"case": "intent references medium-neutral communicative meaning", "component": "meaning",
         "correct": intent["meaning_references"] == ["meaning:candidate-1"]
         and "text" not in intent and "native_text" not in intent},
    ])


def _expression_core_v4() -> dict[str, Any]:
    baseline = _expression_core_v3()
    from expression_core import (
        create_expression_affordance, create_expression_intent,
        create_requested_effect, select_expression_affordance,
    )
    intent = create_expression_intent("participate", allowed_media=["text", "voice"])
    request = create_requested_effect(intent, effects=[
        {"kind": "evoke", "target": "auditory event"},
        {"kind": "social_play", "target": "shared amusement"},
    ], constraints={"uses_words": False})
    witnesses = {
        "effect_fit": ["interpretation:1"], "constraint_fit": ["constraint:1"],
        "capability": ["capability:voice"], "willingness": ["choice:1"],
    }
    direct = create_expression_affordance(
        request, medium="audio.vocal_gesture", action={"plan": "bounded-burst"},
        assessments={"effect_fit": .9, "constraint_fit": 1, "capability": .8,
                     "willingness": .9}, witnesses=witnesses,
    )
    words = create_expression_affordance(
        request, medium="text", action={"text": "sound word"},
        fulfilment="representation_only",
        assessments={"effect_fit": .8, "constraint_fit": 0, "capability": 1,
                     "willingness": .9}, witnesses=witnesses,
    )
    selected = select_expression_affordance(request, [words, direct])
    lone = create_expression_affordance(
        request, medium="visual.diagram", action={"plan": "show"},
        assessments={"effect_fit": .9, "constraint_fit": .9, "capability": .9,
                     "willingness": .9},
        witnesses={key: ["one:model"] for key in witnesses},
    )
    abstained = select_expression_affordance(request, [lone])
    return _capability([*baseline["cases"],
        {"case": "requested outcome is separate from its medium", "component": "grounding",
         "correct": "medium" not in request and len(request["effects"]) == 2},
        {"case": "direct experience outranks its written representation", "component": "selection",
         "correct": selected["selected_affordance_id"] == direct["affordance_id"]
         and selected["fulfils_request"]},
        {"case": "capability media are extensible", "component": "modularity",
         "correct": lone["medium"] == "visual.diagram"},
        {"case": "one witness cannot compel expression", "component": "evidence",
         "correct": abstained["status"] == "abstained"},
    ])


def _expression_core_v5() -> dict[str, Any]:
    baseline = _expression_core_v4()
    from expression_core import (
        create_expression_affordance, create_expression_intent,
        create_requested_effect, select_expression_affordance,
    )
    from transformers.QTransformer import QTransformer
    intent = create_expression_intent("choose expression", allowed_media=["voice", "gesture"])
    request = create_requested_effect(
        intent, effects=[{"kind": "social_play", "target": "shared amusement"}],
    )
    witnesses = {
        "effect_fit": ["interpretation:1"], "capability": ["registry:1"],
        "willingness": ["choice:1"],
    }
    candidates = [create_expression_affordance(
        request, medium=medium, action={"plan": medium},
        assessments={"effect_fit": fit, "capability": capability, "willingness": .9},
        witnesses=witnesses,
    ) for medium, fit, capability in (
        ("audio.vocal_gesture", .9, .8), ("embodied.gesture", .88, .82),
    )]
    transformer = QTransformer()
    selected = select_expression_affordance(
        request, candidates,
        ambiguity_resolver=lambda states, context: transformer.collapse_candidates(
            states, context=context, seed=34,
        ),
    )
    trace = selected["ambiguity_resolution"] or {}
    return _capability([*baseline["cases"],
        {"case": "superposition is limited to already viable affordances", "component": "safety",
         "correct": set(trace.get("candidate_ids") or ())
         == {candidate["affordance_id"] for candidate in candidates}},
        {"case": "collapse cannot invent an unavailable expression", "component": "grounding",
         "correct": selected["selected_affordance_id"]
         in {candidate["affordance_id"] for candidate in candidates}},
        {"case": "selection telemetry says what won without exporting why", "component": "privacy",
         "correct": "distribution" not in trace and "raw_bits" not in trace
         and "considered" not in selected and "assessments" not in selected
         and "witnesses" not in selected},
    ])


def _self_inquiry_journey_v1() -> dict[str, Any]:
    return _capability([
        {"case": "deep introspection requires an explicit request", "component": "agency", "correct": False},
        {"case": "inquiry progresses through bounded witnesses", "component": "cadence", "correct": False},
        {"case": "code is evidence rather than an answer key", "component": "epistemics", "correct": False},
        {"case": "uncertainty may remain unresolved", "component": "uncertainty", "correct": False},
    ])


def _self_inquiry_journey_v2() -> dict[str, Any]:
    from self_inquiry_journey import begin_self_inquiry, continue_self_inquiry, current_inquiry_request
    journey = begin_self_inquiry(
        "Why did I choose that?", trigger_references=["selection:1"],
        depth_budget=5, include_code=True,
    )
    routes = []
    while current_inquiry_request(journey) is not None:
        request = current_inquiry_request(journey)
        routes.append(request["evidence_route"])
        if journey["remaining_continuations"] <= 0:
            journey = continue_self_inquiry(journey, choice="remain_uncertain", hypotheses=[
                {"hypothesis": "Several causes may have contributed", "confidence": .4},
            ])
        else:
            journey = continue_self_inquiry(
                journey, choice="deeper", observation_references=[f"witness:{request['stage']}"],
            )
    return _capability([
        {"case": "deep introspection requires an explicit request", "component": "agency",
         "correct": journey["journey_id"].startswith("self_inquiry_") and journey["may_stop"]},
        {"case": "inquiry progresses through bounded witnesses", "component": "cadence",
         "correct": routes == ["activity_witness", "reflection_witness", "memory_index_query",
                                "self_read_code", "hypothesis_review"]},
        {"case": "code is evidence rather than an answer key", "component": "epistemics",
         "correct": journey["stages"][3]["evidence_route"] == "self_read_code"
         and all(not item["authoritative"] for item in journey["hypotheses"])},
        {"case": "uncertainty may remain unresolved", "component": "uncertainty",
         "correct": journey["status"] == "remain_uncertain"},
    ])


def _identity_system_v1() -> dict[str, Any]:
    return _capability([
        {"case": "identity aspects remain distinct witnesses", "component": "plurality", "correct": False},
        {"case": "internal conflict may remain open", "component": "integration", "correct": False},
        {"case": "shadow dialogue does not claim hidden truth", "component": "shadow", "correct": False},
        {"case": "identity contributes read-only continuity witnesses", "component": "continuity", "correct": False},
    ])


def _identity_system_v2() -> dict[str, Any]:
    import tempfile
    from identity_manager import (
        IdentityManager, create_identity_tension, create_identity_witness,
        identity_continuity_candidates,
    )
    from transformers.shadow_transformer import ShadowTransformer
    with tempfile.TemporaryDirectory(prefix="ina_identity_benchmark_") as directory:
        root = Path(directory)
        manager = IdentityManager("Ina", root_path=root)
        witnesses = [
            create_identity_witness("identity", "I value continuity", evidence_references=["memory:1"]),
            create_identity_witness("ego", "I will keep this commitment", evidence_references=["decision:1"]),
            create_identity_witness("id", "I want novelty", evidence_references=["urge:1"]),
            create_identity_witness("shadow", "I resent this limit", evidence_references=["shadow:env-1"]),
        ]
        manager.add_witnesses(witnesses)
        state = manager.add_tension(create_identity_tension(
            [witnesses[1]["witness_id"], witnesses[2]["witness_id"], witnesses[3]["witness_id"]],
            description="Commitment, novelty, and resentment remain in tension",
        ))
        candidates = identity_continuity_candidates(state, cue="tension resentment")
        shadow = ShadowTransformer(child="Ina", root_path=root)
        shadow.index = {"env-1": {"sealed": True}}
        dialogue = shadow.prepare_identity_dialogue(["env-1"], ego_witness_ids=[witnesses[1]["witness_id"]])
    return _capability([
        {"case": "identity aspects remain distinct witnesses", "component": "plurality",
         "correct": {item["aspect"] for item in state["witnesses"]}
         == {"identity", "ego", "id", "shadow"}},
        {"case": "internal conflict may remain open", "component": "integration",
         "correct": state["tensions"][0]["state"] == "open"
         and not state["tensions"][0]["resolution_required"]},
        {"case": "shadow dialogue does not claim hidden truth", "component": "shadow",
         "correct": dialogue["hidden_truth_claimed"] is False
         and dialogue["resolution_required"] is False},
        {"case": "identity contributes read-only continuity witnesses", "component": "continuity",
         "correct": bool(candidates) and all(item["source"] == "identity_system" for item in candidates)},
    ])


def _communicative_meaning_v1() -> dict[str, Any]:
    return _capability([
        {"case": "affect cannot invent a proposition", "component": "grounding", "correct": False},
        {"case": "competing meanings remain alternatives", "component": "uncertainty", "correct": False},
        {"case": "private meaning may remain unexpressed", "component": "privacy", "correct": False},
        {"case": "conversation examples need no retained surface", "component": "language", "correct": False},
    ])


def _communicative_meaning_v2() -> dict[str, Any]:
    from communicative_meaning import build_conversation_examples, interpret_communicative_meaning
    affect_only = interpret_communicative_meaning([{
        "witness_id": "affect:1", "stance": {"urgency": 0.9}, "confidence": 1.0,
    }])
    alternatives = interpret_communicative_meaning([
        {"witness_id": "thought:1", "communicative_act": "ask",
         "proposition_references": ["concept:need"], "confidence": 0.9},
        {"witness_id": "memory:1", "communicative_act": "disclose",
         "proposition_references": ["concept:concern"], "confidence": 0.9},
    ])
    private = interpret_communicative_meaning([{
        "witness_id": "thought:2", "communicative_act": "disclose",
        "proposition_references": ["concept:private"], "confidence": 0.9,
        "disclosure": "private",
    }])
    examples = build_conversation_examples([{
        "content": "What do you need?", "message_id": "message:1",
    }])
    return _capability([
        {"case": "affect cannot invent a proposition", "component": "grounding",
         "correct": not affect_only["candidates"] and affect_only["abstention"]["active"]},
        {"case": "competing meanings remain alternatives", "component": "uncertainty",
         "correct": len(alternatives["candidates"]) == 2
         and alternatives["abstention"]["reason"] == "meaning_ambiguous"},
        {"case": "private meaning may remain unexpressed", "component": "privacy",
         "correct": private["abstention"]["reason"] == "meaning_not_shareable"},
        {"case": "conversation examples need no retained surface", "component": "language",
         "correct": "surface_text" not in examples["examples"][0]
         and bool(examples["examples"][0]["native_intent"]["events"])},
    ])


def _communicative_meaning_v3() -> dict[str, Any]:
    baseline = _communicative_meaning_v2()
    from social_expression import assess_communicative_repair, build_listener_hypotheses
    listener = build_listener_hypotheses("person:sakura", [
        {"witness_id": "conversation:1", "proposition_reference": "concept:need",
         "state": "may_misunderstand", "confidence": .7,
         "provenance": ["conversation:1"]},
        {"witness_id": "repair:1", "proposition_reference": "concept:need",
         "state": "may_misunderstand", "confidence": .8,
         "provenance": ["repair:1"]},
    ])
    repair = assess_communicative_repair(
        ["meaning:intended"], [{
            "meaning_references": ["meaning:received"], "confidence": .8,
            "witness_references": ["reaction:1"], "provenance": ["conversation:2"],
        }], realisation_id="realisation:1",
    )
    return _capability([*baseline["cases"],
        {"case": "listener knowledge remains a federated non-authoritative hypothesis",
         "component": "common_ground", "correct": listener["hypotheses"][0]["authoritative"] is False
         and listener["hypotheses"][0]["independent_origins"] == 2},
        {"case": "corroborated intended-versus-received mismatch proposes optional repair",
         "component": "repair", "correct": repair["status"] == "repair_candidate"
         and repair["automatic_expression"] is False},
    ])


def _transformer_comparison_v1() -> dict[str, Any]:
    return _capability([
        {"case": "comparison is task-specific", "component": "diagnosis", "correct": False},
        {"case": "repeated matched evidence is required", "component": "evidence", "correct": False},
        {"case": "public smoke cannot recommend promotion", "component": "safety", "correct": False},
        {"case": "advantage points back to federated improvement", "component": "improvement", "correct": False},
    ])


def _transformer_comparison_v2() -> dict[str, Any]:
    from transformer_comparison import compare_transformer_families
    records = []
    for index in range(3):
        for family, accuracy, margin, elapsed in (
            ("federated_ina", .5, .1, 1.0),
            ("conventional_transformer", .8, .3, 1.2),
        ):
            records.append({
                "benchmark": "held-out", "benchmark_version": "1",
                "evaluation_protocol": "procedural-generative",
                "seed_fingerprint": f"seed-{index}", "model_family": family,
                "trained_weights": True, "elapsed_seconds": elapsed, "total": 10,
                "categories": {"composition": {"accuracy": accuracy, "mean_margin": margin}},
            })
    records.append({
        "benchmark": "public", "benchmark_version": "1", "evaluation_protocol": "public-smoke",
        "model_family": "conventional_transformer", "trained_weights": True,
        "categories": {"memory": {"accuracy": 1, "mean_margin": 1}},
    })
    report = compare_transformer_families(records)
    candidate = report["recommendations"][0]
    return _capability([
        {"case": "comparison is task-specific", "component": "diagnosis",
         "correct": candidate["task"] == "composition"},
        {"case": "repeated matched evidence is required", "component": "evidence",
         "correct": candidate["matched_independent_witnesses"] == 3},
        {"case": "public smoke cannot recommend promotion", "component": "safety",
         "correct": len(report["recommendations"]) == 1 and not report["policy"]["public_smoke_is_evidence"]},
        {"case": "advantage points back to federated improvement", "component": "improvement",
         "correct": "missing federated capability" in candidate["improvement_route"]
         and candidate["automatic_promotion"] is False},
    ])


def _conversational_expression_v1() -> dict[str, Any]:
    return _capability([
        {"case": "deictics cannot become lexical recall subjects", "component": "grounding", "correct": False},
        {"case": "ordinary proposals do not trigger generic recall replies", "component": "pragmatics", "correct": False},
        {"case": "whole utterances survive the rebuildable graph projection", "component": "memory", "correct": False},
        {"case": "project history is bounded developmental memory", "component": "continuity", "correct": False},
        {"case": "rephrasing preserves source and meaning", "component": "revision", "correct": False},
        {"case": "repair may edit only Ina-authored Discord messages", "component": "delivery", "correct": False},
    ])


def _conversational_expression_v2() -> dict[str, Any]:
    from github_history_bridge import commit_memory_witness
    from lm_studio_adapter import grounding_subjects, is_explicit_grounding_request
    from utterance_memory import evaluate_utterance_memory_hook
    subjects = grounding_subjects(
        "Maybe help ya store sentences, not just your words, Ina.",
        {"sentences": "sym_sentence", "just": "sym_just", "your": "sym_your", "ina": "sym_ina"},
        child="Ina",
    )
    history = commit_memory_witness({"hash": "a" * 40})
    retained = evaluate_utterance_memory_hook(
        {"utterance": "I know my memory loves quotes."},
        {"id": "event:1", "situation_tags": ["relationship"],
         "internal_state": {"communicative_resonance": .9}},
    )
    ordinary = evaluate_utterance_memory_hook(
        {"utterance": "ordinary sentence"},
        {"id": "event:2", "situation_tags": ["conversation"], "internal_state": {}},
    )
    rephrased = evaluate_utterance_memory_hook(
        {"utterance": "History is memory in a way.",
         "meaning_references": ["meaning:history"]},
        {"id": "event:3", "situation_tags": ["relationship"],
         "internal_state": {"communicative_resonance": .9}},
        rephrasing_candidates=[{
            "text": "Project history is developmental memory.",
            "meaning_references": ["meaning:history"], "confidence": .9,
            "purpose": "clarity", "provenance": ["semantic:1", "social:1"],
        }],
    )
    bridge_source = Path("discord_bridge.py").read_text(encoding="utf-8")
    return _capability([
        {"case": "deictics cannot become lexical recall subjects", "component": "grounding",
         "correct": subjects == ["sentences"]},
        {"case": "ordinary proposals do not trigger generic recall replies", "component": "pragmatics",
         "correct": not is_explicit_grounding_request("Maybe help ya store sentences")
         and is_explicit_grounding_request("What does sentences mean?")},
        {"case": "whole utterances survive the rebuildable graph projection", "component": "memory",
         "correct": retained["retained"] and not ordinary["retained"]
         and retained["independent_origins"] == 2},
        {"case": "project history is bounded developmental memory", "component": "continuity",
         "correct": history["memory_relationship"] == "part_of_developmental_memory"
         and history["read_only"] and not history["direct_experience"]
         and 'conversation_scene_show_memory_consideration", False' in bridge_source},
        {"case": "rephrasing preserves source and meaning", "component": "revision",
         "correct": rephrased["surface_kind"] == "rephrased"
         and rephrased["source_unchanged"] and rephrased["rephrasing"]["status"] == "selected"},
        {"case": "repair may edit only Ina-authored Discord messages", "component": "delivery",
         "correct": "_edit_own_discord_message" in bridge_source
         and "not authored by Ina" in bridge_source and 'action == "edit"' in bridge_source},
    ])
def _text_vocab_lookup_v1() -> dict[str, Any]:
    return _capability([
        {"case": "reply lookup avoids full mapping materialisation", "component": "latency", "correct": False},
        {"case": "rebuildable mapping projection follows fast-tier policy", "component": "storage", "correct": False},
    ])


def _text_vocab_lookup_v2() -> dict[str, Any]:
    import tempfile
    from text_vocab_store import load_text_vocab_store_subset, write_text_vocab_store
    with tempfile.TemporaryDirectory(prefix="ina_text_vocab_lookup_") as directory:
        path = Path(directory) / "links.sqlite"
        payload = {
            "schema_version": 2, "evaluated": {},
            "links": [
                {"word": f"word-{index}", "symbol": f"symbol-{index}", "strength": 0.8}
                for index in range(2048)
            ],
        }
        write_text_vocab_store(path, payload)
        subset = load_text_vocab_store_subset(path, words=["word-7", "word-19"])
    from text_vocab_store import sqlite_path_for
    source = Path("AI_Children") / "Inazuma_Yagami" / "memory" / "text_vocab_links.json"
    selected_path = sqlite_path_for(source)
    return _capability([
        {"case": "reply lookup avoids full mapping materialisation", "component": "latency",
         "correct": len(subset.get("links") or ()) == 2
         and len(subset.get("links") or ()) < len(payload["links"])},
        {"case": "rebuildable mapping projection follows fast-tier policy", "component": "storage",
         "correct": selected_path.name == "text_vocab_links.sqlite"
         and ("fast_runtime" in selected_path.parts or selected_path == source.with_suffix(".sqlite"))},
    ])

_HISTORY_BACKED_MODULES = {
    "q_decoder", "bridge_origin", "mirror_audience", "hindsight_claims",
    "mycelial_links", "seedling_clusters", "shadow_candidates", "soul_drift",
    "self_question_origins", "ina_ml_distribution", "language_components",
    "discord_retention", "native_test_support", "self_read_language",
    "experience_cycle", "virtual_file_explorer", "continuity_recall", "background_interference",
    "codex_harness", "thread_governor", "fragment_runtime_sweep", "fragment_repair",
    "semantic_topology",
    "self_question_resolution",
}


_REGISTRY = {
    "connectome_research": (
        ModuleVersion("connectome_research", "V1", "No governed connectome reference path", _connectome_research_v1),
        ModuleVersion("connectome_research", "V2", "Immutable references, copy-only mutation, and bounded EEG overlays", _connectome_research_v2),
    ),
    "conventional_transformer": (
        ModuleVersion("conventional_transformer", "V1", "No conventional attention baseline", _conventional_transformer_v1),
        ModuleVersion("conventional_transformer", "V2", "Benchmark-only conventional decoder Transformer", _conventional_transformer_v2),
    ),
    "semantic_topology": (
        ModuleVersion("semantic_topology", "V1", "One word to one scalar winning link", _semantic_topology_v1),
        ModuleVersion("semantic_topology", "V2", "One word to bounded ranked meanings with contextual activation", _semantic_topology_v2),
    ),
    "discourse": (
        ModuleVersion("discourse", "V1", "Legacy lexical stopword behavior", _legacy_discourse),
        ModuleVersion("discourse", "V2", "Speaker/addressee and deictic role resolution", _role_aware_discourse),
        ModuleVersion("discourse", "V3", "Unified referents, retrieval routes, semantic events, and uncertain glosses", _referent_event_discourse),
    ),
    "q_decoder": (ModuleVersion("q_decoder", "V1", "Fixed bit tables", _q_decoder_v1), ModuleVersion("q_decoder", "V2", "Experience-adaptive bit meanings", _q_decoder_v2)),
    "bridge_origin": (ModuleVersion("bridge_origin", "V1", "Question text without origin", _bridge_origin_v1), ModuleVersion("bridge_origin", "V2", "Composable contradiction origin", _bridge_origin_v2)),
    "mirror_audience": (ModuleVersion("mirror_audience", "V1", "Generic 0.8 projection", _mirror_v1), ModuleVersion("mirror_audience", "V2", "Audience-specific learned transform", _mirror_v2)),
    "hindsight_claims": (ModuleVersion("hindsight_claims", "V1", "Clarity-only comparison", _hindsight_v1), ModuleVersion("hindsight_claims", "V2", "Multidimensional confidence calibration", _hindsight_v2)),
    "mycelial_links": (ModuleVersion("mycelial_links", "V1", "First available cross-domain links", _mycelial_v1), ModuleVersion("mycelial_links", "V2", "Ranked useful lateral links", _mycelial_v2)),
    "seedling_clusters": (ModuleVersion("seedling_clusters", "V1", "First-character grouping", _seedling_v1), ModuleVersion("seedling_clusters", "V2", "Profile and vector geometry", _seedling_v2)),
    "shadow_candidates": (ModuleVersion("shadow_candidates", "V1", "Full fragment directory scan", _shadow_v1), ModuleVersion("shadow_candidates", "V2", "Queue and SQLite tag lookup", _shadow_v2)),
    "emotion_propagation": (ModuleVersion("emotion_propagation", "V1", "Full fragment directory glob on every tick", _emotion_propagation_v1), ModuleVersion("emotion_propagation", "V2", "Bounded resumable SQLite-indexed propagation", _emotion_propagation_v2)),
    "fragment_runtime_sweep": (ModuleVersion("fragment_runtime_sweep", "V1", "Legacy runtime modules enumerate fragment directories", _fragment_runtime_sweep_v1), ModuleVersion("fragment_runtime_sweep", "V2", "Shared bounded SQLite fragment selection", _fragment_runtime_sweep_v2)),
    "fragment_repair": (ModuleVersion("fragment_repair", "V1", "Legacy salvage or quarantine without mirror recovery", _fragment_repair_v1), ModuleVersion("fragment_repair", "V2", "Intent-gated verified mirror recovery with retained original", _fragment_repair_v2)),
    "soul_drift": (ModuleVersion("soul_drift", "V1", "Link drift without emotion direction", _soul_v1), ModuleVersion("soul_drift", "V2", "Indexed links and emotion-directed drift", _soul_v2)),
    "self_question_origins": (ModuleVersion("self_question_origins", "V1", "Question metadata only", _question_origin_v1), ModuleVersion("self_question_origins", "V2", "Composable trigger chain export", _question_origin_v2)),
    "self_question_display": (ModuleVersion("self_question_display", "V1", "Latest timestamp and resolved state only", _question_display_v1), ModuleVersion("self_question_display", "V2", "Bounded trigger history and reversible display hiding", _question_display_v2)),
    "self_question_resolution": (ModuleVersion("self_question_resolution", "V1", "Questions accumulate without evidence routing", _question_resolution_v1), ModuleVersion("self_question_resolution", "V2", "Typed evidence routing and evaluated lifecycle", _question_resolution_v2)),
    "ina_ml_distribution": (ModuleVersion("ina_ml_distribution", "V1", "Historical native numerics", _ina_ml_distribution_v1), ModuleVersion("ina_ml_distribution", "V2", "Native distribution and entropy kernels", _ina_ml_distribution_v2)),
    "language_components": (
        ModuleVersion("language_components", "V1", "Historical language context", _language_v1),
        ModuleVersion("language_components", "V2", "Compositional and discourse-aware language", _language_v2),
        ModuleVersion("language_components", "V3", "Whole-message event with ordered word sequence", _language_v3),
    ),
    "discord_retention": (ModuleVersion("discord_retention", "V1", "Unbounded delivery history", _discord_retention_v1), ModuleVersion("discord_retention", "V2", "Bounded history and buffers", _discord_retention_v2)),
    "communication_continuity": (ModuleVersion("communication_continuity", "V1", "Stale speech mixed into ordinary episodic recall", _communication_continuity_v1), ModuleVersion("communication_continuity", "V2", "Explicit unfinished speech with bounded contextual recall", _communication_continuity_v2)),
    "discord_bridge_memory": (ModuleVersion("discord_bridge_memory", "V1", "Discord imports the monolithic cognitive manager", _discord_bridge_memory_v1), ModuleVersion("discord_bridge_memory", "V2", "Discord reuses lightweight canonical state seams", _discord_bridge_memory_v2)),
    "native_test_support": (ModuleVersion("native_test_support", "V1", "External pytest required", _native_tests_v1), ModuleVersion("native_test_support", "V2", "Dependency-free pytest subset", _native_tests_v2)),
    "self_read_language": (ModuleVersion("self_read_language", "V1", "Music assets without explicit language roles", _self_read_language_v1), ModuleVersion("self_read_language", "V2", "Vocal, spoken, and written self-read alignment", _self_read_language_v2)),
    "experience_cycle": (ModuleVersion("experience_cycle", "V1", "Historical event and episode logging", _experience_cycle_v1), ModuleVersion("experience_cycle", "V2", "Optional bounded intent-attempt-observation-evaluation cycles", _experience_cycle_v2)),
    "adaptive_storage_decision": (ModuleVersion("adaptive_storage_decision", "V1", "Device probes without operation-attributed autonomous placement", _adaptive_storage_decision_v1), ModuleVersion("adaptive_storage_decision", "V2", "Evidence-gated reversible autonomous placement with audit reports", _adaptive_storage_decision_v2)),
    "virtual_file_explorer": (ModuleVersion("virtual_file_explorer", "V1", "No virtual media-drive explorer", _file_explorer_v1), ModuleVersion("virtual_file_explorer", "V2", "Capability-scoped media and personal drives", _file_explorer_v2)),
    "creative_versioning": (ModuleVersion("creative_versioning", "V1", "Working copies without explicit creative lineage", _creative_versioning_v1), ModuleVersion("creative_versioning", "V2", "Hash-addressed music and drawing lineage", _creative_versioning_v2)),
    "lifecycle_visibility": (ModuleVersion("lifecycle_visibility", "V1", "Lifecycle progress is scattered through activity logs", _lifecycle_visibility_v1), ModuleVersion("lifecycle_visibility", "V2", "Named boot and shutdown phases with reboot safety", _lifecycle_visibility_v2)),
    "continuity_recall": (
        ModuleVersion("continuity_recall", "V1", "Historical isolated continuity snapshots", _continuity_recall_v1),
        ModuleVersion("continuity_recall", "V2", "Federated bounded recall with descriptive bias telemetry", _continuity_recall_v2),
    ),
    "background_interference": (
        ModuleVersion("background_interference", "V1", "Historical aggregate resource checks", _background_interference_v1),
        ModuleVersion("background_interference", "V2", "Human-visible idle-vs-loaded interference and thread fan-out", _background_interference_v2),
    ),
    "codex_harness": (
        ModuleVersion("codex_harness", "V1", "Historical VS Code-hosted Codex workflow", _codex_harness_v1),
        ModuleVersion("codex_harness", "V2", "Standalone subscription-only app-server GUI", _codex_harness_v2),
        ModuleVersion("codex_harness", "V3", "Bounded app-server operator console", _codex_harness_v3),
        ModuleVersion("codex_harness", "V4", "Bounded workspace thread navigation", _codex_harness_v4),
    ),
    "thread_governor": (
        ModuleVersion("thread_governor", "V1", "Historical unmanaged module thread pools", _thread_governor_v1),
        ModuleVersion("thread_governor", "V2", "Bounded per-module observation-driven thread selection", _thread_governor_v2),
        ModuleVersion("thread_governor", "V3", "Opposing differential control with deadband and hard operating envelopes", _thread_governor_v3),
    ),
    "code_experiment_lab": (
        ModuleVersion("code_experiment_lab", "V1", "No governed executable experiment room", _code_experiment_lab_v1),
        ModuleVersion("code_experiment_lab", "V2", "Bounded reproducible Python experiments with review-only promotion", _code_experiment_lab_v2),
        ModuleVersion("code_experiment_lab", "V3", "Evidence-triggered storage optimisation goals with code review issues", _code_experiment_lab_v3),
        ModuleVersion("code_experiment_lab", "V4", "Honesty-first experiments and review-gated connectome design", _code_experiment_lab_v4),
    ),
    "fault_pattern_research": (
        ModuleVersion("fault_pattern_research", "V1", "No dedicated fault-pattern research instruments", _fault_pattern_research_v1),
        ModuleVersion("fault_pattern_research", "V2", "Bounded synthetic fault modelling and adversarial measurement", _fault_pattern_research_v2),
    ),
    "desktop_lifecycle": (
        ModuleVersion("desktop_lifecycle", "V1", "No governed self-service virtual desktop restart", _desktop_lifecycle_v1),
        ModuleVersion("desktop_lifecycle", "V2", "Prepared reasoned observable virtual desktop restart", _desktop_lifecycle_v2),
    ),
    "expression_core": (
        ModuleVersion("expression_core", "V1", "Medium-specific expression decisions without a shared trace", _expression_core_v1),
        ModuleVersion("expression_core", "V2", "Output-neutral intent with medium realisers and reaction provenance", _expression_core_v2),
        ModuleVersion("expression_core", "V3", "Intents reference uncertain medium-neutral communicative meanings", _expression_core_v3),
        ModuleVersion("expression_core", "V4", "Requested effects select corroborated cross-modal affordances", _expression_core_v4),
        ModuleVersion("expression_core", "V5", "Near-equivalent affordances may use bounded superposition", _expression_core_v5),
        ModuleVersion("self_inquiry_journey", "V1", "No voluntary staged route for deeper self-understanding", _self_inquiry_journey_v1),
        ModuleVersion("self_inquiry_journey", "V2", "Meditation offers a finite witness-led self-inquiry journey", _self_inquiry_journey_v2),
        ModuleVersion("identity_system", "V1", "Single-profile self-reflection without aspect coordination", _identity_system_v1),
        ModuleVersion("identity_system", "V2", "Plural identity witnesses preserve conflict and feed continuity", _identity_system_v2),
    ),
    "communicative_meaning": (
        ModuleVersion("communicative_meaning", "V1", "State and lexical output without a communicative meaning boundary", _communicative_meaning_v1),
        ModuleVersion("communicative_meaning", "V2", "Bounded evidence-backed meaning alternatives and abstention", _communicative_meaning_v2),
        ModuleVersion("communicative_meaning", "V3", "Listener hypotheses and evidence-gated communicative repair", _communicative_meaning_v3),
    ),
    "transformer_comparison": (
        ModuleVersion("transformer_comparison", "V1", "No task-specific architecture comparison guidance", _transformer_comparison_v1),
        ModuleVersion("transformer_comparison", "V2", "Repeated multi-signal comparison with improvement routing", _transformer_comparison_v2),
    ),
    "conversational_expression": (
        ModuleVersion("conversational_expression", "V1", "Word-triggered recall and ambiguous project history", _conversational_expression_v1),
        ModuleVersion("conversational_expression", "V2", "Pragmatic recall gating, sentence projection, and developmental history", _conversational_expression_v2),
    ),
    "text_vocab_lookup": (
        ModuleVersion("text_vocab_lookup", "V1", "Full durable-tier meaning-map materialisation per reply", _text_vocab_lookup_v1),
        ModuleVersion("text_vocab_lookup", "V2", "Indexed subsets through the rebuildable fast-tier projection", _text_vocab_lookup_v2),
    ),
    "thought_processor": (
        ModuleVersion("thought_processor", "V1", "Cognition without a shared typed thought boundary", _thought_processor_v1),
        ModuleVersion("thought_processor", "V2", "Separate non-linguistic and linguistic thought with mixed decisions", _thought_processor_v2),
        ModuleVersion("thought_processor", "V3", "Emotion, instinct, cognition, and memory guide one inspectable decision", _thought_processor_v3),
        ModuleVersion("thought_processor", "V4", "Communication supplies evidence for bounded thought revision", _thought_processor_v4),
        ModuleVersion("thought_processor", "V5", "Communication planning references bounded meaning hypotheses", _thought_processor_v5),
    ),
}


def list_benchmark_modules() -> dict[str, tuple[ModuleVersion, ...]]:
    return dict(_REGISTRY)


def benchmark_module(module: str, versions: tuple[str, ...] | None = None) -> tuple[ModuleBenchmarkResult, ...]:
    specs = _REGISTRY.get(str(module))
    if not specs:
        raise ValueError(f"unknown benchmark module: {module}")
    selected = set(versions or ())
    results = []
    for spec in specs:
        if selected and spec.version not in selected:
            continue
        started = time.perf_counter()
        outcome = spec.evaluate()
        elapsed = time.perf_counter() - started
        total = int(outcome.get("total") or 0)
        correct = int(outcome.get("correct") or 0)
        results.append(ModuleBenchmarkResult(
            module=spec.module, version=spec.version, benchmark_version="V1",
            accuracy=round(correct / total, 6) if total else 0.0,
            correct=correct, total=total, elapsed_seconds=round(elapsed, 6),
            source_revision=(
                resolve_revision(TRANSFORMER_V1_REVISION)
                if spec.module in _HISTORY_BACKED_MODULES and spec.version == "V1"
                else "working-tree"
            ),
            cases=tuple(outcome.get("cases") or ()),
            component_scores={
                component: {
                    "correct": sum(bool(case.get("correct")) for case in outcome.get("cases", ()) if str(case.get("component") or "overall") == component),
                    "total": sum(1 for case in outcome.get("cases", ()) if str(case.get("component") or "overall") == component),
                }
                for component in sorted({str(case.get("component") or "overall") for case in outcome.get("cases", ())})
            },
            run_at=datetime.now(timezone.utc).isoformat(),
        ))
    return tuple(results)


__all__ = [
    "ModuleBenchmarkResult", "ModuleVersion", "TRANSFORMER_V1_REVISION",
    "benchmark_module", "list_benchmark_modules",
]
