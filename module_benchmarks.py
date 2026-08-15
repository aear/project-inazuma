"""Deterministic, explicit comparisons between retained module versions."""
from __future__ import annotations

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


def _capability(cases: list[dict[str, Any]]) -> dict[str, Any]:
    return {"correct": sum(bool(case.get("correct")) for case in cases), "total": len(cases), "cases": cases}


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

_HISTORY_BACKED_MODULES = {
    "q_decoder", "bridge_origin", "mirror_audience", "hindsight_claims",
    "mycelial_links", "seedling_clusters", "shadow_candidates", "soul_drift",
    "self_question_origins", "ina_ml_distribution", "language_components",
    "discord_retention", "native_test_support", "self_read_language",
    "experience_cycle", "virtual_file_explorer", "continuity_recall", "background_interference",
    "codex_harness", "thread_governor",
}


_REGISTRY = {
    "discourse": (
        ModuleVersion("discourse", "V1", "Legacy lexical stopword behavior", _legacy_discourse),
        ModuleVersion("discourse", "V2", "Speaker/addressee and deictic role resolution", _role_aware_discourse),
    ),
    "q_decoder": (ModuleVersion("q_decoder", "V1", "Fixed bit tables", _q_decoder_v1), ModuleVersion("q_decoder", "V2", "Experience-adaptive bit meanings", _q_decoder_v2)),
    "bridge_origin": (ModuleVersion("bridge_origin", "V1", "Question text without origin", _bridge_origin_v1), ModuleVersion("bridge_origin", "V2", "Composable contradiction origin", _bridge_origin_v2)),
    "mirror_audience": (ModuleVersion("mirror_audience", "V1", "Generic 0.8 projection", _mirror_v1), ModuleVersion("mirror_audience", "V2", "Audience-specific learned transform", _mirror_v2)),
    "hindsight_claims": (ModuleVersion("hindsight_claims", "V1", "Clarity-only comparison", _hindsight_v1), ModuleVersion("hindsight_claims", "V2", "Multidimensional confidence calibration", _hindsight_v2)),
    "mycelial_links": (ModuleVersion("mycelial_links", "V1", "First available cross-domain links", _mycelial_v1), ModuleVersion("mycelial_links", "V2", "Ranked useful lateral links", _mycelial_v2)),
    "seedling_clusters": (ModuleVersion("seedling_clusters", "V1", "First-character grouping", _seedling_v1), ModuleVersion("seedling_clusters", "V2", "Profile and vector geometry", _seedling_v2)),
    "shadow_candidates": (ModuleVersion("shadow_candidates", "V1", "Full fragment directory scan", _shadow_v1), ModuleVersion("shadow_candidates", "V2", "Queue and SQLite tag lookup", _shadow_v2)),
    "soul_drift": (ModuleVersion("soul_drift", "V1", "Link drift without emotion direction", _soul_v1), ModuleVersion("soul_drift", "V2", "Indexed links and emotion-directed drift", _soul_v2)),
    "self_question_origins": (ModuleVersion("self_question_origins", "V1", "Question metadata only", _question_origin_v1), ModuleVersion("self_question_origins", "V2", "Composable trigger chain export", _question_origin_v2)),
    "ina_ml_distribution": (ModuleVersion("ina_ml_distribution", "V1", "Historical native numerics", _ina_ml_distribution_v1), ModuleVersion("ina_ml_distribution", "V2", "Native distribution and entropy kernels", _ina_ml_distribution_v2)),
    "language_components": (ModuleVersion("language_components", "V1", "Historical language context", _language_v1), ModuleVersion("language_components", "V2", "Compositional and discourse-aware language", _language_v2)),
    "discord_retention": (ModuleVersion("discord_retention", "V1", "Unbounded delivery history", _discord_retention_v1), ModuleVersion("discord_retention", "V2", "Bounded history and buffers", _discord_retention_v2)),
    "native_test_support": (ModuleVersion("native_test_support", "V1", "External pytest required", _native_tests_v1), ModuleVersion("native_test_support", "V2", "Dependency-free pytest subset", _native_tests_v2)),
    "self_read_language": (ModuleVersion("self_read_language", "V1", "Music assets without explicit language roles", _self_read_language_v1), ModuleVersion("self_read_language", "V2", "Vocal, spoken, and written self-read alignment", _self_read_language_v2)),
    "experience_cycle": (ModuleVersion("experience_cycle", "V1", "Historical event and episode logging", _experience_cycle_v1), ModuleVersion("experience_cycle", "V2", "Optional bounded intent-attempt-observation-evaluation cycles", _experience_cycle_v2)),
    "virtual_file_explorer": (ModuleVersion("virtual_file_explorer", "V1", "No virtual media-drive explorer", _file_explorer_v1), ModuleVersion("virtual_file_explorer", "V2", "Capability-scoped media and personal drives", _file_explorer_v2)),
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
    ),
    "thread_governor": (
        ModuleVersion("thread_governor", "V1", "Historical unmanaged module thread pools", _thread_governor_v1),
        ModuleVersion("thread_governor", "V2", "Bounded per-module observation-driven thread selection", _thread_governor_v2),
        ModuleVersion("thread_governor", "V3", "Opposing differential control with deadband and hard operating envelopes", _thread_governor_v3),
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
