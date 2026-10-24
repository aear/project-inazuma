"""Explicit synthetic component assessments; never a score for Ina herself."""
from __future__ import annotations

from pathlib import Path
import tempfile

BASELINE = "a071fbc08f126fa6430ee65e76a5f7e42c6c507c"


def language_cases(module, context_module):
    from discourse_context import build_discourse_context
    rows = []
    def check(name, component, correct):
        rows.append({"case": name, "component": component, "correct": bool(correct)})
    # Unfamiliar names/objects and swapped conversational roles share no fixed answer identity.
    for speaker, listener, text, surface, expected in (
        ("Mira", "Rowan", "I remember your lantern.", "your", "rowan"),
        ("Rowan", "Mira", "You remember my compass.", "my", "rowan"),
        ("Aster", "Vale", "I opened the gate.", "i", "aster"),
        ("Vale", "Aster", "You found the map.", "you", "aster"),
    ):
        memory = module.DiscourseEntityMemory(); memory.mention("Distractor")
        discourse = build_discourse_context(text, speaker=speaker, addressee=listener)
        result = module.analyze_utterance(text, context={"discourse": discourse}, discourse=memory)
        matches = [ref for ref in result["referents"] if ref["surface"] == surface]
        check(text, "participant_transfer", bool(matches) and matches[0]["resolved"] == expected)
    memory = module.DiscourseEntityMemory(); memory.mention("Distractor")
    for text in ("I found my key.", "You found your key."):
        refs = module.analyze_utterance(text, discourse=memory)["referents"]
        check("unknown participant: " + text, "uncertainty", bool(refs) and all(ref["resolved"] is None for ref in refs))
    context = build_discourse_context("They found their key.", speaker="Mira", addressee="Rowan", mentioned_entities=("Vale", "Aster"))
    result = module.analyze_utterance("They found their key.", context={"discourse": context})
    check("authoritative competing entities retained", "discourse", all({r["id"] for r in ref["candidates"]} == {"vale", "aster"} and ref["resolved"] is None for ref in result["referents"]))
    for text, predicate, negated in (("Mira didn't open the gate.", "open", True), ("Mira opened the gate.", "open", False)):
        clause = module.analyze_utterance(text)["clauses"][0]
        check(text, "composition_morphology", clause["predicate"] == predicate and clause["negated"] == negated)
    memory = module.DiscourseEntityMemory(); module.analyze_utterance("Rowan arrived.", discourse=memory)
    check("ordinary prior-turn recall retained", "memory", module.analyze_utterance("He remembered it.", discourse=memory)["referents"][0]["resolved"] == "rowan")
    subject_context = build_discourse_context("I opened the gate.", speaker="Mira", addressee="Rowan")
    subject_result = module.analyze_utterance("I opened the gate.", context={"discourse": subject_context})
    alternatives = subject_result["whole_utterance_interpretations"]
    check("subject alternatives do not become possession", "expression_fidelity", bool(alternatives) and all("subject" in row["changes"] and "possessor" not in row["meaning"] for row in alternatives))
    learner = module.ConstructionLearner()
    a = module.analyze_utterance("Mira gave Rowan a lantern.", constructions=learner)
    b = module.analyze_utterance("Vale gave Aster a compass.", constructions=learner)
    check("construction transfers across names and objects", "construction_learning", a["constructions"][0]["pattern"] == b["constructions"][0]["pattern"] and b["constructions"][0]["observations"] == 2)
    with tempfile.TemporaryDirectory(prefix="ina_construction_assessment_") as directory:
        path = Path(directory)/"patterns.json"
        learner = module.ConstructionLearner(path)
        try:
            first = learner.observe(a["clauses"], source_event="fixture:1")
            restored = module.ConstructionLearner(path)
            replay = restored.observe(a["clauses"], source_event="fixture:1")
            fresh = restored.observe(b["clauses"], source_event="fixture:2")
            check("reload does not learn same event twice", "learning_provenance", replay[0]["observations"] == first[0]["observations"] and replay[0]["replay"])
            check("new event still learns", "learning_provenance", fresh[0]["observations"] == first[0]["observations"]+1)
            analyzed = module.analyze_utterance("Mira gave Rowan a lantern.", constructions=restored, source_event="fixture:2")
            check("analysis forwards replay identity", "learning_provenance", analyzed["constructions"][0]["replay"])
        except TypeError:
            check("reload does not learn same event twice", "learning_provenance", False)
            check("new event still learns", "learning_provenance", False)
            check("analysis forwards replay identity", "learning_provenance", False)
    snapshot = context_module.build_language_context_snapshot({"source_text": "I remember your lantern.",
        "current_speaker": {"id": "mira", "name": "Mira"}, "learned_media_guidance": {},
        "language_state_signals": {"current_prediction": {}, "machine_semantics": {}, "emotion_snapshot": {}}},
        child="Fixture", logic_reader=False)
    refs = {ref["surface"]: ref["resolved"] for ref in snapshot["linguistic_analysis"]["referents"]}
    check("conversation context reaches auxiliary analysis", "context_integration", refs.get("i") == "mira" and refs.get("your") == "self")
    return {"cases": rows, "correct": sum(r["correct"] for r in rows), "total": len(rows)}


def language_v1():
    from historical_source import historical_module
    language = historical_module("language_intelligence.py", BASELINE)
    context = historical_module("language_context.py", BASELINE)
    context.analyze_utterance = language.analyze_utterance
    context.DiscourseEntityMemory = language.DiscourseEntityMemory
    result = language_cases(language, context)
    result["baseline_revision"] = BASELINE
    return result


def language_v2():
    import language_intelligence, language_context
    return language_cases(language_intelligence, language_context)


def speech_v1():
    from historical_source import historical_text
    source = historical_text("personal_tool_runtime.py", BASELINE)
    # Historical absence of this tool; not an assessment of native symbolic speech.
    present = '"speech_render"' in source
    cases = [{"case": "optional local text speech tool available", "component": "tool_access", "correct": present}]
    return {"cases": cases, "correct": int(present), "total": 1, "baseline_revision": BASELINE}


def speech_v2():
    from personal_tool_runtime import execute_personal_tool_command, capability_catalog
    rows = [{"case": "optional local text speech tool available", "component": "tool_access", "correct": "speech_render" in capability_catalog()["commands"]}]
    with tempfile.TemporaryDirectory(prefix="ina_speech_assessment_") as directory:
        for text in ("The lantern is beside the gate.", "I did not say she took the key.", "Whose compass did you find?"):
            value = execute_personal_tool_command({"action": "speech_render", "text": text}, child="Fixture", project_root=directory,
                config={"ina_hdd_writable_path": str(Path(directory)/"personal")})["value"]
            rows.append({"case": text, "component": "audio_render", "correct": value["status"] == "rendered",
                         "status": value["status"], "duration_seconds": value.get("duration_seconds"),
                         "audio_sha256": value.get("audio_sha256"), "intelligibility": "not_assessed"})
            rows.append({"case": "no delivery or automatic learning: " + text, "component": "choice_preservation",
                         "correct": not value["played"] and not value["published"] and not value["automatic_learning"]})
    return {"cases": rows, "correct": sum(r["correct"] for r in rows), "total": len(rows)}


def transfer_limits(module):
    """Untuned challenge set retained even when it exposes unsupported grammar."""
    rows = []
    for text, expected in (("Mira polished the lens.", "polish"),
                           ("Vale repaired the lantern.", "repair"),
                           ("Aster taught Rowan the song.", "teach")):
        actual = module.analyze_utterance(text)["clauses"][0]["predicate"]
        rows.append({"case": text, "component": "unfamiliar_predicate", "expected": expected,
                     "actual": actual, "correct": actual == expected})
    clause = module.analyze_utterance("The compass was found by Mira.")["clauses"][0]
    rows.append({"case": "irregular passive preserves agent and theme", "component": "semantic_roles",
                 "expected": {"voice": "passive", "subject": "mira", "theme": "the compass"},
                 "actual": {"voice": clause["voice"], "subject": clause["subject"], "theme": clause["arguments"].get("theme")},
                 "correct": clause["voice"] == "passive" and clause["subject"] == "mira" and clause["arguments"].get("theme") == "the compass"})
    refs = module.analyze_utterance("Mira gave Rowan her lantern. Aster gave Vale his compass.")["referents"]
    actual = [ref["surface"] for ref in refs if ref["relation"] == "possessor"]
    rows.append({"case": "both sentence-level possessors retained", "component": "reading_continuity",
                 "expected": ["her", "his"], "actual": actual, "correct": actual == ["her", "his"]})
    return {"cases": rows, "correct": sum(row["correct"] for row in rows), "total": len(rows)}


def limits_v1():
    from historical_source import historical_module
    result = transfer_limits(historical_module("language_intelligence.py", BASELINE))
    result["baseline_revision"] = BASELINE
    return result


def limits_v2():
    import language_intelligence
    return transfer_limits(language_intelligence)
