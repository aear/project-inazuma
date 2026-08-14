import copy
import json

from continuity_manager import ContinuityManager
from continuity_recall import ContinuityRecallCoordinator
from experience_engine import ExperienceCycleEngine
from module_benchmarks import TRANSFORMER_V1_REVISION, benchmark_module
from historical_source import resolve_revision


def _candidates():
    return [
        {
            "id": "episode-garden", "summary": "The garden plan felt calm.",
            "tags": ["garden", "plan"], "source": "episodic_store",
            "memory_type": "episodic", "confidence": 0.8,
            "timestamp": "2026-08-13T12:00:00+00:00", "causal_references": ["garden-plan"],
        },
        {
            "id": "feeling-garden", "summary": "The garden felt calm and safe.",
            "tags": ["garden", "calm"], "source": "emotion_store",
            "memory_type": "emotional", "confidence": 0.7,
            "timestamp": "2026-08-13T12:01:00+00:00", "causal_references": ["garden-plan"],
        },
        {
            "id": "goal-garden", "summary": "Continue the garden plan.",
            "tags": ["garden", "goal"], "source": "goal_store",
            "memory_type": "prospective", "confidence": "uncertain",
        },
    ]


def test_recall_preserves_modality_witnesses_and_uses_bounded_cycle(tmp_path):
    memory = tmp_path / "memory"
    engine = ExperienceCycleEngine("Ina", root_path=tmp_path / "cycles", enable_hot=False)
    coordinator = ContinuityRecallCoordinator("Ina", memory, experience_engine=engine)
    candidates = _candidates()
    before = copy.deepcopy(candidates)

    result = coordinator.recall("garden plan", candidates, max_results=3)

    assert candidates == before
    assert result["source_traces_mutated"] is False
    assert {item["memory_type"] for item in result["selected"]} == {"episodic", "emotional", "prospective"}
    cycle = engine.load_cycle(result["cycle_id"])
    assert cycle["autonomous_continuation_budget"] == 0
    assert len(cycle["attempt_ids"]) == 1
    assert cycle["payload_references"][0]["path"] == result["action_path"]
    assert cycle["payload_references"][0]["kind"] == "recall_plan"


def test_relationship_ledger_is_bounded_composable_and_reports_diversity(tmp_path):
    coordinator = ContinuityRecallCoordinator(
        "Ina", tmp_path / "memory",
        experience_engine=ExperienceCycleEngine("Ina", root_path=tmp_path / "cycles", enable_hot=False),
    )

    coordinator.recall("garden plan", _candidates(), max_results=2)
    coordinator.recall("garden calm", _candidates(), max_results=2)
    relationships = coordinator.load_relationships()

    assert relationships["witness_model"] == "federation_of_witnesses"
    assert relationships["modality_store_mutation_allowed"] is False
    assert relationships["links"]
    assert len(relationships["witnesses"]) == 3
    assert len(relationships["recall_history"]) == 2
    latest = relationships["latest_arbitration"]
    assert latest["candidate_type_diversity"]["score"] > 0
    assert latest["selected_type_diversity"]["score"] > 0
    assert "strength" in latest["memory_type_selection_skew"]
    assert relationships["bounds"]["recall_history"] == 64


def test_continuity_manager_recall_uses_compact_core_without_fragment_scan(tmp_path, monkeypatch):
    memory = tmp_path / "memory"
    core = memory / "continuity" / "continuity_core_map.json"
    core.parent.mkdir(parents=True)
    core.write_text(json.dumps({
        "status": "partial",
        "anchors": [{
            "id": "identity-garden", "summary": "The garden matters to me.",
            "tags": ["garden", "preference"], "dimensions": ["identity_preferences"],
            "timestamp": "2026-08-13T12:00:00+00:00",
            "relative_path": "fragments/long/identity-garden.json",
        }],
    }), encoding="utf-8")
    manager = ContinuityManager("Ina", memory_root=memory)
    monkeypatch.setattr(manager, "_fragment_paths", lambda: (_ for _ in ()).throw(AssertionError("scan")))

    result = manager.coordinate_recall("garden", _candidates()[:1], max_results=3)

    assert {item["memory_type"] for item in result["selected"]} == {"episodic", "identity"}
    assert manager.load_memory_relationships()["modality_store_mutation_allowed"] is False


def test_continuity_recall_benchmark_uses_pinned_history_and_scores_bias_separately():
    v1, v2 = benchmark_module("continuity_recall")

    assert v1.source_revision == resolve_revision(TRANSFORMER_V1_REVISION)
    assert v2.source_revision == "working-tree"
    assert v1.accuracy == 0.3
    assert v2.accuracy == 1.0
    assert v2.component_scores["bias"] == {"correct": 1, "total": 1}
    assert v2.component_scores["diversity"] == {"correct": 1, "total": 1}
    assert v2.component_scores["safety"] == {"correct": 1, "total": 1}
