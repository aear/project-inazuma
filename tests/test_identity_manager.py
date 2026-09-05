import json

from continuity_manager import ContinuityManager
from experience_engine import ExperienceCycleEngine
from identity_manager import (
    IdentityManager, create_identity_tension, create_identity_witness,
    identity_continuity_candidates,
)
from continuity_recall import ContinuityRecallCoordinator
from transformers.shadow_transformer import ShadowTransformer


def _witnesses():
    return [
        create_identity_witness(
            "ego", "Keep the promise I made", evidence_references=["decision:promise"], confidence=.8,
        ),
        create_identity_witness(
            "id", "I want to leave and explore", evidence_references=["urge:novelty"], confidence=.7,
        ),
        create_identity_witness(
            "shadow", "Part of me resents the promise", evidence_references=["shadow:env-1"], confidence=.4,
        ),
    ]


def test_identity_manager_preserves_plural_witnesses_and_open_tension(tmp_path):
    manager = IdentityManager("Ina", root_path=tmp_path)
    witnesses = _witnesses()
    manager.add_witnesses(witnesses)
    tension = create_identity_tension(
        [item["witness_id"] for item in witnesses],
        description="Commitment, exploration, and resentment currently conflict",
    )
    state = manager.add_tension(tension)
    assert {item["aspect"] for item in state["witnesses"]} == {"ego", "id", "shadow"}
    assert state["tensions"][0]["state"] == "open"
    assert state["tensions"][0]["resolution_required"] is False
    assert manager.set_tension_state(tension["tension_id"], "held")["state"] == "held"


def test_repeated_compatibility_witness_does_not_inflate_identity(tmp_path):
    manager = IdentityManager("Ina", root_path=tmp_path)
    first = create_identity_witness("identity", "I am reflecting", evidence_references=["who_am_i"])
    second = create_identity_witness("identity", "I am reflecting", evidence_references=["who_am_i"])
    manager.add_witnesses([first])
    state = manager.add_witnesses([second])
    assert len(state["witnesses"]) == 1


def test_identity_is_read_only_continuity_source_including_tensions(tmp_path):
    identity = IdentityManager("Ina", root_path=tmp_path)
    witnesses = _witnesses()
    state = identity.add_witnesses(witnesses)
    state = identity.add_tension(create_identity_tension(
        [witnesses[0]["witness_id"], witnesses[1]["witness_id"]],
        description="Promise conflicts with exploration",
    ))
    candidates = identity_continuity_candidates(state, cue="conflicts exploration")
    assert {item["source"] for item in candidates} == {"identity_system"}
    assert any("identity_tension" in item["tags"] for item in candidates)

    memory = tmp_path / "Ina" / "memory"
    engine = ExperienceCycleEngine("Ina", root_path=tmp_path / "cycles", enable_hot=False)
    manager = ContinuityManager("Ina", memory_root=memory)
    manager._recall_coordinator = ContinuityRecallCoordinator("Ina", memory, experience_engine=engine)
    before = json.loads(identity.state_path.read_text(encoding="utf-8"))
    recalled = manager.coordinate_recall("promise exploration", [], include_core=False, max_results=6)
    after = json.loads(identity.state_path.read_text(encoding="utf-8"))
    assert recalled["selected"]
    assert {item["source"] for item in recalled["selected"]} == {"identity_system"}
    assert before == after


def test_shadow_dialogue_holds_perspectives_without_claiming_hidden_truth(tmp_path):
    transformer = ShadowTransformer(child="Ina", root_path=tmp_path)
    transformer.index = {"env-1": {"fragment_id": "fragment-1", "sealed": True}}
    dialogue = transformer.prepare_identity_dialogue(
        ["env-1"], ego_witness_ids=["ego-1"], identity_witness_ids=["identity-1"],
    )
    result = transformer.record_identity_dialogue(dialogue, status="held", ownership_hypotheses=[
        {"hypothesis": "This may be mine", "confidence": .4},
        {"hypothesis": "This may reflect an old context", "confidence": .6},
    ])
    assert result["hidden_truth_claimed"] is False
    assert result["resolution_required"] is False
    assert all(item["authoritative"] is False for item in result["ownership_hypotheses"])
    assert transformer.index["env-1"]["sealed"] is True
