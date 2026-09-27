import pytest

from creativity_observatory import build_profile, make_evidence


def evidence(dimension, origin, evaluator, *, mode="emergent", reference=""):
    return make_evidence(
        "music", dimension, .8, artifact_id="song-1", origin=origin,
        method="blind review", evaluator_type=evaluator, material_mode=mode,
        reference_set_id=reference,
    )


def test_creativity_is_a_profile_not_one_score_or_intent_test():
    report = build_profile([])
    assert report["policy"]["single_creativity_score"] is False
    assert report["policy"]["intent_required"] is False
    assert report["policy"]["novelty_alone_is_creativity"] is False
    assert "score" not in report


def test_originality_requires_named_comparison_set_and_two_signal_types():
    with pytest.raises(ValueError, match="reference_set_id"):
        evidence("originality", "reviewer-a", "human")
    rows = [
        evidence("originality", "reviewer-a", "human", reference="music-reference-v1"),
        evidence("originality", "similarity-tool", "objective", reference="music-reference-v1"),
    ]
    result = build_profile(rows)["domains"]["music"]["dimensions"]["originality"]
    assert result["evidenced"] is True
    assert result["score"] == .8


def test_novelty_metric_cannot_establish_purposeful_surprise():
    rows = [
        evidence("purposeful_surprise", "metric-a", "objective"),
        evidence("purposeful_surprise", "metric-b", "objective"),
    ]
    result = build_profile(rows)["domains"]["music"]["dimensions"]["purposeful_surprise"]
    assert result["evidenced"] is False
    assert "needs human evidence" in result["blockers"]


def test_emergent_material_is_accepted_without_invented_intention():
    row = evidence("coherence", "reviewer-a", "human", mode="emergent")
    assert row["material_mode"] == "emergent"
    assert "intent" not in row
    assert "rationale" not in row


def test_source_references_are_bounded():
    with pytest.raises(ValueError, match="bounded"):
        make_evidence(
            "image", "transformation", .5, artifact_id="image-1", origin="tool-a",
            method="comparison", evaluator_type="objective",
            source_references=[f"source-{number}" for number in range(17)],
        )
