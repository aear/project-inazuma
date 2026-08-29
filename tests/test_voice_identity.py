from array import array
import json

from voice_identity import (
    conversation_urge_influence,
    observe_discord_voice,
    pcm_s16le_signature,
    recognition_candidates,
)


def _pcm(values):
    return array("h", values).tobytes()


def test_discord_voice_identity_benchmark_v1_unattributed_mix_vs_v2_profiles(tmp_path):
    """V2 retains authoritative per-speaker labels and candidate-only matches."""
    path = tmp_path / "profiles.json"
    sakura = pcm_s16le_signature(_pcm([100, -100, 300, -300] * 64))
    friend = pcm_s16le_signature(_pcm([100, 200, 300, 400] * 64))
    observe_discord_voice(
        path, discord_user_id="11", display_name="Sakura", signature=sakura,
        observed_at="2026-08-29T12:00:00+00:00", guild_id="1", channel_id="2",
    )
    observe_discord_voice(
        path, discord_user_id="22", display_name="Friend", signature=friend,
        observed_at="2026-08-29T12:00:01+00:00", guild_id="1", channel_id="2",
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    candidates = recognition_candidates(path, sakura)
    assert set(payload["profiles"]) == {"11", "22"}
    assert payload["profiles"]["11"]["identity_basis"] == "discord_user_id"
    assert candidates[0]["discord_user_id"] == "11"
    assert candidates[0]["status"] == "candidate_only"


def test_conversation_urge_benchmark_v1_fixed_reduction_vs_v2_signed_evidence():
    """V2 can invite speech, relieve its pressure, or leave it neutral."""
    base = {
        "timestamp": 1000.0,
        "voiced_seconds": 8.0,
        "speakers": [{"discord_user_id": "11", "is_bot": False, "novelty": 1.0}],
    }
    inviting = conversation_urge_influence(base, now=1000.0, curiosity=0.9, isolation=0.1)
    familiar = dict(base, speakers=[{"discord_user_id": "11", "is_bot": False, "novelty": 0.0}])
    relieving = conversation_urge_influence(familiar, now=1000.0, curiosity=0.0, isolation=1.0)
    stale = conversation_urge_influence(base, now=1201.0, curiosity=1.0, isolation=0.0)

    assert inviting["adjustment"] > 0.0
    assert inviting["reason"] == "reply_invitation"
    assert relieving["adjustment"] < 0.0
    assert relieving["reason"] == "social_presence_relief"
    assert stale["adjustment"] == 0.0
    assert stale["reason"] == "stale_conversation"


def test_group_voice_benchmark_v1_single_mix_vs_v2_uncertain_turn_pressure():
    group = {
        "timestamp": 1000.0,
        "voiced_seconds": 8.0,
        "concurrency_load": 0.9,
        "speakers": [
            {"discord_user_id": "11", "is_bot": False, "novelty": 0.2},
            {"discord_user_id": "22", "is_bot": False, "novelty": 0.2},
            {"discord_user_id": "33", "is_bot": False, "novelty": 0.2},
        ],
    }
    cautious = conversation_urge_influence(group, now=1000.0, curiosity=0.1, isolation=0.0)
    curious = conversation_urge_influence(group, now=1000.0, curiosity=1.0, isolation=0.0)

    assert cautious["group_size"] == 3
    assert cautious["turn_uncertainty"] > 0.0
    assert curious["reply_invitation"] > cautious["reply_invitation"]
    assert curious["adjustment"] > cautious["adjustment"]
