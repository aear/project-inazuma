from datetime import datetime, timezone

from music_exploration import artist_identity, build_listening_plan


NOW = datetime(2026, 8, 30, tzinfo=timezone.utc)


def test_music_exploration_benchmark_v1_tied_to_local_files_vs_v2_provider_plan():
    policy = {"enabled": True, "default_provider": "local_mpris"}
    plan = build_listening_plan({"query": "an unfamiliar shoegaze artist"}, policy, now=NOW)

    assert plan == {
        "status": "ready",
        "provider": "local_mpris",
        "query": "an unfamiliar shoegaze artist",
        "requested_at": "2026-08-30T00:00:00+00:00",
        "purpose": "explore_other_people_music",
        "one_bounded_attempt": True,
        "action": "open_spotify_for_selection",
    }


def test_discord_music_bot_requires_an_explicit_channel():
    blocked = build_listening_plan(
        {"provider": "discord_music_bot", "query": "Helldivers soundtrack"},
        {"enabled": True}, now=NOW,
    )
    assert blocked["status"] == "blocked"
    assert blocked["reason"] == "discord_music_channel_not_configured"


def test_artist_identity_does_not_guess_a_spotify_profile():
    assert artist_identity({"artist_identity": {"name": "Inazuma"}}) == {
        "name": "Inazuma",
        "spotify_url": "",
    }
