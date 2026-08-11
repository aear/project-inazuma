import json
from types import SimpleNamespace

from conversation_scene import (
    ConversationSceneBuffer,
    scene_with_memory_consideration,
)
from lm_studio_adapter import LMStudioAdapter


def _message(
    message_id: str,
    text: str,
    *,
    direction: str = "inbound",
    channel_id: str = "room-a",
    speaker: str = "Sakura",
    is_self: bool = False,
    metadata=None,
):
    return SimpleNamespace(
        id=message_id,
        backend="discord",
        direction=direction,
        text=text,
        created_at="2026-08-09T10:00:00+00:00",
        reply_to_id=None,
        metadata=metadata or {},
        sender=SimpleNamespace(
            display_name=speaker,
            internal_id=speaker.lower(),
            backend_id=speaker.lower(),
            is_self=is_self,
        ),
        channel=SimpleNamespace(
            internal_id=channel_id,
            backend_id=channel_id,
            name=channel_id,
        ),
    )


def test_scene_is_channel_local_and_bounded() -> None:
    scenes = ConversationSceneBuffer(max_turns=3, max_total_chars=240)

    for index in range(5):
        snapshot = scenes.observe(_message(str(index), f"turn {index} about the garden"))

    assert snapshot["turn_count"] == 3
    assert [turn["source_id"] for turn in snapshot["turns"]] == ["2", "3", "4"]
    assert snapshot["signals"]["has_prior_context"] is True
    assert "garden" in snapshot["topic_terms"]

    isolated = scenes.observe(_message("other", "separate", channel_id="room-b"))
    assert isolated["turn_count"] == 1
    assert isolated["scene_id"] != snapshot["scene_id"]


def test_scene_marks_questions_and_continuity_without_forcing_a_reply() -> None:
    scenes = ConversationSceneBuffer()
    scenes.observe(_message("1", "Do you remember the garden?"))
    scenes.observe(
        _message("2", "I remember the garden path", direction="outbound", speaker="Ina")
    )
    snapshot = scenes.observe(_message("3", "Was the garden peaceful?"))

    assert snapshot["signals"]["reply_expected"] is True
    assert snapshot["signals"]["current_is_question"] is True
    assert snapshot["signals"]["continuity_terms"] == ["garden"]


def test_scene_keeps_external_and_self_speakers_distinct() -> None:
    scenes = ConversationSceneBuffer()
    scenes.observe(_message("1", "Hello", speaker="Rowan", is_self=False))
    snapshot = scenes.observe(
        _message("2", "Hello Rowan", direction="outbound", speaker="Ina", is_self=True)
    )
    assert [(turn["speaker"], turn["is_self"]) for turn in snapshot["turns"]] == [
        ("Rowan", False), ("Ina", True)
    ]


def test_final_consideration_preserves_bounded_rejection_description() -> None:
    scene = {"scene_id": "scene_test", "signals": {}}
    consideration = {
        "accepted": [],
        "rejected": [
            {
                "event_id": "event_old",
                "cue": "garden",
                "summary": "unrelated lexical match",
                "consideration": {
                    "decision": "rejected",
                    "reason": "insufficient_scene_support",
                    "description": "I recalled an old garden memory but decided it was not relevant.",
                },
            }
        ],
    }

    enriched = scene_with_memory_consideration(scene, consideration, max_chars=100)

    assert enriched["memory_candidates_considered"] == 1
    assert enriched["memory_references"] == []
    assert enriched["memory_rejections"][0]["consideration"]["decision"] == "rejected"
    assert "not relevant" in enriched["memory_rejections"][0]["consideration"]["description"]


def test_indexed_retrieval_then_final_consideration_can_accept_and_reject(tmp_path) -> None:
    graph_path = tmp_path / "Ina" / "memory" / "experiences" / "experience_graph.json"
    graph_path.parent.mkdir(parents=True)
    graph_path.write_text(
        json.dumps(
            {
                "events": [
                    {
                        "id": "garden_event",
                        "narrative": "We walked through the garden after the rain.",
                        "situation_tags": ["outside", "garden"],
                    },
                    {
                        "id": "bank_event",
                        "narrative": "A financial account was opened.",
                        "situation_tags": ["finance"],
                    },
                ],
                "words_index": {
                    "garden": ["garden_event"],
                    "bank": ["bank_event"],
                },
            }
        ),
        encoding="utf-8",
    )
    adapter = object.__new__(LMStudioAdapter)
    adapter.child = "Ina"
    adapter._base_path = tmp_path
    adapter._relevance_cache_signature = None
    adapter._relevance_cache = None

    candidates = adapter.recall_relevant("garden bank", max_items=2)
    considered = adapter.consider_recalled_memories(
        "garden bank",
        candidates,
        scene={"topic_terms": ["rain"], "signals": {"continuity_terms": ["path"]}},
    )

    assert [item["event_id"] for item in considered["accepted"]] == ["garden_event"]
    assert [item["event_id"] for item in considered["rejected"]] == ["bank_event"]
    assert considered["rejected"][0]["consideration"]["description"]


def test_retrieval_declines_when_graph_exceeds_bound(tmp_path) -> None:
    graph_path = tmp_path / "Ina" / "memory" / "experiences" / "experience_graph.json"
    graph_path.parent.mkdir(parents=True)
    graph_path.write_text('{"events": [], "words_index": {}, "padding": "xxxxxxxx"}', encoding="utf-8")
    adapter = object.__new__(LMStudioAdapter)
    adapter.child = "Ina"
    adapter._base_path = tmp_path
    adapter._relevance_cache_signature = None
    adapter._relevance_cache = None

    assert adapter.recall_relevant("anything", max_graph_bytes=16) == []
