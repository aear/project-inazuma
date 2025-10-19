import json
import sys
from pathlib import Path
from typing import List

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import language_processing
from experience_logger import ExperienceLogger
from live_experience_bridge import LiveExperienceBridge
from memory_graph import build_experience_graph
from language_processing import (
    associate_symbol_with_word,
    describe_word_grounding,
    ensure_word_grounded,
    is_word_grounded,
    suggest_words_for_context,
)


@pytest.fixture()
def temp_child(tmp_path):
    base = tmp_path / "AI_Children"
    child = "TestChild"
    yield child, base


def test_experience_logging_and_grounding(temp_child, monkeypatch):
    child, base = temp_child
    logger = ExperienceLogger(child=child, base_path=base)
    bridge = LiveExperienceBridge(child=child, base_path=base, logger=logger)

    episode_id = logger.start_episode(situation_tags=["kitchen"], intent="tidy-up")
    event_id = logger.log_event(
        situation_tags=["kitchen", "cup"],
        perceived_entities=[{"id": "cup-1", "label": "cup"}],
        actions=[{"verb": "pick", "object": "cup"}],
        outcome={"success": True},
        internal_state={"curiosity": 0.7},
        narrative="Picked up the cup from the table.",
    )
    logger.attach_word_usage(
        event_id,
        speaker="parent",
        utterance="This is a cup",
        words=["cup"],
        entity_links=[{"entity_id": "cup-1"}],
    )
    logger.add_event_to_episode(event_id)

    bridge._describe_frame = lambda frame: (["status", "online"], [0.1, 0.2, 0.3])
    frame = [
        [[255, 255, 255], [120, 120, 120]],
        [[0, 0, 0], [64, 64, 64]],
    ]
    screen_event = bridge.log_screen_snapshot(
        frame,
        tags=["kitchen", "screen"],
        narrative="Reviewed instructions on the screen.",
        metadata={"window": "recipe"},
    )

    audio_features = {"volume_db": -12.5, "pitch_hz": 210.0}
    conversation_event = bridge.log_conversation_turn(
        "The cup is full now",
        speaker="parent",
        tags=["kitchen", "audio"],
        entity_links=[{"entity_id": "cup-1"}],
        audio_features=audio_features,
    )

    motor_event = bridge.log_motor_feedback(
        "grip",
        success=True,
        sensor_readings={"pressure": 0.82},
        tags=["kitchen", "motor"],
        vocabulary=["grip"],
        narrative="Applied pressure to grip the cup handle.",
        entities=[{"type": "object", "name": "cup"}],
    )

    summary = logger.finish_and_narrate_episode(
        result={"status": "complete"},
        feedback_hooks=[{"type": "operator_review"}],
    )
    assert summary is not None
    assert summary["episode_id"] == episode_id
    assert summary["narrative"]

    episode_path = (
        base
        / child
        / "memory"
        / "experiences"
        / "episodes"
        / f"{episode_id}.json"
    )
    with open(episode_path, "r", encoding="utf-8") as fh:
        episode_payload = json.load(fh)
    assert len(episode_payload["feedback_hooks"]) == len(episode_payload["events"])
    assert any(hook["event_id"] == screen_event for hook in episode_payload["feedback_hooks"])

    events_dir = base / child / "memory" / "experiences" / "events"
    with open(events_dir / f"{conversation_event}.json", "r", encoding="utf-8") as fh:
        conversation_payload = json.load(fh)
    assert conversation_payload["internal_state"]["audio_features"] == audio_features

    with open(events_dir / f"{motor_event}.json", "r", encoding="utf-8") as fh:
        motor_payload = json.load(fh)
    assert motor_payload["internal_state"]["motor_feedback"] == {"pressure": 0.82}

    screen_meta = (
        base
        / child
        / "memory"
        / "experiences"
        / "live_media"
        / f"{screen_event}_screen.json"
    )
    assert screen_meta.exists()

    triggered: List[str] = []

    def capture_question(prompt: str) -> None:
        triggered.append(prompt)

    monkeypatch.setattr(language_processing, "seed_self_question", capture_question)

    associate_symbol_with_word(
        child,
        "sym_cup",
        "cup",
        grounding={
            "event_id": conversation_event,
            "speaker": "coach",
            "utterance": "cup",
            "entity_links": [{"entity_id": "cup-1"}],
        },
        base_path=base,
    )
    assert not triggered

    associate_symbol_with_word(
        child,
        "sym_unknown",
        "mystery",
        base_path=base,
    )
    assert triggered and "mystery" in triggered[-1]

    graph = build_experience_graph(child, base_path=base)
    graph_path = base / child / "memory" / "experiences" / "experience_graph.json"
    assert graph_path.exists()

    with open(graph_path, "r", encoding="utf-8") as fh:
        saved = json.load(fh)
    assert len(saved["events"]) >= 4
    assert saved["words_index"].get("cup")
    assert saved["words_index"].get("grip")

    groundings = describe_word_grounding(child, "cup", base_path=base)
    assert groundings
    assert any(entry["event_id"] == conversation_event for entry in groundings)

    assert is_word_grounded(child, "cup", base_path=base)
    assert is_word_grounded(child, "grip", base_path=base)
    assert not is_word_grounded(child, "unknown", base_path=base)

    suggestions = suggest_words_for_context(
        child,
        situation_tags=["kitchen"],
        entity_labels=["cup"],
        base_path=base,
    )
    assert suggestions == ["cup", "grip"]

    triggered.clear()
    assert ensure_word_grounded(child, "cup", base_path=base)
    assert not triggered
    assert not ensure_word_grounded(child, "phantom", base_path=base)
    assert triggered and "phantom" in triggered[-1]
