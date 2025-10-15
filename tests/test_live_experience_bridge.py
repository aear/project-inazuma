import json
import sys
from types import ModuleType

import pytest

np = pytest.importorskip("numpy")

from experience_logger import ExperienceLogger
from live_experience_bridge import LiveExperienceBridge


def test_log_screen_snapshot_creates_event(tmp_path, monkeypatch):
    child = "TestChild"
    bridge = LiveExperienceBridge(child=child, base_path=tmp_path)

    dummy_tokens = ["Hello", "World"]

    fake_module = ModuleType("vision_digest")
    fake_module.run_text_recognition = lambda frame, child_name: dummy_tokens
    monkeypatch.setitem(sys.modules, "vision_digest", fake_module)

    frame = np.full((4, 4, 3), 255, dtype=np.uint8)
    event_id = bridge.log_screen_snapshot(frame, metadata={"window": "browser"})

    event_path = (
        tmp_path
        / child
        / "memory"
        / "experiences"
        / "events"
        / f"{event_id}.json"
    )
    with open(event_path, "r", encoding="utf-8") as fh:
        event = json.load(fh)

    assert event["narrative"].startswith("Observed")
    assert event["perceived_entities"]
    assert event["word_usage"]

    screen_meta = (
        tmp_path
        / child
        / "memory"
        / "experiences"
        / "live_media"
        / f"{event_id}_screen.json"
    )
    assert screen_meta.exists()


def test_log_conversation_turn_attach_existing_event(tmp_path):
    child = "TestChild"
    logger = ExperienceLogger(child=child, base_path=tmp_path)
    bridge = LiveExperienceBridge(child=child, base_path=tmp_path, logger=logger)

    event_id = bridge.log_screen_snapshot(np.zeros((2, 2, 3), dtype=np.uint8))
    bridge.log_conversation_turn(
        "This is a live description", speaker="operator", event_id=event_id
    )

    event_path = (
        tmp_path
        / child
        / "memory"
        / "experiences"
        / "events"
        / f"{event_id}.json"
    )
    with open(event_path, "r", encoding="utf-8") as fh:
        event = json.load(fh)

    assert len(event["word_usage"]) >= 1

    dialogue_meta = (
        tmp_path
        / child
        / "memory"
        / "experiences"
        / "live_media"
        / f"{event_id}_dialogue.json"
    )
    assert dialogue_meta.exists()

