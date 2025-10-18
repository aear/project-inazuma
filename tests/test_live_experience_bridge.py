import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experience_logger import ExperienceLogger
from live_experience_bridge import LiveExperienceBridge


@pytest.fixture()
def bridge(tmp_path):
    base = tmp_path / "AI_Children"
    child = "TestChild"
    logger = ExperienceLogger(child=child, base_path=base)
    bridge = LiveExperienceBridge(child=child, base_path=base, logger=logger)
    return bridge, child, base


def test_log_conversation_turn_creates_event_with_word_usage(bridge):
    bridge_instance, child, base = bridge

    event_id = bridge_instance.log_conversation_turn("Hello Ina", speaker="parent")

    # Event file should exist and include attached word usage
    events_dir = base / child / "memory" / "experiences" / "events"
    event_file = events_dir / f"{event_id}.json"
    assert event_file.exists()
    event_data = json.loads(event_file.read_text(encoding="utf-8"))
    assert event_data["word_usage"], "word usage annotations should be recorded"
    assert event_data["word_usage"][0]["words"] == ["hello", "ina"]

    # Logger should allow us to attach another turn to the same event
    bridge_instance.log_conversation_turn(
        "Nice to meet you", speaker="parent", event_id=event_id
    )
    updated_data = json.loads(event_file.read_text(encoding="utf-8"))
    utterances = [entry["utterance"] for entry in updated_data["word_usage"]]
    assert "Nice to meet you" in utterances

    # Ensure the dialogue metadata file aggregates turns
    meta_path = (
        base
        / child
        / "memory"
        / "experiences"
        / "live_media"
        / f"{event_id}_dialogue.json"
    )
    assert meta_path.exists()
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    assert len(payload["turns"]) == 2
    assert payload["turns"][0]["utterance"] == "Hello Ina"
    assert payload["turns"][1]["utterance"] == "Nice to meet you"
