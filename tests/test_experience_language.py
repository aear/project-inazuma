import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experience_logger import ExperienceLogger
from memory_graph import build_experience_graph
from language_processing import (
    describe_word_grounding,
    is_word_grounded,
    suggest_words_for_context,
)


@pytest.fixture()
def temp_child(tmp_path):
    base = tmp_path / "AI_Children"
    child = "TestChild"
    yield child, base


def test_experience_logging_and_grounding(temp_child):
    child, base = temp_child
    logger = ExperienceLogger(child=child, base_path=base)

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
    logger.finish_episode(result={"status": "complete"})
    logger.narrate_episode(episode_id)

    graph = build_experience_graph(child, base_path=base)
    graph_path = base / child / "memory" / "experiences" / "experience_graph.json"
    assert graph_path.exists()

    with open(graph_path, "r", encoding="utf-8") as fh:
        saved = json.load(fh)
    assert saved["events"], "Graph should include at least one event"
    assert saved["words_index"].get("cup"), "Word index should include the grounded word"

    groundings = describe_word_grounding(child, "cup", base_path=base)
    assert groundings, "Groundings should return the event that introduced the word"
    assert groundings[0]["event_id"] == event_id
    assert "cup" in groundings[0]["words"]

    assert is_word_grounded(child, "cup", base_path=base)
    assert not is_word_grounded(child, "unknown", base_path=base)

    suggestions = suggest_words_for_context(
        child,
        situation_tags=["kitchen"],
        entity_labels=["cup"],
        base_path=base,
    )
    assert suggestions == ["cup"]
