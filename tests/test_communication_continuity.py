import json

from lm_studio_adapter import LMStudioAdapter
from module_benchmarks import benchmark_module


def _adapter(tmp_path):
    adapter = object.__new__(LMStudioAdapter)
    adapter.child = "TestChild"
    adapter._base_path = tmp_path
    return adapter


def test_unfinished_communication_needs_continuity_and_topic_cues(tmp_path):
    archive = tmp_path / "TestChild" / "memory" / "typed_outbox_archive.jsonl"
    archive.parent.mkdir(parents=True)
    archive.write_text(json.dumps({
        "id": "old-1", "text": "I wanted to improve garden recall",
        "communication_state": "unfinished", "delivery_state": "not_delivered",
    }) + "\n", encoding="utf-8")
    adapter = _adapter(tmp_path)

    assert adapter._recall_unfinished_communication(
        "Do you have suggestions?", max_items=3, max_chars=800
    ) == []
    assert adapter._recall_unfinished_communication(
        "Continue your unfinished thoughts about weather", max_items=3, max_chars=800
    ) == []
    recalled = adapter._recall_unfinished_communication(
        "Continue your unfinished thoughts about garden recall", max_items=3, max_chars=800
    )
    assert recalled[0]["memory_type"] == "prospective_communication"
    assert recalled[0]["communication_state"] == "unfinished"


def test_stale_outbox_episode_is_not_ordinary_word_grounding(monkeypatch, tmp_path):
    adapter = _adapter(tmp_path)
    monkeypatch.setattr("lm_studio_adapter.describe_word_grounding", lambda *args, **kwargs: [
        {"event_id": "stale", "narrative": "old unsent text", "situation_tags": ["typed_outbox", "archive", "stale_buffer"]},
        {"event_id": "real", "narrative": "a lived example", "situation_tags": ["conversation"]},
    ])
    assert adapter._summarise_grounding("recall")["event_id"] == "real"


def test_communication_continuity_benchmark_compares_versions():
    v1, v2 = benchmark_module("communication_continuity")
    assert v2.accuracy > v1.accuracy
    assert set(v2.component_scores) == {"separation", "persistence", "routing"}
