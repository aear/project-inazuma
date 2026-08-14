from learned_media_lessons import load_output_guidance, record_media_lesson
import model_manager


def _language_lesson(fragment_id, role, *, cadence, text=""):
    return {
        "id": fragment_id,
        "source": f"media/{fragment_id}",
        "modality": "text" if text else "audio",
        "text": text,
        "tags": ["self_read", role],
        "symbols": ["snd_a", "snd_b"],
        "proto_words": ["snd_a_snd_b"],
        "language_learning": {
            "role": role,
            "alignment_keys": ["shared_work"],
            "supports_pronunciation": not bool(text),
            "supports_cadence": cadence,
            "supports_written_alignment": True,
        },
    }


def test_output_guidance_keeps_domains_and_cadence_authority_separate(tmp_path):
    vocal = _language_lesson("vocal", "isolated_vocal_stem", cadence=True)
    essay = _language_lesson("essay", "video_essay", cadence=False)
    script = _language_lesson("script", "spoken_script", cadence=False, text="A written line")
    cover = {
        "id": "cover", "source": "media/cover.png", "tags": ["album_cover"],
        "visual_learning": {
            "role": "album_cover", "alignment_keys": ["shared_work"],
            "study_dimensions": ["composition", "colour"],
        },
    }
    for fragment in (vocal, essay, script, cover):
        assert record_media_lesson("Ina", fragment, base_path=tmp_path)

    daw = load_output_guidance("Ina", "daw", base_path=tmp_path)
    speech = load_output_guidance("Ina", "speech", base_path=tmp_path)
    text = load_output_guidance("Ina", "text", base_path=tmp_path)
    drawing = load_output_guidance("Ina", "drawing", base_path=tmp_path)
    assert any(row["role"] == "isolated_vocal_stem" for row in daw["lessons"])
    assert all(row["role"] != "video_essay" for row in daw["lessons"])
    assert {row["supports_cadence"] for row in speech["lessons"]} == {True, False}
    assert text["lessons"][0]["role"] == "spoken_script"
    assert drawing["lessons"][0]["study_dimensions"] == ["composition", "colour"]


def test_drawing_seed_uses_cover_as_composition_reference_without_copying(monkeypatch):
    updates = []
    monkeypatch.setattr(model_manager, "get_inastate", lambda key: [] if key == "paint_command_queue" else None)
    monkeypatch.setattr(model_manager, "update_inastate", lambda key, value: updates.append((key, value)))
    monkeypatch.setattr(
        model_manager,
        "load_output_guidance",
        lambda child, consumer: {
            "lessons": [{
                "role": "album_cover", "source": "Song/cover.png",
                "alignment_keys": ["song"],
                "study_dimensions": ["composition", "colour"],
            }]
        },
    )
    model_manager._queue_autonomous_paint_seed({"curiosity": 0.2, "joy": 0.1, "intensity": 0.1})
    commands = dict(updates)["paint_command_queue"]
    assert commands[0]["pattern"] == "burst"
    reference = commands[0]["motivation"]["learned_visual_reference"]
    assert reference["source"] == "Song/cover.png"
    assert reference["copying_required"] is False
