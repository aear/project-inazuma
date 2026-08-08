import os
import sys
import types

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import language_processing as lp



def test_text_length_profile_accepts_varied_words_and_sentences():
    profile = lp.text_length_profile("I. extraordinarily vary words now! Two more?")
    assert profile["sentence_word_counts"] == [1, 4, 2]
    assert profile["word_lengths"][1] == len("extraordinarily")


def test_ina_expression_length_is_bounded_and_drive_sensitive():
    quiet = lp.adaptive_symbol_limit(
        "same input", 6, child="Ina", available_symbols=6,
        context={"expression_drive": 0.0, "tags": ["discord"]},
    )
    expressive = lp.adaptive_symbol_limit(
        "same input", 6, child="Ina", available_symbols=6,
        context={"expression_drive": 1.0, "tags": ["discord"]},
    )
    assert 1 <= quiet <= expressive <= 6
    assert lp.adaptive_symbol_limit("tiny", 6, available_symbols=2) == 2


def test_expression_symbol_choice_has_no_legacy_three_or_six_symbol_cut(monkeypatch):
    monkeypatch.setattr(lp, "_stable_symbol_seed", lambda _value: 23)
    symbols = [f"sym_{index}" for index in range(24)]

    chosen = lp.choose_expression_symbols(
        symbols,
        "Ina has more to say.",
        24,
        child="Ina",
        context={"expression_drive": 1.0},
    )

    assert chosen == symbols


def test_mirroring_score_drops_with_boredom_and_repeat_streak():
    open_state = {
        "emotion_snapshot": {
            "values": {"curiosity": 0.9, "novelty": 0.8, "familiarity": -0.7}
        },
        "emotion_boredom": 0.0,
        "emotion_playfulness_level": 0.4,
    }
    tired_state = {
        **open_state,
        "emotion_boredom": 1.0,
        "last_text_expression_decision": {
            "strategy": "mirror",
            "mirror_streak": 2,
        },
    }

    open_score = lp.score_text_mirroring(
        open_state,
        mapping_coverage=1.0,
        expression_drive=0.8,
    )
    tired_score = lp.score_text_mirroring(
        tired_state,
        mapping_coverage=1.0,
        expression_drive=0.8,
    )

    assert open_score["score"] > tired_score["score"]
    assert tired_score["previous_mirror_streak"] == 2


def test_build_dual_symbolic_message_combines_native_and_guess(monkeypatch):
    monkeypatch.setattr(
        lp,
        "load_symbol_to_token",
        lambda child, base_path=None: {
            "sym_hello": {"word": "hello"},
            "sym_calm": {"word": "calm"},
        },
    )

    payload = lp.build_dual_symbolic_message(["sym_hello", "sym_calm"], child="TestChild")

    assert payload is not None
    assert payload["native_text"] == "sym_hello sym_calm"
    assert payload["gloss_text"] == "hello calm"
    assert payload["text"] == "Native: sym_hello sym_calm\nHuman guess: hello calm"



def test_build_dual_symbolic_message_prefers_supplied_human_text(monkeypatch):
    monkeypatch.setattr(
        lp,
        "load_symbol_to_token",
        lambda child, base_path=None: {"sym_wave": {"word": "wave"}},
    )

    payload = lp.build_dual_symbolic_message(
        ["sym_wave"],
        child="TestChild",
        human_text="hello there",
    )

    assert payload is not None
    assert payload["native_text"] == "sym_wave"
    assert payload["gloss_text"] == "hello there"
    assert payload["text"] == "Native: sym_wave\nHuman guess: hello there"


def test_symbolic_message_language_choice_keeps_native_translation():
    payload = {
        "text": "Native: glyph_wave\nHuman guess: hello there",
        "native_text": "glyph_wave",
        "gloss_text": "hello there",
    }

    assert lp.select_symbolic_message_text(payload, "english") == ("hello there", "english")
    assert lp.select_symbolic_message_text(payload, "native") == (payload["text"], "native")
    assert lp.select_symbolic_message_text(payload, "mixed") == (payload["text"], "mixed")
    assert lp.select_symbolic_message_text(payload, "auto") == (payload["text"], "mixed")


def test_english_choice_does_not_hide_an_untranslated_native_message():
    payload = {
        "text": "Native: glyph_private\nHuman guess: glyph_private",
        "native_text": "glyph_private",
        "gloss_text": "glyph_private",
    }

    assert lp.select_symbolic_message_text(payload, {"mode": "english"}) == (
        payload["text"],
        "mixed",
    )


def test_build_dual_symbolic_message_uses_contextual_text_vocab_links(tmp_path, monkeypatch):
    monkeypatch.setattr(lp, "load_symbol_to_token", lambda child, base_path=None: {})
    memory_root = tmp_path / "TestChild" / "memory"
    memory_root.mkdir(parents=True)
    (memory_root / "text_vocab_links.json").write_text(
        """{
  "links": [
    {"word": "zero", "symbol": "sym_ambiguous", "count": 999, "similarity": 1.0},
    {"word": "heart", "symbol": "sym_ambiguous", "count": 1, "similarity": 1.0},
    {"word": "ina", "symbol": "sym_ina", "count": 10, "similarity": 1.0}
  ]
}
""",
        encoding="utf-8",
    )

    payload = lp.build_dual_symbolic_message(
        ["sym_ambiguous", "sym_ina"],
        child="TestChild",
        base_path=tmp_path,
        context={"tokens": ["heart", "ina"], "tags": ["discord"]},
        fallback_to_symbol_to_token=False,
    )

    assert payload is not None
    assert payload["native_text"] == "sym_ambiguous sym_ina"
    assert payload["gloss_text"] == "heart ina"
    assert payload["gloss_sources"] == {
        "sym_ambiguous": "text_vocab_links",
        "sym_ina": "text_vocab_links",
    }



def test_build_dual_symbolic_message_resolves_raw_symbol_words_from_text_vocab_links(tmp_path, monkeypatch):
    monkeypatch.setattr(lp, "load_symbol_to_token", lambda child, base_path=None: {})
    memory_root = tmp_path / "TestChild" / "memory"
    memory_root.mkdir(parents=True)
    (memory_root / "text_vocab_links.json").write_text(
        "{\"links\": [{\"word\": \"heart\", \"symbol\": \"sym_heart\", \"symbol_word\": \"glyph_heart\", \"count\": 4}]}",
        encoding="utf-8",
    )

    payload = lp.build_dual_symbolic_message(
        ["glyph_heart", "glyph_unmapped"],
        child="TestChild",
        base_path=tmp_path,
        context={"tokens": ["heart"], "tags": ["discord"]},
        fallback_to_symbol_to_token=False,
        native_style="glyphs",
    )

    assert payload is not None
    assert payload["native_text"] == "glyph_heart glyph_unmapped"
    assert payload["gloss_text"] == "heart glyph_unmapped"
    assert payload["text"] == "Native: glyph_heart glyph_unmapped\nHuman guess: heart glyph_unmapped"
    assert payload["unresolved_symbols"] == ["glyph_unmapped"]



def test_build_dual_symbolic_message_preserves_native_vocab_as_unresolved_guess(monkeypatch):
    def fail_load_symbol_to_token(child, base_path=None):
        raise AssertionError("symbol_to_token fallback should stay lazy when a vocab is supplied")

    monkeypatch.setattr(lp, "load_text_vocab_links", lambda child, base_path=None: {})
    monkeypatch.setattr(lp, "load_symbol_to_token", fail_load_symbol_to_token)

    payload = lp.build_dual_symbolic_message(
        ["sym_private"],
        child="TestChild",
        fallback_to_symbol_to_token=False,
        native_style="glyphs",
        symbol_to_token_vocab={"sym_private": {"word": "glyph_private"}},
    )

    assert payload is not None
    assert payload["native_text"] == "glyph_private"
    assert payload["gloss_text"] == "glyph_private"
    assert payload["text"] == "Native: glyph_private\nHuman guess: glyph_private"
    assert payload["unresolved_symbols"] == ["sym_private"]


def test_build_dual_symbolic_message_can_skip_symbol_to_token_fallback(tmp_path, monkeypatch):
    def fail_load_symbol_to_token(child, base_path=None):
        raise AssertionError("symbol_to_token fallback should stay lazy")

    monkeypatch.setattr(lp, "load_symbol_to_token", fail_load_symbol_to_token)
    memory_root = tmp_path / "TestChild" / "memory"
    memory_root.mkdir(parents=True)
    (memory_root / "text_vocab_links.json").write_text(
        '{"links": [{"word": "ina", "symbol": "sym_ina", "count": 1}]}',
        encoding="utf-8",
    )

    payload = lp.build_dual_symbolic_message(
        ["sym_unknown", "sym_ina"],
        child="TestChild",
        base_path=tmp_path,
        context={"tokens": ["ina"]},
        fallback_to_symbol_to_token=False,
    )

    assert payload is not None
    assert payload["gloss_text"] == "sym_unknown ina"
    assert payload["unresolved_symbols"] == ["sym_unknown"]


def test_build_dual_symbolic_message_can_use_symbol_word_as_native(tmp_path, monkeypatch):
    monkeypatch.setattr(lp, "load_symbol_to_token", lambda child, base_path=None: {})
    memory_root = tmp_path / "TestChild" / "memory"
    memory_root.mkdir(parents=True)
    (memory_root / "text_vocab_links.json").write_text(
        '{"links": [{"word": "ina", "symbol": "sym_ina", "symbol_word": "glyph_ina", "count": 1}]}',
        encoding="utf-8",
    )

    payload = lp.build_dual_symbolic_message(
        ["sym_ina"],
        child="TestChild",
        base_path=tmp_path,
        context={"tokens": ["ina"]},
        fallback_to_symbol_to_token=False,
        native_style="glyphs",
    )

    assert payload is not None
    assert payload["native_text"] == "glyph_ina"
    assert payload["gloss_text"] == "ina"
    assert payload["native_sources"] == {"sym_ina": "text_vocab_links"}


def test_generate_symbolic_reply_uses_text_vocab_links_without_symbol_to_token(tmp_path, monkeypatch):
    def fail_load_symbol_to_token(child, base_path=None):
        raise AssertionError("symbol_to_token fallback should stay lazy")

    monkeypatch.setattr(lp, "load_symbol_to_token", fail_load_symbol_to_token)
    monkeypatch.setattr(lp, "speak_symbolically", lambda *args, **kwargs: None)
    monkeypatch.setattr(lp, "_build_reply_transformer_insights", lambda *args, **kwargs: None)
    memory_root = tmp_path / "TestChild" / "memory"
    memory_root.mkdir(parents=True)
    (memory_root / "text_vocab_links.json").write_text(
        '{"links": [{"word": "ina", "symbol": "sym_ina", "symbol_word": "glyph_ina", "count": 1}, {"word": "3", "symbol": "sym_heart", "symbol_word": "glyph_heart", "count": 1}]}',
        encoding="utf-8",
    )

    payload = lp.generate_symbolic_reply_from_text(
        "Ina <3",
        child="TestChild",
        base_path=tmp_path,
    )

    assert payload is not None
    assert payload["symbols"] == ["sym_ina", "sym_heart"]
    assert payload["native_text"] == "glyph_ina glyph_heart"
    assert payload["gloss_text"] == "ina 3"


def test_generate_symbolic_reply_forwards_file_only_render_options(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        lp,
        "load_text_vocab_links",
        lambda child, base_path=None: {"links": [{"word": "ina", "symbol": "sym_ina"}]},
    )
    monkeypatch.setattr(
        lp,
        "_build_text_vocab_word_symbol_index",
        lambda _payload: {"ina": "sym_ina"},
    )
    monkeypatch.setattr(
        lp,
        "speak_symbolically",
        lambda symbols, **kwargs: calls.append((symbols, kwargs)),
    )
    monkeypatch.setattr(lp, "build_dual_symbolic_message", lambda *args, **kwargs: None)
    monkeypatch.setattr(lp, "_build_reply_transformer_insights", lambda *args, **kwargs: None)
    destination = tmp_path / "local_symbols.wav"

    payload = lp.generate_symbolic_reply_from_text(
        "Ina",
        child="TestChild",
        base_path=tmp_path,
        playback=False,
        record_path=destination,
        record_format="wav",
    )

    assert payload is not None
    assert calls == [
        (
            ["sym_ina"],
            {
                "child": "TestChild",
                "playback": False,
                "record_path": destination,
                "record_format": "wav",
            },
        )
    ]


def test_speak_symbolically_logs_heard_voice_only_after_playback_starts(tmp_path, monkeypatch):
    status_lines = []
    playback_attempts = []
    logger_instances = []
    playback_state = {"fail": False}

    sounddevice = types.ModuleType("sounddevice")

    def fake_play(audio, *, samplerate):
        playback_attempts.append((len(audio), samplerate))
        if playback_state["fail"]:
            raise RuntimeError("no output device")

    sounddevice.play = fake_play
    monkeypatch.setitem(sys.modules, "sounddevice", sounddevice)
    monkeypatch.setattr(
        lp,
        "load_config",
        lambda: {
            "allow_polyphonic_voice": False,
            "voice_sample_rate": 8_000,
            "feedback_heard_voice": True,
        },
    )
    monkeypatch.setattr(
        lp,
        "load_sound_symbol_map",
        lambda _child: {"sym_test": {"sound_features": {"pitch_mean": 440.0}}},
    )
    monkeypatch.setattr(
        lp,
        "_render_symbolic_synthesis_plan",
        lambda _plan: np.linspace(-0.2, 0.2, 64, dtype=np.float32),
    )
    monkeypatch.setattr(sys.modules["gui_hook"], "log_to_statusbox", status_lines.append)

    class FakeExperienceLogger:
        def __init__(self, *, child):
            self.child = child
            self.events = []
            self.word_usage = []
            logger_instances.append(self)

        def log_event(self, **payload):
            self.events.append(payload)
            return "voice-event"

        def attach_word_usage(self, event_id, **payload):
            self.word_usage.append((event_id, payload))

    monkeypatch.setattr(lp, "ExperienceLogger", FakeExperienceLogger)

    file_only = tmp_path / "file_only.wav"
    lp.speak_symbolically("sym_test", child="TestChild", playback=False, record_path=file_only)

    assert file_only.read_bytes().startswith(b"RIFF")
    assert playback_attempts == []
    assert logger_instances == []

    playback_state["fail"] = True
    unavailable_device = tmp_path / "unavailable_device.wav"
    lp.speak_symbolically(
        "sym_test",
        child="TestChild",
        playback=True,
        record_path=unavailable_device,
    )

    assert unavailable_device.read_bytes().startswith(b"RIFF")
    assert len(playback_attempts) == 1
    assert logger_instances == []
    assert any("failed to start" in line for line in status_lines)

    playback_state["fail"] = False
    lp.speak_symbolically("sym_test", child="TestChild", playback=True)

    assert len(playback_attempts) == 2
    assert len(logger_instances) == 1
    assert logger_instances[0].events[0]["narrative"] == "Ina listened to her own synthesized voice."
    assert logger_instances[0].word_usage[0][0] == "voice-event"


def test_synthesize_clamps_finite_sample_rates_to_safe_bounds():
    fingerprint = {"pitch_mean": 440.0}
    low_rate = lp.synthesize_from_fingerprint(fingerprint, duration_ms=1, sr=1)
    high_rate = lp.synthesize_from_fingerprint(
        fingerprint,
        duration_ms=1,
        sr=lp.SYMBOLIC_VOICE_MAX_SAMPLE_RATE * 10,
    )

    assert low_rate.shape == (lp.SYMBOLIC_VOICE_MIN_SAMPLE_RATE // 1000,)
    assert high_rate.shape == (lp.SYMBOLIC_VOICE_MAX_SAMPLE_RATE // 1000,)


@pytest.mark.parametrize("sample_rate", [float("nan"), float("inf"), float("-inf"), True])
def test_synthesize_rejects_non_finite_or_boolean_sample_rates(sample_rate):
    with pytest.raises(ValueError, match="sample rate"):
        lp.synthesize_from_fingerprint({"pitch_mean": 440.0}, sr=sample_rate)


def test_frequency_replication_inputs_are_bounded():
    explicit = lp._resolve_frequency_layers(
        {"frequency_layers": [{"freq": 220.0 + idx} for idx in range(100)]},
        440.0,
        {},
        "sym_explicit",
    )
    ratios = lp._resolve_frequency_layers(
        {},
        440.0,
        {
            "enabled": True,
            "ratios": [1.0 + idx / 100.0 for idx in range(100)],
            "replicas_per_layer": 100,
            "detune_cents": 0,
        },
        "sym_ratios",
    )
    replicas = lp._resolve_frequency_layers(
        {},
        440.0,
        {
            "enabled": True,
            "ratios": [1.0],
            "replicas_per_layer": 100,
            "detune_cents": 6,
        },
        "sym_replicas",
    )
    combined = lp._resolve_frequency_layers(
        {},
        440.0,
        {
            "enabled": True,
            "ratios": [1.0] * 100,
            "replicas_per_layer": 100,
            "detune_cents": 6,
        },
        "sym_combined",
    )

    assert len(explicit) == lp.MAX_SYMBOLIC_FREQUENCY_LAYERS
    assert len(ratios) == lp.MAX_SYMBOLIC_FREQUENCY_RATIOS
    assert len(replicas) == lp.MAX_SYMBOLIC_REPLICAS_PER_LAYER
    assert len(combined) == lp.MAX_SYMBOLIC_FREQUENCY_LAYERS


def test_render_sample_budget_is_checked_before_audio_allocation(monkeypatch):
    monkeypatch.setattr(
        lp,
        "_render_symbolic_synthesis_plan",
        lambda _plan: pytest.fail("audio allocation started before sample-budget validation"),
    )
    excessive_duration_ms = (
        (lp.MAX_SYMBOLIC_RENDER_SAMPLES + 1)
        * 1000
        / lp.SYMBOLIC_VOICE_MIN_SAMPLE_RATE
    )

    with pytest.raises(ValueError, match="sample budget"):
        lp.synthesize_from_fingerprint(
            {"pitch_mean": 440.0},
            duration_ms=excessive_duration_ms,
            sr=lp.SYMBOLIC_VOICE_MIN_SAMPLE_RATE,
        )


def test_render_work_budget_is_checked_before_audio_allocation(monkeypatch):
    monkeypatch.setattr(
        lp,
        "_render_symbolic_synthesis_plan",
        lambda _plan: pytest.fail("audio allocation started before work-budget validation"),
    )
    target_samples = (
        lp.MAX_SYMBOLIC_RENDER_WORK // lp.MAX_SYMBOLIC_FREQUENCY_LAYERS
    ) + 10
    duration_ms = target_samples * 1000 / lp.SYMBOLIC_VOICE_MIN_SAMPLE_RATE
    fingerprint = {
        "frequency_layers": [
            {"freq": 220.0 + idx}
            for idx in range(lp.MAX_SYMBOLIC_FREQUENCY_LAYERS)
        ]
    }

    with pytest.raises(ValueError, match="work budget"):
        lp.synthesize_from_fingerprint(
            fingerprint,
            duration_ms=duration_ms,
            sr=lp.SYMBOLIC_VOICE_MIN_SAMPLE_RATE,
        )


def test_speak_preflights_total_sample_budget_before_rendering(monkeypatch):
    samples_per_symbol = int(lp.SYMBOLIC_VOICE_MIN_SAMPLE_RATE * 1.5)
    symbol_count = lp.MAX_SYMBOLIC_RENDER_SAMPLES // samples_per_symbol + 1
    symbols = [f"sym_{idx}" for idx in range(symbol_count)]
    symbol_map = {
        symbol: {"sound_features": {"pitch_mean": 440.0}}
        for symbol in symbols
    }
    monkeypatch.setattr(
        lp,
        "load_config",
        lambda: {
            "allow_polyphonic_voice": False,
            "voice_sample_rate": lp.SYMBOLIC_VOICE_MIN_SAMPLE_RATE,
            "feedback_heard_voice": False,
        },
    )
    monkeypatch.setattr(lp, "load_sound_symbol_map", lambda _child: symbol_map)
    monkeypatch.setattr(
        lp,
        "_render_symbolic_synthesis_plan",
        lambda _plan: pytest.fail("audio rendering began before aggregate preflight"),
    )

    with pytest.raises(ValueError, match="total rendered-sample budget"):
        lp.speak_symbolically(symbols, child="TestChild", playback=False)


def test_speak_preflights_total_work_budget_before_rendering(monkeypatch):
    samples_per_symbol = int(lp.SYMBOLIC_VOICE_MIN_SAMPLE_RATE * 1.5)
    work_per_symbol = samples_per_symbol * lp.MAX_SYMBOLIC_FREQUENCY_LAYERS
    symbol_count = lp.MAX_SYMBOLIC_RENDER_WORK // work_per_symbol + 1
    symbols = [f"sym_work_{idx}" for idx in range(symbol_count)]
    fingerprint = {
        "frequency_layers": [
            {"freq": 220.0 + layer_idx}
            for layer_idx in range(lp.MAX_SYMBOLIC_FREQUENCY_LAYERS)
        ]
    }
    symbol_map = {
        symbol: {"sound_features": fingerprint}
        for symbol in symbols
    }
    monkeypatch.setattr(
        lp,
        "load_config",
        lambda: {
            "allow_polyphonic_voice": False,
            "voice_sample_rate": lp.SYMBOLIC_VOICE_MIN_SAMPLE_RATE,
            "feedback_heard_voice": False,
        },
    )
    monkeypatch.setattr(lp, "load_sound_symbol_map", lambda _child: symbol_map)
    monkeypatch.setattr(
        lp,
        "_render_symbolic_synthesis_plan",
        lambda _plan: pytest.fail("audio rendering began before aggregate preflight"),
    )

    with pytest.raises(ValueError, match="total synthesis-work budget"):
        lp.speak_symbolically(symbols, child="TestChild", playback=False)


def test_non_finite_config_sample_rate_falls_back_for_file_only_wav(
    tmp_path,
    monkeypatch,
):
    import wave

    monkeypatch.setattr(
        lp,
        "load_config",
        lambda: {
            "allow_polyphonic_voice": False,
            "voice_sample_rate": float("nan"),
            "feedback_heard_voice": False,
        },
    )
    monkeypatch.setattr(
        lp,
        "load_sound_symbol_map",
        lambda _child: {"sym_test": {"sound_features": {"pitch_mean": 440.0}}},
    )
    destination = tmp_path / "bounded_file_only.wav"

    lp.speak_symbolically(
        "sym_test",
        child="TestChild",
        playback=False,
        record_path=destination,
    )

    with wave.open(str(destination), "rb") as wav_file:
        assert wav_file.getframerate() == lp.SYMBOLIC_VOICE_DEFAULT_SAMPLE_RATE
        assert wav_file.getnframes() == int(lp.SYMBOLIC_VOICE_DEFAULT_SAMPLE_RATE * 1.5)
