import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

import discord_bridge as db


def test_discord_chunking_preserves_full_paragraph_text():
    text = ("A" * 1200) + "\n\n" + ("B" * 1200) + "\nFinal sentence."

    chunks = db.split_discord_message(text)

    assert len(chunks) == 2
    assert all(len(chunk) <= db.DISCORD_MESSAGE_CHAR_LIMIT for chunk in chunks)
    assert "".join(chunks) == text
    assert chunks[0].endswith("\n\n")


def test_chunk_budget_includes_native_and_english_guess():
    native = "λ" * 1450
    english = "word " * 180
    paired = f"Native: {native}\nHuman guess: {english}"

    chunks = db.split_discord_message(paired)

    assert len(paired) > db.DISCORD_MESSAGE_CHAR_LIMIT
    assert len(chunks) >= 2
    assert all(len(chunk) <= db.DISCORD_MESSAGE_CHAR_LIMIT for chunk in chunks)
    assert "".join(chunks) == paired


def test_discord_sender_chunks_only_at_delivery_and_attaches_once():
    sent = []
    files = []

    class File:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    class Destination:
        async def send(self, text, file=None):
            sent.append(text)
            files.append(file)

    async def pace():
        return None

    client = SimpleNamespace(
        _outbox_policy={
            "max_send_retries": 1,
            "rate_limit_padding_seconds": 0,
        },
        _pace_discord_send=pace,
    )
    original = ("first paragraph. " * 130) + "\n\n" + ("second paragraph. " * 130)
    made_files = []

    def file_factory():
        file = File()
        made_files.append(file)
        return file

    delivered = asyncio.run(
        db.InaDiscordClient.send_discord_message(
            client,
            Destination(),
            original,
            file_factory=file_factory,
            reason="test-full-expression",
        )
    )

    assert delivered is True
    assert "".join(sent) == original
    assert all(len(chunk) <= db.DISCORD_MESSAGE_CHAR_LIMIT for chunk in sent)
    assert len(sent) > 1
    assert files[0] is made_files[0]
    assert all(file is None for file in files[1:])
    assert made_files[0].closed is True


class _Adapter:
    def __init__(self, response="adapter reply"):
        self.calls = []
        self.response = response

    def handle_prompt(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return self.response


def _message(text, *, attachments=None, context=None):
    return SimpleNamespace(
        text=text,
        metadata={
            "image_attachments": attachments or [],
            "conversation_context": context or [],
            "is_dm": False,
        },
        sender=SimpleNamespace(
            backend_id="1", display_name="Sakura", internal_id="sakura"
        ),
        channel=SimpleNamespace(backend_id="2", name="ina-text"),
    )


def _enable_replying(monkeypatch):
    monkeypatch.setattr(
        db, "load_root_config", lambda: {"ignore_urge_for_typing": True}
    )
    monkeypatch.setattr(db, "get_current_child", lambda: "TestChild")
    monkeypatch.setattr(db, "update_inastate", lambda *args, **kwargs: None)


def _force_expression_strategy(monkeypatch, strategy):
    monkeypatch.setattr(
        db,
        "choose_text_expression_strategy",
        lambda *args, **kwargs: {
            "strategy": strategy,
            "reason": "test_selection",
            "scores": {},
            "mapping_coverage": 1.0,
            "mapped_count": kwargs.get("mapped_count", 0),
            "token_count": kwargs.get("token_count", 0),
            "mirror_streak": 1 if strategy == "mirror" else 0,
            "signals": {},
        },
    )


def test_complete_symbolic_reply_can_be_english_when_ina_prefers_it(monkeypatch, tmp_path):
    _enable_replying(monkeypatch)
    monkeypatch.chdir(tmp_path)
    state_path = tmp_path / "AI_Children" / "TestChild" / "memory" / "inastate.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text(
        '{"discord_language_preference": "english", '
        '"text_expression_intent": {"strategy": "mirror"}}',
        encoding="utf-8",
    )
    adapter = _Adapter()
    monkeypatch.setattr(db, "get_chat_adapter", lambda: adapter)
    monkeypatch.setattr(
        db,
        "generate_symbolic_reply_from_text",
        lambda *args, **kwargs: {
            "text": "Native: glyph_wave\nHuman guess: hello",
            "native_text": "glyph_wave",
            "gloss_text": "hello",
            "symbols": ["sym_wave"],
            "unknown": [],
        },
    )

    result = db.process_inbound_message(_message("hello"))

    assert result.text == "hello"
    assert result.metadata["effective_language_mode"] == "english"
    assert result.metadata["symbolic_native_text"] == "glyph_wave"
    assert result.metadata["expression_decision"]["reason"] == "explicit_intent"
    assert result.metadata["adapter"] == "deliberate_mirror"
    assert adapter.calls == []


def test_complete_mapping_is_comprehension_not_default_reply(monkeypatch, tmp_path):
    _enable_replying(monkeypatch)
    monkeypatch.chdir(tmp_path)
    state_path = tmp_path / "AI_Children" / "TestChild" / "memory" / "inastate.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text(
        '{"discord_language_preference": "english"}',
        encoding="utf-8",
    )
    adapter = _Adapter(response="my selected response")
    monkeypatch.setattr(db, "get_chat_adapter", lambda: adapter)

    def symbolic(text, **kwargs):
        if text == "hello":
            return {
                "text": "Native: glyph_wave\nHuman guess: hello",
                "native_text": "glyph_wave",
                "gloss_text": "hello",
                "symbols": ["sym_wave"],
                "unknown": [],
            }
        return None

    monkeypatch.setattr(db, "generate_symbolic_reply_from_text", symbolic)

    result = db.process_inbound_message(_message("hello"))

    assert result.text == "my selected response"
    assert result.metadata["expression_decision"]["strategy"] == "respond"
    assert result.metadata["comprehension_adapter"] == "language_processing"
    assert adapter.calls[0][0] == "hello"


def test_emotion_signal_is_bounded_but_can_expose_all_24():
    values = {f"slider_{index:02d}": (index - 12) / 12 for index in range(24)}
    state = {"emotion_snapshot": {"values": values}}

    concise = db.format_emotion_signal(state)
    complete = db.format_emotion_signal(state, max_items=24)

    assert concise["shown"] == 6
    assert concise["available"] == 24
    assert complete["shown"] == 24
    assert len(complete["sliders"]) == 24


def test_code_pointer_signal_validates_modules_and_functions(tmp_path):
    (tmp_path / "signal_target.py").write_text(
        "def indicate_problem():\n    return True\n",
        encoding="utf-8",
    )
    state = {
        "text_expression_intent": {
            "strategy": "code_pointer",
            "pointers": [
                "signal_target.py:indicate_problem",
                "signal_target.py:missing_function",
                "../outside.py",
            ],
        }
    }

    signal = db.format_code_pointer_signal(state, repo_root=tmp_path)

    assert signal["text"] == "Code pointer: signal_target.py → indicate_problem"
    assert signal["pointers"] == [
        {"module": "signal_target.py", "functions": ["indicate_problem"]}
    ]
    assert {item["reason"] for item in signal["rejections"]} == {
        "function_not_found",
        "outside_repository",
    }


def test_expression_arbiter_honours_explicit_code_pointer():
    state = {
        "text_expression_intent": {"strategy": "code_pointer"},
        "emotion_snapshot": {"values": {"curiosity": 0.8, "clarity": 0.9}},
    }

    decision = db.choose_text_expression_strategy(
        state,
        mapped_count=1,
        token_count=1,
        adapter_available=True,
        expression_drive=0.8,
        emotion_available=True,
        code_pointer_available=True,
    )

    assert decision["strategy"] == "code_pointer"
    assert decision["reason"] == "explicit_intent"



def test_incomplete_selected_translation_stays_full_english(monkeypatch):
    monkeypatch.setattr(
        db,
        "generate_symbolic_reply_from_text",
        lambda *args, **kwargs: {
            "symbols": ["sym_snd_internal"],
            "unknown": ["grounding", "restart", "strategies"],
        },
    )
    monkeypatch.setattr(
        db,
        "build_dual_symbolic_message",
        lambda *args, **kwargs: {
            "text": (
                "Native: glyph_known sym_snd_internal\n"
                "Human guess: I do not have grounding for restart strategies."
            ),
            "native_text": "glyph_known sym_snd_internal",
            "gloss_text": "I do not have grounding for restart strategies.",
            "native_sources": {},
        },
    )
    response = "I do not have grounding for restart strategies."

    rendered, metadata = db.encode_selected_text_expression(
        response,
        child="TestChild",
        language_preference="auto",
        max_symbols=24,
    )

    assert rendered == response
    assert metadata["effective_language_mode"] == "english_incomplete_native"
    assert metadata["native_translation_complete"] is False
    assert set(metadata["native_translation_rejections"]) == {
        "unknown_response_words",
        "incomplete_token_coverage",
        "unresolved_native_symbols",
    }
    assert metadata["native_translation_unresolved_symbols"] == [
        "sym_snd_internal"
    ]


def test_complete_selected_translation_can_render_native_and_english(monkeypatch):
    monkeypatch.setattr(
        db,
        "generate_symbolic_reply_from_text",
        lambda *args, **kwargs: {
            "symbols": ["sym_calm", "sym_here"],
            "unknown": [],
        },
    )
    monkeypatch.setattr(
        db,
        "build_dual_symbolic_message",
        lambda *args, **kwargs: {
            "text": "Native: λcalm λhere\nHuman guess: calm here",
            "native_text": "λcalm λhere",
            "gloss_text": "calm here",
            "native_sources": {
                "sym_calm": "text_vocab_links",
                "sym_here": "text_vocab_links",
            },
        },
    )

    rendered, metadata = db.encode_selected_text_expression(
        "calm here",
        child="TestChild",
        language_preference="mixed",
        max_symbols=24,
    )

    assert rendered == "Native: λcalm λhere\nHuman guess: calm here"
    assert metadata["effective_language_mode"] == "mixed"
    assert metadata["native_translation_complete"] is True
    assert metadata["native_translation_rejections"] == []

def test_history_parser_recovers_native_english_pairs_without_cross_pairing():
    assert db.extract_symbolic_history_alignments(
        "Native: glyph_wave glyph_calm\nHuman guess: hello calm"
    ) == [("glyph_wave glyph_calm", "hello calm")]
    parsed = db.parse_symbolic_history_message(
        "Native: orphan\nNative: glyph_calm\nHuman guess: calm\nHuman guess: orphan"
    )
    assert parsed["pairs"] == [("glyph_calm", "calm")]
    assert parsed["rejection_counts"] == {
        "orphan_native": 1,
        "orphan_human_guess": 1,
    }
    assert db.extract_symbolic_history_alignments("Just English this time.") == []


def test_language_review_policy_is_bounded(monkeypatch):
    monkeypatch.setattr(
        db,
        "get_discord_config",
        lambda: {
            "language_review": {
                "pressure_messages": 0,
                "history_limit": 2,
                "mapping_batch": 0,
                "revisit_mappings": 0,
                "cooldown_seconds": -1,
            }
        },
    )

    policy = db.get_language_review_policy()

    assert policy["pressure_messages"] == 4
    assert policy["history_limit"] == 10
    assert policy["mapping_batch"] == 1
    assert policy["revisit_mappings"] == 1
    assert policy["cooldown_seconds"] == 0.0


def test_history_review_learns_old_text_and_deduplicates_live_text(monkeypatch):
    captured = {}
    monkeypatch.setattr(db, "get_memory_guard_level", lambda: "normal")
    monkeypatch.setattr(
        db,
        "get_inastate",
        lambda key: {
            "history_cursors": {},
            "live_vocab_message_ids": ["10"],
        } if key == "discord_language_review" else None,
    )

    def review(observations, alignments, **kwargs):
        captured["observations"] = observations
        captured["alignments"] = alignments
        return {"pairs": [{"native": "glyph_wave", "english": "hello"}]}

    monkeypatch.setattr(db, "review_text_evidence", review)
    monkeypatch.setattr(db, "build_text_symbol_links", lambda *args, **kwargs: True)
    stamp = datetime.now(timezone.utc)
    human = SimpleNamespace(id=1, bot=False, display_name="Human")
    ina = SimpleNamespace(id=999, bot=True, display_name="Ina")
    messages = [
        SimpleNamespace(id=9, author=human, content="older word", created_at=stamp),
        SimpleNamespace(id=10, author=human, content="already live", created_at=stamp),
        SimpleNamespace(
            id=11,
            author=ina,
            content="Native: glyph_wave\nHuman guess: hello",
            created_at=stamp,
        ),
    ]

    class Channel:
        id = 5

        async def history(self, **kwargs):
            for message in messages:
                yield message

    class HistoryBridge:
        def __init__(self):
            self.turns = []

        def log_conversation_turn(self, text, **kwargs):
            self.turns.append(text)

    class Client:
        text_channel = Channel()
        user = SimpleNamespace(id=999)
        child = "TestChild"
        history_bridge = HistoryBridge()

        def get_user(self, user_id):
            return None

        async def fetch_user(self, user_id):
            return None

    client = Client()
    result = asyncio.run(
        db.InaDiscordClient._ingest_message_history(
            client, limit=10, mapping_batch=3, revisit_mappings=1
        )
    )

    assert captured["observations"] == [
        {"text": "older word", "tags": ["discord", "history", "human"]}
    ]
    assert captured["alignments"] == [("glyph_wave", "hello")]
    assert client.history_bridge.turns == ["older word"]
    assert result["history_cursors"] == {"5": "11"}


def test_forced_history_review_routes_app_authored_structured_pairs(monkeypatch):
    captured = {}
    monkeypatch.setattr(db, "get_memory_guard_level", lambda: "normal")
    monkeypatch.setattr(
        db,
        "get_inastate",
        lambda key: {"history_cursors": {"5": "11"}} if key == "discord_language_review" else None,
    )

    def review(observations, alignments, **kwargs):
        captured["observations"] = observations
        captured["alignments"] = alignments
        return {
            "pairs": [{"native": "°·'φam", "english": "ina"}],
            "alignment_candidates": 1,
            "accepted_alignment_candidates": 1,
            "alignment_rejections": [],
        }

    mapping_calls = []
    monkeypatch.setattr(db, "review_text_evidence", review)
    monkeypatch.setattr(
        db,
        "build_text_symbol_links",
        lambda *args, **kwargs: mapping_calls.append((args, kwargs)) or True,
    )
    stamp = datetime.now(timezone.utc)
    app_author = SimpleNamespace(id=888, bot=False, display_name="Inazuma")
    message = SimpleNamespace(
        id=11,
        author=app_author,
        content="Native: °·'φam λ⊙··\nHuman guess: ina λ⊙··",
        created_at=stamp,
    )

    class Channel:
        id = 5

        async def history(self, **kwargs):
            yield message

    class Client:
        text_channel = Channel()
        user = SimpleNamespace(id=999)
        child = "TestChild"
        history_bridge = SimpleNamespace(log_conversation_turn=lambda *args, **kwargs: None)

        def get_user(self, user_id):
            return None

        async def fetch_user(self, user_id):
            return None

    result = asyncio.run(
        db.InaDiscordClient._ingest_message_history(
            Client(),
            limit=10,
            mapping_batch=7,
            revisit_mappings=3,
            force_history=True,
        )
    )

    assert captured["observations"] == []
    assert captured["alignments"] == [("°·'φam λ⊙··", "ina λ⊙··")]
    assert result["new_messages"] == 0
    assert result["revisited_messages"] == 1
    assert result["alignment_candidates"] == 1
    assert mapping_calls[0][1] == {
        "mapping_batch": 7,
        "revisit_existing": 3,
    }


def test_attachment_does_not_override_complete_symbolic_reply(monkeypatch):
    _enable_replying(monkeypatch)
    _force_expression_strategy(monkeypatch, "mirror")
    adapter = _Adapter()
    monkeypatch.setattr(db, "get_chat_adapter", lambda: adapter)
    monkeypatch.setattr(
        db,
        "generate_symbolic_reply_from_text",
        lambda *args, **kwargs: {
            "text": "known symbolic reply",
            "symbols": ["sym"],
            "unknown": [],
        },
    )

    result = db.process_inbound_message(
        _message(
            "Read em and weep",
            attachments=[{"original_filename": "chart.png"}],
            context=[{"author_name": "Someone", "content": "prior context"}],
        )
    )

    assert result.text == "known symbolic reply"
    assert adapter.calls == []


def test_adapter_only_receives_operator_text(monkeypatch):
    _enable_replying(monkeypatch)
    adapter = _Adapter("grounding response")
    monkeypatch.setattr(db, "get_chat_adapter", lambda: adapter)
    monkeypatch.setattr(db, "generate_symbolic_reply_from_text", lambda *a, **k: None)

    result = db.process_inbound_message(
        _message(
            "novelword",
            context=[{"author_name": "ContextUser", "content": "contextonlyword"}],
        )
    )

    assert result.text == "grounding response"
    assert adapter.calls[0][0] == "novelword"


def test_adapter_turn_is_linked_to_visual_experience(monkeypatch):
    _enable_replying(monkeypatch)
    adapter = _Adapter("grounding response")
    monkeypatch.setattr(db, "get_chat_adapter", lambda: adapter)
    monkeypatch.setattr(db, "generate_symbolic_reply_from_text", lambda *a, **k: None)
    perception = {
        "event_id": "evt_visual",
        "recognized_symbols": [],
        "orientation": "landscape",
        "brightness": 0.2,
        "contrast": 0.4,
    }

    result = db.process_inbound_message(
        _message(
            "What do you see?",
            attachments=[{
                "original_filename": "chart.png",
                "vision_perception": perception,
            }],
        )
    )

    assert result.text == "grounding response"
    links = adapter.calls[0][1]["entity_links"]
    assert any(
        link.get("type") == "vision_perception"
        and link.get("event_id") == "evt_visual"
        for link in links
    )
    assert result.metadata["vision_context"]["perceptions"] == [perception]


def test_caption_words_feed_linked_visual_token_evidence(monkeypatch):
    _enable_replying(monkeypatch)
    adapter = _Adapter("grounding response")
    monkeypatch.setattr(db, "get_chat_adapter", lambda: adapter)
    monkeypatch.setattr(db, "generate_symbolic_reply_from_text", lambda *a, **k: None)
    calls = []

    def learn(event_ids, words, **kwargs):
        calls.append((event_ids, words, kwargs))
        return {
            "status": "learned",
            "updated_clusters": ["vtoken_form"],
            "hypotheses": [],
        }

    monkeypatch.setattr(db, "observe_visual_words", learn)
    perception = {
        "event_id": "evt_visual",
        "recognized_symbols": [],
        "visual_token_learning": {
            "candidate_ids": ["vtoken_form"],
            "matches": [],
        },
    }

    db.process_inbound_message(
        _message(
            "zabble form",
            attachments=[{
                "original_filename": "forms.png",
                "vision_perception": perception,
            }],
        )
    )

    assert calls
    assert calls[0][0] == ["evt_visual"]
    assert calls[0][1] == ["zabble", "form"]
    assert calls[0][2]["child"] == "TestChild"


def test_image_only_turn_uses_stored_image_acknowledgement(monkeypatch):
    _enable_replying(monkeypatch)
    adapter = _Adapter("should not be used")
    monkeypatch.setattr(db, "get_chat_adapter", lambda: adapter)
    monkeypatch.setattr(db, "generate_symbolic_reply_from_text", lambda *a, **k: None)

    result = db.process_inbound_message(
        _message("", attachments=[{"original_filename": "chart.png"}])
    )

    assert "vision pass did not produce a usable perception" in result.text
    assert result.metadata["adapter"] == "image_acknowledgement"
    assert adapter.calls == []


def test_recognized_visual_symbol_participates_in_reply(monkeypatch):
    _enable_replying(monkeypatch)
    _force_expression_strategy(monkeypatch, "mirror")
    adapter = _Adapter("should not be used")
    monkeypatch.setattr(db, "get_chat_adapter", lambda: adapter)
    monkeypatch.setattr(db, "generate_symbolic_reply_from_text", lambda *a, **k: None)
    seen = {}

    def build_message(symbols, **kwargs):
        seen["symbols"] = symbols
        seen["context"] = kwargs["context"]
        return {"text": "visual symbolic reply", "native_text": "vision-native"}

    monkeypatch.setattr(db, "build_dual_symbolic_message", build_message)
    perception = {
        "event_id": "evt_vision",
        "recognized_symbols": ["vision_symbol_known"],
        "orientation": "landscape",
        "brightness": 0.4,
        "contrast": 0.2,
    }

    result = db.process_inbound_message(
        _message(
            "",
            attachments=[
                {
                    "original_filename": "chart.png",
                    "vision_perception": perception,
                }
            ],
        )
    )

    assert result.text == "visual symbolic reply"
    assert result.metadata["symbols"] == ["vision_symbol_known"]
    assert result.metadata["vision_context"]["event_ids"] == ["evt_vision"]
    assert seen["symbols"] == ["vision_symbol_known"]
    assert seen["context"]["vision"]["perceptions"] == [perception]
    assert adapter.calls == []


def test_mature_visual_word_hypothesis_enters_symbolic_language(monkeypatch):
    _enable_replying(monkeypatch)
    _force_expression_strategy(monkeypatch, "mirror")
    adapter = _Adapter("should not be used")
    monkeypatch.setattr(db, "get_chat_adapter", lambda: adapter)
    seen = {}

    def generate(text, **kwargs):
        seen["input"] = text
        seen["context"] = kwargs["context"]
        return {
            "text": "self-read symbolic response",
            "symbols": ["symbol_for_zabble"],
            "unknown": [],
        }

    monkeypatch.setattr(db, "generate_symbolic_reply_from_text", generate)
    perception = {
        "event_id": "evt_visual",
        "recognized_symbols": [],
        "orientation": "landscape",
        "brightness": 0.4,
        "contrast": 0.2,
        "visual_token_learning": {
            "candidate_ids": ["vtoken_form"],
            "matches": [{
                "cluster_id": "vtoken_form",
                "hypotheses": [{
                    "word": "zabble",
                    "support": 4,
                    "confidence": 0.75,
                }],
            }],
        },
    }

    result = db.process_inbound_message(
        _message(
            "",
            attachments=[{
                "original_filename": "forms.png",
                "vision_perception": perception,
            }],
        )
    )

    assert result.text == "self-read symbolic response"
    assert seen["input"] == "zabble"
    assert seen["context"]["source_text"] == ""
    assert seen["context"]["visual_inference_words"] == ["zabble"]
    assert result.metadata["visual_inference_words"] == ["zabble"]
    assert adapter.calls == []


def test_unrecognized_image_ack_reports_measured_visual_properties(monkeypatch):
    _enable_replying(monkeypatch)
    monkeypatch.setattr(db, "get_chat_adapter", lambda: None)
    monkeypatch.setattr(db, "generate_symbolic_reply_from_text", lambda *a, **k: None)

    result = db.process_inbound_message(
        _message(
            "",
            attachments=[
                {
                    "original_filename": "chart.png",
                    "vision_perception": {
                        "recognized_symbols": [],
                        "orientation": "landscape",
                        "brightness": 0.2,
                        "contrast": 0.4,
                    },
                }
            ],
        )
    )

    assert "landscape, dark, high-contrast" in result.text
    assert "did not match a visual symbol" in result.text


def test_dependency_free_image_pass_reaches_vision_and_runtime_state(tmp_path, monkeypatch):
    class VisionBridge:
        calls = []

        def __init__(self, **kwargs):
            pass

        def log_screen_snapshot(self, frame, **kwargs):
            self.calls.append((frame, kwargs))
            return "evt_visual"

    image_path = tmp_path / "sample.pgm"
    pixels = bytes([0, 255] * (80 * 40 // 2))
    image_path.write_bytes(b"P5\n80 40\n255\n" + pixels)
    learned = db.extract_image_features(image_path, limit=1024)["features"][:512]
    state_updates = []

    monkeypatch.setattr(db, "Image", None)
    monkeypatch.setattr(db, "np", None)
    monkeypatch.setattr(db, "LiveExperienceBridge", VisionBridge)
    monkeypatch.setattr(
        db,
        "observe_visual_tokens",
        lambda *a, **k: {
            "status": "observed",
            "candidate_ids": ["vtoken_test"],
            "matches": [],
        },
    )
    monkeypatch.setattr(
        db,
        "load_generated_symbols",
        lambda *a, **k: [{"id": "vision_symbol", "image_features": learned}],
    )
    monkeypatch.setattr(
        db, "update_inastate", lambda key, value: state_updates.append((key, value))
    )
    monkeypatch.setattr(db, "should_accept_fragment", lambda **kwargs: (False, "test"))

    fragment = db._build_discord_image_fragment(
        path=image_path,
        child="TestChild",
        fragment_id="frag_test",
        tags=["discord", "image"],
        summary="test image",
        source_context={"discord_message_id": "1"},
        rel_path="discord_attachments/sample.pgm",
    )

    perception = fragment["vision_perception"]
    assert fragment["stored"] is False
    assert perception["decoder"] == "simple_image_fallback"
    assert perception["event_id"] == "evt_visual"
    assert perception["recognized_symbols"] == ["vision_symbol"]
    assert perception["orientation"] == "landscape"
    assert perception["visual_token_learning"]["candidate_ids"] == ["vtoken_test"]
    assert "vtoken_test" in fragment["tags"]
    assert state_updates[-1] == ("last_discord_vision", perception)
    assert VisionBridge.calls
