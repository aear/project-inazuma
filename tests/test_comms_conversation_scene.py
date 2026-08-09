import comms_core as cc


class _ExperienceLogger:
    def log_event(self, **_kwargs):
        return None

    def attach_word_usage(self, *_args, **_kwargs):
        return None


def test_comms_core_attaches_scene_before_processing(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(cc, "ExperienceLogger", _ExperienceLogger)
    monkeypatch.setattr(cc, "record_text_observation", None)
    monkeypatch.setattr(cc, "increment_inastate_metric", lambda *_args, **_kwargs: None)
    observed = []

    def process(message):
        observed.append(message.metadata["conversation_scene"])
        return cc.CommsResponse(text="I heard you.", metadata={})

    core = cc.CommsCore(log_dir=tmp_path, process_inbound=process)
    core.register_backend("discord", lambda _message: None)
    sender = cc.SenderInfo("sakura", "1", "Sakura", backend="discord")
    channel = cc.ChannelInfo("room", "2", "room", backend="discord")

    core.receive_inbound(
        backend="discord",
        backend_message_id="first",
        sender=sender,
        channel=channel,
        text="Do you remember the garden?",
    )
    core.receive_inbound(
        backend="discord",
        backend_message_id="second",
        sender=sender,
        channel=channel,
        text="What did it feel like?",
    )

    assert observed[0]["turn_count"] == 1
    # First inbound + Ina's outbound + second inbound.
    assert observed[1]["turn_count"] == 3
    assert observed[1]["signals"]["reply_expected"] is True
