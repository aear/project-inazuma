from creative_experience import (
    NEXT_CHOICES, begin_experience, choose_next, experience_command_fields,
    experience_path, record_experiment, save_experience,
)


def test_creative_experience_is_one_attempt_then_an_explicit_choice(tmp_path):
    session = begin_experience("daw", "Try a quiet pulse", hypothesis="space may help the rhythm")
    session = record_experiment(session, {"id": "daw-command-60"}, observation="one pulse heard")

    assert session["stage"] == "observation"
    assert session["experiment_count"] == 1
    assert session["experiments"][0]["choice"] is None
    assert session["next_choices"] == list(NEXT_CHOICES)
    assert session["may_pause"] is True
    assert choose_next(session, "revisit", reflection="listen again later")["stage"] == "revisit"

    save_experience("Ina", session, root=tmp_path, history_limit=2)
    assert experience_path("Ina", tmp_path).exists()


def test_drawing_and_motor_commands_share_non_forcing_experience_metadata():
    drawing = begin_experience("drawing", "Try one line")
    motor = begin_experience("motor", "Try one step")

    for session in (drawing, motor):
        fields = experience_command_fields(session)["creative_experience"]
        assert fields["may_pause"] is True
        assert fields["may_stop"] is True
        assert set(fields["next_choices"]) == set(NEXT_CHOICES)
