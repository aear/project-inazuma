from eeg_rendering import dangling_endpoint_ids
from safe_popen import _stderr_tag


def test_dangling_endpoint_ids_reports_each_missing_reference_once():
    nodes = [{"id": "sound:known"}, {"id": "word:known"}]
    edges = [
        {"source": "sound:known", "target": "word:known"},
        {"source": "sound:sym_emotion_0001", "target": "word:known"},
        {"source": "sound:sym_emotion_0001", "target": "word:known"},
        {"source": "sound:sym_emotion_0002", "target": "word:known"},
    ]

    assert dangling_endpoint_ids(nodes, edges) == (
        "sound:sym_emotion_0001",
        "sound:sym_emotion_0002",
    )


def test_wayland_activation_limit_is_a_warning_not_an_app_error():
    assert _stderr_tag(
        "qt.qpa.wayland: Wayland does not support QWindow::requestActivate()"
    ) == "WARN"
    assert _stderr_tag("Traceback: render failed") == "ERROR"
