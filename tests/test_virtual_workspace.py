import json
from types import SimpleNamespace

from ina_desktop import audio
from ina_desktop.client import launch_environment, share_file
from ina_desktop.paths import display_number
from ina_desktop.service import VirtualWorkspaceService


def test_launch_environment_routes_display_and_private_audio(monkeypatch):
    monkeypatch.setattr("ina_desktop.client.workspace_status", lambda child: {
        "ready": True,
        "display": ":117",
        "audio": {
            "output_sink": "ina_workspace_output",
            "input_source": "ina_workspace_input.monitor",
        },
    })
    env = launch_environment("Ina", {"KEEP": "yes"})
    assert env["KEEP"] == "yes"
    assert env["DISPLAY"] == ":117"
    assert env["PULSE_SINK"] == "ina_workspace_output"
    assert env["PULSE_SOURCE"] == "ina_workspace_input.monitor"
    assert env["INA_VIRTUAL_WORKSPACE"] == "1"


def test_display_assignment_is_stable_and_bounded():
    assert display_number("Ina") == display_number("Ina")
    assert 100 <= display_number("Ina") < 150


def test_share_file_copies_without_moving_source(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "painting.png"
    source.write_bytes(b"paint")
    result = share_file("Ina", source)
    assert result["ok"] is True
    assert source.exists()
    assert (tmp_path / result["path"]).read_bytes() == b"paint"
    assert (tmp_path / result["manifest"]).exists()


def test_audio_bus_creation_is_idempotent(monkeypatch):
    sinks = set()
    next_module = iter((40, 41))

    def run(command):
        if command[:4] == ["pactl", "list", "sinks", "short"]:
            text = "\n".join(f"1\t{name}\tmodule-null-sink" for name in sorted(sinks))
            return SimpleNamespace(returncode=0, stdout=text, stderr="")
        name = next(part.split("=", 1)[1] for part in command if part.startswith("sink_name="))
        sinks.add(name)
        return SimpleNamespace(returncode=0, stdout=str(next(next_module)), stderr="")

    monkeypatch.setattr(audio, "_run", run)
    first = audio.ensure_audio_buses()
    second = audio.ensure_audio_buses()
    assert first["ready"] and second["ready"]
    assert first["module_ids"] == [40, 41]
    assert second["module_ids"] == []


def test_service_dispatch_exposes_full_input_and_bounded_capture(tmp_path):
    calls = []

    class Desktop:
        def mouse_move(self, x, y): calls.append(("move", x, y))
        def mouse_button(self, button, pressed): calls.append(("button", button, pressed))
        def key(self, keysym, pressed): calls.append(("key", keysym, pressed))
        def type_text(self, text): calls.append(("text", text))
        def focus(self, window_id): calls.append(("focus", window_id))
        def tile(self): return [{"window_id": 1}]
        def windows(self): return []
        def save_ppm(self, path): path.write_bytes(b"P6\n1 1\n255\n\0\0\0"); return path

    service = VirtualWorkspaceService("Ina")
    service.root = tmp_path
    service.desktop = Desktop()
    assert service._dispatch({"action": "mouse_move", "x": 4, "y": 8})["ok"]
    assert service._dispatch({"action": "mouse_button", "button": 1, "pressed": True})["ok"]
    assert service._dispatch({"action": "key", "keysym": "a", "pressed": True})["ok"]
    assert service._dispatch({"action": "type_text", "text": "hello"})["ok"]
    assert service._dispatch({"action": "focus", "window_id": 7})["ok"]
    assert service._dispatch({"action": "tile"})["windows"] == [{"window_id": 1}]
    assert service._dispatch({"action": "capture"})["path"].endswith("latest.ppm")
    assert calls == [
        ("move", 4, 8), ("button", 1, True), ("key", "a", True),
        ("text", "hello"), ("focus", 7),
    ]
