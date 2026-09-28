import os
import pytest

from obs_bridge import OBSWebSocketBridge


def test_obs_bridge_is_loopback_only_and_mutations_default_off():
    with pytest.raises(ValueError, match="loopback"):
        OBSWebSocketBridge(host="example.com")
    prior = os.environ.get("OBS_TEST_PASSWORD")
    try:
        os.environ["OBS_TEST_PASSWORD"] = "secret"
        bridge = OBSWebSocketBridge.from_config({
            "enabled": True, "host": "127.0.0.1", "password": "ignored-inline",
            "password_env": "OBS_TEST_PASSWORD", "use_replay_buffer": True,
        })
        assert bridge.password == "secret"
        assert bridge.allow_mutations is False
        assert bridge.can_save_replay is False
    finally:
        if prior is None:
            os.environ.pop("OBS_TEST_PASSWORD", None)
        else:
            os.environ["OBS_TEST_PASSWORD"] = prior


def test_obs_frame_decoder_rejects_oversized_input():
    assert OBSWebSocketBridge._decode_image("data:image/png;base64," + "A" * (12 * 1024 * 1024 + 1)) is None
