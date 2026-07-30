import threading
from unittest.mock import Mock

import ina_client


def test_handle_state_feeds_authoritative_pose_to_motor(monkeypatch):
    client = ina_client.InaClient("/tmp/unused.sock")
    motor = Mock()
    motor.bounds = (-10.0, 10.0, -10.0, 10.0)
    client._motor = motor
    monkeypatch.setattr(client, "_update_world_touch", Mock())
    monkeypatch.setattr(client, "_update_world_pose", Mock())

    client._handle_state(
        {
            "bounds": {"min_x": -4, "max_x": 5, "min_y": -6, "max_y": 7},
            "entities": {
                "ina": {
                    "position": [1, 2, 0],
                    "velocity": [0.5, -0.25, 0],
                    "yaw_deg": 45,
                }
            },
        }
    )

    assert motor.bounds == (-4.0, 5.0, -6.0, 7.0)
    motor.observe_state.assert_called_once_with(
        position=(1.0, 2.0, 0.0),
        velocity=(0.5, -0.25, 0.0),
        yaw_deg=45.0,
    )


def test_stale_reader_cannot_disconnect_replacement_socket(monkeypatch):
    client = ina_client.InaClient("/tmp/unused.sock")
    old_sock = Mock()
    new_sock = Mock()
    new_file = Mock()
    client.sock = new_sock
    client.file = new_file
    client._connected_event.set()
    monkeypatch.setattr(client, "_mark_disconnected", Mock())

    client._handle_disconnect(expected_sock=old_sock)

    assert client.sock is new_sock
    assert client.file is new_file
    assert client._connected_event.is_set()
    new_sock.close.assert_not_called()
    new_file.close.assert_not_called()


def test_heartbeat_waits_for_reconnect_instead_of_exiting(monkeypatch):
    client = ina_client.InaClient("/tmp/unused.sock")
    sent = []
    sleeps = 0

    def fake_sleep(_duration):
        nonlocal sleeps
        sleeps += 1
        if sleeps == 1:
            client._connected_event.set()
        else:
            client._stop_event.set()

    monkeypatch.setattr(client, "_sleep_with_stop", fake_sleep)
    monkeypatch.setattr(client, "send", sent.append)
    monkeypatch.setattr(client, "_set_inastate", Mock())

    worker = threading.Thread(target=client._heartbeat_loop)
    worker.start()
    worker.join(timeout=1)

    assert not worker.is_alive()
    assert sent == [{"type": "ping"}]
