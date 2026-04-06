import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from player_client import PlayerClient


def test_player_client_state_view_and_snapshot_are_separate():
    client = PlayerClient(tcp_host="127.0.0.1", tcp_port=7777, local_socket="/tmp/test_player_client.sock")
    state = {"entities": {"player": {"position": [1.0, 2.0, 3.0]}}}

    client._update_state(state)

    assert client.get_state_version() == 1
    assert client.get_state_view() is state

    snapshot = client.get_state_snapshot()

    assert snapshot == state
    assert snapshot is not state
    snapshot["entities"]["player"]["position"][0] = 99.0
    assert client.get_state_view()["entities"]["player"]["position"][0] == 1.0


def test_player_client_state_version_increments_on_update():
    client = PlayerClient(tcp_host="127.0.0.1", tcp_port=7777, local_socket="/tmp/test_player_client.sock")

    client._update_state({"entities": {}})
    client._update_state({"entities": {"ina": {"position": [0.0, 0.0, 0.0]}}})

    assert client.get_state_version() == 2
