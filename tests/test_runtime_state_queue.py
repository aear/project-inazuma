from concurrent.futures import ThreadPoolExecutor
import threading

import pytest

import runtime_state as rs


def test_get_and_update_inastate_can_scope_a_specific_child(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(rs, "_current_child", lambda: "ActiveChild")

    rs.update_inastate("mood", "active")
    rs.update_inastate("mood", "other", child="OtherChild")

    assert rs.get_inastate("mood") == "active"
    assert rs.get_inastate("mood", child="OtherChild") == "other"
    assert rs.get_inastate("missing", "fallback", child="OtherChild") == "fallback"


def test_append_is_fifo_bounded_and_drain_claims_oldest_batch(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    results = [
        rs.append_inastate_queue("commands", value, queue_limit=3, child="Ina")
        for value in ("first", "second", "third", "rejected")
    ]

    assert [result["queued"] for result in results] == [True, True, True, False]
    assert results[-1] == {
        "queued": False,
        "remaining": 3,
        "dropped": 1,
        "invalid": False,
    }
    claim = rs.drain_inastate_queue(
        "commands", batch_limit=2, queue_limit=3, child="Ina"
    )
    assert claim == {
        "batch": ["first", "second"],
        "remaining": 1,
        "dropped": 0,
        "invalid": False,
    }
    assert rs.get_inastate("commands", child="Ina") == ["third"]


def test_queue_primitives_support_legacy_object_and_clear_invalid_state(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rs.update_inastate("commands", {"id": "legacy"}, child="Ina")

    legacy = rs.drain_inastate_queue(
        "commands", batch_limit=1, queue_limit=4, child="Ina"
    )
    assert legacy["batch"] == [{"id": "legacy"}]
    assert legacy["remaining"] == 0
    assert not legacy["invalid"]

    rs.update_inastate("commands", "not-a-queue", child="Ina")
    invalid = rs.drain_inastate_queue(
        "commands", batch_limit=1, queue_limit=4, child="Ina"
    )
    assert invalid == {"batch": [], "remaining": 0, "dropped": 0, "invalid": True}
    assert rs.get_inastate("commands", child="Ina") == []


@pytest.mark.parametrize("name", ["batch_limit", "queue_limit"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_drain_rejects_invalid_limits(tmp_path, monkeypatch, name, value):
    monkeypatch.chdir(tmp_path)
    kwargs = {"batch_limit": 1, "queue_limit": 4}
    kwargs[name] = value
    with pytest.raises(ValueError, match=f"{name} must be a positive integer"):
        rs.drain_inastate_queue("commands", child="Ina", **kwargs)


def test_concurrent_appends_do_not_overwrite_each_other(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    count = 64
    barrier = threading.Barrier(count)

    def enqueue(value):
        barrier.wait()
        return rs.append_inastate_queue(
            "commands", value, queue_limit=count, child="Ina"
        )

    with ThreadPoolExecutor(max_workers=count) as pool:
        results = list(pool.map(enqueue, range(count)))

    assert all(result["queued"] for result in results)
    stored = rs.get_inastate("commands", child="Ina")
    assert len(stored) == count
    assert set(stored) == set(range(count))


def test_concurrent_claims_neither_duplicate_nor_lose_commands(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rs.update_inastate("commands", list(range(60)), child="Ina")
    barrier = threading.Barrier(6)

    def claim(_index):
        barrier.wait()
        return rs.drain_inastate_queue(
            "commands", batch_limit=10, queue_limit=60, child="Ina"
        )

    with ThreadPoolExecutor(max_workers=6) as pool:
        claims = list(pool.map(claim, range(6)))

    claimed = [item for result in claims for item in result["batch"]]
    assert len(claimed) == 60
    assert set(claimed) == set(range(60))
    assert rs.get_inastate("commands", child="Ina") == []


def test_concurrent_append_and_claim_are_one_atomic_transition(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rs.update_inastate("commands", list(range(20)), child="Ina")
    barrier = threading.Barrier(2)

    def append():
        barrier.wait()
        return rs.append_inastate_queue(
            "commands", 20, queue_limit=32, child="Ina"
        )

    def claim():
        barrier.wait()
        return rs.drain_inastate_queue(
            "commands", batch_limit=10, queue_limit=32, child="Ina"
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        append_future = pool.submit(append)
        claim_future = pool.submit(claim)
        append_result = append_future.result()
        claim_result = claim_future.result()

    assert append_result["queued"]
    assert claim_result["batch"] == list(range(10))
    assert rs.get_inastate("commands", child="Ina") == list(range(10, 21))
