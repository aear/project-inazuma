import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from memory.history import MistakeHistory


class DummyAlignmentModule:
    def __init__(self):
        self.received = None

    def receive_mistake_logs(self, logs):
        self.received = logs


def test_logging_and_review_and_exposure():
    history = MistakeHistory()
    history.log_mistake("act1", "bad", "law1")
    history.log_mistake("act2", "worse", "law1")
    history.log_mistake("act3", "bad", "law2")

    summary = history.review_mistakes()
    assert summary["law1"] == 2
    assert summary["law2"] == 1

    dummy = DummyAlignmentModule()
    history.expose_to_alignment(dummy)
    assert dummy.received == history.get_logs()
