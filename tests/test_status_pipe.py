import os
import select

from status_pipe import open_persistent_fifo_reader


def _write_once(path, message):
    fd = os.open(path, os.O_WRONLY | os.O_NONBLOCK)
    try:
        os.write(fd, message.encode("utf-8"))
    finally:
        os.close(fd)


def test_persistent_fifo_reader_survives_separate_writers(tmp_path):
    path = tmp_path / "status.pipe"
    os.mkfifo(path)
    with open_persistent_fifo_reader(path) as reader:
        _write_once(path, "first\n")
        assert reader.readline() == "first\n"
        # The reader retains its own write endpoint, so the first writer's
        # close is not observed as EOF before the next writer arrives.
        assert select.select([reader], [], [], 0.02)[0] == []
        _write_once(path, "second\n")
        assert reader.readline() == "second\n"
