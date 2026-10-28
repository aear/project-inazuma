"""One bounded, event-driven daemon worker for optional Discord storage I/O.

A timeout stops waiting, not an OS read already in progress. Keep its slot until
it really completes; never spawn replacements or queue unbounded pending work.
Unlike ThreadPoolExecutor this worker does not register an interpreter-exit join
that can hold bridge shutdown behind a stuck filesystem call.
"""
from __future__ import annotations
import asyncio
from concurrent.futures import Future
import math
import queue
import threading


class StorageUnavailable(RuntimeError):
    """No result is available within the bounded storage opportunity."""


class BoundedDiscordStorage:
    def __init__(self):
        self._slots = threading.BoundedSemaphore(2)
        self._queue = queue.Queue(maxsize=2)
        self._lock = threading.Lock()
        self._closed = False
        self._thread = threading.Thread(target=self._work, name='discord_storage', daemon=True)
        self._thread.start()

    def _work(self):
        while True:
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                return
            future, function, args, kwargs = item
            try:
                with self._lock:
                    if self._closed:
                        future.cancel()
                    run = not self._closed and future.set_running_or_notify_cancel()
                if run:
                    try:
                        value = function(*args, **kwargs)
                    except BaseException as exc:
                        future.set_exception(exc if isinstance(exc, Exception) else RuntimeError('storage operation terminated'))
                    else:
                        future.set_result(value)
            finally:
                self._slots.release()
                self._queue.task_done()

    async def call(self, function, *args, timeout=1.0, **kwargs):
        if not math.isfinite(timeout) or not 0 < timeout <= 5:
            raise ValueError('storage deadline must be finite, positive and at most five seconds')
        with self._lock:
            if self._closed or not self._slots.acquire(blocking=False):
                raise StorageUnavailable('storage worker closed or busy')
            future = Future()
            self._queue.put_nowait((future, function, args, kwargs))
        try:
            return await asyncio.wait_for(asyncio.wrap_future(future), timeout)
        except TimeoutError as exc:
            # Cancel queued work; running reads retain their occupied slot.
            future.cancel()
            raise StorageUnavailable('storage deadline exceeded') from exc
        except asyncio.CancelledError:
            future.cancel()
            raise

    def close(self):
        """Cancel queued jobs without waiting for an uninterruptible OS read."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            while True:
                try:
                    item = self._queue.get_nowait()
                except queue.Empty:
                    break
                if item is not None:
                    item[0].cancel()
                    self._slots.release()
                self._queue.task_done()
            self._queue.put_nowait(None)
