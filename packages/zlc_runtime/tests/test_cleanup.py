"""Shutting things down without waiting forever for them."""

from __future__ import annotations

import threading

import pytest

from zlc_runtime.cleanup import join_worker

def test_a_worker_that_will_not_stop_is_reported_rather_than_waited_on() -> None:
    """The failure a bare join turns into a frozen window with no message.

    What calls close is usually something shutting down -- a panel, a window, a
    session -- and it is often the GUI thread.  A worker wedged in a driver then
    takes the application down with it, silently.
    """

    stuck = threading.Event()
    worker = threading.Thread(target=stuck.wait, daemon=True)
    worker.start()
    try:
        with pytest.raises(TimeoutError, match="the wedged device"):
            join_worker(worker, what="the wedged device", timeout=0.05)
    finally:
        stuck.set()
        worker.join(5.0)


def test_a_worker_that_stops_is_simply_waited_for() -> None:
    worker = threading.Thread(target=lambda: None, daemon=True)
    worker.start()

    join_worker(worker, what="a worker that finished", timeout=5.0)

    assert not worker.is_alive()


def test_a_worker_cannot_be_asked_to_wait_for_itself() -> None:
    """Which is a deadlock, and reads as a hang rather than a mistake."""

    failure: list[BaseException] = []

    def _join_self() -> None:
        try:
            join_worker(threading.current_thread(), what="itself", timeout=1.0)
        except BaseException as error:
            failure.append(error)

    worker = threading.Thread(target=_join_self, daemon=True)
    worker.start()
    worker.join(5.0)

    assert failure and isinstance(failure[0], RuntimeError)
