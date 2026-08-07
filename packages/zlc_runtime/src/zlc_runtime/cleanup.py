"""Plan cleanup diagnostics from domain-owned session closure."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import threading


@dataclass(frozen=True)
class CleanupReport:
    """Errors raised while a plan closes its device-specific sessions."""

    errors: tuple[BaseException, ...] = ()

    def __post_init__(self) -> None:
        errors = tuple(self.errors)
        if any(not isinstance(error, BaseException) for error in errors):
            raise TypeError("cleanup errors must contain exceptions")
        object.__setattr__(self, "errors", errors)

    @classmethod
    def complete(
        cls,
        *,
        errors: tuple[BaseException, ...] = (),
    ) -> "CleanupReport":
        return cls(tuple(errors))


def run_cleanup_steps(
    *steps: Callable[[], CleanupReport],
) -> CleanupReport:
    """Run every physical cleanup step and aggregate all failures."""

    errors: list[BaseException] = []
    for step in steps:
        try:
            report = step()
            if not isinstance(report, CleanupReport):
                raise TypeError("hardware cleanup step must return CleanupReport")
        except BaseException as error:
            errors.append(error)
            continue
        errors.extend(report.errors)
    return CleanupReport.complete(errors=tuple(errors))


#: How long a worker that has been told to stop is given to stop.
#:
#: Generous, because the point is not to be quick: it is to be finite.  A close
#: that waits forever is a close that can freeze whatever called it, and what
#: calls close is usually a window shutting down -- so "the camera did not
#: release" arrives as a frozen application with no message in it.
WORKER_STOP_SECONDS = 30.0


def join_worker(
    thread: threading.Thread,
    *,
    what: str,
    timeout: float = WORKER_STOP_SECONDS,
) -> None:
    """Wait for a worker that has already been told to stop, but not forever.

    A bare ``join()`` is right only when the worker cannot get stuck, and a
    worker that talks to hardware always can.  The one it blocks is whoever
    called close, which is the GUI thread often enough that this is the
    difference between an error message and a hung window.

    Raises rather than returning a flag: a device that did not release is
    something the operator has to know before they reach for the power switch,
    and a boolean nobody reads is how they find out from the next run instead.
    """

    if not isinstance(thread, threading.Thread):
        raise TypeError("join_worker needs a thread")
    if threading.current_thread() is thread:
        raise RuntimeError(f"{what} cannot wait for itself")
    seconds = float(timeout)
    if not seconds > 0.0:
        raise ValueError("worker stop timeout must be positive")
    thread.join(seconds)
    if thread.is_alive():
        raise TimeoutError(
            f"{what} did not stop within {seconds:g}s after being told to; "
            "it is still holding whatever it was using"
        )


__all__ = [
    "CleanupReport",
    "join_worker",
    "run_cleanup_steps",
    "WORKER_STOP_SECONDS",
]
