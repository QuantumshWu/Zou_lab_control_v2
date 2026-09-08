"""Where a scan point's value comes from.

Both engines do the same three things at every point -- take a value, write it
into the plan's slot, move on -- and differ only in where that value comes
from.  Two answers exist on this bench.

A SCAN NODE WATCHES SOMEBODY ELSE'S SIGNAL.  The operator says which one, and
the value of a point is whatever that signal publishes next: camera frames, a
processor's counts, anything live.  That is the whole point of the scan nodes
-- they scan a knob against a quantity the bench is already producing.

A TASK THAT OWNS ITS CAMERA TAKES THE FRAMES ITSELF.  Release-recapture is not
"scan t_off against whatever happens to be running": the two probe windows of
one cycle ARE the measurement, so the Task holds the camera, arms it for the
whole table, and reads the cycles the fired program triggers.  That source is
a camera capture wearing this protocol, so it lives with the camera
(``camera_measurement.measurement.CameraCycleSource``) -- no scan node uses
it, and the scan package is what the scan NODES stand on.

Both are sources: open before the board is loaded, validate the actual played
program before LOAD, arm just before the fire, return one value plus its exact
causal publication per played point, and close at the end.  A Task-owned camera
has no upstream publication and returns ``None`` for that half.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from threading import Event, RLock

from zlc_data import canonical_text

from zlc_runtime import SignalPublication, SignalValue
from zlc_runtime.streams import SourceGenerationEnded, StreamEndedEarly


def check_cancelled(context: object) -> None:
    """Leave now if the operator has pressed Stop.

    One sentence for both engines and for the wait below: a scan that ends
    because it was stopped says so the same way wherever it noticed.
    """

    if context.cancel_requested():
        raise RuntimeError("the scan was cancelled")


def settle(context: object, seconds: float) -> None:
    """Give the bench its authored settle time, and stay stoppable meanwhile.

    The settle is the longest thing either engine does between SAFE and the
    next fire, and one that sleeps it whole cannot see a Stop that arrives
    during it: the next point was tuned, loaded and fired before the flag
    was read.  Slept in slices with the flag read between them, Stop ends
    the settle within a slice and nothing new reaches the bench.
    """

    deadline = time.monotonic() + float(seconds)
    while True:
        check_cancelled(context)
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            return
        time.sleep(min(remaining, 0.1))


def wait_for_report(sequencer: object, context: object) -> object:
    """Wait for the board's own report of the shot, and stay stoppable meanwhile.

    Stop is a cooperative flag; a wait that cannot see it is a wait an operator
    cannot end.  So the bench is asked in slices and the flag is read between
    them: pressing Stop leaves this within one slice, the caller's ``finally``
    drives the outputs safe, and nothing waits for the rest of the table to
    play out -- ending a scan is not the same act as letting it finish.

    Both engines wait here.  Passing ``None`` to ``wait_done`` is what made a
    stopped seamless scan run to the end of its table (measured: Stop at
    4.018 s, terminal at 8.469 s, the whole difference asleep inside the
    streamer), while its sibling had this loop written out beside it.

    The report comes back whole, fault and all: what a fault MEANS for the
    data is the caller's question (a scan point is lost; a Task that counted
    its own camera frames may know better), and only the waiting is shared.

    NOTHING is waited for unconditionally.  ``wait_done`` answers None for two
    different situations and only one of them is "not yet": a board that is no
    longer firing has no report left to give and says so IMMEDIATELY, which
    turned this -- the one wait in a shot with no deadline of its own -- into
    a spin that never ended and never said anything.  Every other wait in a
    shot batch is bounded and names its own failure; so is this one now.
    """

    while True:
        check_cancelled(context)
        report = sequencer.wait_done(0.1)
        if report is not None:
            return report
        state = sequencer.snapshot()
        if not isinstance(state, Mapping):
            raise TypeError("sequencer snapshot must be a mapping")
        if not state.get("firing"):
            raise RuntimeError(
                "the board stopped without reporting this shot: it is no "
                "longer firing and its report has already been taken or was "
                "never written, so nothing further is coming for this batch"
            )


def wait_for_board(sequencer: object, context: object) -> None:
    """Wait for the board and refuse any shot it reports as faulted."""

    report = wait_for_report(sequencer, context)
    if report.fault:
        raise RuntimeError(f"the pulse failed: {report.fault}")


def watched_signal_source(
    signal_plane: object,
    source_signal: str,
) -> "PublishedSignalSource":
    """Watch the declared source, including one with no generation/data yet.

    Authoring checks the declaration's contract. A configured panel fit has
    no runtime output until its first real frame is fitted, so the source may
    wait for that publication while Scan fires its own pulse. It never starts
    a camera or creates an initial fit value.
    """

    return PublishedSignalSource(signal_plane, source_signal)


class PublishedSignalSource:
    """The point's value is the next publication of a signal somebody runs.

    A producer commits publications independently of display.  This source
    follows that exact stream directly; it never pumps the presentation front
    and therefore cannot make a scan's cadence depend on an open panel.

    A source that restarts mid-scan is not a source with a gap; it is a
    different generation, and the scan says so instead of stitching.
    """

    def __init__(
        self,
        signal_plane: object,
        signal_name: str,
    ) -> None:
        self.signal_plane = signal_plane
        self.signal_name = canonical_text(signal_name, "scan source signal")
        self._tap = None
        self._lock = RLock()
        self._arrival = Event()
        self._opened = False
        self._unsubscribe = None

    def open(self, context: object, *, cycles: int) -> None:
        """Subscribe before the board is loaded, so nothing played is missed.

        An existing live generation binds now; a not-yet-published fit route
        binds on its first real arrival. A sealed old route is not replayed.
        Once bound, generation end stays loud through StreamEndedEarly.
        """

        del context, cycles
        self.close()
        with self._lock:
            self._opened = True
            self._arrival.clear()
            self._unsubscribe = self.signal_plane.subscribe_publications(self._bind_arrival)
            self._bind_arrival(replay=False)

    def _bind_arrival(self, *, replay: bool = True) -> None:
        # Called synchronously at publication arrival, not by polling latest:
        # the first event enters the existing ordered tap before another can
        # replace it. The same lock closes the arrival/close race.
        with self._lock:
            if not self._opened or self._tap is not None:
                return
            try:
                _baseline, tap = self.signal_plane.follow_publications(
                    self.signal_name, replay=replay,
                )
            except (LookupError, SourceGenerationEnded):
                return  # No live route yet; never adopt a sealed old fit.
            self._tap = tap
            self._arrival.set()

    def validate(
        self,
        program: object,
        table: object = None,
        *,
        run_repeats: int = 1,
        scan_repeats: int = 1,
    ) -> None:
        """A watched signal imposes no pulse/camera compatibility constraint."""

        del program, table, run_repeats, scan_repeats

    def arm(self) -> None:
        """Everything published so far belongs to the world before this point."""

        self.discard_pending()

    def discard_pending(self) -> None:
        """Discard every publication completed before a sampling boundary."""

        with self._lock:
            if not self._opened:
                raise RuntimeError("the scan source was not opened")
            tap = self._tap
        if tap is None:
            return
        while True:
            try:
                tap.next(0.0)
            except TimeoutError:
                return
            except StreamEndedEarly:
                raise RuntimeError(
                    "the source signal restarted during the scan"
                ) from None

    def next_value(
        self, context: object
    ) -> tuple[SignalValue, SignalPublication]:
        """The next value and the exact publication the scan consumed."""

        while True:
            if context.cancel_requested():
                raise RuntimeError("the scan was cancelled")
            with self._lock:
                if not self._opened:
                    raise RuntimeError("the scan source was not opened")
                tap = self._tap
            if tap is None:
                self._arrival.wait(0.1)
                continue
            try:
                publication = tap.next(0.1)
            except TimeoutError:
                continue
            except StreamEndedEarly:
                raise RuntimeError(
                    "the source signal restarted during the scan"
                ) from None
            value = publication.value(self.signal_name)
            if not isinstance(value, SignalValue):
                raise RuntimeError("the source publication lost the selected signal")
            return value, publication

    def close(self) -> None:
        with self._lock:
            self._opened = False
            tap, self._tap = self._tap, None
            unsubscribe, self._unsubscribe = self._unsubscribe, None
            self._arrival.set()
        if unsubscribe is not None:
            unsubscribe()
        if tap is not None:
            tap.close()

    def describe(self) -> dict[str, object]:
        return {"source_signal": self.signal_name}


__all__ = [
    "PublishedSignalSource",
    "check_cancelled",
    "settle",
    "wait_for_board",
    "wait_for_report",
    "watched_signal_source",
]
