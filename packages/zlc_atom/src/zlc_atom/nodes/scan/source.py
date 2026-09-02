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

from zlc_runtime import SignalPublication, SignalValue
from zlc_runtime.streams import SourceGenerationEnded, StreamEndedEarly


def check_cancelled(context: object) -> None:
    """Leave now if the operator has pressed Stop.

    One sentence for both engines and for the wait below: a scan that ends
    because it was stopped says so the same way wherever it noticed.
    """

    if context.cancel_requested():
        raise RuntimeError("the scan was cancelled")


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
    """

    while True:
        check_cancelled(context)
        report = sequencer.wait_done(0.1)
        if report is not None:
            return report


def wait_for_board(sequencer: object, context: object) -> None:
    """Wait for the board and refuse any shot it reports as faulted."""

    report = wait_for_report(sequencer, context)
    if report.fault:
        raise RuntimeError(f"the pulse failed: {report.fault}")


def watched_signal_source(
    signal_plane: object,
    source_signal: str,
) -> "PublishedSignalSource":
    """The armed-signal gate both scan engines start from.

    A scan takes each point's value from a signal somebody else runs, so
    that chain must be ARMED -- a live generation -- but not necessarily
    publishing yet: an externally triggered camera publishes nothing until
    this scan's own pulse fires, and zero frames before the first trigger
    is exactly the aligned start.  What stays refused is a chain that is
    not running at all, named so the operator starts it -- the scan never
    starts anybody's camera and never judges frame alignment; both facts
    belong to the chain's owner and to the trigger wiring.
    """

    name = str(source_signal)
    if not signal_plane.is_generation_live(name):
        raise ValueError(
            f"the scan watches {name!r} for each point's value, and its "
            "producer chain is not armed on this bench -- start the "
            "measurement that publishes it (its pulse may stay stopped: "
            "the scan fires its own)"
        )
    return PublishedSignalSource(signal_plane, name)


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
        self.signal_name = str(signal_name)
        self._tap = None

    def open(self, context: object, *, cycles: int) -> None:
        """Subscribe before the board is loaded, so nothing played is missed.

        The generation followed is whatever is armed NOW.  It need not have
        published: an externally triggered chain publishes nothing until
        this scan's own pulse fires its first trigger, and that silence is
        the cleanest possible start -- frame one IS point one.  A
        generation that ends afterwards stays loud through the tap's
        ``StreamEndedEarly``.
        """

        del context, cycles
        try:
            _baseline, tap = self.signal_plane.follow_publications(
                self.signal_name,
                replay=False,
            )
        except SourceGenerationEnded:
            raise RuntimeError(
                "the source signal's generation ended before the scan began"
            ) from None
        self._tap = tap

    def _require_tap(self):
        tap = self._tap
        if tap is None:
            raise RuntimeError("the scan source was not opened")
        return tap

    def validate(
        self,
        program: object,
        table: object = None,
        *,
        cycles: int | None = None,
    ) -> None:
        """A watched signal imposes no pulse/camera compatibility constraint."""

        del program, table, cycles

    def arm(self) -> None:
        """Everything published so far belongs to the world before this point."""

        self.discard_pending()

    def discard_pending(self) -> None:
        """Discard every publication completed before a sampling boundary."""

        tap = self._require_tap()
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

        tap = self._require_tap()
        while True:
            if context.cancel_requested():
                raise RuntimeError("the scan was cancelled")
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
        tap, self._tap = self._tap, None
        if tap is not None:
            tap.close()

    def describe(self) -> dict[str, object]:
        return {"source_signal": self.signal_name}


__all__ = ["PublishedSignalSource", "watched_signal_source"]
