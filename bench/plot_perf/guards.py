"""The measurement mistakes this bench has already made, as code.

Every guard here exists because a number was reported that turned out to be
about the harness rather than the product.  Each one names the failure it
prevents; read the docstring before deciding to pass it a waiver.

None of these reimplement product behaviour.  They ASK the product objects
what they are doing -- the renderer for its density, the board for its
interval, the plane for its revisions -- so a change in product logic moves
the guard with it instead of silently invalidating it.
"""
from __future__ import annotations

import time


class HarnessError(AssertionError):
    """The harness, not the product, is what this number would describe."""


# ---------------------------------------------------------------- density
def display_density(renderer) -> dict:
    """The pixel count this renderer is actually working on.

    Offscreen Qt hands out a small surface at device pixel ratio 1.  A 4x4
    panel measured there is 826x609 -- one NINTH of the 2478x1827 the real
    display gives at DPR 3 -- and every pixel-bound seam comes back three
    to nine times too cheap.  A whole per-kind matrix was collected that
    way before anyone noticed the header.

    Returns the facts; :func:`require_real_density` is the assertion.
    """

    figure = renderer.figure
    return {
        "figure_px": (
            int(round(float(figure.bbox.width))),
            int(round(float(figure.bbox.height))),
        ),
        "device_pixel_ratio": float(renderer.plan.device_pixel_ratio),
        "dpi": float(renderer.plan.dpi),
        "megapixels": round(
            float(figure.bbox.width) * float(figure.bbox.height) / 1e6, 2
        ),
    }


def require_real_density(renderer, *, minimum_ratio: float = 2.0) -> dict:
    """Refuse to report timings taken on a toy surface."""

    facts = display_density(renderer)
    if facts["device_pixel_ratio"] < minimum_ratio:
        raise HarnessError(
            "device pixel ratio %s: this is an offscreen or low-density "
            "surface (%s px), and pixel-bound seams measured here are not "
            "the operator's. Run without QT_QPA_PLATFORM=offscreen, or pass "
            "minimum_ratio=1.0 to say you meant it."
            % (facts["device_pixel_ratio"], facts["figure_px"])
        )
    return facts


# ------------------------------------------------------------------ scope
def require_panels(presenter, expected: int) -> tuple[str, ...]:
    """Assert the console holds exactly the panels you think it does.

    A console always carries more than the panel under test.  Timing seams
    on the RENDERER CLASS therefore counts every panel at once, and a curve
    panel's profile came back carrying an image panel's work -- the wrong
    seam looked like the bottleneck.  :func:`bench.plot_perf.probe.watch`
    binds to one instance; this says how many instances there are, so the
    mistake is visible even when the taps are right.
    """

    ids = tuple(presenter.panels)
    if len(ids) != expected:
        raise HarnessError(
            "console holds %d panels %s, not %d. Either close the others or "
            "bind every probe to the one renderer under test."
            % (len(ids), ids, expected)
        )
    return ids


def require_distinct_labels(labels) -> tuple[str, ...]:
    """Assert no two panels report under the same name.

    Counting panels is not enough: two panels of the SAME KIND passed that
    check and then shared a probe prefix, merging both renderers' self-times
    into one key and dividing the sum by one panel's frame count.  A name is
    what every report joins on, so it has to identify a panel.
    """

    names = tuple(labels)
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise HarnessError(
            "panels report under duplicate names %s: every seam roll-up and "
            "frame count joins on this name, so it must identify one panel."
            % (duplicates,)
        )
    return names


# ------------------------------------------------------------------- beat
class ProductBeat:
    """Drive the console's beat the way the product drives it.

    The console beats on a QTimer at the board's own interval.  A bench that
    calls ``presenter.beat()`` in a tight loop instead runs it about fifty
    times faster, and then the harness IS the load: one such loop reported
    ten busy cores for a console that was idle.

    Used as a context manager so the timer cannot outlive the window.
    """

    def __init__(self, app, presenter):
        from PyQt5 import QtCore

        self._app = app
        self._QtCore = QtCore
        self.interval_ms = int(presenter.board.base_interval_ms)
        self._timer = QtCore.QTimer()
        self._timer.setInterval(self.interval_ms)
        self._timer.timeout.connect(presenter.beat)

    def __enter__(self) -> "ProductBeat":
        self._timer.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._timer.stop()

    def run(self, seconds: float, tick=None) -> float:
        """Let the console run for a wall-clock window; return what elapsed.

        ``tick`` is called on every pass so a front counter can SEE the
        frames.  Polling only at the end reports one frame for a window that
        drew fifty, which is what a bench that forgot this said.
        """

        started = time.perf_counter()
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self._app.processEvents(self._QtCore.QEventLoop.AllEvents, 20)
            if tick is not None:
                tick()
            time.sleep(0.001)
        return time.perf_counter() - started

    def run_until(self, predicate, timeout: float, tick=None) -> bool:
        """Drive the console until a predicate holds, or time runs out.

        Same pacing as :meth:`run`, because there is only one rate at which
        this console is driven and a bench that has two of them is a bench
        that measures whichever one it happened to be in.
        """

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self._app.processEvents(self._QtCore.QEventLoop.AllEvents, 20)
            if tick is not None:
                tick()
            if predicate():
                return True
            time.sleep(0.001)
        return False


# --------------------------------------------------------------- producer
def free_running(session) -> None:
    """Start acquisition the way the product does.

    The product fires the sequencer free-running and the device produces on
    its own thread.  A bench that instead calls the BLOCKING ``session.fire``
    from a Qt timer puts the producer on the GUI thread, where it competes
    with the very rendering being measured -- that fixture is where a
    reported "93 ms per drag move" came from, and the real figure was less
    than half of it.
    """

    # Idempotent in both halves.  A bench measures several things in one
    # process and asks for acquisition before each; the second ask would
    # otherwise raise DeviceUseBusy against the claim the first one took,
    # and the second fire would raise against the pulse already playing.
    # Already running is the state this function exists to reach.
    try:
        session._acquire_pulse_device()
    except Exception:
        pass
    try:
        session.sequencer.fire(cycles=None)
    except RuntimeError as error:
        if "still playing" not in str(error):
            raise


class SourceRate:
    """How fast a signal is actually publishing, measured not assumed.

    ``camera_measurement`` with ``repeat: 0`` keeps the virtual camera
    producing whether or not a pulse is fired: about twenty distinct
    revisions a second with nothing else happening.  Several probes were
    labelled "no producer" while one was running, and the frames they timed
    were rendering genuinely new data.  Ask, then claim.
    """

    def __init__(self, session, signal_name: str):
        self._session = session
        self._name = str(signal_name)

    def _revision(self):
        value = self._session.signal_plane.freeze().value(self._name)
        snapshot = getattr(value, "snapshot", value)
        ref = getattr(snapshot, "ref", None)
        return None if ref is None else int(ref.revision.value)

    def measure(self, app, seconds: float) -> dict:
        seen = set()
        started = time.perf_counter()
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            app.processEvents()
            revision = self._revision()
            if revision is not None:
                seen.add(revision)
            time.sleep(0.002)
        elapsed = time.perf_counter() - started
        return {
            "signal": self._name,
            "distinct_revisions": len(seen),
            "per_second": round(len(seen) / elapsed, 1),
            "quiet": len(seen) <= 1,
        }


def require_quiet(rate: dict) -> dict:
    """Assert a "no producer" window really had none."""

    if not rate["quiet"]:
        raise HarnessError(
            "%s published %d revisions (%.1f/s) during a window that was "
            "meant to be quiet: whatever was timed includes rendering real "
            "new data."
            % (rate["signal"], rate["distinct_revisions"], rate["per_second"])
        )
    return rate


# --------------------------------------------------------------- gestures
def require_effect(before, after, what: str):
    """Assert a gesture changed the thing it was supposed to change.

    Synthesised pointer calls into ``host._pointer_event`` build the gesture
    but silently drop its moves: a middle-button orbit reported an
    ``_OrbitGesture`` and ``height_bars_dragging`` true, and twelve moves
    later the camera had not turned.  Only real QMouseEvents on the widget
    drive it.  A bench cannot tell the difference by looking at latency, so
    it has to look at the QUANTITY THE GESTURE OWNS -- the camera angle, the
    committed region, the view limits -- before and after.
    """

    if before == after:
        raise HarnessError(
            "the gesture left %s unchanged (%r): it was not delivered. "
            "Send real QMouseEvents to the card's widget rather than "
            "calling the host's pointer entry point." % (what, before)
        )
    return after


def committed_region(panel) -> tuple:
    """The region a panel has actually stored, as comparable numbers."""

    state = panel.state.selector or {}
    return tuple(
        (
            str(item.get("domain")),
            round(float(item["lower"]), 6),
            round(float(item["upper"]), 6),
        )
        for item in state.get("ranges", ())
    )


def applied(answer, what: str):
    """Assert a state change was ACCEPTED, not silently refused.

    ``update_panel_state`` takes a fixed vocabulary and returns whether it
    took the patch.  A bench once sent a field that is not in it, never
    looked at the answer, and read the resulting stale surface as a product
    defect.
    """

    if answer is False or answer is None:
        raise HarnessError(
            "%s was refused by the product API; the surface still holds what "
            "it held before." % what
        )
    return answer
