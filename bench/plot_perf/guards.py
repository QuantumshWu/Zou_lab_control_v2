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

    def __init__(self, app, presenter, *, drive_timer: bool = True):
        from PyQt5 import QtCore

        self._app = app
        self._QtCore = QtCore
        self.interval_ms = int(presenter.board.base_interval_ms)
        self._timer = None
        self._presenter = presenter
        self._drive_timer = drive_timer

    def __enter__(self) -> "ProductBeat":
        if self._drive_timer:
            from zlc_workbench.board import attach_qt

            self._timer = attach_qt(
                self._presenter.beat,
                interval_ms=self.interval_ms,
                board=self._presenter.board,
            )
        return self

    def __exit__(self, *_exc) -> None:
        if self._timer is not None:
            self._timer.stop()

    def run(self, seconds: float, tick=None) -> float:
        """Let the console run for a wall-clock window; return what elapsed.

        ``tick`` observes after Qt has processed events, before it sleeps.
        The GUI runs its ordinary event loop throughout the measurement.
        """

        started = time.perf_counter()
        self.run_until(lambda: False, seconds, tick=tick)
        return time.perf_counter() - started

    def run_until(self, predicate, timeout: float, tick=None) -> bool:
        """Drive the console until a predicate holds, or time runs out.

        Observe each completed Qt event batch without a polling timer. The
        one-shot bounds this wait; the product timer still owns refresh.
        """

        from math import ceil

        if timeout <= 0:
            return False
        loop = self._QtCore.QEventLoop()
        end = self._QtCore.QTimer()
        end.setSingleShot(True)
        end.setTimerType(self._QtCore.Qt.PreciseTimer)
        end.timeout.connect(loop.quit)
        dispatcher = self._QtCore.QAbstractEventDispatcher.instance()
        ready = False
        errors = []

        def observe() -> None:
            nonlocal ready
            try:
                if tick is not None:
                    tick()
                ready = bool(predicate())
            except BaseException as error:
                # Let the caller handle the same exception as before; it
                # must not escape through a Qt signal and abort the process.
                errors.append(error)
            if ready or errors:
                loop.quit()

        dispatcher.aboutToBlock.connect(observe)
        try:
            end.start(ceil(timeout * 1000))
            observe()
            if not ready and not errors:
                loop.exec_()
        finally:
            end.stop()
            dispatcher.aboutToBlock.disconnect(observe)
        if errors:
            raise errors[0]
        return ready


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
        session.sequencer.fire(run_repeats=0, scan_repeats=1)
    except RuntimeError as error:
        if "still playing" not in str(error):
            raise


class SourceRate:
    """Whether the producer delivered, asked of the plane and not assumed.

    ``camera_measurement`` with ``repeat: 0`` keeps the virtual camera
    producing whether or not a pulse is fired: about twenty distinct
    revisions a second with nothing else happening.  Several probes were
    labelled "no producer" while one was running, and the frames they timed
    were rendering genuinely new data.  Ask, then claim.
    """

    def __init__(self, session, signal_name: str):
        self._session = session
        self._name = str(signal_name)

    def revision(self):
        """This signal's current revision, or None while it has published none."""

        publication = self._session.signal_plane.latest_publication(self._name)
        if publication is None:
            return None
        return int(publication.value(self._name).snapshot.ref.revision.value)


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
