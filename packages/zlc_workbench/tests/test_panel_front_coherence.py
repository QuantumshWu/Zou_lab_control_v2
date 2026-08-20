"""A panel reads everything it draws out of ONE front.

An image and the judgement annotating it are two signals of one shot.  The
plane already guarantees they freeze together -- but only for signals the
board DECLARES it displays.  An annotation left out of that declaration
floats at its own latest publication, so on a free-running camera the rings
describe a cycle the picture never showed.  These guards pin both halves:
the declaration, and the reading.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

from zlc_runtime.presentation import HarmonicClock, SurfaceBatchArbiter
from zlc_workbench.presentation import PlotPanelPort


class _Host:
    """The narrowest host a port will accept: it only takes the input."""

    host_id = "host-1"

    def __init__(self) -> None:
        self.received: list[object] = []

    def update_data(self, plot_input: object):
        from concurrent.futures import Future

        self.received.append(plot_input)
        return Future()


def test_a_panels_annotation_reaches_the_planes_coherent_front_set() -> None:
    """What the board DECLARES to the plane must include the annotation.

    The plane holds a signal at its source's shot only for signals in the
    declared set; one left out floats at its own latest publication, which
    on a free-running camera is a different cycle every tick.  This asserts
    the set the plane actually receives, not a helper that computes it.
    """

    from zlc_runtime.presentation import BoardScheduler, OwnerChannels
    from zlc_runtime.plane import SignalFront

    class _Sink:
        def request_owner_wake(self) -> None:
            return None

    class _Plane:
        def __init__(self) -> None:
            self.declared: frozenset[str] = frozenset()

        def set_front_signals(self, names) -> None:
            self.declared = frozenset(names)

        def freeze(self):
            return SignalFront({}, {}, {})

        def direct_parent_publications(self, _publication):
            return ()

        def follower_edges(self):
            return frozenset()

    port = PlotPanelPort(
        "panel-1",
        "@logic/camera/frames",
        _Host(),
        display_interval_ms=100,
        companion_signals=lambda: ("@logic/occupancy/occupied",),
    )
    plane = _Plane()
    scheduler = BoardScheduler(
        plane,
        HarmonicClock((100, 200, 400, 800)),
        SurfaceBatchArbiter(OwnerChannels(_Sink())),
        lambda: (port,),
    )

    scheduler.on_tick()

    assert plane.declared == {
        "@logic/camera/frames",
        "@logic/occupancy/occupied",
    }, "the annotation never reached the plane's coherent set"


def test_the_projection_is_handed_the_front_it_was_prepared_from() -> None:
    """``prepare`` gives the projection the exact freeze, not a lookup key.

    The projection used to receive only one publication, so an annotation on
    a different publication had to be FETCHED -- and the only thing available
    to fetch was the plane's latest.
    """

    seen: list[object] = []
    front = object()

    class _Reached(Exception):
        """Raised once the projection has been handed its front."""

    def project(_value: object, _publication: object, given: object) -> object:
        seen.append(given)
        raise _Reached

    port = PlotPanelPort(
        "panel-1",
        "@logic/camera/frames",
        _Host(),
        display_interval_ms=100,
        project_input=project,
    )

    try:
        port.prepare(_Value(), _Publication(), front)
    except _Reached:
        pass

    assert seen == [front], "the projection was not given the prepared front"


class _Value:
    def __init__(self) -> None:
        self.snapshot = _Snapshot()
        self.name = "@logic/camera/frames"


class _Snapshot:
    class ref:  # noqa: D106 - the port only reads a revision
        class revision:
            value = 1

        class stream_generation:
            value = "g"


class _Publication:
    event_ref = object()

    def value(self, _name: str) -> None:
        return None
