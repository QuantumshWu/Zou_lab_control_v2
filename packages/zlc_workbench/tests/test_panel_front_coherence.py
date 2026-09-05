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


def _submit_now(work):
    from concurrent.futures import Future

    completed = Future()
    try:
        completed.set_result(work())
    except BaseException as error:
        completed.set_exception(error)
    return completed


class _Host:
    """The narrowest host a port will accept: it only takes the input."""

    host_id = "host-1"

    def __init__(self) -> None:
        self.received: list[object] = []

    def update_data(self, plot_input: object):
        from concurrent.futures import Future

        self.received.append(plot_input)
        return Future()


def _stage_on(host):
    def replace_host(plot_input, _value, _publication):
        return host, host.update_data(plot_input)

    return replace_host


def test_a_panels_annotation_reaches_the_planes_coherent_front_set() -> None:
    """What the board DECLARES to the plane must include the annotation.

    The plane holds a signal at its source's shot only for signals in the
    declared set; one left out floats at its own latest publication, which
    on a free-running camera is a different cycle every tick.  This asserts
    the set the plane actually receives, not a helper that computes it.
    """

    from zlc_runtime.presentation import BoardScheduler
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
            return SignalFront({})

        def direct_parent_publications(self, _publication):
            return ()

        def follower_edges(self):
            return frozenset()

        def latest_publication(self, _signal):
            return None

    host = _Host()
    port = PlotPanelPort(
        "panel-1",
        "@logic/camera/frames",
        display_interval_ms=100,
        submit_projection=_submit_now,
        replace_host=_stage_on(host),
        companion_signals=lambda _target: ("@logic/occupancy/occupied",),
    )
    plane = _Plane()
    scheduler = BoardScheduler(
        plane,
        HarmonicClock((100, 200, 400, 800)),
        SurfaceBatchArbiter(_Sink()),
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
    from test_signal_front import _publication

    publication = _publication(
        "camera",
        "generation",
        1,
        "camera/frames",
    )
    value = publication.value("camera/frames")
    assert value is not None

    class _Reached(Exception):
        """Raised once the projection has been handed its front."""

    def project(
        _value: object,
        _publication: object,
        given: object,
        _target: object,
    ) -> object:
        seen.append(given)
        raise _Reached

    host = _Host()
    port = PlotPanelPort(
        "panel-1",
        "camera/frames",
        display_interval_ms=100,
        submit_projection=_submit_now,
        replace_host=_stage_on(host),
        project_input=project,
    )

    update = port.prepare(value, publication, front)
    assert update is not None
    try:
        update.future.result()
    except _Reached:
        pass

    assert seen == [front], "the projection was not given the prepared front"
