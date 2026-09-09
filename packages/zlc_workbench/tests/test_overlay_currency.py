"""An annotation belongs to the picture it describes.

Both guards here are about the gap between staging a render and running
it.  A panel's overlay arrives as its own signal of the same shot, so the
port stages the pair and draws it a moment later -- and in that moment
the next camera frame can land, or the operator can turn the overlay off.

Synthetic publications on purpose: a virtual installation per test is
what this suite is already at the ceiling of, and none of this needs a
camera.
"""

from __future__ import annotations

import os
from concurrent.futures import Future
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

from test_signal_front import _publication  # type: ignore[import-not-found]

from zlc_runtime.plane import SignalFront
from zlc_workbench.presentation import PlotPanelPort

SIGNAL = "camera/frames"
COMPANION = "occupancy/occupied"


@pytest.fixture
def panels():
    """Every port this file opens is closed when its test ends."""

    opened: list = []
    try:
        yield opened
    finally:
        for port, deferred in opened:
            port.close()
            deferred.queued.clear()


def _operation(spec: str = "test"):
    return SimpleNamespace(value=SimpleNamespace(spec=spec))


class _Deferred:
    """A projection submitter that runs nothing until told to.

    The race this file is about lives BETWEEN preparing a render and
    running it, so the test has to be able to stand in that gap.
    """

    def __init__(self) -> None:
        self.queued: list[tuple[object, Future]] = []

    def __call__(self, work):
        future: Future = Future()
        self.queued.append((work, future))
        return future

    def run(self, index: int = 0) -> Future:
        work, future = self.queued.pop(index)
        # A real executor never runs work whose future was cancelled
        # while it waited in the queue.
        if future.cancelled():
            return future
        try:
            future.set_result(work())
        except BaseException as error:  # noqa: BLE001 - mirrored to the caller
            future.set_exception(error)
        return future


class _RevisionHost:
    """A host with the real one's rule: revisions only ever increase."""

    def __init__(self, revision: int) -> None:
        self.host_id = object()
        self.revision = revision
        self.inputs: list[object] = []
        self.overlays: list[object] = []
        self.refusals: list[str] = []

    def update_data(self, plot_input, **_render):
        snapshot = getattr(plot_input, "snapshot", plot_input)
        revision = int(snapshot.block.revision.value)
        if revision <= self.revision:
            message = (
                f"data revision must increase: {revision} <= {self.revision}"
            )
            self.refusals.append(message)
            raise ValueError(message)
        self.revision = revision
        self.inputs.append(plot_input)
        future: Future = Future()
        future.set_result(_operation("data"))
        return future

    def configure(self, **changes):
        if "image_overlay" in changes:
            self.overlays.append(changes["image_overlay"])
        future: Future = Future()
        future.set_result(_operation("overlay"))
        return future


def _mounts(host):
    def replace_host(plot_input, _value, _publication, _target):
        return host, host.update_data(plot_input)

    return replace_host


def _shot(sequence: int, companion_sequence: int):
    """One shot: the picture, and the annotation describing it."""

    picture = _publication("camera", "run-1", sequence, SIGNAL)
    annotation = _publication(
        "occupancy", "run-1", companion_sequence, COMPANION
    )
    front = SignalFront(
        {
            SIGNAL: picture.value(SIGNAL),
            COMPANION: annotation.value(COMPANION),
        },
        {SIGNAL: picture, COMPANION: annotation},
    )
    return picture.value(SIGNAL), picture, front


def _overlay_projection():
    from zlc_plot.primitives import ImageFrame, ImagePointOverlay

    def project(primary, publication, exact_front, target):
        # The console's own rule: no overlay signal, no ImageFrame -- the
        # picture is the snapshot and nothing is drawn over it.
        records = ((publication, primary.event_record),)
        if not getattr(target, "overlay_signal", COMPANION):
            return primary.snapshot, records
        annotation = exact_front.publication(COMPANION)
        return (
            ImageFrame(
                primary.snapshot,
                ImagePointOverlay.empty(annotation.event_ref.sequence),
            ),
            (*records, (annotation, primary.event_record)),
        )

    return project


def test_finite_overlay_uses_the_exact_canonical_prefix_and_last_scope() -> None:
    from dataclasses import replace
    import numpy as np
    from data_factory import axis, repeat_domain, make_dataset_schema, make_snapshot
    from zlc_data import READOUT_EVENT, SITE, SPATIAL_X, SPATIAL_Y, DatasetSchema, DomainSpec, ValueSchema
    from zlc_plot import AxisRef, ImagePlot, FacetGridPlot, Reduction
    from zlc_plot.primitives import IMAGE_POINT_OVERLAY_GEOMETRY_RECORD, PointStatus, image_point_overlay_geometry
    from zlc_runtime import SignalDataPlane, DatasetCoverage, DatasetOutputDeclaration, LiveDatasetOutput
    from zlc_workbench.console import ConsolePresenter
    from test_fit_projection import _projection

    frames = axis("frame", role=READOUT_EVENT, values=(0, 1, 2))
    points = DomainSpec((3,), (frames,), ((0, 1, 2),))
    site = axis("site", role=SITE, values=(0, 1))
    image_schema = make_dataset_schema(repeat_domain(size=50), points,
        cell_axes=(axis("y", role=SPATIAL_Y, size=1), axis("x", role=SPATIAL_X, size=2)))
    status_schema = DatasetSchema(image_schema.repeat_domain, points,
        DomainSpec((2,), (site,)), ValueSchema.scalar(np.dtype(bool), "1"))
    geometry = image_point_overlay_geometry(make_snapshot(image_schema, np.zeros((50, 3, 1, 2)), revision=0),
        ((0.0, 0.0), (1.0, 0.0)), ("s0", "s1"), status_axis=site)
    image_output = DatasetOutputDeclaration("frames", "test.frames")
    status_output = DatasetOutputDeclaration("status", "test.status")
    image_node = SimpleNamespace(instance_id="camera", dataset_output_declarations=(image_output,),
                                 signal_key=lambda name: f"camera/{name}")
    status_node = SimpleNamespace(instance_id="status", dataset_output_declarations=(status_output,),
                                  signal_key=lambda name: f"status/{name}")
    plane = SignalDataPlane()
    presenter = object.__new__(ConsolePresenter)
    presenter.session = SimpleNamespace(signal_plane=plane)
    image_spec = ImagePlot(AxisRef.cell_data("x"), AxisRef.cell_data("y"))
    facet_spec = FacetGridPlot(AxisRef.point("frame"), image_spec)
    try:
        plane.begin_generation(image_node)
        plane.begin_generation(status_node)
        for index, origin in enumerate((0, 49)):
            event_repeat = repeat_domain(size=1)
            record = {"device_settings": {"camera": {"device_session_id": "camera-session",
                "epoch_ranges": [[index, index]], "mixed": False}}}
            plane.commit_live(image_node, {"frames": LiveDatasetOutput(image_output,
                make_snapshot(replace(image_schema, repeat_domain=event_repeat),
                    np.full((1, 3, 1, 2), float(origin)) + np.arange(3)[None, :, None, None], revision=index + 1),
                DatasetCoverage((index + 1) * 3, 150), canonical_schema=image_schema, cell_origin=(origin, 0))})
            image_publication = plane.latest_publication("camera/frames")
            values = np.asarray([[[True, False], [False, True], [True, True]]], dtype=bool)
            if index:
                values = ~values
            valid = np.ones((1, 3, 2), dtype=bool)
            valid[:, 2] = False
            plane.commit_live(status_node, {"status": LiveDatasetOutput(status_output,
                make_snapshot(replace(status_schema, repeat_domain=event_repeat), values,
                              revision=index + 1, validity=valid),
                DatasetCoverage((index + 1) * 3, 150), {IMAGE_POINT_OVERLAY_GEOMETRY_RECORD: geometry},
                canonical_schema=status_schema, cell_origin=(origin, 0), event_record=record)},
                worker_source=("camera/frames", image_publication))
            status_publication = plane.latest_publication("status/status")
            exact_front = SimpleNamespace(publication=lambda name: status_publication if name == "status/status" else image_publication)
            image = plane.current_dataset("camera/frames", image_publication)
            overlay, used_publication, used_record = presenter._image_point_overlay(
                exact_front, image_publication, "status/status", image, index + 1)
            assert used_publication is status_publication
            assert overlay.status is plane.current_dataset("status/status", status_publication)
            assert overlay.status.block.values.shape == (50, 3, 2)
            assert used_record == plane.current_dataset_view("status/status", status_publication)[1]
            assert used_record["device_settings"]["camera"]["epoch_ranges"] == ((0, index),)
            assert overlay.statuses_for(facet_spec, 0) is None, "Mean over repeats cannot invent a judgement"
            scope_zero = replace(facet_spec, scope=((AxisRef.repeat("repeat"), 0),))
            assert overlay.statuses_for(scope_zero, 0) == (PointStatus.OCCUPIED, PointStatus.EMPTY)
            last = replace(facet_spec, cell=replace(image_spec, reduction=Reduction.LAST))
            assert overlay.statuses_for(last, 0) == (
                None if not index else (PointStatus.EMPTY, PointStatus.OCCUPIED))
            assert overlay.statuses_for(last, 2) is None, "invalid latest cell cannot reuse an earlier judgement"
            scoped_last = replace(last, scope=((AxisRef.repeat("repeat"), 0),))
            assert overlay.statuses_for(scoped_last, 1) == (PointStatus.EMPTY, PointStatus.OCCUPIED)
            explicit_last = replace(facet_spec, scope=((AxisRef.repeat("repeat"), 49),))
            # The actual image payload, not merely two overlay interpreters:
            # Last and an explicit last-coordinate Scope must name one picture.
            last_payload = _projection(last, snapshot=image)._payload
            scoped_payload = _projection(explicit_last, snapshot=image)._payload
            for cell, scoped_cell in zip(last_payload.cells, scoped_payload.cells, strict=True):
                np.testing.assert_array_equal(cell.payload.z.canonical, scoped_cell.payload.z.canonical)
                np.testing.assert_array_equal(cell.payload.valid, scoped_cell.payload.valid)
                assert overlay.statuses_for(last, cell.facet_value_canonical) == overlay.statuses_for(
                    explicit_last, scoped_cell.facet_value_canonical)
    finally:
        plane.close()


def test_an_overlay_for_a_replaced_picture_is_dropped_not_pushed(
    panels,
) -> None:
    """The shot it annotates is gone, so the annotation is gone with it.

    The port stages the picture and its annotation together and renders
    them a moment later, and in that moment the next camera frame can
    land.  The stale pair was then pushed as DATA -- the host holds a
    newer revision, revisions only ever increase, and the refusal showed
    on the card as "data revision must increase" until the next shot.

    Nothing about the older shot is worth restoring: it is not on screen
    and never will be again.
    """

    pytest.importorskip("zlc_plot")
    value, publication, front = _shot(1, 101)
    host = _RevisionHost(0)
    deferred = _Deferred()
    port = PlotPanelPort(
        "panel",
        SIGNAL,
        initial_target=SimpleNamespace(overlay_signal=COMPANION),
        display_interval_ms=100,
        submit_projection=deferred,
        replace_host=_mounts(host),
        companion_signals=lambda _target: (COMPANION,),
        project_input=_overlay_projection(),
    )
    panels.append((port, deferred))

    mount = port.prepare(value, publication, front)
    assert mount is not None
    deferred.run()
    assert port.accept(mount, mount.future.result(timeout=0))

    # The occupancy for THIS shot is recomputed, and its render queues.
    _same, _same_publication, restated = _shot(1, 102)
    late = port.prepare(value, publication, restated)
    assert late is not None, "the annotation for the shown shot must stage"

    # Before it runs, the next camera frame lands and takes the screen.
    newer_value, newer_publication, newer_front = _shot(2, 201)
    fresh = port.prepare(newer_value, newer_publication, newer_front)
    assert fresh is not None
    deferred.run(1)
    assert port.accept(fresh, fresh.future.result(timeout=0))

    deferred.run(0)              # the annotation, landing too late
    try:
        operation = late.future.result(timeout=0)
    except BaseException:
        operation = None
    if operation is None:
        port.finish_unpresented(late)
    else:
        port.accept(late, operation)

    assert host.refusals == [], host.refusals
    assert port.last_error is None, port.last_error


def test_a_render_projected_under_a_revoked_setting_never_lands(
    panels,
) -> None:
    """Turning the overlay off must not be undone by a frame in flight.

    The operator's decision reaches the host as a configure, and the port
    keeps projecting frames from its own copy of the panel's target.  A
    frame projected under the OLD copy -- staged before the decision,
    rendered after it -- put the rings back, and with a slow producer
    they stayed there: the panel showed a setting the operator had
    already revoked.
    """

    pytest.importorskip("zlc_plot")
    with_overlay = SimpleNamespace(overlay_signal=COMPANION)
    without_overlay = SimpleNamespace(overlay_signal="")
    value, publication, front = _shot(1, 101)
    host = _RevisionHost(0)
    deferred = _Deferred()
    port = PlotPanelPort(
        "panel",
        SIGNAL,
        initial_target=with_overlay,
        display_interval_ms=100,
        submit_projection=deferred,
        replace_host=_mounts(host),
        companion_signals=lambda target: (
            (COMPANION,) if target.overlay_signal else ()
        ),
        project_input=_overlay_projection(),
    )
    panels.append((port, deferred))

    mount = port.prepare(value, publication, front)
    assert mount is not None
    deferred.run()
    assert port.accept(mount, mount.future.result(timeout=0))
    assert hasattr(host.inputs[-1], "overlay"), "the rings started on"

    # A newer frame is staged while the overlay is still on ...
    newer_value, newer_publication, newer_front = _shot(2, 201)
    inflight = port.prepare(newer_value, newer_publication, newer_front)
    assert inflight is not None

    # ... and the operator turns the overlay off before it renders.
    host.configure(image_overlay=None)
    port.retarget(without_overlay)
    assert host.overlays == [None]

    deferred.run()
    try:
        operation = inflight.future.result(timeout=0)
    except BaseException:
        operation = None
    if operation is None:
        port.finish_unpresented(inflight)
    else:
        port.accept(inflight, operation)
    assert host.overlays == [None], (
        "a frame projected under the revoked setting put the rings back"
    )

    # The next frame is projected under the decision that stands.
    again = port.prepare(newer_value, newer_publication, newer_front)
    assert again is not None
    deferred.run()
    assert port.accept(again, again.future.result(timeout=0))
    assert not hasattr(host.inputs[-1], "overlay"), (
        "the frame after the decision still carried the overlay"
    )


def test_a_missing_companion_waits_without_staging_a_partial_shot(
    panels,
) -> None:
    """A pending companion is not an invalid result for the new image."""

    pytest.importorskip("zlc_plot")
    picture = _publication("camera", "run-1", 1, SIGNAL)
    bare_front = SignalFront(
        {SIGNAL: picture.value(SIGNAL)}, {SIGNAL: picture}
    )
    host = _RevisionHost(0)
    deferred = _Deferred()

    port = PlotPanelPort(
        "panel",
        SIGNAL,
        initial_target=SimpleNamespace(overlay_signal=COMPANION),
        display_interval_ms=100,
        submit_projection=deferred,
        replace_host=_mounts(host),
        companion_signals=lambda target: (
            (COMPANION,) if target.overlay_signal else ()
        ),
        project_input=_overlay_projection(),
    )
    panels.append((port, deferred))

    update = port.prepare(picture.value(SIGNAL), picture, bare_front)
    assert update is None
    assert deferred.queued == []
    assert host.inputs == []
    assert COMPANION in port.waiting_condition

    value2, publication2, full_front = _shot(2, 202)
    complete = port.prepare(value2, publication2, full_front)
    assert complete is not None
    deferred.run()
    assert port.accept(complete, complete.future.result(timeout=0))
    assert port.waiting_condition == ""

    accepted = port.accepted_surface()
    picture3 = _publication("camera", "run-1", 3, SIGNAL)
    missing = SignalFront(
        {SIGNAL: picture3.value(SIGNAL)}, {SIGNAL: picture3}
    )
    assert port.prepare(picture3.value(SIGNAL), picture3, missing) is None
    assert port.accepted_surface() is accepted
    assert len(host.inputs) == 1
    assert deferred.queued == []
    assert COMPANION in port.waiting_condition

    # The operator explicitly disconnects the overlay.  The required group
    # changes immediately, even before a new accepted surface exists.
    port.retarget(SimpleNamespace(overlay_signal=""))
    assert port.front_signals == (SIGNAL,)
    unlinked = port.prepare(picture3.value(SIGNAL), picture3, missing)
    assert unlinked is not None
    deferred.run()
    assert port.accept(unlinked, unlinked.future.result(timeout=0))
    assert port.waiting_condition == ""
