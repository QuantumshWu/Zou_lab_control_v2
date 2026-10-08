from __future__ import annotations

import numpy as np
import pytest

from data_factory import (
    axis,
    make_dataset_schema,
    make_snapshot,
    mapped_domain_from_columns,
    repeat_domain,
)

from zlc_data import OwnedSnapshot
from zlc_plot import (
    AxisRef,
    CurvePlot,
    FacetGridPlot,
    HistogramPlot,
    ImagePlot,
    PlotSession,
    RasterPlotHost,
    curve,
)
from zlc_plot.fit import FacetFitBatchResult
from zlc_plot.primitives import ImageFrame, ImagePointOverlay, PointStatus
from zlc_plot.selectors import SelectorKind

def _snapshot(*, revision: int = 0, repeats: int = 1) -> OwnedSnapshot:
    x = np.arange(6, dtype=np.float64)
    schema = make_dataset_schema(
        repeat_domain(size=repeats),
        mapped_domain_from_columns({"x": x, "facet": np.repeat([0.0, 1.0], 3)}),
        dtype=np.float64,
    )
    values = np.tile(x, (repeats, 1))
    return make_snapshot(schema, values, revision=revision)

def _image_snapshot(*, indexed: bool = False) -> OwnedSnapshot:
    from zlc_data import PRIMARY_INDEX
    from zlc_data.snapshot_projection import PRIMARY_INDEX_AXIS_ID

    schema = make_dataset_schema(
        repeat_domain(size=1),
        (
            mapped_domain_from_columns(
                {"source index": [0]},
                ids={"source index": str(PRIMARY_INDEX_AXIS_ID)},
                roles={"source index": PRIMARY_INDEX},
            )
            if indexed
            else mapped_domain_from_columns({"sample": np.array([0.0])})
        ),
        cell_axes=(
            axis("row", size=2),
            axis("column", size=3),
        ),
        dtype=np.float64,
    )
    values = np.arange(6, dtype=np.float64).reshape(1, 1, 2, 3)
    return make_snapshot(schema, values, revision=0)

def test_convenience_api_requires_an_explicit_axis_domain() -> None:
    with pytest.raises(TypeError, match="explicit AxisRef"):
        curve(_snapshot(), "x")  # type: ignore[arg-type]

def test_session_replace_spec_reuses_the_existing_surface() -> None:
    session = PlotSession(_snapshot(), CurvePlot(AxisRef.point("x")))
    figure = session._renderer.figure
    try:
        session.replace_spec(HistogramPlot(), parameters={"bin_count": 12})
        assert session.spec == HistogramPlot()
        assert session._renderer.figure is figure
        assert session.display_state["bin_count"] == 12
    finally:
        session.close()

@pytest.mark.parametrize("entry", ("direct", "live", "configure", "process"))
def test_image_frames_replace_the_complete_layer_and_keep_new_run_overlay(entry) -> None:
    """Every data entry point replaces its point layer with the current frame."""

    from zlc_data import owned_snapshot_from_arrays
    from zlc_plot import RenderProcess

    base = _image_snapshot(indexed=True)
    first = owned_snapshot_from_arrays(
        base.block.schema,
        base.block.values,
        10,
        validity=base.block.validity,
        stream_generation="image-run-a",
    )
    bare = owned_snapshot_from_arrays(
        base.block.schema,
        base.block.values + 1.0,
        11,
        validity=base.block.validity,
        stream_generation="image-run-b",
    )
    second = owned_snapshot_from_arrays(
        base.block.schema, base.block.values + 2.0, 10,
        validity=base.block.validity, stream_generation="image-run-b",
    )
    overlay = ImagePointOverlay(
        10, np.asarray(((1.0, 0.5),)),
        static_statuses=(PointStatus.OCCUPIED,),
        paths_xy=np.asarray((((1.0, 0.5), (2.0, 1.0), (1.0, 1.0)),)),
    )
    incoming = ImagePointOverlay(
        0, np.asarray(((1.0, 0.5),)),
        static_statuses=(PointStatus.EMPTY,),
        paths_xy=np.asarray((((1.0, 0.5), (1.0, 0.5), (0.0, 1.0)),)),
    )
    spec = ImagePlot(AxisRef.cell_data("column"), AxisRef.cell_data("row"))
    parameters = {"show_image": entry != "process"}
    service = RenderProcess("image-frame-contract") if entry == "process" else None
    host = (
        service.build_host(ImageFrame(first, overlay), spec, parameters=parameters)
        if service is not None
        else RasterPlotHost.from_plot(ImageFrame(first, overlay), spec, parameters=parameters)
    )
    try:
        host.wait_for_front(timeout=30)

        def update(data):
            if entry == "direct":
                return host.dispatch(
                    lambda: host._require_session().update_data(data)
                ).result(timeout=30)
            if entry == "configure":
                return host.configure(data=data).result(timeout=30)
            return host.update_data(data).result(timeout=30)

        operation = update(ImageFrame(second, incoming))
        assert operation.front.identity.image_overlay_revision == 0
        assert host.describe_display().result(timeout=30).value.spec == spec
        assert host.front is not None
        assert host.front.identity.data_generation == "image-run-b"
        assert host.front.identity.data_revision == 10
        with_paths = operation.front.buffer.as_rgba().copy()

        operation = update(bare)
        assert operation.front.identity.data_revision == 11
        assert operation.front.identity.image_overlay_revision is None
        reference = RasterPlotHost.from_plot(ImageFrame(first, overlay), spec, parameters=parameters)
        try:
            reference.wait_for_front(timeout=30)
            next_frame = reference.update_data(ImageFrame(second, incoming)).result(timeout=30)
            np.testing.assert_array_equal(with_paths, next_frame.front.buffer.as_rgba())
            clean = reference.update_data(bare).result(timeout=30).front
            np.testing.assert_array_equal(
                operation.front.buffer.as_rgba(), clean.buffer.as_rgba(),
            )
        finally:
            reference.close(timeout=30)

    finally:
        host.close(timeout=30)
        if service is not None:
            service.close(timeout=30)

def test_ordered_image_paths_survive_figure_roundtrip_and_render(tmp_path) -> None:
    """An XY path is ordered geometry, not a sorted X-axis curve."""

    from dataclasses import replace
    from matplotlib.colors import to_rgba
    from matplotlib.path import Path as DrawPath
    from PIL import Image
    from zlc_data.figure_archive import read_archive
    from zlc_plot import read_figure_plot, save_figure_artifact
    from zlc_plot.errors import RevisionError

    paths = np.asarray((
        ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (1.0, 1.0), (0.0, 0.0)),
        ((2.0, 1.0),) * 5,
        ((1.0, 1.0), (2.0, 0.0), (2.0, 0.0), (2.0, 0.0), (2.0, 0.0)),
    ))
    overlay = ImagePointOverlay(
        0, paths[:, 0, :], point_ids=("moving", "stationary", "later start"), paths_xy=paths,
        static_statuses=(PointStatus.OCCUPIED,)*3,
    )
    assert not overlay.paths_xy.flags.writeable
    empty = ImagePointOverlay(0, np.empty((0, 2)), paths_xy=np.empty((0, 5, 2)))
    assert empty.paths_xy.shape == (0, 5, 2)
    spec = ImagePlot(AxisRef.cell_data("column"), AxisRef.cell_data("row"))
    original = ImageFrame(_image_snapshot(), overlay)
    image, archive = save_figure_artifact(
        tmp_path / "paths", plot_input=original, spec=spec,
        parameters={"show_image": False, "side_distribution": False}, size="2x2",
    )
    info, arrays, datasets = read_archive(archive)
    restored, recipe = read_figure_plot(info, arrays, datasets, "data")
    assert isinstance(restored, ImageFrame)
    np.testing.assert_array_equal(restored.snapshot.block.values, original.snapshot.block.values)
    np.testing.assert_array_equal(restored.overlay.paths_xy, paths)
    assert restored.overlay.point_ids == ("moving", "stationary", "later start")
    session = PlotSession(
        restored, recipe["spec"], parameters=recipe["parameters"], size=recipe["size"],
    )
    try:
        artists = session._renderer._artists
        joined = artists["image:point-paths"]
        np.testing.assert_array_equal(joined._zlc_point_path_inputs[1], paths)
        assert len(joined.get_paths()) == 1
        np.testing.assert_array_equal(joined.get_facecolors()[0],
            to_rgba(
                session._renderer.style.artists.point_occupied.color,
                session._renderer.style.artists.point_occupied.alpha))
        assert not artists["image"].get_visible()
        assert "image:point-path-ends" not in artists and "image:point-path-arrows" not in artists
        path_labels = [label.get_text() for label in artists["image:point-path-labels"] if label.get_visible()]
        assert "f2–3" in path_labels and "f0–4" not in path_labels
        assert "f4" in path_labels
        assert artists["image:point-path-timebase"].get_text() == "f0 → f4"
        coincident = [label.get_position() for label in artists["image:point-path-labels"]
                      if label.get_visible() and tuple(label.xy) == (1.0, 1.0)]
        assert len(coincident) == 1
        composed = session.rgba().copy()
        session._renderer.draw()
        # Same tolerance as the existing native/Agg image parity cases.
        assert np.max(np.abs(composed.astype(np.int16) - session.rgba().astype(np.int16))) <= 4
        session.set_parameter("show_image", True)
        assert artists["image"].get_visible()
        assert not np.array_equal(composed, session.rgba())
        session.set_parameter("show_image", False)
        np.testing.assert_array_equal(composed, session.rgba())
        session.set_parameter("show_point_labels", False)
        assert not any(label.get_visible() for label in
                       (*artists["image:point-labels"], *artists["image:point-path-labels"]))
        assert not artists["image:point-path-timebase"].get_visible()
        assert artists["image:point-paths"].get_visible()
        session.set_parameter("show_point_labels", True)
        np.testing.assert_array_equal(composed, session.rgba())
        changed = paths.copy()
        changed[0, 1, 0] = 0.5
        with pytest.raises(RevisionError, match="different content"):
            session.configure(image_overlay=replace(restored.overlay, paths_xy=changed))
    finally:
        session.close()
    redrawn, _archive = save_figure_artifact(
        tmp_path / "reopened", plot_input=restored, spec=recipe["spec"],
        parameters=recipe["parameters"], size=recipe["size"],
    )
    with Image.open(image) as first, Image.open(redrawn) as second:
        np.testing.assert_array_equal(np.asarray(first), np.asarray(second))

    # Dense renderer-only stress: every explicit ID survives, without the
    # repetitive f0/final stamps that obscured neighbouring source IDs.
    yy, xx = np.meshgrid(10.+6.*np.arange(10), 10.+6.*np.arange(10), indexing="ij")
    points = np.column_stack((xx.ravel(), yy.ravel()))
    dense_paths = points[:, None, :] + np.linspace(0., 1., 5)[None, :, None]*np.asarray((.45, .25))
    schema = make_dataset_schema(repeat_domain(size=1), mapped_domain_from_columns({"sample": [0.]}),
        cell_axes=(axis("row", size=84), axis("column", size=84)), dtype=np.float64)
    values = np.zeros((1, 1, 84, 84))
    values[0, 0, points[:, 1].astype(int), points[:, 0].astype(int)] = 1.
    dense = PlotSession(ImageFrame(make_snapshot(schema, values, 0), ImagePointOverlay(
        0, points, labels=tuple(str(i+1) for i in range(100)), paths_xy=dense_paths,
        static_statuses=(PointStatus.OCCUPIED,)*100)),
        spec, parameters={"show_image": False, "side_distribution": False})
    try:
        dense.rgba()
        joined = dense._renderer._artists["image:point-paths"]
        np.testing.assert_allclose(joined._zlc_point_path_inputs[2], (1.8, 1.8))
        np.testing.assert_array_equal(joined._zlc_point_path_inputs[1], dense_paths)
        # Collinear samples retain scientific time but need only two paint
        # endpoints, not a 64-vertex join at each original sample.
        assert sum(len(path.vertices) for path in joined.get_paths()) < 20000
        texts = dense._renderer._artists["image:point-labels"]
        assert sum(text.get_visible() for text in texts) == 100
        assert not any(text.get_visible() for text in dense._renderer._artists["image:point-path-labels"])
        renderer = dense._renderer.figure.canvas.get_renderer()
        boxes = [text.get_window_extent(renderer) for text in texts]
        assert not any(first.overlaps(second) for i, first in enumerate(boxes) for second in boxes[i+1:])
    finally:
        dense.close()

    # Shared crossing/collinear/source junctions are one filled coverage union,
    # not three successive translucent draws of the same status colour.
    starts = np.asarray(((8.,32.),(32.,8.),(18.,32.)))
    ends = np.asarray(((56.,32.),(32.,56.),(48.,32.)))
    alpha_paths = np.stack((starts, ends), axis=1)
    alpha = PlotSession(ImageFrame(make_snapshot(schema, values, 0), ImagePointOverlay(
        0, starts, paths_xy=alpha_paths, static_statuses=(PointStatus.OCCUPIED,)*3)),
        spec, parameters={"show_image":False,"side_distribution":False,"show_point_labels":False},
        device_pixel_ratio=3.)
    try:
        rgba = alpha.rgba()
        renderer = alpha._renderer
        token = renderer.style.artists.point_occupied
        expected = np.floor(255.*(np.asarray(to_rgba(token.color))[:3]*token.alpha + 1.-token.alpha))
        radius = renderer._artists["image:point-paths"]._zlc_point_path_inputs[2][0]
        assert radius == pytest.approx(10.*renderer.style.artists.point_auto_radius_fraction)
        geometry = renderer._artists["image:point-paths"].get_paths()[0]
        starts_at = np.flatnonzero(geometry.codes == DrawPath.MOVETO)
        pieces = [geometry.vertices[first:last] for first,last in
                  zip(starts_at,np.r_[starts_at[1:],len(geometry.vertices)],strict=True)]
        heads = [piece for piece in pieces if len(piece) == 4]
        assert len(heads) == 3
        for head, start, end in zip(heads, starts, ends, strict=True):
            begin, final = renderer.primary_axes.transData.transform((start,end))
            direction = (final-begin)/np.linalg.norm(final-begin)
            tip, base = head[0], head[1:3].mean(axis=0)
            assert np.dot(final-tip,direction) > 0.
            assert np.dot(tip-base,direction) > 0.
            assert np.dot(base-begin,direction) > 0.
        terminal_pixels = renderer.primary_axes.transData.transform(ends)
        for piece in pieces:
            if len(piece) == len(DrawPath.unit_circle().vertices):
                center = (piece.min(axis=0)+piece.max(axis=0))*.5
                assert np.all(np.linalg.norm(terminal_pixels-center,axis=1) > 1e-6)
        head_point = renderer.primary_axes.transData.inverted().transform(heads[0][:3].mean(axis=0))
        for point in ((32.,32.),(40.,32.),(8.+radius,32.),head_point):
            pixel = renderer.primary_axes.transData.transform(point)
            x,y = int(pixel[0]),rgba.shape[0]-1-int(pixel[1])
            region = rgba[y-1:y+2,x-1:x+2,:3].min(axis=(0,1))
            assert np.all(region >= expected-2), (point,region,expected)
            assert np.max(np.abs(region-expected)) <= 3, (point,region,expected)
    finally:
        alpha.close()
    ordinary = PlotSession(ImageFrame(make_snapshot(schema, values, 0), ImagePointOverlay(
        0, starts, labels=("1", "2", "3"), static_statuses=(PointStatus.OCCUPIED,)*3)), spec)
    try:
        # Full supplied geometry owns the same radius with or without paths.
        for point, label in zip(starts, ordinary._renderer._artists["image:point-labels"], strict=True):
            assert point[0]-label.get_position()[0] == pytest.approx(radius)
    finally:
        ordinary.close()

    yy, xx = np.meshgrid(10.+10.*np.arange(5), 10.+10.*np.arange(7), indexing="ij")
    roster = np.column_stack((xx.ravel(), yy.ravel()))
    roster_paths = np.repeat(roster[:, None, :], 5, axis=1)
    roster_paths[0, :, 0] += np.linspace(0., 20., 5)
    cropped_schema = make_dataset_schema(repeat_domain(size=1), mapped_domain_from_columns({"sample": [0.]}),
        cell_axes=(axis("row", size=18), axis("column", size=24)), dtype=np.float64)
    for snapshot in (make_snapshot(schema, values, 0),
                     make_snapshot(cropped_schema, np.zeros((1, 1, 18, 24)), 0)):
        source_label_properties = source_label_anchors = source_label_boxes = None
        for paths in (None, roster_paths):
            full_roster = PlotSession(ImageFrame(snapshot, ImagePointOverlay(
                0, roster, labels=tuple(str(i+1) for i in range(35)), paths_xy=paths,
                static_statuses=(PointStatus.OCCUPIED,)*35)), spec,
                parameters={"show_image": False, "side_distribution": False})
            try:
                full_roster.rgba()
                artists = full_roster._renderer._artists
                token = full_roster._renderer.style.artists.point_occupied
                site_labels = artists["image:point-labels"]
                properties = [(label.get_text(), label.get_position(), label.get_fontproperties().copy(),
                               label.get_ha(), label.get_va()) for label in site_labels]
                anchors = np.asarray([label.get_transform().transform(label.get_position()) for label in site_labels])
                renderer = full_roster._renderer.figure.canvas.get_renderer()
                boxes = np.asarray([label.get_window_extent(renderer).extents for label in site_labels])
                if paths is None:
                    source_label_properties, source_label_anchors, source_label_boxes = properties, anchors, boxes
                    for point, label in zip(roster, artists["image:point-labels"], strict=True):
                        assert point[0]-label.get_position()[0] == pytest.approx(3.)
                    np.testing.assert_array_equal(artists["image:points"].get_offsets(), roster)
                    np.testing.assert_allclose(artists["image:points"].get_edgecolors(),
                                               np.tile(to_rgba(token.color, token.alpha), (35, 1)))
                else:
                    assert properties == source_label_properties
                    np.testing.assert_array_equal(anchors, source_label_anchors)
                    np.testing.assert_array_equal(boxes, source_label_boxes)
                    np.testing.assert_array_equal(artists["image:point-paths"]._zlc_point_path_inputs[0], roster)
                    np.testing.assert_allclose(artists["image:point-paths"]._zlc_point_path_inputs[2], (3., 3.))
                    np.testing.assert_allclose(artists["image:point-paths"].get_facecolors()[0],
                                               to_rgba(token.color, token.alpha))
            finally:
                full_roster.close()

def test_image_site_numbers_use_their_ring_status_style() -> None:
    """A small ordinal must remain visually attached to its status ring."""

    from matplotlib.colors import to_rgba

    snapshot = _image_snapshot()
    overlay = ImagePointOverlay(
        1,
        np.asarray(((0.5, 0.5), (1.5, 0.5), (2.5, 1.5))),
        point_ids=("trap-a", "trap-b", "trap-c"),
        labels=("1", "2", "3"),
        static_statuses=(
            PointStatus.EMPTY,
            PointStatus.OCCUPIED,
            PointStatus.INVALID,
        ),
    )
    session = PlotSession(
        ImageFrame(snapshot, overlay),
        ImagePlot(AxisRef.cell_data("column"), AxisRef.cell_data("row")),
    )
    try:
        # The ordinals are on by default: an overlay is data, not a mode.
        assert session.display_state["show_point_labels"] is True
        artists = session._renderer._artists["image:point-labels"]
        tokens = (
            session._renderer.style.artists.point_empty,
            session._renderer.style.artists.point_occupied,
            session._renderer.style.artists.point_invalid,
        )
        assert tuple(label.get_text() for label in artists) == ("1", "2", "3")
        assert all(
            to_rgba(label.get_color(), label.get_alpha())
            == to_rgba(token.color, token.alpha)
            for label, token in zip(artists, tokens, strict=True)
        )
        assert all(
            label.get_fontsize() == session._renderer.style.fonts.fit_annotation_pt
            for label in artists
        )
        image_axes = session._renderer._axes["image"][0]
        positions = tuple(label.get_position() for label in artists)
        assert all(
            position[0] < point[0]
            for position, point in zip(positions, overlay.coordinates, strict=True)
        )
        if image_axes.yaxis_inverted():
            assert all(
                position[1] < point[1]
                for position, point in zip(
                    positions, overlay.coordinates, strict=True
                )
            )
        else:
            assert all(
                position[1] > point[1]
                for position, point in zip(
                    positions, overlay.coordinates, strict=True
                )
            )
    finally:
        session.close()
    single = PlotSession(ImageFrame(snapshot, ImagePointOverlay(
        0, np.asarray(((.5, .5),)), labels=("1",), static_statuses=(PointStatus.OCCUPIED,))),
        ImagePlot(AxisRef.cell_data("column"), AxisRef.cell_data("row")))
    try:
        # A single marker on the tiny image retains the existing positive
        # image-span fallback rather than inflating to the cell-pitch cap.
        position = single._renderer._artists["image:point-labels"][0].get_position()
        assert .5-position[0] == pytest.approx(single._renderer.style.artists.point_single_radius_fraction)
    finally:
        single.close()

def test_session_fit_all_facets_returns_one_result_per_painted_cell() -> None:
    spec = FacetGridPlot(
        AxisRef.point("facet"),
        CurvePlot(AxisRef.point("x")),
    )
    session = PlotSession(_snapshot(), spec)
    try:
        result = session.fit("gaussian_offset", fit_all_facets=True, live=False)
        assert isinstance(result, FacetFitBatchResult)
        assert len(result.results) == 2
        assert result.source_revision == 0
        assert result.sample_axes[0][0] == "point"
        assert result.sample_axes[0][1].axis_id.value == "facet"
    finally:
        session.close()

def test_an_identical_static_fit_target_does_no_work(monkeypatch) -> None:
    """A second identical static target IS the fit already painted.

    Only the live target answered "same request" with silence; a static one
    re-solved, re-rendered and stamped a new batch revision every time the
    same complete target came back -- and a Setting form re-sends its whole
    target on every unrelated edit.  Something the fit depends on moving is
    what makes the same target solve again.
    """

    target = {
        "model": "gaussian_offset",
        "fixed": {"amplitude": 1.0, "center": 2.5, "sigma": 1.0, "offset": 0.0},
    }
    from zlc_plot.rendering import MatplotlibRenderer

    presented = []
    original_present = MatplotlibRenderer.present

    def present(self, frame, **kwargs):
        presented.append(bool(frame.fit_overlays))
        return original_present(self, frame, **kwargs)

    monkeypatch.setattr(MatplotlibRenderer, "present", present)
    session = PlotSession(
        _snapshot(), CurvePlot(AxisRef.point("x")),
        initial_configuration={"fit": target, "fit_live": False},
    )
    assert presented == [True]
    fronts: list[None] = []
    release = session.subscribe_surface(lambda: fronts.append(None))
    try:
        session.configure(fit=target, fit_live=False)
        first = session.last_fit
        assert first is not None
        painted = len(fronts)
        session.configure(fit=target, fit_live=False)
        assert session.last_fit is first
        assert len(fronts) == painted
        session.set_x_selector(1.0, 4.0)
        session.configure(fit=target, fit_live=False)
        assert session.last_fit is not first
        artists = session._renderer._selector_artists[SelectorKind.X_RANGE]
        label = next(artist for artist in artists if hasattr(artist, "get_text"))
        assert not label.get_visible(), "visible Fit text owns the ROI annotation space"
        assert all(artist.get_visible() for artist in artists if artist is not label)
        selection = session.selector_state(SelectorKind.X_RANGE, display=False)
        session.clear_fit()
        assert label.get_visible()
        assert session.selector_state(SelectorKind.X_RANGE, display=False) == selection
    finally:
        release()
        session.close()

def test_the_same_indexed_publication_may_be_restated() -> None:
    """Configuring the data a session already holds is zero work, not an error."""

    snapshot = _image_snapshot(indexed=True)
    session = PlotSession(
        snapshot, ImagePlot(AxisRef.cell_data("column"), AxisRef.cell_data("row"))
    )
    fronts: list[None] = []
    release = session.subscribe_surface(lambda: fronts.append(None))
    try:
        described = session.configure(data=snapshot)
        assert described.display_state == session.display_state
        assert fronts == []
    finally:
        release()
        session.close()
