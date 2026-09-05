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
    RollingPlot,
    RasterPlotHost,
    curve,
)
from zlc_plot.fit import FacetFitBatchResult
from zlc_plot.primitives import ImageFrame, ImagePointOverlay, PointStatus

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

def test_image_overlay_is_explicit_data_not_a_display_mode() -> None:
    snapshot = _image_snapshot()
    spec = ImagePlot(AxisRef.cell_data("column"), AxisRef.cell_data("row"))
    session = PlotSession(snapshot, spec)
    curve = PlotSession(_snapshot(), CurvePlot(AxisRef.point("x")))
    try:
        described = session.describe_display()
        assert "site_overlay" not in described.parameter_schema
        assert "site_overlay" not in described.display_state.values
        assert described.display_state["show_point_labels"] is True

        curve_description = curve.describe_display()
        assert "site_overlay" not in curve_description.parameter_schema
        assert "site_overlay" not in curve_description.display_state.values
        with pytest.raises(KeyError, match="site_overlay"):
            curve.set_parameter("site_overlay", "centers")
    finally:
        curve.close()
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
    )
    incoming = ImagePointOverlay(
        0, np.asarray(((1.0, 0.5),)),
        static_statuses=(PointStatus.EMPTY,),
    )
    spec = ImagePlot(AxisRef.cell_data("column"), AxisRef.cell_data("row"))
    service = RenderProcess("image-frame-contract") if entry == "process" else None
    host = (
        service.build_host(ImageFrame(first, overlay), spec)
        if service is not None
        else RasterPlotHost.from_plot(ImageFrame(first, overlay), spec)
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

        operation = update(bare)
        assert operation.front.identity.data_revision == 11
        assert operation.front.identity.image_overlay_revision is None
        reference = RasterPlotHost.from_plot(first, spec)
        try:
            reference.wait_for_front(timeout=30)
            reference.update_data(second).result(timeout=30)
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
        parameters={"show_point_labels": True},
    )
    try:
        artists = session._renderer._artists["image:point-labels"]
        tokens = (
            session._renderer.style.artists.point_empty,
            session._renderer.style.artists.point_occupied,
            session._renderer.style.artists.point_invalid,
        )
        assert tokens[0].alpha <= 0.15
        assert tokens[1].alpha <= 0.60
        assert tokens[0].alpha < tokens[1].alpha
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

def test_session_rolling_history_seeds_per_repeat_then_grows_one_sample_per_revision() -> None:
    rolling = PlotSession(_snapshot(repeats=3), RollingPlot())
    try:
        # A static snapshot is a complete shot record: the repeat axis seeds
        # the history so the initial render already shows every shot.
        payload = rolling._payload
        assert len(payload.series) == 1
        assert payload.series[0].x.canonical.size == 3

        # A later non-indexed publication replaces the former one; Plot never
        # grows a second cross-publication history beside Runtime.
        rolling.update_data(_snapshot(revision=1, repeats=3))
        payload = rolling._payload
        assert payload.series[0].x.canonical.size == 3
    finally:
        rolling.close()

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
        assert result.facet == AxisRef.point("facet")
    finally:
        session.close()
