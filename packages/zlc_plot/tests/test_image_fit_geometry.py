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
from zlc_data import DatasetSchema, OwnedSnapshot, REPEAT, SPATIAL_X, SPATIAL_Y
from zlc_plot import AxisRef, FacetGridPlot, ImagePlot, PlotSession


def _image_snapshot(*, x_unit: str = "m", y_unit: str = "m") -> OwnedSnapshot:
    x = np.linspace(-2.0, 2.0, 21)
    y = np.linspace(-3.0, 3.0, 25)
    schema = make_dataset_schema(
        repeat_domain(size=1),
        mapped_domain_from_columns({"sample": [0.0]}),
        cell_axes=(
            axis("x", values=x, unit=x_unit, role=SPATIAL_X),
            axis("y", values=y, unit=y_unit, role=SPATIAL_Y),
        ),
        dtype=np.float64,
        value_unit="1",
    )
    xx, yy = np.meshgrid(x, y)
    values = 0.4 + 2.0 * np.exp(
        -((xx - 0.35) ** 2 / 0.7**2 + (yy + 0.8) ** 2 / 1.1**2)
    )
    return make_snapshot(schema, values.T[None, None, :, :], revision=0)


def _image_session(*, x_unit: str = "m", y_unit: str = "m") -> PlotSession:
    return PlotSession(
        _image_snapshot(x_unit=x_unit, y_unit=y_unit),
        ImagePlot(AxisRef.cell_data("x"), AxisRef.cell_data("y")),
    )


def test_non_equivalent_image_uses_anisotropic_fit_and_recovers_center() -> None:
    session = _image_session(x_unit="m", y_unit="s")
    try:
        fit_events = []
        session.subscribe_fit(fit_events.append)
        models = {model.model_id for model in session.fit_models}
        assert "anisotropic_gaussian_center" in models
        assert "radial_gaussian_center" not in models
        result = session.fit("anisotropic_gaussian_center")
        assert result.success
        assert abs(result.parameters["center_x"] - 0.35) < 1.0e-9
        assert abs(result.parameters["center_y"] + 0.8) < 1.0e-9
        source = _image_snapshot(x_unit="m", y_unit="s")
        from zlc_data import owned_snapshot_from_arrays

        restarted = owned_snapshot_from_arrays(
            source.block.schema,
            source.block.values,
            source.ref.revision,
            block_id=source.ref.block_id,
            stream_generation="image-fit-restarted",
        )
        session.update_data(restarted)
        assert session.last_fit is None
        assert session.fit_status is None
        assert fit_events[-1] is None
    finally:
        session.close()


def test_anisotropic_image_fit_routes_through_the_regular_image_path() -> None:
    """The separable capability, not a model-id literal, selects the fast path."""

    session = _image_session(x_unit="m", y_unit="s")
    try:
        selection = session.fit_selection("anisotropic_gaussian_center")
        assert selection.regular_image is not None
        assert selection.regular_image.valid_mask is None
    finally:
        session.close()


def test_equivalent_image_keeps_radial_catalogue_entry() -> None:
    session = _image_session()
    try:
        assert "radial_gaussian_center" in {
            model.model_id for model in session.fit_models
        }
    finally:
        session.close()


@pytest.mark.parametrize("faceted", (False, True))
def test_image_fit_ring_uses_the_occupied_point_ring_style(faceted: bool) -> None:
    """Standalone and Facet image fits share the occupied-ring visual token."""

    from matplotlib.colors import to_rgba

    cell = ImagePlot(AxisRef.cell_data("x"), AxisRef.cell_data("y"))
    spec = FacetGridPlot(AxisRef.repeat("repeat"), cell) if faceted else cell
    session = PlotSession(_image_snapshot(), spec)
    try:
        result = session.fit("radial_gaussian_center", live=False)
        assert result.success
        accepted = session._accepted_fit
        assert accepted is not None and len(accepted.overlays) == 1
        glyph = accepted.overlays[0].ellipse_glyph
        assert glyph is not None
        renderer = session._renderer
        if faceted:
            native = renderer._artists.get("facet:fit_native")
            assert isinstance(native, dict)
            assert tuple(native["overlays"]) == accepted.overlays
            assert renderer.style.artists.point_occupied.linewidth > 0.0
            return
        slots = renderer._fit_slots
        ring = slots["ring"]
        center = slots["center"]
        annotation = slots["annotation"]
        token = renderer.style.artists.point_occupied

        assert ring.get_edgecolor() == pytest.approx(
            to_rgba(token.color, token.alpha)
        )
        assert (ring.get_alpha(), ring.get_linewidth()) == (
            token.alpha,
            token.linewidth,
        )
        assert ring.get_visible() and center.get_visible()
        assert ring.get_facecolor()[3] == 0.0
        assert ring.get_center() == pytest.approx(
            (glyph.center_x, glyph.center_y), rel=0.0, abs=0.0
        )
        assert (ring.get_width(), ring.get_height()) == pytest.approx(
            (2.0 * glyph.radius_x, 2.0 * glyph.radius_y), rel=0.0, abs=0.0
        )
        center_x, center_y = center.get_data()
        assert (center_x[0], center_y[0]) == pytest.approx(
            (glyph.center_x, glyph.center_y), rel=0.0, abs=0.0
        )
        assert to_rgba(center.get_markerfacecolor()) == pytest.approx(
            to_rgba(renderer.style.artists.fit_ellipse_color)
        )
        center_area = renderer.style.artists.fit_ellipse_center_area_pt2
        assert center_area == 2.25
        assert center.get_markersize() ** 2 == pytest.approx(center_area)
        assert annotation.get_visible() and annotation.get_text()
    finally:
        session.close()


def test_image_display_unit_change_preserves_canonical_pixel_geometry() -> None:
    session = _image_session()
    try:
        axes = session._renderer.primary_axes
        image = session._renderer._artists["image"]
        before_bbox = tuple(float(value) for value in axes.bbox.bounds)
        before_array = np.asarray(image.get_array()).copy()
        assert axes.get_aspect() == pytest.approx(0.8)

        session.set_axis_unit(AxisRef.cell_data("x"), "mm")

        image = session._renderer._artists["image"]
        after_bbox = tuple(float(value) for value in axes.bbox.bounds)
        assert np.allclose(after_bbox, before_bbox, rtol=0.0, atol=1.0e-9)
        assert after_bbox[2] == pytest.approx(after_bbox[3], abs=1.0e-9)
        assert axes.get_aspect() == pytest.approx(800.0)
        np.testing.assert_array_equal(np.asarray(image.get_array()), before_array)
        # The display extent changes by the unit conversion, while the
        # renderer's physical square and prepared image remain invariant.  The
        # RGBA artist fills the square viewport; the prepared extent remains
        # the real data footprint inside that letterboxed view.
        prepared = session._renderer._artists["image:prepared_current"]
        assert np.isclose(float(prepared.extent[1]), 1000.0 * 2.1)
        assert tuple(map(float, image.get_extent())) == pytest.approx(
            (*map(float, axes.get_xlim()), *map(float, axes.get_ylim()))
        )
    finally:
        session.close()


def test_non_equivalent_image_still_uses_square_screen_cells() -> None:
    session = _image_session(x_unit="m", y_unit="s")
    try:
        axes = session._renderer.primary_axes
        image = session._renderer._artists["image"]
        assert axes.get_aspect() == pytest.approx(0.8)
        bbox = tuple(map(float, axes.bbox.bounds))
        assert bbox[2] == pytest.approx(bbox[3], abs=1.0e-9)
        x = np.linspace(-2.0, 2.0, 21)
        y = np.linspace(-3.0, 3.0, 25)
        origin = axes.transData.transform((x[0], y[0]))
        x_pixels = axes.transData.transform((x[1], y[0]))[0] - origin[0]
        y_pixels = axes.transData.transform((x[0], y[1]))[1] - origin[1]
        assert abs(x_pixels) == pytest.approx(abs(y_pixels), rel=1.0e-9)
        extent = tuple(float(value) for value in image.get_extent())
        assert np.allclose(axes.get_xlim(), extent[:2])
        assert np.allclose(axes.get_ylim(), extent[2:])
    finally:
        session.close()


def test_the_schema_says_which_axes_are_the_image() -> None:
    """Picking by size and position refused datasets that declared their axes.

    A camera frame arrives as (readout-event, spatial-y, spatial-x).  With one
    window the readout axis is length one and gets filtered out, so it worked by
    luck; with two windows there are three significant axes and the image kind
    was refused outright, and a one-pixel-tall ROI strip was refused from the
    other side.  The roles are on the axes precisely so a reader can tell.
    """

    import numpy as np

    from zlc_data import READOUT_EVENT
    from zlc_plot._kinds.image import HANDLER

    def _schema(events: int, height: int, width: int) -> DatasetSchema:
        return make_dataset_schema(
            repeat_domain(size=1),
            mapped_domain_from_columns({"sample": [0.0]}),
            cell_axes=(
                axis("event", size=events, role=READOUT_EVENT),
                axis("y", size=height, role=SPATIAL_Y),
                axis("x", size=width, role=SPATIAL_X),
            ),
            dtype=np.float64,
        )

    for events, height, width in ((1, 60, 80), (2, 60, 80), (1, 1, 80)):
        spec = HANDLER.default_spec(_schema(events, height, width))
        assert spec is not None, (events, height, width)
