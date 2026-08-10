from __future__ import annotations

import numpy as np

from data_factory import Axis, DatasetSchema, DatasetSnapshot, PointTable
from zlc_plot import AxisRef, ImagePlot, PlotSession


def _image_snapshot(*, x_unit: str = "m", y_unit: str = "m") -> DatasetSnapshot:
    x = np.linspace(-2.0, 2.0, 21)
    y = np.linspace(-3.0, 3.0, 25)
    schema = DatasetSchema.create(
        Axis.create("repeat", size=1),
        PointTable.from_columns({"sample": [0.0]}),
        data_axes=(
            Axis.create("x", values=x, canonical_unit=x_unit),
            Axis.create("y", values=y, canonical_unit=y_unit),
        ),
        dtype=np.float64,
        canonical_unit="1",
        generation=f"image-fit-{x_unit}-{y_unit}",
    )
    xx, yy = np.meshgrid(x, y)
    values = 0.4 + 2.0 * np.exp(
        -((xx - 0.35) ** 2 / 0.7**2 + (yy + 0.8) ** 2 / 1.1**2)
    )
    return DatasetSnapshot(schema, values.T[None, None, :, :], revision=0)


def _image_session(*, x_unit: str = "m", y_unit: str = "m") -> PlotSession:
    return PlotSession(
        _image_snapshot(x_unit=x_unit, y_unit=y_unit),
        ImagePlot(AxisRef.data("x"), AxisRef.data("y")),
    )


def test_non_equivalent_image_uses_anisotropic_fit_and_recovers_center() -> None:
    session = _image_session(x_unit="m", y_unit="s")
    try:
        models = {model.model_id for model in session.fit_models}
        assert "anisotropic_gaussian_center" in models
        assert "radial_gaussian_center" not in models
        result = session.fit("anisotropic_gaussian_center", live=False)
        assert result.success
        assert abs(result.parameters["center_x"] - 0.35) < 1.0e-9
        assert abs(result.parameters["center_y"] + 0.8) < 1.0e-9
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


def test_image_display_unit_change_preserves_canonical_pixel_geometry() -> None:
    session = _image_session()
    try:
        axes = session._renderer.primary_axes
        image = session._renderer._artists["image"]
        before_bbox = tuple(float(value) for value in axes.bbox.bounds)
        before_array = np.asarray(image.get_array()).copy()
        assert axes.get_aspect() == 1.0

        session.set_axis_unit(AxisRef.data("x"), "cm")

        image = session._renderer._artists["image"]
        after_bbox = tuple(float(value) for value in axes.bbox.bounds)
        assert np.allclose(after_bbox, before_bbox, rtol=0.0, atol=1.0e-9)
        assert axes.get_aspect() == 100.0
        np.testing.assert_array_equal(np.asarray(image.get_array()), before_array)
        # The display extent changes by the unit conversion, while the
        # renderer's physical box and prepared image remain invariant.
        assert np.isclose(float(image.get_extent()[1]), 100.0 * 2.1)
    finally:
        session.close()


def test_non_equivalent_image_does_not_square_pad_unrelated_axes() -> None:
    session = _image_session(x_unit="m", y_unit="s")
    try:
        axes = session._renderer.primary_axes
        image = session._renderer._artists["image"]
        assert axes.get_aspect() == "auto"
        extent = tuple(float(value) for value in image.get_extent())
        assert np.allclose(axes.get_xlim(), extent[:2])
        assert np.allclose(axes.get_ylim(), extent[2:])
    finally:
        session.close()


def test_an_axis_named_either_supported_way_takes_the_dense_image_path() -> None:
    """The resolver accepts an axis NAME or a full axis id.

    The dense image path used to build its own map keyed on the name alone, so
    the precise spelling dropped silently into the generic per-pixel aggregator
    -- measured 830 ms against 30 ms on one 640x480 frame, growing with the
    pixel count.  One rule decides what an axis reference means.
    """

    snapshot = _image_snapshot()
    axes = snapshot.block.schema.cell_schema.data_axes
    by_name = ImagePlot(AxisRef.data(axes[0].name), AxisRef.data(axes[1].name))
    by_id = ImagePlot(
        AxisRef.data(str(axes[0].axis_id)), AxisRef.data(str(axes[1].axis_id))
    )

    from zlc_data import ReductionMethod
    from zlc_plot.data_view import DataView

    view = DataView(snapshot)
    dense = [
        view._dense_data_image(spec.x, spec.y, ReductionMethod.MEAN) is not None
        for spec in (by_name, by_id)
    ]

    assert dense == [True, True], "one supported spelling missed the dense path"


def test_the_schema_says_which_axes_are_the_image() -> None:
    """Picking by size and position refused datasets that declared their axes.

    A camera frame arrives as (readout-event, spatial-y, spatial-x).  With one
    window the readout axis is length one and gets filtered out, so it worked by
    luck; with two windows there are three significant axes and the image kind
    was refused outright, and a one-pixel-tall ROI strip was refused from the
    other side.  The roles are on the axes precisely so a reader can tell.
    """

    import numpy as np

    from zlc_data import READOUT_EVENT, SPATIAL_X, SPATIAL_Y
    from zlc_plot._kinds.image import HANDLER

    from data_factory import Axis, DatasetSchema, PointTable

    def _schema(events: int, height: int, width: int) -> DatasetSchema:
        return DatasetSchema.create(
            Axis.create("repeat", size=1),
            PointTable.from_columns({"sample": [0.0]}),
            data_axes=(
                Axis.create("event", size=events, role=READOUT_EVENT),
                Axis.create("y", size=height, role=SPATIAL_Y),
                Axis.create("x", size=width, role=SPATIAL_X),
            ),
            dtype=np.float64,
            generation="image-roles",
        )

    for events, height, width in ((1, 60, 80), (2, 60, 80), (1, 1, 80)):
        spec = HANDLER.default_spec(_schema(events, height, width))
        assert spec is not None, (events, height, width)
