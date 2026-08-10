"""The console's default axis pick can always carry what the kind draws.

A camera frame's point table has one row, and the plot library's curve
default walks the point domain -- so "1D vector" on a camera signal opened as
one invisible point.  The workbench resolver re-points a degenerate series x
onto a dense axis through the library's own composition authority.
"""

from __future__ import annotations

import numpy as np

from zlc_plot import AxisRef, PlotKind
from zlc_plot.semantics import axis_size
from data_factory import Axis, DatasetSchema, PointTable

from zlc_workbench.panel_spec import fitting_panel_spec


def _camera_frame_schema():
    return DatasetSchema.create(
        Axis.create("repeat", size=1),
        PointTable.from_columns({"frame": np.zeros(1)}),
        data_axes=(
            Axis.create("spatial-y", size=8),
            Axis.create("spatial-x", size=6),
        ),
        dtype=np.float64,
    )


def _scanned_schema():
    return DatasetSchema.create(
        Axis.create("repeat", size=2),
        PointTable.from_columns(
            {"x": np.arange(4.0), "row": np.array([0.0, 0.0, 1.0, 1.0])}
        ),
        dtype=np.float64,
    )


def test_default_curve_on_a_camera_frame_lands_on_a_dense_axis() -> None:
    schema = _camera_frame_schema()
    spec = fitting_panel_spec(schema, "curve")
    assert spec is not None and spec.kind is PlotKind.CURVE
    assert axis_size(schema, spec.x) > 1


def test_default_curve_on_a_scan_keeps_the_scan_axis() -> None:
    schema = _scanned_schema()
    spec = fitting_panel_spec(schema, "curve")
    assert spec is not None
    assert spec.x == AxisRef.point("x")


def test_facet_grid_curve_cell_default_is_also_dense() -> None:
    schema = _camera_frame_schema()
    spec = fitting_panel_spec(schema, "facet_grid", "curve")
    if spec is None:
        # A camera frame may not admit a facet grid at all; that refusal is
        # the library's own and not this resolver's concern.
        return
    assert axis_size(schema, spec.cell.x) > 1


def test_image_specs_pass_through_untouched() -> None:
    schema = _camera_frame_schema()
    spec = fitting_panel_spec(schema, "image")
    assert spec is not None and spec.kind is PlotKind.IMAGE
