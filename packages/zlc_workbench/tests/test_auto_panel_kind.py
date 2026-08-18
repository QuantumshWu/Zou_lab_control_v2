"""Who decides what a started node draws, and what a grid can draw at all.

Two questions that used to be tangled.  WHICH plot a node's Start opens is
the node's own answer -- it knows what it is about to measure, and a shape
alone does not.  What a facet grid can draw is a question about the data,
and the answer must not be "nothing" for a signal an operator can obviously
see in cells.
"""

from __future__ import annotations

import numpy as np
import pytest

from zlc_data import READOUT_EVENT, SITE, SPATIAL_X, SPATIAL_Y
from zlc_plot import CurvePlot, FacetGridPlot, HistogramPlot, ImagePlot
from zlc_workbench.panel_catalog import task_console_fitting_spec


def _camera_frames(*, cycles: int, frames: int):
    """(cycles, F, y, x) exactly as camera_measurement publishes it."""

    from zlc_atom.data import snapshot_from_array
    from zlc_atom.nodes.camera_measurement.measurement import _frame_point_column
    from zlc_data import READOUT_EVENT

    values = np.zeros((cycles, frames, 6, 8), dtype=np.uint16)
    return snapshot_from_array(
        values,
        producer="cam",
        signal="frames",
        roles=(READOUT_EVENT, SPATIAL_Y, SPATIAL_X),
        point_columns={READOUT_EVENT: _frame_point_column("cam", frames)},
        generation="auto-kind",
        revision=1,
    ).block.schema


def _scalar_source(*, repeats: int = 1):
    """One scalar per shot: what a scan of a device readout captures."""

    from zlc_atom.data import snapshot_from_array

    return snapshot_from_array(
        np.zeros((repeats,), dtype=np.float64),
        producer="device",
        signal="readout",
        generation="auto-kind",
        revision=1,
    ).block.schema


def _scan(source, *, axes: dict[str, int], visits: int = 1):
    """A scan dataset the way ScanDatasetWriter lays one out."""

    import itertools

    from zlc_atom.nodes.scan.dataset import scan_dataset_schema

    rows = tuple(
        itertools.product(*(range(size) for size in axes.values()))
    )
    return scan_dataset_schema(
        source,
        rows,
        tuple((name, "V") for name in axes),
        visits=visits,
    )


def _auto(schema):
    return task_console_fitting_spec(schema, "", "")


def _occupancy_counts(*, frames: int, sites: int):
    """(repeat, frames, sites) exactly as the occupancy processor publishes."""

    from zlc_atom.data import snapshot_from_array
    from zlc_atom.nodes.camera_measurement.measurement import _frame_point_column

    return snapshot_from_array(
        np.zeros((1, frames, sites), dtype=float),
        producer="occupancy",
        signal="counts",
        roles=(READOUT_EVENT, SITE),
        point_columns={READOUT_EVENT: _frame_point_column("cam", frames)},
        generation="auto-kind",
        revision=1,
    ).block.schema


def _scan_of_scalars(*, points: int):
    """(repeat, points) -- one number per scanned point, no cell axis."""

    from zlc_atom.data import snapshot_from_array
    from zlc_data import AxisId, PointColumn, SCAN_POINT

    return snapshot_from_array(
        np.zeros((2, points), dtype=float),
        producer="scan",
        signal="value",
        roles=(SCAN_POINT,),
        point_columns={
            SCAN_POINT: PointColumn(
                AxisId("scan.knob"),
                "knob",
                SCAN_POINT,
                PointColumn.NUMERIC,
                tuple(float(index) for index in range(points)),
            )
        },
        generation="auto-kind",
        revision=1,
    ).block.schema


def test_a_grid_of_judged_frames_draws_one_cell_per_frame() -> None:
    """The reported refusal: 1 x 3 x 35 counts, and the grid said no.

    Three frames judged over thirty-five sites is three cells with a curve
    -- or a histogram -- in each; there is nothing ambiguous about it.  The
    grid refused because only its image-cell branch knew how to face a point
    axis, and a curve cell fell through to "nothing to facet".
    """

    schema = _occupancy_counts(frames=3, sites=35)
    frame_axis = "cam.frames.frame"

    automatic = task_console_fitting_spec(schema, "facet_grid", "")
    assert isinstance(automatic, FacetGridPlot), automatic
    assert automatic.facet.axis_id == frame_axis
    assert isinstance(automatic.cell, CurvePlot)
    # Inside one cell there is one frame, so the curve walks the sites.
    assert automatic.cell.x.axis_id.endswith("site")

    asked_for_curve = task_console_fitting_spec(schema, "facet_grid", "curve")
    assert isinstance(asked_for_curve, FacetGridPlot)
    assert asked_for_curve.facet.axis_id == frame_axis
    assert isinstance(asked_for_curve.cell, CurvePlot)

    asked_for_histogram = task_console_fitting_spec(schema, "facet_grid", "histogram")
    assert isinstance(asked_for_histogram, FacetGridPlot)
    assert isinstance(asked_for_histogram.cell, HistogramPlot)


def test_the_automatic_kind_is_the_flat_kind_the_data_proves() -> None:
    """One rule, and a grid is not part of it: a grid is asked for.

    Reading a shape and deciding the operator wanted cells is guessing at
    intent -- which is why WHICH grid a node opens is the node's own
    declaration, not something inferred here.
    """

    frames = _camera_frames(cycles=3, frames=2)
    assert isinstance(task_console_fitting_spec(frames, "", ""), ImagePlot)

    # A judged cycle proves two coordinates as well -- which frame, and which
    # site -- so it is a map, not thirty-five curves three points long.  One
    # coordinate comes from the point table and one from the cell; a picture
    # does not care which side of the dataset its axes live on.
    counts = _occupancy_counts(frames=3, sites=35)
    counts_spec = task_console_fitting_spec(counts, "", "")
    assert isinstance(counts_spec, ImagePlot), counts_spec
    assert counts_spec.x.axis_id == "cam.frames.frame"
    assert counts_spec.y.axis_id.endswith("site")

    # A scan of a scalar proves one coordinate, so it stays a curve.
    scalar = _scan_of_scalars(points=6)
    assert isinstance(task_console_fitting_spec(scalar, "", ""), CurvePlot)


def test_a_measurement_declares_the_plot_its_own_start_opens() -> None:
    """The node knows what it is about to measure; a shape does not.

    A cycle's frames are its point rows, one picture each, and that is as true
    of a cycle of one as of a cycle of three -- so the kind does not follow the
    count.  It used to, and then changing the count, or restarting the node,
    changed the shape of the panel underneath the operator.
    """

    from zlc_atom.nodes import discover_logic_nodes

    descriptors = {value.api_name: value for value in discover_logic_nodes()}
    camera = descriptors["camera_measurement"]
    assert tuple(
        (spec.output_name, spec.plot_kind) for spec in camera.node_previews
    ) == (
        ("frames", "facet_grid"),
    )

    # A processor answers someone else's signal; nothing about starting it
    # says which of its answers belongs on this board.
    assert descriptors["occupancy"].node_previews == ()
