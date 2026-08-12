"""What a panel opens as when nobody named a kind.

The rule is one question, asked of the data: would a single flat plot have
to average something real away?  A cycle of one frame is a picture; a cycle
of two frames is two pictures.  A one-dimensional scan of scalars is a
curve; a scan of frames is one picture per point.

These are the shapes the bench actually publishes -- camera_measurement's
frames and the scan nodes' datasets -- built through their own production
helpers so this stays a statement about the product, not about a fixture.
"""

from __future__ import annotations

import numpy as np
import pytest

from zlc_data import SPATIAL_X, SPATIAL_Y
from zlc_plot import CurvePlot, FacetGridPlot, ImagePlot
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


def test_a_single_frame_cycle_is_a_picture_and_a_two_frame_cycle_is_a_grid() -> None:
    single = _auto(_camera_frames(cycles=3, frames=1))
    assert isinstance(single, ImagePlot), single

    pair = _auto(_camera_frames(cycles=3, frames=2))
    assert isinstance(pair, FacetGridPlot), pair
    assert isinstance(pair.cell, ImagePlot)


def test_a_one_dimensional_scan_of_scalars_is_the_curve_it_looks_like() -> None:
    """The one shape a flat plot shows completely: one axis, one number."""

    spec = _auto(_scan(_scalar_source(), axes={"da_bias_x": 3}, visits=2))
    assert isinstance(spec, CurvePlot), spec


def test_every_other_scan_opens_as_a_grid_instead_of_averaging_itself() -> None:
    """A scan of frames is one picture per point, not one picture."""

    frames = _camera_frames(cycles=1, frames=1)

    one_axis = _auto(_scan(frames, axes={"da_bias_x": 3}, visits=2))
    assert isinstance(one_axis, FacetGridPlot), one_axis
    assert isinstance(one_axis.cell, ImagePlot)

    two_axes = _auto(
        _scan(frames, axes={"da_bias_x": 3, "da_bias_y": 2}, visits=2)
    )
    assert isinstance(two_axes, FacetGridPlot), two_axes
    assert isinstance(two_axes.cell, ImagePlot)

    # Frames per point AND points per scan: still one cell per thing measured.
    cycles = _camera_frames(cycles=1, frames=2)
    scanned_frames = _auto(_scan(cycles, axes={"da_bias_x": 3}, visits=1))
    assert isinstance(scanned_frames, FacetGridPlot), scanned_frames


def test_a_two_dimensional_scalar_scan_keeps_every_point_it_measured() -> None:
    """Two scan axes and one number per point: the map, or a grid of maps.

    Either way nothing is pooled that the operator did not ask to pool --
    which is the whole question the automatic kind is answering.
    """

    swept_once = _auto(
        _scan(_scalar_source(), axes={"a": 3, "b": 4}, visits=1)
    )
    assert isinstance(swept_once, ImagePlot), swept_once

    swept_again = _auto(
        _scan(_scalar_source(), axes={"a": 3, "b": 4}, visits=3)
    )
    assert isinstance(swept_again, FacetGridPlot), swept_again
    assert isinstance(swept_again.cell, ImagePlot)


def test_a_one_dimensional_scan_of_per_site_values_stays_one_curve_family() -> None:
    """One scan axis and a vector per point is still one flat picture."""

    from zlc_atom.data import snapshot_from_array

    from zlc_data import SITE

    sites = snapshot_from_array(
        np.zeros((1, 6), dtype=np.float64),
        producer="occupancy",
        signal="counts",
        roles=(SITE,),
        generation="auto-kind",
        revision=1,
    ).block.schema
    spec = _auto(_scan(sites, axes={"da_bias_x": 3}, visits=1))
    assert isinstance(spec, CurvePlot), spec


def test_only_a_node_that_owns_its_shot_offers_to_open_a_plot() -> None:
    """Start opens what the node came to show; a processor came to answer.

    Occupancy reacts to a camera it does not own, and which of its five
    answers belongs on this board is a decision about the board -- made by
    adding the panel, not by the node's own declaration.
    """

    from zlc_atom.nodes import discover_logic_nodes
    from zlc_atom.nodes._framework.descriptor import (
        LogicNodeDescriptor,
        NodeKind,
        NodePreviewSpec,
        OutputSpec,
    )
    from zlc_atom.authoring import AuthoringSchema

    declared = {
        descriptor.api_name: tuple(
            spec.output_name for spec in descriptor.node_previews
        )
        for descriptor in discover_logic_nodes()
    }
    assert declared["occupancy"] == ()
    assert declared["camera_measurement"] == ("frames",)
    assert declared["seamless_scan"] == ("scan",)
    assert declared["stepped_scan"] == ("scan",)

    with pytest.raises(ValueError, match="processor"):
        LogicNodeDescriptor(
            "invented",
            NodeKind.PROCESSOR,
            AuthoringSchema(()),
            outputs=(OutputSpec("rate", "invented.rate.v1"),),
            node_previews=(NodePreviewSpec("rate"),),
        )
