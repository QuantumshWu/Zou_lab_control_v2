"""The dense facet path must agree with the generic path, cell for cell.

A facet over the repeat axis or a scan dimension slices whole rows, which
preserves the regularity the dense projections rely on -- so those cells
reduce through the same kernel as the single-kind dense paths instead of
materializing one (position, value) pair per sample.  A fast path that is
allowed to disagree with the path it replaces is not an optimisation, so
every cell payload is compared here, and each case first PROVES the dense
path engaged (otherwise this file would compare the generic path to
itself and never turn red).
"""

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
from zlc_data import OwnedSnapshot, REPEAT, SPATIAL_X, SPATIAL_Y
from zlc_plot import AxisRef, CurvePlot, FacetGridPlot, HistogramPlot, ImagePlot
from zlc_plot.data_view import DataView
from zlc_plot.specs import Reduction

def _scan_of_frames(repeat: int = 2) -> OwnedSnapshot:
    """A 3x2 Cartesian scan of 4x5 uint16 frames with some invalid cells."""

    point_domain = mapped_domain_from_columns(
        {
            "bias_x": np.repeat([-1.0, 0.0, 1.0], 2),
            "bias_y": np.tile([10.0, 20.0], 3),
        }
    )
    schema = make_dataset_schema(
        repeat_domain(size=repeat),
        point_domain,
        cell_axes=(
            axis("sy", values=[0.0, 1.0, 2.0, 3.0], role=SPATIAL_Y),
            axis("sx", values=[0.0, 1.0, 2.0, 3.0, 4.0], role=SPATIAL_X),
        ),
        dtype=np.uint16,
    )
    rng = np.random.default_rng(5)
    values = rng.integers(0, 4000, size=schema.physical_shape, dtype=np.uint16)
    cell_validity = rng.random(schema.physical_shape[:2]) > 0.2
    cell_validity[0, 0] = True  # at least one valid cell in the first facet
    validity = np.broadcast_to(
        cell_validity[..., None, None], schema.physical_shape
    ).copy()
    return make_snapshot(schema, values, revision=3, validity=validity)

def _assert_facets_equal(dense, generic) -> None:
    assert len(dense.cells) == len(generic.cells)
    for ours, theirs in zip(dense.cells, generic.cells):
        assert ours.facet_index == theirs.facet_index
        assert ours.label == theirs.label
        assert ours.facet_value_canonical == theirs.facet_value_canonical
        left, right = ours.payload, theirs.payload
        assert type(left) is type(right)
        if hasattr(left, "z"):
            np.testing.assert_array_equal(
                np.asarray(left.x.canonical), np.asarray(right.x.canonical)
            )
            np.testing.assert_array_equal(
                np.asarray(left.y.canonical), np.asarray(right.y.canonical)
            )
            np.testing.assert_array_equal(left.valid, right.valid)
            np.testing.assert_allclose(
                np.asarray(left.z.canonical, dtype=np.float64)[left.valid],
                np.asarray(right.z.canonical, dtype=np.float64)[right.valid],
                rtol=1e-12,
            )
        elif hasattr(left, "series"):
            assert len(left.series) == len(right.series) == 1
            np.testing.assert_array_equal(
                np.asarray(left.series[0].x.canonical),
                np.asarray(right.series[0].x.canonical),
            )
            np.testing.assert_array_equal(
                left.series[0].valid, right.series[0].valid
            )
            np.testing.assert_array_equal(
                left.series[0].counts, right.series[0].counts
            )
            np.testing.assert_allclose(
                np.asarray(left.series[0].y.canonical)[left.series[0].valid],
                np.asarray(right.series[0].y.canonical)[right.series[0].valid],
                rtol=1e-12,
            )
        else:
            np.testing.assert_array_equal(left.counts, right.counts)
            np.testing.assert_allclose(
                np.asarray(left.edges.canonical),
                np.asarray(right.edges.canonical),
            )

_IMAGE_CELL = ImagePlot(AxisRef.cell_data("sx"), AxisRef.cell_data("sy"))
_EDGES = tuple(float(edge) for edge in np.linspace(0.0, 4000.0, 7))

@pytest.mark.parametrize(
    "spec,bins",
    [
        (FacetGridPlot(AxisRef.point("bias_x"), _IMAGE_CELL), None),
        (FacetGridPlot(AxisRef.repeat("repeat"), _IMAGE_CELL), None),
        (
            FacetGridPlot(
                AxisRef.point("bias_x"),
                ImagePlot(
                    AxisRef.cell_data("sx"),
                    AxisRef.cell_data("sy"),
                    reduction=Reduction.MIN,
                ),
            ),
            None,
        ),
        (
            FacetGridPlot(
                AxisRef.point("bias_y"),
                ImagePlot(
                    AxisRef.cell_data("sx"),
                    AxisRef.cell_data("sy"),
                    reduction=Reduction.SUM,
                ),
            ),
            None,
        ),
        (
            FacetGridPlot(
                AxisRef.point("bias_x"),
                CurvePlot(AxisRef.cell_data("sx")),
            ),
            None,
        ),
        (FacetGridPlot(AxisRef.cell_data("sy"), HistogramPlot()), _EDGES),
        (
            FacetGridPlot(AxisRef.point("bias_x"), HistogramPlot()),
            _EDGES,
        ),
    ],
)
def test_dense_facet_equals_the_generic_path(spec, bins) -> None:
    view = DataView(_scan_of_frames())
    dense = view._factored_facet(spec, False)
    if dense is None and isinstance(spec.cell, HistogramPlot):
        dense = view._dense_histogram_facet(spec, bins)
    assert dense is not None, "a tensor/factored path must actually engage here"
    generic = view._facet_from_positions(spec, bins, view._all_positions())
    _assert_facets_equal(dense, generic)

@pytest.mark.parametrize(
    "spec,bins",
    [
        (FacetGridPlot(AxisRef.point("bias_x"), _IMAGE_CELL), None),
        (FacetGridPlot(AxisRef.cell_data("sy"), HistogramPlot()), _EDGES),
    ],
)
def test_facet_projection_takes_the_dense_tensor_path(
    monkeypatch, spec, bins
) -> None:
    """facet() must use a tensor/factored owner, not materialize positions."""

    view = DataView(_scan_of_frames())

    def forbidden(*_args, **_kwargs):
        raise AssertionError("facet allocated generic element positions")

    monkeypatch.setattr(DataView, "_all_positions", forbidden)
    view.facet(spec, bins=bins)
