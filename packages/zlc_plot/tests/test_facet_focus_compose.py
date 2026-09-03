"""A focused FacetGrid's composed frame equals a full draw, pixel for pixel.

The compose path restores a cached background and repaints dynamic artists;
its contract is full-draw-exact stacking.  Focus is the one layout that
HIDES axes, and the compose loop used to repaint their image artists anyway
-- every hidden cell kept ghosting into the focused view at its old box.
"""

from __future__ import annotations

import numpy as np

from data_factory import (
    axis,
    make_dataset_schema,
    make_snapshot,
    mapped_domain_from_columns,
    repeat_domain,
)
from zlc_data import OwnedSnapshot, REPEAT, SPATIAL_X, SPATIAL_Y
from zlc_plot import AxisRef, FacetGridPlot, ImagePlot
from zlc_plot.session import PlotSession

def _scan_snapshot() -> OwnedSnapshot:
    table = mapped_domain_from_columns({"bias": [-1.0, 0.0, 1.0]})
    schema = make_dataset_schema(
        repeat_domain(size=1),
        table,
        cell_axes=(
            axis(
                "sy", values=tuple(float(v) for v in range(40)), role=SPATIAL_Y
            ),
            axis(
                "sx", values=tuple(float(v) for v in range(60)), role=SPATIAL_X
            ),
        ),
        dtype=np.uint8,
    )
    values = np.zeros((1, 3, 40, 60), dtype=np.uint8)
    xx = np.mgrid[0:40, 0:60][1]
    # Distinct, bright patterns in the cells that get HIDDEN on focus, so a
    # ghost of either is unmissable in a pixel comparison.
    values[0, 0] = np.linspace(0, 255, 60)[None, :].astype(np.uint8)
    values[0, 2] = ((xx // 6 % 2) * 255).astype(np.uint8)
    return make_snapshot(schema, values, revision=1)

_SPEC = FacetGridPlot(
    AxisRef.point("bias"),
    ImagePlot(AxisRef.cell_data("sx"), AxisRef.cell_data("sy")),
)

def _frame(session: PlotSession) -> np.ndarray:
    raw, height, width = session._raster_capture_rgba_bytes()
    return np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 4).copy()

def _assert_native_roundtrip(actual: np.ndarray, expected: np.ndarray) -> None:
    delta = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
    changed = np.any(delta != 0, axis=2)
    assert int(delta.max()) <= 2
    assert float(np.count_nonzero(changed)) / changed.size <= 0.10

def test_focused_compose_equals_a_full_draw() -> None:
    session = PlotSession(_scan_snapshot(), _SPEC, size="4x4")
    try:
        session.focus_facet(1)
        composed = _frame(session)
        # The ground truth: a plain full figure draw, which skips hidden
        # axes together with everything on them.  EXACT equality: any
        # tolerance here is a place for the next ghost to hide -- the first
        # fix stopped at the images and a "sparse antialias tail" that was
        # in fact the hidden cells' tick marks.
        session._renderer.draw()
        reference = _frame(session)
        np.testing.assert_array_equal(composed, reference)
    finally:
        session.close()

def test_overview_after_focus_shows_every_cell_again() -> None:
    session = PlotSession(_scan_snapshot(), _SPEC, size="4x4")
    try:
        before = _frame(session)
        session.focus_facet(1)
        session.show_facet_overview()
        after = _frame(session)
        _assert_native_roundtrip(after, before)
    finally:
        session.close()
