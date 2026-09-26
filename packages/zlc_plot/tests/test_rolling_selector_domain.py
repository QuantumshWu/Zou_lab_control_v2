"""A rolling selector compares shot offsets with shot offsets.

The rolling x is "shots from latest" -- an axis no snapshot has a column
for -- so ``_x_ref()`` hands back a PLACEHOLDER AxisRef where the generic
selector code needs an AxisRef shape.  Reading that token's coordinates
gave point-row ORDINALS (0..P-1) to be compared against those offsets
(-(N-1)..0).  Zero was the only value the two domains shared: point rows
1..P-1 could never be selected, every crosshair candidate reported the
same x so the pick was decided by y alone, and a range drawn over the
older shots came back holding row 0 of every shot.

A record without a primary index carries its shots on the Repeat rows, and
each sample sits where its row is drawn.  Every sample at zero put a range
over the older shots on nothing and one over the latest on every shot.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from zlc_plot import HistogramPlot, PlotSession, RollingPlot, SelectorKind
from data_factory import (
    make_dataset_schema,
    make_snapshot,
    mapped_domain_from_columns,
    repeat_domain,
)
from zlc_data import INVALID, BlockId, OwnedSnapshot
from zlc_data.snapshot_projection import restrict_snapshot

def _snapshot(revision: int, repeats: int = 5, points: int = 3) -> OwnedSnapshot:
    schema = make_dataset_schema(
        repeat_domain(size=repeats),
        mapped_domain_from_columns({"x": np.arange(float(points))}),
        dtype=np.float64,
    )
    values = np.arange(repeats * points, dtype=float).reshape(repeats, points)
    return make_snapshot(schema, values, revision=revision)

def _session() -> PlotSession:
    return PlotSession(_snapshot(0), RollingPlot())

def test_every_sample_of_this_revision_is_drawn_by_the_curve() -> None:
    """Five shots, a window of a hundred: every shot is drawn."""

    session = _session()
    try:
        mask = session._projection._rolling_visible_mask()
        assert mask.all(), (
            "%d of %d samples were called invisible" % ((~mask).sum(), mask.size)
        )
    finally:
        session.close()

def _selected(session: PlotSession, low: float, high: float) -> list[float]:
    assert session.set_x_selector(low, high) is not None
    return sorted(
        float(value)
        for value in session.selector_data(SelectorKind.X_RANGE).canonical_values
    )

def test_a_range_selects_the_shots_it_covers_whole() -> None:
    """The two oldest shots at -4 and -3; the latest alone at 0."""

    session = _session()
    try:
        shots = np.asarray(session._payload.series[0].x.canonical)
        np.testing.assert_array_equal(shots, [-4.0, -3.0, -2.0, -1.0, 0.0])
        assert _selected(session, -4.0, -3.0) == [float(index) for index in range(6)]
        assert _selected(session, -0.5, 0.0) == [12.0, 13.0, 14.0]
    finally:
        session.close()

def test_a_live_finite_run_ends_on_the_last_shot_it_has_begun() -> None:
    """A live run declares all its repeats; three of six have arrived.

    Counted from the declared end, a window of two held none of them and
    "shots from latest" put the latest three shots before zero.
    """

    snapshot = _snapshot(0, repeats=6)
    rows = tuple(restrict_snapshot(
        snapshot, repeat_rows=range(repeat, repeat + 1),
        reference_for=lambda schema, repeat=repeat: replace(
            snapshot.ref, block_id=BlockId(f"run:repeat:{repeat}"),
            schema_fingerprint=schema.fingerprint,
        ),
    ).block.as_segment() for repeat in range(3))
    live = OwnedSnapshot(snapshot.ref, replace(
        snapshot.block, values=None, validity=INVALID, sigma=None, segments=rows,
        segment_origins=np.asarray([(repeat, 0) for repeat in range(3)], dtype=np.int64),
        segment_shapes=np.asarray([(1, 3)] * 3, dtype=np.int64),
    ))
    session = PlotSession(live, RollingPlot())
    try:
        shots = np.asarray(session._payload.series[0].x.canonical)
        np.testing.assert_array_equal(shots, [-2.0, -1.0, 0.0])
        assert _selected(session, -0.5, 0.0) == [6.0, 7.0, 8.0]
        session.set_parameters({"window": 2})
        shots = np.asarray(session._payload.series[0].x.canonical)
        np.testing.assert_array_equal(shots, [-1.0, 0.0])
        np.testing.assert_allclose(session._payload.series[0].y.canonical, [4.0, 7.0])
    finally:
        session.close()
    session = PlotSession(live, HistogramPlot(), parameters={"window": 2})
    try:
        assert int(np.asarray(session._payload.counts).sum()) == 6
    finally:
        session.close()
