"""A rolling selector compares shot offsets with shot offsets.

The rolling x is "shots from latest" -- an axis no snapshot has a column
for -- so ``_x_ref()`` hands back a PLACEHOLDER AxisRef where the generic
selector code needs an AxisRef shape.  Reading that token's coordinates
gave point-row ORDINALS (0..P-1) to be compared against those offsets
(-(N-1)..0).  Zero was the only value the two domains shared: point rows
1..P-1 could never be selected, every crosshair candidate reported the
same x so the pick was decided by y alone, and a range drawn over the
older shots came back holding row 0 of every shot.
"""

from __future__ import annotations

import numpy as np

from zlc_plot import PlotSession, RollingPlot, SelectorKind
from data_factory import (
    make_dataset_schema,
    make_snapshot,
    mapped_domain_from_columns,
    repeat_domain,
)
from zlc_data import OwnedSnapshot

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
    """They share one shot, and the window always ends on it."""

    session = _session()
    try:
        mask = session._projection._rolling_visible_mask()
        assert mask.all(), (
            "%d of %d samples were called invisible" % ((~mask).sum(), mask.size)
        )
    finally:
        session.close()

def test_a_range_over_older_shots_selects_nothing_from_this_revision() -> None:
    """The older points are history entries, not samples of this snapshot."""

    session = _session()
    try:
        shots = np.asarray(session._payload.series[0].x.canonical)
        assert float(shots.max()) == 0.0
        oldest = float(shots.min())
        assert session.set_x_selector(oldest, oldest + 1.0) is not None
        assert list(session.selector_data(SelectorKind.X_RANGE).canonical_values) == []
    finally:
        session.close()

def test_a_range_covering_the_latest_shot_selects_all_of_it() -> None:
    """Not one point row out of every shot: this shot, whole."""

    session = _session()
    try:
        assert session.set_x_selector(-1.0, 0.0) is not None
        selected = sorted(float(value) for value in session.selector_data(SelectorKind.X_RANGE).canonical_values)
        assert selected == [float(index) for index in range(15)], selected
    finally:
        session.close()
