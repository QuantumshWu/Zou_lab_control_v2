"""The spread of a large number is still the spread it has.

``E[x^2] - mean^2`` subtracts two numbers that agree to as many digits as
the offset exceeds the spread.  A fitted resonance centre near 6.834 GHz
with a kilohertz scatter is exactly that shape: the two terms are near
4.7e19 and their difference is near 3.5e5, so six of sixteen digits are
gone before the clip at zero hides what is left.  Measured over eight
samples the variance came out 2.2 per cent wrong, and the error grows with
the ratio -- a linewidth on an absolute optical frequency would be noise.

The standard error does not depend on where the origin is, so the fix is to
put the origin near the data.  These pin that it is there.
"""

from __future__ import annotations

import numpy as np

from data_factory import Axis, DatasetSchema, DatasetSnapshot, PointTable
from zlc_plot import AxisRef, CurvePlot, PlotSession


def _resonance_snapshot(
    *, centre: float, scatter: float, repeats: int = 8, points: int = 5
) -> DatasetSnapshot:
    """A scan whose y is a big number with a small spread."""

    rng = np.random.default_rng(20260828)
    schema = DatasetSchema.create(
        Axis.create("repeat", size=repeats),
        PointTable.from_columns({"x": list(range(points))}),
        dtype=np.float64,
        canonical_unit="Hz",
        generation="sem-conditioning",
    )
    values = centre + rng.normal(0.0, scatter, size=(repeats, points))
    return DatasetSnapshot(schema, values, revision=0), values


def _drawn_sem(centre: float, scatter: float):
    snapshot, values = _resonance_snapshot(centre=centre, scatter=scatter)
    session = PlotSession(
        snapshot,
        CurvePlot(AxisRef.point("x")),
        parameters={"uncertainty": True},
    )
    try:
        session._renderer.draw()
        series = session._projection._payload.series[0]
        return np.asarray(series.sem, dtype=float), values
    finally:
        session.close()


def test_a_large_offset_does_not_eat_the_spread() -> None:
    """The sem of a GHz-scale value with a kHz spread is the kHz spread."""

    centre, scatter = 6.834e9, 1.0e3
    sem, values = _drawn_sem(centre, scatter)
    # numpy's own two-pass answer, per point, over the repeats.
    wanted = values.std(axis=0, ddof=1) / np.sqrt(values.shape[0])
    assert np.all(np.isfinite(sem)), sem
    np.testing.assert_allclose(sem, wanted, rtol=1e-9)


def test_the_answer_does_not_depend_on_where_zero_is() -> None:
    """Translation invariance, which the arithmetic must not break.

    The same spread around 1.0 and around 6.834e9 is the same spread.  A
    formula that squares about zero says otherwise, and says it more the
    further from zero the data sits.
    """

    scatter = 1.0e3
    near, _ = _drawn_sem(1.0e3, scatter)
    far, _ = _drawn_sem(6.834e9, scatter)
    np.testing.assert_allclose(near, far, rtol=1e-9)
