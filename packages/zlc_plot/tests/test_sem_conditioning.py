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

from data_factory import (
    make_dataset_schema,
    make_snapshot,
    mapped_domain_from_columns,
    repeat_domain,
)

from zlc_data import OwnedSnapshot
from zlc_plot import AxisRef, CurvePlot, PlotSession

def _resonance_snapshot(
    *, centre: float, scatter: float, repeats: int = 8, points: int = 5
) -> OwnedSnapshot:
    """A scan whose y is a big number with a small spread."""

    rng = np.random.default_rng(20260828)
    schema = make_dataset_schema(
        repeat_domain(size=repeats),
        mapped_domain_from_columns({"x": list(range(points))}),
        dtype=np.float64,
        value_unit="Hz",
    )
    values = centre + rng.normal(0.0, scatter, size=(repeats, points))
    return make_snapshot(schema, values, revision=0), values

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

def test_the_mean_of_samples_that_carry_their_own_error() -> None:
    """The scatter already contains the errors; the sigma fills its silence.

    Var(m) = (<sigma_i^2> + sigma_pop^2)/n, and E[s^2] is exactly that
    numerator -- so s^2/n is unbiased for the WHOLE of it and adding the
    measurement error again would count it twice.  Substituting the clipped
    moment estimate of sigma_pop^2 instead gives max(<sigma_i^2>, s^2)/n,
    which is the rule physics often uses and is biased high: measured over
    400k Monte Carlo buckets with no genuine variation, 22 per cent high in
    the error bar at n = 2.

    So the per-sample sigma is used where the scatter cannot speak at all.
    """

    from zlc_plot.data_view import _sem_from_moments

    def sem(values, sigmas=None):
        values = np.asarray(values, dtype=float)
        n = np.asarray([values.size])
        mean = np.asarray([values.mean()])
        mean_sq = np.asarray([(values ** 2).mean()])
        sigma_sq = (
            None
            if sigmas is None
            else np.asarray([(np.asarray(sigmas, dtype=float) ** 2).mean()])
        )
        return float(_sem_from_moments(mean, mean_sq, n, sigma_sq)[0])

    values = [10.0, 11.0, 12.0, 13.0]
    scatter = float(np.std(values, ddof=1) / np.sqrt(len(values)))

    # 1. NO per-sample sigma: exactly what it always was.
    assert sem(values) == scatter
    assert np.isnan(sem([7.0]))

    # 2 and 3. WITH per-sample sigmas, large or small, a bucket that HAS a
    #    scatter answers with it: the scatter already contains them, and
    #    taking the larger of the two biases the bar high.
    assert sem(values, [1e-6] * 4) == scatter
    assert sem(values, [50.0] * 4) == scatter

    # 4. ONE sample that knows its own error -- the shape of one fit per
    #    shot, where the answer used to be NaN with the error sitting in a
    #    signal beside it.
    assert sem([6.834e9], [1.0e3]) == 1.0e3
