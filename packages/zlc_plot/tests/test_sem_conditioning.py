"""The spread of a large number is still the spread it has.

``E[x^2] - mean^2`` subtracts two numbers that agree to as many digits as
the offset exceeds the spread.  A fitted resonance centre near 6.834 GHz
with a kilohertz scatter is exactly that shape: the two terms are near
4.7e19 and their difference is near 3.5e5, so six of sixteen digits are
gone before the clip at zero hides what is left.  Measured over eight
samples the variance came out 2.2 per cent wrong, and the error grows with
the ratio -- a linewidth on an absolute optical frequency would be noise.

The standard error does not depend on where the origin is, so the second pass
centres every bucket on that bucket's own first-pass mean and measures both
centred moments.  These pin that rule across the public projection routes.
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

from zlc_data import OwnedSnapshot
from zlc_plot import (
    AxisRef,
    CurvePlot,
    FacetGridPlot,
    PlotSession,
    Reduction,
)
from zlc_plot.data_view import DataView

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


def test_each_projection_centres_every_bucket_on_its_own_mean() -> None:
    """Far-apart buckets retain the small spread each one actually has."""

    offsets = np.asarray([-2.0, -1.0, 1.0, 2.0]) * 1e-3
    centres = np.asarray([0.0, 1.0e6])
    x = AxisRef.point("x")

    def wanted(samples: np.ndarray, axis: int = 0) -> np.ndarray:
        return np.std(samples, axis=axis, ddof=1) / np.sqrt(samples.shape[axis])

    # Dense tensor and the position-based generic oracle over the same data.
    dense_schema = make_dataset_schema(
        repeat_domain(size=offsets.size),
        mapped_domain_from_columns({"x": [0.0, 1.0]}),
        dtype=np.float64,
    )
    dense_values = centres[None, :] + offsets[:, None]
    dense_view = DataView(make_snapshot(dense_schema, dense_values, revision=0))
    expected = wanted(dense_values)
    dense = dense_view._dense_data_curve(x, (), Reduction.MEAN, True)
    assert dense is not None
    np.testing.assert_allclose(dense.series[0].sem, expected, rtol=1e-10)
    generic = dense_view._curve_from_positions(
        x, dense_view._all_positions(), (), Reduction.MEAN, True
    )
    np.testing.assert_allclose(generic.series[0].sem, expected, rtol=1e-10)

    # Mapped rows exercise both the factored fold and axis-code kernel.
    mapped_schema = make_dataset_schema(
        repeat_domain(size=offsets.size),
        mapped_domain_from_columns({"x": [0.0, 1.0, 0.0, 1.0]}),
        dtype=np.float64,
    )
    mapped_values = np.asarray([0.0, 1.0e6, 0.0, 1.0e6])[None, :]
    mapped_values = mapped_values + offsets[:, None]
    mapped_view = DataView(make_snapshot(mapped_schema, mapped_values, revision=0))
    mapped_expected = np.asarray([
        wanted(mapped_values[:, [0, 2]].reshape(-1), axis=0),
        wanted(mapped_values[:, [1, 3]].reshape(-1), axis=0),
    ])
    factored = mapped_view._factored_curve(x, (), Reduction.MEAN, True)
    assert factored is not None
    np.testing.assert_allclose(
        factored.series[0].sem, mapped_expected, rtol=1e-10
    )
    axis_kernel = mapped_view._curve_from_axes(
        x, (), Reduction.MEAN, uncertainty=True
    )
    assert axis_kernel is not None
    np.testing.assert_allclose(
        axis_kernel.series[0].sem, mapped_expected, rtol=1e-10
    )

    # Rolling's regular tensor and irregular repeat routes use the same rule.
    sites = np.asarray([-1.0, 0.0, 1.0]) * 1e-3
    rolling_values = (
        centres[None, :, None]
        + offsets[:, None, None]
        + sites[None, None, :]
    )
    rolling_schema = make_dataset_schema(
        repeat_domain(size=offsets.size),
        mapped_domain_from_columns({"x": [0.0, 1.0]}),
        cell_axes=(axis("site", values=[0.0, 1.0, 2.0]),),
        dtype=np.float64,
    )
    rolling = DataView(
        make_snapshot(rolling_schema, rolling_values, revision=0)
    ).rolling_history(group=x, uncertainty=True)
    rolling_expected = np.std(rolling_values, axis=2, ddof=1) / np.sqrt(
        sites.size
    )
    np.testing.assert_allclose(rolling.sem, rolling_expected, rtol=1e-10)

    irregular_values = np.repeat(rolling_values, 2, axis=1)
    irregular_schema = make_dataset_schema(
        repeat_domain(size=offsets.size),
        mapped_domain_from_columns({"x": [0.0, 0.0, 1.0, 1.0]}),
        cell_axes=(axis("site", values=[0.0, 1.0, 2.0]),),
        dtype=np.float64,
    )
    irregular = DataView(
        make_snapshot(irregular_schema, irregular_values, revision=0)
    ).rolling_history(group=x, uncertainty=True)
    irregular_expected = np.stack(
        [
            np.std(irregular_values[:, :2, :], axis=(1, 2), ddof=1),
            np.std(irregular_values[:, 2:, :], axis=(1, 2), ddof=1),
        ],
        axis=1,
    ) / np.sqrt(6.0)
    np.testing.assert_allclose(irregular.sem, irregular_expected, rtol=1e-10)

    # Facet retains its cell and x axes in one dense reduction.
    facet_values = dense_values[:, :, None] + np.asarray([0.0, 0.5])[None, None, :]
    facet_schema = make_dataset_schema(
        repeat_domain(size=offsets.size),
        mapped_domain_from_columns({"x": [0.0, 1.0]}),
        cell_axes=(axis("facet", values=[0.0, 1.0]),),
        dtype=np.float64,
    )
    facet = DataView(
        make_snapshot(facet_schema, facet_values, revision=0)
    ).facet(
        FacetGridPlot(AxisRef.cell_data("facet"), CurvePlot(x)),
        uncertainty=True,
    )
    for index, cell in enumerate(facet.cells):
        np.testing.assert_allclose(
            cell.payload.series[0].sem,
            wanted(facet_values[:, :, index]),
            rtol=1e-10,
        )


def test_far_apart_constant_buckets_have_exactly_zero_sem() -> None:
    repeats = 12
    values = np.broadcast_to(np.asarray([0.0, 1.0e6]), (repeats, 2)).copy()
    schema = make_dataset_schema(
        repeat_domain(size=repeats),
        mapped_domain_from_columns({"x": [0.0, 1.0]}),
        dtype=np.float64,
    )
    view = DataView(make_snapshot(schema, values, revision=0))
    dense = view._dense_data_curve(
        AxisRef.point("x"), (), Reduction.MEAN, True
    )
    assert dense is not None
    np.testing.assert_array_equal(dense.series[0].sem, np.zeros(2))
    generic = view._curve_from_positions(
        AxisRef.point("x"), view._all_positions(), (), Reduction.MEAN, True
    )
    np.testing.assert_array_equal(generic.series[0].sem, np.zeros(2))

def test_a_constant_bucket_does_not_turn_one_roundoff_bit_into_a_sem() -> None:
    """A one-ulp moment residual is numerical zero, not extreme confidence.

    Even centred moments can subtract two equal non-binary squares when the
    first-pass mean rounded away from a constant sample.  If their last bit
    rounds in opposite directions, clipping only negative residuals leaves a
    fake positive SEM near 1e-9.
    """

    from zlc_plot.data_view import _sem_from_moments

    mean = np.asarray([-0.21644806294627436])
    square = np.square(mean)
    mean_of_squares = np.nextafter(square, np.inf)
    sem = _sem_from_moments(mean, mean_of_squares, np.asarray([20]))
    assert sem[0] == 0.0

    distinguishable = square + (
        128.0 * np.finfo(np.float64).eps * (np.abs(square) + square)
    )
    assert _sem_from_moments(mean, distinguishable, np.asarray([20]))[0] > 0.0

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
