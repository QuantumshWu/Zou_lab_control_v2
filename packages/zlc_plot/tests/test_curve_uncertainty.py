"""The MEAN reduction's companion standard error.

One SEM definition serves every column: for a boolean occupancy column the
sample spread sqrt(p(1-p)) IS the binomial spread, so sem = s/sqrt(n) needs
no binomial special case.  These tests pin that identity mechanically, pin
dense/generic path agreement, and pin that not asking keeps sem absent.
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
from zlc_data import OwnedSnapshot, REPEAT
from zlc_plot.data_view import DataView
from zlc_plot.kinds import AxisRef
from zlc_plot.specs import Reduction


def _snapshot(values: np.ndarray, *, x: list[float], repeats: int) -> OwnedSnapshot:
    point_domain = mapped_domain_from_columns({"x": x})
    schema = make_dataset_schema(
        repeat_domain(size=repeats),
        point_domain,
        dtype=np.float64,
    )
    return make_snapshot(schema, values.reshape(schema.physical_shape), revision=0)


def _expected_sem(samples: np.ndarray) -> float:
    return float(np.std(samples, ddof=1) / np.sqrt(samples.size))


def test_boolean_column_sem_is_the_binomial_error() -> None:
    """occupied in {0,1}: sem == sqrt((E[x^2]-p^2)/(n-1)) == sqrt(p(1-p)/(n-1))."""

    rng = np.random.default_rng(7)
    repeats, points = 40, 3
    values = (rng.random((repeats, points)) < 0.35).astype(np.float64)
    snapshot = _snapshot(values, x=[0.0, 1.0, 2.0], repeats=repeats)
    series = DataView(snapshot).curve(AxisRef.point("x"), uncertainty=True).series[0]
    assert series.sem is not None
    for column in range(points):
        shots = values[:, column]
        p_hat = shots.mean()
        binomial = np.sqrt(p_hat * (1.0 - p_hat) / (repeats - 1))
        np.testing.assert_allclose(series.sem[column], binomial, rtol=1e-12)
        np.testing.assert_allclose(series.sem[column], _expected_sem(shots), rtol=1e-12)


def test_continuous_column_sem_matches_sample_standard_error() -> None:
    rng = np.random.default_rng(11)
    repeats, points = 25, 4
    values = rng.normal(3.0, 0.7, size=(repeats, points))
    snapshot = _snapshot(values, x=[0.0, 1.0, 2.0, 3.0], repeats=repeats)
    series = DataView(snapshot).curve(AxisRef.point("x"), uncertainty=True).series[0]
    for column in range(points):
        np.testing.assert_allclose(
            series.sem[column], _expected_sem(values[:, column]), rtol=1e-12
        )


def test_dense_and_generic_paths_agree_on_sem() -> None:
    """The dense tensor path and the position path are the same statistic."""

    rng = np.random.default_rng(3)
    repeats = 12
    scan = axis("scan", values=[10.0, 20.0, 30.0])
    point_domain = mapped_domain_from_columns({"x": [0.0, 1.0]})
    schema = make_dataset_schema(
        repeat_domain(size=repeats),
        point_domain,
        cell_axes=(scan,),
        dtype=np.float64,
    )
    values = rng.normal(size=schema.physical_shape)
    snapshot = make_snapshot(schema, values, revision=0)
    view = DataView(snapshot)
    dense = view.curve(AxisRef.cell_data("scan"), uncertainty=True).series[0]
    generic = view.curve(
        AxisRef.cell_data("scan"), group_by=(), aggregation=Reduction.MEAN,
        uncertainty=True,
    )
    # Force the generic path via a grouped projection over a single-value
    # group: same buckets, generic machinery.
    np.testing.assert_allclose(
        dense.sem,
        [
            _expected_sem(values[:, :, column].reshape(-1))
            for column in range(3)
        ],
        rtol=1e-12,
    )
    assert generic.series[0].sem is not None
    np.testing.assert_allclose(generic.series[0].sem, dense.sem, rtol=1e-12)


def test_single_sample_bucket_reports_nan_not_zero() -> None:
    values = np.arange(2.0).reshape(1, 2)
    snapshot = _snapshot(values, x=[0.0, 1.0], repeats=1)
    series = DataView(snapshot).curve(AxisRef.point("x"), uncertainty=True).series[0]
    assert np.all(np.isnan(np.asarray(series.sem)))


def test_uncertainty_refuses_non_mean_reductions() -> None:
    values = np.zeros((4, 2))
    snapshot = _snapshot(values, x=[0.0, 1.0], repeats=4)
    with pytest.raises(ValueError, match="Reduction.MEAN only|mean"):
        DataView(snapshot).curve(
            AxisRef.point("x"),
            aggregation=Reduction.MIN,
            uncertainty=True,
        )


def test_sem_is_absent_unless_requested() -> None:
    values = np.zeros((4, 2))
    snapshot = _snapshot(values, x=[0.0, 1.0], repeats=4)
    series = DataView(snapshot).curve(AxisRef.point("x")).series[0]
    assert series.sem is None
