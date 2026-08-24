"""The lattice curve path, held to the generic path series for series.

``_factored_curve`` reduces the pooled dimensions as tensor axes and folds
only a (rows x series) residue, where the generic path builds per-sample
bucket codes.  Two algorithms, ONE contract: for every configuration the
fast path accepts, its output must match ``_curve_from_positions`` -- keys,
labels, x coordinates and counts exactly, values to float tolerance (the
summation orders differ, pairwise against sequential).  Configurations it
must NOT accept fall through, so nothing silently draws a different curve.
"""

from __future__ import annotations

import numpy as np
import pytest

from data_factory import (
    Axis,
    DatasetSchema,
    DatasetSnapshot,
    PointTable,
    PointTopology,
)
from zlc_data import AxisId
from zlc_plot import AxisRef, Reduction
from zlc_plot.data_view import DataView


def _snapshot(*, dtype=np.float64, holes: float = 0.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    R, F, S = 4, 3, 5
    cells = [(i % 6, (i // 6) % 4) for i in range(24)]
    schema = DatasetSchema.create(
        Axis.create("repeat", size=R),
        PointTable.from_columns({
            "ax": np.asarray([float(c[0]) for c in cells]),
            "ay": np.asarray([float(c[1]) for c in cells]),
        }),
        data_axes=(
            Axis.create("frame", values=[0.0, 1.0, 2.0]),
            Axis.create("site", values=[float(i) for i in range(S)]),
        ),
        dtype=np.dtype(dtype),
        point_topology=PointTopology(
            (AxisId("ax"), AxisId("ay")),
            (tuple(float(i) for i in range(6)), tuple(float(i) for i in range(4))),
            tuple(cells),
        ),
    )
    shape = (R, len(cells), F, S)
    if np.dtype(dtype).kind == "u":
        values = rng.integers(0, 200, size=shape).astype(dtype)
    else:
        values = rng.normal(size=shape).astype(dtype)
    validity = None
    if holes:
        # Holes at the (repeat, row) level, broadcast across the cell --
        # the shape the validity contract declares, and the physical one:
        # a shot is judged as a shot.
        validity = np.broadcast_to(
            rng.random(shape[:2] + (1, 1)) > holes, shape
        ).copy()
    return DatasetSnapshot(schema, values, revision=1, validity=validity)


def _assert_same(fast, slow):
    assert fast is not None, "the lattice path refused a configuration it owns"
    assert len(fast.series) == len(slow.series), (
        [s.group_key for s in fast.series],
        [s.group_key for s in slow.series],
    )
    for ours, theirs in zip(fast.series, slow.series):
        assert ours.label == theirs.label
        assert tuple(item.label for item in ours.group_key) == tuple(
            item.label for item in theirs.group_key
        )
        np.testing.assert_array_equal(
            np.asarray(ours.x.canonical), np.asarray(theirs.x.canonical)
        )
        np.testing.assert_array_equal(ours.counts, theirs.counts)
        np.testing.assert_array_equal(ours.valid, theirs.valid)
        np.testing.assert_allclose(
            np.asarray(ours.y.canonical),
            np.asarray(theirs.y.canonical),
            equal_nan=True,
            rtol=1e-12,
            atol=1e-12,
        )
        if theirs.sem is None:
            assert ours.sem is None
        else:
            np.testing.assert_allclose(
                ours.sem, theirs.sem, equal_nan=True, rtol=1e-12, atol=1e-12
            )


GROUPINGS = (
    (),
    (AxisRef.data("site"),),
    (AxisRef.data("frame"), AxisRef.data("site")),
    (AxisRef.repeat(),),
    (AxisRef.repeat(), AxisRef.data("site")),
)


@pytest.mark.parametrize("holes", [0.0, 0.3, 0.995])
@pytest.mark.parametrize("dtype", [np.float64, np.uint8])
@pytest.mark.parametrize(
    "aggregation",
    (Reduction.MEAN, Reduction.SUM, Reduction.MIN, Reduction.MAX),
)
def test_every_owned_configuration_matches_the_generic_path(
    aggregation, dtype, holes
) -> None:
    view = DataView(_snapshot(dtype=dtype, holes=holes, seed=11))
    x = AxisRef.point("ax")
    for groups in GROUPINGS:
        fast = view._factored_curve(x, groups, aggregation)
        slow = view._curve_from_positions(
            x, view._all_positions(), groups, aggregation
        )
        _assert_same(fast, slow)


def test_uncertainty_matches_including_the_binomial_case() -> None:
    view = DataView(_snapshot(holes=0.2, seed=5))
    x = AxisRef.point("ay")
    for groups in GROUPINGS:
        fast = view._factored_curve(x, groups, Reduction.MEAN, True)
        slow = view._curve_from_positions(
            x, view._all_positions(), groups, Reduction.MEAN, True
        )
        _assert_same(fast, slow)


def test_configurations_the_path_does_not_own_fall_through() -> None:
    view = DataView(_snapshot(seed=3))
    x = AxisRef.point("ax")
    assert view._factored_curve(x, (), Reduction.FIRST) is None
    assert (
        view._factored_curve(AxisRef.data("site"), (), Reduction.MEAN)
        is None
    ), "a data-axis x belongs to the dense path, not this one"
    assert (
        view._factored_curve(x, (AxisRef.point("ay"),), Reduction.MEAN)
        is None
    ), "a point-domain group keeps the generic path"


def test_the_public_curve_entry_uses_the_lattice_path(monkeypatch) -> None:
    """The fast path must actually serve curve(); a fast path nothing
    dispatches to is a test fixture, not a fix."""

    view = DataView(_snapshot(seed=9))
    calls = []
    original = DataView._factored_curve

    def spy(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        calls.append(result is not None)
        return result

    monkeypatch.setattr(DataView, "_factored_curve", spy)
    view.curve(AxisRef.point("ax"), group_by=(AxisRef.data("site"),))
    assert calls == [True]
