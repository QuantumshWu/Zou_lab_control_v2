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


@pytest.mark.parametrize("holes", [0.0, 0.3, 0.995])
@pytest.mark.parametrize("dtype", [np.float64, np.uint8])
@pytest.mark.parametrize(
    "aggregation",
    (Reduction.MEAN, Reduction.SUM, Reduction.MIN, Reduction.MAX),
)
def test_factored_image_matches_the_generic_path(
    aggregation, dtype, holes
) -> None:
    """The heatmap twin: pixel for pixel against the generic aggregation."""

    view = DataView(_snapshot(dtype=dtype, holes=holes, seed=17))
    x, y = AxisRef.point("ax"), AxisRef.point("ay")
    fast = view._factored_image(x, y, aggregation)
    assert fast is not None, "the heatmap twin refused its own configuration"
    slow = view._image_from_positions(
        x, y, view._all_positions(), aggregation
    )
    np.testing.assert_array_equal(
        np.asarray(fast.x.canonical), np.asarray(slow.x.canonical)
    )
    np.testing.assert_array_equal(
        np.asarray(fast.y.canonical), np.asarray(slow.y.canonical)
    )
    np.testing.assert_array_equal(fast.valid, slow.valid)
    np.testing.assert_allclose(
        np.asarray(fast.z.canonical),
        np.asarray(slow.z.canonical),
        equal_nan=True,
        rtol=1e-12,
        atol=1e-12,
    )


def test_factored_image_fall_throughs() -> None:
    view = DataView(_snapshot(seed=21))
    x, y = AxisRef.point("ax"), AxisRef.point("ay")
    assert view._factored_image(x, y, Reduction.FIRST) is None
    assert (
        view._factored_image(
            AxisRef.data("frame"), AxisRef.data("site"), Reduction.MEAN
        )
        is None
    ), "two data axes belong to the dense image path"


@pytest.mark.parametrize("holes", [0.0, 0.3])
@pytest.mark.parametrize("uncertainty", [False, True])
def test_factored_facet_matches_the_generic_path(holes, uncertainty) -> None:
    """A lattice facet grid, cell for cell against the generic per-cell run."""

    from zlc_plot import CurvePlot, FacetGridPlot

    view = DataView(_snapshot(holes=holes, seed=29))
    for facet, group in (
        (AxisRef.data("frame"), None),
        (AxisRef.data("frame"), AxisRef.data("site")),
        (AxisRef.repeat(), AxisRef.data("site")),
    ):
        spec = FacetGridPlot(
            facet, CurvePlot(AxisRef.point("ax"), group=group)
        )
        fast = view._factored_facet(spec, uncertainty)
        assert fast is not None, (facet, group)
        slow = view._facet_from_positions(
            spec, None, view._all_positions(), uncertainty
        )
        assert len(fast.cells) == len(slow.cells)
        for ours, theirs in zip(fast.cells, slow.cells):
            assert ours.label == theirs.label
            assert ours.facet_index == theirs.facet_index
            _assert_same(ours.payload, theirs.payload)


def test_factored_facet_fall_throughs() -> None:
    from zlc_plot import CurvePlot, FacetGridPlot, HistogramPlot

    view = DataView(_snapshot(seed=31))
    assert (
        view._factored_facet(
            FacetGridPlot(AxisRef.data("frame"), HistogramPlot()), False
        )
        is None
    ), "histogram cells bin by value and keep their own paths"


@pytest.mark.parametrize("holes", [0.0, 0.3])
@pytest.mark.parametrize("uncertainty", [False, True])
def test_factored_row_facet_matches_the_generic_path(
    holes, uncertainty
) -> None:
    """Cells over a SCAN dimension: the combined row-key fold, cell for
    cell against the generic per-cell run -- including each cell's own
    used-set x domain."""

    from zlc_plot import CurvePlot, FacetGridPlot

    view = DataView(_snapshot(holes=holes, seed=37))
    for group in (None, AxisRef.data("site")):
        spec = FacetGridPlot(
            AxisRef.point_dimension("ax"),
            CurvePlot(AxisRef.point("ay"), group=group),
        )
        fast = view._factored_facet(spec, uncertainty)
        assert fast is not None, group
        slow = view._facet_from_positions(
            spec, None, view._all_positions(), uncertainty
        )
        assert len(fast.cells) == len(slow.cells)
        for ours, theirs in zip(fast.cells, slow.cells):
            assert ours.label == theirs.label
            assert ours.facet_index == theirs.facet_index
            _assert_same(ours.payload, theirs.payload)


def _assert_same_image(ours, theirs) -> None:
    np.testing.assert_array_equal(
        np.asarray(ours.x.canonical), np.asarray(theirs.x.canonical)
    )
    np.testing.assert_array_equal(
        np.asarray(ours.y.canonical), np.asarray(theirs.y.canonical)
    )
    np.testing.assert_array_equal(ours.valid, theirs.valid)
    np.testing.assert_allclose(
        np.asarray(ours.z.canonical),
        np.asarray(theirs.z.canonical),
        equal_nan=True,
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize("holes", [0.0, 0.3])
@pytest.mark.parametrize(
    "aggregation", (Reduction.MEAN, Reduction.SUM, Reduction.MIN)
)
def test_factored_facet_image_cells_match_the_generic_path(
    holes, aggregation
) -> None:
    """Heatmap cells over a DATA-axis facet: pixel for pixel per cell."""

    from zlc_plot import FacetGridPlot, ImagePlot

    view = DataView(_snapshot(holes=holes, seed=41))
    spec = FacetGridPlot(
        AxisRef.data("site"),
        ImagePlot(
            AxisRef.point_dimension("ax"),
            AxisRef.point_dimension("ay"),
            reduction=aggregation,
        ),
    )
    fast = view._factored_facet(spec, False)
    assert fast is not None
    slow = view._facet_from_positions(
        spec, None, view._all_positions(), False
    )
    assert len(fast.cells) == len(slow.cells)
    for ours, theirs in zip(fast.cells, slow.cells):
        assert ours.label == theirs.label
        assert ours.facet_index == theirs.facet_index
        _assert_same_image(ours.payload, theirs.payload)


def test_factored_row_facet_image_cells_compress_to_their_used_sets() -> None:
    """A scan-dimension facet of heatmap cells over a HOLED topology: each
    cell owns only its own present coordinates, exactly as the generic
    per-cell domains do."""

    from zlc_plot import FacetGridPlot, ImagePlot

    rng = np.random.default_rng(43)
    combos = [(i, j, k) for k in range(2) for j in range(3) for i in range(4)]
    cells = [c for c in combos if not (c[2] == 1 and c[0] == 3)]
    rows = len(cells)
    schema = DatasetSchema.create(
        Axis.create("repeat", size=5),
        PointTable.from_columns({
            "ax": np.asarray([float(c[0]) for c in cells]),
            "ay": np.asarray([float(c[1]) for c in cells]),
            "az": np.asarray([float(c[2]) for c in cells]),
        }),
        data_axes=(Axis.create("site", values=[0.0, 1.0, 2.0]),),
        dtype=np.float64,
        point_topology=PointTopology(
            (AxisId("ax"), AxisId("ay"), AxisId("az")),
            ((0.0, 1.0, 2.0, 3.0), (0.0, 1.0, 2.0), (0.0, 1.0)),
            tuple(cells),
        ),
    )
    shape = (5, rows, 3)
    validity = np.broadcast_to(
        rng.random(shape[:2] + (1,)) > 0.2, shape
    ).copy()
    view = DataView(
        DatasetSnapshot(schema, rng.normal(size=shape), 1, validity=validity)
    )
    spec = FacetGridPlot(
        AxisRef.point_dimension("az"),
        ImagePlot(
            AxisRef.point_dimension("ax"), AxisRef.point_dimension("ay")
        ),
    )
    fast = view._factored_facet(spec, False)
    assert fast is not None
    slow = view._facet_from_positions(
        spec, None, view._all_positions(), False
    )
    assert len(fast.cells) == len(slow.cells)
    for ours, theirs in zip(fast.cells, slow.cells):
        assert ours.label == theirs.label
        _assert_same_image(ours.payload, theirs.payload)
