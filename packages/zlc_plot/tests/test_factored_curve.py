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
    axis,
    make_dataset_schema,
    make_snapshot,
    mapped_domain_from_columns,
    repeat_domain,
)
from zlc_data import REPEAT, SITE
from zlc_plot import AxisRef, Reduction
from zlc_plot.data_view import DataView

def _snapshot(*, dtype=np.float64, holes: float = 0.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    R, F, S = 4, 3, 5
    cells = [(i % 6, (i // 6) % 4) for i in range(24)]
    schema = make_dataset_schema(
        repeat_domain(size=R),
        mapped_domain_from_columns({
            "ax": np.asarray([float(c[0]) for c in cells]),
            "ay": np.asarray([float(c[1]) for c in cells]),
        }),
        cell_axes=(
            axis("frame", values=[0.0, 1.0, 2.0]),
            axis("site", values=[float(i) for i in range(S)], role=SITE),
        ),
        dtype=np.dtype(dtype),
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
    return make_snapshot(schema, values, revision=1, validity=validity)

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
    (AxisRef.cell_data("site"),),
    (AxisRef.cell_data("frame"), AxisRef.cell_data("site")),
    (AxisRef.repeat("repeat"),),
    (AxisRef.repeat("repeat"), AxisRef.cell_data("site")),
    (AxisRef.point("ay"),),
    (AxisRef.point("ay"), AxisRef.cell_data("site")),
    (AxisRef.cell_data("site"), AxisRef.point("ay")),
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

@pytest.mark.parametrize("holes", [0.0, 0.3])
@pytest.mark.parametrize("dtype", [np.float64, np.uint8])
@pytest.mark.parametrize(
    "aggregation",
    (Reduction.MEAN, Reduction.SUM, Reduction.MIN, Reduction.MAX),
)
def test_tensor_x_and_group_axes_match_the_generic_path(
    aggregation, dtype, holes
) -> None:
    view = DataView(_snapshot(dtype=dtype, holes=holes, seed=7))
    for x, groups in (
        (AxisRef.cell_data("frame"), (AxisRef.cell_data("site"),)),
        (AxisRef.cell_data("frame"), (AxisRef.repeat("repeat"),)),
        (AxisRef.repeat("repeat"), (AxisRef.cell_data("site"),)),
    ):
        fast = view._dense_data_curve(x, groups, aggregation)
        slow = view._curve_from_positions(
            x, view._all_positions(), groups, aggregation
        )
        _assert_same(fast, slow)

def test_tensor_x_and_group_uncertainty_matches() -> None:
    view = DataView(_snapshot(holes=0.2, seed=13))
    for x, groups in (
        (AxisRef.cell_data("frame"), (AxisRef.cell_data("site"),)),
        (AxisRef.cell_data("frame"), (AxisRef.repeat("repeat"),)),
        (AxisRef.repeat("repeat"), (AxisRef.cell_data("site"),)),
    ):
        fast = view._dense_data_curve(x, groups, Reduction.MEAN, True)
        slow = view._curve_from_positions(
            x, view._all_positions(), groups, Reduction.MEAN, True
        )
        _assert_same(fast, slow)
    fast = view._curve_from_axes(
        AxisRef.cell_data("frame"),
        (AxisRef.point("ay"),),
        Reduction.MEAN,
        uncertainty=True,
    )
    slow = view._curve_from_positions(
        AxisRef.cell_data("frame"),
        view._all_positions(),
        (AxisRef.point("ay"),),
        Reduction.MEAN,
        True,
    )
    _assert_same(fast, slow)

def test_mapped_repeat_siblings_keep_factored_uncertainty_on_repeat_dimension() -> None:
    """Multiple logical Repeat axes share physical dimension zero."""

    from zlc_data import (
        AxisId,
        AxisSpec,
        COMPONENT,
        DatasetSchema as DataSchema,
        DomainSpec,
        REPEAT,
        ValidityContract,
        ValueSchema,
        owned_snapshot_from_arrays,
    )

    sweep = AxisSpec(AxisId("sweep"), "sweep", REPEAT, 2, (0, 1))
    shot = AxisSpec(AxisId("shot"), "shot", REPEAT, 3, (0, 1, 2))
    x = AxisSpec(AxisId("x"), "x", COMPONENT, 4, (0.0, 1.0, 2.0, 3.0))
    schema = DataSchema(
        DomainSpec(
            (6,),
            (sweep, shot),
            ((0, 0, 0, 1, 1, 1), (0, 1, 2, 0, 1, 2)),
        ),
        DomainSpec((1,), (), ()),
        DomainSpec((4,), (x,)),
        ValueSchema(ValidityContract.value(), np.dtype("<f8")),
    )
    values = np.arange(24, dtype=np.float64).reshape((6, 1, 4))
    snapshot = owned_snapshot_from_arrays(
        schema, values, 1, validity=np.ones(values.shape, dtype=np.bool_)
    )
    view = DataView(snapshot)
    fast = view._factored_curve(
        AxisRef.repeat("shot"), (), Reduction.MEAN, uncertainty=True
    )
    slow = view._curve_from_positions(
        AxisRef.repeat("shot"), view._all_positions(), (), Reduction.MEAN, True
    )
    _assert_same(fast, slow)

def _identity_bucket_snapshot(*, grouped: bool, holes: bool, revision: int = 1):
    repeats = 11
    cell_axes = (
        (axis("series", values=[0.0, 1.0, 2.0]),)
        if grouped
        else ()
    )
    schema = make_dataset_schema(
        repeat_domain(values=np.arange(repeats, dtype=float) * 0.001, unit='s'),
        mapped_domain_from_columns({"sample": [0.0]}),
        cell_axes=cell_axes,
        value_unit="s",
        dtype=np.float64,
    )
    series = 3 if grouped else 1
    values = (
        0.010
        + np.arange(repeats, dtype=float)[:, None] * 0.0001
        + np.arange(series, dtype=float)[None, :] * 0.001
    )
    validity = None
    if holes:
        validity = np.ones(values.shape, dtype=bool)
        # The dataset's VALUE validity is common across an undeclared data
        # axis; finite-value holes below still exercise per-series validity.
        validity[2, :] = False
        values[7, series - 1] = np.nan
    return make_snapshot(
        schema,
        values[:, None, :] if grouped else values,
        revision=revision,
        validity=(
            None
            if validity is None
            else validity[:, None, :] if grouped else validity
        ),
    )

def _assert_curve_arrays_exact(left, right) -> None:
    assert len(left.series) == len(right.series)
    for ours, theirs in zip(left.series, right.series, strict=True):
        assert ours.group_key == theirs.group_key
        assert ours.label == theirs.label
        assert ours.x_labels == theirs.x_labels
        for ours_array, theirs_array in (
            (ours.x.canonical, theirs.x.canonical),
            (ours.x.display, theirs.x.display),
            (ours.valid, theirs.valid),
            (ours.counts, theirs.counts),
        ):
            np.testing.assert_array_equal(ours_array, theirs_array)
        # Reduction results where count==0 are explicitly unspecified; the
        # validity plane is their truth.  Every observable value remains
        # byte-for-byte equal, canonical and converted.
        np.testing.assert_array_equal(
            np.asarray(ours.y.canonical)[ours.valid],
            np.asarray(theirs.y.canonical)[theirs.valid],
        )
        np.testing.assert_array_equal(
            np.asarray(ours.y.display)[ours.valid],
            np.asarray(theirs.y.display)[theirs.valid],
        )
        if theirs.sem is None:
            assert ours.sem is None
        else:
            np.testing.assert_array_equal(ours.sem, theirs.sem)

@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("holes", [False, True])
@pytest.mark.parametrize("aggregation", tuple(Reduction))
def test_identity_tensor_buckets_match_every_array_exactly(
    grouped, holes, aggregation
) -> None:
    view = DataView(
        _identity_bucket_snapshot(grouped=grouped, holes=holes),
        axis_display_units={AxisRef.repeat("repeat"): "ms"},
        value_display_unit="ms",
    )
    groups = (AxisRef.cell_data("series"),) if grouped else ()
    fast = view._dense_data_curve(AxisRef.repeat("repeat"), groups, aggregation)
    slow = view._curve_from_positions(
        AxisRef.repeat("repeat"), view._all_positions(), groups, aggregation
    )
    _assert_curve_arrays_exact(fast, slow)

@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("holes", [False, True])
def test_identity_tensor_uncertainty_is_the_same_undefined_single_sample(
    grouped, holes
) -> None:
    view = DataView(_identity_bucket_snapshot(grouped=grouped, holes=holes))
    groups = (AxisRef.cell_data("series"),) if grouped else ()
    fast = view._dense_data_curve(
        AxisRef.repeat("repeat"), groups, Reduction.MEAN, uncertainty=True
    )
    slow = view._curve_from_positions(
        AxisRef.repeat("repeat"), view._all_positions(), groups, Reduction.MEAN, True
    )
    _assert_curve_arrays_exact(fast, slow)

def test_resolved_axis_cache_crosses_only_the_same_schema_and_unit_context() -> None:
    first = DataView(
        _identity_bucket_snapshot(grouped=True, holes=False, revision=1),
        axis_display_units={AxisRef.repeat("repeat"): "ms"},
    )
    resolved = first._resolve(AxisRef.repeat("repeat"))
    second = DataView(
        _identity_bucket_snapshot(grouped=True, holes=True, revision=2),
        axis_display_units={AxisRef.repeat("repeat"): "ms"},
        inherit_domains_from=first,
    )
    assert second._resolve(AxisRef.repeat("repeat")) is resolved

    changed_unit = DataView(
        _identity_bucket_snapshot(grouped=True, holes=False, revision=3),
        axis_display_units={AxisRef.repeat("repeat"): "s"},
        inherit_domains_from=second,
    )
    assert changed_unit._resolve(AxisRef.repeat("repeat")) is not resolved

@pytest.mark.parametrize("holes", [0.0, 0.3])
@pytest.mark.parametrize(
    "aggregation",
    (Reduction.MEAN, Reduction.SUM, Reduction.MIN, Reduction.FIRST),
)
def test_exact_axis_aggregation_covers_remaining_curve_roles(
    holes, aggregation
) -> None:
    view = DataView(_snapshot(holes=holes, seed=19))
    for x, groups in (
        (AxisRef.cell_data("frame"), (AxisRef.point("ay"),)),
        (AxisRef.repeat("repeat"), (AxisRef.point("ay"), AxisRef.cell_data("site"))),
    ):
        fast = view._curve_from_axes(x, groups, aggregation)
        slow = view._curve_from_positions(
            x, view._all_positions(), groups, aggregation
        )
        _assert_same(fast, slow)

def test_configurations_the_path_does_not_own_fall_through() -> None:
    view = DataView(_snapshot(seed=3))
    x = AxisRef.point("ax")
    assert view._factored_curve(x, (), Reduction.FIRST) is None

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
    view.curve(AxisRef.point("ax"), group_by=(AxisRef.cell_data("site"),))
    assert calls == [True]

    def forbidden(*_args, **_kwargs):
        raise AssertionError("a tensor curve allocated generic positions")

    monkeypatch.setattr(DataView, "_all_positions", forbidden)
    view.curve(
        AxisRef.cell_data("frame"), group_by=(AxisRef.cell_data("site"),)
    )

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
            AxisRef.cell_data("frame"), AxisRef.cell_data("site"), Reduction.MEAN
        )
        is None
    ), "two data axes belong to the dense image path"

    tensor = view._dense_data_image(
        AxisRef.cell_data("site"), AxisRef.repeat("repeat"), Reduction.MEAN
    )
    assert tensor is not None
    generic = view._image_from_positions(
        AxisRef.cell_data("site"),
        AxisRef.repeat("repeat"),
        view._all_positions(),
        Reduction.MEAN,
    )
    _assert_same_image(tensor, generic)

@pytest.mark.parametrize("holes", [0.0, 0.3])
@pytest.mark.parametrize(
    "aggregation", (Reduction.MEAN, Reduction.SUM, Reduction.MIN)
)
def test_mixed_point_tensor_images_match_generic(holes, aggregation) -> None:
    view = DataView(_snapshot(holes=holes, seed=23))
    for x, y in (
        (AxisRef.point("ax"), AxisRef.cell_data("site")),
        (AxisRef.cell_data("site"), AxisRef.point("ay")),
    ):
        fast = view._image_from_axes(x, y, aggregation)
        slow = view._image_from_positions(
            x, y, view._all_positions(), aggregation
        )
        _assert_same_image(fast, slow)

@pytest.mark.parametrize("holes", [0.0, 0.3])
@pytest.mark.parametrize("uncertainty", [False, True])
def test_factored_facet_matches_the_generic_path(holes, uncertainty) -> None:
    """A lattice facet grid, cell for cell against the generic per-cell run."""

    from zlc_plot import CurvePlot, FacetGridPlot

    view = DataView(_snapshot(holes=holes, seed=29))
    for facet, group in (
        (AxisRef.cell_data("frame"), None),
        (AxisRef.cell_data("frame"), AxisRef.cell_data("site")),
        (AxisRef.repeat("repeat"), AxisRef.cell_data("site")),
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

    spec = FacetGridPlot(
        AxisRef.cell_data("site"), CurvePlot(AxisRef.cell_data("frame"))
    )
    fast = view._factored_facet(spec, uncertainty)
    assert fast is not None
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
            FacetGridPlot(AxisRef.cell_data("frame"), HistogramPlot()), False
        )
        is None
    ), "histogram cells use the dense tensor-slice path"

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
    for group in (None, AxisRef.cell_data("site")):
        spec = FacetGridPlot(
            AxisRef.point("ax"),
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

    spec = FacetGridPlot(
        AxisRef.point("ax"),
        CurvePlot(
            AxisRef.cell_data("frame"), group=AxisRef.point("ay")
        ),
    )
    fast = view._factored_facet(spec, uncertainty)
    assert fast is not None
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
    for cell in (
        ImagePlot(
            AxisRef.point("ax"),
            AxisRef.point("ay"),
            reduction=aggregation,
        ),
        ImagePlot(
            AxisRef.point("ax"),
            AxisRef.cell_data("frame"),
            reduction=aggregation,
        ),
    ):
        spec = FacetGridPlot(AxisRef.cell_data("site"), cell)
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
    schema = make_dataset_schema(
        repeat_domain(size=5),
        mapped_domain_from_columns({
            "ax": np.asarray([float(c[0]) for c in cells]),
            "ay": np.asarray([float(c[1]) for c in cells]),
            "az": np.asarray([float(c[2]) for c in cells]),
        }),
        cell_axes=(axis("site", values=[0.0, 1.0, 2.0], role=SITE),),
        dtype=np.float64,
    )
    shape = (5, rows, 3)
    validity = np.broadcast_to(
        rng.random(shape[:2] + (1,)) > 0.2, shape
    ).copy()
    view = DataView(
        make_snapshot(schema, rng.normal(size=shape), 1, validity=validity)
    )
    spec = FacetGridPlot(
        AxisRef.point("az"),
        ImagePlot(
            AxisRef.point("ax"), AxisRef.point("ay")
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
