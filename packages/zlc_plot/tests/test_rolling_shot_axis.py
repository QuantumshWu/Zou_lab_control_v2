"""Rolling x counts back from the newest shot, which sits at zero.

A rolling window shows the last N shots, so what a point MEANS is its
distance from now.  The absolute shot number is a fact about the run,
not about the picture: using it slid every tick label forward on every
single revision, so a full window never held still.
"""

from __future__ import annotations

import numpy as np
import pytest

from zlc_data import PRIMARY_INDEX
from zlc_data.snapshot_projection import PRIMARY_INDEX_AXIS_ID
from zlc_plot import AxisRef, PlotSession, Reduction, RollingPlot
from zlc_plot.data_view import DataView
from data_factory import (
    axis,
    make_dataset_schema,
    make_snapshot,
    mapped_domain_from_columns,
    repeat_domain,
)
from zlc_data import OwnedSnapshot

def _snapshot(revision: int, repeats: int = 6) -> OwnedSnapshot:
    schema = make_dataset_schema(
        repeat_domain(size=repeats),
        mapped_domain_from_columns({"x": np.arange(4.0)}),
        dtype=np.float64,
    )
    values = np.arange(repeats * 4.0).reshape(repeats, 4) + float(revision)
    return make_snapshot(schema, values, revision=revision)

def test_seeded_history_ends_at_zero_and_counts_back() -> None:
    session = PlotSession(_snapshot(0), RollingPlot())
    try:
        series = session._payload.series[0]
        np.testing.assert_array_equal(
            np.asarray(series.x.canonical), np.arange(-5.0, 1.0)
        )
        assert series.x.label == "Shots from latest"
        assert float(np.asarray(series.x.canonical).max()) == 0.0
    finally:
        session.close()

def test_nonindexed_revisions_replace_instead_of_extending_the_shot_axis() -> None:
    session = PlotSession(_snapshot(0), RollingPlot())
    try:
        session.update_data(_snapshot(1))
        session.update_data(_snapshot(2))
        x = np.asarray(session._payload.series[0].x.canonical)
        np.testing.assert_array_equal(x, np.arange(-5.0, 1.0))
    finally:
        session.close()

def test_area_selector_display_coordinates_are_identity() -> None:
    """Display selector coordinates pass through unchanged on the shot axis.

    The dead negative-axis mapping used np.interp, which clamps out-of-domain
    input to the endpoints — a selector authored outside the current shot
    range collapsed to a degenerate (0, 0) span and raised.  The ordinal axis
    has no display conversion at all now.
    """

    from zlc_plot import NumericRange

    session = PlotSession(_snapshot(0), RollingPlot())
    try:
        state = session.set_area_selector(
            NumericRange(1.0, 4.0), NumericRange(0.0, 30.0), display=True
        )
        assert state.value.x == NumericRange(1.0, 4.0)

        # Out-of-domain spans stay non-degenerate instead of being clamped.
        state = session.set_area_selector(
            NumericRange(-24.0, -8.0), NumericRange(0.0, 30.0), display=True
        )
        assert state.value.x == NumericRange(-24.0, -8.0)
    finally:
        session.close()

def test_a_full_window_holds_the_same_coordinates_as_it_slides() -> None:
    window = 100
    total = window + 8
    session = PlotSession(
        _snapshot(0, repeats=total),
        RollingPlot(),
        parameters={"window": window},
    )
    try:
        x = np.asarray(session._payload.series[0].x.canonical)
        assert x.size == min(window, total)
        # The window shows the most recent shots, the newest at zero -- the
        # same coordinates however far the run has got.
        np.testing.assert_array_equal(x, np.arange(1.0 - x.size, 1.0))
    finally:
        session.close()

def test_shot_axis_frames_the_full_window_from_the_first_revision() -> None:
    """The axis spans exactly ``window`` shots and then stands still.

    What you configure is what you see: the frame is always
    ``[-(window - 1), 0]``, the young trace grows rightward inside it to
    the newest shot at zero, and the frame never moves again.
    """

    session = PlotSession(
        _snapshot(0, repeats=30),
        RollingPlot(),
        parameters={"window": 20},
    )
    try:
        axes = session._renderer.primary_axes
        assert tuple(map(float, axes.get_xlim())) == (-19.0, 0.0)
    finally:
        session.close()

def test_window_selects_display_without_truncating_retention() -> None:
    """The window is a view, not a destructive cap on measured history.

    Shrinking must narrow the display immediately; committing new data under
    the small window must not discard older shots; enlarging must immediately
    reveal them again.
    """

    session = PlotSession(_snapshot(0, repeats=14), RollingPlot())
    try:
        session.set_parameter("window", 4)
        x = np.asarray(session._payload.series[0].x.canonical)
        np.testing.assert_array_equal(x, np.arange(-3.0, 1.0))

        # Shrinking changes only the view over this Runtime-supplied Dataset.
        session.set_parameter("window", 100)
        x = np.asarray(session._payload.series[0].x.canonical)
        np.testing.assert_array_equal(x, np.arange(-13.0, 1.0))
    finally:
        session.close()

def test_replace_spec_keeps_history_for_an_equivalent_rolling_spec() -> None:
    """A form submit that keeps group and reduction keeps the trace.

    Changing the reduction changes what one history point IS, so that
    replacement reseeds from the current snapshot instead.
    """

    session = PlotSession(_snapshot(0, repeats=10), RollingPlot())
    try:
        session.replace_spec(RollingPlot())
        x = np.asarray(session._payload.series[0].x.canonical)
        np.testing.assert_array_equal(x, np.arange(-9.0, 1.0))

        session.replace_spec(RollingPlot(reduction=Reduction.MIN))
        x = np.asarray(session._payload.series[0].x.canonical)
        np.testing.assert_array_equal(x, np.arange(-9.0, 1.0))
    finally:
        session.close()

@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_primary_index_history_keeps_source_order_holes_and_site_groups(monkeypatch, dtype) -> None:
    source = [-2, -2, 0, 0]
    indexed_schema = make_dataset_schema(
        repeat_domain(size=1),
        mapped_domain_from_columns(
            {"source index": source, "category": [0.0, 1.0, 0.0, 1.0]},
            ids={"source index": str(PRIMARY_INDEX_AXIS_ID)},
            roles={"source index": PRIMARY_INDEX},
        ),
        cell_axes=(axis("site", values=[0.0, 1.0, 2.0]),),
        dtype=dtype,
    )
    indexed_values = np.arange(12, dtype=dtype).reshape(1, 4, 3)
    indexed_valid = np.ones(indexed_values.shape, dtype=np.bool_)
    indexed_valid[:, :2] = False
    snapshot = make_snapshot(
        indexed_schema,
        indexed_values,
        revision=9,
        validity=indexed_valid,
    )
    from dataclasses import replace
    from zlc_data import INVALID

    child_schema = make_dataset_schema(
        repeat_domain(size=1), mapped_domain_from_columns({"category": [0.0, 1.0]}),
        cell_axes=indexed_schema.cell_domain.axes, dtype=dtype,
    )
    children = tuple(make_snapshot(
        child_schema, indexed_values[:, start:start + 2], revision=start,
        validity=indexed_valid[:, start:start + 2],
    ) for start in (0, 2))
    segmented = OwnedSnapshot(snapshot.ref, replace(
        snapshot.block, values=None, validity=INVALID,
        segments=tuple(child.block.as_segment() for child in children),
        segment_origins=np.asarray(((0, 0), (0, 2)), dtype=np.int64),
        segment_shapes=np.asarray(((1, 2), (1, 2)), dtype=np.int64),
    ))
    from zlc_plot import _raster_kernels as kernels

    # The same authored cells must survive both execution engines, including
    # grouping on the record's point-row dimension and absent first-shot data.
    for group in (None, AxisRef.cell_data("site"), AxisRef.point("category")):
        for reduction in (Reduction.MEAN, Reduction.SUM, Reduction.MIN,
                          Reduction.MAX, Reduction.FIRST):
            answers = []
            for engine in ("numpy", "auto"):
                with monkeypatch.context() as active:
                    active.setattr(kernels, "ENGINE", engine)
                    answers.append(DataView(snapshot).rolling_history(
                        group=group, aggregation=reduction,
                    ))
            expected, actual = answers
            for name in ("values", "counts", "valid", "source_indices"):
                np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
            if reduction is Reduction.MEAN:
                np.testing.assert_array_equal(actual.sem, expected.sem)
            segmented_view = DataView(segmented)
            actual = segmented_view.rolling_history(group=group, aggregation=reduction)
            for name in ("values", "counts", "valid", "source_indices"):
                np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
            if reduction is Reduction.MEAN:
                np.testing.assert_allclose(actual.sem, expected.sem, rtol=1e-14)
            inherited = DataView(segmented, inherit_domains_from=segmented_view)
            repeated = inherited.rolling_history(group=group, aggregation=reduction)
            np.testing.assert_array_equal(repeated.values, actual.values)
            assert inherited._samples is None
            assert inherited._rolling_carry[1] == segmented_view._rolling_carry[1]
            assert inherited._rolling_carry[-1] is segmented
            assert segmented.block._materialized is None
    grouped = DataView(segmented)
    grouped.rolling_history(group=AxisRef.point("category"))
    point = indexed_schema.point_domain
    flipped_schema = replace(indexed_schema, point_domain=replace(
        point, axis_codes=(point.axis_codes[0], 1 - np.asarray(point.axis_codes[1])),
    ))
    flipped = OwnedSnapshot(
        replace(segmented.ref, schema_fingerprint=flipped_schema.fingerprint),
        replace(segmented.block, schema=flipped_schema),
    )
    changed = DataView(flipped, inherit_domains_from=grouped).rolling_history(group=AxisRef.point("category"))
    np.testing.assert_array_equal(changed.values[1], [10.0, 7.0])
    assert flipped.block._materialized is None
    session = PlotSession(
        segmented, RollingPlot(group=AxisRef.cell_data("site")),
        parameters={"window": 3, "uncertainty": True},
    )
    try:
        assert session._view._samples is None
        assert segmented.block._materialized is None
        np.testing.assert_array_equal(session._payload.series[0].y.canonical, [np.nan, 7.5])
    finally:
        session.close()
    history = DataView(snapshot).rolling_history(
        group=AxisRef.cell_data("site"), aggregation=Reduction.MEAN
    )
    assert tuple(sample.source_index for sample in history) == (-2, 0)
    assert all(sample.revision == 9 for sample in history)
    assert all(
        sample.generation == snapshot.ref.stream_generation.value
        for sample in history
    )
    assert tuple(key[0].canonical for key in history[0].group_keys) == (
        0.0,
        1.0,
        2.0,
    )
    np.testing.assert_allclose(history[0].values, [np.nan] * 3, equal_nan=True)
    np.testing.assert_array_equal(history[0].valid, [False] * 3)
    np.testing.assert_array_equal(history[0].counts, [0] * 3)
    np.testing.assert_allclose(history[1].values, [7.5, 8.5, 9.5])
    np.testing.assert_array_equal(history[1].valid, [True] * 3)
    np.testing.assert_array_equal(history[1].counts, [2] * 3)
    np.testing.assert_allclose(history[1].sem, [1.5] * 3)
    last = DataView(snapshot).rolling_history(
        group=AxisRef.cell_data("site"), aggregation=Reduction.LAST,
    )
    assert tuple(last.source_indices) == (-2, 0)
    np.testing.assert_array_equal(last.valid[0], [False] * 3)
    np.testing.assert_allclose(last.values[1], [9.0, 10.0, 11.0])
    np.testing.assert_array_equal(last.counts[1], [1] * 3)
    assert np.all(np.isnan(last.sem))

    stated = make_snapshot(indexed_schema, indexed_values, revision=9,
                           validity=indexed_valid, sigma=np.full(indexed_values.shape, 2.0))
    for engine in ("numpy", "auto"):
        with monkeypatch.context() as active:
            active.setattr(kernels, "ENGINE", engine)
            result = DataView(stated).rolling_history(
                group=AxisRef.cell_data("site"), aggregation=Reduction.LAST,
            )
        np.testing.assert_array_equal(result.sem[1], [2.0] * 3)

    # Flat storage carries only declared component masks and sample sigma;
    # the mixed singleton bucket still uses the producer's sigma, not SEM=0.
    from zlc_data import AxisId, ValidityContract

    component_schema = replace(indexed_schema, value_schema=replace(
        indexed_schema.value_schema,
        validity_contract=ValidityContract.components(AxisId("site")),
    ))
    component_valid = indexed_valid.copy()
    component_valid[:, 2, 1] = False
    component = make_snapshot(component_schema, indexed_values, revision=10,
                              validity=component_valid, sigma=np.full(indexed_values.shape, 2.0))
    component_children = tuple(make_snapshot(
        replace(child_schema, value_schema=component_schema.value_schema),
        indexed_values[:, start:start + 2], revision=start,
        validity=component_valid[:, start:start + 2],
        sigma=np.full((1, 2, 3), 2.0),
    ).block.as_segment() for start in (0, 2))
    component_flat = OwnedSnapshot(component.ref, replace(
        component.block, values=None, validity=INVALID, sigma=None,
        segments=component_children, segment_origins=segmented.block.segment_origins,
        segment_shapes=segmented.block.segment_shapes,
    ))
    flat_view = DataView(component_flat)
    actual = flat_view.rolling_history(group=AxisRef.cell_data("site"))
    expected = DataView(component).rolling_history(group=AxisRef.cell_data("site"))
    np.testing.assert_array_equal(actual.counts, expected.counts)
    np.testing.assert_array_equal(actual.values, expected.values)
    np.testing.assert_array_equal(actual.sem, expected.sem)
    np.testing.assert_array_equal(flat_view.samples.valid_mask, component_valid)
    assert actual.sem[1, 1] == 2.0
    assert component_flat.block._materialized is None

    repeat = DataView(_snapshot(0, repeats=3)).rolling_history()
    np.testing.assert_allclose(
        [sample.values[0] for sample in repeat], [1.5, 5.5, 9.5]
    )
    np.testing.assert_array_equal(
        [sample.counts[0] for sample in repeat], [4, 4, 4]
    )
    assert all(
        sample.source_index is None and sample.group_keys == ((),)
        for sample in repeat
    )

    # Record rows are already Rolling's X. Grouping by them used to allocate
    # records squared buckets, only the diagonal of which could be valid.
    # Reject before any sample gather, including the LAST reduction path.
    def no_projection(_view):
        raise AssertionError("invalid rolling axes reached numeric projection")

    monkeypatch.setattr(DataView, "_all_positions", no_projection)
    for view, record in (
        (DataView(snapshot), AxisRef.point(PRIMARY_INDEX_AXIS_ID.value)),
        (DataView(_snapshot(0, repeats=3)), AxisRef.repeat("repeat")),
    ):
        for aggregation in (Reduction.MEAN, Reduction.LAST):
            with pytest.raises(ValueError, match="fixed record axis cannot also be Group"):
                view.rolling_history(group=record, aggregation=aggregation)
    with pytest.raises(ValueError, match="fixed record axis"):
        DataView(snapshot).rolling_history(x=AxisRef.cell_data("site"))

def test_a_one_shot_history_skips_the_band_it_is_told_not_to_draw() -> None:
    """``uncertainty=False`` reaches the single-revision reduction.

    The standard error is a second pass over every value -- squared,
    masked, reduced -- and a one-shot history (one repeat, no shot index:
    the ordinary unindexed monitor) estimated it on every revision with
    the band switched off, because the switch was dropped on the way in.
    """

    snapshot = _snapshot(0, repeats=1)
    view = DataView(snapshot)
    assert view.rolling_history(uncertainty=False).sem is None
    assert view.rolling_history(
        group=AxisRef.point("x"), uncertainty=False
    ).sem is None
    assert view.rolling_history(uncertainty=True).sem is not None
    assert view.rolling_history(
        group=AxisRef.point("x"), uncertainty=True
    ).sem is not None
    # A non-MEAN reduction has no band whatever the switch says.
    assert view.rolling_history(
        aggregation=Reduction.SUM, uncertainty=True
    ).sem is None
