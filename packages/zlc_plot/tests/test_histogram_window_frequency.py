"""The window histogram is moved by the shots that entered and left, exactly.

A pixel-pool histogram at a deep window used to recount every value of the
window on every shot.  The count of each value is a sum over shots, so the
view keeps one frequency table for the window and moves it by the shot
that arrived and the one that left -- and the contract is EXACTNESS: at
every revision the moved table equals a fresh count of the same window,
holes, invalid samples, replacements and window changes included.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest

from data_factory import (
    axis,
    make_dataset_schema,
    make_snapshot,
    mapped_domain_from_columns,
    repeat_domain,
)
from zlc_data import AxisId, BlockId, DataBlock, DatasetSchema
from zlc_data import PRIMARY_INDEX, IndexedWindow, owned_snapshot_from_arrays
from zlc_data.snapshot_projection import PRIMARY_INDEX_AXIS_ID, restrict_snapshot, value_selection
from zlc_plot import AxisRef, HistogramPlot, PlotSession
from zlc_plot.data_view import DataView

HEIGHT, WIDTH = 4, 5


def _schema(offsets: tuple[int, ...], dtype=np.uint16) -> DatasetSchema:
    return make_dataset_schema(
        repeat_domain(size=1),
        mapped_domain_from_columns(
            {"source index": list(offsets)},
            ids={"source index": str(PRIMARY_INDEX_AXIS_ID)},
            roles={"source index": PRIMARY_INDEX},
        ),
        cell_axes=(axis("y", size=HEIGHT), axis("x", size=WIDTH)),
        dtype=dtype,
    )


def _frame(seed: int, dtype=np.uint16, high: int = 40) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, high, size=(HEIGHT, WIDTH)).astype(dtype)


class _History:
    """Shots by absolute number, published as rolled windows the way Runtime does."""

    def __init__(self, capacity: int, dtype=np.uint16, high: int = 40) -> None:
        self.capacity = capacity
        self.dtype = dtype
        self.high = high
        self.shots: dict[int, np.ndarray] = {}
        self.invalid: dict[int, np.ndarray] = {}
        self.revision = 0
        self.stable_since = -1

    def publish(self, index: int, *, frame: np.ndarray | None = None, invalid: np.ndarray | None = None):
        self.revision += 1
        if index in self.shots:
            self.stable_since = self.revision
        self.shots[index] = _frame(index, self.dtype, self.high) if frame is None else frame
        if invalid is not None:
            self.invalid[index] = invalid
        latest = max(self.shots)
        start = max(min(self.shots), latest - self.capacity + 1)
        kept = sorted(i for i in self.shots if i >= start)
        values = np.stack([self.shots[i] for i in kept])[None]
        validity = np.ones(values.shape, dtype=bool)
        for position, i in enumerate(kept):
            if i in self.invalid:
                validity[0, position] = ~self.invalid[i]
        return owned_snapshot_from_arrays(
            schema=_schema(tuple(i - latest for i in kept), self.dtype),
            values=values,
            revision=self.revision,
            validity=validity,
            block_id="roi.indexed",
            stream_generation="roi",
            window=IndexedWindow(start, latest, self.stable_since),
        )


def _fresh_frequency(snapshot, window: int) -> tuple[int, np.ndarray]:
    """What the table must equal: a count of the window from scratch."""

    view = DataView(snapshot)
    values, valid = view.history_values(window)
    selected = np.asarray(values)[np.asarray(valid, dtype=bool)]
    offset = int(np.iinfo(selected.dtype).min)
    return offset, np.bincount(
        selected.astype(np.int64) - offset, minlength=int(np.iinfo(selected.dtype).max) - offset + 1
    )


def _assert_exact(view: DataView, snapshot, window: int) -> None:
    got = view.window_frequency(window)
    assert got is not None
    offset, counts = got
    expected_offset, expected = _fresh_frequency(snapshot, window)
    assert offset == expected_offset
    np.testing.assert_array_equal(counts, expected)


def test_the_table_moves_with_the_window_and_stays_exact(monkeypatch) -> None:
    """Fill-up, steady rolling, a hole, invalid samples, a replacement, a window change."""

    history = _History(capacity=6)
    previous = None
    window = 4
    snapshot = history.publish(0)
    view = DataView(snapshot)
    _assert_exact(view, snapshot, window)
    for index in (1, 2, 3, 4, 5, 6, 7):
        previous, snapshot = view, history.publish(index)
        view = DataView(snapshot, inherit_domains_from=previous)
        _assert_exact(view, snapshot, window)
    # The steady state moved the carried table: same offset, a new array.
    assert view._frequency_carry is not None and previous._frequency_carry is not None
    assert view._frequency_carry.counts is not previous._frequency_carry.counts

    # A hole: shot 8 never arrives, 9 does.
    previous, snapshot = view, history.publish(9)
    view = DataView(snapshot, inherit_domains_from=previous)
    _assert_exact(view, snapshot, window)

    # An invalid shot (a frame the producer withdrew) is not counted ...
    previous, snapshot = view, history.publish(
        10, invalid=np.ones((HEIGHT, WIDTH), dtype=bool)
    )
    view = DataView(snapshot, inherit_domains_from=previous)
    _assert_exact(view, snapshot, window)
    # ... and contributes nothing to subtract when it leaves.
    for index in (11, 12, 13, 14):
        previous, snapshot = view, history.publish(index)
        view = DataView(snapshot, inherit_domains_from=previous)
        _assert_exact(view, snapshot, window)

    # A retained shot replaced: the fence moves past the carried revision,
    # so the table is recounted rather than moved -- and is still exact.
    previous, snapshot = view, history.publish(13, frame=np.full((HEIGHT, WIDTH), 7, dtype=np.uint16))
    assert snapshot.block.window.stable_since == history.revision
    view = DataView(snapshot, inherit_domains_from=previous)
    _assert_exact(view, snapshot, window)

    # A different window on the same view is its own count.
    _assert_exact(view, snapshot, 2)
    _assert_exact(view, snapshot, 6)
    _assert_exact(view, snapshot, 1)

    # The real Runtime changes block_id on EVERY window move. Unchanged
    # immutable event planes, not that range-dependent string, carry counts.
    from zlc_runtime import DatasetOutputDeclaration, LiveDatasetOutput, MonitorCoverage, SignalDataPlane

    event_schema = make_dataset_schema(
        repeat_domain(size=1), mapped_domain_from_columns({"frame": [0, 1]}),
        cell_axes=(axis("y", size=HEIGHT), axis("x", size=WIDTH)), dtype=np.uint16,
    )
    declaration = DatasetOutputDeclaration("value", "test.frequency", index_by_source=True)
    node = SimpleNamespace(instance_id="frequency", dataset_output_declarations=(declaration,),
                           signal_key=lambda name: f"frequency/{name}")
    plane = SignalDataPlane()
    plane.begin_generation(node)
    lease = plane.acquire_indexed_history("frequency/value", 6)
    try:
        for index in range(7):
            event = make_snapshot(event_schema, np.stack((_frame(index), _frame(index + 1)))[None], index)
            plane.commit_live(node, {"value": LiveDatasetOutput(declaration, event, MonitorCoverage(2, 2))})
            snapshot = plane.current_dataset("frequency/value")
            if index == 5:
                previous = DataView(snapshot)
                _assert_exact(previous, snapshot, 4)
                _ = previous.samples  # An earlier raw selector may have packed this window.
        assert previous._snapshot.ref.block_id != snapshot.ref.block_id
        view = DataView(snapshot, inherit_domains_from=previous)
        assert view._packed_carry is not None
        packed = DataBlock.packed_planes
        reads = []
        def observed(block, *, selection=None, sigma=False):
            assert not sigma, "histogram never consumes sample sigma"
            indices = range(len(block.segments)) if selection is None else selection
            reads.extend(block.segments[int(index)][0].size for index in indices)
            return packed(block, selection=selection, sigma=sigma)
        with monkeypatch.context() as patch:
            patch.setattr(DataBlock, "packed_planes", observed)
            view.window_frequency(4)
        assert sum(reads) == 2 * np.prod(event_schema.physical_shape), "only entering and leaving events are read"
        assert view._samples is None and view._packed_segments is None and view._packed_carry is None
        _assert_exact(view, snapshot, 4)
        _assert_exact(view, snapshot, 2)
        assert view._frequency_carry.snapshot is snapshot

        # The normal worker fork must preserve a fixed Scope's slices after
        # the window cutter, without changing a held Frozen projection.
        scoped_session = PlotSession(snapshot, HistogramPlot(scope=((AxisRef.point("frame"), 1),)),
                                     parameters={"window": 4})
        try:
            owner = scoped_session._projection
            frozen = owner._fork_frozen(data=snapshot, revision=snapshot.ref.revision.value, context=owner._context)
            frozen._build_view_and_payload()
            old_cache = frozen._scoped_cache
            old_memo = dict(old_cache[3])
            frozen_counts = frozen.payload.counts.copy()
            event = make_snapshot(event_schema, np.stack((_frame(7), _frame(8)))[None], 7)
            plane.commit_live(node, {"value": LiveDatasetOutput(declaration, event, MonitorCoverage(2, 2))})
            snapshot = plane.current_dataset("frequency/value")
            reads.clear()
            with monkeypatch.context() as patch:
                patch.setattr(DataBlock, "packed_planes", observed)
                scoped_session.update_data(snapshot)
            assert sum(reads) == 2 * HEIGHT * WIDTH, "fixed Scope counts only entering/leaving slices"
            current_cache = scoped_session._projection._scoped_cache
            assert len({id(item) for item in old_cache[1].block.segments}
                       & {id(item) for item in current_cache[1].block.segments}) == 3
            assert frozen._scoped_cache is old_cache and old_cache[3].keys() == old_memo.keys()
            assert all(old_cache[3][key] is entry for key, entry in old_memo.items())
            np.testing.assert_array_equal(frozen.payload.counts, frozen_counts)
            scoped_session.set_parameters({"window": 2})
            smaller = scoped_session._projection._scoped_cache
            assert len(smaller[3]) == 2
            assert all(left is right for left, right in
                       zip(smaller[1].block.segments, current_cache[1].block.segments[-2:]))
            scoped_session.set_parameters({"window": 4})
            assert len(scoped_session._projection._scoped_cache[3]) == 4
            scoped_session.replace_spec(HistogramPlot(scope=((AxisRef.point("frame"), 0),)))
            assert not current_cache[3].keys() & scoped_session._projection._scoped_cache[3].keys()
            selected = np.concatenate([item[0][:, 0:1].reshape(-1) for item in snapshot.block.segments[-4:]])
            expected, _ = np.histogram(selected, bins=scoped_session._payload.edges.canonical)
            np.testing.assert_array_equal(scoped_session._payload.counts, expected[None])
            accepted_cache = scoped_session._projection._scoped_cache
            with pytest.raises(ValueError):
                scoped_session.replace_spec(HistogramPlot(scope=((AxisRef.point("frame"), 99),)))
            assert scoped_session._projection._scoped_cache is accepted_cache
            scoped_session.replace_spec(HistogramPlot(), parameters={"window": 6})
            assert scoped_session._projection._scoped_cache is None
            scoped_session.replace_spec(HistogramPlot(scope=((AxisRef.point("frame"), 1),)))
            assert scoped_session._projection._scoped_cache is not None
        finally:
            scoped_session.close()
        assert scoped_session._projection._scoped_cache is None
    finally:
        lease.close()
        plane.close()


def test_an_unchanged_snapshot_shares_the_table_and_no_provenance_means_no_table() -> None:
    history = _History(capacity=5)
    snapshot = history.publish(0)
    view = DataView(snapshot)
    first = view.window_frequency(3)
    assert first is not None
    assert view.window_frequency(3)[1] is first[1]
    again = DataView(snapshot, inherit_domains_from=view)
    assert again.window_frequency(3)[1] is first[1], "nothing entered or left"

    plain = owned_snapshot_from_arrays(
        schema=_schema((-1, 0)),
        values=np.stack([_frame(1), _frame(2)])[None],
        revision=1,
    )
    assert DataView(plain).window_frequency(2) is None
    assert DataView(plain, inherit_domains_from=view)._frequency_carry is None

    floats = owned_snapshot_from_arrays(
        schema=_schema((-1, 0), np.float64),
        values=np.stack([_frame(1), _frame(2)])[None].astype(np.float64),
        revision=1,
        window=IndexedWindow(0, 1, -1),
    )
    assert DataView(floats).window_frequency(2) is None
    assert DataView(floats, inherit_domains_from=view)._frequency_carry is None

    # Different same-shaped Scope selections can keep block_id, generation,
    # revision and window facts. Their actual immutable planes differ.
    source = history.publish(1)
    previous = None
    for coordinate in (0, 1):
        scoped = restrict_snapshot(
            source, value_selection(source.block.schema, {AxisId("x"): coordinate}),
            reference_for=lambda schema: replace(source.ref, block_id=BlockId("roi.scoped"),
                                                 schema_fingerprint=schema.fingerprint),
        )
        view = DataView(scoped, inherit_domains_from=previous)
        _assert_exact(view, scoped, 3)
        previous = view

    # One immutable two-row plane can occur twice. A changing window also
    # cuts inside it: match occurrences and subtract just the changed rows.
    repeated_schema = _schema((-3, -2, -1, 0))
    planes = plain.block.as_segment()
    block = DataBlock._from_owned_segments(
        BlockId("repeated"), plain.ref.revision, repeated_schema, (planes, planes),
        origins=np.asarray(((0, 0), (0, 2))), shapes=np.asarray(((1, 2), (1, 2))),
        window=IndexedWindow(0, 3, -1),
    )
    repeated = type(plain)(block.ref(plain.ref.stream_generation), block)
    view = DataView(repeated)
    for window in (4, 3, 2, 1, 4):
        _assert_exact(view, repeated, window)


def test_a_wide_integer_table_grows_with_its_values_and_gives_up_past_the_limit() -> None:
    from zlc_plot import data_view as module

    history = _History(capacity=4, dtype=np.int32, high=50)
    snapshot = history.publish(0)
    view = DataView(snapshot)
    offset, counts = view.window_frequency(4)
    assert offset >= 0 and counts.size <= 50

    previous, snapshot = view, history.publish(
        1, frame=np.full((HEIGHT, WIDTH), -300, dtype=np.int32)
    )
    view = DataView(snapshot, inherit_domains_from=previous)
    offset, counts = view.window_frequency(4)
    assert offset == -300
    values, valid = view.history_values(4)
    selected = np.asarray(values)[np.asarray(valid, dtype=bool)]
    np.testing.assert_array_equal(
        counts, np.bincount(selected.astype(np.int64) + 300, minlength=counts.size)
    )

    previous, snapshot = view, history.publish(
        2, frame=np.full((HEIGHT, WIDTH), module._FREQUENCY_LEVEL_LIMIT + 10, dtype=np.int32)
    )
    view = DataView(snapshot, inherit_domains_from=previous)
    assert view.window_frequency(4) is None
    assert view._frequency_carry is None


def test_the_session_histogram_is_the_same_picture_shot_after_shot() -> None:
    """End to end: on every shot the live payload is the window counted against its own edges.

    The edges are the session's business -- a live session keeps the domain
    on screen until the data leaves it -- so the oracle counts the window's
    valid samples into whatever edges the session drew, the way the kind
    counted them before there was a table.
    """

    history = _History(capacity=8, high=300)
    snapshot = history.publish(0)
    window = 5
    live = PlotSession(
        snapshot, HistogramPlot(), parameters={"window": window, "bin_count": 16}
    )
    try:
        for index in range(1, 14):
            snapshot = history.publish(
                index,
                invalid=np.ones((HEIGHT, WIDTH), dtype=bool) if index % 3 == 0 else None,
            )
            live.update_data(snapshot)
            assert live._view._samples is None, "integer frequency edges do not need the raw window"
            edges = np.asarray(live._payload.edges.canonical, dtype=float)
            values, valid = DataView(snapshot).history_values(window)
            selected = np.asarray(values)[np.asarray(valid, dtype=bool)]
            expected, _edges = np.histogram(selected, bins=edges)
            np.testing.assert_array_equal(np.asarray(live._payload.counts), expected[None])
            assert int(expected.sum()) == int(selected.size), "the domain lost samples"
        assert live._view._frequency_carry is not None, "the session never used the table"
        live.replace_spec(HistogramPlot(group=AxisRef.cell_data("x")))
        assert live._view._frequency_carry is None, "Group no longer consumes the whole-pool table"
        live.replace_spec(HistogramPlot())
        assert live._view._frequency_carry is not None
        live.replace_spec(HistogramPlot(reduced=(AxisRef.point(str(PRIMARY_INDEX_AXIS_ID)),)))
        assert live._view._frequency_carry is None, "Reduced no longer consumes the whole-pool table"
    finally:
        live.close()


def test_a_uint64_history_past_int64_declines_the_table_and_still_draws() -> None:
    """The table is addressed by an int64 difference from its offset.

    Data allows uint64, whose upper half no int64 holds: three samples of
    2**63 + k span three levels, the optimisation took them, and the
    subtraction raised OverflowError out of ``window_frequency`` and out
    of the histogram panel drawn from it.  A level int64 cannot hold is
    declined -- by the fresh count and by a later shot that would widen a
    table to it -- and the histogram counts the pool the ordinary way.
    """

    history = _History(capacity=3, dtype=np.uint64)
    view = None
    for index in range(3):
        snapshot = history.publish(
            index, frame=np.full((HEIGHT, WIDTH), 2**63 + index + 1, dtype=np.uint64)
        )
        view = DataView(snapshot) if view is None else DataView(
            snapshot, inherit_domains_from=view
        )
    assert view.window_frequency(3) is None
    session = PlotSession(snapshot, HistogramPlot(), parameters={"window": 3})
    try:
        payload = session._payload
        assert int(np.sum(payload.counts)) == 3 * HEIGHT * WIDTH
    finally:
        session.close()

    # A table that int64 holds is kept until a shot brings a level it
    # cannot hold, and given up then rather than overflowed.
    history = _History(capacity=3, dtype=np.uint64, high=50)
    snapshot = history.publish(0)
    view = DataView(snapshot)
    assert view.window_frequency(3) is not None
    previous, snapshot = view, history.publish(
        1, frame=np.full((HEIGHT, WIDTH), 2**63 + 1, dtype=np.uint64)
    )
    view = DataView(snapshot, inherit_domains_from=previous)
    assert view.window_frequency(3) is None
