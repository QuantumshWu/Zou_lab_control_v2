"""The window histogram is moved by the shots that entered and left, exactly.

A pixel-pool histogram at a deep window used to recount every value of the
window on every shot.  The count of each value is a sum over shots, so the
view keeps one frequency table for the window and moves it by the shot
that arrived and the one that left -- and the contract is EXACTNESS: at
every revision the moved table equals a fresh count of the same window,
holes, invalid samples, replacements and window changes included.
"""

from __future__ import annotations

import numpy as np
import pytest

from data_factory import (
    axis,
    make_dataset_schema,
    mapped_domain_from_columns,
    repeat_domain,
)
from zlc_data import DatasetSchema
from zlc_data import PRIMARY_INDEX, IndexedWindow, owned_snapshot_from_arrays
from zlc_data.snapshot_projection import PRIMARY_INDEX_AXIS_ID
from zlc_plot import HistogramPlot, PlotSession
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


def test_the_table_moves_with_the_window_and_stays_exact() -> None:
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

    floats = owned_snapshot_from_arrays(
        schema=_schema((-1, 0), np.float64),
        values=np.stack([_frame(1), _frame(2)])[None].astype(np.float64),
        revision=1,
        window=IndexedWindow(0, 1, -1),
    )
    assert DataView(floats).window_frequency(2) is None


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
            edges = np.asarray(live._payload.edges.canonical, dtype=float)
            values, valid = DataView(snapshot).history_values(window)
            selected = np.asarray(values)[np.asarray(valid, dtype=bool)]
            expected, _edges = np.histogram(selected, bins=edges)
            np.testing.assert_array_equal(np.asarray(live._payload.counts), expected)
            assert int(expected.sum()) == int(selected.size), "the domain lost samples"
        assert live._view._frequency_carry is not None, "the session never used the table"
    finally:
        live.close()
