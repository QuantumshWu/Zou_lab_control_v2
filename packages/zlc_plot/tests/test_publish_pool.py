"""A published front is recycled, and never while anything can still read it.

The pool trades a fresh eighteen-megabyte allocation per frame for a buffer
that is already resident.  Its whole correctness rests on one property --
a buffer is reissued only after the last reference to the view handed out
for it is gone -- so that property is asserted directly, including the case
that would corrupt a frame: a holder that keeps its front.
"""
from __future__ import annotations

import numpy as np
import pytest

from data_factory import Axis, DatasetSchema, DatasetSnapshot, PointTable

from zlc_plot import AxisRef, CurvePlot, PlotSession
from zlc_plot.raster import RasterBuffer
from zlc_plot.rendering import PublishBufferPool


def _session() -> PlotSession:
    rng = np.random.default_rng(1)
    schema = DatasetSchema.create(
        Axis.create("repeat", size=4),
        PointTable.from_columns({"x": np.arange(48.0)}),
        generation="publish-pool",
    )
    return PlotSession(
        DatasetSnapshot(schema, rng.normal(size=(4, 48)), revision=1),
        CurvePlot(AxisRef.point("x")),
        device_pixel_ratio=2.0,
    )


def test_a_released_buffer_is_the_one_reissued() -> None:
    """Steady state stops allocating: the same storage comes back around."""

    pool = PublishBufferPool()
    writable, published = pool.take(1024)
    # The writable view is taken straight from the recycled block, so its
    # object IS the block; the published view wraps it once more, and that
    # wrapper is deliberately new each time -- only the bytes are reused.
    block = writable.obj
    del writable, published
    again_writable, again_published = pool.take(1024)
    assert again_writable.obj is block


def test_a_held_buffer_is_never_reissued() -> None:
    """The failure mode of a holder is a fresh allocation, not shared pixels."""

    pool = PublishBufferPool()
    held = []
    blocks = []
    for _ in range(PublishBufferPool.DEPTH + 2):
        writable, published = pool.take(1024)
        held.append(published)
        blocks.append(writable.obj)
        del writable
    assert len({id(block) for block in blocks}) == len(blocks)


def test_a_size_change_drops_the_pooled_buffers() -> None:
    """A resized surface must not be served a buffer of the old size."""

    pool = PublishBufferPool()
    writable, published = pool.take(1024)
    del writable, published
    assert pool._free
    other_writable, other_published = pool.take(2048)
    assert other_published.nbytes == 2048
    assert not pool._free


def test_a_published_front_cannot_be_written() -> None:
    """The guarantee to a holder is unchanged: these bytes are immutable."""

    session = _session()
    try:
        front = session.rgba()
        assert not front.flags.writeable
        with pytest.raises(ValueError):
            front[0, 0, 0] = 1
        raw, height, width = session._raster_capture_rgba_bytes()
        assert memoryview(raw).readonly
        buffer = RasterBuffer(width, height, raw)
        assert not buffer.as_rgba().flags.writeable
    finally:
        session.close()


def test_a_front_kept_across_frames_keeps_its_own_pixels() -> None:
    """The whole point, end to end: an old front does not become a new one."""

    session = _session()
    try:
        rng = np.random.default_rng(2)
        schema = session._projection.data.block.schema
        session.rgba()
        kept = session.rgba()
        # The property is about THIS array: whatever it held when it was
        # published, it must still hold after the pool has cycled.
        first = np.array(kept, copy=True)
        for revision in range(2, 8):
            session.update_data(
                DatasetSnapshot(
                    schema, rng.normal(size=(4, 48)) * revision, revision=revision
                )
            )
            session.rgba()
        np.testing.assert_array_equal(np.asarray(kept), first)
    finally:
        session.close()
