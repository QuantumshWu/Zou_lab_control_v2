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

from data_factory import (
    make_dataset_schema,
    make_snapshot,
    mapped_domain_from_columns,
    repeat_domain,
)


from zlc_plot import AxisRef, CurvePlot, PlotSession
from zlc_plot.raster import RasterBuffer
from zlc_plot.rendering import PublishBufferPool, install_publish_pool


def _free_blocks(pool) -> int:
    """How many blocks the shared pool is holding spare, across all sizes."""

    return len(pool._free)

def _session() -> PlotSession:
    rng = np.random.default_rng(1)
    schema = make_dataset_schema(
        repeat_domain(size=4),
        mapped_domain_from_columns({"x": np.arange(48.0)}),
    )
    return PlotSession(
        make_snapshot(schema, rng.normal(size=(4, 48)), revision=1),
        CurvePlot(AxisRef.point("x")),
        device_pixel_ratio=2.0,
    )

@pytest.mark.parametrize("surfaces", (1, 4))
def test_a_released_buffer_is_the_one_reissued(surfaces) -> None:
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

    from zlc_plot.render_process import _SharedFrontPool

    shared = _SharedFrontPool()
    try:
        first = [shared.publish(bytes(1024)) for _ in range(surfaces)]
        held_id, held_name, _size = shared.publish(bytes([7]) * 1024)
        first_names = {name for _lease, name, _size in first}
        assert held_name not in first_names
        for lease, _name, _size in first:
            shared.release(lease, surfaces)
        reissued = [shared.publish(bytes(1024)) for _ in range(surfaces)]
        assert {name for _lease, name, _size in reissued} == first_names
        assert bytes(shared._leased[held_id].memory.buf) == bytes([7]) * 1024
        for lease, _name, _size in reissued:
            shared.release(lease, surfaces)
        shared.trim_free(1)
        assert _free_blocks(shared) == 1
        for size in range(2, 8):
            lease, _name, _size = shared.publish(bytes(size * 1024))
            shared.release(lease, surfaces)
            assert _free_blocks(shared) <= surfaces
        shared.trim_free(0)
        assert not shared._free
        assert held_id in shared._leased
    finally:
        shared.close()

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
                make_snapshot(
                    schema, rng.normal(size=(4, 48)) * revision, revision=revision
                )
            )
            session.rgba()
        np.testing.assert_array_equal(np.asarray(kept), first)
    finally:
        session.close()


def test_a_front_written_in_the_shared_block_is_published_without_a_copy() -> None:
    """The change: the block a renderer fills IS the one the frontend maps.

    A front used to be written into private memory and copied into a shared
    segment afterwards -- eighteen megabytes twice, per frame, per panel.
    The claim is what says the copy is not needed, and a buffer that did not
    come from this pool still has to be copied, so both are asserted.
    """

    from zlc_plot.render_process import _SharedFrontPool

    shared = _SharedFrontPool()
    try:
        writable, published = shared.take(1024)
        writable[:] = bytes([3]) * 1024
        del writable
        claimed = shared.claim(published)
        assert claimed is not None
        lease_id, name, nbytes = claimed
        assert nbytes == 1024
        block = shared._leased[lease_id]
        assert block.memory.name == name
        # No copy happened, so the segment already holds what was written.
        assert bytes(block.memory.buf[:16]) == bytes([3]) * 16

        # A buffer from anywhere else has no lease to give.
        assert shared.claim(memoryview(bytes(1024))) is None
    finally:
        shared.close()


def test_a_shared_block_is_free_only_when_both_hands_let_go() -> None:
    """Two processes hold it; only one of them has an interpreter.

    Recycled on either release alone, the renderer would be filling a block
    the frontend is still painting from -- the failure this pool exists to
    make impossible, and the one that cannot be seen in a screenshot.
    """

    from zlc_plot.render_process import _SharedFrontPool

    shared = _SharedFrontPool()
    try:
        _writable, published = shared.take(1024)
        lease_id, _name, _size = shared.claim(published)

        # The frontend lets go first; the renderer still holds its view.
        shared.release(lease_id, 4)
        assert _free_blocks(shared) == 0
        del published
        assert _free_blocks(shared) == 1

        # And the other order.
        writable, published = shared.take(1024)
        del writable
        lease_id, _name, _size = shared.claim(published)
        del published
        assert _free_blocks(shared) == 0
        shared.release(lease_id, 4)
        assert _free_blocks(shared) == 1
    finally:
        shared.close()


def test_an_installed_pool_is_the_one_a_new_renderer_publishes_through() -> None:
    """The seam the child uses, asserted where it is, not where it is called."""

    from zlc_plot.render_process import _SharedFrontPool

    shared = _SharedFrontPool()
    install_publish_pool(shared)
    try:
        session = _session()
        try:
            raw, _height, _width = session._raster_capture_rgba_bytes()
            claimed = shared.claim(raw)
            assert claimed is not None, "a front was not written in the pool"
        finally:
            session.close()
    finally:
        install_publish_pool(None)
        shared.close()
