"""The compiled front kernels are a SPEED path, never a semantics path.

Each kernel in :mod:`zlc_plot._raster_kernels` mirrors a numpy reference
that stays the specification.  These tests run the same input through both
engines and assert bit equality -- not closeness -- so a kernel cannot
drift from its reference silently.  ``ZLC_PLOT_KERNELS`` is the switch the
comparison turns.
"""
from __future__ import annotations

import numpy as np
import pytest

from zlc_plot import _raster_kernels as kernels
from zlc_plot._image_raster import _area_mean, _reduction_starts
from zlc_plot.data_view import histogram_counts


def _both_engines(call):
    """Run ``call`` under each engine and return ``(reference, compiled)``."""

    previous = kernels.ENGINE
    try:
        kernels.ENGINE = "numpy"
        reference = call()
        kernels.ENGINE = "auto"
        compiled = call()
    finally:
        kernels.ENGINE = previous
    return reference, compiled


def test_the_block_mean_kernel_matches_reduceat_bit_for_bit() -> None:
    """Every block sum the kernel takes is one ``reduceat`` would have made.

    Covers the shapes a real panel produces -- a marginal reduction whose
    blocks are one or two samples wide, an exactly halving one, a ragged
    one -- and the masked case the kernel must decline.
    """

    pytest.importorskip("numba")
    rng = np.random.default_rng(11)
    cases = (
        (512, 378),   # the camera's own ratio, blocks of 1 and 2
        (512, 256),   # exactly halving
        (300, 97),    # ragged, blocks of 3 and 4
        (64, 63),     # one block of two, the rest of one
    )
    for source, target in cases:
        values = rng.integers(0, 65535, size=(source, source), dtype=np.uint16)
        starts = _reduction_starts(source, target, 1.25)
        valid = np.broadcast_to(np.True_, values.shape)
        reference, compiled = _both_engines(
            lambda: _area_mean(values, valid, starts, starts)
        )
        np.testing.assert_array_equal(reference, compiled)
        assert reference.dtype == compiled.dtype

    # A partly invalid plane is not the kernel's question: it must fall
    # through to the reference, which counts the contributing samples.
    values = rng.integers(0, 4095, size=(128, 128), dtype=np.uint16)
    valid = np.ones(values.shape, dtype=bool)
    valid[3:9, 4:20] = False
    starts = _reduction_starts(128, 90, 1.25)
    reference, compiled = _both_engines(
        lambda: _area_mean(values, valid, starts, starts)
    )
    np.testing.assert_array_equal(np.asarray(reference), np.asarray(compiled))


def test_the_block_mean_kernel_declines_sums_float32_cannot_hold() -> None:
    """Exactness is the kernel's licence, and it is judged from the dtype.

    A block wide enough to sum past 2**24 would round inside the
    reference's float32 accumulator, and an exact integer total would no
    longer be that reduction's answer -- so the kernel must not answer.
    """

    starts = np.array([0], dtype=np.intp)
    assert not kernels.block_sums_are_exact(
        np.dtype(np.uint16), starts, starts, (1024, 1024)
    )
    narrow = np.arange(0, 512, 2, dtype=np.intp)
    assert kernels.block_sums_are_exact(
        np.dtype(np.uint16), narrow, narrow, (512, 512)
    )
    assert not kernels.block_sums_are_exact(
        np.dtype(np.float32), narrow, narrow, (512, 512)
    )


def test_the_uniform_histogram_kernel_matches_numpy_bit_for_bit() -> None:
    """The same counts numpy's equal-bin path produces, including its edges.

    Samples are deliberately placed ON the edges, outside the range and at
    the inclusive last edge, because those are the only places where the
    index arithmetic and its two corrections can disagree.
    """

    pytest.importorskip("numba")
    rng = np.random.default_rng(5)
    edges = np.linspace(-3.0, 7.0, 41)
    pools = (
        rng.normal(size=200_003) * 2.0,
        np.concatenate([edges, edges - 1e-12, edges + 1e-12]),
        np.concatenate([rng.random(5_000) * 20.0 - 10.0, [-3.0, 7.0]]),
        (rng.random(50_000) * 6000).astype(np.uint16).astype(np.float64),
        np.full(1000, 7.0),
    )
    for pool in pools:
        reference, compiled = _both_engines(
            lambda: histogram_counts(pool, edges)
        )
        np.testing.assert_array_equal(reference, compiled)

    integers = (rng.random(80_000) * 500).astype(np.uint16)
    integer_edges = np.linspace(0.0, 500.0, 26)
    reference, compiled = _both_engines(
        lambda: histogram_counts(integers, integer_edges)
    )
    np.testing.assert_array_equal(reference, compiled)


def test_the_histogram_kernel_declines_a_float32_pool() -> None:
    """numpy would do that pool's arithmetic in float32; the kernel would not.

    Rather than reproduce a second precision, the dispatch defers -- and
    the counts are still numpy's, which is what the equality above asserts
    for every dtype the kernel does take.
    """

    rng = np.random.default_rng(7)
    pool = (rng.random(10_000) * 10.0).astype(np.float32)
    edges = np.linspace(0.0, 10.0, 21)
    reference, compiled = _both_engines(lambda: histogram_counts(pool, edges))
    np.testing.assert_array_equal(reference, compiled)


def test_the_colour_and_gather_kernels_match_their_references() -> None:
    """Colouring and nearest-neighbour resize, pixel for pixel.

    Both are pure per-element maps, so equality here is exact by
    construction -- the test exists to catch a kernel that stops mirroring
    its reference, not to discover a tolerance.
    """

    pytest.importorskip("numba")
    rng = np.random.default_rng(13)
    lut = rng.integers(0, 255, size=(256, 4), dtype=np.uint8)

    values = (rng.random((97, 131)) * 300.0 - 50.0).astype(np.float32)
    vmin, scale = np.float32(-20.0), np.float32(256.0 / 200.0)
    scaled = values.astype(np.float32, copy=True)
    scaled -= vmin
    scaled *= scale
    np.clip(scaled, 0.0, 255.0, out=scaled)
    reference = lut[scaled.astype(np.uint8)]
    compiled = np.empty(values.shape + (4,), dtype=np.uint8)
    kernels.colour_float32(np.ascontiguousarray(values), lut, vmin, scale,
                           compiled)
    np.testing.assert_array_equal(reference, compiled)

    codes = rng.integers(0, 65535, size=(53, 71), dtype=np.uint16)
    table = rng.integers(0, 255, size=(65536, 4), dtype=np.uint8)
    compiled = np.empty(codes.shape + (4,), dtype=np.uint8)
    kernels.colour_indexed(codes, table, compiled)
    np.testing.assert_array_equal(table[codes], compiled)

    rgba = rng.integers(0, 255, size=(64, 48, 4), dtype=np.uint8)
    row_map = np.minimum(((np.arange(90) + 0.5) * (64 / 90)).astype(np.intp), 63)
    column_map = np.minimum(((np.arange(37) + 0.5) * (48 / 37)).astype(np.intp), 47)
    compiled = np.empty((row_map.size, column_map.size, 4), dtype=np.uint8)
    kernels.gather_rows_columns(rgba, row_map, column_map, compiled)
    np.testing.assert_array_equal(rgba[row_map][:, column_map], compiled)


def test_the_extrema_kernel_matches_the_masked_reductions() -> None:
    """Same three numbers as isfinite + any + min(where=) + max(where=).

    Extrema are order-independent, so parallel partials are exact -- the
    cases that matter are the ones with nothing to reduce: an all-invalid
    pool, an all-NaN pool, and infinities that must not become extremes.
    """

    pytest.importorskip("numba")
    rng = np.random.default_rng(17)
    pools = (
        rng.normal(size=200_003),
        np.concatenate([rng.normal(size=1000), [np.nan, np.inf, -np.inf]]),
        np.full(500, np.nan),
        np.array([3.5]),
    )
    for pool in pools:
        for mask in (None, rng.random(pool.size) > 0.3, np.zeros(pool.size, bool)):
            finite = np.isfinite(pool)
            if mask is not None:
                finite = finite & mask
            expected = (
                int(finite.sum()),
                float(np.min(pool, where=finite, initial=np.inf)),
                float(np.max(pool, where=finite, initial=-np.inf)),
            )
            got = kernels.masked_finite_extrema(pool, mask)
            assert got is not None
            assert got[0] == expected[0]
            if expected[0]:
                assert got[1] == expected[1]
                assert got[2] == expected[2]


def test_the_finite_probe_takes_the_same_leading_values() -> None:
    """Block-built masks pick the same values, in the same order."""

    from zlc_plot.data_view import _finite_probe, finite_probe

    rng = np.random.default_rng(19)
    pool = rng.normal(size=200_000)
    pool[::997] = np.nan
    mask = rng.random(pool.size) > 0.2
    finite = np.isfinite(pool) & mask
    np.testing.assert_array_equal(
        _finite_probe(pool, finite), finite_probe(pool, mask)
    )
