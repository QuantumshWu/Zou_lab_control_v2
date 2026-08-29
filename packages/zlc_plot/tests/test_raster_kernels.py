"""The compiled front kernels are a SPEED path, never a semantics path.

Each kernel in :mod:`zlc_plot._raster_kernels` mirrors a numpy reference
that stays the specification.  These tests run the same input through both
engines and assert bit equality -- not closeness -- so a kernel cannot
drift from its reference silently.  ``ZLC_PLOT_KERNELS`` is the switch the
comparison turns.

ONE kernel cannot promise that, and says so where it is tested: summing a
floating plane in a different order is a different answer in the last
bits, always.  Its contract is the stronger one that bit equality was
standing in for -- it must be at least as close to a float64 reduction as
the reference is -- because the reference accumulates float32 planes in
float32, and the kernel does not.
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

    # A partly invalid plane sums and counts in one compiled pass rather
    # than materialising np.where(valid, values, 0) and reducing twice.
    # For exact integers that is the reference's answer, bit for bit.
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


def test_the_float_block_mean_is_no_further_from_the_truth_than_reduceat() -> None:
    """A floating plane cannot promise bit equality, so it promises more.

    ``np.add.reduceat`` on a float32 plane accumulates in float32 and lands
    about 1e-7 relative away from a float64 reduction of the same numbers.
    The kernel accumulates in float64 and lands on it.  Asserting closeness
    alone would let a future kernel get worse and still pass, so the
    assertion is the comparison itself: for every case, the compiled answer
    is at least as near the float64 truth as the reference answer is.
    """

    pytest.importorskip("numba")
    rng = np.random.default_rng(29)
    cases = (
        (np.float32, 512, 378),
        (np.float32, 512, 256),
        (np.float32, 300, 97),
        (np.float64, 512, 378),
        (np.float64, 300, 97),
    )
    improved = 0
    for dtype, source, target in cases:
        values = (rng.random((source, source)) * 4000.0).astype(dtype)
        starts = _reduction_starts(source, target, 1.25)
        valid = np.broadcast_to(np.True_, values.shape)
        reference, compiled = _both_engines(
            lambda: _area_mean(values, valid, starts, starts)
        )
        assert reference.dtype == compiled.dtype
        truth = _area_mean(
            values.astype(np.float64), valid, starts, starts
        )
        reference_error = np.abs(np.asarray(reference, dtype=np.float64) - truth)
        compiled_error = np.abs(np.asarray(compiled, dtype=np.float64) - truth)
        assert compiled_error.max() <= reference_error.max(), (
            "%s %d->%d: the kernel is further from a float64 reduction "
            "(%.3e) than reduceat is (%.3e)"
            % (np.dtype(dtype).name, source, target,
               compiled_error.max(), reference_error.max())
        )
        if compiled_error.max() < reference_error.max():
            improved += 1
        # And still the same picture: a relative difference far below
        # anything a colour LUT or a bar height can show.
        scale = np.abs(truth).max()
        assert np.abs(
            np.asarray(compiled, dtype=np.float64) - np.asarray(reference,
                                                                dtype=np.float64)
        ).max() <= 1e-6 * scale
    assert improved, (
        "no case improved: the float32 comparison is not exercising the "
        "float32 accumulator this kernel exists to beat"
    )


def test_the_masked_block_mean_counts_what_it_summed() -> None:
    """Sum and count come out of one pass, so they cannot disagree.

    The path this replaced built a whole zero-filled plane and reduced it
    twice, once for each.  A cell with nothing valid in it must still come
    back masked, not as a division by zero.
    """

    pytest.importorskip("numba")
    rng = np.random.default_rng(31)
    values = (rng.random((128, 128)) * 100.0).astype(np.float32)
    valid = np.ones(values.shape, dtype=bool)
    valid[3:9, 4:20] = False
    starts = _reduction_starts(128, 90, 1.25)
    reference, compiled = _both_engines(
        lambda: _area_mean(values, valid, starts, starts)
    )
    np.testing.assert_allclose(
        np.asarray(compiled, dtype=np.float64),
        np.asarray(reference, dtype=np.float64),
        rtol=1e-6,
    )

    # A block with no valid sample at all: masked on both engines, and the
    # mask must agree cell for cell.
    valid[:] = True
    valid[:16, :16] = False
    reference, compiled = _both_engines(
        lambda: _area_mean(values, valid, starts, starts)
    )
    assert isinstance(compiled, np.ma.MaskedArray), (
        "an empty block must come back masked, not divided by zero"
    )
    np.testing.assert_array_equal(
        np.ma.getmaskarray(compiled), np.ma.getmaskarray(reference)
    )


def test_the_kernel_cache_is_a_plainly_named_folder_in_the_checkout() -> None:
    """It belongs to this checkout, and it is not a hidden dotfile.

    The cache holds machine code compiled from THESE sources, so it sits in
    the checkout beside them rather than in some per-user cache area.  It is
    named without a leading dot on purpose: it is a build product an operator
    may want to find and delete, not a private dotfile to hide from them.

    Two modules and one batch file each carried their own copy of the path,
    which is why this asserts there is one owner and that everyone asks it.
    """

    import os
    import pathlib

    from zlc_plot import _kernel_cache

    chosen = pathlib.Path(_kernel_cache.kernel_cache_dir()).resolve()
    checkout = pathlib.Path(_kernel_cache.__file__).resolve().parents[4]
    assert chosen.parent == checkout, (
        "the kernel cache is not at the checkout root: %s" % chosen
    )
    assert not chosen.name.startswith("."), (
        "the kernel cache is hidden behind a leading dot: %s" % chosen.name
    )

    # And the environment override still wins, which is what a sandbox or a
    # read-only checkout needs.
    previous = os.environ.get("NUMBA_CACHE_DIR")
    try:
        os.environ["NUMBA_CACHE_DIR"] = "somewhere/else"
        assert _kernel_cache.install() == "somewhere/else"
    finally:
        if previous is None:
            os.environ.pop("NUMBA_CACHE_DIR", None)
        else:
            os.environ["NUMBA_CACHE_DIR"] = previous


def test_no_module_keeps_its_own_copy_of_the_cache_path() -> None:
    """The path has one owner; a second copy is how a move half-lands."""

    import pathlib

    package = pathlib.Path(__file__).resolve().parents[1] / "src" / "zlc_plot"
    offenders = [
        path.name
        for path in package.glob("*.py")
        if path.name != "_kernel_cache.py"
        and "NUMBA_CACHE_DIR" in path.read_text(encoding="utf-8")
        and "os.environ[\"NUMBA_CACHE_DIR\"] =" in path.read_text(encoding="utf-8")
    ]
    assert not offenders, (
        "these modules set NUMBA_CACHE_DIR themselves instead of asking "
        "_kernel_cache: %s" % offenders
    )


def test_the_centred_square_kernel_is_a_reduction_not_a_copy() -> None:
    """One compiled specialization serves every signal rank.

    The kept axes of a reduction are a BLOCK of the tensor, so the tensor
    always views as (before, block, after) and the kernel is written on
    that three-dimensional spelling alone -- a curve over point rows, a
    grouped band and a two-dimensional scan heatmap all reach the same
    compiled code, which is the only way a cache of compiled kernels is
    worth having.

    THE REFERENCE IS THE EINSUM, not a second numpy spelling of the
    kernel: without the compiled engine the helper declines and its one
    caller takes the einsum path it always had.  So the assertions are
    that it declines when it must, and that where it does answer the
    answer is the einsum's to within a summation order -- which is all a
    different order can ever promise.
    """

    from zlc_plot.data_view import _centred_square_sums

    rng = np.random.default_rng(4)
    shape = (7, 40, 3, 5)
    letters = "abcd"
    plane = rng.normal(0.0, 1.0, shape)
    offset = 0.37
    previous = kernels.ENGINE
    try:
        kernels.ENGINE = "numpy"
        assert _centred_square_sums(plane, offset, None, [1], shape) is None
        kernels.ENGINE = "auto"
        if not kernels.engaged():
            pytest.skip("no compiled engine available")
        for kept in ([1], [0, 1], [1, 2]):
            compiled = _centred_square_sums(plane, offset, None, kept, shape)
            assert compiled is not None
            centred = plane - offset
            einsum = np.einsum(
                f"{letters},{letters}->{''.join(letters[axis] for axis in kept)}",
                centred,
                centred,
            )
            assert compiled.shape == einsum.shape
            assert np.allclose(compiled, einsum, rtol=1e-12, atol=0.0)

        # Kept axes that are not one block have no three-dimensional view,
        # and the helper says so rather than copying to make one.
        assert _centred_square_sums(plane, offset, None, [1, 3], shape) is None

        # A non-contiguous plane is declined for the same reason: the
        # kernel takes C-contiguous input so one layout compiles, not two.
        assert (
            _centred_square_sums(
                np.asfortranarray(plane), offset, None, [1], shape
            )
            is None
        )

        marks = rng.random(shape) > 0.3
        masked = _centred_square_sums(plane, offset, marks, [1], shape)
        assert masked is not None
        assert np.allclose(
            masked,
            np.sum(
                np.square(plane - offset),
                axis=(0, 2, 3),
                where=marks,
                dtype=np.float64,
            ),
            rtol=1e-12,
            atol=0.0,
        )
    finally:
        kernels.ENGINE = previous


def test_an_input_s_mutability_is_not_an_accident_of_where_it_came_from() -> None:
    """One signature per dtype, not two.

    Numba types an array's mutability: ``array(uint16, 2d, C)`` and
    ``readonly array(uint16, 2d, C)`` are different types and compile the
    same kernel twice.  Which one a plane is, is decided by whether
    something upstream had to copy it -- ``ascontiguousarray`` returns a
    read-only contiguous array unchanged but must COPY a strided one, and a
    fresh copy is writable.  Which is to say: by whether the operator had
    zoomed.  Measured across the image dtypes with and without a zoom, 23
    compiled signatures of which 10 were the same code again.
    """

    numba = pytest.importorskip("numba")

    sealed = np.zeros((8, 8), dtype=np.uint16)
    sealed.setflags(write=False)
    strided = np.zeros((8, 16), dtype=np.uint16)[:, 2:10]
    strided.setflags(write=False)
    writable = np.zeros((8, 8), dtype=np.uint16)

    # The premise: without sealing, these are three different numba types.
    raw = {
        str(numba.typeof(np.ascontiguousarray(item)))
        for item in (sealed, strided, writable)
    }
    assert len(raw) > 1, raw

    typed = {
        str(numba.typeof(kernels.readable(item)))
        for item in (sealed, strided, writable)
    }
    assert len(typed) == 1, typed
    assert "readonly" in typed.pop()

    # And sealing a caller's array is a side effect on a value they own.
    assert writable.flags.writeable
    assert not kernels.readable(writable).flags.writeable
    assert np.array_equal(kernels.readable(strided), strided)
