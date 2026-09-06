"""The compiled front kernels are a SPEED path, never a semantics path.

Each kernel in :mod:`zlc_plot._raster_kernels` mirrors a numpy reference
that stays the specification.  These tests run the same input through both
engines and assert bit equality -- not closeness -- so a kernel cannot
drift from its reference silently.  ``ZLC_PLOT_KERNELS`` is the switch the
comparison turns.

ONE kernel cannot promise that, and says so where it is tested: summing a
floating plane in a different order is a different answer in the last
bits, always.  Its contract is the one bit equality was standing in for:
both engines accumulate a floating plane in float64 and narrow only the
quotient, so each lands within float32 rounding of a float64 reduction,
and a finite plane near float32's range comes back finite.
"""
from __future__ import annotations

import numpy as np
import pytest

from zlc_plot import _raster_kernels as kernels
from zlc_plot._image_raster import _area_mean, _reduction_starts
from zlc_plot import Reduction
from zlc_plot.data_view import (
    _aggregate_by_codes,
    _axis_kernel_aggregate,
    _facet_kernel_counts,
    _masked_leading_reduce,
    histogram_counts,
)


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

    values = rng.normal(size=(5, 7, 3, 11))
    valid = rng.random(values.shape) > 0.2
    facet_codes = np.asarray([2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    expected = []
    previous = kernels.ENGINE
    try:
        kernels.ENGINE = "numpy"
        for facet in range(3):
            selected = np.flatnonzero(facet_codes == facet)
            expected.append(histogram_counts(
                values[..., selected],
                edges,
                valid[..., selected],
            ))
        kernels.ENGINE = "auto"
        batched = _facet_kernel_counts(
            values, valid, facet_codes, 3, 3, edges
        )
    finally:
        kernels.ENGINE = previous
    assert batched is not None
    np.testing.assert_array_equal(np.asarray(expected), batched)


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


def test_joint_axis_kernel_matches_the_exact_bucket_reduction() -> None:
    pytest.importorskip("numba")
    rng = np.random.default_rng(11)
    values = rng.normal(size=(3, 7, 5))
    valid = rng.random(values.shape) > 0.2
    row_codes = np.asarray([1, 0, 2, 1, 0, 2, 1], dtype=np.int64)
    data_codes = np.asarray([2, 0, 1, 2, 0], dtype=np.int64)
    combined = np.broadcast_to(
        row_codes.reshape(1, 7, 1) * 3 + data_codes.reshape(1, 1, 5),
        values.shape,
    ).reshape(-1)
    usable = valid.reshape(-1)
    for reduction in Reduction:
        expected, expected_counts = _aggregate_by_codes(
            values.reshape(-1), usable, combined, 9, reduction
        )
        got = _axis_kernel_aggregate(
            values,
            valid,
            (row_codes, data_codes),
            (1, 2),
            (3, 3),
            reduction,
        )
        assert got is not None
        actual, actual_counts, presence = got
        np.testing.assert_array_equal(expected, actual)
        np.testing.assert_array_equal(expected_counts, actual_counts)
        np.testing.assert_array_equal(presence, np.ones(9, dtype=np.bool_))


@pytest.mark.parametrize("reduction", (Reduction.MIN, Reduction.MAX))
def test_joint_axis_extrema_preserve_nan_and_signed_zero(reduction) -> None:
    pytest.importorskip("numba")
    values = np.asarray([[0.0, -0.0, np.nan, 2.0]], dtype=np.float64)
    valid = np.ones(values.shape, dtype=np.bool_)
    codes = np.asarray([0, 0, 1, 1], dtype=np.int64)
    expected, expected_counts = _aggregate_by_codes(
        values.reshape(-1), valid.reshape(-1), codes, 2, reduction
    )
    got = _axis_kernel_aggregate(
        values, valid, (codes,), (1,), (2,), reduction
    )
    assert got is not None
    actual, actual_counts, presence = got
    np.testing.assert_array_equal(expected.view(np.uint64), actual.view(np.uint64))
    np.testing.assert_array_equal(expected_counts, actual_counts)
    np.testing.assert_array_equal(presence, np.ones(2, dtype=np.bool_))


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
    whole_but_last = np.zeros(200_003)
    whole_but_last[-1] = 0.5
    pools = (
        rng.normal(size=200_003),
        np.concatenate([rng.normal(size=1000), [np.nan, np.inf, -np.inf]]),
        np.full(500, np.nan),
        np.array([3.5]),
        np.arange(300.0),
        whole_but_last,
        np.concatenate([np.arange(64.0), [np.nan, np.inf, 7.0, -12.0]]),
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
                bool(np.all(pool == np.floor(pool), where=finite)),
            )
            got = kernels.masked_finite_extrema(pool, mask)
            assert got is not None
            assert got[0] == expected[0]
            if expected[0]:
                assert got[1] == expected[1]
                assert got[2] == expected[2]
            # Whether every finite sample is whole is a fact about every
            # sample: a pool whole but for its last value is not whole.
            assert got[3] is expected[3], (pool.size, mask is None)


def test_the_float_block_mean_is_within_float32_rounding_on_both_engines() -> None:
    """A floating plane cannot promise bit equality, so it promises this.

    Both engines accumulate a floating plane in float64 and narrow only
    the quotient, so each answer is within a few ulps of its own dtype
    from a float64 reduction -- for a float32 plane the rounding of the
    quotient, for a float64 plane the summation order -- and the two
    engines agree to the same.  A reference that accumulated a float32
    plane in float32 landed 1e-7 relative away from the truth, and turned
    a finite plane near float32's range into infinite means.
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
    for dtype, source, target in cases:
        values = (rng.random((source, source)) * 4000.0).astype(dtype)
        starts = _reduction_starts(source, target, 1.25)
        valid = np.broadcast_to(np.True_, values.shape)
        reference, compiled = _both_engines(
            lambda: _area_mean(values, valid, starts, starts)
        )
        assert reference.dtype == compiled.dtype == np.result_type(dtype, np.float32)
        truth = _area_mean(
            values.astype(np.float64), valid, starts, starts
        )
        tolerance = 4 * np.finfo(reference.dtype).eps * np.abs(truth).max()
        for engine, answer in (("reference", reference), ("compiled", compiled)):
            error = np.abs(np.asarray(answer, dtype=np.float64) - truth).max()
            assert error <= tolerance, (
                "%s %d->%d: the %s answer is %.3e from a float64 reduction, "
                "past float32 rounding (%.3e)"
                % (np.dtype(dtype).name, source, target, engine, error, tolerance)
            )
        assert np.abs(
            np.asarray(compiled, dtype=np.float64) - np.asarray(reference,
                                                                dtype=np.float64)
        ).max() <= tolerance


def test_a_finite_float32_plane_near_its_range_has_a_finite_mean() -> None:
    """The block SUM is never narrowed to float32 before the division.

    Four finite float32 samples of 3e38 have a block total of 1.2e39,
    past float32; written back as float32 on the way to the mean, the
    mean of finite samples came out infinite on both engines.  The mean
    itself, 3e38, is a float32 number, and that is the answer.
    """

    pytest.importorskip("numba")
    values = np.full((2, 2), 3.0e38, dtype=np.float32)
    values[0, 1] = np.float32(3.0000001e38)
    valid = np.broadcast_to(np.True_, values.shape)
    starts = np.array([0], dtype=np.intp)
    truth = np.float32(np.mean(values.astype(np.float64)))
    reference, compiled = _both_engines(
        lambda: _area_mean(values, valid, starts, starts)
    )
    for answer in (reference, compiled):
        assert answer.dtype == np.float32
        assert np.isfinite(answer).all()
        assert answer[0, 0] == truth
    # With a hole the same total flows through the masked face.
    holed = np.ones(values.shape, dtype=bool)
    holed[1, 1] = False
    truth = np.float32(np.mean(values[holed].astype(np.float64)))
    reference, compiled = _both_engines(
        lambda: _area_mean(values, holed, starts, starts)
    )
    for answer in (reference, compiled):
        assert np.isfinite(np.asarray(answer)).all()
        assert np.asarray(answer)[0, 0] == truth


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


def test_an_installed_wheel_has_no_checkout_to_cache_in(monkeypatch, tmp_path) -> None:
    """Four parents up from site-packages is nobody's folder.

    ``C:/Python313/Lib/site-packages/zlc_plot/_kernel_cache.py`` counted up
    to ``C:/`` and named ``C:/numba_cache`` the checkout's cache.  Outside a
    checkout the answer is numba's own default beside the module, and
    ``install`` sets nothing so numba keeps its read-only fallback.
    """

    import os

    from zlc_plot import _kernel_cache

    installed = tmp_path / "Lib" / "site-packages" / "zlc_plot" / "_kernel_cache.py"
    installed.parent.mkdir(parents=True)
    installed.write_text("", encoding="utf-8")
    monkeypatch.setattr(_kernel_cache, "__file__", str(installed))
    monkeypatch.delenv("NUMBA_CACHE_DIR", raising=False)
    assert _kernel_cache.kernel_cache_dir() == installed.parent / "__pycache__"
    assert _kernel_cache.install() == ""
    assert "NUMBA_CACHE_DIR" not in os.environ


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

    # The stand-in a maskless call hands the extrema kernel is sealed like
    # a mask, so with and without a mask are ONE signature.
    pool = np.arange(8.0)
    assert kernels.masked_finite_extrema(pool, None) is not None
    assert kernels.masked_finite_extrema(pool, pool > 2.0) is not None
    mask_types = {str(signature[1]) for signature in kernels.finite_extrema.signatures}
    assert len(mask_types) == 1 and "readonly" in mask_types.pop(), mask_types


def test_masked_leading_tensor_matrix_matches_numpy_bit_for_bit() -> None:
    """Pool size, reducer and holes never select different arithmetic."""

    pytest.importorskip("numba")
    rng = np.random.default_rng(29)
    previous = kernels.ENGINE
    try:
        for pool in (1, 2, 4, 8, 32):
            values = rng.normal(size=(pool, 257))
            # Equal signed zeros expose an extrema kernel that keeps the first
            # tie where NumPy keeps the last one.
            values[:, 0] = np.resize(np.asarray((0.0, -0.0)), pool)
            holey = rng.random(values.shape) > 0.25
            holey[:, 1] = False
            for valid in (
                np.broadcast_to(np.asarray(True), values.shape),
                holey,
            ):
                for reduction in Reduction:
                    kernels.ENGINE = "numpy"
                    reference = _masked_leading_reduce(
                        values, valid, reduction
                    )
                    kernels.ENGINE = "auto"
                    compiled = _masked_leading_reduce(
                        values, valid, reduction
                    )
                    np.testing.assert_array_equal(
                        compiled[0], reference[0]
                    )
                    np.testing.assert_array_equal(
                        compiled[1], reference[1]
                    )

        # Above the measured dispatch crossover the compiled path must
        # actually engage, still under every reducer and pool size.  Keeping
        # the output fixed distinguishes pool work from output construction.
        for pool in (2, 4, 8, 32):
            values = rng.normal(size=(pool, 16384))
            valid = rng.random(values.shape) > 0.2
            for reduction in Reduction:
                kernels.ENGINE = "numpy"
                reference = _masked_leading_reduce(values, valid, reduction)
                kernels.ENGINE = "auto"
                compiled = _masked_leading_reduce(values, valid, reduction)
                np.testing.assert_array_equal(compiled[0], reference[0])
                np.testing.assert_array_equal(compiled[1], reference[1])

        # The common (x, pool) source becomes a non-contiguous (pool, x)
        # reduction view.  Copying it to satisfy one compiled signature both
        # costs a full plane and changes NumPy's float summation order; the
        # dispatcher must decline it.  A single output column also stays on
        # NumPy's pairwise reduction because there is no prange width.
        original = rng.normal(size=(8192, 4, 1))
        original_valid = rng.random(original.shape) > 0.2
        strided = np.reshape(
            np.moveaxis(original, (0, 2), (1, 2)), (4, 8192)
        )
        strided_valid = np.reshape(
            np.moveaxis(original_valid, (0, 2), (1, 2)), (4, 8192)
        )
        assert not strided.flags.c_contiguous
        one_column = rng.normal(size=(32768, 1))
        one_valid = rng.random(one_column.shape) > 0.2
        for values, valid in (
            (strided, strided_valid),
            (one_column, one_valid),
        ):
            for reduction in Reduction:
                kernels.ENGINE = "numpy"
                reference = _masked_leading_reduce(values, valid, reduction)
                kernels.ENGINE = "auto"
                compiled = _masked_leading_reduce(values, valid, reduction)
                np.testing.assert_array_equal(compiled[0], reference[0])
                np.testing.assert_array_equal(compiled[1], reference[1])
    finally:
        kernels.ENGINE = previous


def test_a_float32_stack_reduces_in_float64_on_both_engines() -> None:
    """Three float32 samples 1e8, 1, -1e8 sum to 1, and their mean is 1/3.

    Every sample is a float32 number; only the accumulation can lose the
    1, and the plain leading-axis reduction did, in float32, while the
    bucket path beside it accumulated in float64: the same public curve
    answered 0 or 1/3 by which layout the data took.
    """

    values = np.asarray(
        [[1e8, 2e8], [1.0, 2.0], [-1e8, -2e8]], dtype=np.float32
    )
    all_valid = np.broadcast_to(np.asarray(True), values.shape)
    holey = np.ones(values.shape, dtype=np.bool_)
    holey[1, 1] = False
    for valid, mean, total in (
        (all_valid, [1 / 3, 2 / 3], [1.0, 2.0]),
        (holey, [1 / 3, 0.0], [1.0, 0.0]),
    ):
        for aggregation, expected in (
            (Reduction.MEAN, mean), (Reduction.SUM, total)
        ):
            reference, compiled = _both_engines(
                lambda: _masked_leading_reduce(values, valid, aggregation)
            )
            for answer, _counts in (reference, compiled):
                np.testing.assert_allclose(
                    np.asarray(answer, dtype=np.float64), expected, rtol=1e-12
                )


def _ellipse_distance_reference(dx, dy, radius_x, radius_y):
    """Brute force: the nearest of many boundary samples."""

    theta = np.linspace(0.0, 2.0 * np.pi, 65537)
    boundary_x = radius_x * np.cos(theta)
    boundary_y = radius_y * np.sin(theta)
    return float(np.min(np.hypot(boundary_x - dx, boundary_y - dy)))


def _ellipse_distance_map(center_x, center_y, radius_x, radius_y, height, width):
    """Brute force for every pixel centre at once."""

    theta = np.linspace(0.0, 2.0 * np.pi, 16385)
    boundary_x = radius_x * np.cos(theta) + center_x
    boundary_y = radius_y * np.sin(theta) + center_y
    centres_x = np.arange(width) + 0.5
    centres_y = np.arange(height) + 0.5
    distance = np.empty((height, width))
    for row in range(height):
        dx = centres_x[:, None] - boundary_x[None, :]
        dy = centres_y[row] - boundary_y[None, :]
        distance[row] = np.min(np.hypot(dx, dy), axis=1)
    return distance


def test_the_ring_is_stroked_at_the_distance_to_the_ellipse() -> None:
    """A stroke of width w covers the pixels whose centres lie within w/2
    of the ELLIPSE, which is not ``|n - 1| * min(rx, ry)`` unless the
    ellipse is a circle.

    On a 20 x 4 ring, the pixel centred 18 px along the major axis is
    1.58 px from the boundary -- its whole square clear of a 1 px stroke
    -- and that distance formula called it 0.4 and painted it 60 % red.
    The kernel's distance helper is held to a brute-force nearest-point
    search, and the painted set to the exact rule, for rings a fit can
    produce: round, wide, tall and a 8:1 needle.
    """

    pytest.importorskip("numba")
    rng = np.random.default_rng(23)
    rings = ((20.0, 4.0), (4.0, 20.0), (6.0, 4.0), (5.0, 5.0), (8.0, 1.0))
    for radius_x, radius_y in rings:
        for dx, dy in (
            (18.0, 0.0), (0.0, 0.0), (0.0, 3.0), (21.0, 1.0), (0.3, 0.2),
            *((float(px), float(py)) for px, py in
              rng.uniform(-25.0, 25.0, size=(40, 2))),
        ):
            expected = _ellipse_distance_reference(dx, dy, radius_x, radius_y)
            got = kernels.ellipse_boundary_distance(dx, dy, radius_x, radius_y)
            assert abs(got - expected) <= 1e-6 + 1e-9 * expected, (
                (radius_x, radius_y, dx, dy, got, expected)
            )

    height, width = 40, 64
    center_x, center_y = 32.5, 20.5
    for radius_x, radius_y in rings:
        for ring_width in (1.0, 3.0):
            out = np.full((height, width, 4), 255, dtype=np.uint8)
            kernels.raster_fit_ellipses(
                kernels.readable(np.asarray(
                    ((center_x, center_y, radius_x, radius_y),), dtype=np.float64
                )),
                kernels.readable(np.asarray(((255, 0, 0, 255),), dtype=np.uint8)),
                kernels.readable(np.asarray((ring_width,), dtype=np.float64)),
                kernels.readable(np.asarray(((0, 0, 0, 0),), dtype=np.uint8)),
                kernels.readable(np.asarray((0.5,), dtype=np.float64)),
                kernels.readable(np.asarray(((0, 0, width, height),), dtype=np.int32)),
                out,
            )
            painted = np.any(out[:, :, :3] != 255, axis=2)
            half = ring_width / 2.0 + 0.5
            distance = _ellipse_distance_map(
                center_x, center_y, radius_x, radius_y, height, width
            )
            decided = np.abs(distance - half) > 2e-3
            np.testing.assert_array_equal(
                painted[decided], (distance < half)[decided],
                err_msg=str((radius_x, radius_y, ring_width)),
            )
    # The finding's own pixel, on a 20 x 4 ring with a 1 px stroke.
    out = np.full((height, width, 4), 255, dtype=np.uint8)
    kernels.raster_fit_ellipses(
        kernels.readable(np.asarray(((center_x, center_y, 20.0, 4.0),), dtype=np.float64)),
        kernels.readable(np.asarray(((255, 0, 0, 255),), dtype=np.uint8)),
        kernels.readable(np.asarray((1.0,), dtype=np.float64)),
        kernels.readable(np.asarray(((0, 0, 0, 0),), dtype=np.uint8)),
        kernels.readable(np.asarray((0.5,), dtype=np.float64)),
        kernels.readable(np.asarray(((0, 0, width, height),), dtype=np.int32)),
        out,
    )
    assert tuple(out[20, 50]) == (255, 255, 255, 255)
