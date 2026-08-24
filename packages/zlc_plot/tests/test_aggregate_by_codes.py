"""The one grouped-reduction kernel, checked against a plain per-group loop.

``_aggregate_by_codes`` reduces every projection's groups in O(N)
sequential passes and NOTHING in it sorts: SUM/MEAN accumulate through
``bincount``, FIRST is a reversed scatter, MIN/MAX ride the ufunc's
indexed ``at`` loop.  Each pass must agree exactly with the obvious
per-group loop it replaces (a camera-sized facet reduces one group PER
PIXEL, so the loop itself once cost ~10 s per cell).
"""

from __future__ import annotations

import numpy as np
import pytest

from zlc_plot.data_view import _aggregate_by_codes
from zlc_plot.specs import Reduction


def _reference(values, usable, codes, bucket_count, reduction):
    """The obvious per-group loop the kernel must agree with."""

    reduce = {
        Reduction.MEAN: np.mean,
        Reduction.SUM: np.sum,
        Reduction.MIN: np.min,
        Reduction.MAX: np.max,
        Reduction.FIRST: lambda group: group[0],
    }[reduction]
    output = np.full(bucket_count, np.nan)
    counts = np.zeros(bucket_count, dtype=np.int64)
    for code in range(bucket_count):
        group = values[usable & (codes == code)]
        counts[code] = group.size
        if group.size:
            output[code] = reduce(group.astype(np.float64))
    return output, counts


ALL_REDUCTIONS = tuple(Reduction)


@pytest.mark.parametrize("reduction", ALL_REDUCTIONS)
def test_uneven_unsorted_groups_match_the_plain_loop(reduction) -> None:
    rng = np.random.default_rng(7)
    codes = rng.integers(-1, 6, size=200).astype(np.int64)
    values = rng.normal(size=200)
    usable = rng.random(200) > 0.2
    output, counts = _aggregate_by_codes(values, usable, codes, 6, reduction)
    expected_output, expected_counts = _reference(values, usable, codes, 6, reduction)
    np.testing.assert_array_equal(counts, expected_counts)
    np.testing.assert_allclose(output, expected_output)


@pytest.mark.parametrize("reduction", (Reduction.SUM, Reduction.MEAN))
def test_uint8_groups_do_not_wrap_at_256(reduction) -> None:
    """Camera frames arrive as native unsigned bytes; sums must leave them."""

    values = np.full(12, 200, dtype=np.uint8)
    codes = np.repeat(np.arange(3, dtype=np.int64), 4)
    usable = np.ones(12, dtype=bool)
    output, counts = _aggregate_by_codes(values, usable, codes, 3, reduction)
    expected = 800.0 if reduction is Reduction.SUM else 200.0
    np.testing.assert_array_equal(counts, [4, 4, 4])
    np.testing.assert_allclose(output, [expected] * 3)


@pytest.mark.parametrize("reduction", ALL_REDUCTIONS)
def test_single_member_groups_are_the_values_themselves(reduction) -> None:
    """The dense image case: one sample per pixel, every reduction identity."""

    rng = np.random.default_rng(3)
    order = rng.permutation(50).astype(np.int64)
    values = rng.normal(size=50)
    usable = np.ones(50, dtype=bool)
    output, counts = _aggregate_by_codes(values, usable, order, 50, reduction)
    np.testing.assert_array_equal(counts, np.ones(50, dtype=np.int64))
    np.testing.assert_allclose(output[order], values)


@pytest.mark.parametrize("reduction", ALL_REDUCTIONS)
def test_one_bucket_pools_everything_usable(reduction) -> None:
    rng = np.random.default_rng(11)
    values = rng.normal(size=64)
    usable = rng.random(64) > 0.3
    codes = np.zeros(64, dtype=np.int64)
    output, counts = _aggregate_by_codes(values, usable, codes, 1, reduction)
    expected_output, expected_counts = _reference(values, usable, codes, 1, reduction)
    np.testing.assert_array_equal(counts, expected_counts)
    np.testing.assert_allclose(output, expected_output)
