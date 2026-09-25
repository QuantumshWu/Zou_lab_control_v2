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


# Last is an axis-coordinate Scope before this geometry-free numeric kernel.
ALL_REDUCTIONS = tuple(item for item in Reduction if item is not Reduction.LAST)


def _code_cases():
    """(values, usable, codes, bucket count) for every grouping shape a
    projection hands the kernel."""

    # Uneven, unsorted groups, some samples in no group at all.
    rng = np.random.default_rng(7)
    codes = rng.integers(-1, 6, size=200).astype(np.int64)
    values = rng.normal(size=200)
    yield values, rng.random(200) > 0.2, codes, 6
    # The dense image case: one sample per pixel, in scrambled order.
    rng = np.random.default_rng(3)
    order = rng.permutation(50).astype(np.int64)
    yield rng.normal(size=50), np.ones(50, dtype=bool), order, 50
    # One bucket pools everything usable.
    rng = np.random.default_rng(11)
    values = rng.normal(size=64)
    yield values, rng.random(64) > 0.3, np.zeros(64, dtype=np.int64), 1


@pytest.mark.parametrize("reduction", ALL_REDUCTIONS)
def test_every_grouping_matches_the_plain_loop(reduction) -> None:
    for values, usable, codes, bucket_count in _code_cases():
        output, counts = _aggregate_by_codes(values, usable, codes, bucket_count, reduction)
        expected_output, expected_counts = _reference(
            values, usable, codes, bucket_count, reduction
        )
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
