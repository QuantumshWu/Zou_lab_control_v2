"""Named validity expansion and compactness contracts."""

from __future__ import annotations

import numpy as np

from zlc_data.axis import AxisId, AxisSpec, REPEAT, SCAN_POINT, SPATIAL_X
from zlc_data.schema import DatasetSchema, DomainSpec, ValueSchema
from zlc_data.validity import DatasetComponentValidity, ValidityContract
from zlc_data.value import (
    DataBlock,
    DatasetRevision,
    BlockId,
    expand_dataset_validity,
    repeat_coordinate_counts,
)


def _axis(name: str, role, size: int) -> AxisSpec:
    return AxisSpec(AxisId(name), name, role, size, tuple(range(size)))


def test_dataset_component_validity_expands_over_repeat_and_point_carriers():
    repeat = _axis("repeat", REPEAT, 2)
    component = _axis("component", SPATIAL_X, 3)
    value_schema = ValueSchema(
        ValidityContract.components(component.axis_id),
        np.dtype(np.float64),
    )
    point = _axis("point", SCAN_POINT, 2)
    schema = DatasetSchema(
        DomainSpec((2,), (repeat,), ((0, 1),)),
        DomainSpec((2,), (point,), ((0, 1),)),
        DomainSpec((3,), (component,)),
        value_schema,
    )
    validity = DatasetComponentValidity(
        (component.axis_id,),
        np.array(
            [
                [[True, False, True], [False, False, True]],
                [[True, True, True], [False, True, False]],
            ]
        ),
    )
    block = DataBlock(
        BlockId("validity-block"),
        DatasetRevision(0),
        np.zeros(schema.physical_shape),
        validity,
        schema,
    )
    np.testing.assert_array_equal(expand_dataset_validity(block.validity, schema), validity.mask)


def test_a_repeat_axis_counts_the_samples_that_have_landed():
    """Count written rows at the other Repeat coordinates, not components."""
    repeat = _axis("repeat", REPEAT, 3)
    run = _axis("run", REPEAT, 2)
    # Four rows so far of a planned 3 x 2: repeat 0 whole, repeat 1 playing.
    domain = DomainSpec((4,), (repeat, run), ((0, 0, 1, 1), (0, 1, 0, 1)))
    assert repeat_coordinate_counts(domain) == (2, 2)
    assert repeat_coordinate_counts(domain, np.zeros(4, bool)) == (0, 0)
    present = np.array([True, True, True, False])
    assert repeat_coordinate_counts(domain, present) == (1, 1)
    assert repeat_coordinate_counts(domain, present, current_row=2) == (2, 1)
    assert repeat_coordinate_counts(domain, present, current_row=1) == (1, 2)
    # Duplicate carrier rows count a coordinate only once.
    duplicate = DomainSpec((6,), (repeat, run),
                           ((1, 0, 1, 0, 1, 0), (1, 0, 0, 1, 0, 0)))
    assert repeat_coordinate_counts(duplicate, np.array([False, False, True, True, True, True])) == (2, 2)


def test_a_seamless_sweep_counts_the_runs_at_the_selected_point():
    """Unplayed points report zero, even when all runs reached Point 0."""

    scan = AxisSpec(AxisId("scan"), "scan", REPEAT, 1, (0,))
    run = AxisSpec(AxisId("run"), "run", REPEAT, 3, (0, 1, 2))
    domain = DomainSpec((3,), (scan, run), ((0, 0, 0), (0, 1, 2)))
    assert repeat_coordinate_counts(domain, np.zeros(3, bool)) == (0, 0)
    assert repeat_coordinate_counts(domain, np.ones(3, bool)) == (1, 3)


def test_two_repeat_axes_of_one_name_keep_their_own_counts():
    """A count belongs to an axis, and only its id is unique.

    Keyed by name, the second "repeat" axis's count overwrote the first's,
    and a Point axis called "repeat" would have read a Repeat count where
    its size belongs.  The counts travel by position in the schema's order.
    """

    first = AxisSpec(AxisId("first"), "repeat", REPEAT, 2, (0, 1))
    second = AxisSpec(AxisId("second"), "repeat", REPEAT, 3, (0, 1, 2))
    domain = DomainSpec((6,), (first, second), ((0, 0, 0, 1, 1, 1), (0, 1, 2, 0, 1, 2)))
    assert repeat_coordinate_counts(domain) == (2, 3)
    # Rows 2 and 5 are second=2 of either first coordinate, and both
    # faulted: second=2 never landed, while every first coordinate did.
    present = np.array([True, True, False, True, True, False])
    assert repeat_coordinate_counts(domain, present) == (0, 2)
    assert repeat_coordinate_counts(domain, present, current_row=0) == (2, 2)
