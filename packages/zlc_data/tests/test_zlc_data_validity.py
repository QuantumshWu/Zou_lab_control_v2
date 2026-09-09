"""Named validity expansion and compactness contracts."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from zlc_data.axis import AxisId, AxisSpec, REPEAT, SCAN_POINT, SPATIAL_X
from zlc_data.schema import DatasetSchema, DomainSpec, ValueSchema
from zlc_data.validity import DatasetComponentValidity, ValidityContract
from zlc_data.validity import CellValidity, INVALID, VALID
from zlc_data.value import (
    DataBlock,
    DatasetRevision,
    BlockId,
    expand_dataset_validity,
    repeat_validity_counts,
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
    """Count at the other coordinates; never pool validity across contexts."""

    repeat = _axis("repeat", REPEAT, 3)
    run = _axis("run", REPEAT, 2)
    point = _axis("point", SCAN_POINT, 2)
    site = _axis("site", SPATIAL_X, 2)
    value_schema = ValueSchema(ValidityContract.value(), np.dtype(np.float64))
    # Four rows so far of a planned 3 x 2: repeat 0 whole, repeat 1 playing.
    schema = DatasetSchema(
        DomainSpec((4,), (repeat, run), ((0, 0, 1, 1), (0, 1, 0, 1))),
        DomainSpec((2,), (point,), ((0, 1),)),
        DomainSpec((2,), (site,)),
        value_schema,
    )
    assert repeat_validity_counts(VALID, schema) == (2, 2)
    assert repeat_validity_counts(INVALID, schema) == (0, 0)
    # Row 3 is still missing Point 1. Its valid Point 0 cannot fill that hole.
    cells = CellValidity(np.array([[True, True], [True, True], [True, True], [True, False]]))
    assert repeat_validity_counts(cells, schema) == ((1, 2), (1, 2))
    assert repeat_validity_counts(cells, schema, {point.axis_id: 0}) == (2, 2)
    assert repeat_validity_counts(cells, schema, {
        repeat.axis_id: 1, run.axis_id: 1, point.axis_id: 1,
    }) == (1, 1)
    # A target ignores its own pin; an absent intersection for another target
    # is genuinely empty. No undeclared Cartesian rows are invented.
    assert repeat_validity_counts(VALID, schema, {
        repeat.axis_id: 2, run.axis_id: 0, point.axis_id: 0,
    }) == (2, 0)
    # Rows 1 and 3 are run 1 of each repeat, and both faulted: run 1 never landed.
    faulted = CellValidity(np.array([[True, True], [False, False], [True, True], [False, False]]))
    assert repeat_validity_counts(faulted, schema) == ((0, 2), 1)
    # A different site's valid value cannot stand in for an unjudged site.
    component_schema = DatasetSchema(
        schema.repeat_domain,
        schema.point_domain,
        schema.cell_domain,
        ValueSchema(ValidityContract.components(site.axis_id), np.dtype(np.float64)),
    )
    components = DatasetComponentValidity(
        (site.axis_id,),
        np.array([
            [[True, True], [True, True]],
            [[True, True], [True, True]],
            [[True, False], [True, True]],
            [[True, True], [True, True]],
        ]),
    )
    assert repeat_validity_counts(components, component_schema) == ((1, 2), (1, 2))
    assert repeat_validity_counts(components, component_schema, {
        repeat.axis_id: 1, run.axis_id: 0, point.axis_id: 0, site.axis_id: 1,
    }) == (1, 1)
    assert repeat_validity_counts(components, component_schema, {site.axis_id: 0}) == (2, 2)
    # Reordered/duplicate storage rows still count distinct Repeat coordinates.
    order = (3, 0, 2, 1, 2, 0)
    repeated_schema = replace(schema,
        repeat_domain=DomainSpec((6,), (repeat, run),
            tuple(tuple(codes[i] for i in order) for codes in schema.repeat_domain.axis_codes)),
        point_domain=DomainSpec((3,), (point,), ((1, 0, 0),)))
    repeated = cells.mask[np.asarray(order)][:, (1, 0, 0)].copy()
    repeated[:, 1] = False  # another row with the same Point coordinate is valid
    repeated[1] = False    # another row with the same Repeat coordinates is valid
    assert repeat_validity_counts(CellValidity(repeated), repeated_schema) == ((1, 2), (1, 2))
    assert repeat_validity_counts(CellValidity(repeated), repeated_schema,
                                  {point.axis_id: 0}) == (2, 2)


def test_a_seamless_sweep_counts_the_runs_at_the_selected_point():
    """Unplayed points report zero, even when all runs reached Point 0."""

    scan = AxisSpec(AxisId("scan"), "scan", REPEAT, 1, (0,))
    run = AxisSpec(AxisId("run"), "run", REPEAT, 3, (0, 1, 2))
    point = _axis("point", SCAN_POINT, 4)
    site = _axis("site", SPATIAL_X, 2)
    schema = DatasetSchema(
        DomainSpec((3,), (scan, run), ((0, 0, 0), (0, 1, 2))),
        DomainSpec((4,), (point,), ((0, 1, 2, 3),)),
        DomainSpec((2,), (site,)),
        ValueSchema(ValidityContract.value(), np.dtype(np.float64)),
    )
    assert repeat_validity_counts(INVALID, schema) == (0, 0)
    first_point = CellValidity(np.array([[True, False, False, False]] * 3))
    assert repeat_validity_counts(first_point, schema) == ((0, 1), (0, 3))
    assert repeat_validity_counts(first_point, schema, {point.axis_id: 0}) == (1, 3)
    assert repeat_validity_counts(first_point, schema, {point.axis_id: 1}) == (0, 0)


def test_two_repeat_axes_of_one_name_keep_their_own_counts():
    """A count belongs to an axis, and only its id is unique.

    Keyed by name, the second "repeat" axis's count overwrote the first's,
    and a Point axis called "repeat" would have read a Repeat count where
    its size belongs.  The counts travel by position in the schema's order.
    """

    first = AxisSpec(AxisId("first"), "repeat", REPEAT, 2, (0, 1))
    second = AxisSpec(AxisId("second"), "repeat", REPEAT, 3, (0, 1, 2))
    point = AxisSpec(AxisId("point"), "repeat", SCAN_POINT, 2, (0, 1))
    site = _axis("site", SPATIAL_X, 2)
    schema = DatasetSchema(
        DomainSpec((6,), (first, second), ((0, 0, 0, 1, 1, 1), (0, 1, 2, 0, 1, 2))),
        DomainSpec((2,), (point,), ((0, 1),)),
        DomainSpec((2,), (site,)),
        ValueSchema(ValidityContract.value(), np.dtype(np.float64)),
    )
    assert repeat_validity_counts(VALID, schema) == (2, 3)
    # Rows 2 and 5 are second=2 of either first coordinate, and both
    # faulted: second=2 never landed, while every first coordinate did.
    cells = CellValidity(
        np.array([[True, True]] * 2 + [[False, False]] + [[True, True]] * 2 + [[False, False]])
    )
    assert repeat_validity_counts(cells, schema) == ((0, 2), 2)
    assert repeat_validity_counts(cells, schema, {
        first.axis_id: 0, second.axis_id: 2, point.axis_id: 0,
    }) == (0, 2)
