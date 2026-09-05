"""Named validity expansion and compactness contracts."""

from __future__ import annotations

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


def test_a_repeat_axis_counts_only_the_samples_that_landed_whole():
    """34\u2713 beside a repeat axis is the number of COMPLETE repeats.

    Two repeat axes over one carrier: ``repeat`` (which play of the table)
    and ``run`` (which run within it).  A row is a sample; a coordinate of
    either axis is whole when every row it owns is valid at every point.
    The repeat still being played, and any run of it that faulted, are not
    counted -- and a coordinate the carrier does not hold yet never is.
    """

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
    assert repeat_validity_counts(VALID, schema) == {"repeat": 2, "run": 2}
    assert repeat_validity_counts(INVALID, schema) == {"repeat": 0, "run": 0}
    # Row 3 (repeat 1, run 1) has one point still to land.
    cells = CellValidity(np.array([[True, True], [True, True], [True, True], [True, False]]))
    assert repeat_validity_counts(cells, schema) == {"repeat": 1, "run": 1}
    # A component mask reads the same way: any invalid site breaks its row.
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
    assert repeat_validity_counts(components, component_schema) == {"repeat": 1, "run": 1}
