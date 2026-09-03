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
