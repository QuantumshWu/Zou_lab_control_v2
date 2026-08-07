"""Named validity expansion and compactness contracts."""

from __future__ import annotations

import numpy as np
import pytest

from zlc_data.axis import AxisId, AxisSpec, REPEAT, SITE, SPATIAL_X
from zlc_data.schema import DatasetSchema, PointTable, ValueSchema
from zlc_data.validity import (
    VALID,
    CellValidity,
    ComponentValidity,
    DatasetComponentValidity,
    ValidityContract,
)
from zlc_data.value import (
    DataBlock,
    DatasetRevision,
    BlockId,
    expand_component_validity,
    expand_dataset_validity,
    expand_value_validity,
)


def _axis(name: str, role, size: int) -> AxisSpec:
    return AxisSpec(AxisId(name), name, role, size, tuple(range(size)))


def test_expand_component_validity_uses_declared_axis_identity():
    site = _axis("site", SITE, 2)
    component = _axis("component", SPATIAL_X, 2)
    schema = ValueSchema(
        (site, component),
        ValidityContract.components(site.axis_id, component.axis_id),
        np.dtype(np.float64),
    )
    site_mask = ComponentValidity((site.axis_id,), np.array([True, False]))
    component_mask = ComponentValidity((component.axis_id,), np.array([True, False]))
    np.testing.assert_array_equal(
        expand_component_validity(site_mask, schema),
        [[True, True], [False, False]],
    )
    np.testing.assert_array_equal(
        expand_component_validity(component_mask, schema),
        [[True, False], [True, False]],
    )
    with pytest.raises(ValueError, match="COMPONENTS"):
        expand_component_validity(VALID, ValueSchema.scalar(np.dtype(np.float64)))


def test_dataset_component_validity_expands_over_repeat_and_point_carriers():
    repeat = _axis("repeat", REPEAT, 2)
    component = _axis("component", SPATIAL_X, 3)
    cell_schema = ValueSchema(
        (component,),
        ValidityContract.components(component.axis_id),
        np.dtype(np.float64),
    )
    schema = DatasetSchema(repeat, PointTable(2), None, cell_schema)
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
    np.testing.assert_array_equal(
        expand_value_validity(
            ComponentValidity((component.axis_id,), np.array([True, False, True])),
            cell_schema,
        ),
        [True, False, True],
    )
