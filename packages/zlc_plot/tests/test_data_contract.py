from __future__ import annotations

import numpy as np
import pytest
from zlc_data import AxisId, GridTopology

from data_factory import Axis, DatasetSchema, DatasetSnapshot, PointTable, PointTopology


@pytest.fixture
def schema() -> DatasetSchema:
    repeat = Axis.create("repeat", size=2)
    points = PointTable.from_columns(
        {"x": [0.0, 1.0, 2.0]},
        units={"x": "m"},
    )
    scan = Axis.create("scan", values=[10, 20], canonical_unit="1")
    return DatasetSchema.create(
        repeat,
        points,
        data_axes=(scan,),
        dtype=np.float32,
        metadata={"producer": {"name": "test"}, "tags": ["a", "b"]},
    )


def test_schema_exposes_fixed_r_p_data_geometry(schema: DatasetSchema) -> None:
    assert schema.shape == (2, 3, 2)
    assert schema.ndim == 3
    assert schema.R == 2
    assert schema.P == 3
    assert schema.cell_schema.dtype == np.dtype(np.float32)


def test_snapshot_makes_owned_readonly_arrays_and_validity(
    schema: DatasetSchema,
) -> None:
    source = np.arange(12, dtype=np.float32).reshape(schema.shape)
    validity = np.ones(schema.shape, dtype=np.bool_)
    snapshot = DatasetSnapshot(schema, source, revision=7, validity=validity)

    source[0, 0, 0] = -100
    validity[0, 0, 0] = False
    values = snapshot.block.values
    dense_validity = snapshot.expanded_validity()

    assert values.flags.writeable is False
    assert dense_validity.flags.writeable is False
    assert values[0, 0, 0] != -100
    assert bool(dense_validity[0, 0, 0])
    with pytest.raises((TypeError, ValueError)):
        values[0, 0, 0] = 0


@pytest.mark.parametrize(
    "factory",
    [
        lambda schema: DatasetSnapshot(schema, np.zeros((2, 3), dtype=np.float32), 0),
        lambda schema: DatasetSnapshot(schema, np.zeros(schema.shape, dtype=np.float64), 0),
        lambda schema: DatasetSnapshot(schema, np.zeros(schema.shape, dtype=np.float32), 0, validity=np.ones((schema.R, schema.P - 1, schema.shape[-1]), dtype=np.bool_)),
        lambda schema: DatasetSnapshot(schema, np.zeros(schema.shape, dtype=np.float32), -1),
    ],
)
def test_snapshot_rejects_invalid_shape_dtype_validity_and_revision(
    schema: DatasetSchema,
    factory,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        factory(schema)


def test_point_topology_is_explicit_and_readonly() -> None:
    bx = Axis.create("b_x", values=[-1.0, 1.0])
    by = Axis.create("b_y", values=[0.0, 2.0])
    table = PointTable.from_columns({"signal": [10.0, 20.0, 30.0, 40.0]})
    topology = PointTopology.from_cartesian((bx, by))
    assert topology.shape == (2, 2)
    assert topology.logical_shape == (2, 2)
    assert topology.row_to_cell == ((0, 0), (0, 1), (1, 0), (1, 1))
    # W-round topology dimensions are allowed to be absent from PointTable.
    schema = DatasetSchema.create(
        Axis.create("repeat", size=1),
        table,
        point_topology=topology,
    )
    assert schema.grid_topology is topology
    with pytest.raises(ValueError):
        GridTopology((AxisId("bx"),), ((0.0,),), ((2,),))
