"""Core invariants for zlc_data's named multidimensional values."""

from __future__ import annotations

import sys
import subprocess
from fractions import Fraction

import numpy as np
import pytest

from zlc_data._arrays import immutable_array, is_intrinsically_immutable_array
from zlc_data.axis import (
    REPEAT,
    SCAN_POINT,
    SPATIAL_X,
    SPATIAL_Y,
    AxisId,
    AxisSpec,
)
from zlc_data.codec import dataset_schema_from_tree, dataset_schema_to_tree
from zlc_data.schema import (
    DatasetSchema,
    GridTopology,
    PointColumn,
    PointTable,
    ValueSchema,
)
from zlc_data.validity import (
    VALID,
    CellValidity,
    DatasetComponentValidity,
    ValidityContract,
)
from zlc_data.value import (
    BlockId,
    DataBlock,
    DatasetRevision,
    StreamGenerationId,
)


def axis(name: str, role, size: int) -> AxisSpec:
    return AxisSpec(AxisId(name), name, role, size, tuple(range(size)))


def image_schema(*, component_validity: bool = False) -> ValueSchema:
    y = axis("camera.image.y", SPATIAL_Y, 3)
    x = axis("camera.image.x", SPATIAL_X, 4)
    contract = ValidityContract.components(y.axis_id, x.axis_id) if component_validity else ValidityContract.value()
    return ValueSchema((y, x), contract, np.dtype(np.uint16), value_unit="count")


def dataset_schema(*, explicit: bool = False, component_validity: bool = False) -> DatasetSchema:
    repeat = axis("capture.repeat", REPEAT, 2)
    detuning = axis("scan.detuning", SCAN_POINT, 3)
    point_values = (2, 0) if explicit else (0, 1, 2)
    point_table = PointTable(
        len(point_values),
        (
            PointColumn(
                detuning.axis_id,
                detuning.name,
                detuning.role,
                PointColumn.NUMERIC,
                point_values,
            ),
        ),
    )
    topology = GridTopology(
        (detuning.axis_id,),
        ((0, 1, 2),),
        tuple((value,) for value in point_values),
    )
    return DatasetSchema(
        repeat,
        point_table,
        topology,
        image_schema(component_validity=component_validity),
    )


def test_scalar_has_the_canonical_length_one_carrier_axis():
    scalar_schema = ValueSchema.scalar(np.dtype(np.float64), "count")
    assert scalar_schema.is_scalar
    assert scalar_schema.data_shape == (1,)


def test_intrinsically_immutable_strided_views_cross_value_and_dataset_without_copy():
    mutable = np.arange(12, dtype=np.uint16).reshape(3, 4)
    frozen = immutable_array(
        mutable,
        dtype=np.dtype("<u2"),
        shape=mutable.shape,
    )
    transposed = frozen.T
    y = axis("camera.transposed.y", SPATIAL_Y, 4)
    x = axis("camera.transposed.x", SPATIAL_X, 3)
    value_schema = ValueSchema(
        (y, x),
        ValidityContract.value(),
        np.dtype("<u2"),
        "count",
    )
    repeat = axis("camera.transposed.repeat", REPEAT, 1)
    schema = DatasetSchema(repeat, PointTable(1), None, value_schema)
    block = DataBlock(
        BlockId("immutable-transpose"),
        DatasetRevision(0),
        transposed.reshape(schema.physical_shape),
        VALID,
        schema,
    )
    assert np.shares_memory(block.values, transposed)
    assert is_intrinsically_immutable_array(block.values)

    mutable[:] = 0
    assert np.any(block.values != 0)
    with pytest.raises(ValueError):
        block.values.setflags(write=True)


def test_dataset_rejects_duplicate_axis_identity_across_axis_families():
    repeat = axis("same", REPEAT, 1)
    point = axis("same", SCAN_POINT, 1)
    point_table = PointTable(
        1,
        (
            PointColumn(
                point.axis_id,
                point.name,
                point.role,
                PointColumn.NUMERIC,
                (0,),
            ),
        ),
    )
    with pytest.raises(ValueError, match="unique"):
        DatasetSchema(repeat, point_table, None, image_schema())


def test_dataset_component_validity_includes_repeat_and_physical_point_rows():
    schema = dataset_schema(component_validity=True)
    site_like_x = schema.cell_schema.data_axes[1]
    validity = DatasetComponentValidity(
        (site_like_x.axis_id,),
        np.ones((2, 3, 4), dtype=bool),
    )
    block = DataBlock(
        BlockId("capture-1"),
        DatasetRevision(0),
        np.zeros(schema.physical_shape, dtype=np.uint16),
        validity,
        schema,
    )
    assert block.validity.mask.shape == (2, 3, 4)


def test_datablock_owns_intrinsically_immutable_bytes():
    schema = dataset_schema()
    source = np.arange(np.prod(schema.physical_shape), dtype=np.uint16).reshape(
        schema.physical_shape
    )[..., ::-1]
    assert not source.flags.c_contiguous
    expected = source.copy()
    block = DataBlock(
        BlockId("capture-immutable"),
        DatasetRevision(7),
        source,
        CellValidity(np.ones((2, 3), dtype=bool)),
        schema,
    )
    np.testing.assert_array_equal(block.values, expected)
    before = block.values.copy()
    source[...] = 0
    np.testing.assert_array_equal(block.values, before)
    with pytest.raises(ValueError):
        block.values.setflags(write=True)
    with pytest.raises(ValueError):
        block.validity.mask.setflags(write=True)


def test_dataset_ref_carries_complete_identity():
    schema = dataset_schema(explicit=True)
    restored = dataset_schema_from_tree(dataset_schema_to_tree(schema))
    assert restored.point_table == schema.point_table
    assert restored.grid_topology == schema.grid_topology
    block = DataBlock(
        BlockId("sparse-capture"),
        DatasetRevision(3),
        np.zeros(schema.physical_shape, dtype=np.uint16),
        VALID,
        schema,
    )
    ref = block.ref(StreamGenerationId("camera-generation-8"))
    assert ref.block_id == block.block_id
    assert ref.revision == block.revision
    assert ref.schema_fingerprint == schema.fingerprint


def test_grid_topology_can_own_dimensions_without_point_columns():
    repeat = axis("capture.repeat", REPEAT, 1)
    topology = GridTopology(
        (AxisId("b.x"), AxisId("b.y")),
        ((0, 1), (10, 20)),
        ((0, 0), (1, 0), (0, 1), (1, 1)),
    )
    schema = DatasetSchema(
        repeat,
        PointTable(4),
        topology,
        image_schema(),
    )

    restored = dataset_schema_from_tree(dataset_schema_to_tree(schema))

    assert schema.point_table.columns == ()
    assert restored == schema
    assert restored.grid_topology == topology


def test_grid_topology_checks_a_matching_point_column_but_allows_absent_ones():
    repeat = axis("capture.repeat", REPEAT, 1)
    b_x = AxisId("b.x")
    b_y = AxisId("b.y")
    topology = GridTopology(
        (b_x, b_y),
        ((0, 1), (10, 20)),
        ((0, 0), (1, 0), (0, 1), (1, 1)),
    )
    consistent = PointTable(
        4,
        (
            PointColumn(
                b_x,
                "b.x",
                SCAN_POINT,
                PointColumn.NUMERIC,
                (0, 1, 0, 1),
            ),
        ),
    )
    DatasetSchema(repeat, consistent, topology, image_schema())

    inconsistent = PointTable(
        4,
        (
            PointColumn(
                b_x,
                "b.x",
                SCAN_POINT,
                PointColumn.NUMERIC,
                (0, 99, 0, 1),
            ),
        ),
    )
    with pytest.raises(ValueError, match="match their PointTable columns"):
        DatasetSchema(repeat, inconsistent, topology, image_schema())


def test_dataset_schema_tree_matches_the_independent_current_grammar():
    schema = dataset_schema()
    literal = {
        "schema": "zlc_data.DatasetSchema",
        "repeat_axis": {
            "schema": "zlc_data.AxisSpec",
            "axis_id": "capture.repeat",
            "name": "capture.repeat",
            "role": "repeat",
            "size": 2,
            "coordinates": [0, 1],
                "unit": None,
                "coordinate_frame": None,
                "index_origin": 0,
                "coordinate_labels": None,
        },
        "point_table": {
            "schema": "zlc_data.PointTable",
            "row_count": 3,
            "columns": [
                {
                    "schema": "zlc_data.PointColumn",
                    "coordinate_id": "scan.detuning",
                    "name": "scan.detuning",
                    "role": "scan-point",
                    "value_kind": "NUMERIC",
                    "values": [0, 1, 2],
                        "unit": None,
                        "coordinate_frame": None,
                        "coordinate_labels": None,
                }
            ],
        },
        "grid_topology": {
            "schema": "zlc_data.GridTopology",
            "dimension_ids": ["scan.detuning"],
            "coordinate_domains": [[0, 1, 2]],
            "row_to_cell": [[0], [1], [2]],
        },
        "cell_schema": {
            "schema": "zlc_data.ValueSchema",
            "data_axes": [
                {
                    "schema": "zlc_data.AxisSpec",
                    "axis_id": "camera.image.y",
                    "name": "camera.image.y",
                    "role": "spatial-y",
                    "size": 3,
                    "coordinates": [0, 1, 2],
                        "unit": None,
                        "coordinate_frame": None,
                        "index_origin": 0,
                        "coordinate_labels": None,
                },
                {
                    "schema": "zlc_data.AxisSpec",
                    "axis_id": "camera.image.x",
                    "name": "camera.image.x",
                    "role": "spatial-x",
                    "size": 4,
                    "coordinates": [0, 1, 2, 3],
                        "unit": None,
                        "coordinate_frame": None,
                        "index_origin": 0,
                        "coordinate_labels": None,
                },
            ],
            "validity_contract": {"mode": "VALUE", "component_axis_ids": []},
            "dtype": "<u2",
            "value_unit": "count",
        },
    }

    assert dataset_schema_to_tree(schema) == literal
    restored = dataset_schema_from_tree(literal)
    assert restored == schema
    assert restored.fingerprint == schema.fingerprint

    malformed = dict(literal, revision=1)
    with pytest.raises(ValueError, match="exactly"):
        dataset_schema_from_tree(malformed)


def test_schema_fingerprint_covers_point_rows_topology_and_component_validity():
    sparse = dataset_schema(explicit=True, component_validity=True)
    dense = dataset_schema(explicit=False, component_validity=True)
    value_only = dataset_schema(explicit=True, component_validity=False)
    assert dense.fingerprint != sparse.fingerprint
    assert value_only.fingerprint != sparse.fingerprint


def test_schema_fingerprint_normalizes_dtype_endianness():
    little = ValueSchema.scalar(np.dtype("<i2"))
    big = ValueSchema.scalar(np.dtype(">i2"))
    assert little.fingerprint == big.fingerprint


def test_immutable_schema_fingerprints_are_computed_once(monkeypatch):
    schema = dataset_schema()
    dataset_fingerprint = schema.fingerprint
    value_fingerprint = schema.cell_schema.fingerprint
    import zlc_data.codec as codec

    def forbidden(*_args, **_kwargs):
        raise AssertionError("immutable schema fingerprint was recomputed")

    monkeypatch.setattr(codec, "dataset_schema_fingerprint", forbidden)
    monkeypatch.setattr(codec, "value_schema_fingerprint", forbidden)
    assert schema.fingerprint == dataset_fingerprint
    assert schema.cell_schema.fingerprint == value_fingerprint


def test_value_schema_rejects_non_numeric_payload_dtypes():
    with pytest.raises(TypeError, match="numeric"):
        ValueSchema.scalar(np.dtype("U4"))


def test_axis_coordinates_reject_nonfinite_values():
    with pytest.raises(ValueError, match="finite"):
        AxisSpec(AxisId("bad"), "bad", SCAN_POINT, 1, (float("nan"),))


def test_numeric_coordinates_have_one_python_and_fingerprint_identity():
    negative_zero = AxisSpec(
        AxisId("scan.coordinate"),
        "coordinate",
        SCAN_POINT,
        2,
        (-0.0, np.float64(1.0)),
    )
    integers = AxisSpec(
        AxisId("scan.coordinate"),
        "coordinate",
        SCAN_POINT,
        2,
        (0, 1),
    )
    assert negative_zero == integers
    assert negative_zero.coordinates == (0, 1)
    assert all(type(value) is int for value in negative_zero.coordinates)

    repeat = axis("repeat", REPEAT, 1)
    left_column = PointColumn(
        negative_zero.axis_id,
        negative_zero.name,
        negative_zero.role,
        PointColumn.NUMERIC,
        negative_zero.coordinates,
    )
    right_column = PointColumn(
        integers.axis_id,
        integers.name,
        integers.role,
        PointColumn.NUMERIC,
        integers.coordinates,
    )
    left = DatasetSchema(repeat, PointTable(2, (left_column,)), None, image_schema())
    right = DatasetSchema(repeat, PointTable(2, (right_column,)), None, image_schema())
    assert left == right
    assert left.fingerprint == right.fingerprint

    with pytest.raises(TypeError, match="finite number"):
        AxisSpec(AxisId("bool"), "bool", SCAN_POINT, 1, (True,))
    fractional = AxisSpec(
        AxisId("fraction"), "fraction", SCAN_POINT, 1, (Fraction(1, 2),)
    )
    assert fractional.coordinates == (0.5,)


def test_repeat_role_has_exactly_one_structural_owner():
    repeat = axis("repeat", REPEAT, 1)
    with pytest.raises(ValueError, match="point-domain"):
        PointColumn(
            AxisId("counterfeit.point"),
            "counterfeit.point",
            REPEAT,
            PointColumn.NUMERIC,
            (0, 1),
        )

    counterfeit_data = axis("counterfeit.data", REPEAT, 2)
    with pytest.raises(ValueError, match="only"):
        DatasetSchema(
            repeat,
            PointTable(1),
            None,
            ValueSchema(
                (counterfeit_data,),
                ValidityContract.value(),
                np.dtype(np.float64),
            ),
        )


def test_import_is_headless_and_does_not_pull_legacy_domain():
    code = """
import sys
import zlc_data
for forbidden in ('matplotlib', 'PyQt5', 'Zou_lab_control'):
    assert forbidden not in sys.modules, (forbidden, sorted(sys.modules))
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_a_schema_is_not_named_until_someone_asks() -> None:
    """A name costs 23 us and most schemas are never named.

    Derived operations build intermediate schemas, and most are compared as
    objects without ever asking for a persisted fingerprint.
    """

    import zlc_data.codec as codec

    calls = []
    real = codec.dataset_schema_fingerprint

    def counted(schema):
        calls.append(schema)
        return real(schema)

    codec.dataset_schema_fingerprint = counted
    try:
        schema = dataset_schema()
        assert calls == [], "a schema was named before anyone asked"
        first = schema.fingerprint
        assert len(calls) == 1
        assert schema.fingerprint == first
        assert len(calls) == 1, "the name was computed twice"
    finally:
        codec.dataset_schema_fingerprint = real
