"""Core invariants for zlc_data's named multidimensional values."""

from __future__ import annotations

import os
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

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
    DomainSpec,
    SCALAR_DOMAIN,
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


def domain(axis: AxisSpec) -> DomainSpec:
    return DomainSpec(
        (axis.size,), (axis,), (tuple(range(axis.size)),)
    )


def image_schema(
    *, component_validity: bool = False
) -> tuple[DomainSpec, ValueSchema]:
    y = axis("camera.image.y", SPATIAL_Y, 3)
    x = axis("camera.image.x", SPATIAL_X, 4)
    contract = ValidityContract.components(y.axis_id, x.axis_id) if component_validity else ValidityContract.value()
    return (
        DomainSpec((y.size, x.size), (y, x)),
        ValueSchema(contract, np.dtype(np.uint16), value_unit="count"),
    )


def dataset_schema(*, explicit: bool = False, component_validity: bool = False) -> DatasetSchema:
    repeat = axis("capture.repeat", REPEAT, 2)
    detuning = axis("scan.detuning", SCAN_POINT, 3)
    point_values = (2, 0) if explicit else (0, 1, 2)
    return DatasetSchema(
        domain(repeat),
        DomainSpec((len(point_values),), (detuning,), (point_values,)),
        *image_schema(component_validity=component_validity),
    )


def test_scalar_has_the_canonical_length_one_carrier_axis():
    scalar_schema = ValueSchema.scalar(np.dtype(np.float64), "count")
    assert scalar_schema.validity_contract == ValidityContract.value()
    assert SCALAR_DOMAIN.shape == (1,)


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
    cell_domain = DomainSpec((y.size, x.size), (y, x))
    value_schema = ValueSchema(ValidityContract.value(), np.dtype("<u2"), "count")
    repeat = axis("camera.transposed.repeat", REPEAT, 1)
    schema = DatasetSchema(
        domain(repeat),
        DomainSpec((1,), (), ()),
        cell_domain,
        value_schema,
    )
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
    with pytest.raises(ValueError, match="unique"):
        DatasetSchema(
            domain(repeat),
            DomainSpec((1,), (point,), ((0,),)),
            *image_schema(),
        )


def test_dataset_component_validity_includes_repeat_and_physical_point_rows():
    schema = dataset_schema(component_validity=True)
    site_like_x = schema.cell_domain.axes[1]
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
    assert restored.repeat_domain == schema.repeat_domain
    assert restored.point_domain == schema.point_domain
    for codes in (
        restored.repeat_domain.codes(restored.repeat_domain.axes[0].axis_id),
        restored.point_domain.codes(restored.point_domain.axes[0].axis_id),
        restored.cell_domain.codes(restored.cell_domain.axes[0].axis_id),
    ):
        assert is_intrinsically_immutable_array(codes)
        with pytest.raises(ValueError):
            codes.setflags(write=True)
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


def test_point_domain_owns_multiple_axes_and_their_only_row_mapping():
    repeat = axis("capture.repeat", REPEAT, 1)
    b_x = AxisSpec(AxisId("b.x"), "b.x", SCAN_POINT, 2, (0, 1))
    b_y = AxisSpec(AxisId("b.y"), "b.y", SCAN_POINT, 2, (10, 20))
    point_domain = DomainSpec(
        (4,),
        (b_x, b_y),
        ((0, 1, 0, 1), (0, 0, 1, 1)),
    )
    schema = DatasetSchema(
        domain(repeat),
        point_domain,
        *image_schema(),
    )

    restored = dataset_schema_from_tree(dataset_schema_to_tree(schema))

    assert restored == schema
    np.testing.assert_array_equal(restored.point_domain.codes(b_x.axis_id), (0, 1, 0, 1))
    assert not restored.point_domain.codes(b_x.axis_id).flags.writeable
    assert restored.point_domain.physical_dimension(b_x.axis_id) == 0
    y_axis, x_axis = restored.cell_domain.axes
    assert restored.cell_domain.physical_dimension(y_axis.axis_id) == 0
    assert restored.cell_domain.physical_dimension(x_axis.axis_id) == 1
    assert restored.cell_domain.codes(x_axis.axis_id).shape == (x_axis.size,)
    assert not restored.cell_domain.codes(x_axis.axis_id).flags.writeable


def test_repeat_domain_can_name_multiple_axes_over_one_physical_carrier():
    repeat = axis("capture.repeat", REPEAT, 2)
    shot = axis("capture.shot-per-point", REPEAT, 3)
    schema = DatasetSchema(
        DomainSpec(
            (6,),
            (repeat, shot),
            ((0, 0, 0, 1, 1, 1), (0, 1, 2, 0, 1, 2)),
        ),
        DomainSpec((1,), (), ()),
        *image_schema(),
    )

    assert schema.physical_shape[:2] == (6, 1)
    np.testing.assert_array_equal(
        schema.repeat_domain.codes(shot.axis_id), (0, 1, 2, 0, 1, 2)
    )


def test_domain_rejects_ambiguous_axis_coordinates_and_out_of_range_codes():
    repeat = axis("capture.repeat", REPEAT, 1)
    b_x = AxisSpec(AxisId("b.x"), "b.x", SCAN_POINT, 2, (0, 1))
    repeated = DomainSpec((2,), (b_x,), ((0, 0),))
    assert repeated.axis_codes == ((0, 0),)
    with pytest.raises(ValueError, match="outside"):
        DomainSpec((2,), (b_x,), ((0, 2),))
    ambiguous = AxisSpec(AxisId("ambiguous"), "ambiguous", SCAN_POINT, 2, (1, 1))
    with pytest.raises(ValueError, match="coordinates must be unique"):
        DomainSpec((2,), (ambiguous,), ((0, 1),))
    with pytest.raises(ValueError, match="named axis"):
        DatasetSchema(
            domain(repeat),
            DomainSpec((2,), (), ()),
            *image_schema(),
        )


def test_dataset_schema_tree_matches_the_independent_current_grammar():
    schema = dataset_schema()
    literal = {
        "schema": "zlc_data.DatasetSchema",
        "repeat_domain": {
            "schema": "zlc_data.DomainSpec",
            "shape": [2],
            "axes": [
                {
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
                }
            ],
            "axis_codes": [[0, 1]],
        },
        "point_domain": {
            "schema": "zlc_data.DomainSpec",
            "shape": [3],
            "axes": [
                {
                    "schema": "zlc_data.AxisSpec",
                    "axis_id": "scan.detuning",
                    "name": "scan.detuning",
                    "role": "scan-point",
                    "size": 3,
                    "coordinates": [0, 1, 2],
                    "unit": None,
                    "coordinate_frame": None,
                    "index_origin": 0,
                    "coordinate_labels": None,
                }
            ],
            "axis_codes": [[0, 1, 2]],
        },
        "cell_domain": {
            "schema": "zlc_data.DomainSpec",
            "shape": [3, 4],
            "axes": [
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
            "axis_codes": None,
        },
        "value_schema": {
            "schema": "zlc_data.ValueSchema",
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


def test_schema_fingerprint_covers_index_codes_and_component_validity():
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
    value_fingerprint = schema.value_schema.fingerprint
    import zlc_data.codec as codec

    def forbidden(*_args, **_kwargs):
        raise AssertionError("immutable schema fingerprint was recomputed")

    monkeypatch.setattr(codec, "dataset_schema_fingerprint", forbidden)
    monkeypatch.setattr(codec, "value_schema_fingerprint", forbidden)
    assert schema.fingerprint == dataset_fingerprint
    assert schema.value_schema.fingerprint == value_fingerprint


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
    left = DatasetSchema(
        domain(repeat),
        DomainSpec((2,), (negative_zero,), ((0, 1),)),
        *image_schema(),
    )
    right = DatasetSchema(
        domain(repeat),
        DomainSpec((2,), (integers,), ((0, 1),)),
        *image_schema(),
    )
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
    counterfeit_point = axis("counterfeit.point", REPEAT, 2)
    with pytest.raises(ValueError, match="Repeat domain"):
        DatasetSchema(
            domain(repeat),
            domain(counterfeit_point),
            *image_schema(),
        )

    counterfeit_data = axis("counterfeit.data", REPEAT, 2)
    with pytest.raises(ValueError, match="only"):
        DatasetSchema(
            domain(repeat),
            DomainSpec((1,), (), ()),
            DomainSpec((2,), (counterfeit_data,)),
            ValueSchema(
                ValidityContract.value(),
                np.dtype(np.float64),
            ),
        )


def test_import_is_headless():
    import tempfile
    import zou_lab_control
    import zlc_data

    repo_root = Path(__file__).resolve().parents[3]
    environment = dict(os.environ)
    if environment.get("ZLC_TEST_INSTALLED") == "1":
        environment["PYTHONPATH"] = ""
    else:
        environment["PYTHONPATH"] = os.pathsep.join(
            (
                str(repo_root),
                str(repo_root / "packages" / "zlc_data" / "src"),
            )
        )
    code = """
from pathlib import Path
import sys

import zou_lab_control
import zlc_data

root_file = Path(zou_lab_control.__file__).resolve()
data_file = Path(zlc_data.__file__).resolve()
print("root", root_file)
print("zlc_data", data_file)
assert root_file == Path(sys.argv[1]).resolve()
assert data_file == Path(sys.argv[2]).resolve()
for forbidden in ('matplotlib', 'PyQt5'):
    assert forbidden not in sys.modules, (forbidden, sorted(sys.modules))
"""
    with tempfile.TemporaryDirectory() as folder:
        subprocess.run(
            [
                sys.executable,
                "-c",
                code,
                str(Path(zou_lab_control.__file__).resolve()),
                str(Path(zlc_data.__file__).resolve()),
            ],
            check=True,
            cwd=folder,
            env=environment,
        )


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
