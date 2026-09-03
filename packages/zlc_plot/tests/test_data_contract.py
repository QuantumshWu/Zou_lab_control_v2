from __future__ import annotations

import numpy as np
import pytest
from zlc_data import AxisId, DatasetSchema, DomainSpec, REPEAT, SCAN_POINT

from data_factory import (
    axis,
    cartesian_domain,
    make_dataset_schema,
    make_snapshot,
    mapped_domain_from_columns,
    repeat_domain,
)
from zlc_plot import AxisRef
from zlc_plot.semantics import axis_choices_for_schema, schema_structure


@pytest.fixture
def schema() -> DatasetSchema:
    repeat = repeat_domain(size=2)
    points = mapped_domain_from_columns(
        {"x": [0.0, 1.0, 2.0]},
        units={"x": "m"},
    )
    scan = axis("scan", values=[10, 20], unit="1")
    return make_dataset_schema(
        repeat,
        points,
        cell_axes=(scan,),
        dtype=np.float32,
    )


def test_schema_exposes_fixed_r_p_data_geometry(schema: DatasetSchema) -> None:
    assert schema.physical_shape == (2, 3, 2)
    assert len(schema.physical_shape) == 3
    assert schema.repeat_domain.size == 2
    assert schema.point_domain.size == 3
    assert schema.value_schema.dtype == np.dtype(np.float32)


def test_snapshot_makes_owned_readonly_arrays_and_validity(
    schema: DatasetSchema,
) -> None:
    source = np.arange(12, dtype=np.float32).reshape(schema.physical_shape)
    validity = np.ones(schema.physical_shape, dtype=np.bool_)
    snapshot = make_snapshot(schema, source, revision=7, validity=validity)

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
        lambda schema: make_snapshot(schema, np.zeros((2, 3), dtype=np.float32), 0),
        lambda schema: make_snapshot(schema, np.zeros(schema.physical_shape, dtype=np.float64), 0),
        lambda schema: make_snapshot(
            schema,
            np.zeros(schema.physical_shape, dtype=np.float32),
            0,
            validity=np.ones(
                (schema.repeat_domain.size, schema.point_domain.size - 1, schema.physical_shape[-1]),
                dtype=np.bool_,
            ),
        ),
        lambda schema: make_snapshot(schema, np.zeros(schema.physical_shape, dtype=np.float32), -1),
    ],
)
def test_snapshot_rejects_invalid_shape_dtype_validity_and_revision(
    schema: DatasetSchema,
    factory,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        factory(schema)


def test_point_domain_codes_are_explicit_and_readonly() -> None:
    bx = axis("b_x", values=[-1.0, 1.0], role=SCAN_POINT)
    by = axis("b_y", values=[0.0, 2.0], role=SCAN_POINT)
    points = cartesian_domain((bx, by))
    assert points.logical_shape == (2, 2)
    assert points.codes(AxisId("b_x")).tolist() == [0, 0, 1, 1]
    assert points.codes(AxisId("b_y")).tolist() == [0, 1, 0, 1]
    assert not points.codes(AxisId("b_x")).flags.writeable
    schema = make_dataset_schema(
        repeat_domain(size=1),
        points,
    )
    assert schema.point_domain is points
    with pytest.raises(ValueError):
        DomainSpec((1,), (bx,), ((2,),))


def test_scalar_carrier_is_not_an_authored_plot_axis() -> None:
    scalar = make_dataset_schema(
        repeat_domain(size=1),
        mapped_domain_from_columns({"x": [0.0, 1.0]}),
    )
    assert AxisRef.cell_data("zlc_data.scalar") not in axis_choices_for_schema(scalar)
    assert all(
        name != "value"
        for group in schema_structure(scalar)
        for name, _size in group
    )



def test_a_dimension_resolves_its_labels_on_a_cropped_view() -> None:
    """The labels are the domain's: cropping the rows must not lose them."""

    import numpy as np
    from zlc_data import (
        READOUT_EVENT,
        REPEAT,
        SITE,
        AxisId,
        AxisSpec,
        DatasetSchema,
        DomainSpec,
        ValidityContract,
        ValueSchema,
    )
    from zlc_data.snapshot_projection import restricted_schema
    from zlc_plot.data_contract import resolve_axis
    from zlc_plot.kinds import AxisRef

    labels = ("0-1", "0-2", "1-2")
    pair = AxisSpec(
        AxisId("pair"), "pair", READOUT_EVENT, 3, (0, 1, 2),
        coordinate_labels=labels,
    )
    site = AxisSpec(AxisId("site"), "site", SITE, 2, (0, 1))
    schema = DatasetSchema(
        DomainSpec(
            (1,),
            (AxisSpec(AxisId("repeat"), "repeat", REPEAT, 1, (0,)),),
            ((0,),),
        ),
        DomainSpec((3,), (pair,), ((0, 1, 2),)),
        DomainSpec((2,), (site,)),
        ValueSchema(
            ValidityContract.components(AxisId("site")), np.dtype("<f8"), "1"
        ),
    )
    cropped = restricted_schema(schema, range(1), (2,), {AxisId("site"): range(2)})
    resolved = resolve_axis(cropped, AxisRef.point("pair"))
    assert tuple(resolved.coordinates) == (2,)
    assert resolved.coordinate_labels == ("1-2",)
    assert resolved.name == "pair"
