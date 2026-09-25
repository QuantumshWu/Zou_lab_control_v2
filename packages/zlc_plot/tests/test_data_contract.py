from __future__ import annotations

import numpy as np
import pytest
from zlc_data import DatasetSchema

from data_factory import (
    axis,
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


def test_structure_keeps_repeat_point_and_cell_brackets() -> None:
    """Three brackets: (repeat) x (points) x (data).

    Pair and site are both dimensions of one atomic cell payload, so they
    share the third bracket.  A READOUT_EVENT cell axis is instead a fact
    about WHEN within one point, so it joins the points bracket after the
    scan dimensions: (20) x (10x10x10x3) x (34).
    """

    from zlc_data import (
        COMPONENT,
        DomainSpec,
        READOUT_EVENT,
        SITE,
        AxisId,
        AxisSpec,
        DatasetSchema as Schema,
        REPEAT,
        SCALAR_DOMAIN,
        SPATIAL_X,
        SPATIAL_Y,
        ValidityContract,
        ValueSchema,
    )
    from zlc_plot.semantics import schema_structure

    def _schema_for(axes):
        return Schema(
            DomainSpec(
                (1,),
                (AxisSpec(AxisId("cycle"), "cycle", REPEAT, 1),),
                ((0,),),
            ),
            DomainSpec((1,), (), ()),
            DomainSpec(tuple(axis.size for axis in axes), axes),
            ValueSchema(
                ValidityContract.components(axes[0].axis_id),
                np.dtype("<f8"),
                "1",
            ),
        )

    categorical = _schema_for(
        (
            AxisSpec(AxisId("fs.pair"), "pair", COMPONENT, 3),
            AxisSpec(AxisId("occ.site"), "site", SITE, 33),
        )
    )
    groups = schema_structure(categorical)
    assert tuple(tuple(name for name, _size in group) for group in groups) == (
        ("cycle",),
        (),
        ("pair", "site"),
    )

    scanned = Schema(
        DomainSpec(
            (20,),
            (AxisSpec(AxisId("cycle"), "cycle", REPEAT, 20),),
            (tuple(range(20)),),
        ),
        DomainSpec(
            (8,),
            tuple(
                AxisSpec(AxisId(name), name, COMPONENT, 2, (0.0, 1.0))
                for name in ("ax", "ay", "az")
            ),
            tuple(
                tuple(cell[position] for cell in tuple(
                    (i % 2, (i // 2) % 2, i // 4) for i in range(8)
                ))
                for position in range(3)
            ),
        ),
        DomainSpec(
            (3, 34),
            (
                AxisSpec(AxisId("cm.frame"), "frame", READOUT_EVENT, 3),
                AxisSpec(AxisId("occ.site"), "site", SITE, 34),
            ),
        ),
        ValueSchema(
            ValidityContract.components(AxisId("occ.site")),
            np.dtype("<f8"),
            "1",
        ),
    )
    groups = schema_structure(scanned)
    assert tuple(tuple(name for name, _size in group) for group in groups) == (
        ("cycle",),
        ("ax", "ay", "az"),
        ("frame", "site"),
    )

    picture = _schema_for(
        (
            AxisSpec(AxisId("cam.y"), "y", SPATIAL_Y, 4),
            AxisSpec(AxisId("cam.x"), "x", SPATIAL_X, 5),
        )
    )
    groups = schema_structure(picture)
    assert tuple(tuple(name for name, _size in group) for group in groups) == (
        ("cycle",),
        (),
        ("y", "x"),
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
