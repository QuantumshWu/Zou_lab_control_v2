"""Public Dataset snapshot projection contracts."""

from __future__ import annotations

from zlc_data.validation import DIGEST_HEX

import numpy as np
import pytest

from zlc_data.axis import (
    AxisId,
    AxisSpec,
    CoordinateFrameId,
    PRIMARY_INDEX,
    REPEAT,
    SITE,
    SPATIAL_X,
)
from zlc_data.schema import DatasetSchema, PointColumn, PointTable, ValueSchema
from zlc_data.selection import IndexRangeSelection, Selection, take_indices
from zlc_data.snapshot_projection import (
    PRIMARY_INDEX_AXIS_ID,
    indexed_schemas_compatible,
    materialize_derived_dataset,
    restrict_snapshot,
)
from zlc_data.validity import VALID, ValidityContract
from zlc_data.value import (
    BlockId,
    DatasetRevision,
    DatasetRevisionRef,
    StreamGenerationId,
    owned_snapshot_from_arrays,
)


def _source_ref() -> DatasetRevisionRef:
    return DatasetRevisionRef(
        BlockId("source-block"),
        StreamGenerationId("source-generation"),
        "a" * DIGEST_HEX,
        DatasetRevision(7),
    )


def _reference_for(block_id: str = "derived-block"):
    def reference(schema: DatasetSchema) -> DatasetRevisionRef:
        return DatasetRevisionRef(
            BlockId(block_id),
            StreamGenerationId("source-generation"),
            schema.fingerprint,
            DatasetRevision(7),
        )
    return reference


def _scalar_schema() -> DatasetSchema:
    repeat = AxisSpec(AxisId("repeat"), "repeat", REPEAT, 1, (0,))
    return DatasetSchema(
        repeat,
        PointTable(1),
        None,
        ValueSchema.scalar(np.dtype("<f4"), "count"),
    )


def test_materialize_derived_dataset_preserves_schema_and_source_revision():
    schema = _scalar_schema()
    snapshot = materialize_derived_dataset(
        _source_ref(),
        np.asarray([[[2.5]]], dtype="<f4"),
        schema=schema,
        validity=VALID,
        reference_for=_reference_for(),
    )

    assert snapshot.block.schema == schema
    assert snapshot.ref.revision == DatasetRevision(7)
    assert snapshot.ref.block_id == BlockId("derived-block")
    np.testing.assert_array_equal(snapshot.block.values, [[[2.5]]])


def test_materialize_derived_dataset_rejects_bad_schema_and_reused_reference():
    with pytest.raises(TypeError, match="schema"):
        materialize_derived_dataset(
            _source_ref(),
            [[[1.0]]],
            schema=object(),  # type: ignore[arg-type]
            validity=VALID,
            reference_for=_reference_for(),
        )
    with pytest.raises(ValueError, match="reuse"):
        materialize_derived_dataset(
            _source_ref(),
            [[[1.0]]],
            schema=_scalar_schema(),
            validity=VALID,
            reference_for=_reference_for("source-block"),
        )
    with pytest.raises(ValueError, match="revision"):
        materialize_derived_dataset(
            _source_ref(),
            [[[1.0]]],
            schema=_scalar_schema(),
            validity=VALID,
            reference_for=lambda schema: DatasetRevisionRef(
                BlockId("other-block"),
                StreamGenerationId("source-generation"),
                schema.fingerprint,
                DatasetRevision(8),
            ),
        )


def test_restriction_projects_values_validity_coordinates_labels_and_units_together():
    repeat_id = AxisId("capture.repeat")
    site_id = AxisId("calibration.site")
    x_id = AxisId("camera.x")
    camera_frame = CoordinateFrameId("camera.sensor")
    schema = DatasetSchema(
        AxisSpec(
            repeat_id,
            "Shot",
            REPEAT,
            3,
            ("dark", "signal", "reference"),
            coordinate_labels=("Dark", "Signal", "Reference"),
        ),
        PointTable(
            3,
            (
                PointColumn(
                    site_id,
                    "Site",
                    SITE,
                    PointColumn.TEXT,
                    ("site-a", "site-b", "site-c"),
                    coordinate_frame=CoordinateFrameId("trap.array"),
                    coordinate_labels=("A", "B", "C"),
                ),
            ),
        ),
        None,
        ValueSchema(
            (
                AxisSpec(
                    x_id,
                    "Sensor x",
                    SPATIAL_X,
                    4,
                    (10.0, 11.0, 12.0, 13.0),
                    "px",
                    camera_frame,
                    coordinate_labels=("x10", "x11", "x12", "x13"),
                ),
            ),
            ValidityContract.components(x_id),
            np.dtype("<f4"),
            "count",
        ),
    )


    values = np.arange(np.prod(schema.physical_shape), dtype="<f4").reshape(
        schema.physical_shape
    )
    validity = values % 3 != 0
    source = owned_snapshot_from_arrays(
        schema,
        values,
        7,
        validity=validity,
        block_id="projection-source",
        stream_generation="projection-generation",
    )
    selection = Selection(
        (
            IndexRangeSelection(repeat_id, 1, 3),
            IndexRangeSelection(site_id, 0, 2),
            IndexRangeSelection(x_id, 1, 4),
        )
    )

    projected = restrict_snapshot(
        source,
        selection,
        reference_for=_reference_for("projection-result"),
    )

    projected_schema = projected.block.schema
    assert projected_schema.repeat_axis.coordinates == ("signal", "reference")
    assert projected_schema.repeat_axis.coordinate_labels == ("Signal", "Reference")
    site = projected_schema.point_table.column(site_id)
    assert site.values == ("site-a", "site-b")
    assert site.coordinate_labels == ("A", "B")
    assert site.coordinate_frame == CoordinateFrameId("trap.array")
    x = projected_schema.cell_schema.axis(x_id)
    assert x.coordinates == (11, 12, 13)
    assert x.coordinate_labels == ("x11", "x12", "x13")
    assert x.unit == "px"
    assert x.coordinate_frame == camera_frame
    assert projected_schema.cell_schema.value_unit == "count"
    np.testing.assert_array_equal(projected.block.values, values[1:3, 0:2, 1:4])
    np.testing.assert_array_equal(
        projected.expanded_validity(), validity[1:3, 0:2, 1:4]
    )


def _indexed_schema(offsets: tuple[int, ...]) -> DatasetSchema:
    repeat = AxisSpec(AxisId("repeat"), "repeat", REPEAT, 1, (0,))
    primary = PointColumn(
        PRIMARY_INDEX_AXIS_ID,
        "source index",
        PRIMARY_INDEX,
        PointColumn.NUMERIC,
        offsets,
    )
    return DatasetSchema(
        repeat,
        PointTable(len(offsets), (primary,)),
        None,
        ValueSchema.scalar(np.dtype("<f4"), "count"),
    )


def test_relative_indexed_windows_share_one_event_layout() -> None:
    assert indexed_schemas_compatible(
        _indexed_schema((-1, 0)),
        _indexed_schema((-2, -1, 0)),
    )
    assert not indexed_schemas_compatible(
        _indexed_schema((4, 5)),
        _indexed_schema((5, 6)),
    )


def test_take_indices_rejects_stepped_range_instead_of_silently_ignoring_step():
    with pytest.raises(ValueError, match="step"):
        take_indices(np.arange(6), range(0, 6, 2), axis=0)
