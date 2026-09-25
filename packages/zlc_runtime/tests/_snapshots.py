"""One way to build a test snapshot, shared by every test that needs one.

Three test modules had grown their own copy of this builder with slightly
different axis ids and revisions, which is how a fixture stops being a fixture:
a change to the schema vocabulary has to be made three times, and the copies
drift until they no longer describe the same thing.  The same holds for the
outputs committed from it and the nodes that commit them.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from zlc_data import (
    REPEAT,
    SCAN_POINT,
    AxisId,
    AxisSpec,
    BlockId,
    CellValidity,
    DataBlock,
    DatasetRevision,
    DatasetSchema,
    DomainSpec,
    OwnedSnapshot,
    SCALAR_DOMAIN,
    StreamGenerationId,
    ValueSchema,
)
from zlc_runtime.dataset import DatasetCoverage, MonitorCoverage
from zlc_runtime.dataset_output import DatasetOutputDeclaration, LiveDatasetOutput


def snapshot_schema(name: str) -> DatasetSchema:
    """A one-repeat, one-point, scalar schema named after its producer."""

    repeat = AxisSpec(AxisId(f"{name}.repeat"), "repeat", REPEAT, 1, (0,))
    point = AxisSpec(AxisId(f"{name}.point"), "point", SCAN_POINT, 1, (0,))
    return DatasetSchema(
        DomainSpec((1,), (repeat,), ((0,),)),
        DomainSpec((1,), (point,), ((0,),)),
        SCALAR_DOMAIN,
        ValueSchema.scalar(np.dtype("float64"), "count"),
    )


def snapshot(name: str, revision: int, *, value: float | None = None) -> OwnedSnapshot:
    """One cell carrying ``value`` (default: the revision, so shots differ).

    Every revision of one name belongs to that name's one generation, as the
    shots of one run do.
    """

    cell = float(revision) if value is None else float(value)
    block = DataBlock(
        BlockId(f"{name}-{revision}"),
        DatasetRevision(revision),
        np.asarray([[[cell]]], dtype=np.float64),
        CellValidity(np.ones((1, 1), dtype=np.bool_)),
        snapshot_schema(name),
    )
    return OwnedSnapshot(
        block.ref(StreamGenerationId(f"{name}-generation")),
        block,
    )


def monitor_output(
    declaration: DatasetOutputDeclaration,
    value: float,
    *,
    event_record: dict[str, object] | None = None,
) -> LiveDatasetOutput:
    """One monitor shot of ``declaration`` carrying ``value``."""

    return LiveDatasetOutput(
        declaration,
        snapshot(declaration.name, 1, value=value),
        MonitorCoverage(1, 1),
        event_record=event_record,
    )


def finite_output(
    declaration: DatasetOutputDeclaration,
    *,
    value: float,
    total: int,
    origin: int,
    written: int,
    event_record: dict[str, object] | None = None,
) -> LiveDatasetOutput:
    """The shot at repeat ``origin`` of a ``total``-repeat finite run."""

    event = snapshot(declaration.name, origin + 1, value=value)
    schema = event.block.schema
    (repeat,) = schema.repeat_domain.axes
    canonical = DatasetSchema(
        DomainSpec(
            (total,),
            (
                AxisSpec(
                    repeat.axis_id,
                    repeat.name,
                    repeat.role,
                    total,
                    tuple(range(total)),
                ),
            ),
            (tuple(range(total)),),
        ),
        schema.point_domain,
        schema.cell_domain,
        schema.value_schema,
    )
    return LiveDatasetOutput(
        declaration,
        event,
        DatasetCoverage(written, total),
        canonical_schema=canonical,
        cell_origin=(origin, 0),
        event_record=event_record,
    )


def producer(instance_id: str, *declarations: DatasetOutputDeclaration):
    """The smallest object the plane accepts as a producer."""

    return SimpleNamespace(
        instance_id=instance_id,
        dataset_output_declarations=declarations,
        signal_key=lambda name: f"{instance_id}/{name}",
    )


def paused_lane(instance_id: str, *declarations: DatasetOutputDeclaration):
    """A latest-lane node whose results are committed by the test itself.

    The paused lane is the existing public contract for a display-paced
    derivation that already holds its answers; evaluating is an error.
    """

    def refuse(*_arguments):
        raise AssertionError("paused processor must not evaluate")

    def fail(error):
        raise error

    return SimpleNamespace(
        instance_id=instance_id,
        dataset_output_declarations=declarations,
        signal_key=lambda name: f"{instance_id}/{name}",
        validate_processor_source=lambda _source: None,
        evaluate_processor=refuse,
        accept_processor_result=lambda *_arguments: None,
        accept_processor_failure=fail,
        accept_processor_cancelled=lambda: None,
        accept_processor_ended=lambda _error: None,
        request_processor_owner_wake=lambda: None,
    )
