"""One way to build a test snapshot, shared by every test that needs one.

Three test modules had grown their own copy of this builder with slightly
different axis ids and revisions, which is how a fixture stops being a fixture:
a change to the schema vocabulary has to be made three times, and the copies
drift until they no longer describe the same thing.
"""

from __future__ import annotations

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
    """One cell carrying ``value`` (default: the revision, so shots differ)."""

    cell = float(revision) if value is None else float(value)
    block = DataBlock(
        BlockId(f"{name}-{revision}"),
        DatasetRevision(revision),
        np.asarray([[[cell]]], dtype=np.float64),
        CellValidity(np.ones((1, 1), dtype=np.bool_)),
        snapshot_schema(name),
    )
    return OwnedSnapshot(
        block.ref(StreamGenerationId(f"{name}-generation-{revision}")),
        block,
    )
