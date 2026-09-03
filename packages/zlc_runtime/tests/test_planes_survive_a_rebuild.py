"""What a producer states about a sample survives being re-assembled.

Runtime rebuilds a dataset in two places: the indexed history a Rolling
panel's lease turns on, and the exact run assembled from its chunks.  Both
allocate a blank dataset and fill it from the snapshots they were given,
and both used to allocate and fill only the values and the validity -- so a
fitted parameter's own error, published correctly and restamped correctly,
was destroyed on the last hop before the panel saw it.  The rolling trace
then pooled one sample per shot, found no scatter over one sample, and drew
no band anywhere.

These hold both directions: what is stated survives, and what is NOT stated
stays unstated.  A cell no producer spoke for gets NaN, never zero -- zero
is a claim of certainty nobody made.
"""

from __future__ import annotations

import numpy as np
import pytest

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
    IndexedWindow,
    OwnedSnapshot,
    SCALAR_DOMAIN,
    StreamGenerationId,
    ValueSchema,
)
from zlc_runtime.plane import (
    SignalDataPlane,
    _IndexedMaterialization,
    _materialize_indexed_dataset,
)

GENERATION = StreamGenerationId("plane-rebuild")


def _schema(name: str) -> DatasetSchema:
    repeat = AxisSpec(AxisId(f"{name}.repeat"), "repeat", REPEAT, 1, (0,))
    point = AxisSpec(AxisId(f"{name}.point"), "point", SCAN_POINT, 1, (0,))
    return DatasetSchema(
        DomainSpec((1,), (repeat,), ((0,),)),
        DomainSpec((1,), (point,), ((0,),)),
        SCALAR_DOMAIN,
        ValueSchema.scalar(np.dtype("float64"), "count"),
    )


def _shot(
    schema: DatasetSchema, value: float, sigma: float | None
) -> OwnedSnapshot:
    block = DataBlock(
        BlockId("shot"),
        DatasetRevision(1),
        np.asarray([[[value]]], dtype=np.float64),
        CellValidity(np.ones((1, 1), dtype=np.bool_)),
        schema,
        None if sigma is None else np.asarray([[[sigma]]], dtype=np.float64),
    )
    return OwnedSnapshot(block.ref(GENERATION), block)


def test_indexed_history_keeps_each_shots_stated_error() -> None:
    """The path a Rolling panel's lease turns on."""

    schema = _schema("fit")
    events = tuple(
        (index, _shot(schema, 4.0 + index, 0.1 * (index + 1)))
        for index in range(3)
    )
    built = _materialize_indexed_dataset(
        _IndexedMaterialization(
            "@logic/fit/amplitude",
            GENERATION,
            7,
            schema,
            None,
            events,
            0,
            2,
            None,
            {},
            {},
        )
    )
    assert built.block.sigma is not None
    np.testing.assert_allclose(
        np.asarray(built.block.sigma).reshape(-1), (0.1, 0.2, 0.3)
    )
    np.testing.assert_allclose(
        built.block.values.reshape(-1), (4.0, 5.0, 6.0)
    )


def test_an_index_nobody_published_has_an_unknown_error_not_a_zero_one() -> None:
    """A hole in the history is unknown, and NaN is how that is written."""

    schema = _schema("fit")
    events = ((0, _shot(schema, 4.0, 0.1)), (2, _shot(schema, 6.0, 0.3)))
    built = _materialize_indexed_dataset(
        _IndexedMaterialization(
            "@logic/fit/amplitude",
            GENERATION,
            7,
            schema,
            None,
            events,
            0,
            2,
            None,
            {},
            {},
        )
    )
    sigma = np.asarray(built.block.sigma).reshape(-1)
    assert sigma[0] == pytest.approx(0.1)
    assert np.isnan(sigma[1])
    assert sigma[2] == pytest.approx(0.3)


def test_a_history_of_shots_that_state_nothing_states_nothing() -> None:
    """Absent stays absent: a camera signal grows no sigma plane."""

    schema = _schema("camera")
    events = tuple(
        (index, _shot(schema, float(index), None)) for index in range(3)
    )
    built = _materialize_indexed_dataset(
        _IndexedMaterialization(
            "camera/frame",
            GENERATION,
            7,
            schema,
            None,
            events,
            0,
            2,
            None,
            {},
            {},
        )
    )
    assert built.block.sigma is None


def test_the_exact_run_keeps_the_error_of_every_chunk() -> None:
    """The other materializer, on the finite/exact path."""

    chunk_schema = _schema("run")
    run_schema = DatasetSchema(
        DomainSpec(
            (2,),
            (AxisSpec(AxisId("run.repeat"), "repeat", REPEAT, 2, (0, 1)),),
            ((0, 1),),
        ),
        chunk_schema.point_domain,
        chunk_schema.cell_domain,
        chunk_schema.value_schema,
    )
    chunks = (
        (_shot(chunk_schema, 4.0, 0.1), (0, 0)),
        (_shot(chunk_schema, 5.0, 0.2), (1, 0)),
    )
    built = SignalDataPlane._materialize_dataset(
        "scan/value", 3, run_schema, GENERATION, chunks
    )
    assert built.block.sigma is not None
    np.testing.assert_allclose(
        np.asarray(built.block.sigma).reshape(-1), (0.1, 0.2)
    )


def test_an_indexed_materialization_is_stamped_with_its_window() -> None:
    """Where the block sits in its history, in absolute shot numbers, on the block."""

    schema = _schema("camera")
    events = tuple(
        (index, _shot(schema, float(index), None)) for index in range(5, 8)
    )
    built = _materialize_indexed_dataset(
        _IndexedMaterialization(
            "camera/frame",
            GENERATION,
            9,
            schema,
            None,
            events,
            5,
            7,
            None,
            {},
            {},
            stable_since=4,
        )
    )
    assert built.block.window == IndexedWindow(5, 7, 4)
    assert built.block.revision == DatasetRevision(9)

