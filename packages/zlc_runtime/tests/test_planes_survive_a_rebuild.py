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
    StreamGenerationId,
)
from zlc_runtime import DatasetCoverage, DatasetOutputDeclaration, LiveDatasetOutput
from zlc_runtime.plane import (
    SignalDataPlane,
    _IndexedMaterialization,
    _materialize_indexed_dataset,
)

from _snapshots import producer, snapshot_schema

GENERATION = StreamGenerationId("plane-rebuild")


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


def _finite_run(schema, chunks, read_after):
    declaration = DatasetOutputDeclaration("value", "test.value")
    node = producer("scan", declaration)
    plane = SignalDataPlane()
    views = {}
    try:
        plane.begin_generation(node)
        for sequence, (snapshot, origin) in enumerate(chunks, 1):
            plane.commit_live(node, {"value": LiveDatasetOutput(
                declaration, snapshot, DatasetCoverage(sequence, len(chunks)), schema, origin,
            )})
            if sequence in read_after:
                views[sequence] = plane.current_dataset("scan/value")
        plane.latest_publication("scan/value").value("scan/value").snapshot.block.as_segment()
        _, tap = plane.follow_publications("scan/value")
        plane.seal_committed(node)
        plane.retire(node)
        try:
            for snapshot, _origin in chunks:
                replayed = tap.next(0).value("scan/value").snapshot
                np.testing.assert_array_equal(replayed.block.values, snapshot.block.values)
                np.testing.assert_array_equal(replayed.block.as_segment()[0], snapshot.block.values)
                np.testing.assert_equal(replayed.block.sigma, snapshot.block.sigma)
        finally:
            tap.close()
        return views
    finally:
        plane.close()


def test_indexed_history_keeps_each_shots_stated_error_and_its_window() -> None:
    """The path a Rolling panel's lease turns on.

    Each shot keeps the error it stated; a hole in the history is unknown,
    and NaN is how that is written, not a zero; and where the block sits in
    its history, in absolute shot numbers, is on the block.
    """

    schema = snapshot_schema("fit")
    events = ((5, _shot(schema, 4.0, 0.1)), (7, _shot(schema, 6.0, 0.3)))
    built = _materialize_indexed_dataset(
        _IndexedMaterialization(
            "@logic/fit/amplitude",
            GENERATION,
            9,
            schema,
            None,
            events,
            5,
            7,
            5,
            None,
            (),
            None,
        )
    )
    assert built.block.window == IndexedWindow(5, 7)
    assert built.block.revision == DatasetRevision(9)
    built = built.materialize()
    values = built.block.values.reshape(-1)
    assert (values[0], values[2]) == (4.0, 6.0)
    sigma = np.asarray(built.block.sigma).reshape(-1)
    assert sigma[0] == pytest.approx(0.1)
    assert np.isnan(sigma[1])
    assert sigma[2] == pytest.approx(0.3)


def test_a_history_of_shots_that_state_nothing_states_nothing() -> None:
    """Absent stays absent: a camera signal grows no sigma plane."""

    schema = snapshot_schema("camera")
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
            0,
            None,
            (),
            None,
        )
    )
    assert built.materialize().block.sigma is None


def test_extending_a_run_gives_what_rebuilding_it_would_have() -> None:
    """The basis is the same answer, reached without re-answering it.

    A canonical Dataset never rewrites a cell it already holds, so the
    cells assembled for an earlier sequence are still those cells at a
    later one.  What that buys is cost -- the shot instead of the run --
    and what it must not cost is a single different number, in any of
    the three planes.  The exact run keeps the error of every chunk that
    stated one.
    """

    chunk_schema = snapshot_schema("run")
    run_schema = DatasetSchema(
        DomainSpec(
            (3,),
            (AxisSpec(AxisId("run.repeat"), "repeat", REPEAT, 3, (0, 1, 2)),),
            ((0, 1, 2),),
        ),
        chunk_schema.point_domain,
        chunk_schema.cell_domain,
        chunk_schema.value_schema,
    )
    chunks = (
        (_shot(chunk_schema, 4.0, 0.1), (0, 0)),
        (_shot(chunk_schema, 5.0, None), (1, 0)),
        (_shot(chunk_schema, 6.0, 0.3), (2, 0)),
    )
    whole = _finite_run(run_schema, chunks, (3,))[3].materialize()
    views = _finite_run(run_schema, chunks, (2, 3))
    prefix, extended = views[2].materialize(), views[3].materialize()
    assert prefix.block.values.reshape(-1).tolist() == [4.0, 5.0, 0.0]
    np.testing.assert_array_equal(
        np.asarray(extended.block.values), np.asarray(whole.block.values)
    )
    np.testing.assert_array_equal(
        np.asarray(extended.expanded_validity()),
        np.asarray(whole.expanded_validity()),
    )
    np.testing.assert_array_equal(
        np.asarray(extended.block.sigma), np.asarray(whole.block.sigma)
    )
    np.testing.assert_allclose(
        np.asarray(whole.block.sigma).reshape(-1), (0.1, np.nan, 0.3)
    )


def test_a_run_that_states_no_error_gains_none_from_a_basis() -> None:
    """What is not stated stays unstated across an extension too."""

    chunk_schema = snapshot_schema("plain")
    run_schema = DatasetSchema(
        DomainSpec(
            (2,),
            (AxisSpec(AxisId("plain.repeat"), "repeat", REPEAT, 2, (0, 1)),),
            ((0, 1),),
        ),
        chunk_schema.point_domain,
        chunk_schema.cell_domain,
        chunk_schema.value_schema,
    )
    first = (_shot(chunk_schema, 4.0, None), (0, 0))
    second = (_shot(chunk_schema, 5.0, None), (1, 0))
    extended = _finite_run(run_schema, (first, second), (1, 2))[2].materialize()
    assert extended.block.sigma is None
