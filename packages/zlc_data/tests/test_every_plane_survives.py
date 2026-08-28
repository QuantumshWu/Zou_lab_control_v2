"""A block's planes travel together, through every door that copies one.

A ``DataBlock`` is values, validity, schema and -- for a producer whose
samples know their own error -- sigma.  Three doors copy a block: the
restamp that gives committed bytes a Runtime identity, the cutter that
narrows a snapshot to a scope, and the file format.  Each was written
before the sigma plane existed and each listed the fields it carried, so
each dropped it silently the day it arrived.

Listing fields is the defect these pin against.  ``replacing`` copies what
it was not asked to change, the cutter cuts every plane by the same
indices, and the format writes the key only when there is one to write.
"""

from __future__ import annotations

import io

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
    OwnedSnapshot,
    PointColumn,
    PointTable,
    StreamGenerationId,
    ValueSchema,
    load_npz,
    save_npz,
)
from zlc_data.selection import IndexRangeSelection, Selection
from zlc_data.snapshot_projection import restrict_snapshot

POINTS = 4
VALUES = np.asarray([[[1.0], [2.0], [3.0], [4.0]]])
SIGMA = np.asarray([[[0.1], [0.2], [0.3], [0.4]]])


def _schema() -> DatasetSchema:
    return DatasetSchema(
        AxisSpec(AxisId("t.repeat"), "repeat", REPEAT, 1, (0,)),
        PointTable(
            POINTS,
            (
                PointColumn(
                    AxisId("t.point"),
                    "point",
                    SCAN_POINT,
                    PointColumn.NUMERIC,
                    tuple(range(POINTS)),
                ),
            ),
        ),
        None,
        ValueSchema.scalar(np.dtype("float64"), "count"),
    )


def _snapshot(*, with_sigma: bool = True) -> OwnedSnapshot:
    schema = _schema()
    block = DataBlock(
        BlockId("planes"),
        DatasetRevision(3),
        VALUES,
        CellValidity(np.ones((1, POINTS), dtype=np.bool_)),
        schema,
        SIGMA if with_sigma else None,
    )
    return OwnedSnapshot(block.ref(StreamGenerationId("planes-gen")), block)


def _derived_ref(schema: DatasetSchema):
    """A derived dataset gets its own identity, never its source's."""

    from zlc_data import DatasetRevisionRef

    return DatasetRevisionRef(
        block_id=BlockId("planes.cut"),
        stream_generation=StreamGenerationId("planes-gen"),
        schema_fingerprint=schema.fingerprint,
        revision=DatasetRevision(3),
    )


def test_replacing_carries_the_planes_it_was_not_asked_about() -> None:
    """Change the identity; the content -- all of it -- comes along."""

    block = _snapshot().block
    restamped = block.replacing(
        block_id=BlockId("elsewhere"), revision=DatasetRevision(9)
    )
    assert str(restamped.block_id) == "elsewhere"
    assert int(restamped.revision.value) == 9
    np.testing.assert_array_equal(restamped.sigma, SIGMA)


def test_the_cutter_cuts_the_error_with_the_sample() -> None:
    """A scope says which samples survive, not what is known of each."""

    snapshot = _snapshot()
    cut = restrict_snapshot(
        snapshot,
        Selection((IndexRangeSelection(AxisId("t.point"), 1, 3),)),
        reference_for=lambda schema: _derived_ref(schema),
    )
    np.testing.assert_allclose(cut.block.values.reshape(-1), (2.0, 3.0))
    assert cut.block.sigma is not None
    np.testing.assert_allclose(
        np.asarray(cut.block.sigma).reshape(-1), (0.2, 0.3)
    )


def test_a_cut_of_something_that_states_nothing_states_nothing() -> None:
    snapshot = _snapshot(with_sigma=False)
    cut = restrict_snapshot(
        snapshot,
        Selection((IndexRangeSelection(AxisId("t.point"), 1, 3),)),
        reference_for=lambda schema: _derived_ref(schema),
    )
    assert cut.block.sigma is None


def test_a_saved_dataset_comes_back_with_its_error() -> None:
    """An archived run that loses its uncertainty is an archived lie."""

    stream = io.BytesIO()
    save_npz(stream, _snapshot())
    stream.seek(0)
    loaded = load_npz(stream)
    assert loaded.block.sigma is not None
    np.testing.assert_array_equal(loaded.block.sigma, SIGMA)
    assert loaded.exactly_equals(_snapshot())


def test_a_saved_dataset_that_stated_nothing_still_states_nothing() -> None:
    """Absent must not come back as zero, which would claim certainty."""

    stream = io.BytesIO()
    save_npz(stream, _snapshot(with_sigma=False))
    stream.seek(0)
    assert load_npz(stream).block.sigma is None


def test_two_blocks_that_differ_only_in_their_error_are_different() -> None:
    """Identity comparison sees every plane, or a rebuild can hide a loss."""

    assert not _snapshot().exactly_equals(_snapshot(with_sigma=False))


def test_a_negative_error_is_refused_rather_than_absolved() -> None:
    schema = _schema()
    with pytest.raises(ValueError):
        DataBlock(
            BlockId("planes"),
            DatasetRevision(3),
            VALUES,
            CellValidity(np.ones((1, POINTS), dtype=np.bool_)),
            schema,
            -SIGMA,
        )
