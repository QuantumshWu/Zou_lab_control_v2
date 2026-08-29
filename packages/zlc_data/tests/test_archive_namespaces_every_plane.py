"""A figure keeps every plane of every dataset it holds, under one namespace.

A figure archive is one flat member namespace shared by several datasets,
so which members a dataset owns has to be decided once.  It was decided
three times -- in the claim list, in the writer's keyword arguments, and
again as a fixed pair the reader compared against -- and when the block
grew a sigma plane two of the three were updated.  The third wrote it as
the bare name ``sigma``: outside any dataset's namespace, so nothing
claimed it, two sigma-carrying datasets overwrote each other, and the
reader rejected the file the writer had just produced.

Which is reachable from the operator's hands: a fitted parameter is
published carrying its own sigma, a panel plots it, Save Figure hands that
snapshot to the writer, and the figure viewer could then never open it.
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
)
from zlc_data.figure_archive import read_archive, write_figure_archive

PNG = b"\x89PNG\r\n\x1a\n" + b"figure bytes"


def _schema() -> DatasetSchema:
    return DatasetSchema(
        AxisSpec(AxisId("t.repeat"), "repeat", REPEAT, 1, (0,)),
        PointTable(
            4,
            (
                PointColumn(
                    AxisId("t.point"),
                    "point",
                    SCAN_POINT,
                    PointColumn.NUMERIC,
                    (0, 1, 2, 3),
                ),
            ),
        ),
        None,
        ValueSchema.scalar(np.dtype("float64"), "count"),
    )


def _snapshot(name: str, scale: float, *, sigma: bool = True) -> OwnedSnapshot:
    schema = _schema()
    block = DataBlock(
        BlockId(name),
        DatasetRevision(1),
        np.asarray([[[1.0], [2.0], [3.0], [4.0]]]) * scale,
        CellValidity(np.ones((1, 4), dtype=np.bool_)),
        schema,
        np.asarray([[[0.1], [0.2], [0.3], [0.4]]]) * scale if sigma else None,
    )
    return OwnedSnapshot(block.ref(StreamGenerationId("figure")), block)


def _round_trip(arrays):
    stream = io.BytesIO()
    write_figure_archive(stream, "fig.png", arrays=arrays, sections={})
    stream.seek(0)
    return read_archive(stream)


def test_a_saved_figure_opens_again_when_its_dataset_states_an_error() -> None:
    """The writer produced files the reader refused; it no longer can."""

    _round_trip({"data": _snapshot("data", 1.0)})


def test_two_datasets_that_both_state_errors_keep_their_own() -> None:
    """One flat namespace, so the members have to be namespaced.

    Written under one bare name, the second dataset's sigma silently
    replaced the first -- and the collision check, which exists for
    exactly this, never saw the name because nothing claimed it.
    """

    _info, members = _round_trip(
        {"first": _snapshot("first", 1.0), "second": _snapshot("second", 10.0)}
    )
    assert "first.sigma" in members and "second.sigma" in members
    assert not np.array_equal(members["first.sigma"], members["second.sigma"])


def test_a_dataset_that_states_no_error_writes_no_member_for_one() -> None:
    _info, members = _round_trip({"data": _snapshot("data", 1.0, sigma=False)})
    assert not [name for name in members if name.endswith(".sigma")]


def test_every_member_a_dataset_writes_lives_under_its_own_name() -> None:
    """The rule the reader enforces, stated as the rule and not as a list.

    A fixed pair of accepted names has to be edited for every plane a
    block grows, and the edit that was missed is what produced files the
    reader rejected.  This is the property that must hold for the plane
    after next, without anyone editing anything.
    """

    _info, members = _round_trip(
        {"alpha": _snapshot("alpha", 1.0), "beta": _snapshot("beta", 2.0)}
    )
    for member in members:
        assert member.split(".", 1)[0] in ("alpha", "beta"), member
