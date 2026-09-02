"""An indexed history's layout is read once, from the schema, by everyone.

The primary-index column names each shot's relative offset once per event
row.  What the plot's window mask, the rolling trace's shot codes, the
compatibility gate and the title's shot count all need is the same three
facts -- the shots, the rows under each, the event that repeats -- and this
is the one place that derives them.
"""

from __future__ import annotations

import numpy as np
import pytest
from zlc_data import (
    PRIMARY_INDEX,
    READOUT_EVENT,
    REPEAT,
    SITE,
    AxisId,
    AxisSpec,
    DatasetSchema,
    GridTopology,
    PointColumn,
    PointTable,
    ValidityContract,
    ValueSchema,
)
from zlc_data.snapshot_projection import (
    PRIMARY_INDEX_AXIS_ID,
    indexed_history_layout,
    indexed_schemas_compatible,
)

REPEAT_AXIS = AxisSpec(AxisId("repeat"), "repeat", REPEAT, 1, (0,))
SITE_AXIS = AxisSpec(AxisId("site"), "site", SITE, 2, (0, 1))
CELL = ValueSchema(
    (SITE_AXIS,), ValidityContract.components(AxisId("site")), np.dtype("<f8"), "count"
)


def _primary(offsets, role=PRIMARY_INDEX) -> PointColumn:
    return PointColumn(
        PRIMARY_INDEX_AXIS_ID, "source index", role, PointColumn.NUMERIC, tuple(offsets)
    )


def _frame(values, labels=None) -> PointColumn:
    return PointColumn(
        AxisId("frame"),
        "frame",
        READOUT_EVENT,
        PointColumn.NUMERIC,
        tuple(values),
        coordinate_labels=labels,
    )


def _schema(offsets, *columns, topology=None) -> DatasetSchema:
    columns = (_primary(offsets), *columns)
    return DatasetSchema(REPEAT_AXIS, PointTable(len(offsets), columns), topology, CELL)


def test_the_layout_reads_shots_rows_and_the_repeating_event() -> None:
    schema = _schema((-3, -3, -1, -1, 0, 0), _frame((0, 1, 0, 1, 0, 1), ("a", "b") * 3))
    layout = indexed_history_layout(schema)
    assert layout is not None
    assert layout.cells.tolist() == [-3, -1, 0]  # a hole at -2 is a shot never received
    assert layout.inner_count == 2
    assert layout.shot_count == 3 and layout.row_count == 6
    assert layout.codes().tolist() == [0, 0, 1, 1, 2, 2]
    assert layout.row_mask(1).tolist() == [False] * 4 + [True] * 2
    assert layout.row_mask(2).tolist() == [False] * 2 + [True] * 4
    assert layout.row_mask(50).tolist() == [True] * 6
    assert not layout.cells.flags.writeable
    # Read once: the schema carries the answer for every later consumer.
    assert indexed_history_layout(schema) is layout


def test_a_schema_without_a_shot_index_has_no_layout() -> None:
    plain = DatasetSchema(REPEAT_AXIS, PointTable(2, (_frame((0, 1)),)), None, CELL)
    assert indexed_history_layout(plain) is None
    assert indexed_history_layout(plain) is None


@pytest.mark.parametrize(
    ("offsets", "reason"),
    (
        ((-1.5, 0.0), "integer"),
        ((0, -1), "ordered"),
        ((-1, -1, 0), "same event rows"),
        ((-2, -1), "latest offset 0"),
    ),
)
def test_a_broken_shot_index_is_refused_not_read_leniently(offsets, reason) -> None:
    with pytest.raises(ValueError, match=reason):
        indexed_history_layout(_schema(offsets))


def test_the_event_must_repeat_under_every_shot() -> None:
    with pytest.raises(ValueError, match="point columns"):
        indexed_history_layout(_schema((-1, -1, 0, 0), _frame((0, 1, 0, 2))))
    # The shot index under another role is a producer error, found when
    # the layout is read, not silently "no history".
    mislabelled = DatasetSchema(
        REPEAT_AXIS,
        PointTable(2, (_primary((-1, 0), role=READOUT_EVENT),)),
        None,
        CELL,
    )
    with pytest.raises(ValueError, match="primary-index role"):
        indexed_history_layout(mislabelled)


def test_the_topology_leads_with_the_shot_index_and_repeats_too() -> None:
    frame = _frame((0, 1, 0, 1))
    good = GridTopology(
        (PRIMARY_INDEX_AXIS_ID, AxisId("frame")),
        ((-1, 0), (0, 1)),
        ((0, 0), (0, 1), (1, 0), (1, 1)),
    )
    layout = indexed_history_layout(_schema((-1, -1, 0, 0), frame, topology=good))
    assert layout is not None and layout.event[3] == (
        (AxisId("frame"),),
        ((0, 1),),
        ((0,), (1,)),
        None,
    )
    # A dimension no column names can drift between shots without the
    # schema noticing; the layout does.
    twisted = GridTopology(
        (PRIMARY_INDEX_AXIS_ID, AxisId("frame"), AxisId("extra")),
        ((-1, 0), (0, 1), (0, 1)),
        ((0, 0, 0), (0, 1, 0), (1, 0, 1), (1, 1, 1)),
    )
    with pytest.raises(ValueError, match="topology"):
        indexed_history_layout(_schema((-1, -1, 0, 0), frame, topology=twisted))


def test_two_windows_of_one_history_are_compatible_and_two_events_are_not() -> None:
    short = _schema((-1, -1, 0, 0), _frame((0, 1, 0, 1)))
    longer = _schema((-2, -2, -1, -1, 0, 0), _frame((0, 1) * 3))
    assert indexed_schemas_compatible(short, longer)
    other_event = _schema((-1, -1, 0, 0), _frame((0, 2, 0, 2)))
    assert not indexed_schemas_compatible(short, other_event)
    plain = DatasetSchema(REPEAT_AXIS, PointTable(2, (_frame((0, 1)),)), None, CELL)
    assert not indexed_schemas_compatible(short, plain)
