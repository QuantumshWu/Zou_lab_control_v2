"""An indexed history's layout is read once from Point-domain codes."""

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
    DomainSpec,
    ValidityContract,
    ValueSchema,
)
from zlc_data.snapshot_projection import (
    PRIMARY_INDEX_AXIS_ID,
    indexed_history_layout,
    indexed_schemas_compatible,
)


REPEAT_AXIS = AxisSpec(AxisId("repeat"), "repeat", REPEAT, 1, (0,))
REPEAT_DOMAIN = DomainSpec((1,), (REPEAT_AXIS,), ((0,),))
SITE_AXIS = AxisSpec(AxisId("site"), "site", SITE, 2, (0, 1))
CELL_DOMAIN = DomainSpec((2,), (SITE_AXIS,))
VALUE = ValueSchema(
    ValidityContract.components(SITE_AXIS.axis_id), np.dtype("<f8"), "count"
)


def _schema(
    offsets,
    primary_codes,
    *,
    frame_coordinates=None,
    frame_codes=None,
    primary_role=PRIMARY_INDEX,
) -> DatasetSchema:
    primary = AxisSpec(
        PRIMARY_INDEX_AXIS_ID,
        "source index",
        primary_role,
        len(offsets),
        tuple(offsets),
    )
    axes = [primary]
    codes = [tuple(primary_codes)]
    if frame_coordinates is not None:
        axes.append(
            AxisSpec(
                AxisId("frame"),
                "frame",
                READOUT_EVENT,
                len(frame_coordinates),
                tuple(frame_coordinates),
            )
        )
        codes.append(tuple(frame_codes))
    return DatasetSchema(
        REPEAT_DOMAIN,
        DomainSpec((len(primary_codes),), tuple(axes), tuple(codes)),
        CELL_DOMAIN,
        VALUE,
    )


def test_the_layout_reads_shots_rows_and_the_repeating_event() -> None:
    schema = _schema(
        (-3, -1, 0),
        (0, 0, 1, 1, 2, 2),
        frame_coordinates=(0, 1),
        frame_codes=(0, 1, 0, 1, 0, 1),
    )
    layout = indexed_history_layout(schema)
    assert layout is not None
    assert layout.cells.tolist() == [-3, -1, 0]
    assert layout.inner_count == 2
    assert layout.shot_count == 3 and layout.row_count == 6
    assert layout.codes().tolist() == [0, 0, 1, 1, 2, 2]
    assert layout.row_mask(1).tolist() == [False] * 4 + [True] * 2
    assert layout.row_mask(2).tolist() == [False] * 2 + [True] * 4
    assert layout.row_mask(50).tolist() == [True] * 6
    assert not layout.cells.flags.writeable
    with pytest.raises(ValueError):
        layout.cells.setflags(write=True)
    assert indexed_history_layout(schema) is layout


def test_a_schema_without_a_shot_index_has_no_layout() -> None:
    frame = AxisSpec(AxisId("frame"), "frame", READOUT_EVENT, 2, (0, 1))
    plain = DatasetSchema(
        REPEAT_DOMAIN,
        DomainSpec((2,), (frame,), ((0, 1),)),
        CELL_DOMAIN,
        VALUE,
    )
    assert indexed_history_layout(plain) is None
    assert indexed_history_layout(plain) is None


@pytest.mark.parametrize(
    ("offsets", "codes", "reason"),
    (
        ((-1.5, 0.0), (0, 1), "integer"),
        ((0, -1), (0, 1), "ordered"),
        ((-1, 0), (0, 0, 1), "same event rows"),
        ((-2, -1), (0, 1), "latest offset 0"),
    ),
)
def test_a_broken_shot_index_is_refused_not_read_leniently(
    offsets, codes, reason
) -> None:
    with pytest.raises(ValueError, match=reason):
        indexed_history_layout(_schema(offsets, codes))


def test_the_event_codes_must_repeat_under_every_shot() -> None:
    with pytest.raises(ValueError, match="Point domain"):
        indexed_history_layout(
            _schema(
                (-1, 0),
                (0, 0, 1, 1),
                frame_coordinates=(0, 1, 2),
                frame_codes=(0, 1, 0, 2),
            )
        )

    mislabelled = _schema((0,), (0,), primary_role=READOUT_EVENT)
    with pytest.raises(ValueError, match="primary-index role"):
        indexed_history_layout(mislabelled)


def test_two_windows_of_one_history_are_compatible_and_two_events_are_not() -> None:
    short = _schema(
        (-1, 0),
        (0, 0, 1, 1),
        frame_coordinates=(0, 1),
        frame_codes=(0, 1, 0, 1),
    )
    longer = _schema(
        (-2, -1, 0),
        (0, 0, 1, 1, 2, 2),
        frame_coordinates=(0, 1),
        frame_codes=(0, 1, 0, 1, 0, 1),
    )
    assert indexed_schemas_compatible(short, longer)

    other_event = _schema(
        (-1, 0),
        (0, 0, 1, 1),
        frame_coordinates=(0, 2),
        frame_codes=(0, 1, 0, 1),
    )
    assert not indexed_schemas_compatible(short, other_event)

    frame = AxisSpec(AxisId("frame"), "frame", READOUT_EVENT, 2, (0, 1))
    plain = DatasetSchema(
        REPEAT_DOMAIN,
        DomainSpec((2,), (frame,), ((0, 1),)),
        CELL_DOMAIN,
        VALUE,
    )
    assert not indexed_schemas_compatible(short, plain)


def test_a_sliding_history_keeps_the_structure_it_advances_through() -> None:
    """Deepening and sliding a window is not a new world to a gesture.

    The FULL name is the dataset's identity and must move on every shot: the
    coordinates really did change.  The STRUCTURE name answers a different
    question -- what an interaction was measured on -- and a bounded window
    filling up and then advancing is the same axes throughout.  Reading the
    full name for that question, or a structure name that still carried the
    window's DEPTH, threw the operator's zoom away on every shot and rebuilt
    the panel's host with it.
    """

    filling = _schema((-2, -1, 0), (0, 1, 2))
    deeper = _schema((-3, -2, -1, 0), (0, 1, 2, 3))
    slid = _schema((-3, -2, -1, 0), (0, 1, 2, 3))

    assert filling.fingerprint != deeper.fingerprint
    assert filling.structure_fingerprint == deeper.structure_fingerprint
    assert slid.structure_fingerprint == deeper.structure_fingerprint

    # An inner event axis rides the same flat carrier, so its codes lengthen
    # with the window too, and that is not a change of structure either.
    inner = _schema(
        (-1, 0),
        (0, 0, 1, 1),
        frame_coordinates=(0, 1),
        frame_codes=(0, 1, 0, 1),
    )
    inner_deeper = _schema(
        (-2, -1, 0),
        (0, 0, 1, 1, 2, 2),
        frame_coordinates=(0, 1),
        frame_codes=(0, 1, 0, 1, 0, 1),
    )
    assert inner.fingerprint != inner_deeper.fingerprint
    assert inner.structure_fingerprint == inner_deeper.structure_fingerprint

    # And a REAL change of axes still renames the structure: one more frame
    # per shot is a different world, however the window is doing.
    three_frames = _schema(
        (-2, -1, 0),
        (0, 0, 0, 1, 1, 1, 2, 2, 2),
        frame_coordinates=(0, 1, 2),
        frame_codes=(0, 1, 2, 0, 1, 2, 0, 1, 2),
    )
    assert inner_deeper.structure_fingerprint != three_frames.structure_fingerprint

    # A plain axis that no history advances keeps its size in the structure.
    ordinary = _schema((-1, 0), (0, 1), primary_role=READOUT_EVENT)
    ordinary_deeper = _schema((-2, -1, 0), (0, 1, 2), primary_role=READOUT_EVENT)
    assert (
        ordinary.structure_fingerprint != ordinary_deeper.structure_fingerprint
    )
