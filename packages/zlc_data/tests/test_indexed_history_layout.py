"""An indexed history's layout is read once from Point-domain codes."""

from __future__ import annotations

import numpy as np
import pytest
from dataclasses import replace

from zlc_data import (
    PRIMARY_INDEX,
    READOUT_EVENT,
    REPEAT,
    SITE,
    SHOT_TIME,
    SAMPLE_TIME,
    AxisId,
    AxisSpec,
    BlockId,
    DatasetRevision,
    DatasetRevisionRef,
    DatasetSchema,
    DomainSpec,
    StreamGenerationId,
    ValidityContract,
    ValueSchema,
    owned_snapshot_from_arrays,
)
from zlc_data.snapshot_projection import (
    PRIMARY_INDEX_AXIS_ID,
    SHOT_TIME_AXIS_ID,
    indexed_history_layout,
    indexed_schemas_compatible,
    restrict_snapshot,
    value_selection,
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


@pytest.mark.parametrize("factored", (False, True))
def test_the_layout_reads_shots_rows_and_the_repeating_event(factored) -> None:
    schema = _schema(
        (-3, -1, 0),
        (0, 0, 1, 1, 2, 2),
        frame_coordinates=(0, 1),
        frame_codes=(0, 1, 0, 1, 0, 1),
    )
    if factored:
        schema = replace(schema, point_domain=replace(
            schema.point_domain, axis_codes=(range(3), range(2)),
            axis_code_repeats=((2, 1), (1, 3)),
        ))
    layout = indexed_history_layout(schema)
    assert layout is not None
    assert layout.cells.tolist() == [-3, -1, 0]
    assert layout.inner_count == 2
    assert layout.shot_count == 3 and layout.row_count == 6
    assert layout.codes(np.asarray((5, 0, 2))).tolist() == [2, 0, 1]
    assert layout.codes().tolist() == [0, 0, 1, 1, 2, 2]
    assert layout.row_mask(1).tolist() == [False] * 4 + [True] * 2
    assert layout.row_mask(2).tolist() == [False] * 2 + [True] * 4
    assert layout.row_mask(50).tolist() == [True] * 6
    assert not layout.cells.flags.writeable
    with pytest.raises(ValueError):
        layout.cells.setflags(write=True)
    assert indexed_history_layout(schema) is layout
    for alternate in (None, PRIMARY_INDEX_AXIS_ID):
        time = AxisSpec(SHOT_TIME_AXIS_ID, "shot time", SHOT_TIME, 3, (0.1, 0.3, 0.4),
                        coordinate_of=alternate)
        point = schema.point_domain
        stamped = replace(schema, point_domain=DomainSpec(
            point.shape, (*point.axes, time), (*point.axis_codes, range(3)),
            (*(point.axis_code_repeats or ((1, 1), (1, 1))), (2, 1)),
        ))
        timed = indexed_history_layout(stamped)
        assert timed.event == layout.event
        assert timed.times.tolist() == [0.1, 0.3, 0.4]
        if alternate is None:
            with pytest.raises(ValueError, match="primary index's rows"):
                indexed_history_layout(replace(stamped, point_domain=replace(
                    stamped.point_domain, axis_codes=(*point.axis_codes, range(2, -1, -1)),
                )))


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
    ("offsets", "codes", "repeats", "reason"),
    (
        ((-1.5, 0.0), (0, 1), (1, 1), "integer"),
        ((0, -1), (0, 1), (1, 1), "ordered"),
        ((-1, 1), (0, 1), (1, 1), "latest offset 0"),
        ((-2, -1, 0), range(0, 3, 2), (2, 1), "ordered contiguous"),
        ((-1, 0), range(2), (2, 2), "ordered contiguous"),
        ((-1, 0), (0, 1, 0, 1), (1, 1), "ordered contiguous"),
        ((-1, 0), (0, 0), (1, 1), "ordered contiguous"),
    ),
)
def test_a_broken_shot_index_is_refused_not_read_leniently(
    offsets, codes, repeats, reason
) -> None:
    schema = _schema(offsets, codes)
    schema = replace(schema, point_domain=replace(
        schema.point_domain, shape=(len(codes) * repeats[0] * repeats[1],),
        axis_codes=(codes,), axis_code_repeats=(repeats,),
    ))
    with pytest.raises(ValueError, match=reason):
        indexed_history_layout(schema)


def test_a_history_restricted_to_past_shots_keeps_their_coordinates() -> None:
    """A Scope on the source index is a Scope like any other.

    Runtime materializes the latest shot as 0, and restricting that Dataset
    to an earlier shot keeps the shot's coordinate, -1, the way restricting
    a site axis keeps the site's.  The reader took "the last offset is 0"
    for part of the contract and refused the cropped history as a broken
    producer, so a Curve scoped to a past shot the history fully held could
    not be drawn.  What it refuses is an offset ABOVE 0: an absolute
    ordinal that never became a relative coordinate.
    """

    schema = _schema(
        (-2, -1, 0),
        (0, 0, 1, 1, 2, 2),
        frame_coordinates=(0, 1),
        frame_codes=(0, 1, 0, 1, 0, 1),
    )
    source = owned_snapshot_from_arrays(
        schema,
        np.arange(12.0).reshape(1, 6, 2),
        7,
        block_id="history",
        stream_generation="g",
    )
    past = restrict_snapshot(
        source,
        value_selection(schema, {PRIMARY_INDEX_AXIS_ID: -1}),
        reference_for=lambda derived: DatasetRevisionRef(
            BlockId("past"),
            StreamGenerationId("g"),
            derived.fingerprint,
            DatasetRevision(7),
        ),
    )
    np.testing.assert_array_equal(past.block.values, [[[4.0, 5.0], [6.0, 7.0]]])

    layout = indexed_history_layout(past.block.schema)
    assert layout is not None
    assert layout.cells.tolist() == [-1]
    assert layout.inner_count == 2
    assert indexed_schemas_compatible(schema, past.block.schema)


def test_cropped_records_keep_their_actual_row_membership() -> None:
    layout = indexed_history_layout(_schema(
        (-1, 0), (0, 0, 1),
        frame_coordinates=(0, 1, 2), frame_codes=(0, 1, 2),
    ))
    assert layout.inner_count is None
    assert layout.row_count == 3
    assert layout.codes().tolist() == [0, 0, 1]
    assert layout.row_mask(1).tolist() == [False, False, True]
    schema = _schema((-1, 0), (0, 0, 1))
    repeated = replace(schema, point_domain=replace(
        schema.point_domain, shape=(6,), axis_code_repeats=((2, 1),),
    ))
    layout = indexed_history_layout(repeated)
    assert layout.inner_count is None
    assert layout.codes().tolist() == [0, 0, 0, 0, 1, 1]
    assert layout.row_mask(1).tolist() == [False] * 4 + [True] * 2
    single = _schema((0,), (0,))
    single = replace(single, point_domain=replace(
        single.point_domain, shape=(12,), axis_codes=(range(1),), axis_code_repeats=((3, 4),),
    ))
    layout = indexed_history_layout(single)
    assert layout.inner_count == 12 and layout.row_mask(1).all()
    assert layout.codes().tolist() == [0] * 12

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

    # Compare the compact periodicity decision with the actual expanded
    # event rows, including nonconstant periods that cross inner repeats.
    for base, inner, outer, per_shot in (
        (range(2), 1, 4, 2), ((0, 1, 0, 1), 2, 2, 4),
        ((0, 0, 1, 1), 2, 1, 4), ((0, 1), 2, 3, 3),
        ((0, 0), 2, 3, 3), (range(3), 2, 2, 3),
    ):
        expanded = np.tile(np.repeat(base, inner), outer)
        shots = len(expanded) // per_shot
        source = _schema(range(1 - shots, 1), np.repeat(np.arange(shots), per_shot),
                         frame_coordinates=range(max(base) + 1), frame_codes=expanded)
        compact = replace(source, point_domain=replace(
            source.point_domain, axis_codes=(range(shots), base),
            axis_code_repeats=((per_shot, 1), (inner, outer)),
        ))
        layout = indexed_history_layout(compact)
        rows = expanded.reshape(shots, per_shot)
        repeated = bool(np.all(rows == rows[0]))
        expected = expanded[:per_shot] if repeated else expanded
        assert layout.inner_count == per_shot
        assert layout.event[3][0] == (len(expected),)
        assert layout.event[3][2] == (tuple(expected),)
        assert indexed_schemas_compatible(source, compact)


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

    factored = []
    for count in (2, 3):
        base = _schema(tuple(range(1 - count, 1)), np.repeat(np.arange(count), 3))
        sample = AxisSpec(
            AxisId("sample"), "sample time", SAMPLE_TIME, count * 3,
            np.asarray((0.0, 0.1, 0.2)), unit="s", coordinate_origins=np.arange(count),
        )
        point = replace(
            base.point_domain, axes=base.point_domain.axes + (sample,),
            axis_codes=(range(count), range(count * 3)), axis_code_repeats=((3, 1), (1, 1)),
        )
        factored.append(replace(base, point_domain=point))
    assert factored[0].fingerprint != factored[1].fingerprint
    assert factored[0].structure_fingerprint == factored[1].structure_fingerprint
