"""Logical coordinate labels live once on each DomainSpec axis."""

from __future__ import annotations

import numpy as np
import pytest
import pickle
from dataclasses import replace

from zlc_data import (
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
from zlc_data.codec import domain_from_tree, domain_to_tree
from zlc_data.snapshot_projection import restricted_schema


PAIR = AxisId("pair")
SITE_ID = AxisId("site")
LABELS = ("0-1", "0-2", "1-2")


def _schema() -> DatasetSchema:
    pair = AxisSpec(
        PAIR,
        "pair",
        READOUT_EVENT,
        3,
        (0, 1, 2),
        coordinate_labels=LABELS,
    )
    repeat = AxisSpec(AxisId("repeat"), "repeat", REPEAT, 1, (0,))
    site = AxisSpec(SITE_ID, "site", SITE, 2, (0, 1))
    return DatasetSchema(
        DomainSpec((1,), (repeat,), ((0,),)),
        DomainSpec((3,), (pair,), ((0, 1, 2),)),
        DomainSpec((2,), (site,)),
        ValueSchema(
            ValidityContract.components(SITE_ID), np.dtype("<f8"), "1"
        ),
    )


def test_axis_validates_one_label_per_coordinate() -> None:
    with pytest.raises(ValueError, match="length"):
        AxisSpec(
            PAIR,
            "pair",
            READOUT_EVENT,
            3,
            (0, 1, 2),
            coordinate_labels=("a", "b"),
        )
    with pytest.raises(ValueError):
        AxisSpec(
            PAIR,
            "pair",
            READOUT_EVENT,
            3,
            (0, 1, 2),
            coordinate_labels=(1, 2, 3),
        )


def test_a_cropped_domain_crops_coordinates_labels_and_codes_together() -> None:
    schema = _schema()
    cropped = restricted_schema(schema, range(1), (2,), {SITE_ID: range(2)})
    pair = cropped.point_domain.axis(PAIR)
    np.testing.assert_array_equal(pair.coordinate_values(), (2,))
    assert pair.coordinate_labels == ("1-2",)
    np.testing.assert_array_equal(cropped.point_domain.axis_codes[0], (0,))


def test_domain_labels_round_trip_through_the_codec() -> None:
    domain = _schema().point_domain
    assert domain_from_tree(domain_to_tree(domain)) == domain
    alternate = AxisSpec(AxisId("pair.time"), "time", READOUT_EVENT, 3, (0.1, 0.3, 0.8), "s", coordinate_of=PAIR)
    paired = replace(domain, axes=domain.axes + (alternate,), axis_codes=domain.axis_codes * 2)
    assert paired.logical_shape == (3,)
    assert paired.coordinate_axis(alternate.axis_id) is domain.axes[0]
    assert paired.codes(PAIR) is paired.codes(alternate.axis_id)
    assert paired.axis_codes[0] is paired.axis_codes[1]
    assert paired.coordinate_counts() == (3, 3)
    assert domain_from_tree(domain_to_tree(paired)) == paired
    cropped = restricted_schema(replace(_schema(), point_domain=paired), range(1), (1,), {SITE_ID: range(2)})
    assert cropped.point_domain.axis(alternate.axis_id).coordinate_of == PAIR
    np.testing.assert_array_equal(cropped.point_domain.axis(alternate.axis_id).coordinate_values(), (0.3,))
    with pytest.raises(ValueError, match="same physical rows"):
        replace(paired, axis_codes=(domain.axis_codes[0], (2, 1, 0)))

    from zlc_data import SAMPLE_TIME

    offsets = np.asarray((0.0, 0.1, 0.2))
    origins = np.asarray((0.0, 1.0))
    sample = AxisSpec(AxisId("sample"), "sample time", SAMPLE_TIME, 6, offsets,
                      unit="s", coordinate_origins=origins)
    times = DomainSpec((6,), (sample,), (np.arange(6),))
    offsets[:] = -1
    origins[:] = -1
    np.testing.assert_array_equal(sample.coordinate_values(), (0.0, 0.1, 0.2, 1.0, 1.1, 1.2))
    assert sample.coordinate_at(3) == 1 and type(sample.coordinate_at(3)) is int
    assert len(sample) == 6 and sample[-1] == 1.2
    np.testing.assert_array_equal(sample[2:5], (0.2, 1.0, 1.1))
    np.testing.assert_array_equal(np.asarray(sample), sample.coordinate_values())
    assert sample.coordinate_position(1.1) == 4
    assert sample.coordinate_position(0.8) is None
    for missing in (None, "missing", -1, 2):
        assert sample.coordinate_position(missing) is None
    overlapping = replace(sample, coordinate_origins=np.asarray((0.0, 0.15)))
    assert overlapping.coordinate_position(0.15) == 3
    descending = replace(sample, coordinates=sample.coordinates[::-1])
    assert descending.coordinate_position(1.1) == 4
    assert domain_from_tree(domain_to_tree(times)) == times
    assert len(domain_to_tree(times)["axes"][0]["coordinates"]) == 3
    cropped = restricted_schema(replace(_schema(), point_domain=times), range(1), (4,), {SITE_ID: range(2)})
    assert cropped.point_domain.axes[0].coordinate_at(0) == 1.1
    with pytest.raises(ValueError, match="unique"):
        replace(sample, coordinate_origins=np.asarray((0.0, 0.1)))

    record = AxisSpec(AxisId("record"), "record", READOUT_EVENT, 2)
    record_time = AxisSpec(AxisId("record.time"), "record time", READOUT_EVENT, 2,
                           (0, 1), coordinate_of=record.axis_id)
    mapped = DomainSpec(
        (12,), (record, record_time, domain.axes[0]), (range(2), range(2), range(3)),
        ((3, 2), (3, 2), (1, 4)),
    )
    expected = np.tile(np.repeat(np.arange(2), 3), 2)
    assert mapped.code_mapping(record.axis_id) == (range(2), 3, 2)
    rows = np.asarray((11, 0, 7, 4, 4), dtype=np.int64)
    selected = mapped.codes(record_time.axis_id, rows)
    np.testing.assert_array_equal(selected, expected[rows])
    np.testing.assert_array_equal(mapped.codes(record.axis_id, rows[:0]), expected[:0])
    for selected_rows in (range(2, 10), range(11, -1, -3), range(0)):
        np.testing.assert_array_equal(mapped.codes(record.axis_id, selected_rows), expected[list(selected_rows)])
    with pytest.raises(ValueError):
        selected.setflags(write=True)
    with pytest.raises(IndexError):
        mapped.codes(record.axis_id, np.asarray((-1,)))
    with pytest.raises(IndexError):
        mapped.codes(record.axis_id, np.asarray((12,)))
    with pytest.raises(TypeError):
        mapped.codes(record.axis_id, np.asarray((0.5,)))
    dense = DomainSpec((2, 3), (record, domain.axes[0]))
    np.testing.assert_array_equal(dense.codes(PAIR, np.asarray((2, 0))), (2, 0))
    assert dense.code_at(PAIR, 2) == 2
    with pytest.raises(IndexError):
        dense.codes(PAIR, np.asarray((3,)))
    np.testing.assert_array_equal(mapped.codes(record.axis_id), expected)
    np.testing.assert_array_equal([mapped.code_at(record.axis_id, row) for row in range(12)], expected)
    np.testing.assert_array_equal(mapped.codes(PAIR), np.tile(np.arange(3), 4))
    assert mapped.codes(record_time.axis_id) is mapped.codes(record.axis_id)
    assert domain_from_tree(domain_to_tree(mapped)) == mapped
    restored_times, restored_mapping = pickle.loads(pickle.dumps((sample, mapped), protocol=5))
    assert restored_times == sample and restored_mapping == mapped
    assert restored_times.coordinate_position(1.1) == 4
    np.testing.assert_array_equal(restored_mapping.codes(record.axis_id), expected)
    for array in (restored_times.coordinates, restored_times.coordinate_origins, restored_mapping.codes(record.axis_id)):
        with pytest.raises(ValueError):
            array.setflags(write=True)
