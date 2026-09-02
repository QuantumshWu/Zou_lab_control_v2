"""A dimension's labels live with its domain, not with the rows that survive.

A scoped view crops the point table to the rows it shows and keeps the
topology's whole domain ("cropped, not dropped").  Labels joined through
the surviving rows vanished for every cropped coordinate, so the domain
carries them itself and a matching column must agree.
"""

from __future__ import annotations

import numpy as np
import pytest
from zlc_data import (
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
from zlc_data.codec import grid_topology_from_tree, grid_topology_to_tree
from zlc_data.snapshot_projection import restricted_schema

PAIR = AxisId("pair")
SITE_ID = AxisId("site")
LABELS = ("0-1", "0-2", "1-2")


def _schema(*, column_labels, topology_labels):
    pair = PointColumn(
        PAIR, "pair", READOUT_EVENT, PointColumn.NUMERIC, (0, 1, 2),
        coordinate_labels=column_labels,
    )
    topology = GridTopology(
        (PAIR,), ((0, 1, 2),), ((0,), (1,), (2,)),
        coordinate_labels=topology_labels,
    )
    site = AxisSpec(SITE_ID, "site", SITE, 2, (0, 1))
    cell = ValueSchema(
        (site,), ValidityContract.components(SITE_ID), np.dtype("<f8"), "1"
    )
    repeat = AxisSpec(AxisId("repeat"), "repeat", REPEAT, 1, (0,))
    return DatasetSchema(repeat, PointTable(3, (pair,)), topology, cell)


def test_labels_are_validated_against_the_domain() -> None:
    with pytest.raises(ValueError):
        GridTopology((PAIR,), ((0, 1, 2),), ((0,), (1,), (2,)), coordinate_labels=(("a", "b"),))
    with pytest.raises(ValueError):
        GridTopology((PAIR,), ((0, 1, 2),), ((0,), (1,), (2,)), coordinate_labels=((1, 2, 3),))
    with pytest.raises(ValueError):
        GridTopology((PAIR,), ((0, 1, 2),), ((0,), (1,), (2,)), coordinate_labels=(LABELS, LABELS))
    unlabelled = GridTopology((PAIR,), ((0, 1, 2),), ((0,), (1,), (2,)), coordinate_labels=(None,))
    assert unlabelled.coordinate_labels is None


def test_a_labelled_column_needs_the_topology_to_carry_the_same_labels() -> None:
    with pytest.raises(ValueError):
        _schema(column_labels=LABELS, topology_labels=None)
    with pytest.raises(ValueError):
        _schema(column_labels=LABELS, topology_labels=(("x", "y", "z"),))
    schema = _schema(column_labels=LABELS, topology_labels=(LABELS,))
    assert schema.grid_topology.coordinate_labels == (LABELS,)
    # A topology may label a dimension its column does not.
    assert _schema(column_labels=None, topology_labels=(LABELS,)).grid_topology


def test_a_cropped_view_keeps_every_coordinate_label() -> None:
    schema = _schema(column_labels=LABELS, topology_labels=(LABELS,))
    cropped = restricted_schema(schema, range(1), (2,), {SITE_ID: range(2)})
    assert cropped.point_table.column(PAIR).coordinate_labels == ("1-2",)
    assert cropped.grid_topology.coordinate_domains == ((0, 1, 2),)
    assert cropped.grid_topology.coordinate_labels == (LABELS,)


def test_topology_labels_round_trip_through_the_codec() -> None:
    labelled = GridTopology((PAIR,), ((0, 1, 2),), ((0,), (1,), (2,)), coordinate_labels=(LABELS,))
    assert grid_topology_from_tree(grid_topology_to_tree(labelled)) == labelled
    plain = GridTopology((PAIR,), ((0, 1, 2),), ((0,), (1,), (2,)))
    assert "coordinate_labels" not in grid_topology_to_tree(plain)
    assert grid_topology_from_tree(grid_topology_to_tree(plain)) == plain
