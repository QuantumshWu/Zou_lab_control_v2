"""Logical coordinate labels live once on each DomainSpec axis."""

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
    assert pair.coordinates == (2,)
    assert pair.coordinate_labels == ("1-2",)
    assert cropped.point_domain.axis_codes == ((0,),)


def test_domain_labels_round_trip_through_the_codec() -> None:
    domain = _schema().point_domain
    assert domain_from_tree(domain_to_tree(domain)) == domain
