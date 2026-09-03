from __future__ import annotations

import numpy as np
import pytest

from zlc_data.axis import AxisId, AxisSpec, REPEAT, SITE, SPATIAL_X
from zlc_data.schema import DatasetSchema, DomainSpec, SCALAR_DOMAIN, ValueSchema
from zlc_data.snapshot_projection import axis_catalog, selection_indices, value_selection
from zlc_data.validity import ValidityContract



def test_a_coordinate_selection_resolves_on_an_implicit_axis() -> None:
    """An implicit axis HAS coordinates: index_origin + i, and says so.

    Refusing one here meant a box drawn on a camera frame only worked because
    the producer had written out the very tuple an implicit axis exists to
    avoid -- 2048 validated elements per frame to say what "none" already says.
    """

    from zlc_data.selection import CoordinateRangeSelection, resolve_selection_indices
    from zlc_data.axis import AxisId, AxisSpec
    from zlc_data import SPATIAL_X

    axis = AxisSpec(AxisId("cam.x"), "x", SPATIAL_X, 8)

    indices, dropped = resolve_selection_indices(
        axis, CoordinateRangeSelection(AxisId("cam.x"), 2.0, 5.0, None)
    )

    assert tuple(indices) == (2, 3, 4, 5)
    assert dropped is False


def test_an_implicit_axis_selection_respects_its_index_origin() -> None:
    """A cropped implicit axis records where it starts, and a box must land."""

    from zlc_data.selection import CoordinateRangeSelection, resolve_selection_indices
    from zlc_data.axis import AxisId, AxisSpec
    from zlc_data import SPATIAL_X

    axis = AxisSpec(AxisId("cam.x"), "x", SPATIAL_X, 4, index_origin=100)

    indices, _dropped = resolve_selection_indices(
        axis, CoordinateRangeSelection(AxisId("cam.x"), 101.0, 102.0, None)
    )

    assert tuple(indices) == (1, 2)


def test_a_coordinate_selection_off_an_implicit_axis_is_refused() -> None:
    from zlc_data.selection import CoordinateRangeSelection, resolve_selection_indices
    from zlc_data.axis import AxisId, AxisSpec
    from zlc_data import SPATIAL_X

    axis = AxisSpec(AxisId("cam.x"), "x", SPATIAL_X, 4)

    with pytest.raises(ValueError, match="empty"):
        resolve_selection_indices(
            axis, CoordinateRangeSelection(AxisId("cam.x"), 10.0, 20.0, None)
        )


def test_value_selection_resolves_axis_id_and_text_coordinate() -> None:
    site_id = AxisId("measurement.site")
    repeat = AxisSpec(AxisId("measurement.repeat"), "repeat", REPEAT, 1, (0,))
    site = AxisSpec(site_id, "site", SITE, 2, ("dark", "bright"))
    schema = DatasetSchema(
        DomainSpec((1,), (repeat,), ((0,),)),
        DomainSpec(
            (3,),
            (site,),
            ((0, 1, 0),),
        ),
        SCALAR_DOMAIN,
        ValueSchema.scalar(np.dtype("<f4"), "count"),
    )

    selection = value_selection(schema, {site_id: "dark"})
    _repeat, points, _data = selection_indices(schema, selection)

    assert points == (0, 2)


def test_value_selection_rejects_non_unique_human_axis_name() -> None:
    repeat = AxisSpec(AxisId("measurement.repeat"), "shared", REPEAT, 2, (0, 1))
    x = AxisSpec(AxisId("camera.x"), "shared", SPATIAL_X, 2, (0, 1))
    schema = DatasetSchema(
        DomainSpec((2,), (repeat,), ((0, 1),)),
        DomainSpec((1,), (), ()),
        DomainSpec((2,), (x,)),
        ValueSchema(ValidityContract.value(), np.dtype("<f4")),
    )

    with pytest.raises(ValueError, match="not uniquely present"):
        value_selection(schema, {"shared": 0})


def test_numeric_coordinate_range_skips_missing_coordinates() -> None:
    from zlc_data.selection import CoordinateRangeSelection, resolve_selection_indices

    axis = AxisSpec(
        AxisId("scan.frequency"),
        "frequency",
        SPATIAL_X,
        3,
        (0.0, None, 2.0),
    )

    indices, dropped = resolve_selection_indices(
        axis,
        CoordinateRangeSelection(axis.axis_id, 0.0, 2.0, None),
    )

    assert indices == (0, 2)
    assert dropped is False


def test_axis_catalog_preserves_point_coordinate_labels() -> None:
    site_id = AxisId("measurement.site")
    repeat = AxisSpec(AxisId("measurement.repeat"), "repeat", REPEAT, 1, (0,))
    site = AxisSpec(
        site_id,
        "site",
        SITE,
        2,
        ("site-a", "site-b"),
        coordinate_labels=("A", "B"),
    )
    schema = DatasetSchema(
        DomainSpec((1,), (repeat,), ((0,),)),
        DomainSpec((2,), (site,), ((0, 1),)),
        SCALAR_DOMAIN,
        ValueSchema.scalar(np.dtype("<f4"), "count"),
    )

    axis = next(item[2] for item in axis_catalog(schema) if item[1] == site_id)

    assert axis.coordinates == ("site-a", "site-b")
    assert axis.coordinate_labels == ("A", "B")
