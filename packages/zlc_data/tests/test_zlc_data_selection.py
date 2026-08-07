import pytest



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
