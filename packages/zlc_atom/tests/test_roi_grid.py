"""Selecting a region means covering it, whichever sensor is behind it.

Both camera adapters implement the same ``CameraAdapter.set_roi``, driven by
the same authored roi_x/roi_y/roi_width/roi_height and the same image-area
selector.  They each wrote their own snapping rule, and the two rules
disagreed: the qCMOS covered the request, the Basler rounded the size down,
so on a Width increment of 4 a requested 1918 became 1916 and the right-hand
columns of the selection were never read out.  Silently -- the applied region
is what everything downstream then measures.
"""

from __future__ import annotations

import pytest

from zlc_atom.devices.camera import dcam, pylon, roi_grid
from zlc_atom.devices.camera.roi_grid import snap_roi_axis


def test_every_adapter_reads_the_one_rule() -> None:
    """Not two copies that agree today: one rule, two grids."""

    assert dcam.snap_roi_axis is roi_grid.snap_roi_axis
    assert pylon.snap_roi_axis is roi_grid.snap_roi_axis


@pytest.mark.parametrize("origin_step", (1, 2, 4, 8))
@pytest.mark.parametrize("extent_step", (1, 2, 4))
@pytest.mark.parametrize("origin", (0, 1, 7, 51, 101, 1917))
@pytest.mark.parametrize("extent", (1, 3, 16, 481, 641))
def test_the_applied_region_contains_the_requested_one(
    origin_step: int, extent_step: int, origin: int, extent: int
) -> None:
    sensor = 1920
    start, size = snap_roi_axis(
        origin,
        extent,
        origin_step=origin_step,
        extent_step=extent_step,
        sensor_extent=sensor,
    )

    assert start % origin_step == 0
    assert size % extent_step == 0
    assert 0 <= start and start + size <= sensor

    wanted_start = max(0, min(origin, sensor - 1))
    wanted_stop = max(wanted_start + 1, min(origin + extent, sensor))
    assert start <= wanted_start
    assert start + size >= wanted_stop


def test_a_sensor_minimum_larger_than_one_step_is_honoured() -> None:
    """Basler declares Width min 16 on an increment of 4."""

    start, size = snap_roi_axis(
        1000, 3, origin_step=4, extent_step=4, sensor_extent=1920, minimum_extent=16
    )
    assert (start, size) == (1000, 16)


def test_a_region_at_the_far_edge_stays_on_the_sensor() -> None:
    start, size = snap_roi_axis(
        1919, 40, origin_step=4, extent_step=4, sensor_extent=1920, minimum_extent=16
    )
    assert start + size <= 1920
    assert size == 16 and start % 4 == 0
