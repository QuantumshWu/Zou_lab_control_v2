"""How a selected region meets the grid a sensor is willing to take.

A sensor can only start and end a region on the steps it declares, so a box
drawn on an image is never exactly the box a camera takes.  Which way it is
rounded is a choice, and it is the SAME choice for every sensor -- so it is
made once, here, and each adapter supplies its own grid rather than its own
answer.

It was made twice, in opposite directions.  The qCMOS adapter covered the
request; the Basler adapter rounded the size down, so on a Width increment of
4 a requested 1918 became 1916 and the right-hand columns of the selected
region were never read out -- silently, because the applied region is what
everything downstream then measures.  Both adapters implement the same
``CameraAdapter.set_roi``, driven by the same authored roi_x/roi_y/roi_width/
roi_height and the same image-area selector, so the operator's box changed
meaning with the camera behind it.
"""

from __future__ import annotations


def snap_roi_axis(
    origin: int,
    extent: int,
    *,
    origin_step: int,
    extent_step: int,
    sensor_extent: int,
    minimum_extent: int | None = None,
) -> tuple[int, int]:
    """Snap one axis so the region asked for is COVERED, not clipped.

    Selecting a region means covering it: the origin rounds DOWN to its step
    and the far edge rounds UP to the size step, so the applied region
    contains what was asked for, clamped only by the sensor itself.

    ``minimum_extent`` is the smallest region the sensor accepts on this axis
    when that is larger than one size step (Basler declares a Width minimum
    of 16 on an increment of 4); it is itself rounded up onto the grid.
    """

    origin_step = max(1, int(origin_step))
    extent_step = max(1, int(extent_step))
    sensor_extent = int(sensor_extent)
    floor = extent_step if minimum_extent is None else max(extent_step, int(minimum_extent))
    floor = -(-floor // extent_step) * extent_step
    start = max(0, min(int(origin), sensor_extent - 1))
    stop = max(start + 1, min(int(origin) + int(extent), sensor_extent))
    snapped_origin = (start // origin_step) * origin_step
    covered = stop - snapped_origin
    snapped_extent = max(floor, -(-covered // extent_step) * extent_step)
    if snapped_extent > sensor_extent:
        # Larger than the sensor: nothing covers it, so take the whole sensor.
        return 0, max(floor, (sensor_extent // extent_step) * extent_step)
    if snapped_origin + snapped_extent > sensor_extent:
        # The far edge cannot be reached from where the region starts -- an
        # odd origin under an even size step, say.  Walk the origin back onto
        # its own step, the direction it already rounds, rather than dropping
        # the far edge: shrinking here would reinstate exactly the clipping
        # this rule exists to prevent.
        snapped_origin = ((sensor_extent - snapped_extent) // origin_step) * origin_step
    return snapped_origin, snapped_extent


__all__ = ["snap_roi_axis"]
