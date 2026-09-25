"""A multi-frame camera cycle is a drawable image, frames pooled.

``camera_measurement`` publishes one cycle as ``(R cycles, F frame points,
y, x)``: the frames are POINTS, because different frames fire at different
places in the pulse.  Asking for a standalone image of that box names two
cell-data axes and no point axis -- which the projection layer used to
refuse outright, so a panel could be offered a picture it could never draw.

Pooling is one rule with no exceptions: R, P and D all meet the fate the
authored ``reduction`` states.  Here that means "average the frames", and
the guard checks the actual numbers, not merely that a build succeeded.
"""

from __future__ import annotations

import numpy as np
import pytest

from zlc_plot import PlotSession
from zlc_plot.data_view import DataView
from zlc_plot.kinds import AxisRef
from zlc_plot.specs import ImagePlot, Reduction

from data_factory import (
    axis,
    make_dataset_schema,
    make_snapshot,
    mapped_domain_from_columns,
    repeat_domain,
)


CYCLES, FRAMES, HEIGHT, WIDTH = 3, 4, 5, 6


def _camera_cycle(cycles: int = CYCLES, frame_count: int = FRAMES):
    """The reference shape, built the way the producer builds it."""

    from zlc_data import READOUT_EVENT, SPATIAL_X, SPATIAL_Y

    frames = mapped_domain_from_columns(
        {"frame": list(range(frame_count))},
        roles={"frame": READOUT_EVENT},
    )
    schema = make_dataset_schema(
        repeat_domain(size=cycles),
        frames,
        cell_axes=(
            axis("y", size=HEIGHT, role=SPATIAL_Y),
            axis("x", size=WIDTH, role=SPATIAL_X),
        ),
        dtype=np.float64,
    )
    assert schema.physical_shape == (cycles, frame_count, HEIGHT, WIDTH)
    rng = np.random.default_rng(11)
    values = rng.normal(size=schema.physical_shape)
    return make_snapshot(schema, values, revision=0), values


@pytest.mark.parametrize(
    "reduction, pool",
    [
        (Reduction.MEAN, lambda box: box.mean(axis=(0, 1))),
        (Reduction.MAX, lambda box: box.max(axis=(0, 1))),
    ],
    ids=["mean", "max"],
)
# A single-frame cycle takes the identical path, no branch of its own.
@pytest.mark.parametrize(
    "cycles, frame_count", [(CYCLES, FRAMES), (1, 1)], ids=["cycle", "one-frame"]
)
def test_a_camera_cycle_draws_as_one_image_with_its_frames_pooled(
    reduction, pool, cycles, frame_count
) -> None:
    snapshot, values = _camera_cycle(cycles, frame_count)
    view = DataView(snapshot)

    # No refusal: the two named axes are cell data axes, and the cycles and
    # the frame POINTS both pool under the declared reduction.
    view.validate_image(AxisRef.cell_data("x"), AxisRef.cell_data("y"))
    payload = view.image(
        AxisRef.cell_data("x"), AxisRef.cell_data("y"), aggregation=reduction
    )

    assert np.asarray(payload.z.canonical).shape == (HEIGHT, WIDTH)
    np.testing.assert_allclose(np.asarray(payload.z.canonical), pool(values))
    assert np.all(np.asarray(payload.valid))


def test_the_pooled_cycle_image_actually_renders() -> None:
    """Drawable, not merely projectable: the session builds and paints."""

    snapshot, _values = _camera_cycle()
    spec = ImagePlot(
        AxisRef.cell_data("x"), AxisRef.cell_data("y"), reduction=Reduction.MEAN
    )
    session = PlotSession(snapshot, spec)
    try:
        assert session.rgba().size
    finally:
        session.close()
