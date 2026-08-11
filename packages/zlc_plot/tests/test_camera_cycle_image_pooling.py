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

from data_factory import Axis, DatasetSchema, DatasetSnapshot, PointTable


CYCLES, FRAMES, HEIGHT, WIDTH = 3, 4, 5, 6


def _camera_cycle():
    """The reference shape, built the way the producer builds it."""

    from zlc_data import READOUT_EVENT, SPATIAL_X, SPATIAL_Y

    frames = PointTable.from_columns(
        {"frame": list(range(FRAMES))},
        roles={"frame": READOUT_EVENT},
    )
    schema = DatasetSchema.create(
        Axis.create("repeat", size=CYCLES),
        frames,
        data_axes=(
            Axis.create("y", size=HEIGHT, role=SPATIAL_Y),
            Axis.create("x", size=WIDTH, role=SPATIAL_X),
        ),
        dtype=np.float64,
        generation="camera-cycle",
    )
    assert schema.shape == (CYCLES, FRAMES, HEIGHT, WIDTH)
    rng = np.random.default_rng(11)
    values = rng.normal(size=schema.shape)
    return DatasetSnapshot(schema, values, revision=0), values


@pytest.mark.parametrize(
    "reduction, pool",
    [
        (Reduction.MEAN, lambda box: box.mean(axis=(0, 1))),
        (Reduction.MEDIAN, lambda box: np.median(box, axis=(0, 1))),
        (Reduction.MAX, lambda box: box.max(axis=(0, 1))),
    ],
    ids=["mean", "median", "max"],
)
def test_a_camera_cycle_draws_as_one_image_with_its_frames_pooled(
    reduction, pool
) -> None:
    snapshot, values = _camera_cycle()
    view = DataView(snapshot)

    # No refusal: the two named axes are cell data axes, and the cycles and
    # the frame POINTS both pool under the declared reduction.
    view.validate_image(AxisRef.data("x"), AxisRef.data("y"))
    payload = view.image(
        AxisRef.data("x"), AxisRef.data("y"), aggregation=reduction
    )

    assert np.asarray(payload.z.canonical).shape == (HEIGHT, WIDTH)
    np.testing.assert_allclose(np.asarray(payload.z.canonical), pool(values))
    assert np.all(np.asarray(payload.valid))


def test_the_pooled_cycle_image_actually_renders() -> None:
    """Drawable, not merely projectable: the session builds and paints."""

    snapshot, values = _camera_cycle()
    spec = ImagePlot(
        AxisRef.data("x"), AxisRef.data("y"), reduction=Reduction.MEAN
    )
    session = PlotSession(snapshot, spec)
    try:
        rgba = session.rgba()
        assert rgba.size
        np.testing.assert_allclose(
            np.asarray(session._payload.z.canonical), values.mean(axis=(0, 1))
        )
    finally:
        session.close()


def test_one_frame_is_not_a_special_case_of_the_same_rule() -> None:
    """A single-frame cycle takes the identical path, no branch of its own."""

    from zlc_data import READOUT_EVENT, SPATIAL_X, SPATIAL_Y

    frames = PointTable.from_columns(
        {"frame": [0]}, roles={"frame": READOUT_EVENT}
    )
    schema = DatasetSchema.create(
        Axis.create("repeat", size=1),
        frames,
        data_axes=(
            Axis.create("y", size=HEIGHT, role=SPATIAL_Y),
            Axis.create("x", size=WIDTH, role=SPATIAL_X),
        ),
        dtype=np.float64,
        generation="camera-cycle-single",
    )
    values = np.arange(HEIGHT * WIDTH, dtype=np.float64).reshape(schema.shape)
    payload = DataView(DatasetSnapshot(schema, values, revision=0)).image(
        AxisRef.data("x"), AxisRef.data("y")
    )
    np.testing.assert_allclose(
        np.asarray(payload.z.canonical), values[0, 0]
    )
