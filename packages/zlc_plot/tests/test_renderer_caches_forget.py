"""Two caches keyed by identity, and what they may not keep.

An answer about an array is not a reason to keep the array: held strongly,
one composed RGBA plane per revision stayed alive on a live image panel --
about 4 MB each at 1024x1024 and DPR 2, so roughly a gigabyte over 256
shots, dropped to nothing, and up again, per panel.

And five box caches are keyed by ``id(axes)`` while ``relayout`` is
exactly where the old Axes are dropped, so a later generation could be
allocated at a freed address and read a stale answer about itself.
"""

from __future__ import annotations

import gc

import numpy as np

from data_factory import Axis, DatasetSchema, DatasetSnapshot, PointTable
from zlc_plot import AxisRef, CurvePlot, PlotSession


def _snapshot(revision: int = 0) -> DatasetSnapshot:
    schema = DatasetSchema.create(
        Axis.create("repeat", size=1),
        PointTable.from_columns({"x": [0.0, 1.0, 2.0]}),
        dtype=np.float64,
        generation="renderer-caches",
    )
    return DatasetSnapshot(
        schema, np.array([[1.0, 2.0, 3.0]]) + revision, revision=revision
    )


def _renderer(session: PlotSession):
    return session._renderer


def test_a_settled_opacity_does_not_keep_its_plane_alive() -> None:
    session = PlotSession(_snapshot(), CurvePlot(AxisRef.point("x")))
    try:
        renderer = _renderer(session)
        front = np.full((4, 4, 4), 255, dtype=np.uint8)
        assert renderer._front_is_opaque(front) is True
        assert renderer._front_is_opaque(front) is True
        assert len(renderer._front_opacity) == 1

        del front
        gc.collect()
        assert not renderer._front_opacity, (
            "the answer outlived the array it was about"
        )
    finally:
        session.close()


def test_relayout_forgets_the_boxes_of_the_axes_it_drops() -> None:
    """Five caches keyed by an address that relayout is about to free."""

    session = PlotSession(_snapshot(), CurvePlot(AxisRef.point("x")))
    try:
        session.set_size("2x2")
        session.rgba()
        renderer = _renderer(session)
        caches = (
            renderer._owned_box,
            renderer._box_exact,
            renderer._planned_ratio,
            renderer._quantized_bounds,
        )
        assert any(cache for cache in caches) or renderer._owned_axes, (
            "nothing was recorded, so this proves nothing"
        )
        session.set_size("4x4")
        session.rgba()
        live = {id(axes) for axes in renderer._figure.get_axes()}
        for cache in caches:
            assert set(cache) <= live, sorted(set(cache) - live)
        assert renderer._owned_axes <= live
    finally:
        session.close()
