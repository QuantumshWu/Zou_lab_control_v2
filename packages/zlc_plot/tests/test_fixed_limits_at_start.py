"""A session built over a stored fixed colour pair starts, and keeps the pair."""

from __future__ import annotations

import numpy as np

from data_factory import (
    axis,
    make_dataset_schema,
    make_snapshot,
    mapped_domain_from_columns,
    repeat_domain,
)
from zlc_data import SPATIAL_X, SPATIAL_Y
from zlc_plot import AxisRef, ImagePlot, PlotSession


def _image_snapshot():
    x = np.linspace(-2.0, 2.0, 21)
    y = np.linspace(-3.0, 3.0, 25)
    schema = make_dataset_schema(
        repeat_domain(size=1),
        mapped_domain_from_columns({"sample": [0.0]}),
        cell_axes=(
            axis("x", values=x, unit="m", role=SPATIAL_X),
            axis("y", values=y, unit="m", role=SPATIAL_Y),
        ),
        dtype=np.float64,
        value_unit="1",
    )
    xx, yy = np.meshgrid(x, y)
    values = 0.4 + 2.0 * np.exp(-((xx - 0.35) ** 2 / 0.7**2 + (yy + 0.8) ** 2 / 1.1**2))
    return make_snapshot(schema, values.T[None, None, :, :], revision=0)


def test_a_session_built_over_a_stored_fixed_colour_pair_starts() -> None:
    """The Edit tab and Save build a NEW host from the panel's stored display.
    With a colour range set by hand that display holds a complete fixed pair,
    and the host died at start with a bare KeyError('color_min'): the
    constructor's configure carries no limit in its patch, and the limit
    transition asked the patch for one instead of the store."""

    session = PlotSession(
        _image_snapshot(),
        ImagePlot(AxisRef.cell_data("x"), AxisRef.cell_data("y")),
        parameters={"relim_mode": "fixed", "color_min": 0.5, "color_max": 2.0},
    )
    try:
        values = session.display_state.values
        assert (values["relim_mode"], values["color_min"], values["color_max"]) == ("fixed", 0.5, 2.0)
        limits = session.resolved_color_limits()
        assert (limits.low, limits.high) == (0.5, 2.0)
    finally:
        session.close()
