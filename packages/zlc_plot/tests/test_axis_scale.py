"""Pointing and drawing must agree about how an axis is divided.

Matplotlib owns the scale for DRAWING: one ``set_yscale`` call in the
histogram painter, and everything drawn in data coordinates lands correctly
ever after.  Nothing owned it for POINTING.  ``AxisTransform`` -- "one
immutable axes transform shared by native and raster interaction" -- carried
the limits and not the scale, so every pixel-to-data conversion was a
straight interpolation between the ends.

The two authorities then disagreed silently, and the disagreement looked
like a drawing bug: on a count axis limited to (0.8, 1200) a press at the
vertical middle of the plot box reported 600.4 where the middle of the
picture is 30.98, and Matplotlib faithfully drew that corner at nine per
cent from the top.  The box did not follow the pointer because the number
under the pointer was wrong.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from data_factory import Axis, DatasetSchema, DatasetSnapshot, PointTable
from zlc_plot import HistogramPlot, PlotSession
from zlc_plot._axis_scale import LINEAR, LOG, axis_space, axis_value, midpoint
from zlc_plot._axis_transform import AxisTransform
from zlc_plot.selectors import DragHandle, NumericRange, _drag_numeric_range


_BOX = (0.0, 0.0, 1.0, 1.0)


def _transform(**overrides) -> AxisTransform:
    fields = {
        "role": "main",
        "cell_index": None,
        "bounds": _BOX,
        "x_limits": (0.0, 10.0),
        "y_limits": (0.8, 1200.0),
        "canonical_x_limits": (0.0, 10.0),
        "canonical_y_limits": (0.8, 1200.0),
        "x_scale": LINEAR,
        "y_scale": LINEAR,
    }
    fields.update(overrides)
    return AxisTransform(**fields)


def test_the_middle_of_a_log_box_is_the_geometric_mean() -> None:
    """The number this reported was the arithmetic mean, four decades out."""

    linear = _transform()
    logarithmic = _transform(y_scale=LOG)

    assert linear.canonical_from_normalized(0.5, 0.5).y == pytest.approx(
        (0.8 + 1200.0) / 2.0
    )
    assert logarithmic.canonical_from_normalized(0.5, 0.5).y == pytest.approx(
        math.sqrt(0.8 * 1200.0)
    )
    # And it is not the answer the linear map gives, which is the defect.
    assert logarithmic.canonical_from_normalized(0.5, 0.5).y != pytest.approx(
        linear.canonical_from_normalized(0.5, 0.5).y
    )


def test_pixel_and_data_round_trip_under_every_scale() -> None:
    """Whatever goes out one side must come back in the other."""

    for scale in (LINEAR, LOG):
        transform = _transform(y_scale=scale)
        for fraction in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0):
            point = transform.display_from_normalized(fraction, fraction)
            back = transform.display_to_normalized(point.x, point.y)
            assert back[0] == pytest.approx(fraction, abs=1e-9)
            assert back[1] == pytest.approx(fraction, abs=1e-9)


def _histogram_session() -> PlotSession:
    schema = DatasetSchema.create(
        Axis.create("repeat", size=1),
        PointTable.from_columns({"shot": np.asarray([0.0])}),
        data_axes=(Axis.create("site", values=[float(i) for i in range(64)]),),
        dtype=np.float64,
        generation="axis-scale",
    )
    rng = np.random.default_rng(5)
    values = rng.poisson(6.0, size=(1, 1, 64)).astype(np.float64)
    session = PlotSession(DatasetSnapshot(schema, values, 0), HistogramPlot())
    session.set_size("2x2")
    return session


def test_the_transform_agrees_with_matplotlib_on_a_log_axis() -> None:
    """The two authorities, asked the same question about the same pixels.

    This is the whole invariant: for every fraction of the plot box, the
    transform the pointer path uses and the transData Matplotlib draws with
    must name the same value -- under every scale the renderer can install,
    not only under the one it was written for.
    """

    session = _histogram_session()
    try:
        for log_y in (False, True):
            session.set_parameters({"log_y": log_y})
            session.rgba()
            renderer = session._renderer
            axes = renderer.primary_axes
            transform = session._axis_transform_for_axis(axes)
            assert transform.y_scale == (LOG if log_y else LINEAR), (
                "the transform did not capture the scale the renderer set"
            )
            left, top, right, bottom = transform.bounds
            for fraction in np.linspace(0.02, 0.98, 20):
                # The transform speaks in FIGURE-normalized, top-origin
                # coordinates; the axes box is only part of the figure.
                point = transform.display_from_normalized(
                    left + 0.5 * (right - left),
                    top + fraction * (bottom - top),
                )
                truth = float(
                    axes.transData.inverted().transform(
                        axes.transAxes.transform((0.5, 1.0 - fraction))
                    )[1]
                )
                assert point.y == pytest.approx(truth, rel=1e-9), (
                    "log_y=%s at %.3f: transform says %r, matplotlib says %r"
                    % (log_y, fraction, point.y, truth)
                )
    finally:
        session.close()


def test_a_body_drag_slides_the_box_instead_of_stretching_it() -> None:
    """Dragging a body means SLIDE, and sliding is a screen operation.

    Adding a data delta to both ends slides on a linear axis and stretches
    on a logarithmic one, where equal screen distances are equal ratios.  A
    fix confined to the pixel-to-data entrance would have turned "the box
    does not follow the pointer" into "it follows but changes height".
    """

    original = NumericRange(10.0, 100.0)
    moved = _drag_numeric_range(
        original,
        handle=DragHandle.BODY,
        origin=20.0,
        position=200.0,
        scale=LOG,
    )
    # One decade of pointer travel is one decade of box travel, and the box
    # keeps its size ON SCREEN -- which on a log axis is its ratio.
    assert moved.low == pytest.approx(100.0)
    assert moved.high == pytest.approx(1000.0)
    assert moved.high / moved.low == pytest.approx(original.high / original.low)

    # The same call on a linear axis is the addition it always was.
    straight = _drag_numeric_range(
        original,
        handle=DragHandle.BODY,
        origin=20.0,
        position=30.0,
        scale=LINEAR,
    )
    assert straight.low == pytest.approx(20.0)
    assert straight.high == pytest.approx(110.0)


def test_an_edge_handle_sits_on_the_edge_it_belongs_to() -> None:
    """``(low + high) / 2`` is the middle of the box only on a linear axis."""

    assert midpoint(0.8, 1200.0, LINEAR) == pytest.approx((0.8 + 1200.0) / 2.0)
    assert midpoint(0.8, 1200.0, LOG) == pytest.approx(math.sqrt(0.8 * 1200.0))
    # Where the arithmetic mean would have put it, as a fraction of the box.
    transform = _transform(y_scale=LOG)
    # display_to_normalized answers in TOP-origin fractions, so a value near
    # the top of the axis is a small number.
    arithmetic = transform.display_to_normalized(0.0, (0.8 + 1200.0) / 2.0)[1]
    assert arithmetic < 0.1, (
        "the arithmetic mean of a log range renders near the top: %.3f"
        % arithmetic
    )
    geometric = transform.display_to_normalized(0.0, midpoint(0.8, 1200.0, LOG))[1]
    assert geometric == pytest.approx(0.5)


def test_every_axis_fact_the_renderer_can_set_is_captured() -> None:
    """The transform is built once; what it forgets, nothing can recover.

    The scale was not dropped somewhere downstream -- it was never picked
    up.  This asserts mechanically that the one builder asks for the scale
    wherever it asks for the limits, so the next axis property cannot be
    forgotten the same way.
    """

    import inspect

    from zlc_plot import session as session_module

    source = inspect.getsource(session_module.PlotSession._axis_transform_for_axis)
    for limits, scale in (("get_xlim", "get_xscale"), ("get_ylim", "get_yscale")):
        assert limits in source
        assert scale in source, (
            "the transform builder reads %s but never %s" % (limits, scale)
        )
    assert {"x_scale", "y_scale"} <= set(AxisTransform.__slots__)


def test_the_notebook_carries_the_scale_to_the_browser() -> None:
    """Two frontends share this transform so they cannot fail differently."""

    from zlc_plot.notebook import _axis_to_dict

    payload = _axis_to_dict(_transform(y_scale=LOG))
    assert payload["y_scale"] == LOG
    assert payload["x_scale"] == LINEAR


def test_axis_space_is_reversible_and_guards_a_stale_value() -> None:
    """A value from before the scale changed must not raise, only clamp."""

    assert axis_value(axis_space(37.0, LOG), LOG) == pytest.approx(37.0)
    assert axis_value(axis_space(37.0, LINEAR), LINEAR) == pytest.approx(37.0)
    assert axis_space(0.0, LOG) == -math.inf
    assert axis_space(-5.0, LOG) == -math.inf
    assert axis_space(-5.0, LINEAR) == -5.0
