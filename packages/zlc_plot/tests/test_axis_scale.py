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

from concurrent.futures import ThreadPoolExecutor
import math
from threading import Barrier

import numpy as np
import pytest

from data_factory import (
    axis,
    make_dataset_schema,
    make_snapshot,
    mapped_domain_from_columns,
    repeat_domain,
)
from zlc_data import REPEAT, SITE
from zlc_plot import AxisRef, CurvePlot, HistogramPlot, ImagePlot, PlotSession, SelectorKind
from zlc_plot._axis_scale import LINEAR, LOG, axis_space, axis_value, midpoint
from zlc_plot._axis_transform import AxisTransform
from zlc_plot.selectors import DragHandle, NumericRange, SelectorState, _drag_numeric_range


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
    schema = make_dataset_schema(
        repeat_domain(size=1),
        mapped_domain_from_columns({"shot": np.asarray([0.0])}),
        cell_axes=(axis("site", values=[float(i) for i in range(64)], role=SITE),),
        dtype=np.float64,
    )
    rng = np.random.default_rng(5)
    values = rng.poisson(6.0, size=(1, 1, 64)).astype(np.float64)
    session = PlotSession(make_snapshot(schema, values, 0), HistogramPlot())
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
            transform = session._axis_transform_for_axis(axes, session._projected)
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

    # Image coordinates define equally spaced cells, including descending
    # nonuniform scans and nonlinear display units. Pointer -> canonical ->
    # painted selector must return to the same pixel between sample centers.
    from zlc_data.units import DEFAULT_UNITS
    for coordinates, unit in (((135., 191., 247.), "dBm"), ((247., 160., 135.), "mVpp")):
        schema = make_dataset_schema(
            repeat_domain(size=1), mapped_domain_from_columns({"shot": [0.]}),
            cell_axes=(axis("y", values=(0., 1., 2.), role=SITE),
                       axis("power", values=coordinates, role=SITE, unit="mVpp")),
            dtype=np.float64,
        )
        image = PlotSession(
            make_snapshot(schema, np.arange(9.).reshape(1, 1, 3, 3), 0),
            ImagePlot(AxisRef.cell_data("power"), AxisRef.cell_data("y")),
            parameters={"x_display_unit": unit},
        )
        try:
            image.rgba()
            axes = image._renderer.primary_axes
            transform = image._axis_transform_for_axis(axes, image._projected)
            display_values = DEFAULT_UNITS.convert(np.asarray(coordinates), "mVpp", unit)
            displayed = float((display_values[0] + display_values[1]) / 2)
            nx, ny = transform.display_to_normalized(displayed, 1.)
            canonical = transform.canonical_from_normalized(nx, ny)
            expected = float(DEFAULT_UNITS.convert(displayed, unit, "mVpp"))
            assert canonical.x == pytest.approx(expected)
            if unit == "dBm":
                assert canonical.x == pytest.approx(math.sqrt(135. * 191.))
            painted = image._painted_selector_state(SelectorState(SelectorKind.CROSSHAIR, canonical))
            np.testing.assert_allclose(transform.display_to_normalized(painted.value.x, painted.value.y), (nx, ny))
            from zlc_plot.notebook import _axis_from_dict, _axis_to_dict
            import json
            restored = _axis_from_dict(json.loads(json.dumps(_axis_to_dict(transform))))
            assert restored == transform
            assert restored.canonical_from_normalized(nx, ny) == canonical
            import pickle
            from zlc_plot.render_process import _encode_message
            assert pickle.loads(_encode_message(transform)) == transform
            image.set_area_selector(
                NumericRange(min(coordinates), max(coordinates)), NumericRange(0., 2.), display=False,
            )
            selection = image.selectors
            for step in (-1., 1.):
                image._raster_pointer_event("scroll", nx, ny, step=step, axes_snapshot=transform)
                image.rgba()
                transform = image._axis_transform_for_axis(axes, image._projected)
                box = axes.get_window_extent()
                assert box.width == pytest.approx(box.height)
                lattice_span = abs(np.diff(axis_space(np.asarray(axes.get_xlim()), transform.x_scale))[0])
                pitch = abs(np.diff(axis_space(display_values, transform.x_scale))[0])
                assert lattice_span / pitch == pytest.approx(abs(np.diff(axes.get_ylim())[0]))
                assert image.selectors == selection
            left, top, right, bottom = transform.bounds
            origin = (left + .4 * (right - left), top + .4 * (bottom - top))
            target = (left + .6 * (right - left), top + .6 * (bottom - top))
            for action, point in (("press", origin), ("move", target), ("release", target)):
                image._raster_pointer_event(action, *point, button=2, axes_snapshot=transform)
            assert image.selectors == selection
        finally:
            image.close()


def test_a_curve_shown_in_another_unit_points_where_it_draws() -> None:
    """A curve's axis is drawn straight in its DISPLAY unit.

    Only the image built a pointer scale for a nonlinear unit pair; every
    other surface interpolated in the canonical unit between converted
    limits, so on a dBm axis shown in mW a press at the middle of the plot
    read -15 dBm where the picture says -3 -- and a range or a threshold
    was committed there.
    """

    from zlc_data.units import DEFAULT_UNITS

    schema = make_dataset_schema(
        repeat_domain(size=1), mapped_domain_from_columns({"shot": [0.]}),
        cell_axes=(axis("power", values=(-30., -20., -10., -3., 0.), role=SITE, unit="dBm"),),
        dtype=np.float64,
    )
    session = PlotSession(
        make_snapshot(schema, np.arange(5.).reshape(1, 1, 5), 0),
        CurvePlot(AxisRef.cell_data("power")),
        parameters={"x_display_unit": "mW"},
    )
    try:
        session.rgba()
        transform = session._axis_transform_for_axis(
            session._renderer.primary_axes, session._projected
        )
        left, top, right, bottom = transform.bounds
        middle = top + 0.5 * (bottom - top)
        for fraction in (0.1, 0.5, 0.9):
            nx = left + fraction * (right - left)
            shown = transform.display_from_normalized(nx, middle).x
            pointed = transform.canonical_from_normalized(nx, middle).x
            assert pointed == pytest.approx(float(DEFAULT_UNITS.convert(shown, "mW", "dBm")))
    finally:
        session.close()


def test_an_axis_ending_where_its_unit_has_no_value_still_takes_a_drag() -> None:
    """A dBm value shown in mW is drawn from 0 mW, and 0 mW is -inf dBm.

    The selector bounds were built from that end, and the press and every
    move of every drag on the axis raised.  A hand below the axis, where no
    dBm exists at all, reads nothing there: a new box stays where the last
    reading drew it, and letting go there keeps it.  A box slid down past
    0 mW stops there on that axis alone; a hand that went there is no
    click; an Area's side, its body and an X range follow what the hand
    still reads there; and a threshold on the axis is grabbed within a
    fraction of what is drawn.
    """

    from zlc_data.units import DEFAULT_UNITS

    schema = make_dataset_schema(
        repeat_domain(size=1), mapped_domain_from_columns({"shot": [0.]}),
        cell_axes=(axis("site", values=(0., 1., 2., 3., 4.), role=SITE),),
        dtype=np.float64,
        value_unit="dBm",
    )
    session = PlotSession(
        make_snapshot(schema, np.asarray((-30., -20., -10., -3., 0.)).reshape(1, 1, 5), 0),
        CurvePlot(AxisRef.cell_data("site")),
        parameters={"value_display_unit": "mW"},
    )
    try:

        def hand(*steps: tuple[str, float, float]) -> AxisTransform:
            # One left-button gesture, read through the frame drawn as it
            # begins, at fractions of the plot box, top-origin like the box.
            session.rgba()
            transform = session._axis_transform_for_axis(
                session._renderer.primary_axes, session._projected
            )
            left, top, right, bottom = transform.bounds
            for action, fraction_x, fraction_y in steps:
                session._raster_pointer_event(
                    action,
                    left + fraction_x * (right - left),
                    top + fraction_y * (bottom - top),
                    button=1,
                    axes_snapshot=transform,
                )
            return transform

        def mw(value: float) -> float:
            return float(DEFAULT_UNITS.convert(value, "dBm", "mW"))

        transform = hand(
            ("press", 0.1, 0.1),
            ("move", 0.3, 0.6),
            ("move", 0.35, 1.05),
            ("release", 0.35, 1.05),
        )
        assert transform.y_limits[0] == 0.0
        assert transform.canonical_y_limits[0] == -math.inf
        (area,) = session.selectors
        assert area.kind is SelectorKind.AREA
        top_mw = transform.y_limits[1]
        assert mw(area.value.y.low) == pytest.approx(0.4 * top_mw, rel=0.05)
        assert mw(area.value.y.high) == pytest.approx(0.9 * top_mw, rel=0.05)
        x_low, x_high = transform.x_limits
        span = x_high - x_low
        assert area.value.x.high == pytest.approx(x_low + 0.3 * span, abs=0.02 * span)

        # Slid 0.2 down, then 0.5 down and 0.1 right, where its bottom
        # would be below 0 mW: it stays where the last move drew it on y
        # and still slides on x.  Every such move raised.
        hand(
            ("press", 0.2, 0.35),
            ("move", 0.2, 0.55),
            ("move", 0.3, 0.85),
            ("release", 0.3, 0.85),
        )
        (area,) = session.selectors
        assert mw(area.value.y.low) == pytest.approx(0.2 * top_mw, rel=0.05)
        assert mw(area.value.y.high) == pytest.approx(0.7 * top_mw, rel=0.05)
        assert area.value.x.low == pytest.approx(x_low + 0.2 * span, abs=0.02 * span)

        # A new box dragged straight below the axis read nothing on the
        # way, and letting go there took the committed one away as a click.
        hand(("press", 0.7, 0.97), ("move", 0.75, 1.05), ("release", 0.75, 1.05))
        assert session.selectors == (area,)

        # Its left side moves x alone, so it follows a hand that drifts
        # below the axis, where y reads nothing.  Required of both, it
        # stayed where it was.
        hand(("press", 0.2, 0.55), ("move", 0.1, 1.05), ("release", 0.1, 1.05))
        (area,) = session.selectors
        assert area.value.x.low == pytest.approx(x_low + 0.1 * span, abs=0.02 * span)
        assert area.value.x.high == pytest.approx(x_low + 0.4 * span, abs=0.02 * span)
        assert mw(area.value.y.low) == pytest.approx(0.2 * top_mw, rel=0.05)

        # Its body, carried a little down and then with the hand below the
        # axis, keeps the height the last reading gave and still slides
        # along x, as it does when the box itself reaches 0 mW.
        hand(
            ("press", 0.25, 0.55),
            ("move", 0.3, 0.6),
            ("move", 0.45, 1.05),
            ("release", 0.45, 1.05),
        )
        (area,) = session.selectors
        assert area.value.x.low == pytest.approx(x_low + 0.3 * span, abs=0.02 * span)
        assert mw(area.value.y.low) == pytest.approx(0.15 * top_mw, rel=0.05)
        assert mw(area.value.y.high) == pytest.approx(0.65 * top_mw, rel=0.05)

        # A fraction of a canonical span ending at -inf dBm reached
        # nothing: the threshold could not be grabbed.
        session.set_threshold_selector(0.5 * top_mw)
        hand(("press", 0.8, 0.5), ("move", 0.8, 0.7), ("release", 0.8, 0.7))
        threshold = session.selector_state(SelectorKind.THRESHOLD)
        assert mw(threshold.value) == pytest.approx(0.3 * top_mw, rel=0.05)

        # An X range reads x alone: it follows a hand that drifts below the
        # axis, where y reads nothing, and is let go there.
        session.set_x_selector(x_low + 0.6 * span, x_low + 0.9 * span)
        hand(("press", 0.75, 0.2), ("move", 0.8, 1.05), ("release", 0.8, 1.05))
        x_range = session.selector_state(SelectorKind.X_RANGE)
        assert x_range.value.low == pytest.approx(x_low + 0.65 * span, abs=0.02 * span)
        assert session.selector_state(SelectorKind.AREA) == area
    finally:
        session.close()


def test_a_unit_with_no_value_for_a_selector_names_it_and_lets_the_view_go() -> None:
    """A mW value drawn from 0 mW has nothing in dBm at its floor: -inf.

    A box dragged to the bottom of that axis refused the switch to dBm as
    "range low must be finite", with nothing to say which selector to
    move.  The refusal names it, and nothing changed; a fixed value limit
    there is refused by name the same way, where it was stored as -inf
    and refused by the drawing unnamed.  A view reaching below 0 mW is
    where the hand left it, not a choice to keep: that axis lets go of its
    navigation and the switch goes through.  A curve's fit reads x alone
    and stays current; an image's keeps to the rows on screen, and letting
    its y go takes the fit's domain with it.
    """

    schema = make_dataset_schema(
        repeat_domain(size=1), mapped_domain_from_columns({"shot": [0.]}),
        cell_axes=(axis("site", values=(0., 1., 2., 3., 4.), role=SITE),),
        dtype=np.float64,
        value_unit="mW",
    )
    session = PlotSession(
        make_snapshot(schema, np.asarray((0.5, 1., 2., 4., 8.)).reshape(1, 1, 5), 0),
        CurvePlot(AxisRef.cell_data("site")),
    )
    try:
        session.rgba()
        session.set_area_selector(NumericRange(1.0, 3.0), NumericRange(0.0, 4.0))
        before = session.display_state.values
        with pytest.raises(ValueError, match="area selector y low 0 mW has no value in dBm"):
            session.set_parameters({"value_display_unit": "dBm"})
        assert session.display_state.values == before
        (area,) = session.selectors
        assert area.value.y.low == 0.0

        session.remove_selector(SelectorKind.AREA)
        session.set_parameters({"relim_mode": "fixed", "y_min": 0.0, "y_max": 4.0})
        fixed = session.display_state.values
        with pytest.raises(ValueError, match="y_min 0 mW has no value in dBm"):
            session.set_parameters({"value_display_unit": "dBm"})
        assert session.display_state.values == fixed
        session.set_parameters({"relim_mode": before["relim_mode"]})

        session.set_viewport(NumericRange(1.0, 3.0), NumericRange(-1.0, 9.0))
        generation = session._fit_context_generation
        session.set_parameters({"value_display_unit": "dBm"})
        assert session.display_state["value_display_unit"] == "dBm"
        x, y = session.viewport
        assert (x.low, x.high) == pytest.approx((1.0, 3.0))
        assert y is None
        assert session._fit_context_generation == generation
        assert np.asarray(session.rgba()).size
    finally:
        session.close()

    schema = make_dataset_schema(
        repeat_domain(size=1), mapped_domain_from_columns({"shot": [0.]}),
        cell_axes=(axis("power", values=(0.5, 1., 1.5), role=SITE, unit="mW"),
                   axis("site", values=(0., 1., 2.), role=SITE)),
        dtype=np.float64,
    )
    image = PlotSession(
        make_snapshot(schema, np.arange(9.).reshape(1, 1, 3, 3), 0),
        ImagePlot(AxisRef.cell_data("site"), AxisRef.cell_data("power")),
    )
    try:
        image.rgba()
        image.set_viewport(NumericRange(0.0, 2.0), NumericRange(-1.0, 1.5))
        x, y = image.viewport
        assert y.low < 0.0
        generation = image._fit_context_generation
        image.set_parameters({"y_display_unit": "dBm"})
        assert image.viewport == (x, None)
        assert image._fit_context_generation == generation + 1
    finally:
        image.close()


def test_a_view_where_its_canonical_unit_has_no_value_is_not_navigated_there() -> None:
    """A dBm value shown in mW and zoomed out below 0 mW has no dBm there.

    Converted to dBm whole, that view raised "range low must be finite"
    after it had been committed and drawn: no viewport notice went out,
    and every unit switch -- back to dBm included -- was refused until the
    view was reset.  That axis is not navigated in dBm: the notice carries
    x alone, and the switch lets y go.  An edit that leaves y's own unit
    alone -- a window -- leaves y's view as drawn; rebuilt through a dBm it
    has none in, the zoom was let go.
    """

    schema = make_dataset_schema(
        repeat_domain(size=1), mapped_domain_from_columns({"shot": [0.]}),
        cell_axes=(axis("site", values=(0., 1., 2., 3., 4.), role=SITE),),
        dtype=np.float64,
        value_unit="dBm",
    )
    session = PlotSession(
        make_snapshot(schema, np.asarray((-30., -20., -10., -3., 0.)).reshape(1, 1, 5), 0),
        CurvePlot(AxisRef.cell_data("site")),
        parameters={"value_display_unit": "mW"},
    )
    notices: list = []
    try:
        session.rgba()
        session.subscribe_viewport(notices.append)
        session.set_viewport(NumericRange(1.0, 3.0), NumericRange(-0.1, 1.1))
        (notice,) = notices
        assert notice.display == session.viewport
        x, y = notice.canonical
        assert (x.low, x.high) == pytest.approx((1.0, 3.0))
        assert y is None

        view = session.viewport
        session.set_parameters({"window": 2})
        assert session.viewport == view

        # A curve's fit reads x alone, so letting y go leaves a fit in
        # flight, and a repeated one, where they were.
        generation = session._fit_context_generation
        session.set_parameters({"value_display_unit": "dBm"})
        assert session.display_state["value_display_unit"] == "dBm"
        x, y = session.viewport
        assert (x.low, x.high) == pytest.approx((1.0, 3.0))
        assert y is None
        assert session._fit_context_generation == generation
        assert notices[-1].display == session.viewport
        assert np.asarray(session.rgba()).size
    finally:
        session.close()


def test_concurrent_live_draws_and_exports_share_one_safe_mathtext_parser(
    tmp_path,
) -> None:
    workers = 6
    sessions = tuple(_histogram_session() for _ in range(workers))
    gate = Barrier(workers)

    def paint(item: tuple[int, PlotSession]) -> None:
        index, session = item
        gate.wait(timeout=30.0)
        for enabled in (True, False, True):
            session.set_parameters({"log_y": enabled})
        session.save(tmp_path / f"concurrent-{index}.png")
        assert np.asarray(session.rgba()).size

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            tuple(pool.map(paint, enumerate(sessions)))
        assert len(tuple(tmp_path.glob("concurrent-*.png"))) == workers
    finally:
        for session in sessions:
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

    # A nonuniform image lattice draws its coordinates as equal cells: a box
    # over four cells grabbed at 1.5 and carried three cells right covers
    # the next four, not a thirteen-and-a-half wide shift of the numbers.
    lattice = (0.0, 1.0, 2.0, 3.0, 10.0, 20.0, 30.0)
    carried = _drag_numeric_range(
        NumericRange(0.0, 3.0),
        handle=DragHandle.BODY,
        origin=1.5,
        position=15.0,
        scale=lattice,
    )
    assert carried.low == pytest.approx(3.0)
    assert carried.high == pytest.approx(30.0)

    # The wall stops the slide where the axis is straight too.  Pushed back
    # by a data delta, a decade-tall box slid 0.1 decade into the top of a
    # (0.8, 1200) count axis came back a decade and a quarter tall, and
    # 0.2 decade made it the whole axis.
    for decades in (0.1, 0.2):
        walled = _drag_numeric_range(
            NumericRange(100.0, 1000.0),
            handle=DragHandle.BODY,
            origin=10.0,
            position=10.0 * 10.0**decades,
            bounds=NumericRange(0.8, 1200.0),
            scale=LOG,
        )
        assert walled.high == pytest.approx(1200.0)
        assert walled.high / walled.low == pytest.approx(10.0)
    # A box sticking out of a view zoomed to (10, 1200) is pulled inside
    # as the box it is.  Pushed inside by a data delta first, (1, 100)
    # became (10, 109): one decade where there were two.
    held = _drag_numeric_range(
        NumericRange(1.0, 100.0),
        handle=DragHandle.BODY,
        origin=50.0,
        position=50.0,
        bounds=NumericRange(10.0, 1200.0),
        scale=LOG,
    )
    assert held.low == pytest.approx(10.0)
    assert held.high == pytest.approx(1000.0)
    # A box drawn down to 0 on a linear count axis, then shown on a log
    # one, has an end the scale cannot place: it is drawn from the bottom
    # of the view and slides from there.  Taken at -inf, it became the
    # whole view at the first move.
    lifted = _drag_numeric_range(
        NumericRange(0.0, 60.0),
        handle=DragHandle.BODY,
        origin=10.0,
        position=100.0,
        bounds=NumericRange(0.8, 1200.0),
        scale=LOG,
    )
    assert lifted.low == pytest.approx(8.0)
    assert lifted.high == pytest.approx(600.0)
    # Carried five cells right against the end of the lattice, the
    # four-cell box stops three cells over, not stretched over the axis.
    stopped = _drag_numeric_range(
        NumericRange(0.0, 3.0),
        handle=DragHandle.BODY,
        origin=1.5,
        position=35.0,
        bounds=NumericRange(0.0, 30.0),
        scale=lattice,
    )
    assert stopped.low == pytest.approx(3.0)
    assert stopped.high == pytest.approx(30.0)


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


def test_a_box_end_a_log_axis_cannot_place_is_grabbed_where_it_is_drawn() -> None:
    """A box drawn down to 0 counts, then shown on a log count axis.

    0 is -inf on a log axis: the bottom handles sat at -inf pixels and the
    middle of the sides was NaN, so neither the side handles nor the sides
    were drawn, and a press on the left side took the body and slid the
    whole box.  Drawn and grabbed at the wall it is clipped to, the side
    moves alone.
    """

    session = _histogram_session()
    try:
        session.rgba()
        transform = session._axis_transform_for_axis(
            session._renderer.primary_axes, session._projected
        )
        x_low, x_high = transform.x_limits
        left, right = x_low + 0.3 * (x_high - x_low), x_low + 0.6 * (x_high - x_low)
        top = 0.5 * transform.y_limits[1]
        session.set_area_selector(NumericRange(left, right), NumericRange(0.0, top))
        session.set_parameters({"log_y": True})
        session.rgba()
        transform = session._axis_transform_for_axis(
            session._renderer.primary_axes, session._projected
        )
        nx, ny = transform.display_to_normalized(
            left, midpoint(transform.y_limits[0], top, LOG)
        )
        for action, x in (("press", nx), ("move", nx - 0.05), ("release", nx - 0.05)):
            session._raster_pointer_event(action, x, ny, button=1, axes_snapshot=transform)
        area = session.selector_state(SelectorKind.AREA)
        assert area.value.x.low < left
        assert area.value.x.high == pytest.approx(right)
        assert area.value.y.low == 0.0
        assert area.value.y.high == pytest.approx(top)
    finally:
        session.close()


def test_axis_space_is_reversible_and_guards_a_stale_value() -> None:
    """A value from before the scale changed must not raise, only clamp."""

    assert axis_value(axis_space(37.0, LOG), LOG) == pytest.approx(37.0)
    assert axis_value(axis_space(37.0, LINEAR), LINEAR) == pytest.approx(37.0)
    assert axis_space(0.0, LOG) == -math.inf
    assert axis_space(-5.0, LOG) == -math.inf
    assert axis_space(-5.0, LINEAR) == -5.0
    for coordinates in ((1.0, 3.0, 10.0), (10.0, 3.0, 1.0)):
        # The scale itself stays increasing. Descending axes are reversed
        # by their limits, exactly as uniform axes, not a second time here.
        np.testing.assert_array_equal(axis_space(np.asarray((1., 3., 10.)), coordinates), (0., 1., 2.))
        positions = np.asarray((-.5, 0., .5, 1., 1.5, 2., 2.5))
        values = axis_value(positions, coordinates)
        np.testing.assert_allclose(axis_space(values, coordinates), positions)
        for position, value in zip(positions, values):
            assert axis_space(float(value), coordinates) == pytest.approx(position)
            assert axis_value(float(position), coordinates) == pytest.approx(value)
