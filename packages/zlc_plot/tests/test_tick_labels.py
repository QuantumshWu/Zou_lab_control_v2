"""A tick label says what distinguishes its tick, and nothing else.

The failure these guard against: an axis over 1000000.0 .. 1000000.4 printed
``10000000 10000001 10000002`` beside a ``x1e-1``, eight characters each to
say four tenths.  The offset was taken only when the labels did not FIT the
axis width -- a question that is never asked of a y axis, so a y axis never
took one at all, whatever it was showing.
"""

from __future__ import annotations

import matplotlib
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import zlc_plot.ticks as ticks_module
from zlc_plot.layout import Room
from zlc_plot.ticks import (
    MIN_TICK_LABEL_PT,
    SmartOffsetLocator,
    apply_declared_ticks,
    apply_smart_ticks,
    count_label,
    declare_room,
)

#: Ranges this bench actually produces.  Each is (name, low, high).
RANGES = (
    ("an absolute shot number late in a run", 1234500.0, 1234600.0),
    ("a wavelength window", 1000000.0, 1000000.4),
    ("counts about a pedestal", 5180.0, 5220.0),
    ("a control voltage", -1.0, 2.0),
    ("an optical frequency in Hz", 384227900000.0, 384228100000.0),
    ("a normalised rate", 0.0, 1.0),
    ("a pulse length in seconds", 1.0e-6, 5.0e-6),
)

#: What a label may cost.  Four significant characters plus a sign and a
#: decimal point: "-0.25" is a label, "10000001" is a coordinate that forgot
#: to take its common part out.
LONGEST_LABEL = 6

#: What the caller is about to draw these labels at.  The policy prices its
#: candidates against exactly this, and may shrink it when two labels can no
#: longer clear each other.
LABEL_PT = 6.5

def _drawn(low: float, high: float, *, surface: str, inches: float):
    figure = plt.figure(figsize=(inches, inches * 0.75), dpi=100)
    axes = figure.add_subplot(111)
    axes.set_xlim(low, high)
    axes.set_ylim(low, high)
    if surface == "cell":
        # A grid cell's neighbours are other cells: half a gap each way.
        declare_room(axes, Room(0.02, 0.02, 0.02, 0.02))
    apply_smart_ticks(axes, label_pt=LABEL_PT)
    figure.canvas.draw()
    return figure, axes

def _labels(axis) -> list[str]:
    return [text.get_text() for text in axis.get_ticklabels() if text.get_text()]

@pytest.mark.parametrize("name,low,high", RANGES)
@pytest.mark.parametrize("surface,inches", (("panel", 6.0), ("cell", 1.2)))
def test_no_axis_prints_a_label_longer_than_its_own_range_needs(
    name: str, low: float, high: float, surface: str, inches: float
) -> None:
    figure, axes = _drawn(low, high, surface=surface, inches=inches)
    try:
        for which, axis in (("x", axes.xaxis), ("y", axes.yaxis)):
            labels = _labels(axis)
            assert labels, f"{name}: {which} drew no labels"
            longest = max(len(value) for value in labels)
            assert longest <= LONGEST_LABEL, (
                f"{name}: {which} labels {labels} beside "
                f"offset {axis.get_offset_text().get_text()!r}"
            )
    finally:
        plt.close(figure)

def test_a_y_axis_far_from_zero_takes_the_offset_too() -> None:
    """The regression itself: the width question is not asked of y."""

    figure, axes = _drawn(1000000.0, 1000000.4, surface="panel", inches=6.0)
    try:
        assert axes.yaxis.get_offset_text().get_text() == "+1000000"
        assert axes.xaxis.get_offset_text().get_text() == "+1000000"
    finally:
        plt.close(figure)

def test_labels_that_are_already_short_keep_their_coordinates() -> None:
    """An offset is a cost: it is paid only where it buys something."""

    figure, axes = _drawn(5180.0, 5220.0, surface="panel", inches=6.0)
    try:
        assert axes.yaxis.get_offset_text().get_text() == ""
        assert _labels(axes.yaxis)[0] == "5180"
    finally:
        plt.close(figure)

def test_a_scale_and_an_offset_stay_two_statements() -> None:
    """Run together, "x1e4+3.84e11" reads as arithmetic that means nothing."""

    figure, axes = _drawn(
        384227900000.0, 384228100000.0, surface="panel", inches=6.0
    )
    try:
        text = axes.yaxis.get_offset_text().get_text()
        assert "×1e" in text and "+" in text
        assert " " in text or "\n" in text, text
    finally:
        plt.close(figure)

@pytest.mark.parametrize("surface,inches", (("panel", 6.0), ("cell", 1.2)))
def test_the_offset_is_written_in_the_figures_own_corners(
    surface: str, inches: float
) -> None:
    """Both surfaces, both parts, inside the canvas.

    In axes fractions -- what this was, and what the reference GUI uses -- the
    x offset sits a tenth of the data box BELOW the axes.  That reference
    reserves a hundred-pixel bottom margin for it; a panel laid out to fit its
    own labels does not, so the two-line form was drawn off the canvas.
    """

    figure, axes = _drawn(
        384227900000.0, 384228100000.0, surface=surface, inches=inches
    )
    try:
        for axis, position in (
            (axes.xaxis, (0.995, 0.008)),
            (axes.yaxis, (0.008, 0.995)),
        ):
            offset = axis.get_offset_text()
            assert offset.get_text(), "a range this far from zero has an offset"
            assert offset.get_position() == position
            assert offset.get_transform() is figure.transFigure
            window = offset.get_window_extent(figure.canvas.get_renderer())
            assert figure.bbox.containsx(window.x0) and figure.bbox.containsx(
                window.x1
            ), f"{surface} {axis.axis_name} offset runs off the canvas: {window}"
            assert figure.bbox.containsy(window.y0) and figure.bbox.containsy(
                window.y1
            ), f"{surface} {axis.axis_name} offset runs off the canvas: {window}"
    finally:
        plt.close(figure)

def test_a_caller_says_where_it_draws_and_how_big_never_how_many() -> None:
    """The one fact a caller owns, and the two it does not.

    How many labels an axis carries follows from its extent, its room and
    its labels; a caller supplies only the size they are drawn at (required,
    because the policy prices against it).  What lies past the edge is the
    layout's to declare, never a caller's flag.
    """

    figure = plt.figure(figsize=(4.0, 3.0), dpi=100)
    try:
        axes = figure.add_subplot(111)
        with pytest.raises(TypeError):
            apply_smart_ticks(axes)
        with pytest.raises(TypeError):
            apply_smart_ticks(axes, label_pt=LABEL_PT, prune_edges=True)
        with pytest.raises(ValueError, match="at least two"):
            apply_declared_ticks(axes, "x", 1, label_pt=LABEL_PT)
        with pytest.raises(ValueError, match="label_pt"):
            apply_smart_ticks(axes, label_pt=0.0)
        with pytest.raises(ValueError, match="which"):
            apply_smart_ticks(axes, "diagonal", label_pt=LABEL_PT)
    finally:
        plt.close(figure)

def test_identical_tick_input_reuses_layout_but_range_and_extent_do_not(
    monkeypatch,
) -> None:
    figure = plt.figure(figsize=(4.0, 3.0), dpi=100)
    try:
        axes = figure.add_subplot(111)
        axes.set_xlim(0.0, 10.0)
        apply_smart_ticks(axes, "x", label_pt=LABEL_PT)
        locator = axes.xaxis.get_major_locator()
        assert isinstance(locator, SmartOffsetLocator)
        native = locator._unit
        calls = 0

        def counted(low: float, high: float):
            nonlocal calls
            calls += 1
            return native(low, high)

        monkeypatch.setattr(locator, "_unit", counted)
        forward = locator.tick_values(0.0, 10.0)
        assert locator.tick_values(0.0, 10.0) is forward
        assert calls == 1

        reverse = locator.tick_values(10.0, 0.0)
        assert reverse == list(reversed(forward))
        assert calls == 2

        axes.set_position((0.2, 0.2, 0.35, 0.6))
        locator.tick_values(10.0, 0.0)
        assert calls == 3
    finally:
        plt.close(figure)

def test_a_settled_fine_unit_is_rejected_before_a_large_range_is_enumerated(
    monkeypatch,
) -> None:
    """A representation-scale jump must not enumerate the old tick lattice."""

    locator = SmartOffsetLocator(max_ticks=8, label_pt=LABEL_PT)
    locator.tick_values(0.0, 1.0)
    native_range = range

    def bounded_range(*args: int):
        candidate = native_range(*args)
        assert len(candidate) <= locator.max_ticks + 1
        return candidate

    monkeypatch.setattr(ticks_module, "range", bounded_range, raising=False)
    values = locator.tick_values(0.0, 4_200_000.0)
    fresh = SmartOffsetLocator(max_ticks=8, label_pt=LABEL_PT)

    assert 2 <= len(values) <= locator.max_ticks
    assert values == fresh.tick_values(0.0, 4_200_000.0)

def _rendered_labels(renderer):
    """Every label a real render drew, with the box it drew it in."""

    figure = renderer.figure
    figure.canvas.draw()
    canvas = figure.canvas.get_renderer()
    for role, axes_list in renderer.axes.items():
        for index, axes in enumerate(axes_list):
            if not axes.get_visible():
                continue
            for name, axis in (("x", axes.xaxis), ("y", axes.yaxis)):
                drawn = [
                    (label.get_text(), float(label.get_fontsize()),
                     label.get_window_extent(canvas))
                    for label in axis.get_ticklabels()
                    if label.get_text() and label.get_visible()
                ]
                if drawn:
                    yield f"{role}:{index}:{name}", name, drawn

def _frame_snapshot(height: int, width: int, *, peak: float):
    """A camera-like frame: dark background, a bright spot, noise -- so the
    rail's bound is the four or five digits a real frame gives it."""

    import numpy as np

    from data_factory import (
        axis,
        make_dataset_schema,
        make_snapshot,
        mapped_domain_from_columns,
        repeat_domain,
    )
    from zlc_data import READOUT_EVENT, SPATIAL_X, SPATIAL_Y

    rng = np.random.default_rng(7)
    yy, xx = np.mgrid[0:height, 0:width]
    spot = peak * np.exp(
        -(((xx - width / 2) ** 2 + (yy - height / 2) ** 2) / (2 * (max(width, height) / 8) ** 2))
    )
    values = 100.0 + spot + rng.normal(0.0, 8.0, size=(height, width))
    frames = mapped_domain_from_columns({"frame": [0]}, roles={"frame": READOUT_EVENT})
    schema = make_dataset_schema(
        repeat_domain(size=1),
        frames,
        cell_axes=(
            axis("spatial-y", size=height, role=SPATIAL_Y),
            axis("spatial-x", size=width, role=SPATIAL_X),
        ),
        dtype=np.float32,
    )
    return make_snapshot(schema, values[None, None].astype(np.float32), 0)


def _frames_snapshot(count: int, height: int, width: int):
    import numpy as np

    from data_factory import (
        axis,
        make_dataset_schema,
        make_snapshot,
        mapped_domain_from_columns,
        repeat_domain,
    )

    schema = make_dataset_schema(
        repeat_domain(size=1),
        mapped_domain_from_columns({"frame": [float(i) for i in range(count)]}),
        cell_axes=(axis("spatial-y", size=height), axis("spatial-x", size=width)),
        dtype=np.float32,
    )
    values = np.zeros((1, count, height, width), dtype=np.float32)
    values[..., ::7] = 1.0
    return make_snapshot(schema, values, 0)


def _shots_snapshot(shots: int, sites: int):
    import numpy as np

    from data_factory import (
        make_dataset_schema,
        make_snapshot,
        mapped_domain_from_columns,
        repeat_domain,
    )

    rng = np.random.default_rng(3)
    schema = make_dataset_schema(
        repeat_domain(size=shots),
        mapped_domain_from_columns({"site": np.arange(float(sites))}),
    )
    return make_snapshot(schema, rng.normal(size=(shots, sites)) * 1000.0 + 5000.0, 1)


def _surface_sessions(case: str, preset: str, dpr: float):
    """Every kind this bench opens, through the product's own session."""

    from zlc_plot import (
        AxisRef,
        CurvePlot,
        FacetGridPlot,
        HistogramPlot,
        ImagePlot,
        RollingPlot,
    )
    from zlc_plot.session import PlotSession

    image = ImagePlot(AxisRef.cell_data("spatial-x"), AxisRef.cell_data("spatial-y"))
    if case == "image-512":
        return PlotSession(_frame_snapshot(512, 512, peak=900.0), image, size=preset, device_pixel_ratio=dpr)
    if case == "image-roi":
        return PlotSession(_frame_snapshot(60, 83, peak=900.0), image, size=preset, device_pixel_ratio=dpr)
    if case == "image-qcmos":
        return PlotSession(_frame_snapshot(2304, 4096, peak=900.0), image, size=preset, device_pixel_ratio=dpr)
    if case == "rolling":
        return PlotSession(_shots_snapshot(400, 8), RollingPlot(group=AxisRef.point("site")), size=preset, device_pixel_ratio=dpr)
    if case == "curve":
        return PlotSession(_shots_snapshot(6, 12), CurvePlot(AxisRef.point("site")), size=preset, device_pixel_ratio=dpr)
    if case == "histogram":
        return PlotSession(_shots_snapshot(40, 12), HistogramPlot(), size=preset, device_pixel_ratio=dpr)
    if case == "histogram-log":
        return PlotSession(
            _shots_snapshot(40, 12), HistogramPlot(), size=preset,
            parameters={"log_y": True}, device_pixel_ratio=dpr,
        )
    if case in ("facet", "facet-focus"):
        session = PlotSession(
            _frames_snapshot(9, 60, 83),
            FacetGridPlot(AxisRef.point("frame"), image),
            size=preset,
            device_pixel_ratio=dpr,
        )
        if case == "facet-focus":
            session.focus_facet(4)
        return session
    raise ValueError(case)


_CASES = (
    "image-512", "image-roi", "image-qcmos", "rolling", "curve",
    "histogram", "histogram-log", "facet", "facet-focus",
)


@pytest.mark.parametrize("dpr", (1.0, 3.0), ids=("dpr1", "dpr3"))
@pytest.mark.parametrize("preset", ("1x2", "2x2", "4x4", "8x8"))
@pytest.mark.parametrize("case", _CASES)
def test_every_surface_prints_two_labels_inside_its_room(
    case: str, preset: str, dpr: float
) -> None:
    """The law, measured on what was DRAWN through the product's own path.

    Every kind this bench opens, at the presets an operator picks, at the
    bench's own DPR: every labelled axis carries two distinct labels -- a
    count rail its bound, and its zero only when it clears -- no label of
    one axis prints over a label of another, none leaves the figure, none
    is smaller than the readable floor, and within an axis two labels come
    closer than a digit only when the axis has reached that floor.
    """

    from zlc_plot.ticks import _label_size_pt

    session = _surface_sessions(case, preset, dpr)
    try:
        session.rgba()
        renderer = session._renderer
        figure = renderer.figure
        dpi = float(figure.dpi)
        canvas_box = figure.bbox
        boxes = []
        for where, name, drawn in _rendered_labels(renderer):
            texts = [text for text, _pt, _box in drawn]
            role = where.split(":")[0]
            if role == "distribution" and name == "x":
                axes = renderer.axes["distribution"][int(where.split(":")[1])]
                bound = float(axes.get_xlim()[1])
                spellings = {count_label(bound, length) for length in (None, 4, 3)}
                assert texts and texts[-1] in spellings, (where, texts, bound)
            else:
                assert len(set(texts)) >= 2, (where, texts)
            for text, size_pt, box in drawn:
                assert size_pt >= MIN_TICK_LABEL_PT - 1e-6, (where, text, size_pt)
                assert canvas_box.contains(box.x0, box.y0) and canvas_box.contains(box.x1, box.y1), (
                    f"{where} {text!r} leaves the figure: {box}"
                )
                boxes.append((where, text, box))
            if len(drawn) >= 2:
                size_pt = max(pt for _text, pt, _box in drawn)
                required = _label_size_pt("0", size_pt)[0]
                spans = sorted(
                    (box.x0, box.x1) if name == "x" else (box.y0, box.y1)
                    for _text, _pt, box in drawn
                )
                clear = min(
                    (second[0] - first[1]) / dpi * 72.0
                    for first, second in zip(spans, spans[1:])
                )
                if clear + 1e-6 < required:
                    assert size_pt <= MIN_TICK_LABEL_PT + 1e-6, (
                        f"{where}: {len(drawn)} labels at {size_pt}pt, "
                        f"clear {clear:.2f} < {required:.2f}"
                    )
        for first in range(len(boxes)):
            for second in range(first + 1, len(boxes)):
                one, two = boxes[first], boxes[second]
                if one[0] == two[0]:
                    continue
                assert not one[2].overlaps(two[2]), (
                    f"{one[0]} {one[1]!r} prints over {two[0]} {two[1]!r}"
                )
        if case == "rolling":
            history = [text for where, _n, drawn in _rendered_labels(renderer)
                       if where.startswith("history") and where.endswith(":x")
                       for text, _pt, _box in drawn]
            assert "0" in history, "the newest shot is the anchor of the shot axis"
    finally:
        session.close()


def _strip(width_pt: float, *, room_left_pt: float, room_right_pt: float):
    """A bare axes ``width_pt`` wide with the room a rail gets beside an image."""

    figure_pt = 216.0
    figure = plt.figure(figsize=(figure_pt / 72.0, 1.0), dpi=100)
    left = 0.5
    axes = figure.add_axes((left, 0.3, width_pt / figure_pt, 0.4))
    # A rail names no y coordinate; a bare axes would, and its labels would
    # then own the corner the x labels are priced against.
    axes.tick_params(axis="y", labelleft=False)
    declare_room(
        axes,
        Room(room_left_pt / figure_pt, room_right_pt / figure_pt, 0.2, 0.2),
    )
    return figure, axes


def _drawn_x(figure, axes):
    figure.canvas.draw()
    canvas = figure.canvas.get_renderer()
    return [
        (label.get_text(), float(label.get_fontsize()), label.get_window_extent(canvas))
        for label in axes.xaxis.get_ticklabels()
        if label.get_text()
    ]


@pytest.mark.parametrize(
    "width_pt,expect_zero,at_least_pt",
    (
        (46.2, True, LABEL_PT),
        (23.2, False, LABEL_PT),
        (11.5, False, MIN_TICK_LABEL_PT),
        (5.8, False, MIN_TICK_LABEL_PT),
    ),
    ids=("8x8", "4x4", "2x2", "1x2"),
)
def test_a_declared_rail_prints_its_bound_before_its_zero(
    width_pt: float, expect_zero: bool, at_least_pt: float
) -> None:
    """The bound is the information; the zero is the one optional label.

    The rail widths of the 8x8, 4x4, 2x2 and 1x2 presets, with half a gap
    on each side: the widest prints both; 4x4 has no room for a zero a
    digit clear of a five-digit bound and prints the bound at full size;
    2x2 prints it smaller; 1x2 prints a shorter spelling -- never the zero
    alone, which is what a width test used to leave.
    """

    figure, axes = _strip(width_pt, room_left_pt=1.4, room_right_pt=1.4)
    try:
        axes.set_xlim(0.0, 28608.0)
        apply_declared_ticks(axes, "x", 2, label_pt=LABEL_PT)
        drawn = _drawn_x(figure, axes)
        texts = [text for text, _pt, _box in drawn]
        assert texts[-1] in {count_label(28608.0, n) for n in (None, 4, 3)}, texts
        assert ("0" in texts) == expect_zero, texts
        size = drawn[-1][1]
        assert size >= at_least_pt - 1e-6, size
        box = drawn[-1][2]
        dots = figure.dpi / 72.0
        assert box.x0 >= axes.bbox.x0 - 1.4 * dots - 0.5
        assert box.x1 <= axes.bbox.x1 + 1.4 * dots + 0.5
    finally:
        plt.close(figure)


@pytest.mark.parametrize("width_pt,room_right_pt", ((43.2, 0.73), (86.4, 1.45)), ids=("1x2", "2x2"))
@pytest.mark.parametrize("high", (2048.0, 50.0, 1000.0))
def test_no_coordinate_label_reaches_past_its_room(
    width_pt: float, room_right_pt: float, high: float
) -> None:
    """An image's x axis ends at a rail; its last label ends there too.

    Pruning an edge tick and then putting it back when fewer than two were
    left is how "2000" hung eight points over the image's right edge onto
    the rail's zero.  Now the label is anchored inward when it must be, and
    a tick whose label cannot be placed is simply not one of the axis's.
    """

    figure, axes = _strip(width_pt, room_left_pt=26.4, room_right_pt=room_right_pt)
    try:
        axes.set_xlim(0.0, high)
        apply_smart_ticks(axes, "x", label_pt=LABEL_PT)
        drawn = _drawn_x(figure, axes)
        assert len(drawn) >= 2, drawn
        dots = figure.dpi / 72.0
        for text, _pt, box in drawn:
            assert box.x1 <= axes.bbox.x1 + room_right_pt * dots + 0.5, (text, box)
    finally:
        plt.close(figure)


def test_the_ladder_takes_fewer_before_smaller() -> None:
    """At the drawn size, fewer labels; smaller ones only at the floor of two."""

    figure, axes = _strip(30.0, room_left_pt=26.4, room_right_pt=26.4)
    try:
        axes.set_xlim(0.0, 7.0)
        apply_smart_ticks(axes, "x", label_pt=LABEL_PT)
        drawn = _drawn_x(figure, axes)
        assert len(drawn) >= 2
        assert all(abs(pt - LABEL_PT) < 1e-6 for _t, pt, _b in drawn), drawn
    finally:
        plt.close(figure)
    figure, axes = _strip(9.0, room_left_pt=0.5, room_right_pt=0.5)
    try:
        axes.set_xlim(0.0, 7.0)
        apply_smart_ticks(axes, "x", label_pt=LABEL_PT)
        drawn = _drawn_x(figure, axes)
        assert len(drawn) == 2, drawn
        assert all(pt < LABEL_PT for _t, pt, _b in drawn), drawn
        assert all(pt >= MIN_TICK_LABEL_PT - 1e-6 for _t, pt, _b in drawn), drawn
    finally:
        plt.close(figure)


@pytest.mark.parametrize("name,low,high", RANGES)
@pytest.mark.parametrize(
    "inches", (0.3, 0.45, 0.6, 0.9, 1.2), ids=lambda v: f"{v}in"
)
def test_every_axis_shows_at_least_two_labels_even_as_a_tiny_cell(
    name, low, high, inches
) -> None:
    """A single label names a point, not a scale.

    The floor of two holds on every linear axis at every size this bench
    can shrink a facet cell to -- crowding a neighbour is the accepted
    price, an image cell whose one label reads "0" is not.  (The one
    surface allowed a single label is a zero-anchored counts rail, whose
    bound alone is its information; it is guarded below.)
    """

    figure, axes = _drawn(low, high, surface="cell", inches=inches)
    try:
        for axis in (axes.xaxis, axes.yaxis):
            shown = [
                text.get_text()
                for text in axis.get_ticklabels()
                if text.get_text()
            ]
            assert len(shown) >= 2, (name, inches, axis.axis_name, shown)
    finally:
        plt.close(figure)
