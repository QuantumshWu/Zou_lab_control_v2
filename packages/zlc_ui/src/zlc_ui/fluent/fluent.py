"""Reusable PyQt Fluent-style widgets and window chrome.

The visual constants and widget shapes are shared frontend primitives with no
application state or domain model.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Sequence
import logging
import sys
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_EVEN, localcontext
import math
import re
import weakref

from PyQt5 import QtCore, QtGui, QtWidgets

from qframelesswindow import FramelessWindow, StandardTitleBar

from zlc_data.units import (
    DEFAULT_UNITS,
    UnitError,
    decimal_of,
    format_quantity,
)

from ..qt import ensure_qt_app
from ..status import STATUS_SEVERITIES

from .style import (
    ACCENT,
    ACCENT_HOVER,
    ACCENT_STRONG,
    ACCENT_TINT,
    DERIVED_INK,
    DISABLED_INK,
    MUTED_INK,
    NEUTRAL_FILL,
    SELECTED,
    SELECTED_HOVER,
    SELECTION_EDGE,
    AUTO_SCALE_BASIS,
    BUTTON_PADDING_H,
    BUTTON_TINT,
    AUTO_SCALE_MARGIN,
    BG,
    CARD_PAD,
    CARD_TITLE_PX,
    COMBO_TRI_SIZE,
    COMBO_WIDTH,
    DIVIDER,
    EDIT_PADDING_H,
    FLUENT_SCALE_MAX,
    FLUENT_SCALE_MIN,
    FONT,
    FONT_SIZE,
    GREEN,
    GRAPHITE,
    GREY,
    HOVER,
    MUTED_LABEL_STYLE,
    ORANGE,
    ORANGE_DARK,
    PADDING_V,
    PLACEHOLDER,
    RADIUS,
    RED,
    RED_DARK,
    RED_TINT,
    TEXT,
    TITLE_ICON_GAP,
    TITLE_ICON_RATIO,
    TITLE_LEFT_INSET,
    WINDOW_FALLBACK_MIN_PX,
    WINDOW_FALLBACK_PX,
    WINDOW_MARGIN_FLOOR_PX,
    WINDOW_MARGIN_PX,
    WINDOW_MAX_FLOOR_PX,
    WINDOW_MIN_FLOOR_PX,
    WINDOW_MIN_PX,
    WINDOW_PAD,
    WINDOW_SCREEN_FRACTION,
    WINDOW_TITLEBAR_FLOOR_PX,
    WINDOW_TITLEBAR_PX,
    SURFACE,)


def window_pad(mult: float = 1.0) -> int:
    """The window padding / card-gap at ``mult`` x the single :data:`WINDOW_PAD` unit, display-scaled."""
    return scaled_px(round(WINDOW_PAD * mult))

_FLUENT_SCALE = 1.0
_FLUENT_SCALE_INITIALIZED = False
# Matches one float token; used by align_to_resolution to snap numbers inside a value.
_FLOAT_TOKEN_RE = re.compile(r"(?<![A-Za-z_])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")

#: The integers Qt's own integer controls and validators can hold.  A Python
#: int has no width; this is the one place that says where Qt's ends, so a
#: form choosing between a spin box and a text field and a validator choosing
#: between Qt's bound check and its own agree about it.
QT_INT_RANGE = (-(2**31), 2**31 - 1)
_INT_TEXT_RE = QtCore.QRegularExpression(r"[+-]?\d*")


# The fluent design-basis window: the automatic scale fits this logical size
# into the primary screen's available geometry.  EVERY GUI window must resolve
# its automatic scale through this ONE rule -- two windows on the same screen
# must end up with identical control sizes across every hosted view.
def resolve_fluent_auto_scale(app: QtWidgets.QApplication | None = None) -> float:
    """The shared automatic fluent scale for the current primary screen."""

    app = app or QtWidgets.QApplication.instance()
    screen = app.primaryScreen() if app is not None else None
    if screen is None:
        return 1.0
    available = screen.availableGeometry()
    basis_w, basis_h = AUTO_SCALE_BASIS
    margin_w, margin_h = AUTO_SCALE_MARGIN
    return min(
        1.0,
        max(0.1, (available.width() - margin_w) / basis_w),
        max(0.1, (available.height() - margin_h) / basis_h),
    )


def set_fluent_scale(scale: float | None = None) -> float:
    """Set the PROCESS-GLOBAL scale used by subsequently created Fluent widgets.

    ``None`` means AUTOMATIC: resolve from the primary screen via
    :func:`resolve_fluent_auto_scale` (never a silent 1.0 -- a fixed fallback
    is how two GUIs ended up with different control sizes on the same screen).

    The fluent scale is ONE process-wide value BY DESIGN: every GUI window on a
    screen must agree on control sizes (the scale-sharing contract), so there is
    no per-window scale, and the app font follows it."""

    global _FLUENT_SCALE, _FLUENT_SCALE_INITIALIZED
    if scale is None:
        scale = resolve_fluent_auto_scale()
    _FLUENT_SCALE = max(FLUENT_SCALE_MIN, min(FLUENT_SCALE_MAX, float(scale)))
    _FLUENT_SCALE_INITIALIZED = True
    _apply_app_font()
    return _FLUENT_SCALE


def ensure_fluent_scale() -> float:
    """Initialize the shared Fluent scale before any caller constructs widgets.

    ``ensure_qt_app`` calls this hook after creating the QApplication.  Keeping
    the initialization at the Qt entry point prevents a body from being built
    at the default scale and only later wrapped by ``launch_fluent_window``.
    An explicit ``set_fluent_scale(...)`` remains authoritative for the rest of
    the process.
    """

    if not _FLUENT_SCALE_INITIALIZED:
        return set_fluent_scale(None)
    _apply_app_font()
    return _FLUENT_SCALE


def _apply_app_font() -> None:
    """The one writer of the application font: Segoe UI at the scaled size.

    Qt broadcasts every ``setFont`` to every widget of every open window,
    equal font or not, and each window and dialog opened asks for the font
    again -- so an unchanged font is not written.
    """

    app = QtWidgets.QApplication.instance()
    if app is None:
        return
    font = QtGui.QFont(FONT, fluent_font_size())
    if app.font() != font:
        app.setFont(font)


def scaled_px(value: int | float, *, minimum: int = 1) -> int:
    return max(int(minimum), int(round(float(value) * _FLUENT_SCALE)))


def control_height() -> int:
    """The ONE height of a control in a row -- a field, a combo, a button,
    a switch, a grid card's face, a section title -- display-scaled."""

    return scaled_px(30, minimum=22)


# ---------------------------------------------------------------- frame geometry
#
# THE ONE GEOMETRY RULE for a frame with parts inside it (a spin box's
# stepper, a combo's drop column, a table's rules, a card's title band):
# every inner part is placed FROM THE FRAME, snapped to device pixels --
# never from a sub-control rectangle nudged by a magic offset.  Offsets tuned
# at one display scale broke at the next: the spin divider stopped half a
# pixel under the top edge and ran into the bottom one at every DPR.
#
#   * A divider is a FRAME SEGMENT: a filled rectangle, a whole number of
#     device pixels wide, in the frame's current edge colour, antialiasing
#     off, running between the edge's own rows (frame_inner_rect) -- so its
#     T-joints have no seam, and at a fractional ratio, where Qt draws the
#     top and bottom edge part-covered, it does not cross them as a darker
#     nub.
#   * A fill (a hover tint, a band) tiles exactly the room between the
#     dividers and the frame's inner edges, concentric at a rounded corner.
#   * A one-edge stroke is a filled rectangle or a whole-device-pixel pen on
#     a half-device-pixel centre with a flat cap.
#   * A round thing has an even size.

#: Logical px of every field/card frame edge.  Sheets write
#: ``f"{FIELD_BORDER_PX}px"`` and Qt's high-DPI scaling makes it
#: ``FIELD_BORDER_PX * devicePixelRatioF()`` device px; painted code reads the
#: same constant.  Nothing else defines the edge width.
FIELD_BORDER_PX = 1
#: QLineEditPrivate::horizontalMargin: the text inset Qt keeps inside a line
#: edit's contents rect on the left and right (logical px, never scaled).
_QLINEEDIT_HMARGIN = 2


def scaled_even_px(value: float, *, minimum: int = 2) -> int:
    """``scaled_px`` rounded up to an even number: a circle drawn as a
    ``size / 2`` radius is only round on a whole-pixel centre."""

    size = scaled_px(value, minimum=minimum)
    return size + (size & 1)


def device_snap(
    widget: QtWidgets.QWidget, value: float, mode: str = "round", *, axis: str = "x",
    grid: float = 1.0,
) -> float:
    """``value`` (a logical ``axis`` coordinate in ``widget``) moved onto a
    device-pixel boundary of the window it is painted into -- or, with
    ``grid=0.5``, onto a boundary or a pixel's middle, where a part with
    an odd number of device pixels is centred.

    At a whole device-pixel ratio every logical pixel is one; at a
    fractional one (1.25, 1.5) a widget's own origin can fall inside a
    device pixel, so the boundary is found in window coordinates."""

    ratio = float(widget.devicePixelRatioF()) / float(grid)
    snap = {"round": lambda v: math.floor(v + 0.5), "floor": math.floor, "ceil": math.ceil}[mode]
    offset = 0.0
    if ratio != int(ratio):
        window = widget.window()
        if window is not widget:
            origin = widget.mapTo(window, QtCore.QPoint(0, 0))
            offset = float(origin.x() if axis == "x" else origin.y())
    return snap((float(value) + offset) * ratio) / ratio - offset


def frame_border_px(widget: QtWidgets.QWidget) -> float:
    """The frame edge in LOGICAL px, as a whole number (>= 1) of device px."""

    ratio = float(widget.devicePixelRatioF())
    return max(1, round(FIELD_BORDER_PX * ratio)) / ratio


def frame_inner_rect(widget: QtWidgets.QWidget) -> QtCore.QRectF:
    """``widget.rect()`` minus its edge, snapped INWARD to the first device
    pixel the edge leaves untouched (ceil top/left, floor bottom/right)."""

    edge = FIELD_BORDER_PX
    left = device_snap(widget, edge, "ceil")
    top = device_snap(widget, edge, "ceil", axis="y")
    right = device_snap(widget, widget.width() - edge, "floor")
    bottom = device_snap(widget, widget.height() - edge, "floor", axis="y")
    return QtCore.QRectF(left, top, right - left, bottom - top)


#: The edge a field sheet gives a resting / focused field: the field sheets
#: (``fluent_spinbox_stylesheet``, ``FluentLineEdit``) are written from these.
_FIELD_EDGE_REST = PLACEHOLDER
_FIELD_EDGE_FOCUS = ACCENT


def frame_stroke_rect(widget: QtWidgets.QWidget, rect: QtCore.QRectF | None = None) -> QtCore.QRectF:
    """``rect`` (default: the widget's) inset by half the frame edge, so a
    pen of ``frame_border_px`` stroked on it lies wholly inside."""

    half = frame_border_px(widget) / 2.0
    return QtCore.QRectF(widget.rect() if rect is None else rect).adjusted(half, half, -half, -half)


def frame_pen(widget: QtWidgets.QWidget, colour) -> QtGui.QPen:
    """The pen a painted frame edge is drawn with: ``frame_border_px``
    wide, the same edge the field sheets write as ``FIELD_BORDER_PX``."""

    pen = QtGui.QPen(QtGui.QColor(colour))
    pen.setWidthF(frame_border_px(widget))
    return pen


def stepper_column(widget: QtWidgets.QWidget) -> QtCore.QRectF:
    """The right-hand stepper / drop cell inside the edge.

    One source for a spin box's tint and chevrons AND a combo's chevron
    column, so the two chevrons of a form column share a centre."""

    inner = frame_inner_rect(widget)
    right = widget.width() - FIELD_BORDER_PX
    left = right - scaled_px(COMBO_WIDTH)
    return QtCore.QRectF(left, inner.top(), right - left, inner.height())


def fluent_stepper_extent() -> int:
    """The logical width a spin box's stepper adds over a plain field: the
    margin column its sheet reserves, the cell and the edge beside it."""

    return scaled_px(COMBO_WIDTH) + 2 * FIELD_BORDER_PX


def paint_chevron(
    painter: QtGui.QPainter,
    widget: QtWidgets.QWidget,
    centre: QtCore.QPointF,
    *,
    up: bool,
    colour: QtGui.QColor,
) -> None:
    """The one chevron a spin box and a combo draw: one size, one pen, its
    ink centred on ``centre`` (x on a device pixel, or on a pixel's middle
    when the stroke is an odd number of device pixels wide; y on a device
    pixel or a pixel's middle, so a chevron centred on a frame with an odd
    number of device rows inside it is centred on them, not half a pixel
    low)."""

    ratio = float(widget.devicePixelRatioF())
    half = scaled_px(COMBO_TRI_SIZE) / 2.0
    width = max(1.0, scaled_px(3) / 2.0)
    cx = device_snap(widget, centre.x())
    if round(width * ratio) % 2:
        cx += 0.5 / ratio
    cy = device_snap(widget, centre.y(), axis="y", grid=0.5)
    sign = -1.0 if up else 1.0
    pen = QtGui.QPen(colour)
    pen.setWidthF(width)
    pen.setCapStyle(QtCore.Qt.RoundCap)
    pen.setJoinStyle(QtCore.Qt.RoundJoin)
    painter.save()
    painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
    painter.setPen(pen)
    painter.setBrush(QtCore.Qt.NoBrush)
    painter.drawPolyline(QtGui.QPolygonF([
        QtCore.QPointF(cx - half, cy - sign * half / 2.0),
        QtCore.QPointF(cx, cy + sign * half / 2.0),
        QtCore.QPointF(cx + half, cy - sign * half / 2.0),
    ]))
    painter.restore()


def paint_centred_ink(
    painter: QtGui.QPainter,
    widget: QtWidgets.QWidget,
    rect: QtCore.QRectF,
    font: QtGui.QFont,
    text: str,
    colour,
) -> QtCore.QRectF:
    """Fill ``text`` so its INK is centred on ``rect``; return the ink's box.

    The one way a small symbol or label is centred inside a part of a
    field, a card or a button (a ×, a ⋯, a step, a badge letter).  Drawn
    with ``AlignCenter`` it is centred on the font's LINE BOX, which puts a
    × 2.5 px and a ⋯ 1.2 px below the middle of its face, and a digit or a
    capital a pixel low.  The text is laid out as a path and moved so the
    centre of its own bounding box lands on ``rect``'s, by a translation
    snapped to the device pixels of ``widget`` (the surface the painter
    paints on) -- the same glyph then renders alike wherever it is, and the
    eye sees it centred to the pixel.
    """

    path = QtGui.QPainterPath()
    path.addText(0.0, 0.0, font, str(text))
    ink = path.boundingRect()
    centre = QtCore.QRectF(rect).center()
    dx = device_snap(widget, centre.x() - ink.center().x())
    dy = device_snap(widget, centre.y() - ink.center().y(), axis="y")
    painter.save()
    painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
    painter.translate(dx, dy)
    painter.fillPath(path, QtGui.QColor(colour))
    painter.restore()
    return ink.translated(dx, dy)


def centred_line_top(top: float, height: float, font: QtGui.QFont) -> int:
    """The top of the ONE line box that centres ``font``'s capitals on the
    span ``top .. top + height``, in whole logical px: where a field's
    text, a row label, a button's label and a combo's collapsed text all
    put their line, so a row of them shares one baseline.

    Qt centres the line BOX, by two rules: a label or a button on the
    box's fractional height, a QLineEdit on its integer height with the odd
    pixel above.  A field's text sat two device pixels under the label
    beside it, and both sat low, because Segoe UI's ascent holds the room
    for accents: the capitals of a field were 1.3 px under its middle.  The
    capitals -- and the digits, which are as tall -- are what the eye
    centres.  Whole pixels, because a QLineEdit can put its line only on
    one: everything that paints its line on the same integer top, over the
    same ascent, lands on the same device row at every ratio."""

    metrics = QtGui.QFontMetricsF(font)
    centre = float(top) + float(height) / 2.0
    return math.floor(centre + metrics.capHeight() / 2.0 - metrics.ascent() + 0.5)


def _available_geometry_at(
    point: QtCore.QPoint, *, widget: QtWidgets.QWidget | None = None
) -> QtCore.QRect | None:
    """The usable area of the screen a point falls on, or None if there is none.

    "Where may a floating surface land" is ONE question -- a combo popup, a
    modal card -- and it had grown two separate walks from a point to a
    screen to its available geometry.  A third was about to be written for
    the modal that opens off the edge of the monitor; that is what the
    copies cost.

    ``widget`` is asked only when the point is on no screen at all: a
    window dragged past every edge still belongs to one.
    """

    application = QtWidgets.QApplication.instance()
    screen_at = getattr(application, "screenAt", None) if application is not None else None
    screen = screen_at(point) if callable(screen_at) else None
    if screen is None and widget is not None and hasattr(widget, "screen"):
        screen = widget.screen()
    if screen is None and application is not None:
        screen = application.primaryScreen()
    return None if screen is None else screen.availableGeometry()


def _placed_on_screen(
    wanted: QtCore.QPoint,
    size: QtCore.QSize,
    *,
    widget: QtWidgets.QWidget | None = None,
) -> QtCore.QPoint:
    """``wanted``, moved the least it takes to put the whole window on a screen.

    A platform window manager never opens a window off the monitor.  Every
    window this project places by hand -- a modal card centred on its
    parent, a dialog window centred on the window that opened it -- can,
    because a parent near an edge has its centre near that edge.  Modal is
    what makes that fatal: nothing else takes a click, so a window past the
    edge is one nobody can answer or move.
    """

    available = _available_geometry_at(wanted, widget=widget)
    if available is None:
        return wanted
    return QtCore.QPoint(
        min(
            max(wanted.x(), available.left()),
            max(available.left(), available.right() - size.width() + 1),
        ),
        min(
            max(wanted.y(), available.top()),
            max(available.top(), available.bottom() - size.height() + 1),
        ),
    )


def screen_fit_window_size(window_ratio: float) -> QtCore.QSize:
    """Initial window size for a top-level GUI, fit to the primary screen.

    Every hosted view shares this ONE rule (it was duplicated
    verbatim in callers, so a change in one silently diverged
    the other, the same failure class as the scale rule).  ``window_ratio`` is the
    fraction of the screen the window aims for; the result is clamped to a usable
    minimum and to what the screen can actually show."""

    app = QtWidgets.QApplication.instance()
    screen = app.primaryScreen() if app is not None else None
    if screen is None:
        return QtCore.QSize(
            scaled_px(WINDOW_FALLBACK_PX[0], minimum=WINDOW_FALLBACK_MIN_PX[0]),
            scaled_px(WINDOW_FALLBACK_PX[1], minimum=WINDOW_FALLBACK_MIN_PX[1]))
    available = screen.availableGeometry()
    titlebar_allowance = scaled_px(WINDOW_TITLEBAR_PX, minimum=WINDOW_TITLEBAR_FLOOR_PX)
    margin_w = scaled_px(WINDOW_MARGIN_PX[0], minimum=WINDOW_MARGIN_FLOOR_PX[0])
    margin_h = scaled_px(WINDOW_MARGIN_PX[1], minimum=WINDOW_MARGIN_FLOOR_PX[1])
    max_w = max(WINDOW_MAX_FLOOR_PX[0], available.width() - margin_w)
    max_h = max(WINDOW_MAX_FLOOR_PX[1], available.height() - titlebar_allowance - margin_h)
    min_w = min(scaled_px(WINDOW_MIN_PX[0], minimum=WINDOW_MIN_FLOOR_PX[0]), max_w)
    min_h = min(scaled_px(WINDOW_MIN_PX[1], minimum=WINDOW_MIN_FLOOR_PX[1]), max_h)
    desired_w = min(max_w, int(available.width() * window_ratio))
    desired_h = min(max_h, int(available.height() * window_ratio) - titlebar_allowance)
    return QtCore.QSize(max(min_w, desired_w), max(min_h, desired_h))


def frameless_content_top_margin(
    title_bar: QtWidgets.QWidget,
    *,
    fallback: int = 32,
) -> int:
    """Return the body margin required to clear native frameless chrome.

    The title bar is native Qt chrome and is not governed by the Fluent content
    scale.  A scaled design token is therefore only a fallback: when the
    native title bar reports a larger hint/actual height, the body must start
    below that real boundary.  Every frameless host uses this one rule.
    """

    if not isinstance(title_bar, QtWidgets.QWidget):
        raise TypeError("title_bar must be a QWidget")
    if int(fallback) < 0:
        raise ValueError("fallback must be non-negative")
    hint_height = int(title_bar.sizeHint().height())
    actual_height = int(title_bar.height())
    return max(int(fallback), hint_height, actual_height)


def center_window_on_primary_screen(window: QtWidgets.QWidget, app: QtWidgets.QApplication | None = None) -> None:
    """Centre a top-level GUI window on the primary screen's available area.

    Every hosted view shares this ONE rule -- it used to live in individual
    launchers, so windows opened inconsistently.  Single-sourced here next to
    ``screen_fit_window_size`` (which
    sizes the same windows) so the open POSITION cannot silently diverge from the open SIZE."""
    app = app or QtWidgets.QApplication.instance()
    screen = app.primaryScreen() if app is not None else None
    if screen is None:
        return
    available = screen.availableGeometry()
    frame = window.frameGeometry()
    frame.moveCenter(available.center())
    window.move(frame.topLeft())


def release_window(window: QtWidgets.QWidget) -> None:
    """Release one top-level window from the shared application retention registry.

    Current workbench windows close asynchronously and intentionally remain ordinary
    ``QWidget`` instances (they do not opt into ``WA_DeleteOnClose``).  Consequently a
    successful ``close()`` can hide a window without emitting ``destroyed``.  Their committed
    close path must call this function after worker/resource shutdown so the application does
    not retain an otherwise dead board forever.  The operation is idempotent.
    """

    app = QtWidgets.QApplication.instance()
    registry = None if app is None else getattr(app, "_zlc_retained_windows", None)
    if registry is None:
        return
    while True:
        try:
            registry.remove(window)
        except ValueError:
            return


def retain_window(window: QtWidgets.QWidget, *extras) -> None:
    """Keep a launched top-level GUI ``window`` alive until its committed close.

    The ``show_*`` launchers must hold a Python
    reference to the window they open -- a notebook cell often drops the return value, and with
    no reference the window would be garbage-collected mid-display.  Each launcher used to
    append to its OWN app-level list and never remove anything, so every reopened standalone
    window was retained forever.  This is the ONE registry
    (``app._zlc_retained_windows``) with the ONE pruning rule:

    * ``window.destroyed`` always removes the entry;
    * a genuine close (the ``closed`` signal, when the window has one) also removes it -- but
      ONLY for windows that really close on X.  ``hide_on_close`` windows hide instead and MUST
      stay retained, so a later session call restores the SAME interface.
    * asynchronous workbench windows explicitly call :func:`release_window` only after their
      cancellation/reap/worker shutdown protocol has reached its committed close.

    ``extras`` are for objects the caller needs kept alive that the window does NOT own; they
    ride the window's lifetime.  A widget the window parents (``FluentWindow(widget=...)``
    re-parents it) needs no extra retention -- do not double-retain children."""

    app = QtWidgets.QApplication.instance()
    if app is None:
        raise RuntimeError("retain_window needs a QApplication (call ensure_qt_app first).")
    registry = getattr(app, "_zlc_retained_windows", None)
    if registry is None:
        registry = []
        app._zlc_retained_windows = registry
    if extras:
        window._zlc_retained_extras = tuple(extras)   # live and die with the window
    if window not in registry:
        registry.append(window)
    window_ref = weakref.ref(window)

    def _prune(*_args) -> None:
        candidate = window_ref()
        if candidate is not None:
            release_window(candidate)

    window.destroyed.connect(_prune)
    closed = getattr(window, "closed", None)
    connect_closed = getattr(closed, "connect", None)
    if callable(connect_closed) and not getattr(window, "_hide_on_close", False):
        def _prune_committed_close(*_args) -> None:
            candidate = window_ref()
            if candidate is not None and getattr(
                candidate,
                "_zlc_close_committed",
                True,
            ):
                _prune()

        connect_closed(_prune_committed_close)


def fluent_font_size() -> int:
    return max(8, int(round(FONT_SIZE * _FLUENT_SCALE)))


def fluent_text_width(metrics: QtGui.QFontMetrics, text: str) -> int:
    """Return text width on old and new PyQt5 builds."""

    if hasattr(metrics, "horizontalAdvance"):
        return int(metrics.horizontalAdvance(text))
    return int(metrics.width(text))


def _radius() -> int:
    return scaled_px(RADIUS)


def popup_gap() -> int:
    """The few-px OUTER gap a Fluent popup sits OFF the control that anchored it -- the ONE source for
    every below-the-anchor Fluent surface: the FluentComboBox drop-down, the Setting FluentPopup, the
    FluentTabWidget overflow menu, the FluentLineEdit context menu.  Sharing this is what makes the
    ``...`` overflow list and a right-click menu open with the SAME breathing room as a combo drop-down,
    instead of one flush against its anchor and another floating off it."""
    return scaled_px(4, minimum=3)


def fluent_widget_stylesheet() -> str:
    return f'QWidget {{ background: {BG}; color: {TEXT}; font: {fluent_font_size()}pt "{FONT}"; }}'


def fluent_scrollbar_thickness() -> int:
    """The width (vertical bar) / height (horizontal bar) a Fluent scrollbar OCCUPIES, in device px.

    The ONE source for that dimension: ``fluent_scrollbar_stylesheet`` paints the bar this thick,
    AND any layout that must RESERVE the bar's gutter (so word-wrapped content never slides UNDER a
    vertical scrollbar -- the classic "text past the right edge" bug) reserves exactly this many px.
    Whenever the two disagreed, content underlapped the bar; keeping them on one constant closes it."""

    return scaled_px(12, minimum=10)


def fluent_scrollbar_stylesheet(
    selector: str = "QScrollBar", *, frame_width: int = 0
) -> str:
    """Return shared Fluent scrollbar CSS for scroll areas and popup lists."""

    thickness = fluent_scrollbar_thickness()
    # Bars sit INSIDE the frame, and the handle a margin inside its track,
    # so it never runs flush into a card's or a table's edge.  Concentric
    # corners need the inner radius, not the outer frame radius translated
    # inward by its border width.
    radius = max(0, _radius() - int(frame_width))
    return f"""
    {selector}:vertical {{
        background: transparent;
        border: none;
        border-radius: {radius}px;
        width: {thickness}px;
        margin: 0px;
    }}
    {selector}:horizontal {{
        background: transparent;
        border: none;
        border-radius: {radius}px;
        height: {thickness}px;
        margin: 0px;
    }}
    {selector}::handle:vertical, {selector}::handle:horizontal {{
        background: {NEUTRAL_FILL};
        border: none;
        border-radius: {radius}px;
        margin: {scaled_px(2)}px;
        min-height: {scaled_px(28)}px;
        min-width: {scaled_px(28)}px;
    }}
    {selector}::handle:vertical:hover, {selector}::handle:horizontal:hover {{
        background: {ACCENT};
    }}
    {selector}::add-line, {selector}::sub-line {{
        width: 0px;
        height: 0px;
        border: none;
        background: transparent;
    }}
    {selector}::add-page, {selector}::sub-page {{
        background: transparent;
    }}
    """


def status_dot_stylesheet(color: str, *, radius: int = 8) -> str:
    return f"background:{color}; border-radius:{scaled_px(radius)}px;"


# Fluent cards are delineated by a flat 1 px ``DIVIDER`` border, NOT a drop shadow.  A
# ``QGraphicsDropShadowEffect`` re-rasterised the whole widget subtree + re-ran a Gaussian blur on EVERY
# paint (~250 ms of a 35-site grid card's cold load at 3x dpr); a 1 px border costs nothing and reads the
# same "raised card" boundary.  Each card owns its edge: ``FluentGroupBox`` paints a CONTINUOUS border
# UNDER its title pill (so the pill never notches it); ``FluentTabWidget`` has NO border at all -- a plain
# white rounded card (any border read as boxing the pivot tabs, or as a line between the tabs and the
# content below); ``FluentFrame`` paints its edge via this one helper (a ``QFrame`` stylesheet border
# would cascade onto child QFrames -- painting does not), exactly like ``FluentPopup`` strokes its own edge.
def stroke_card_border(widget: QtWidgets.QWidget, *, radius: int | None = None) -> None:
    """Stroke a 1 px antialiased ``DIVIDER`` rounded border hugging ``widget``'s edge, in a paintEvent.

    Cascade-proof (painted, not a stylesheet rule) so a container that nests its own widget type does
    not border every child.  Call from the widget's ``paintEvent`` AFTER its body is painted."""
    _paint_fluent_card(
        widget, radius, QtGui.QColor(DIVIDER), QtGui.QColor(QtCore.Qt.transparent)
    )


@contextlib.contextmanager
def signals_blocked(*widgets: QtWidgets.QWidget | None):
    """Temporarily block Qt signals on each (non-None) widget, restoring after.

    Handy when updating many existing widgets' values in place (e.g. refreshing
    a form from a model) without each ``setText`` / ``setCurrentText`` firing a
    feedback signal -- the standard alternative to destroying + rebuilding the
    widgets.

    Restored LAST BLOCKED FIRST.  A widget named twice in one call is blocked
    twice, and the second block records "already blocked" as the state to
    restore; restoring in the order of blocking put that stale True back
    after the genuine False and left the widget silent for good.
    """

    saved = [(w, w.blockSignals(True)) for w in widgets if w is not None]
    try:
        yield
    finally:
        for widget, previous in reversed(saved):
            widget.blockSignals(previous)


def format_compact_number(value: float, *, digits: int = 12) -> str:
    """One snapped number, short enough to sit inside an authored expression.

    Deliberately NOT :func:`zlc_data.units.format_quantity`, which shows every
    digit a value has because a box an operator reads a device through may not
    round.  Snapping to a resolution produces 0.30000000000000004 where the
    author wrote 0.3, and pasting that back into their own scan program is not
    fidelity, it is noise.  Two jobs, two rules.
    """

    if not math.isfinite(float(value)):
        return str(value)
    text = f"{float(value):.{digits}g}"
    return text.replace("e+0", "e").replace("e+", "e")


def align_to_resolution(value: str | float, resolution: float, *, allow_any: bool = True) -> str:
    """Snap the numeric part of a simple value/x expression to ``resolution``."""

    if resolution is None or float(resolution) <= 0:
        return str(value)
    text = str(value).strip()
    if not text:
        return text

    def snap_number(match: re.Match[str]) -> str:
        token = match.group(0)
        try:
            number = float(token)
        except ValueError:
            return token
        snapped = round(number / float(resolution)) * float(resolution)
        if not allow_any and snapped <= 0:
            snapped = float(resolution)
        return format_compact_number(snapped)

    if any(char in text for char in "[](),"):
        return _FLOAT_TOKEN_RE.sub(snap_number, text)
    return snap_number(re.match(r".+", text))


class FluentStatusDot(QtWidgets.QLabel):
    #: Class attribute: set_color runs from __init__ before any assignment.
    _color = ""

    def __init__(self, parent=None, *, color: str = GREY, size: int = 16):
        super().__init__(parent)
        # Even, so the half-size radius is a whole pixel and the dot round.
        self._size = scaled_even_px(size)
        self.setFixedSize(self._size, self._size)
        self.set_color(color)

    def set_color(self, color: str) -> None:
        """This colour, written once.

        A stylesheet write repolishes and repaints whether or not it says
        what the last one said, and a control re-projected every beat wrote
        it every beat: one row of every device control repainted ten times
        a second for nothing.
        """

        wanted = str(color)
        if wanted == self._color:
            return
        self._color = wanted
        self.setStyleSheet(status_dot_stylesheet(wanted, radius=max(1, self._size // 2)))


def _paint_label_line(label: QtWidgets.QLabel) -> bool:
    """Paint a one-line, plain-text, vertically centred ``label`` with its
    line on :func:`centred_line_top` of its contents -- the line the field
    beside it puts its text on -- and say whether it did.

    Everything else QLabel does is kept: the frame and the sheet's
    background, the margin, the indent, the horizontal alignment, the
    palette and the enabled state.  A label that is not one plain line (it
    wraps, is rich text, is selectable, has a buddy's mnemonic) or has no
    room for its line is left to QLabel."""

    text = QtWidgets.QLabel.text(label)
    kind = label.textFormat()
    rich = kind == QtCore.Qt.RichText or (kind == QtCore.Qt.AutoText and QtCore.Qt.mightBeRichText(text))
    interactive = int(label.textInteractionFlags()) & int(
        QtCore.Qt.TextSelectableByMouse | QtCore.Qt.TextSelectableByKeyboard | QtCore.Qt.TextEditable
    )
    if (not text or "\n" in text or rich or interactive or label.wordWrap()
            or label.buddy() is not None or not int(label.alignment()) & int(QtCore.Qt.AlignVCenter)):
        return False
    margin = label.margin()
    rect = label.contentsRect().adjusted(margin, margin, -margin, -margin)
    metrics = label.fontMetrics()
    line = metrics.height()
    if rect.height() < line:
        return False
    align = int(QtWidgets.QStyle.visualAlignment(label.layoutDirection(), label.alignment()))
    indent = label.indent()
    if indent < 0 and label.frameWidth():
        indent = metrics.horizontalAdvance("x") // 2 - margin
    if indent > 0:
        if align & int(QtCore.Qt.AlignLeft):
            rect.setLeft(rect.left() + indent)
        if align & int(QtCore.Qt.AlignRight):
            rect.setRight(rect.right() - indent)
    top = centred_line_top(rect.top(), rect.height(), label.font())
    rect.setTop(max(rect.top(), min(top, rect.bottom() + 1 - line)))
    painter = QtGui.QPainter(label)
    label.drawFrame(painter)
    label.style().drawItemText(
        painter, rect, (align & int(QtCore.Qt.AlignHorizontal_Mask)) | int(QtCore.Qt.AlignTop),
        label.palette(), label.isEnabled(), text, label.foregroundRole(),
    )
    painter.end()
    return True


class FluentLabel(QtWidgets.QLabel):
    """Plain text in the body font.  One line of it, centred in its box, is
    painted on the line a field of that box's height puts its text on
    (:func:`_paint_label_line`), so a row label reads on the baseline of
    the field it names."""

    def __init__(self, text: str = "", parent=None):
        super().__init__(text, parent)
        self.setStyleSheet(f'QLabel {{ color: {TEXT}; font: {fluent_font_size()}pt "{FONT}"; background: transparent; }}')

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if not _paint_label_line(self):
            super().paintEvent(event)


def muted_note_label(text: str) -> "QtWidgets.QLabel":
    """A greyed, transparent, borderless note label in the fluent body font -- THE one muted-note
    factory (an optional empty-state note, an inline status line, ...), so the grey-note
    style is spelled once instead of re-typed per caller.  Art-bearing, so it stays internal (never
    re-exported from ``frontend/__init__``)."""
    lbl = FluentLabel(text)
    lbl.setStyleSheet(f'{MUTED_LABEL_STYLE} font: {fluent_font_size()}pt "{FONT}";')
    return lbl


class FluentFrame(QtWidgets.QFrame):
    def __init__(self, parent=None, *, bordered: bool = True, rule: bool = False,
                 round: tuple[str, ...] = ("NW", "NE", "SE", "SW")):
        super().__init__(parent)
        # ``bordered`` = a delineated flat card (1 px DIVIDER edge); False = a plain, edgeless region
        # (a header / button strip that must not read as a boxed card).  ``rule`` = one item of a
        # list inside a card: no box, a DIVIDER hairline along its bottom -- a card inside a card
        # doubled every border.  The edge is PAINTED, never a stylesheet ``QFrame { border }`` rule --
        # that would cascade onto every child QFrame.
        self._bordered = bool(bordered) and not rule
        self._rule = bool(rule)
        corners = {corner.upper() for corner in round}
        top_left = _radius() if "NW" in corners else 0
        top_right = _radius() if "NE" in corners else 0
        bottom_right = _radius() if "SE" in corners else 0
        bottom_left = _radius() if "SW" in corners else 0
        self.setStyleSheet(
            f"""
            QFrame {{
                background: white;
                border: none;
                border-top-left-radius: {top_left}px;
                border-top-right-radius: {top_right}px;
                border-bottom-right-radius: {bottom_right}px;
                border-bottom-left-radius: {bottom_left}px;
            }}
            """
        )

    def set_rule(self, on: bool) -> None:
        """Whether this list item draws its hairline.  A rule SEPARATES
        items; it never closes a list -- under the last item it doubled
        the card's own bottom edge."""

        if bool(on) != self._rule:
            self._rule = bool(on)
            self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().paintEvent(event)
        if self._bordered:
            stroke_card_border(self)
        elif self._rule:
            # The bottom edge-width of the item, whole device pixels: an
            # integer pen at h-1 sat a device pixel above the edge at 2x/3x.
            edge = frame_border_px(self)
            top = device_snap(self, self.height(), axis="y") - edge
            painter = QtGui.QPainter(self)
            painter.setRenderHint(QtGui.QPainter.Antialiasing, False)
            painter.fillRect(QtCore.QRectF(0.0, top, float(self.width()), edge), QtGui.QColor(DIVIDER))
            painter.end()


class FluentPopup(QtWidgets.QFrame):
    """A top-level ``Qt.Popup`` with a CLEAN rounded white-card chrome.

    A rounded card on a popup must NOT be done with a stylesheet ``border-radius``
    on an opaque top-level window: the square window stays opaque behind the arc,
    so the corner triangles outside the radius show a square white nub past the
    rounded border (and Windows adds a native popup shadow) -- that nub reads as
    the stray line at the bottom corners.  WA_TranslucentBackground alone is not
    enough either: it stops the stylesheet background painting at all, leaving the
    card see-through.

    So this paints the card itself: a TRANSLUCENT, frameless popup whose
    paintEvent strokes ONE antialiased rounded rect (fill + 1 px border).  The
    corners are truly rounded with nothing outside them, and because the border is
    painted (not a stylesheet rule) it can never cascade onto child widgets."""

    #: What the desktop is told this surface IS.  A menu by default; a
    #: subclass that is a frame the operator keeps says so instead, and says
    #: it HERE -- setting window flags after construction re-creates the
    #: native window underneath a widget that has already been polished.
    WINDOW_TYPE = QtCore.Qt.Popup

    #: CLASS attributes, not instance ones.  The owner's event filter is
    #: installed on a window that is already delivering events, and a
    #: subclass may reach this widget through Qt before ``__init__`` below
    #: has assigned anything -- QFrame's own constructor with a parent sends
    #: events, and so does WA_TranslucentBackground.  An ``eventFilter`` that
    #: read ``self._owner_window`` in that instant raised AttributeError out
    #: of a Qt filter, which is not a traceback but the end of the process.
    #: Declared here, there is no instant in which they are missing.
    _owner_window = None
    _toggle_anchor_ref = None

    def __init__(self, parent=None, *, radius: float | None = None,
                 border: str = DIVIDER, fill: str = "white"):
        super().__init__(parent, self.WINDOW_TYPE | QtCore.Qt.FramelessWindowHint
                         | QtCore.Qt.NoDropShadowWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        self._radius = None if radius is None else float(radius)
        self._border = QtGui.QColor(border)
        self._fill = QtGui.QColor(fill)
        self._owner_window = (
            parent.window() if isinstance(parent, QtWidgets.QWidget) else None
        )
        if self._owner_window is not None and self._owner_window is not self:
            self._owner_window.installEventFilter(self)

    def eventFilter(self, watched, event):  # noqa: N802 - Qt naming
        owner = self._owner_window
        if owner is not None and watched is owner:
            event_type = event.type()
            owner_retired = event_type in (
                QtCore.QEvent.WindowDeactivate,
                QtCore.QEvent.Hide,
                QtCore.QEvent.Close,
                QtCore.QEvent.Move,
                QtCore.QEvent.Resize,
            )
            if event_type == QtCore.QEvent.WindowStateChange:
                owner_retired = bool(
                    watched.windowState() & QtCore.Qt.WindowMinimized
                )
            # A MENU follows its owner by going away: it is anchored to a
            # widget in that window, and the moment the window moves,
            # resizes or loses the desktop the anchor means nothing.  A
            # frame the operator keeps open is not a window at all -- see
            # FluentOverlayFrame, which lives inside its page instead.
            if owner_retired and self.isVisible():
                self.hide()
        return super().eventFilter(watched, event)

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        # Qt clears WA_NoMouseReplay before each mouse event. Suppress only
        # the outside press on our own toggle, before QWidget closes us;
        # another control must still receive Qt's normal replayed click.
        reference = self._toggle_anchor_ref
        anchor = None if reference is None else reference()
        if (
            self.windowType() == QtCore.Qt.Popup
            and event.button() == QtCore.Qt.LeftButton
            and not self.rect().contains(event.pos())
            and anchor is not None
        ):
            hit = QtWidgets.QApplication.widgetAt(event.globalPos())
            while hit is not None:
                if hit is anchor:
                    self.setAttribute(QtCore.Qt.WA_NoMouseReplay)
                    break
                hit = hit.parentWidget()
        super().mousePressEvent(event)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        _paint_fluent_card(self, self._radius, self._border, self._fill)


def _paint_fluent_card(
    widget: QtWidgets.QWidget,
    radius: float | None,
    border: QtGui.QColor,
    fill: QtGui.QColor,
) -> None:
    """The one painted Fluent card: rounded, hairline-bordered, filled."""

    painter = QtGui.QPainter(widget)
    painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
    painter.setPen(frame_pen(widget, border))
    painter.setBrush(fill)
    # Inset by half the edge, so the stroke hugs the widget's edge unclipped.
    rect = frame_stroke_rect(widget)
    corner = float(_radius() if radius is None else radius)
    painter.drawRoundedRect(rect, corner, corner)
    # Ended here, not at garbage collection: a QMenu paints its items with
    # its own painter right after this card.
    painter.end()


class FluentOverlayFrame(QtWidgets.QWidget):
    """An auxiliary frame that lives INSIDE the page it serves.

    Same painted card as :class:`FluentPopup`, no relationship with the
    desktop at all.  The Setting frame was a ``Qt.Tool`` top-level window
    once -- "the operator carries it anywhere" -- and that ability was the
    product defect: a top-level survives its page being hidden, so
    switching the console to another tab left the frame floating over the
    wrong page, and nothing bounded a drag to the panel area.  As an
    ordinary child of the panel page the desktop questions dissolve
    instead of needing answers: it is clipped by the page, hides and
    returns with the tab, travels with the window, minimises with it,
    closes with it, and there is no Z order for anything to cut into.
    """

    def __init__(self, parent, *, radius: float | None = None,
                 border: str = DIVIDER, fill: str = "white"):
        if not isinstance(parent, QtWidgets.QWidget):
            raise TypeError("an overlay frame must live inside a widget")
        super().__init__(parent)
        self._radius = None if radius is None else float(radius)
        self._border = QtGui.QColor(border)
        self._fill = QtGui.QColor(fill)
        self.hide()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        _paint_fluent_card(self, self._radius, self._border, self._fill)


def _popup_content_chrome(
    popup: FluentPopup,
    content: QtWidgets.QWidget,
) -> tuple[int, int]:
    """Measure the decoration between a popup's edge and its mounted content.

    Walks the real ancestor chain -- layout margins, scroll-area frames, the
    reserved vertical scrollbar, and fixed siblings stacked in vertical box
    layouts (a drag handle above the scroll) -- so the popup is sized from
    what is actually there instead of a hand-tuned padding constant.  A magic
    pad drifts the moment the chrome changes, and a too-small guess clips the
    content's right edge under a scrollbar that was never accounted for.
    """

    width = height = 0
    child: QtWidgets.QWidget = content
    while child is not popup:
        widget = child.parentWidget()
        if widget is None:
            break
        if isinstance(widget, QtWidgets.QAbstractScrollArea):
            frame = 2 * widget.frameWidth()
            width += frame
            height += frame
            if widget.verticalScrollBarPolicy() != QtCore.Qt.ScrollBarAlwaysOff:
                width += widget.verticalScrollBar().sizeHint().width()
        layout = widget.layout()
        if layout is not None:
            margins = layout.contentsMargins()
            width += margins.left() + margins.right()
            height += margins.top() + margins.bottom()
            if (
                isinstance(layout, QtWidgets.QBoxLayout)
                and layout.direction()
                in (
                    QtWidgets.QBoxLayout.TopToBottom,
                    QtWidgets.QBoxLayout.BottomToTop,
                )
            ):
                for index in range(layout.count()):
                    item = layout.itemAt(index)
                    sibling = None if item is None else item.widget()
                    if sibling is None or sibling is child or sibling.isHidden():
                        continue
                    if not sibling.isAncestorOf(child):
                        height += sibling.sizeHint().height() + layout.spacing()
        child = widget
    return width, height


def show_fluent_popup_for_anchor(
    popup: FluentPopup,
    anchor: QtWidgets.QWidget,
    content: QtWidgets.QWidget,
    *,
    minimum_width: int = 360,
    minimum_height: int = 300,
    maximum_height: int | None = None,
    content_width: int | None = None,
) -> None:
    """Size and place one Fluent popup beside its anchor on the active screen.

    This is the single owner of the Setting-popup placement contract used by
    Workbench surfaces: prefer below, fall back above, clamp to the current
    screen, and leave the shared Fluent outer gap.  Popup content and its draft
    lifecycle remain entirely owned by the caller.

    ``content_width`` is the content's own required width in device pixels
    (a form's widest row).  Passing it explicitly lets a width-bounded scroll
    content stay a width CONSUMER -- no manual minimum pinned onto the widget
    that would clip when the screen clamps the popup narrower.

    This runs ONCE, when a popup opens.  It used to run again whenever the
    mounted content's required size changed, and an ``origin`` argument
    existed to keep a dragged popup from being thrown back beside its
    button by that second run.  Re-placing an open frame is what was wrong:
    it resized the operator's window under them and moved everything in it.
    The caller now measures content without re-placing, so there is one
    placement per open and nothing to preserve across it.
    """

    if not isinstance(popup, (FluentPopup, FluentOverlayFrame)):
        raise TypeError("popup must be FluentPopup or FluentOverlayFrame")
    if not isinstance(anchor, QtWidgets.QWidget):
        raise TypeError("anchor must be QWidget")
    if not isinstance(content, QtWidgets.QWidget):
        raise TypeError("content must be QWidget")
    for value, name in (
        (minimum_width, "minimum_width"),
        (minimum_height, "minimum_height"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if maximum_height is not None and (
        isinstance(maximum_height, bool)
        or not isinstance(maximum_height, int)
        or maximum_height <= 0
    ):
        raise ValueError("maximum_height must be a positive integer or None")
    if content_width is not None and (
        isinstance(content_width, bool)
        or not isinstance(content_width, int)
        or content_width <= 0
    ):
        raise ValueError("content_width must be a positive integer or None")

    # MEASURE WITHOUT MOVING ANYTHING.  This used to call popup.adjustSize(),
    # which is not a measurement at all: it RESIZES the popup to its unbounded
    # size hint before the cap below is applied.  For the one instant that
    # oversized popup existed, the scroll viewport grew, its vertical range
    # collapsed to fit, and Qt clamped the operator's scroll VALUE into the
    # smaller range.  Restoring the real size restored the range but not the
    # value, so a form that merely grew one row -- picking a fit model adds
    # one -- scrolled itself, and every row the operator was reading moved.
    #
    # Nothing below needs a resized popup: the chrome walk reads only layout
    # margins, spacing, frame widths and size hints, and the content's own
    # hint is layout-derived.  Polish is what adjustSize was really
    # contributing (style metrics for the reserved scrollbar), and polishing
    # has no geometry side effect.
    popup.ensurePolished()
    hint = content.sizeHint()
    # ONE algorithm, two coordinate spaces.  A window popup is placed in
    # global coordinates and clamped to the screen; an overlay frame is
    # placed in ITS PAGE's coordinates and clamped to the page -- which is
    # exactly what keeps the Setting frame inside the panel area.
    page = None if popup.isWindow() else popup.parentWidget()
    if page is None:
        anchor_top_left = anchor.mapToGlobal(QtCore.QPoint(0, 0))
        anchor_bottom_right = anchor.mapToGlobal(
            QtCore.QPoint(
                max(0, anchor.width() - 1), max(0, anchor.height() - 1)
            )
        )
        anchor_center = QtCore.QPoint(
            (anchor_top_left.x() + anchor_bottom_right.x()) // 2,
            (anchor_top_left.y() + anchor_bottom_right.y()) // 2,
        )
        available = _available_geometry_at(anchor_center, widget=anchor)
    else:
        anchor_top_left = anchor.mapTo(page, QtCore.QPoint(0, 0))
        anchor_bottom_right = anchor.mapTo(
            page,
            QtCore.QPoint(
                max(0, anchor.width() - 1), max(0, anchor.height() - 1)
            ),
        )
        available = page.rect()
    chrome_width, chrome_height = _popup_content_chrome(popup, content)
    desired_width = max(
        scaled_px(minimum_width),
        hint.width() + chrome_width,
        content.minimumWidth() + chrome_width,
        (0 if content_width is None else content_width) + chrome_width,
    )
    desired_height = max(
        scaled_px(minimum_height, minimum=260),
        hint.height() + chrome_height,
    )
    if maximum_height is not None:
        desired_height = min(desired_height, maximum_height)
    gap = popup_gap()
    if available is not None:
        desired_width = min(
            desired_width,
            max(1, available.width() - 2 * gap),
        )
        desired_height = min(
            desired_height,
            max(1, int(available.height() * 0.90)),
        )
        below_top = anchor_bottom_right.y() + 1 + gap
        above_bottom = anchor_top_left.y() - gap - 1
        below_space = max(0, available.bottom() - below_top + 1)
        above_space = max(0, above_bottom - available.top() + 1)
        if desired_height <= below_space:
            top = below_top
        elif desired_height <= above_space:
            top = above_bottom - desired_height + 1
        elif above_space >= below_space:
            desired_height = max(1, min(desired_height, above_space))
            top = above_bottom - desired_height + 1
        else:
            desired_height = max(1, min(desired_height, below_space))
            top = below_top
    else:
        top = anchor_bottom_right.y() + 1 + gap
    popup.resize(desired_width, desired_height)
    left = anchor_bottom_right.x() - popup.width() + 1
    if available is not None:
        left = max(
            available.left(),
            min(left, available.right() - popup.width() + 1),
        )
        top = max(
            available.top(),
            min(top, available.bottom() - popup.height() + 1),
        )
    popup.move(left, top)
    popup.show()
    popup.raise_()


class FluentSettingsPopupAnchor:
    """Own the true-toggle contract between one Setting button and its popup.

    The popup itself prevents replay of the one outside press on this anchor;
    closing a row or pressing Escape never delays a subsequent open. ``prepare``
    runs only before showing; ``present`` preserves the caller's placement.
    """

    def __init__(
        self,
        popup: FluentPopup,
        anchor: QtWidgets.QWidget,
    ) -> None:
        if not isinstance(popup, (FluentPopup, FluentOverlayFrame)):
            raise TypeError("popup must be FluentPopup or FluentOverlayFrame")
        if not isinstance(anchor, QtWidgets.QWidget):
            raise TypeError("anchor must be QWidget")
        self._popup = popup
        self._anchor = anchor
        if isinstance(popup, FluentPopup):
            popup._toggle_anchor_ref = weakref.ref(anchor)

    def toggle(
        self,
        content: QtWidgets.QWidget,
        *,
        prepare=None,
        present=None,
        minimum_width: int = 360,
        minimum_height: int = 300,
    ) -> None:
        """Close a visible popup, otherwise prepare and show it beside the anchor."""

        if prepare is not None and not callable(prepare):
            raise TypeError("prepare must be callable or None")
        if present is not None and not callable(present):
            raise TypeError("present must be callable or None")
        if not self._anchor.isEnabled():
            return
        popup = self._popup
        if popup.isVisible():
            popup.hide()
            return
        if prepare is not None:
            prepare()
        if present is None:
            show_fluent_popup_for_anchor(
                popup,
                self._anchor,
                content,
                minimum_width=minimum_width,
                minimum_height=minimum_height,
            )
        else:
            present()


class FluentGroupBox(QtWidgets.QGroupBox):
    def __init__(self, title: str = "", parent=None, *, title_strip_px: int = CARD_TITLE_PX,
                 title_color: str = TEXT):
        super().__init__(title, parent)
        # How much room the strip above the body takes.  A card that draws its
        # OWN strip -- because it has more to say than one line of plain text
        # -- passes zero and lays that strip out as an ordinary first row, so
        # the card's height stays the sum of what is in it.
        self._zlc_strip_px = scaled_px(int(title_strip_px)) if title_strip_px else 0
        self._zlc_title_color = str(title_color)
        #: The resting surface: body fill (white when None), dashed edge, edge ink.
        self._zlc_surface: tuple[str | None, bool, str | None] = (None, False, None)
        self._apply_chrome()

    def _apply_chrome(self) -> None:
        # Flat card delineated by a CONTINUOUS 1 px DIVIDER border painted in ``paintEvent`` (below) --
        # NOT Qt's own ``QGroupBox`` frame, which cuts a NOTCH in the top border where the title sits.
        # The title is a semibold heading INSIDE the card, inset like its content and centred in the
        # strip ``padding-top`` reserves -- painted by paintEvent, because a sheet cannot set the
        # weight of a group box title.  It was a grey pill on the card's corner, which read as the
        # stub of a tab, and section titles came in two styles: the pill and the bold section label.
        # The ::title rule only keeps Qt's own title text out of sight.
        self.setStyleSheet(
            f"""
            QGroupBox {{
                background: transparent;
                border: none;
                border-radius: {_radius()}px;
                margin-top: 0px;
                padding-top: {self._zlc_strip_px}px;
                color: {TEXT};
                font: {fluent_font_size()}pt "{FONT}";
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                background: transparent;
                color: transparent;
            }}
            """
        )

    def set_title_color(self, color: str) -> None:
        """The ink of the title.

        A bracket post wears its bracket's colour here, the colour the
        preview's loop is drawn in; a spacer wears grey.  The title is
        painted, so a new ink is a repaint.
        """

        value = str(color or TEXT)
        if getattr(self, "_zlc_title_color", None) != value:
            self._zlc_title_color = value
            self.update()

    def set_selected(self, selected: bool) -> None:
        """Mark this card as the one the next edit acts on, or unmark it.

        Selection is a FILL, not a frame: the body takes the accent tint and
        the edge stays the card's own (a spacer's dashes, a bracket's
        colour, the divider grey).  A 2 px accent border was tried first and
        read as an outline drawn around the card rather than as the card
        being picked; a softer accent edge was still a second line.  Painted
        in paintEvent, never via stylesheet -- an unscoped ``border:``
        cascades to every CHILD widget, so each inner checkbox/lineedit grew
        its own box.
        """

        value = bool(selected)
        if getattr(self, "_zlc_selected", False) != value:
            self._zlc_selected = value
            self.update()

    def is_selected(self) -> bool:
        """Whether this box is wearing the selection fill.

        Readable because "is this the selected one" is a question about what
        is on screen, and the only other way to ask it is to name the private
        attribute the painter happens to keep it in.
        """

        return bool(getattr(self, "_zlc_selected", False))

    def set_surface(
        self, fill: str | None = None, *, dashed: bool = False, edge: str | None = None,
    ) -> None:
        """The resting look of this box: its body colour (white when None),
        whether its edge is dashed, and the edge's own colour (the divider
        grey when None).  A spacer card is the page's own grey with a dashed
        edge, so it reads as something to look past; a bracket post wears
        its bracket's colour on the edge; selection fills the body and
        keeps the edge."""

        state = (
            None if fill is None else str(fill),
            bool(dashed),
            None if edge is None else str(edge),
        )
        if getattr(self, "_zlc_surface", (None, False, None)) != state:
            self._zlc_surface = state
            self.update()

    def paintEvent(self, event) -> None:
        # The body, a CONTINUOUS rounded border, then the title.  Nothing of Qt's own is drawn: a
        # Fluent group box is never checkable and has no frame of its own.
        del event
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        radius = float(_radius())
        fill, dashed, edge = getattr(self, "_zlc_surface", (None, False, None))
        selected = bool(getattr(self, "_zlc_selected", False))
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QColor(ACCENT_TINT if selected else (fill or "white")))
        painter.drawRoundedRect(QtCore.QRectF(self.rect()), radius, radius)
        pen = frame_pen(self, edge or DIVIDER)
        if dashed:
            pen.setStyle(QtCore.Qt.DashLine)
        painter.setPen(pen)
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.drawRoundedRect(frame_stroke_rect(self), radius, radius)
        if self.title() and self._zlc_strip_px:
            font = QtGui.QFont(FONT, fluent_font_size())
            font.setWeight(QtGui.QFont.DemiBold)
            painter.setFont(font)
            painter.setPen(QtGui.QColor(self._zlc_title_color))
            pad = scaled_px(CARD_PAD)
            band = QtCore.QRect(pad, 0, max(0, self.width() - 2 * pad), self._zlc_strip_px)
            title = painter.fontMetrics().elidedText(self.title(), QtCore.Qt.ElideRight, band.width())
            painter.drawText(band, QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter, title)
        painter.end()


#: What a button DOES, which is all its look may say.  A colour chosen per
#: call gave one verb a different hue in every window -- Remove orange here,
#: grey there, blue in the editor -- with white text at 1.65-2.55:1 on top.
#: ``primary`` is a surface's ONE run or commit action (the deep
#: ACCENT_STRONG fill, white ink) -- at most one per surface; ``standard``
#: every other command (the light-blue tint); ``destructive`` a command that
#: takes something away or stops or shuts something down -- Remove, Delete,
#: Clear, Unload, a device's Close, Shutdown devices, every Stop (a labelled
#: one sits on the RED_TINT face; a one-glyph remover repeated down rows is
#: a ghost in quiet ink that turns red only under the pointer, since red at
#: rest marks problems); and ``destructive_primary`` the confirm of a
#: destructive question -- the one white-on-red fill.
BUTTON_ROLES = ("primary", "standard", "destructive", "destructive_primary")

_BUTTON_STYLE_CACHE: tuple[float, str] | None = None


def _destructive_glyph_faces() -> tuple[str, str]:
    """The faces of a one-glyph remover under the pointer and held down.

    One source for FluentButton's ghost × and the × a selection grid paints
    at the end of a row, so the two are the same object to the eye."""

    return RED_TINT, _mixed(RED_TINT, RED, 0.3).name()


def _button_stylesheet() -> str:
    """The one sheet every FluentButton wears; role and state pick the rules.

    The layers are told apart by FILL, never by an edge: every face's
    border is its own fill, kept one edge-width wide so a state change
    never moves a label.  Standard is the light-blue BUTTON_TINT with HOVER
    ink (7.7:1), and held down it is SELECTED, the one colour that means
    "picked"; primary the deep ACCENT_STRONG with white ink (5.2:1), the one
    dark face on its surface.  Destructive has ONE look for every labelled
    command -- Remove, Delete, Clear, Close, Stop: the RED_TINT face,
    RED_DARK ink (5.4:1), as far from white as the light-blue face is -- so
    a Remove reads the same in a toolbar, a button grid or alone; a
    one-glyph remover repeated down a list of rows (x) stays a ghost with
    the red wash only under the pointer, because a column of red faces
    shouted louder than the rows it removed.  destructive_primary is the
    RED_DARK fill with white ink.  A command ON A TINTED CARD (``on_tint``,
    the green config Subpulse) is a white face with HOVER ink (9.9:1): no
    blue is laid on green, and the white face against the card is the
    command's whole boundary.  Pointer and press each step the face a
    shade.  Dirty (something new to save) is ONE rule for every save or
    commit: clean it is the role it was given, unsaved it is the primary
    with a trailing star.  Disabled is a flat DIVIDER face with DISABLED_INK
    and no frame; a disabled ghost stays a ghost.
    """

    global _BUTTON_STYLE_CACHE
    if _BUTTON_STYLE_CACHE is not None and _BUTTON_STYLE_CACHE[0] == _FLUENT_SCALE:
        return _BUTTON_STYLE_CACHE[1]
    tint_hover = _mixed(BUTTON_TINT, ACCENT, 0.25).name()
    strong_hover = QtGui.QColor(ACCENT_STRONG).darker(110).name()
    strong_pressed = QtGui.QColor(ACCENT_STRONG).darker(122).name()
    red_hover = _mixed(RED_TINT, RED, 0.19).name()
    # A labelled destructive command and a ghost × are pressed alike.
    glyph_hover, red_tint_pressed = _destructive_glyph_faces()
    red_strong_hover = QtGui.QColor(RED_DARK).darker(112).name()
    red_strong_pressed = QtGui.QColor(RED_DARK).darker(126).name()
    on_tint_pressed = _mixed(BG, DIVIDER, 0.6).name()
    style = f"""
    QPushButton {{
        background: {BUTTON_TINT};
        color: {HOVER};
        border: {FIELD_BORDER_PX}px solid {BUTTON_TINT};
        border-radius: {_radius()}px;
        padding: 0px {scaled_px(BUTTON_PADDING_H, minimum=6)}px;
        font: {fluent_font_size()}pt "{FONT}";
        outline: none;
    }}
    QPushButton:hover {{ background: {tint_hover}; border-color: {tint_hover}; }}
    QPushButton:pressed {{ background: {SELECTED}; border-color: {SELECTED}; }}
    QPushButton[on_tint="true"] {{
        background: {SURFACE}; border-color: {SURFACE}; color: {HOVER};
    }}
    QPushButton[on_tint="true"]:hover {{ background: {BG}; border-color: {BG}; }}
    QPushButton[on_tint="true"]:pressed {{
        background: {on_tint_pressed}; border-color: {on_tint_pressed};
    }}
    QPushButton[role="primary"] {{
        background: {ACCENT_STRONG}; border-color: {ACCENT_STRONG}; color: white;
    }}
    QPushButton[role="primary"]:hover {{
        background: {strong_hover}; border-color: {strong_hover};
    }}
    QPushButton[role="primary"]:pressed {{
        background: {strong_pressed}; border-color: {strong_pressed};
    }}
    QPushButton[role="destructive"] {{
        background: {RED_TINT}; border-color: {RED_TINT}; color: {RED_DARK};
    }}
    QPushButton[role="destructive"]:hover {{
        background: {red_hover}; border-color: {red_hover};
    }}
    QPushButton[role="destructive"]:pressed {{
        background: {red_tint_pressed}; border-color: {red_tint_pressed};
    }}
    QPushButton[role="destructive_primary"] {{
        background: {RED_DARK}; border-color: {RED_DARK}; color: white;
    }}
    QPushButton[role="destructive_primary"]:hover {{
        background: {red_strong_hover}; border-color: {red_strong_hover};
    }}
    QPushButton[role="destructive_primary"]:pressed {{
        background: {red_strong_pressed}; border-color: {red_strong_pressed};
    }}
    QPushButton[glyph="true"] {{
        padding: 0px {scaled_px(4, minimum=2)}px;
        font-size: {round(fluent_font_size() * 1.35)}pt;
    }}
    QPushButton[role="destructive"][glyph="true"] {{
        background: transparent; border-color: transparent; color: {MUTED_INK};
    }}
    QPushButton[role="destructive"][glyph="true"]:hover {{
        background: {glyph_hover}; border-color: {glyph_hover}; color: {RED_DARK};
    }}
    QPushButton[role="destructive"][glyph="true"]:pressed {{
        background: {red_tint_pressed}; border-color: {red_tint_pressed}; color: {RED_DARK};
    }}
    QPushButton:disabled, QPushButton[role="primary"]:disabled,
    QPushButton[role="destructive"]:disabled, QPushButton[on_tint="true"]:disabled,
    QPushButton[role="destructive_primary"]:disabled {{
        background: {DIVIDER}; border-color: {DIVIDER}; color: {DISABLED_INK};
    }}
    QPushButton[role="destructive"][glyph="true"]:disabled {{
        background: transparent; border-color: transparent; color: {DISABLED_INK};
    }}
    """
    _BUTTON_STYLE_CACHE = (_FLUENT_SCALE, style)
    return style


class FluentButton(QtWidgets.QPushButton):
    """A command, drawn by its ``role`` (see :data:`BUTTON_ROLES`).

    A command only: an on/off setting is a :class:`FluentSwitch`, never a
    checkable button.  The label is never elided: the button asks for its
    whole text plus padding, and a layout too narrow for it is the layout's
    to fix.
    """

    def __init__(self, text: str = "", parent=None, *, role: str = "standard"):
        self._label = str(text)
        self._dirty = False
        super().__init__(text, parent)
        self.setProperty("on_tint", False)
        self.setMinimumHeight(control_height())
        # Default sizing is set ONCE here: a caller that opts into Expanding
        # (a button grid) keeps it.
        self.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self._role = ""
        self._mark_glyph()
        self.set_role(role)
        self.setStyleSheet(_button_stylesheet())

    def setText(self, text: str) -> None:  # noqa: N802 - Qt API name
        self._label = str(text)
        self._show_label()

    def _show_label(self) -> None:
        """The label as shown: the caller's words and the dirty star."""

        label = self._label
        if self._dirty and not label.endswith("*"):
            label += "*"
        if QtWidgets.QPushButton.text(self) != label:
            QtWidgets.QPushButton.setText(self, label)
        self._mark_glyph()

    def sizeHint(self) -> QtCore.QSize:  # noqa: N802 - Qt API name
        hint = super().sizeHint()
        if self.property("glyph"):
            # The enlarged symbol must not make its button taller than the
            # word buttons beside it.
            hint.setHeight(self.minimumHeight())
        return hint

    def _glyph_ink(self) -> str:
        """The ink the sheet gives this glyph button's label in its state."""

        role = self.role()
        if not self.isEnabled():
            return DISABLED_INK
        if role == "destructive":
            return RED_DARK if self.underMouse() or self.isDown() else MUTED_INK
        if role in ("primary", "destructive_primary"):
            return SURFACE
        return HOVER

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """The face is drawn by the style; the label by the row's rules.

        A word sits on :func:`centred_line_top` of the face, the line the
        field and the label beside the button put their text on -- the
        style centred its line box, two device pixels from theirs.  A
        one-symbol label is centred on its INK (:func:`paint_centred_ink`).
        A button with an icon is the style's to lay out."""

        option = QtWidgets.QStyleOptionButton()
        self.initStyleOption(option)
        glyph = bool(self.property("glyph"))
        if not glyph and not option.icon.isNull():
            super().paintEvent(event)
            return
        painter = QtWidgets.QStylePainter(self)
        painter.drawControl(QtWidgets.QStyle.CE_PushButtonBevel, option)
        if glyph:
            paint_centred_ink(painter, self, QtCore.QRectF(self.rect()), self.font(), option.text,
                              self._glyph_ink())
        else:
            style = self.style()
            area = style.subElementRect(QtWidgets.QStyle.SE_PushButtonContents, option, self)
            area.setTop(centred_line_top(0, self.height(), self.font()))
            flags = QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop | QtCore.Qt.TextShowMnemonic
            if not style.styleHint(QtWidgets.QStyle.SH_UnderlineShortcut, option, self):
                flags |= QtCore.Qt.TextHideMnemonic
            style.drawItemText(painter, area, int(flags), option.palette, self.isEnabled(),
                               option.text, QtGui.QPalette.ButtonText)
        painter.end()

    def setEnabled(self, enabled: bool) -> None:  # noqa: N802 - Qt API name
        """Disabling a button that holds the focus lets the focus go first.

        Qt hands the focus of a widget being disabled to the next one in the
        tab chain: a clicked On Pulse disabled while it runs passed it to
        Stop Pulse, and the next Space stopped the pulse the operator had
        just started.  Cleared here, the focus goes nowhere.  (A button is
        still focused by a click: a field commits what was typed into it
        when the focus leaves, so a command that took no focus would act on
        the number from before the typing.)"""

        if not enabled and self.hasFocus():
            self.clearFocus()
        super().setEnabled(enabled)

    def setVisible(self, visible: bool) -> None:  # noqa: N802 - Qt API name
        """Hiding a focused button lets the focus go first, for the reason
        :meth:`setEnabled` gives."""

        if not visible and self.hasFocus():
            self.clearFocus()
        super().setVisible(visible)

    def enterEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().enterEvent(event)
        if self.property("glyph"):
            self.update()

    def leaveEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().leaveEvent(event)
        if self.property("glyph"):
            self.update()

    def _mark_glyph(self) -> None:
        """A one-symbol label (×, ⋯, ▾) is drawn larger, with little padding.

        At body size such a symbol was a speck in the middle of a button
        padded for a word.
        """

        glyph = len(self._label) == 1 and not self._label.isalnum()
        if self.property("glyph") != glyph:
            self.setProperty("glyph", glyph)
            self._repolish()

    def set_on_tint(self, on_tint: bool) -> None:
        """Whether the button sits on a TINTED card (a config Subpulse).

        There a light-blue face laid a second hue on the card's; it wears a
        plain white face instead, whose contrast with the card is its whole
        boundary, so no blue is brought onto green.
        """

        value = bool(on_tint)
        if self.property("on_tint") != value:
            self.setProperty("on_tint", value)
            self._repolish()

    def role(self) -> str:
        """The role it is drawn with now: primary while dirty."""

        return str(self.property("role"))

    def set_role(self, role: str) -> None:
        """What this command does; the sheet is repolished only on a change."""

        value = str(role)
        if value not in BUTTON_ROLES:
            raise ValueError(f"button role must be one of {BUTTON_ROLES}, not {value!r}")
        self._role = value
        self._show_role()

    def _show_role(self) -> None:
        shown = "primary" if self._dirty else self._role
        if self.property("role") != shown:
            self.setProperty("role", shown)
            self._repolish()

    def _repolish(self) -> None:
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()

    def set_dirty(self, dirty: bool = True) -> None:
        """Mark that pressing this would save or apply something new.

        One rule for every save or commit: clean, the button wears the role
        it was given (a Save is a standard command); unsaved, it is drawn
        primary with a trailing ``*`` -- the one thing on its surface to
        press.  A Save that was primary while there was nothing to save
        shouted for no reason, and an orange tint on one tab beside a
        primary on the next made one verb two looks.  Idempotent and
        width-stable: the label never accumulates stars.
        """

        dirty = bool(dirty)
        if self._label.endswith("*"):
            self._label = self._label[:-1]
        self._dirty = dirty
        self._show_label()
        self._show_role()

    def is_dirty(self) -> bool:
        return self._dirty


def _apply_fluent_context_menu(widget, event, *, extra=()) -> None:
    """Pop ``widget``'s right-click menu with the Fluent rounded-card chrome instead of the platform's
    opaque SQUARE "black box" menu.  The ONE shared context-menu builder for every editable text widget
    (``FluentLineEdit``, ``FluentReadoutMultiline``, ``FluentCodeEdit``): it REUSES Qt's own standard
    context menu -- so Undo / Cut / Copy / Paste / Delete / Select-All stay wired to THAT widget -- and
    only swaps the container, re-homing the actions into a :class:`_FluentRoundedMenu` (ownership moved
    via ``setParent`` before the throwaway standard menu is disposed, so its teardown cannot delete them
    out from under the Fluent menu).  A read-only widget's standard menu is just Copy / Select-All; it is
    styled identically -- no widget special-cases the menu.  ``_FluentRoundedMenu`` resolves at call time,
    so this may sit above its definition.

    The menu is a ONE-SHOT object: built for this click, gone after it.  It is
    parented to the widget so it inherits the widget's screen and stacking,
    and a parented QObject outlives the Python name that made it -- every
    right-click used to leave one more menu, and its actions, on the widget
    until the window died.

    ``extra`` is ``(text, callable)`` commands a control adds below the
    standard ones -- a numeric box's "Edit step…".  Each runs after the menu
    has closed, so a dialog it opens is not stacked on a closing menu."""
    std = widget.createStandardContextMenu()
    menu = _FluentRoundedMenu(widget)
    for act in list(std.actions()):
        act.setParent(menu)
        menu.addAction(act)
    std.deleteLater()
    commands = {}
    if extra:
        menu.addSeparator()
        for text, command in extra:
            commands[menu.addAction(str(text))] = command
    try:
        chosen = menu.exec_(event.globalPos())
    finally:
        retire_widget(menu)
    event.accept()
    command = commands.get(chosen)
    if command is not None:
        command()


def _line_edit_text_line(edit: QtWidgets.QLineEdit) -> QtCore.QRect:
    """The line box QLineEdit lays its text in: its contents less the text
    margins and its own horizontal inset, one line high, centred with Qt's
    integer rounding.  Its top plus the font's ascent is the baseline."""

    option = QtWidgets.QStyleOptionFrame()
    # initFrom, not initStyleOption: a spin box's edit is made by Qt, and
    # PyQt refuses a protected call on an object it did not create.
    option.initFrom(edit)
    contents = edit.style().subElementRect(QtWidgets.QStyle.SE_LineEditContents, option, edit)
    margins = edit.textMargins()
    area = contents.adjusted(margins.left(), margins.top(), -margins.right(), -margins.bottom())
    line = edit.fontMetrics().height()
    top = area.y() + (area.height() - line + 1) // 2
    return QtCore.QRect(
        area.x() + _QLINEEDIT_HMARGIN, top, max(0, area.width() - 2 * _QLINEEDIT_HMARGIN), line
    )


def _seat_line_edit_text(edit: QtWidgets.QLineEdit) -> None:
    """Give ``edit`` the top and bottom text margins that move Qt's line box
    onto :func:`centred_line_top` of its own height.

    The edit's height is its field's: a plain field is its own frame, and
    the editor inside a spin box or an editable combo is laid out
    symmetrically inside theirs.  The margins are equal and opposite -- the
    room Qt centres the line in is moved, not shrunk -- so the line moves
    by exactly their size and the edit asks for no other height than it
    did.  The left and right margins stay the owner's (a readout's inset,
    a badge's room)."""

    option = QtWidgets.QStyleOptionFrame()
    option.initFrom(edit)
    contents = edit.style().subElementRect(QtWidgets.QStyle.SE_LineEditContents, option, edit)
    line = edit.fontMetrics().height()
    qt_top = contents.y() + (contents.height() - line + 1) // 2
    shift = centred_line_top(0, edit.height(), edit.font()) - qt_top
    margins = edit.textMargins()
    if (margins.top(), margins.bottom()) != (shift, -shift):
        edit.setTextMargins(margins.left(), shift, margins.right(), -shift)


def field_text_inset() -> int:
    """Where a field's text starts inside its box, from the box's left
    edge: the edge, the padding and Qt's own text margin (logical px).

    A read-only value standing in a column of fields -- a popup's Default
    and Effective under its Config name, a card's Total under its Name --
    indents its text by this, so the values and the typed text start on
    one x instead of the values starting at the boxes' edge."""

    return FIELD_BORDER_PX + scaled_px(EDIT_PADDING_H) + _QLINEEDIT_HMARGIN


def line_edit_text_right(edit: QtWidgets.QLineEdit) -> int:
    """Where the text of ``edit`` may end with NO right text margin, in its
    own coordinates: its contents (inside a sheet's border and padding)
    less Qt's own inset.  A part laid over a field's right end asks this
    for the margin that keeps the text clear of it."""

    option = QtWidgets.QStyleOptionFrame()
    option.initFrom(edit)
    contents = edit.style().subElementRect(QtWidgets.QStyle.SE_LineEditContents, option, edit)
    return contents.right() + 1 - _QLINEEDIT_HMARGIN


def _numeric_text_width(edit: QtWidgets.QLineEdit, right_edge: int | None = None) -> int:
    option = QtWidgets.QStyleOptionFrame()
    option.initFrom(edit)
    contents = edit.style().subElementRect(QtWidgets.QStyle.SE_LineEditContents, option, edit)
    right = contents.right() + 1 if right_edge is None else min(contents.right() + 1, right_edge)
    margins = edit.textMargins()
    inset = edit.style().pixelMetric(QtWidgets.QStyle.PM_FocusFrameHMargin, option, edit)
    return max(1, right - contents.left() - margins.left() - margins.right() - 2 * inset)


class FluentLineEdit(QtWidgets.QLineEdit):
    valueNormalized = QtCore.pyqtSignal()
    #: How the text is elided while the field is not being edited; None
    #: scrolls it as any line edit does (see set_display_elide).
    _display_elide = None

    def __init__(self, text: str = "", parent=None):
        super().__init__(text, parent)
        self._res_step: float | None = None
        self._allow_any = True
        self._numeric_bounds: tuple[float | None, float | None, str] = (None, None, "float")
        self._numeric_validator = False
        #: ``(kind, bottom, top, decimals)`` of the validator
        #: set_numeric_validator installed; None once anything else replaced it.
        self._validator_spec: tuple | None = None
        self._quantity_unit = ""
        self._shown_quantity_unit = ""
        self._normalization_timer: QtCore.QTimer | None = None
        self._normalizing = False
        self._numeric_layout_key: tuple | None = None
        self.setMinimumHeight(control_height())
        self._apply_style()
        self.set_placeholder_prompt(False)
        self.setText(str(text))
        self.editingFinished.connect(self._snap_to_resolution)
        self.textEdited.connect(self._normalize_visible_number)

    def _apply_style(self) -> None:
        self.setStyleSheet(
            f"""
            QLineEdit {{
                background: white;
                border: {FIELD_BORDER_PX}px solid {_FIELD_EDGE_REST};
                border-radius: {_radius()}px;
                padding: {scaled_px(PADDING_V)}px {scaled_px(EDIT_PADDING_H)}px;
                color: {TEXT};
                font: {fluent_font_size()}pt "{FONT}";
            }}
            QLineEdit:focus {{ border: {FIELD_BORDER_PX}px solid {_FIELD_EDGE_FOCUS}; }}
            QLineEdit:disabled {{ background: {BG}; color: {DISABLED_INK}; }}
            """
        )

    def contextMenuEvent(self, event) -> None:  # noqa: N802 - Qt naming
        _apply_fluent_context_menu(self, event)

    def set_placeholder_prompt(self, prompt: bool = True) -> None:
        """Draw the placeholder as a pure PROMPT (the text ink at half
        strength, Qt's own 2.9:1) instead of as information (MUTED_INK,
        5.6:1, every field's default).

        A placeholder that says what a blank field DOES -- "full sensor",
        "camera's current" --, what to type there ("filter names and
        values"), or that nothing is chosen yet is something the operator
        must read, so information is the default; only a field's own name
        repeated inside it ("File name", "role") may stay faint.
        """

        colour = QtGui.QColor(TEXT if prompt else MUTED_INK)
        if prompt:
            colour.setAlpha(128)
        palette = self.palette()
        palette.setColor(QtGui.QPalette.PlaceholderText, colour)
        self.setPalette(palette)

    def set_display_elide(self, mode) -> None:
        """Elide what does not fit (``QtCore.Qt.ElideMiddle``, ...) while the
        field is not being edited; ``None`` scrolls as a line edit does.

        A path shown from its head hid the very file name it was there to
        name.  Only the PAINTING elides: ``text()`` stays whole, and the
        field shows all of it again the moment it takes focus.
        """

        self._display_elide = mode
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        mode = self._display_elide
        if mode is None or self.hasFocus() or not self.text():
            super().paintEvent(event)
            return
        # Exactly where QLineEdit would put the text, so the elided spelling
        # does not move against the whole one it replaces.
        area = _line_edit_text_line(self)
        metrics = self.fontMetrics()
        if metrics.horizontalAdvance(self.text()) <= area.width():
            super().paintEvent(event)
            return
        option = QtWidgets.QStyleOptionFrame()
        self.initStyleOption(option)
        painter = QtGui.QPainter(self)
        self.style().drawPrimitive(QtWidgets.QStyle.PE_PanelLineEdit, option, painter, self)
        painter.setPen(QtGui.QColor(TEXT if self.isEnabled() else DISABLED_INK))
        painter.setFont(self.font())
        painter.drawText(
            area, QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop,
            metrics.elidedText(self.text(), mode, area.width()),
        )
        painter.end()

    def width_for(self, sample: str) -> int:
        """Width for complete text, including the actual style and text margins."""
        self.ensurePolished()
        chrome = self.width() - _numeric_text_width(self)
        return self.fontMetrics().horizontalAdvance(str(sample)) + 2 + chrome

    def setText(self, text: str) -> None:  # noqa: N802 - Qt API name
        text = str(text)
        if self.text() == text:
            self._normalize_visible_number()
            return
        super().setText(text)
        self._normalize_visible_number()
        if not self.hasFocus():
            self.setCursorPosition(0)

    def _normalize_visible_number(self, *_args) -> None:
        if self._normalizing or not self._numeric_validator:
            return
        if _args and self._normalization_timer is not None:
            self._normalization_timer.stop()
        if self.isReadOnly() or not self.isEnabled():
            # Width belongs to an editor, not to a read-only projection.
            # Drop any old display-fit refusal; re-entering the editable state
            # below re-evaluates the same text against the then-current width.
            self.setProperty("numericError", "")
            self._numeric_layout_key = None
            if self._normalization_timer is not None:
                self._normalization_timer.stop()
            return
        if not self.isVisible() or self.width() <= 1:
            return
        if _args:
            text = self.text()
            if self.validator().validate(text, len(text))[0] != QtGui.QValidator.Acceptable:
                return
            if self.fontMetrics().horizontalAdvance(text) + 2 <= _numeric_text_width(self):
                # A spelling that fits is not too narrow, whatever the last
                # full evaluation of an earlier text said.  Left standing,
                # that refusal met the next commit as a message about a
                # number the box was already showing whole: 10000 refused,
                # then 1000 refused with it, then 100 accepted.
                self.setProperty("numericError", "")
                return  # Keep an in-progress spelling such as 0. or 0.00.
        margins = self.textMargins()
        key = (self.text(), self.width(), margins.left(), margins.right(),
               self._numeric_bounds, self._res_step, self._allow_any,
               self._quantity_unit, self._shown_quantity_unit)
        if key == self._numeric_layout_key:
            return
        try:
            number = _finite_decimal(self.text())
        except (ValueError, TypeError):
            self._numeric_layout_key = key
            return  # Incomplete input remains an editable draft.
        bottom, top, kind = self._numeric_bounds
        requested_number = number
        if kind == "quantity":
            bottom = _bound_in_unit(bottom, self._quantity_unit, self._shown_quantity_unit)
            top = _bound_in_unit(top, self._quantity_unit, self._shown_quantity_unit)
        low = None if bottom is None else _finite_decimal(bottom)
        high = None if top is None else _finite_decimal(top)
        available = _numeric_text_width(self)
        integral = kind == "int"
        # An int|float box's text is its value's type, so a float keeps its
        # point: rewritten as 10000000000000000, or narrowed to 12345678, a
        # float read back as an int.
        point = self._keeps_point(self.text())
        try:
            rounded, text, minimum = _visible_decimal(
                number, low, high, integral, available, self.fontMetrics(), point=point,
            )
            if self._res_step and not self._allow_any:
                aligned = _finite_decimal(align_to_resolution(text, self._res_step, allow_any=False))
                if aligned != rounded:
                    rounded, text, minimum = _visible_decimal(
                        aligned, aligned, aligned, integral, available, self.fontMetrics(),
                        point=point,
                    )
        except ValueError as error:
            self.setProperty("numericError", str(error))
            self._queue_number_notification()
            return
        self.setProperty("numericError", "")
        current = self.text()
        if (
            rounded == requested_number
            and "e" not in current.lower()
            and self.fontMetrics().horizontalAdvance(current) + 2 <= available
        ):
            # Normalization owns width and bounds, not spelling: a number it
            # leaves unchanged that already fits keeps its text.  Rewritten,
            # 100.0 became 100, and an int|float field read back an int.
            text = current
        self._normalizing = True
        try:
            if text != self.text():
                # A width-only rewrite is passive normalization.  Emitting
                # textChanged here made Form/Device Control treat a resize as
                # a user edit before valueNormalized could identify it.
                with signals_blocked(self):
                    super().setText(text)
        finally:
            self._normalizing = False
        self._numeric_layout_key = (self.text(), self.width(), *key[2:])
        if rounded != requested_number and not _args:
            self._queue_number_notification()

    def _keeps_point(self, text: str) -> bool:
        """Whether a rewrite of ``text`` must keep a decimal point.

        Only an int|float box's text is its value's type: there ``1.5`` or
        ``1e3`` is a float, and every spelling the box writes for it -- a
        width rewrite, a bound it is pulled back to -- stays one.
        """

        return self._numeric_bounds[2] == "number" and any(
            mark in text.lower() for mark in ".e"
        )

    def _queue_number_notification(self) -> None:
        if self._normalization_timer is None:
            timer = QtCore.QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(self._emit_normalized_value)
            self._normalization_timer = timer
        self._normalization_timer.start(0)

    def _emit_normalized_value(self) -> None:
        if self.isEnabled() and not self.isReadOnly():
            self.valueNormalized.emit()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        _seat_line_edit_text(self)
        self._normalize_visible_number()

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self._normalize_visible_number()

    def changeEvent(self, event) -> None:  # noqa: N802
        super().changeEvent(event)
        if event.type() in (QtCore.QEvent.FontChange, QtCore.QEvent.StyleChange):
            _seat_line_edit_text(self)
        if event.type() in (
            QtCore.QEvent.FontChange,
            QtCore.QEvent.StyleChange,
            QtCore.QEvent.EnabledChange,
            QtCore.QEvent.ReadOnlyChange,
        ):
            if hasattr(self, "_numeric_layout_key"):
                self._numeric_layout_key = None
                self._normalize_visible_number()

    def set_resolution(self, step: float | None) -> None:
        self._res_step = None if step in (None, 0) else float(step)

    def set_allow_any(self, allow_any: bool = True) -> None:
        self._allow_any = bool(allow_any)

    def setValidator(self, validator) -> None:  # noqa: N802
        replaced = self.validator()
        super().setValidator(validator)
        if replaced is not None and replaced is not validator and replaced.parent() is self:
            # QLineEdit only points at its validator: one it replaced stays a
            # child of the edit until the edit dies, and every projection
            # re-declares the bounds of the boxes it keeps.
            replaced.deleteLater()
        self._validator_spec = None
        self._numeric_validator = isinstance(validator, (QtGui.QIntValidator, QtGui.QDoubleValidator))
        if self._numeric_validator:
            self._numeric_bounds = (
                validator.bottom() if math.isfinite(validator.bottom()) else None,
                validator.top() if math.isfinite(validator.top()) else None,
                "int" if isinstance(validator, QtGui.QIntValidator) else "float",
            )

    def set_numeric_validator(
        self,
        kind: str = "float",
        *,
        bottom: float | None = None,
        top: float | None = None,
        decimals: int = 12,
    ) -> None:
        """Restrict typed input to a number (a numeric validator on the line edit).

        ``kind="float"`` accepts only digits, a decimal point, ``e``/``E`` and a sign
        (scientific notation); ``kind="int"`` accepts only an integer.  ``kind="number"``
        types like a float and holds an ``int | float`` value whose text is its type, so
        a float keeps its decimal point through every width rewrite.  Optional
        ``bottom``/``top`` bound the value (an int field with a ``top`` also blocks
        clearly-too-big entries as you type).  Other characters are simply rejected at
        the keystroke, so the field can never hold non-numeric junk.

        A bound field also CORRECTS itself when the operator leaves it, and
        that correction is what makes the field commit at all.  A Qt validator
        alone is worse than none here: 700 typed into a -512..511 field
        validates as Intermediate -- no more digits than 511, so it could still
        be on its way to something legal -- Qt lets Intermediate text sit in
        the box, AND refuses to emit ``editingFinished`` for it.  The value is
        then neither accepted nor refused nor reported: the operator types 700,
        presses the button, and nothing happens at all.  Clamping before Qt
        decides whether to emit means the field always leaves in a state it can
        commit.

        The same bounds declared again are the validator already installed:
        a projection re-declares every field it keeps."""

        self._numeric_bounds = (bottom, top, kind)
        spec = (kind, bottom, top, int(decimals))
        if spec == self._validator_spec:
            self._numeric_validator = True
            return
        if kind == "int":
            low, high = QT_INT_RANGE
            if all(
                edge is None or low <= int(edge) <= high for edge in (bottom, top)
            ):
                validator: QtGui.QValidator = QtGui.QIntValidator(self)
                if bottom is not None:
                    validator.setBottom(int(bottom))
                if top is not None:
                    validator.setTop(int(top))
            else:
                # A Python int has no width, and an owner may bound one
                # beyond what Qt's integer validator can hold; that validator
                # raises for such a bound, so the keystroke check is only
                # "digits and a sign" and the bound is applied exactly, by
                # the clamp below, when the operator is done.
                validator = QtGui.QRegularExpressionValidator(_INT_TEXT_RE, self)
        else:
            validator = QtGui.QDoubleValidator(self)
            validator.setNotation(QtGui.QDoubleValidator.ScientificNotation)
            validator.setDecimals(int(decimals))
            if bottom is not None:
                validator.setBottom(float(bottom))
            if top is not None:
                validator.setTop(float(top))
        # Force a dot decimal separator regardless of the system locale.
        validator.setLocale(QtCore.QLocale.c())
        self.setValidator(validator)
        self._validator_spec = spec
        self._numeric_bounds = (bottom, top, kind)
        self._numeric_validator = True

    def set_quantity_validator(
        self,
        unit: str,
        *,
        bottom: float | None = None,
        top: float | None = None,
    ) -> None:
        """This field holds a number OF ``unit``, and shows digits alone.

        Which spelling of that unit the digits are in is the picker's to say,
        beside the field -- so this takes a plain number, and the bounds,
        which are the OWNER's and are in the owner's unit, are converted
        rather than compared across two different scales.
        """

        wanted = str(unit).strip() or "1"
        if wanted != self._quantity_unit:
            # A DIFFERENT unit is a different quantity, read in its own
            # spelling.  The same unit re-declared -- every projection
            # re-declares the bounds of the field it keeps -- leaves the
            # spelling the operator chose on screen where it is.
            self._quantity_unit = wanted
            self._shown_quantity_unit = wanted
        self.set_numeric_validator("float", bottom=None, top=None)
        self._numeric_bounds = (bottom, top, "quantity")

    def setShownUnit(self, unit: str) -> None:  # noqa: N802 - Qt API name
        """Read this field in another spelling of its own unit.

        The value does not move: what the owner holds is what it holds.  The
        digits on screen are converted, and so is whatever is typed next, so
        an RF floor declared in dBm can be worked in mW without anyone doing
        that arithmetic by hand.
        """

        wanted = str(unit).strip() or self._quantity_unit
        previous = self._shown_quantity_unit
        if wanted == previous:
            return
        # Raises for an incompatible unit BEFORE anything on screen moves.
        DEFAULT_UNITS.convert(0.0, self._quantity_unit, wanted)
        text = self.text().strip()
        if not text:
            self._shown_quantity_unit = wanted
            return
        try:
            number = _finite_decimal(text)
        except ValueError:
            return
        converted = DEFAULT_UNITS.convert_decimal(number, previous, wanted)
        self._shown_quantity_unit = wanted
        self.setText(format_quantity(converted, "1"))
        if self.property("numericError"):
            self._shown_quantity_unit = previous
            super().setText(text)
            self._numeric_layout_key = None

    def shownUnit(self) -> str:  # noqa: N802 - Qt API name
        return self._shown_quantity_unit

    def valueUnit(self) -> str:  # noqa: N802 - Qt API name
        return self._quantity_unit

    def focusOutEvent(self, event) -> None:  # noqa: N802
        # BEFORE super(), which is what decides whether editingFinished is
        # emitted: an out-of-range entry that is corrected first is acceptable
        # by the time Qt looks at it, so the field both shows the right number
        # and tells its consumer about it.
        self._clamp_to_numeric_bounds()
        super().focusOutEvent(event)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
            self._clamp_to_numeric_bounds()
        super().keyPressEvent(event)

    def _clamp_to_numeric_bounds(self) -> None:
        """Pull an out-of-range entry back to the nearest end of the range."""

        bottom, top, kind = self._numeric_bounds
        if bottom is None and top is None:
            return
        text = self.text().strip()
        if not text:
            return
        if kind == "int":
            # In integers, exactly: a bound past 2**53 read through a float
            # is a different number.
            try:
                value = int(text, 10)
            except ValueError:
                return
            clamped = value
            if bottom is not None:
                clamped = max(clamped, int(bottom))
            if top is not None:
                clamped = min(clamped, int(top))
            if clamped != value:
                self.setText(str(clamped))
            return
        try:
            shown = _finite_decimal(text)
            # The bounds are the owner's, in the owner's unit; the text is in
            # whichever spelling is on screen.  Comparing them directly was
            # comparing two different scales.
            value = (
                DEFAULT_UNITS.convert_decimal(
                    shown, self._shown_quantity_unit, self._quantity_unit
                )
                if kind == "quantity"
                else shown
            )
        except (UnitError, ValueError):
            # Incomplete text is still owned by the numeric validator.
            return
        clamped = value
        if bottom is not None:
            clamped = max(clamped, _finite_decimal(bottom))
        if top is not None:
            clamped = min(clamped, _finite_decimal(top))
        if clamped == value:
            return
        if kind == "quantity":
            self.setText(
                format_quantity(
                    DEFAULT_UNITS.convert_decimal(
                        clamped, self._quantity_unit, self._shown_quantity_unit
                    ),
                    "1",
                )
            )
            return
        # Spelled as the box spells every number, in the type the operator
        # typed: str() of the bound wrote 1.5 pulled back to an int bound as
        # 1, and the int|float field committed the int its operator never
        # typed (2 pulled back to a float bound 1.0 the other way round).
        self.setText(_decimal_text(clamped, point=self._keeps_point(text)))

    def _snap_to_resolution(self) -> None:
        if self._normalization_timer is not None:
            self._normalization_timer.stop()
        if self._res_step:
            before = self.text()
            after = align_to_resolution(before, self._res_step, allow_any=self._allow_any)
            if after != before:
                self.setText(after)
        self._normalize_visible_number()


class FluentReadoutEdit(FluentLineEdit):
    """A read-only BUT selectable / copyable value -- the THIRD line-edit kind.

    Three visually distinct text widgets, so a user tells at a glance what a field is:

      * :class:`FluentLineEdit`    -- WHITE box + accent focus edge: an EDITABLE input.
      * :class:`FluentLabel`       -- plain text: pure DISPLAY (cannot select).
      * :class:`FluentReadoutEdit` -- plain TEXT-ink text on the surface, no box: you can
        select + Ctrl-C it but NOT edit it (a command result, a resolved file path, a
        readout value).  A value too long for its row is elided in the MIDDLE -- a path
        keeps its head and its file name -- and the tooltip carries all of it.

    It wore a grey box with a faint edge, which is the disabled button's chrome: a path
    the operator was meant to read looked like a control that was switched off.

    Inherits FluentLineEdit's sizing / API (``setText`` / ``text``) so it drops into the same
    rows.  Being text, it starts where text starts -- flush with its widget's left edge, so a
    value sits exactly under its column header or beside the label it continues.  It kept a
    field's inner padding when it lost the box, and every value then started nine pixels to
    the right of the header above it."""

    def __init__(self, text: str = "", parent=None):
        super().__init__(text, parent)
        # QLineEdit keeps two pixels before its text; a readout has no box to keep them from.
        self.setTextMargins(-2, 0, 0, 0)
        self.setReadOnly(True)
        self.setCursor(QtCore.Qt.IBeamCursor)        # the I-beam signals "selectable text"
        self.set_display_elide(QtCore.Qt.ElideMiddle)
        self.setToolTip(self.text())

    def setText(self, text: str) -> None:  # noqa: N802 - Qt API name
        super().setText(text)
        self.setToolTip(self.text())

    def _apply_style(self) -> None:
        # No box and no side inset: the transparent top and bottom edges keep a field's height.
        self.setStyleSheet(
            f"""
            QLineEdit {{
                background: transparent;
                border: {FIELD_BORDER_PX}px solid transparent;
                border-left-width: 0px;
                border-right-width: 0px;
                padding: {scaled_px(PADDING_V)}px 0px;
                color: {TEXT};
                font: {fluent_font_size()}pt "{FONT}";
            }}
            """
        )


class FluentReadoutMultiline(QtWidgets.QPlainTextEdit):
    """A MULTI-LINE :class:`FluentReadoutEdit` -- the read-only-but-selectable text field for a
    value TOO LONG for one line (a resolved path, a metadata blob, or a data shape).

    Same look/role as :class:`FluentReadoutEdit` (plain TEXT-ink text on the surface, no box:
    you can select + Ctrl-C but not edit), but it SOFT-WRAPS a long value at the
    field width (``WrapAtWordBoundaryOrAnywhere``) and AUTO-SIZES its height to the WHOLE wrapped
    content -- so every line is shown in full and nothing is ever truncated the way a single-line edit
    clips it.  There is NO row cap and NO inner scrollbar: the field grows to fit all its lines and the
    outer scroll area (the Info column's ``FluentScrollArea``) scrolls the whole form -- an inner
    scrollbar on a field that already shows everything is exactly the "why scroll when it's all here"
    confusion.  Uses a dynamic plain-text layout (scrollbars OFF, count the VISUAL
    wrapped lines, size to fit), on the house Fluent readout chrome.  All colours / radius / font come
    from the style tokens (one source), never re-picked."""

    def __init__(self, text: str = "", parent=None, *, plain: bool = False,
                 max_height: int | None = None):
        super().__init__(parent)
        # plain=True -> transparent, borderless (reads as plain MESSAGE text, e.g. a warning/error
        #   body) with no padding and no caret; still selectable + wrap-anywhere.
        # max_height set -> the field grows to fit its content UP TO this many px, then STOPS and
        #   scrolls vertically (the QPlainTextEdit reserves its own scrollbar gutter and re-wraps the
        #   text narrower, so nothing ever lands under the bar).  None -> grow unbounded (default).
        self._plain = bool(plain)
        self._max_height = None if max_height is None else int(max_height)
        self.setReadOnly(True)
        self.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse
                                     | QtCore.Qt.TextSelectableByKeyboard)
        self.setCursor(QtCore.Qt.IBeamCursor)          # the I-beam signals "selectable text"
        # Soft-wrap a long value at the field width (break inside a long token like a path too), so it
        # wraps to as many lines as it needs instead of overflowing / clipping.
        self.setLineWrapMode(QtWidgets.QPlainTextEdit.WidgetWidth)
        self.setWordWrapMode(QtGui.QTextOption.WrapAtWordBoundaryOrAnywhere)
        # Horizontal scrollbar OFF forever: width-wrapped -> never sideways.  Vertical starts OFF and,
        # WITHOUT a max_height, stays off (the field auto-sizes to fit EVERY line, so a vertical bar
        # would have nothing hidden to scroll to); WITH a max_height, _adjust_height
        # flips it to AsNeeded once the content exceeds the cap.  (The Fluent bar look is applied by
        # _apply_style, which folds in fluent_scrollbar_stylesheet.)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        # Match the document's inner margin to the single-line edit's vertical padding (PADDING_V): the
        # default QPlainTextEdit documentMargin (4 px) makes one text row's content box TALLER than the
        # FluentReadoutEdit row, so a one-line field could not both equal that height AND fully contain
        # its text.  At PADDING_V the row's content box lines up with the line edit's, so the anchored
        # height formula (single-row height + one lineSpacing per extra line) shows every line in full.
        self.document().setDocumentMargin(scaled_px(PADDING_V))
        # That margin is on every side; a readout's text still starts at its widget's left edge,
        # under the header or beside the label it continues, so the viewport takes it back.
        self.setViewportMargins(-scaled_px(PADDING_V), 0, 0, 0)
        self._apply_style()
        if self._plain:
            # A MESSAGE body is not an input: hide the blinking text caret and don't steal the dialog's
            # initial focus (so it lands on the OK button -> Enter dismisses).  Mouse selection + copy
            # still work regardless of focus policy.
            self.setCursorWidth(0)
            self.setFocusPolicy(QtCore.Qt.ClickFocus)
        if text:
            self.setPlainText(str(text))
        # Recompute height when the text changes AND when the wrapped layout size changes (a resize
        # re-wraps the same text into a different line count) -- both changes trigger remeasurement.
        self.textChanged.connect(self._adjust_height)
        self.document().documentLayout().documentSizeChanged.connect(self._adjust_height)
        self._adjust_height()

    def contextMenuEvent(self, event) -> None:  # noqa: N802 - Qt naming
        _apply_fluent_context_menu(self, event)   # Fluent right-click (Copy / Select-All), not the native menu

    def _apply_style(self) -> None:
        if self._plain:
            # Transparent + borderless: reads as plain MESSAGE text (a warning / error body), still
            # selectable + wrap-anywhere, flush with the card's text -- no field padding.
            base = f"""
            QPlainTextEdit {{
                background: transparent;
                border: none;
                padding: 0px;
                color: {TEXT};
                font: {fluent_font_size()}pt "{FONT}";
            }}
            """
        else:
            # The SAME readout look as FluentReadoutEdit: plain TEXT-ink text on the surface, no
            # box (a grey box was the disabled button's chrome) and no side inset.
            base = f"""
            QPlainTextEdit {{
                background: transparent;
                border: {FIELD_BORDER_PX}px solid transparent;
                border-left-width: 0px;
                border-right-width: 0px;
                padding: {scaled_px(PADDING_V)}px 0px;
                color: {TEXT};
                font: {fluent_font_size()}pt "{FONT}";
            }}
            """
        # Fold in the shared Fluent scrollbar look: a ``max_height`` field shows a vertical bar when it
        # overflows, and this ``setStyleSheet`` would otherwise clobber a separately-applied bar style.
        self.setStyleSheet(base + fluent_scrollbar_stylesheet(
            "QScrollBar", frame_width=0 if self._plain else 1
        ))

    def setText(self, text: str) -> None:  # noqa: N802 - match FluentReadoutEdit's line-edit API
        """Quack like a line edit (``setText`` / ``text``) so it drops into the same rows/fillers."""
        self.setPlainText(str(text))

    def text(self) -> str:
        return self.toPlainText()

    def resizeEvent(self, event):  # noqa: N802 - Qt naming
        # Wrapping depends on the viewport width, so a width change re-wraps -> re-measure the height.
        super().resizeEvent(event)
        self._adjust_height()

    @QtCore.pyqtSlot()
    def _adjust_height(self) -> None:
        """Size the field to fit ALL its wrapped content (no row cap -> every line shown, no inner
        scroll).

        Qt's actual ``QTextLayout`` line boxes are the height authority.  Font
        ``lineSpacing`` is smaller on some platforms, and the scrollbar range
        is measured in logical block positions rather than pixels; using either
        as geometry left one hidden line position even when every glyph looked
        visible.  Summing each block's laid-out rectangle plus the document and
        widget margins gives the exact wrapped content box at the live width.

        Re-entrancy guard: measuring the wrapped line count calls ``document().adjustSize()`` and the
        resize re-lays-out the document, either of which can emit ``documentSizeChanged`` -- a connected
        trigger of this method -- so an unguarded call recurses.  The flag lets the outermost call settle
        the height; a nested emission is dropped, and the trailing ``documentSizeChanged`` from the final
        resize re-runs it once cleanly (converging, since a fixed WIDTH gives a stable wrapped count)."""
        if getattr(self, "_adjusting_height", False):
            return
        self._adjusting_height = True
        try:
            document = self.document()
            document.setTextWidth(self.viewport().width())
            document.adjustSize()
            layout = document.documentLayout()
            content_height = 0.0
            block = document.begin()
            while block.isValid():
                content_height += layout.blockBoundingRect(block).height()
                block = block.next()
            margins = self.contentsMargins()
            total = math.ceil(
                content_height + 2.0 * document.documentMargin()
            ) + margins.top() + margins.bottom()
            cap = self._max_height
            if cap is not None and total > cap:
                # Too tall to fit: STOP at the cap and SCROLL the overflow.  The vertical bar shows
                # (AsNeeded) and, being a QPlainTextEdit, RESERVES its own gutter -- the text re-wraps
                # narrower, so nothing lands under the bar and nothing overflows sideways.
                self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
                if self.height() != cap:
                    self.setFixedHeight(cap)
            else:
                self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
                if total != self.height():
                    self.setFixedHeight(total)
                self.horizontalScrollBar().setRange(0, 0)
                self.verticalScrollBar().setRange(0, 0)
                self.verticalScrollBar().setValue(0)
            self.updateGeometry()
        finally:
            self._adjusting_height = False


class FluentPathEdit(QtWidgets.QWidget):
    """A path field + a ``Browse…`` button -- the ONE reusable control for every path
    parameter (a data folder, a saved template, or a calibration file).

    The user can TYPE a path or click Browse to pick it in the native file/folder
    dialog, so a path is never a bare line-edit you must hand-type into.  ``mode="dir"``
    opens a folder chooser; ``mode="file"`` an open-file chooser filtered by
    ``file_filter``.  Quacks like a line edit (``text`` / ``setText`` /
    ``setPlaceholderText`` / ``setToolTip``) and emits ``changed(str)`` so a form can
    treat it exactly like a ``FluentLineEdit``.

    ``refreshable=True`` adds a Refresh button that emits ``refresh_requested``
    without changing the path.  A file field points at something that is edited
    elsewhere -- a pulse in the Pulse Editor, a calibration a run has just
    written -- and the only way to re-read it was to pick the same file again
    through the dialog, or to touch some other field so the whole draft
    re-finalized.  Both worked by accident; this says it."""

    changed = QtCore.pyqtSignal(str)
    selected = QtCore.pyqtSignal(str)
    refresh_requested = QtCore.pyqtSignal()

    def __init__(self, text: str = "", *, mode: str = "file", caption: str = "Choose",
                 file_filter: str = "All files (*)", base_dir: str = "",
                 refreshable: bool = False, parent=None):
        super().__init__(parent)
        self._mode = "dir" if str(mode) == "dir" else "file"
        self._caption = str(caption)
        self._filter = str(file_filter)
        # The folder the Browse dialog opens in when the field doesn't point at an existing
        # path -- e.g. a template directory or a calibration directory --
        # so the operator lands where the files actually live, not the process CWD.
        self._base_dir = str(base_dir)
        self.setStyleSheet("background: transparent;")
        row = QtWidgets.QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(scaled_px(6, minimum=4))
        # Every child is born WITH its parent.  A widget that has none is a
        # top-level WINDOW the moment it is made visible, and this row set
        # the Refresh button visible before the layout adopted it: on the
        # one GUI that asks for that button, a 130x66 window titled
        # "python" flashed on the desktop for a frame before the real
        # window opened.  Parenting at construction makes the stray window
        # unreachable rather than short-lived.
        self.edit = FluentLineEdit(str(text), self)
        # a modest floor so the field + button still fit a narrow form column (the
        # edit grows with the column via the stretch below); never a 150px floor that
        # pushes the Browse button off a narrow Edit tab.
        self.edit.setMinimumWidth(scaled_px(96, minimum=72))
        # The file name is the end of a path: a field too narrow for all of it
        # keeps the head and the name, and its tooltip carries the whole path.
        self.edit.set_display_elide(QtCore.Qt.ElideMiddle)
        self._hint = ""
        self.edit.textChanged.connect(lambda t: self.changed.emit(str(t)))
        self.edit.textChanged.connect(lambda _text: self._show_tooltip())
        self.browse = FluentButton("Browse…", self)
        self.browse.clicked.connect(self._browse)
        self.refresh = FluentButton("Refresh", self)
        self.refresh.setToolTip("Re-read this file from disk")
        self.refresh.clicked.connect(self.refresh_requested)
        self.refresh.setVisible(bool(refreshable))
        self.refresh.setEnabled(bool(refreshable))
        row.addWidget(self.edit, 1)
        row.addWidget(self.browse, 0)
        row.addWidget(self.refresh, 0)

    def _browse(self) -> None:
        start = self._dialog_start()
        if self._mode == "dir":
            picked = QtWidgets.QFileDialog.getExistingDirectory(self, self._caption, start)
        else:
            picked, _ = QtWidgets.QFileDialog.getOpenFileName(self, self._caption, start, self._filter)
        if picked:
            self.edit.setText(picked)   # fires textChanged -> changed
            self.selected.emit(picked)

    def _dialog_start(self) -> str:
        """Where the Browse dialog opens: the current path if it's a real file/dir or sits in
        a real directory the user typed; otherwise ``base_dir`` (the folder these files live
        in, when it exists) so a BARE default like ``mot_field_template.json`` still lands in
        the configured base directory rather than the CWD; otherwise the CWD."""
        import os
        cur = self.edit.text().strip()
        if cur and os.path.exists(cur):
            return cur
        parent = os.path.dirname(cur)
        if parent and os.path.isdir(parent):       # user typed a real-dir-qualified path
            return cur
        if self._base_dir:
            base = os.path.abspath(self._base_dir)
            if os.path.isdir(base):
                return base
        return os.getcwd()

    def text(self) -> str:
        return self.edit.text()

    def setText(self, text: str) -> None:  # noqa: N802 - Qt API name
        self.edit.setText("" if text is None else str(text))

    def clear(self) -> None:
        self.edit.clear()

    def setPlaceholderText(self, text: str) -> None:  # noqa: N802 - Qt API name
        self.edit.setPlaceholderText(str(text))

    def setToolTip(self, text: str) -> None:  # noqa: N802 - Qt API name
        self._hint = str(text)
        self._show_tooltip()
        self.browse.setToolTip(self._hint)

    def _show_tooltip(self) -> None:
        self.edit.setToolTip("\n".join(part for part in (self.edit.text(), self._hint) if part))


class FluentSectionLabel(FluentLabel):
    """THE section title: a group's name on its own line, or the title of a
    header row whose commands and summary sit beside it.

    One style, everywhere a section is named: TEXT ink at the weight of a
    card's title and a table's column names (600), as tall as a control,
    its line on the line a field of that height puts its text on.  It kept
    a top margin for an own-line header, and as the title of a row of
    fields that margin put it two pixels under their text; the space above
    a section is the section gap of the layout it stands in."""

    def __init__(self, text: str = "", parent=None):
        super().__init__(text, parent)
        # The weight goes in the SHEET, the one that renders: a QFont-only
        # weight set before a sheet is dropped when the sheet repolishes.
        self.setStyleSheet(
            f'QLabel {{ color: {TEXT}; font: 600 {fluent_font_size()}pt "{FONT}"; '
            f"background: transparent; border: none; }}"
        )
        self.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.setMinimumHeight(control_height())


def setting_label_width(labels, *, minimum: int = 72) -> int:
    """The ONE label-column width for a settings/edit/param form: wide enough that the LONGEST
    of ``labels`` shows in full, never below ``minimum`` (scaled).  Every form computes its
    column with THIS function, so each form's rows align and the width rule is identical across
    the whole frontend (a form with longer labels simply gets a wider column).  This is the single
    source of the "[label | control]" column width -- callers never hand-pick a px value."""
    metrics = QtGui.QFontMetrics(QtGui.QFont(FONT, fluent_font_size()))
    widest = 0
    for text in labels:
        try:
            widest = max(widest, metrics.horizontalAdvance(str(text)))
        except AttributeError:  # pragma: no cover - very old Qt
            widest = max(widest, metrics.width(str(text)))
    return max(scaled_px(int(minimum), minimum=48), widest + scaled_px(10))


class FluentSettingRow(QtWidgets.QWidget):
    """One settings row -- the ONE param-row primitive for the whole frontend:
    ``[fixed-width label | control flush-left, expanding]``.

    THE design system (one logic everywhere): a form is built from exactly two primitives --
    :class:`FluentSectionLabel` (semibold, dark) for a GROUP header, and this row (regular
    weight, fixed-width column) for each control.  The section/row hierarchy is shown by
    weight ONLY -- never by indentation.  Labels are TEXT: grey read as disabled beside a
    required field, and grey is what a HINT wears.  Share ONE column width
    per form via :func:`setting_label_width` so the rows align.  Do NOT use a ``QFormLayout`` (its
    labels render dark + auto-width = a different style) and do NOT use a bare ``FluentLabel`` as a
    row/section label using the shared ``[QLabel | widget]`` row template."""

    def __init__(
        self,
        label: str | QtWidgets.QWidget,
        control: QtWidgets.QWidget,
        *,
        label_width: int | None = None,
        before: tuple[QtWidgets.QWidget, ...] = (),
        after: tuple[tuple[QtWidgets.QWidget, int], ...] = (),
        parent=None,
    ):
        super().__init__(parent)
        # A form row is a width consumer, never a width author.  Combo/path/error
        # size hints can be arbitrarily long; the enclosing Setting/Edit viewport
        # owns the available width and the control must contract inside it.
        self.setMinimumWidth(0)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Ignored,
            QtWidgets.QSizePolicy.Preferred,
        )
        h = QtWidgets.QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(scaled_px(6, minimum=4))
        if isinstance(label, QtWidgets.QWidget):
            lbl = label
            lbl.setParent(self)
        else:
            lbl = FluentLabel(str(label), self)
        lbl.setFixedWidth(int(label_width) if label_width is not None else scaled_px(60, minimum=48))
        if isinstance(lbl, QtWidgets.QLabel):
            lbl.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
            lbl.setStyleSheet(f"color: {TEXT}; background: transparent; border: none;")
        self._label = lbl                       # introspected by the layout-uniformity contract test
        h.addWidget(lbl)
        # Cells a host puts BESIDE the control -- a device's current reading
        # before it, its live switch, Apply and status after it -- belong to
        # the row from the moment it exists.  A host that took the row apart
        # to add them after the form had shown it painted a row with half
        # its cells, once, on every control it opened.
        for cell in before:
            h.addWidget(cell)
        # A control that FILLS the cell (a combo / line-edit / spin / switch -- Expanding/Preferred
        # horizontal policy) is stretched so its left edge sits flush against the label column.  A
        # control pinned to its sizeHint (FIXED horizontal policy -- compact controls,
        # a FluentButton) must NOT be stretched: a stretch cell CENTRES a non-filling widget, so the
        # toggle would float in the middle of the row instead of left-aligning under its label.  Add it
        # left-aligned with a trailing stretch instead, so
        # EVERY control -- filling or fixed -- starts at the SAME left edge.  A CAPPED control (a
        # number box) stretches up to its cap beside a trailing stretch that takes the rest.
        if control.sizePolicy().horizontalPolicy() == QtWidgets.QSizePolicy.Fixed:
            h.addWidget(control, 0, QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
            if not after:
                h.addStretch(1)
        else:
            h.addWidget(control, 1)
            if not after and control.maximumWidth() < QtWidgets.QWIDGETSIZE_MAX:
                h.addStretch(1)
        for cell, stretch in after:
            h.addWidget(cell, int(stretch))

    def set_label(self, text: str, *, width: int | None = None) -> None:
        """Update this stable row's label without replacing either control."""

        if hasattr(self._label, "setText"):
            self._label.setText(str(text))
        if width is not None:
            self._label.setFixedWidth(int(width))


class _FluentRoundedMenu(QtWidgets.QMenu):
    """A ``QMenu`` with the SAME antialiased rounded-card chrome as every other Fluent popup
    (the FluentPopup Setting card, the FluentComboBox drop-down), reusable frontend-owned.

    A stylesheet ``border-radius`` alone is NOT enough on a menu: its window stays opaque + SQUARE
    behind the arc, so the corners outside the radius show a white nub (and the platform adds a native
    shadow) -- reading as a hard-cornered popup beside the antialiased round ones.  So the window is
    made TRANSLUCENT + frameless and this PAINTS the card itself (fill + 1 px border) via the same
    ``drawRoundedRect`` single source, BEFORE the items.  A menu draws its own items,
    so the card is composited UNDER them (super paints the labels on top) rather than replacing them."""

    def __init__(self, parent=None, *, radius: float | None = None,
                 border: str = DIVIDER, fill: str = "white"):
        super().__init__(parent)
        self._radius = float(_radius() if radius is None else radius)
        self._border = QtGui.QColor(border)
        self._fill = QtGui.QColor(fill)
        # Translucent frameless window so nothing opaque/square sits behind the painted arc.
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.FramelessWindowHint
                            | QtCore.Qt.NoDropShadowWindowHint)
        # Transparent menu body (the card is painted, not a stylesheet background) + a rounded ACCENT
        # pill on the selected/hover row -- the same item chrome the combo drop-down uses.
        # The item has a horizontal MARGIN inside the menu's own padding so the selected/hover pill's
        # rounded rect sits fully WITHIN the card's rounded corners (a pill flush to the padding edge
        # gets clipped by the arc -> a square-cornered grey band bleeding past the corner).  The
        # ``::indicator`` is collapsed to nothing (image:none + 0x0) so a (formerly checkable) action
        # never reserves a tick gutter that shifts the labels right / draws an un-Fluent check box.
        self.setStyleSheet(
            f"""
            QMenu {{ background: transparent; border: none;
                     padding: {scaled_px(4, minimum=3)}px;
                     font: {fluent_font_size()}pt "{FONT}"; color: {TEXT}; }}
            QMenu::item {{ padding: {scaled_px(5, minimum=4)}px {scaled_px(14, minimum=10)}px;
                          margin: 0px {scaled_px(2, minimum=2)}px;
                          border-radius: {_radius()}px; }}
            QMenu::item:selected {{ background: {ACCENT_TINT}; color: {TEXT}; }}
            QMenu::separator {{ height: {FIELD_BORDER_PX}px; background: {DIVIDER};
                               margin: {scaled_px(4, minimum=3)}px {scaled_px(2, minimum=2)}px; }}
            QMenu::indicator {{ image: none; width: 0px; height: 0px; }}
            """
        )

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        # Paint the rounded card FIRST, then let QMenu draw its items on top.
        _paint_fluent_card(self, self._radius, self._border, self._fill)
        super().paintEvent(event)


class FluentCardDialog(QtWidgets.QDialog):
    """THE chrome every modal this project owns wears.

    A native dialog frame puts the platform title bar -- and with it the
    interpreter's own icon -- in the corner of a window whose every other
    pixel is this project's.  One off-brand corner is all it takes to read
    as somebody else's dialog, so no modal here has a native frame:
    frameless and translucent, each PAINTS the same rounded white card
    (fill + 1 px DIVIDER border, the shared ``drawRoundedRect`` recipe) as
    every other Fluent surface.

    The chrome lives HERE and not in each dialog, because three copies of
    it is three chances for one of them to drift back to a platform frame
    -- which is exactly how the two that did got there.  A subclass builds
    its own contents and renders its own title (a frameless card has no
    title bar to put one in); it inherits the card.

    Taking the frame away takes two things with it, so the card supplies
    both or it is not a usable modal.  WHERE IT OPENS: a platform frame is
    placed by the window manager, which never puts one off the screen; a
    card placed by hand at its parent's centre goes wherever that centre
    is, and a window near an edge put the whole card past it -- modal, so
    nothing else would take a click, and no button anyone could reach.
    HOW IT MOVES: a platform frame has a title bar to drag, and a card has
    none, so the card itself is the handle.
    """

    def __init__(self, parent=None, *, title: str = "") -> None:
        super().__init__(parent, QtCore.Qt.FramelessWindowHint | QtCore.Qt.Dialog)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        self.setModal(True)
        self.setFont(QtGui.QFont(FONT, fluent_font_size()))
        self._radius = float(_radius())
        self._drag_origin: QtCore.QPoint | None = None
        if title:
            self.setWindowTitle(str(title))

    def exec_(self) -> int:
        """Place the card where it can be used, then run the modal loop.

        Every card modal goes through here, so none of them can be the one
        that opens somewhere unreachable.
        """

        # Before the nested loop starts, not after: what it is about to
        # paint has to be what is actually still there.
        retire_pending_widgets()
        self.adjustSize()
        self.move(self._opening_position())
        return super().exec_()

    def _opening_position(self) -> QtCore.QPoint:
        """Centred on the parent, but ALWAYS on the screen it lands on.

        The centring is what makes a modal feel attached to the window it
        came from.  The clamp is what keeps that from being fatal: a parent
        near an edge -- or dragged half off one -- centres the card past
        the screen, and a modal nobody can click is a window nobody can
        close.
        """

        anchor = self.parentWidget()
        # ``size``, not ``frameGeometry`` -- a frameless window has no frame,
        # so the two are the same thing once it exists, and before it does
        # only this one is right (the other is a pixel short in each
        # direction, which is a pixel off the screen).
        size = self.size()
        if anchor is not None:
            centre = anchor.mapToGlobal(anchor.rect().center())
            wanted = QtCore.QPoint(
                centre.x() - size.width() // 2, centre.y() - size.height() // 2
            )
        else:
            wanted = self.pos()
        return _placed_on_screen(wanted, size, widget=anchor or self)

    # -- the card is its own title bar -----------------------------------
    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Begin a drag, when nothing in the card wanted the press.

        No widget list decides that: Qt only delivers a press HERE once
        every child has declined it, so a button still clicks, a text body
        still selects, and the dead space and the title label -- which
        decline by nature -- are the handle.
        """

        if event.button() == QtCore.Qt.LeftButton:
            self._drag_origin = (
                event.globalPos() - self.frameGeometry().topLeft()
            )
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt naming
        origin = self._drag_origin
        if origin is not None and event.buttons() & QtCore.Qt.LeftButton:
            self.move(event.globalPos() - origin)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt naming
        self._drag_origin = None
        super().mouseReleaseEvent(event)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        _paint_fluent_card(
            self, self._radius, QtGui.QColor(DIVIDER), QtGui.QColor("white")
        )


class _FluentMessageDialog(FluentCardDialog):
    """A modal message / warning box on the shared card: the section title
    (``FluentSectionLabel``), the wrapped message and a single Fluent OK
    button, centred on its parent.  ``fluent_message`` is the ONE entry
    point -- use it in place of ``QMessageBox.information`` / ``.warning``."""

    def __init__(
        self,
        parent,
        title: str,
        text: str,
        *,
        kind: str = "info",
        confirm: bool = False,
        confirm_text: str = "Confirm",
        cancel_text: str = "Cancel",
    ):
        super().__init__(parent)
        pad = scaled_px(16, minimum=10)
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(pad, pad, pad, pad)
        outer.setSpacing(scaled_px(10, minimum=6))
        outer.addWidget(FluentSectionLabel(str(title)))
        # The body is a wrap-ANYWHERE read-only text block (FluentReadoutMultiline, plain chrome): a
        # long unbreakable token -- a Windows path, a dotted traceback frame, a fingerprint -- BREAKS
        # mid-token instead of running past the card edge / under the scrollbar the way a QLabel's
        # word-BOUNDARY wrap does (that clipping was the bug).  It hugs a short message and, past the
        # height cap, stops and scrolls (the QPlainTextEdit reserves its own vertical-bar gutter, so
        # nothing lands under the bar).  Selectable, too, so a user can copy an error / traceback.
        body_width = scaled_px(360, minimum=240)
        body = FluentReadoutMultiline(str(text), plain=True, max_height=scaled_px(320, minimum=180))
        body.setFixedWidth(body_width)
        outer.addWidget(body)
        row = QtWidgets.QHBoxLayout()
        row.addStretch(1)
        if confirm:
            cancel = FluentButton(str(cancel_text))
            cancel.clicked.connect(self.reject)
            row.addWidget(cancel)
            # A warning question confirms something that cannot be taken
            # back, so its confirm is the one red fill this package draws.
            accept = FluentButton(
                str(confirm_text),
                role="destructive_primary" if str(kind) == "warning" else "primary",
            )
            accept.clicked.connect(self.accept)
            row.addWidget(accept)
        else:
            # Acknowledging a message destroys nothing, warning or not.
            ok = FluentButton("OK", role="primary")
            ok.clicked.connect(self.accept)
            row.addWidget(ok)
        outer.addLayout(row)


def retire_widget(widget: QtWidgets.QWidget) -> None:
    """Take a widget off the screen for good, without it ever being a window.

    THE one way to retire a widget.  It is hidden EXPLICITLY, then deleted
    on the next pass of the loop, and it keeps its parent until then.

    What it must never be is unparented first.  Adding a widget to a layout
    under a visible parent QUEUES a show of it; ``setParent(None)`` on a
    widget that was not explicitly hidden clears the explicit-hide mark; so
    a widget added and then unparented in the same pass -- a row rebuilt
    twice because the draft it wrote re-projected it -- is shown by that
    queued event AFTER it has lost its parent, which on Windows is a real
    top-level window on the desktop for the one frame before its deferred
    delete runs.  That is the small window that flashed at Add axis, at a
    pulse reopen, at a closed editor tab: thirteen retirements, one idiom.
    An explicit hide is what the queued show checks for, so it is the one
    line that makes the difference, and it lives here so nobody writes the
    idiom without it.
    """

    widget.hide()
    widget.deleteLater()


def detach_widget(widget: QtWidgets.QWidget) -> None:
    """Take a widget out of its host to be mounted somewhere else later.

    The counterpart of :func:`retire_widget` for a widget that lives on: a
    plot surface handed from one card to another, a row the handle still
    owns.  It is hidden EXPLICITLY before it is unparented, for the reason
    given there -- a queued show that lands after the unparenting maps a
    top-level window -- and whoever mounts it next shows it.
    """

    widget.hide()
    widget.setParent(None)


def retire_pending_widgets() -> None:
    """Run the deletions the outer event loop has not got to yet.

    ``deleteLater`` runs when the loop it was called from returns to its top
    level -- and a modal dialog does not return there, it opens a NESTED loop.
    So a rebuild that retires widgets and then reports what it did leaves the
    retired widgets alive for exactly as long as the message is up, and the
    nested loop repaints the window behind the dialog with both sets on it.

    Measured on a pulse reopen: 6 old period cards still alive beside the 4 new
    ones until the dialog closed, which is when the text stopped being doubled.
    """

    application = QtWidgets.QApplication.instance()
    if application is not None:
        application.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)


def fluent_message(parent, title: str, text: str, *, kind: str = "info") -> None:
    """Show a modal Fluent message dialog (``kind`` = ``"info"`` | ``"warning"``) -- the ONE on-brand
    replacement for ``QMessageBox.information`` / ``.warning`` (no native frame, no stray app icon).
    Centres on ``parent``.  Blocks until the user clicks OK (or presses Escape).

    The dialog is made for this one message and retired the moment it has
    been answered.  It is parented to ``parent`` so it centres on it, and a
    parented dialog is kept alive by that parent, not by the local name here:
    left as it was, every message a window ever showed stayed in its object
    tree until the window itself was destroyed."""
    dialog = _FluentMessageDialog(parent, title, text, kind=kind)
    try:
        dialog.exec_()
    finally:
        retire_widget(dialog)


def stamped_file_name(stem: str, extension: str, *, stamp: str | None = None) -> str:
    """What a file saved out of a window is called.

    A name the operator recognises plus the moment it was written, because two
    saves of the same board are two files and the second must not silently be
    the first.  ONE rule: the Edit tab's Save Fig and the console's own Save
    Layout / Save Screenshot had begun to grow their own.
    """

    from datetime import datetime  # noqa: PLC0415 -- only a save needs the clock

    text = str(stem).strip().replace("/", "_").replace("\\", "_") or "untitled"
    moment = stamp if stamp is not None else datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = str(extension).lstrip(".")
    return f"{text}_{moment}.{suffix}" if suffix else f"{text}_{moment}"


def fluent_save_path(parent, caption: str, suggested: str, filter: str) -> str:
    """Ask where to write, offering a name; "" when the operator declines.

    The suggestion is a PATH, not a directory: Qt fills the name box from it,
    and a dialog that opens with an empty name box makes every save begin with
    typing.  Written once -- three windows had a copy of this call, and two of
    them passed a bare folder.
    """

    path, _selected = QtWidgets.QFileDialog.getSaveFileName(
        parent, str(caption), str(suggested), str(filter)
    )
    return str(path)


def fluent_open_path(parent, caption: str, start: str, filter: str) -> str:
    """Ask which file to read; "" when the operator declines."""

    path, _selected = QtWidgets.QFileDialog.getOpenFileName(
        parent, str(caption), str(start), str(filter)
    )
    return str(path)


def fluent_confirm(
    parent,
    title: str,
    text: str,
    *,
    confirm_text: str = "Confirm",
    cancel_text: str = "Cancel",
    kind: str = "warning",
) -> bool:
    """Ask one modal yes/no question with the shared Fluent chrome.

    One question, one dialog: it is retired once the answer has been read,
    for the reason :func:`fluent_message` gives."""
    dlg = _FluentMessageDialog(
        parent,
        title,
        text,
        kind=kind,
        confirm=True,
        confirm_text=confirm_text,
        cancel_text=cancel_text,
    )
    try:
        return dlg.exec_() == QtWidgets.QDialog.Accepted
    finally:
        retire_widget(dlg)


#: Max rows a Fluent drop-down shows before it SCROLLS (the shared fluent scrollbar) -- one source for
#: both the flat combo and the collapsible tree picker, so a long choice list never overflows the screen.
_COMBO_MAX_VISIBLE_ITEMS = 12

_COMBO_POPUP_STYLE_CACHE: tuple[float, str] | None = None


def _combo_popup_stylesheet() -> str:
    """The one visual declaration used by materialized list/tree popups."""

    global _COMBO_POPUP_STYLE_CACHE
    if (
        _COMBO_POPUP_STYLE_CACHE is not None
        and _COMBO_POPUP_STYLE_CACHE[0] == _FLUENT_SCALE
    ):
        return _COMBO_POPUP_STYLE_CACHE[1]
    gap = popup_gap()
    style = f"""
QAbstractItemView {{
    background: transparent;
    border: none;
    padding: {gap}px;
    color: {TEXT};
    selection-background-color: {SELECTED};
    selection-color: {TEXT};
    font: {fluent_font_size()}pt "{FONT}";
    outline: none;
}}
QAbstractItemView::item {{
    padding: {scaled_px(3, minimum=2)}px {scaled_px(6, minimum=4)}px;
    border: {FIELD_BORDER_PX}px solid transparent;
    border-radius: {_radius()}px;
    color: {TEXT};
}}
QAbstractItemView::item:hover {{
    background-color: {ACCENT_TINT};
    color: {TEXT};
}}
QAbstractItemView::item:selected {{
    background-color: {SELECTED};
    color: {TEXT};
}}
QAbstractItemView::item:selected:hover {{
    background-color: {SELECTED_HOVER};
}}
QAbstractItemView::corner {{
    background: transparent;
    border: none;
}}
{fluent_scrollbar_stylesheet('QScrollBar')}
"""
    _COMBO_POPUP_STYLE_CACHE = (_FLUENT_SCALE, style)
    return style


class FluentComboBox(QtWidgets.QAbstractButton):
    """A Fluent combo with an optional, lazily created text editor.

    The collapsed control owns a plain item model.  Its list/tree popup is a
    real :class:`FluentPopup` created only on first use, so Qt cannot silently
    materialize a private QListView while an ancestor stylesheet is polished.
    """

    activated = QtCore.pyqtSignal(int)
    currentIndexChanged = QtCore.pyqtSignal(int)
    currentTextChanged = QtCore.pyqtSignal(str)
    _line_edit = None

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(control_height())
        self.setCheckable(False)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred,
            QtWidgets.QSizePolicy.Fixed,
        )
        self._current_index = -1
        self._minimum_contents_length = 0
        self._model = QtGui.QStandardItemModel(self)
        self._popup_view: QtWidgets.QAbstractItemView | None = None
        self._popup: FluentPopup | None = None
        self._popup_anchor: FluentSettingsPopupAnchor | None = None
        self._popup_refit_pending = False
        self._cached_content_width: int | None = None
        self._bind_width_model(self._model)
        self.clicked.connect(self._toggle_popup)

    def model(self) -> QtCore.QAbstractItemModel:
        return self._model

    def setEditable(self, editable: bool) -> None:  # noqa: N802
        if bool(editable) == (self._line_edit is not None):
            return
        if editable:
            text = self.currentText()
            self._line_edit = FluentLineEdit(text, self)
            self._line_edit.setStyleSheet(
                f'QLineEdit {{ background: transparent; border: none; padding: 0; '
                f'color: {TEXT}; font: {fluent_font_size()}pt "{FONT}"; }}'
            )
            # The combo draws the field; its editor takes exactly the text
            # room inside it.  A field's own minimum height made the editor
            # taller than that room, so the typed text sat below the
            # collapsed text's line and below every label beside the box.
            self._line_edit.setMinimumHeight(0)
            self._line_edit.textChanged.connect(self.currentTextChanged)
            self._line_edit.installEventFilter(self)
            self.setFocusProxy(self._line_edit)
            self._line_edit.setGeometry(self._editor_rect())
            self._line_edit.show()
        else:
            self.setFocusProxy(None)
            retire_widget(self._line_edit)
            self._line_edit = None
        self.update()

    def isEditable(self) -> bool:  # noqa: N802
        return self._line_edit is not None

    def lineEdit(self):  # noqa: N802
        return self._line_edit

    @property
    def editingFinished(self):  # noqa: N802
        if self._line_edit is None:
            raise AttributeError("a non-editable combo has no text editor")
        return self._line_edit.editingFinished

    def setEditText(self, text: str) -> None:  # noqa: N802
        if self._line_edit is None:
            raise RuntimeError("this combo is not editable")
        self._line_edit.setText(str(text))

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self._line_edit is not None:
            self._line_edit.setGeometry(self._editor_rect())

    def modelColumn(self) -> int:  # noqa: N802 - Qt API name
        return 0

    def count(self) -> int:
        return int(self._model.rowCount())

    def addItem(self, text: object, userData: object = None) -> None:  # noqa: N802
        item = QtGui.QStandardItem(str(text))
        item.setData(userData, QtCore.Qt.UserRole)
        item.setEditable(False)
        self._model.appendRow(item)
        if self._current_index < 0:
            self.setCurrentIndex(0)

    def addItems(self, texts) -> None:  # noqa: N802 - Qt API name
        for text in texts:
            self.addItem(text)

    def clear(self) -> None:
        previous = self._current_index
        if self.count():
            self._model.removeRows(0, self.count())
        self._current_index = -1
        if self._line_edit is not None:
            self._line_edit.clear()
        if previous != -1:
            self.currentIndexChanged.emit(-1)
            self.currentTextChanged.emit("")
        self.update()

    def itemText(self, index: int) -> str:  # noqa: N802 - Qt API name
        model_index = self._model.index(int(index), 0)
        value = model_index.data(QtCore.Qt.DisplayRole)
        return "" if value is None else str(value)

    def itemData(
        self,
        index: int,
        role: int = QtCore.Qt.UserRole,
    ) -> object:  # noqa: N802
        return self._model.index(int(index), 0).data(role)

    def setItemData(
        self,
        index: int,
        value: object,
        role: int = QtCore.Qt.UserRole,
    ) -> None:  # noqa: N802
        model_index = self._model.index(int(index), 0)
        if model_index.isValid():
            self._model.setData(model_index, value, role)

    def findData(
        self,
        value: object,
        role: int = QtCore.Qt.UserRole,
    ) -> int:  # noqa: N802
        for index in range(self.count()):
            candidate = self.itemData(index, role)
            if type(candidate) is type(value) and candidate == value:
                return index
        return -1

    def findText(self, text: object) -> int:  # noqa: N802 - Qt API name
        selected = str(text)
        return next(
            (index for index in range(self.count()) if self.itemText(index) == selected),
            -1,
        )

    def currentIndex(self) -> int:  # noqa: N802 - Qt API name
        return self._current_index

    def setCurrentIndex(self, index: int) -> None:  # noqa: N802 - Qt API name
        selected = int(index)
        if selected < 0 or selected >= self.count():
            selected = -1
        if self._line_edit is not None:
            with signals_blocked(self._line_edit):
                self._line_edit.setText(self.itemText(selected) if selected >= 0 else "")
        if selected == self._current_index:
            return
        self._current_index = selected
        self.currentIndexChanged.emit(selected)
        self.currentTextChanged.emit(self.currentText())
        if self._popup_view is not None:
            self._sync_popup_selection()
        self.update()

    def currentText(self) -> str:  # noqa: N802 - Qt API name
        if self._line_edit is not None:
            return self._line_edit.text()
        return self.itemText(self._current_index) if self._current_index >= 0 else ""

    def setCurrentText(self, text: object) -> None:  # noqa: N802 - Qt API name
        index = self.findText(text)
        if index >= 0:
            self.setCurrentIndex(index)
        elif self._line_edit is not None:
            self.setEditText(str(text))

    def currentData(self, role: int = QtCore.Qt.UserRole) -> object:  # noqa: N802
        if self._line_edit is not None and self.currentText() != self.itemText(self._current_index):
            return None
        return self.itemData(self._current_index, role) if self._current_index >= 0 else None

    def minimumContentsLength(self) -> int:  # noqa: N802 - Qt API name
        return self._minimum_contents_length

    def setMinimumContentsLength(self, length: int) -> None:  # noqa: N802
        selected = max(0, int(length))
        if selected != self._minimum_contents_length:
            self._minimum_contents_length = selected
            self.updateGeometry()
            self._schedule_popup_refit()

    def _create_popup_view(self) -> QtWidgets.QAbstractItemView:
        """Create this combo family's one popup view, on first use only."""

        view = QtWidgets.QListView(self)
        view.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        view.viewport().setAutoFillBackground(False)
        view.viewport().setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        view_palette = view.palette()
        view_palette.setColor(QtGui.QPalette.Base, QtCore.Qt.transparent)
        view_palette.setColor(QtGui.QPalette.Window, QtCore.Qt.transparent)
        view.setPalette(view_palette)
        return view

    def _popup_view_created(self, view: QtWidgets.QAbstractItemView) -> None:
        """Subclass hook for signals owned by the one materialized view."""

        view.clicked.connect(self._commit_flat_index)
        view.activated.connect(self._commit_flat_index)

    def _commit_flat_index(self, index: QtCore.QModelIndex) -> None:
        if not index.isValid() or not (
            index.flags() & QtCore.Qt.ItemIsEnabled
            and index.flags() & QtCore.Qt.ItemIsSelectable
        ):
            return
        self.setCurrentIndex(index.row())
        self.hidePopup()
        # A completed popup choice returns keyboard focus to its collapsed
        # control. Popup teardown itself posts a later focus change, so an immediate
        # setFocus is overwritten after this callback returns.  Restore on
        # the next owner turn, after that teardown has completed.
        QtCore.QTimer.singleShot(0, self._restore_collapsed_focus)
        self.activated.emit(index.row())

    def _restore_collapsed_focus(self) -> bool:
        try:
            if self.isVisible() and self.isEnabled():
                self.setFocus(QtCore.Qt.MouseFocusReason)
                return True
        except RuntimeError:
            pass
        return False

    def _ensure_popup_view(self) -> QtWidgets.QAbstractItemView:
        view = self._popup_view
        if view is not None:
            return view
        view = self._create_popup_view()
        if not isinstance(view, QtWidgets.QAbstractItemView):
            raise TypeError("Fluent combo popup factory must return QAbstractItemView")
        view.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        view.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        view.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        view.installEventFilter(self)
        view.setStyleSheet(_combo_popup_stylesheet())
        view.setModel(self._model)
        self._popup_view = view
        self._popup_view_created(view)
        popup = FluentPopup(self)
        layout = QtWidgets.QVBoxLayout(popup)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(view)
        self._popup = popup
        self._popup_anchor = FluentSettingsPopupAnchor(popup, self)
        return view

    def view(self):  # noqa: N802 - Qt API name
        """Materialize the correct family view when a caller explicitly asks."""

        return self._ensure_popup_view()

    def eventFilter(self, watched, event):  # noqa: N802 - Qt API name
        if self._line_edit is not None and watched is self._line_edit:
            if event.type() in (QtCore.QEvent.FocusIn, QtCore.QEvent.FocusOut):
                self.update()
            elif event.type() == QtCore.QEvent.KeyPress and event.key() in (QtCore.Qt.Key_Up, QtCore.Qt.Key_Down, QtCore.Qt.Key_F4):
                self.keyPressEvent(event)
                return True
        if (
            watched is self._popup_view
            and event.type() == QtCore.QEvent.KeyPress
            and event.key() == QtCore.Qt.Key_Escape
        ):
            self.hidePopup()
            return True
        return super().eventFilter(watched, event)

    def _bind_width_model(self, model: QtCore.QAbstractItemModel) -> None:
        callback = self._invalidate_content_width
        for name in (
            "rowsInserted",
            "rowsRemoved",
            "rowsMoved",
            "columnsInserted",
            "columnsRemoved",
            "columnsMoved",
            "modelReset",
            "layoutChanged",
        ):
            signal = getattr(model, name, None)
            if signal is not None:
                signal.connect(callback)
        data_changed = getattr(model, "dataChanged", None)
        if data_changed is not None:
            data_changed.connect(self._model_data_changed)
        self._invalidate_content_width()

    def _model_data_changed(self, top_left, bottom_right, roles=()) -> None:
        display_changed = (
            not roles
            or QtCore.Qt.DisplayRole in roles
            or QtCore.Qt.EditRole in roles
        )
        if display_changed:
            self._invalidate_content_width()
        if (
            display_changed
            and self._current_index >= 0
            and top_left.row() <= self._current_index <= bottom_right.row()
            and top_left.column() <= 0 <= bottom_right.column()
        ):
            self.currentTextChanged.emit(self.currentText())
            self.update()

    def _invalidate_content_width(self, *_args) -> None:
        was_measured = self._cached_content_width is not None
        self._cached_content_width = None
        if was_measured:
            self.updateGeometry()
            self.update()
        self._schedule_popup_refit()

    def changeEvent(self, event) -> None:  # noqa: N802 - Qt API name
        super().changeEvent(event)
        if (
            event.type() == QtCore.QEvent.EnabledChange
            and not self.isEnabled()
            and hasattr(self, "_popup")
        ):
            self.hidePopup()
        if event.type() in {
            QtCore.QEvent.FontChange,
            QtCore.QEvent.ApplicationFontChange,
            QtCore.QEvent.StyleChange,
        } and hasattr(self, "_cached_content_width"):
            # The floor follows the scale, but it is a FLOOR: an owner that
            # fixed this combo's height owns it.  Re-applying unconditionally
            # dropped the minimum back under the owner's number on the first
            # style polish, so a combo asked for a 26 px row settled at 22 and
            # every row laid out beside it drifted.
            if self.minimumHeight() != self.maximumHeight():
                self.setMinimumHeight(control_height())
            self._invalidate_content_width()

    def event(self, event) -> bool:
        result = super().event(event)
        # Qt5's PyQt enum omits ScreenChangeInternal, whose stable Qt value is
        # 216.  Keep it in the derived-metrics owner rather than spreading a
        # native-screen callback through every consumer.
        if int(event.type()) == 216 and hasattr(self, "_cached_content_width"):
            self._invalidate_content_width()
        return result

    def hideEvent(self, event) -> None:  # noqa: N802 - Qt API name
        if hasattr(self, "_popup"):
            self.hidePopup()
        super().hideEvent(event)

    def _collapsed_font(self) -> QtGui.QFont:
        return QtGui.QFont(self.font())

    @staticmethod
    def _collapsed_text_chrome() -> tuple[int, int, int]:
        """Return ``(left inset, right pad, drop width)`` for collapsed text.

        The inset is a line edit's own: the edge, the field padding and the
        text margin Qt keeps inside a QLineEdit, so the text of a combo
        starts on the column a line edit or spin box beside it starts on.
        The drop width is the stepper column and the edge beside it."""

        pad = scaled_px(EDIT_PADDING_H)
        return (
            FIELD_BORDER_PX + pad + _QLINEEDIT_HMARGIN,
            pad,
            scaled_px(COMBO_WIDTH) + FIELD_BORDER_PX,
        )

    def _collapsed_text_rect(self) -> QtCore.QRect:
        """Where the collapsed text's line box sits: on the line every
        field's text takes (:func:`centred_line_top` of the box), so the
        typed text of an editable combo, a line edit beside it and the
        label before it share its baseline."""

        left, right, drop_width = self._collapsed_text_chrome()
        width = max(0, self.width() - drop_width - left - right)
        font = self._collapsed_font()
        line = QtGui.QFontMetrics(font).height()
        return QtCore.QRect(left, centred_line_top(0, self.height(), font), width, line)

    def _editor_rect(self) -> QtCore.QRect:
        """The editable combo's line edit: the collapsed text's columns plus
        the inset the line edit keeps before its text, and the room inside
        the edge and padding, as high above the box's middle as below it --
        the edit seats its line on its own middle, so typed text sits
        exactly where the collapsed text was."""

        text = self._collapsed_text_rect()
        edge = FIELD_BORDER_PX + scaled_px(PADDING_V)
        return QtCore.QRect(
            text.left() - _QLINEEDIT_HMARGIN, edge,
            text.width() + 2 * _QLINEEDIT_HMARGIN, max(0, self.height() - 2 * edge),
        )

    def _collapsed_text_candidates(self) -> tuple[str, ...]:
        return tuple(self.itemText(index) for index in range(self.count()))

    def _collapsed_content_width(self) -> int:
        if self._cached_content_width is None:
            # CEILED, from the fractional advance.  The painter elides with
            # the text engine's fractional width, and the integer metric
            # rounds it -- so a box sized to exactly its widest item, which
            # is what a Fixed-width combo is, was a fraction of a pixel too
            # narrow for that item and painted it with an ellipsis: a power
            # picker sized to "dBm" showed "dB...".
            metrics = QtGui.QFontMetricsF(self._collapsed_font())
            self._cached_content_width = max(
                (
                    int(math.ceil(metrics.horizontalAdvance(text)))
                    for text in self._collapsed_text_candidates()
                ),
                default=0,
            )
        return self._cached_content_width

    def sizeHint(self):  # noqa: N802 - Qt API name
        font = self._collapsed_font()
        metrics = QtGui.QFontMetrics(font)
        minimum_text = (
            fluent_text_width(metrics, "M" * self.minimumContentsLength())
            if self.minimumContentsLength() > 0
            else 0
        )
        content_width = max(self._collapsed_content_width(), minimum_text)
        left, right, drop_width = self._collapsed_text_chrome()
        return QtCore.QSize(
            content_width + left + right + drop_width,
            max(self.minimumHeight(), metrics.height() + scaled_px(4)),
        )

    def minimumSizeHint(self):  # noqa: N802 - Qt API name
        return self.sizeHint()

    def paintEvent(self, event):
        """A field like a line edit beside it, with a chevron and no block.

        The solid accent block every combo carried repeated down a whole
        form, and an open combo filled with tint.  The field says its states
        the way an edit does -- the accent edge under the pointer or while it
        is open -- and only a cycle choice whose wheel is ACTIVE takes the
        SELECTED fill, because that is a state it holds; its edge stays the
        field's own.
        """

        del event
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        radius = float(_radius())
        outer = frame_stroke_rect(self)
        enabled = self.isEnabled()
        current = enabled and self.isChecked()
        engaged = enabled and (
            (self._popup is not None and self._popup.isVisible())
            or (self._line_edit is not None and self._line_edit.hasFocus())
            or self.underMouse()
        )
        fill = SELECTED if current else SURFACE if enabled else BG
        edge = ACCENT if engaged else PLACEHOLDER
        painter.setPen(frame_pen(self, edge))
        painter.setBrush(QtGui.QColor(fill))
        painter.drawRoundedRect(outer, radius, radius)

        if self._line_edit is None:
            text_rect = self._collapsed_text_rect()
            painter.setPen(QtGui.QColor(TEXT if enabled else DISABLED_INK))
            painter.setFont(self._collapsed_font())
            text = painter.fontMetrics().elidedText(
                self._display_text(), QtCore.Qt.ElideRight, text_rect.width()
            )
            painter.drawText(text_rect, QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop, text)

        # The one chevron, on the stepper column a spin box below would use,
        # centred on the frame's inner rect like every part inside a field.
        paint_chevron(
            painter, self,
            QtCore.QPointF(stepper_column(self).center().x(), frame_inner_rect(self).center().y()),
            up=False,
            colour=QtGui.QColor(TEXT if current else GRAPHITE if enabled else PLACEHOLDER),
        )
        painter.end()

    def enterEvent(self, event) -> None:  # noqa: N802 - Qt API name
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802 - Qt API name
        self.update()
        super().leaveEvent(event)

    def showPopup(self):  # noqa: N802 - Qt naming
        view = self._ensure_popup_view()
        popup = self._popup
        if popup is None:
            raise RuntimeError("Fluent combo popup owner was not constructed")
        view.setMinimumHeight(0)
        view.setMaximumHeight(QtWidgets.QWIDGETSIZE_MAX)
        popup.setMinimumHeight(0)
        popup.setMaximumHeight(QtWidgets.QWIDGETSIZE_MAX)
        self._sync_popup_selection()
        self._resize_popup_to_contents()
        self._place_popup()
        popup.show()
        popup.raise_()
        view.setFocus(QtCore.Qt.PopupFocusReason)
        self.update()

    def hidePopup(self) -> None:  # noqa: N802 - Qt API name
        if self._popup is not None:
            self._popup.hide()
            self.update()

    def _toggle_popup(self) -> None:
        view = self._ensure_popup_view()
        anchor = self._popup_anchor
        if anchor is None:
            raise RuntimeError("Fluent combo popup anchor was not constructed")
        anchor.toggle(view, present=self.showPopup)

    def _popup_available_geometry(self) -> QtCore.QRect | None:
        """The screen containing the anchor, shared by cap/space/placement."""

        return _available_geometry_at(
            self.mapToGlobal(self.rect().center()), widget=self
        )

    def _popup_width_cap(self) -> int:
        available = self._popup_available_geometry()
        return (
            QtWidgets.QWIDGETSIZE_MAX
            if available is None
            else int(available.width())
        )

    def _popup_space_below(self) -> tuple[int, QtCore.QRect | None]:
        """Return usable space below the field on its active screen.

        Fluent combo popups have one spatial contract: the field remains the
        visible anchor and overflow is handled by the view's native scrollbar.
        Qt's default "pick the roomier side" policy is deliberately not part of
        that contract because it makes a list jump above the control near the
        bottom of a screen.
        """

        available = self._popup_available_geometry()
        if available is None:
            return (QtWidgets.QWIDGETSIZE_MAX, None)
        bottom = self.mapToGlobal(QtCore.QPoint(0, self.height())).y()
        below_top = max(available.top(), bottom + popup_gap())
        below = max(0, available.bottom() - below_top + 1)
        return below, available

    def _popup_origin(self, popup_width: int) -> QtCore.QPoint:
        anchor_bottom = self.mapToGlobal(QtCore.QPoint(0, self.height()))
        left = (
            anchor_bottom.x()
            if popup_width <= self.width()
            else anchor_bottom.x() + self.width() - popup_width
        )
        available = self._popup_available_geometry()
        top = anchor_bottom.y() + popup_gap()
        if available is not None:
            left = max(available.left(), min(left, available.right() - popup_width + 1))
        return QtCore.QPoint(int(left), int(top))

    def _place_popup(self) -> None:
        """Anchor the open drop-down below its field.

        Wide content grows left from the field's right edge; vertical overflow
        never changes the anchor and is instead consumed by native scrolling.
        The popup is positioned once from the current anchor geometry.
        """
        popup = self._popup
        if popup is None:
            return
        popup.move(self._popup_origin(popup.width()))

    def _visible_popup_row_count(self) -> int:
        return min(self.count(), _COMBO_MAX_VISIBLE_ITEMS)

    def _popup_row_total(self) -> int:
        """Every row the open view holds; more than are shown is what the
        vertical bar is for."""

        return self.count()

    def _apply_popup_geometry(
        self, width: int, height: int, *, column_width: int
    ) -> None:
        """Give the popup this geometry and its one column this width."""

        view = self._popup_view
        popup = self._popup
        if view is None or popup is None:
            return
        layout = popup.layout()
        if layout is not None:
            layout.setContentsMargins(0, popup_gap(), 0, 0)
        # THE COLUMN FIRST.  A tree keeps a header even when it hides it,
        # and the header's length is what its horizontal range is computed
        # from on every geometry update -- so the section is set before the
        # size that triggers one, never from a viewport read back afterwards.
        header = getattr(view, "header", None)
        if callable(header):
            header().resizeSection(self.modelColumn(), int(column_width))
        # The popup is the sole geometry owner.  Its zero-horizontal-margin
        # layout gives the view the exact remaining rectangle; fixing both
        # parent and child would create two competing width authorities.
        view.setMinimumSize(0, 0)
        view.setMaximumSize(
            QtWidgets.QWIDGETSIZE_MAX,
            QtWidgets.QWIDGETSIZE_MAX,
        )
        popup.setFixedSize(max(1, int(width)), max(0, int(height)))
        if layout is not None:
            layout.activate()
        view.doItemsLayout()

    def _resize_popup_to_contents(self, *_) -> None:
        """The popup's geometry, a pure function of its content.

        The rows revealed, the delegate's row height, the content's column
        width, the collapsed field's width, the space below it and the
        screen's width: everything the geometry follows from is known before
        the popup is shown, and nothing is read back from the scroll area.
        It used to be -- ``view.width - viewport.width`` stood in for the
        frame-and-scrollbar chrome -- and a scroll area lays itself out
        lazily: hidden, a resize is only pended; visible, the vertical bar
        shows or hides on a queued slot.  So the read was the PREVIOUS
        opening's chrome, each opening left the opposite residue, and the
        same picker alternated between a clean card and one with two
        scrollbars cutting its rows.  Which bars there are is decided here,
        from the numbers, and the view is told.
        """

        view = self._popup_view
        popup = self._popup
        if view is None or popup is None:
            return
        popup.ensurePolished()
        view.ensurePolished()
        column = self.modelColumn()
        rows = self._visible_popup_row_count()
        row_height = int(view.sizeHintForRow(0)) if rows > 0 else 0
        if rows > 0 and row_height <= 0:
            row_height = view.fontMetrics().height() + scaled_px(8)
        # The view's frame is its border with the stylesheet padding folded
        # in, on every side; above the view the popup keeps one gap.
        side = int(view.frameWidth())
        content_width = max(0, int(view.sizeHintForColumn(column)))
        content_height = popup_gap() + 2 * side + rows * row_height
        below, _available = self._popup_space_below()
        bar = fluent_scrollbar_thickness()
        needs_vertical = self._popup_row_total() > rows or content_height > below
        width = min(
            self._popup_width_cap(),
            max(
                self.width(),
                content_width + 2 * side + (bar if needs_vertical else 0),
            ),
        )
        viewport_width = max(0, width - 2 * side - (bar if needs_vertical else 0))
        needs_horizontal = content_width > viewport_width
        height = min(below, content_height + (bar if needs_horizontal else 0))
        view.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarAlwaysOn
            if needs_vertical
            else QtCore.Qt.ScrollBarAlwaysOff
        )
        view.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarAlwaysOn
            if needs_horizontal
            else QtCore.Qt.ScrollBarAlwaysOff
        )
        self._apply_popup_geometry(
            width, height, column_width=max(content_width, viewport_width)
        )
        if popup.isVisible():
            self._place_popup()

    def _schedule_popup_refit(self) -> None:
        popup = getattr(self, "_popup", None)
        if (
            popup is None
            or not popup.isVisible()
            or self._popup_refit_pending
        ):
            return
        self._popup_refit_pending = True
        QtCore.QTimer.singleShot(0, self._refit_open_popup)

    def _refit_open_popup(self) -> None:
        self._popup_refit_pending = False
        popup = self._popup
        if popup is not None and popup.isVisible():
            self._resize_popup_to_contents()

    def _display_text(self) -> str:
        """The text painted in the COLLAPSED combo (the seam a tree combo overrides to show a
        fuller source-qualified name).  Default: the current item's own text."""
        return self.currentText()

    def _sync_popup_selection(self) -> None:
        """Project the collapsed flat selection into the materialized view."""

        view = self._popup_view
        if view is None:
            return
        index = self._model.index(self._current_index, 0)
        if not index.isValid():
            view.clearSelection()
            view.setCurrentIndex(QtCore.QModelIndex())
            return
        view.setCurrentIndex(index)
        view.selectionModel().select(
            index,
            QtCore.QItemSelectionModel.ClearAndSelect
            | QtCore.QItemSelectionModel.Rows,
        )
        view.scrollTo(index)

    def wheelEvent(self, event):
        view = self._popup_view
        if view is not None and view.isVisible():
            super().wheelEvent(event)
            return
        event.ignore()

    def _selectable_row(self, start: int, step: int) -> int:
        row = int(start)
        while 0 <= row < self.count():
            flags = self._model.index(row, 0).flags()
            if (
                flags & QtCore.Qt.ItemIsEnabled
                and flags & QtCore.Qt.ItemIsSelectable
            ):
                return row
            row += int(step)
        return -1

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt API name
        key = event.key()
        if key in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter, QtCore.Qt.Key_Space, QtCore.Qt.Key_F4):
            self._toggle_popup()
            event.accept()
            return
        if key == QtCore.Qt.Key_Escape and self._popup is not None:
            self.hidePopup()
            event.accept()
            return
        if key in (
            QtCore.Qt.Key_Up,
            QtCore.Qt.Key_Down,
            QtCore.Qt.Key_Home,
            QtCore.Qt.Key_End,
        ) and self.count():
            step = -1 if key == QtCore.Qt.Key_Up else 1
            if key == QtCore.Qt.Key_Home:
                step, start = 1, 0
            elif key == QtCore.Qt.Key_End:
                step, start = -1, self.count() - 1
            elif self._current_index < 0:
                start = self.count() - 1 if step < 0 else 0
            else:
                start = self._current_index + step
            candidate = self._selectable_row(start, step)
            if candidate >= 0 and candidate != self._current_index:
                self.setCurrentIndex(candidate)
                self.activated.emit(candidate)
            event.accept()
            return
        super().keyPressEvent(event)


class FluentCycleComboBox(FluentComboBox):
    """A small popup choice with one lazily cycled sub-domain.

    The popup contains ordinary actions plus one action such as ``Scope``.
    Selecting that action paints ``Scope: <value>`` in the collapsed control;
    its text toggles wheel activation without opening the popup, while the
    arrow still opens the ordinary choices. A wheel changes ``<value>`` only
    while activated, focused and under the pointer. The
    sub-domain never enters the Qt item model, so a large scientific axis does
    not create a large popup or thousands of QStandardItems.
    """

    def __init__(self, parent=None):
        self._cycle_choices: Sequence[tuple[object, str]] | None = None
        self._cycle_label = ""
        self._cycle_row = -1
        self._cycle_position = -1
        super().__init__(parent)
        self.setCheckable(True)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)

    def _set_cycle_active(self, active: bool) -> None:
        if active == self.isChecked():
            return
        self.setChecked(active)
        app = QtWidgets.QApplication.instance()
        if app is not None:
            if active:
                app.installEventFilter(self)
            else:
                app.removeEventFilter(self)

    def nextCheckState(self) -> None:  # noqa: N802 - Qt API name
        # Arrow clicks are ordinary popup actions, not activation toggles.
        pass

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt API name
        _left, _right, drop_width = self._collapsed_text_chrome()
        if (
            event.button() == QtCore.Qt.LeftButton
            and self.isCycleSelected()
            and event.pos().x() < self.width() - drop_width
        ):
            self.setFocus(QtCore.Qt.MouseFocusReason)
            self._set_cycle_active(not self.isChecked())
            event.accept()
            return
        super().mousePressEvent(event)

    def showPopup(self) -> None:  # noqa: N802 - Qt API name
        self._set_cycle_active(False)
        super().showPopup()

    def _restore_collapsed_focus(self) -> bool:
        restored = super()._restore_collapsed_focus()
        if restored:
            self._set_cycle_active(self.hasFocus() and self.isCycleSelected())
        return restored

    def focusOutEvent(self, event) -> None:  # noqa: N802 - Qt API name
        self._set_cycle_active(False)
        super().focusOutEvent(event)

    def hideEvent(self, event) -> None:  # noqa: N802 - Qt API name
        self._set_cycle_active(False)
        super().hideEvent(event)

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt API name
        if (
            event.type() == QtCore.QEvent.MouseButtonPress
            and self.isChecked()
            and isinstance(watched, QtWidgets.QWidget)
            and watched is not self
            and not self.isAncestorOf(watched)
        ):
            # Blank labels/pages may have NoFocus, so focusOut alone cannot
            # represent an explicit click elsewhere. Listen only while active.
            self._set_cycle_active(False)
        return super().eventFilter(watched, event)

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt API name
        if event.key() == QtCore.Qt.Key_Escape and self.isChecked():
            self._set_cycle_active(False)
            event.accept()
            return
        super().keyPressEvent(event)

    @staticmethod
    def _typed_equal(left: object, right: object) -> bool:
        return type(left) is type(right) and bool(left == right)

    def clear(self) -> None:
        super().clear()
        self._cycle_choices = None
        self._cycle_label = ""
        self._cycle_row = -1
        self._cycle_position = -1

    def setCycleChoices(  # noqa: N802 - Qt API name
        self,
        label: str,
        choices: Sequence[tuple[object, str]],
    ) -> None:
        """Install the one popup action backed by ``choices``.

        Re-installable, because the domain behind the action is DATA: an
        axis that is still filling offers one more coordinate per shot.
        Requiring a clear() first meant the only way to say "one more
        coordinate" was to rebuild the whole popup -- which closed an open
        dropdown and swallowed the click already travelling to it, once
        per shot.  The action keeps its row; only what it cycles through
        changes, and a position already chosen is kept where it still
        exists -- looked for at that POSITION first, because an axis that
        grew by one shot still holds the same coordinate where it held it,
        and a domain of a million coordinates is only scanned when the
        coordinate really moved.
        """

        if not isinstance(label, str) or not label.strip():
            raise ValueError("cycle label must be non-empty text")
        if not isinstance(choices, Sequence) or not len(choices):
            raise ValueError("cycle choices must be a non-empty sequence")
        previous = None
        if self._cycle_row >= 0 and self._cycle_choices is not None:
            if 0 <= self._cycle_position < len(self._cycle_choices):
                previous = self._cycle_choices[self._cycle_position]
        held = self._cycle_position
        self._cycle_choices = choices
        self._cycle_label = label.strip()
        self._cycle_position = 0
        if previous is not None:
            self._cycle_position = max(0, self._position_of(previous[0], first=held))
        if self._cycle_row < 0:
            self._cycle_row = self.count()
            self.addItem(self._cycle_label)
        elif self.itemText(self._cycle_row) != self._cycle_label:
            self.setItemData(self._cycle_row, self._cycle_label, QtCore.Qt.DisplayRole)

    def _position_of(self, value: object, *, first: int = -1) -> int:
        """Where ``value`` sits in the lazy domain, or -1 when it does not.

        ``first`` is a position to try before scanning -- the one the value
        held a moment ago.  The domain is a lazy sequence, each read of which
        may reach into the data, so the common questions ("is it still where
        it was", "select the value already selected") must cost one read and
        not the whole axis.
        """

        choices = self._cycle_choices
        if choices is None:
            raise RuntimeError("cycle choices are not configured")
        if 0 <= first < len(choices) and self._typed_equal(
            self._cycle_choice(first)[0], value
        ):
            return first
        # A domain that locates a value itself (an axis's Scope domain) is
        # asked, and one position is read; only a plain sequence is walked.
        locate = getattr(choices, "position_of", None)
        if locate is not None:
            position = locate(value)
            if position is not None and self._typed_equal(
                self._cycle_choice(position)[0], value
            ):
                return int(position)
            return -1
        return next(
            (
                position
                for position in range(len(choices))
                if self._typed_equal(self._cycle_choice(position)[0], value)
            ),
            -1,
        )

    def _cycle_choice(self, position: int) -> tuple[object, str]:
        choices = self._cycle_choices
        if choices is None:
            raise RuntimeError("cycle choices are not configured")
        choice = choices[int(position)]
        if (
            not isinstance(choice, tuple)
            or len(choice) != 2
            or not isinstance(choice[1], str)
            or not choice[1].strip()
        ):
            raise TypeError(
                "cycle choices must yield (value, non-empty label) pairs"
            )
        return choice[0], choice[1]

    def setCyclePosition(self, position: int) -> None:  # noqa: N802 - Qt API name
        """Select one lazy sub-domain position without scanning its values."""

        choices = self._cycle_choices
        selected = int(position)
        if choices is None:
            raise RuntimeError("cycle choices are not configured")
        if not 0 <= selected < len(choices):
            raise IndexError("cycle position is outside the configured choices")
        changed_value = selected != self._cycle_position
        self._cycle_position = selected
        changed_row = self.currentIndex() != self._cycle_row
        super().setCurrentIndex(self._cycle_row)
        if changed_value and not changed_row:
            self.currentTextChanged.emit(self.currentText())
            self.update()

    def cyclePosition(self) -> int:  # noqa: N802 - Qt API name
        """Return the selected lazy-domain position without reverse lookup."""

        if self._cycle_choices is None or self._cycle_position < 0:
            raise RuntimeError("cycle choices are not selected")
        return self._cycle_position

    def isCycleSelected(self) -> bool:  # noqa: N802 - Qt API name
        return self.currentIndex() == self._cycle_row and self._cycle_position >= 0

    def setCurrentIndex(self, index: int) -> None:  # noqa: N802 - Qt API name
        selected = int(index)
        if selected == self._cycle_row and self._cycle_position < 0:
            self._cycle_position = 0
        super().setCurrentIndex(selected)
        # A form may clear/rebuild ordinary rows before restoring its cycle
        # value. Keep activation across that intermediate row population.
        if self._cycle_row >= 0 and selected != self._cycle_row:
            self._set_cycle_active(False)

    def currentData(self, role: int = QtCore.Qt.UserRole) -> object:  # noqa: N802
        if (
            role == QtCore.Qt.UserRole
            and self.currentIndex() == self._cycle_row
            and self._cycle_position >= 0
        ):
            return self._cycle_choice(self._cycle_position)[0]
        return super().currentData(role)

    def currentText(self) -> str:  # noqa: N802 - Qt API name
        if self.currentIndex() == self._cycle_row and self._cycle_position >= 0:
            return f"{self._cycle_label}: {self._cycle_choice(self._cycle_position)[1]}"
        return super().currentText()

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt API name
        view = self._popup_view
        if view is not None and view.isVisible():
            super().wheelEvent(event)
            return
        if (
            not self.isChecked()
            or not self.hasFocus()
            or not self.isCycleSelected()
            or not self.rect().contains(event.pos())
        ):
            event.ignore()
            return
        delta = int(event.angleDelta().y())
        if delta == 0:
            event.ignore()
            return
        choices = self._cycle_choices
        if choices is None:
            event.ignore()
            return
        step = -1 if delta > 0 else 1
        candidate = self._cycle_position + step
        current_value = self._cycle_choice(self._cycle_position)[0]
        while 0 <= candidate < len(choices):
            if not self._typed_equal(
                self._cycle_choice(candidate)[0], current_value
            ):
                break
            candidate += step
        if not (0 <= candidate < len(choices)):
            event.accept()
            return
        self._cycle_position = candidate
        self.currentTextChanged.emit(self.currentText())
        self.update()
        self.activated.emit(self._cycle_row)
        event.accept()


class _ExpandableTreeView(QtWidgets.QTreeView):
    """A ``QTreeView`` whose PARENT (header) rows toggle expansion on a click ANYWHERE on the row
    -- the text as well as the disclosure triangle -- so a signal-group header is easy to open
    (G2 follow-up: "click the name OR the arrow"), exactly once (the click is swallowed so Qt's
    native triangle toggle never double-fires).  LEAF rows fall through to the normal click path
    (``clicked`` -> ``FluentTreeComboBox._commit_tree_index`` selects them).  Reusable: the behaviour
    lives on the widget, not in any caller."""

    def mousePressEvent(self, event):  # noqa: N802 - Qt naming
        index = self.indexAt(event.pos())
        model = index.model() if index.isValid() else None
        item = model.itemFromIndex(index) if isinstance(model, QtGui.QStandardItemModel) else None
        # a source header = has children + carries no leaf payload (UserRole) -> toggle on any click
        if item is not None and item.hasChildren() and item.data(QtCore.Qt.UserRole) is None:
            self.setExpanded(index, not self.isExpanded(index))
            return                                # swallow: no native double-toggle, no header "select"
        super().mousePressEvent(event)


class FluentTreeComboBox(FluentComboBox):
    """A keyed-choice picker whose drop-down is a collapsible source tree.

    Each leaf carries an opaque choice key and the collapsed box paints the
    injected source-qualified label.  The widget owns only the Qt model; the
    choice projection remains in the companion plain-data picker helpers.
    """

    #: emitted with the opaque choice key when the user picks a leaf
    keyed_choice_picked = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._full_by_key: dict[str, str] = {}
        self._selected_index = QtCore.QPersistentModelIndex()

    def _create_popup_view(self) -> QtWidgets.QAbstractItemView:
        tree = _ExpandableTreeView(self)         # parent rows toggle on a name OR triangle click
        tree.setHeaderHidden(True)
        # The column has one owner: the popup sizes it from its content.
        tree.header().setStretchLastSection(False)
        tree.setRootIsDecorated(True)            # show the expand/collapse triangles
        tree.setItemsExpandable(True)
        tree.setUniformRowHeights(True)
        tree.setExpandsOnDoubleClick(False)      # single click on the triangle expands
        tree.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        tree.viewport().setAutoFillBackground(False)
        tree.viewport().setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        pal = tree.palette()
        pal.setColor(QtGui.QPalette.Base, QtCore.Qt.transparent)
        pal.setColor(QtGui.QPalette.Window, QtCore.Qt.transparent)
        tree.setPalette(pal)
        return tree

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt API name
        if event.key() in (
            QtCore.Qt.Key_Up,
            QtCore.Qt.Key_Down,
            QtCore.Qt.Key_Home,
            QtCore.Qt.Key_End,
        ):
            self.showPopup()
            event.accept()
            return
        super().keyPressEvent(event)

    def _popup_view_created(self, view: QtWidgets.QAbstractItemView) -> None:
        if not isinstance(view, _ExpandableTreeView):
            raise TypeError("Fluent tree combo requires its tree popup view")
        view.clicked.connect(self._commit_tree_index)
        view.activated.connect(self._commit_tree_index)
        # When a parent expands/collapses, RE-GROW the open popup so the children are fully visible
        # (Qt sizes the popup once at open and never re-fits a tree expand) -- the box "变长".
        view.expanded.connect(self._resize_popup_to_contents)
        view.collapsed.connect(self._resize_popup_to_contents)

    def _collapsed_text_candidates(self) -> tuple[str, ...]:
        none_labels = tuple(
            item.text()
            for row in range(self._model.rowCount())
            for item in (self._model.item(row),)
            if item is not None and item.data(QtCore.Qt.UserRole) == ""
        )
        return (*none_labels, *self._full_by_key.values())

    def _popup_row_total(self) -> int:
        return self._visible_popup_row_count()

    def _visible_popup_row_count(self) -> int:
        """Count the rows revealed by the current tree expansion state."""

        view = self._popup_view
        if not isinstance(view, QtWidgets.QTreeView):
            return 0
        model = self._model
        rows = 0

        def _visit(parent_index):
            nonlocal rows
            for r in range(model.rowCount(parent_index)):
                rows += 1
                idx = model.index(r, 0, parent_index)
                if view.isExpanded(idx):
                    _visit(idx)

        _visit(QtCore.QModelIndex())
        return rows

    def _display_text(self) -> str:
        """Derive the collapsed text live from the current key and source label."""
        key = self.current_choice_key()
        full = self._full_by_key.get(key, "")
        if full:
            return full
        index = self._selected_model_index()
        item = self._model.itemFromIndex(index) if index.isValid() else None
        return "" if item is None else item.text()

    def _selected_model_index(self) -> QtCore.QModelIndex:
        if not self._selected_index.isValid():
            return QtCore.QModelIndex()
        return QtCore.QModelIndex(self._selected_index)

    def currentIndex(self) -> int:  # noqa: N802 - Qt API name
        index = self._selected_model_index()
        return index.row() if index.isValid() else -1

    def currentText(self) -> str:  # noqa: N802 - Qt API name
        index = self._selected_model_index()
        item = self._model.itemFromIndex(index) if index.isValid() else None
        return "" if item is None else item.text()

    def currentData(self, role: int = QtCore.Qt.UserRole) -> object:  # noqa: N802
        index = self._selected_model_index()
        item = self._model.itemFromIndex(index) if index.isValid() else None
        return None if item is None else item.data(role)

    def _set_selected_index(self, index: QtCore.QModelIndex) -> bool:
        previous = self._selected_model_index()
        if previous == index:
            return False
        self._selected_index = (
            QtCore.QPersistentModelIndex(index)
            if index.isValid()
            else QtCore.QPersistentModelIndex()
        )
        self.currentIndexChanged.emit(self.currentIndex())
        self.currentTextChanged.emit(self.currentText())
        self.update()
        return True

    def _sync_popup_selection(self) -> None:
        view = self._popup_view
        if not isinstance(view, QtWidgets.QTreeView):
            return
        index = self._selected_model_index()
        if not index.isValid():
            view.clearSelection()
            view.setCurrentIndex(QtCore.QModelIndex())
            return
        parent = index.parent()
        if parent.isValid():
            view.setExpanded(parent, True)
        view.setCurrentIndex(index)
        view.selectionModel().select(
            index,
            QtCore.QItemSelectionModel.ClearAndSelect
            | QtCore.QItemSelectionModel.Rows,
        )
        view.scrollTo(index)

    def _commit_tree_index(self, index: QtCore.QModelIndex) -> None:
        item = self._model.itemFromIndex(index)
        if item is None or item.data(QtCore.Qt.UserRole) is None:
            return
        self._set_selected_index(index)
        self.hidePopup()
        # repaint() NOT update(): the box lives inside the Setting Qt.Popup,
        # whose child popup has just closed, so a queued update can otherwise
        # leave the collapsed source-qualified label stale until reopen.
        self.repaint()
        self.activated.emit(item.row())
        self.keyed_choice_picked.emit(self.current_choice_key())

    def _find_choice_index(self, key: str) -> QtCore.QModelIndex:
        match = QtCore.QModelIndex()

        def visit(parent: QtCore.QModelIndex) -> bool:
            nonlocal match
            for row in range(self._model.rowCount(parent)):
                index = self._model.index(row, 0, parent)
                item = self._model.itemFromIndex(index)
                if item is None:
                    continue
                payload = item.data(QtCore.Qt.UserRole)
                if payload is not None and str(payload).strip() == key:
                    match = index
                    return True
                if item.hasChildren() and visit(index):
                    return True
            return False

        visit(QtCore.QModelIndex())
        return match

    def select_choice_key(self, bare_key) -> bool:
        """Select one nested choice leaf by its opaque ``UserRole`` key.

        A flat ``findData`` only searches top-level rows.  Choice leaves live
        below source rows, so using it to restore a
        saved value silently misses every real choice.  This method is the one
        programmatic-selection path for the tree: it walks all nested leaves,
        stores one persistent model index, and paints the collapsed
        source-qualified label immediately without materializing a view.

        ``None`` and ``""`` clear the selection unless the model contains an
        explicit empty leaf (the optional ``none_label`` row), in which case
        that row is selected.  A non-empty key that is absent also leaves the
        combo blank and returns ``False``; it never lands on a source header.
        """

        key = "" if bare_key is None else str(bare_key).strip()
        match = self._find_choice_index(key)
        found = match.isValid()
        if self._set_selected_index(
            match if found else QtCore.QModelIndex()
        ):
            self._sync_popup_selection()
            self.repaint()
        return found

    def set_choice_tree(self, groups, *, current="", none_label=None) -> None:
        """Reconcile one grouped choice topology without rebuilding its items."""

        normalized = []
        source_names: set[str] = set()
        choice_keys: set[str] = set()
        full_by_key: dict[str, str] = {}
        for source, leaves in groups:
            source_name = str(source)
            if source_name in source_names:
                raise ValueError("choice tree contains a duplicate source")
            source_names.add(source_name)
            normalized_leaves = []
            for leaf_label, bare, full_label in leaves:
                choice_key = str(bare)
                if not choice_key or choice_key in choice_keys:
                    raise ValueError("choice tree contains a duplicate or empty key")
                choice_keys.add(choice_key)
                full = str(full_label)
                normalized_leaves.append((str(leaf_label), choice_key, full))
                full_by_key[choice_key] = full
            normalized.append((source_name, tuple(normalized_leaves)))

        view = self._popup_view
        expanded = set()
        vertical = horizontal = 0
        if isinstance(view, QtWidgets.QTreeView):
            vertical = view.verticalScrollBar().value()
            horizontal = view.horizontalScrollBar().value()
            expanded = {
                item.text()
                for row in range(self._model.rowCount())
                for item in (self._model.item(row),)
                if (
                    item is not None
                    and item.data(QtCore.Qt.UserRole) is None
                    and view.isExpanded(item.index())
                )
            }

        existing_none = None
        existing_parents = {}
        for row in range(self._model.rowCount()):
            item = self._model.item(row)
            if item is None:
                continue
            payload = item.data(QtCore.Qt.UserRole)
            if payload == "":
                existing_none = item
            elif payload is None:
                existing_parents[item.text()] = item

        desired_top = []
        if none_label is not None:
            none_item = existing_none or QtGui.QStandardItem()
            if none_item.text() != str(none_label):
                none_item.setText(str(none_label))
            if none_item.data(QtCore.Qt.UserRole) != "":
                none_item.setData("", QtCore.Qt.UserRole)
            if none_item.data(QtCore.Qt.UserRole + 1) != "":
                none_item.setData("", QtCore.Qt.UserRole + 1)
            if none_item.isEditable():
                none_item.setEditable(False)
            desired_top.append(none_item)

        for source, leaves in normalized:
            parent = existing_parents.get(source) or QtGui.QStandardItem()
            if parent.text() != source:
                parent.setText(source)
            if parent.isSelectable():
                parent.setSelectable(False)
            if parent.isEditable():
                parent.setEditable(False)
            font = parent.font()
            if not font.bold():
                font.setBold(True)
                parent.setFont(font)
            existing_children = {
                str(child.data(QtCore.Qt.UserRole)): child
                for row in range(parent.rowCount())
                for child in (parent.child(row),)
                if child is not None and child.data(QtCore.Qt.UserRole)
            }
            desired_children = []
            for leaf_label, bare, full_label in leaves:
                child = existing_children.get(bare) or QtGui.QStandardItem()
                if child.text() != leaf_label:
                    child.setText(leaf_label)
                if child.data(QtCore.Qt.UserRole) != bare:
                    child.setData(bare, QtCore.Qt.UserRole)
                if child.data(QtCore.Qt.UserRole + 1) != full_label:
                    child.setData(full_label, QtCore.Qt.UserRole + 1)
                if child.isEditable():
                    child.setEditable(False)
                desired_children.append(child)
            desired_child_ids = {id(child) for child in desired_children}
            for row in range(parent.rowCount() - 1, -1, -1):
                if id(parent.child(row)) not in desired_child_ids:
                    parent.removeRow(row)
            for row, child in enumerate(desired_children):
                if child.parent() is parent:
                    current_row = child.row()
                    if current_row != row:
                        parent.insertRow(row, parent.takeRow(current_row))
                else:
                    parent.insertRow(row, child)
            desired_top.append(parent)

        desired_top_ids = {id(item) for item in desired_top}
        for row in range(self._model.rowCount() - 1, -1, -1):
            if id(self._model.item(row)) not in desired_top_ids:
                self._model.removeRow(row)
        for row, item in enumerate(desired_top):
            if item.parent() is None and item.model() is self._model:
                current_row = item.row()
                if current_row != row:
                    self._model.insertRow(row, self._model.takeRow(current_row))
            else:
                self._model.insertRow(row, item)

        if self._full_by_key != full_by_key:
            self._full_by_key = full_by_key
            self._invalidate_content_width()
            self.update()
        self.select_choice_key(current)
        if isinstance(view, QtWidgets.QTreeView):
            # ``expanded`` is captured before the keyed model reconcile.  A
            # source can disappear while a live topology is refreshed; its
            # old QStandardItem is then owned/deleted by Qt when removed from
            # the model.  Never dereference that stale item.  Resolve the
            # expansion state from the reconciled model, which is the sole
            # widget model owner.
            reconciled_parents = {
                self._model.item(row).text(): self._model.item(row)
                for row in range(self._model.rowCount())
                if (
                    self._model.item(row) is not None
                    and self._model.item(row).data(QtCore.Qt.UserRole) is None
                )
            }
            for source in expanded:
                parent = reconciled_parents.get(source)
                if parent is not None:
                    view.setExpanded(parent.index(), True)
            view.verticalScrollBar().setValue(vertical)
            view.horizontalScrollBar().setValue(horizontal)
            view.viewport().update()

    def current_choice_key(self) -> str:
        """Return the selected opaque choice key (empty for none/header)."""
        index = self._selected_model_index()
        item = self._model.itemFromIndex(index) if index.isValid() else None
        data = None if item is None else item.data(QtCore.Qt.UserRole)
        return str(data).strip() if data else ""


class FluentUnitPicker(FluentTreeComboBox):
    """Which spelling of one unit a field is read in.

    A number's unit is its owner's -- what the device holds, what the range is
    in -- and it never moves.  Which spelling of it an operator reads is a
    different question, and theirs: the power a board holds in dBm is the same
    power in mW, and nobody should have to do that arithmetic by hand to work
    in the other one.  Since a prefix is not typed, this is the ONLY way to
    ask for one, which is why it belongs beside every field that has a ladder
    rather than inside the one form that first needed it.

    A TREE, because the choices are one thing: each base carries its own
    prefixes on its branch, and a unit that is not a prefixed base (dBm)
    stands beside them as its own.  A flat list of eighteen spellings is the
    same information with the structure thrown away.
    """

    #: emitted with the chosen spelling of the field's own unit
    unit_picked = QtCore.pyqtSignal(str)

    def __init__(self, unit: str, parent=None) -> None:
        super().__init__(parent)
        self._unit = ""
        #: Every spelling this picker can show collapsed.  The base combo
        #: measures its width over its TOP-LEVEL rows, which in a tree are
        #: the trunks -- "W", one letter -- so the box was sized for "W" and
        #: showed "dBm" as "dB...".  The leaves are what it shows.
        self._spellings: tuple[str, ...] = ()
        self.set_unit(unit)
        # As wide as its widest spelling and no wider; as tall as every other
        # control in the row.  Only the horizontal policy is this widget's to
        # say -- a Preferred height let the row's tallest neighbour stretch
        # it, so the picker stood taller than the box it belongs to.
        policy = self.sizePolicy()
        policy.setHorizontalPolicy(QtWidgets.QSizePolicy.Fixed)
        self.setSizePolicy(policy)
        self.setToolTip("Read this field in another unit")
        self.keyed_choice_picked.connect(self.unit_picked)

    def _collapsed_text_candidates(self) -> tuple[str, ...]:
        return self._spellings or super()._collapsed_text_candidates()

    def unit(self) -> str:
        """The unit this field's NUMBER is in, whatever is on screen."""

        return self._unit

    def set_unit(self, unit: str) -> None:
        """Offer the spellings of ``unit`` instead: the row's port changed.

        In place.  The tree is reconciled, not the picker rebuilt, so a row
        that keeps its picker keeps its place in the layout and in the
        focus chain -- and a unit that did not change is only selected
        again.  That is how a refused pick is put back: the tree names the
        leaf before it asks, and left there the next number typed would be
        read in a unit the picker does not say.
        """

        wanted = str(unit).strip() or "1"
        if wanted == self._unit:
            self.select_choice_key(wanted)
            return
        tree = unit_choice_tree(wanted)
        self._unit = wanted
        self._spellings = tuple(
            str(label) for _trunk, leaves in tree for label, _key, _full in leaves
        )
        self.set_choice_tree(tree, current=wanted)
        self._invalidate_content_width()


def unit_choice_tree(unit: str) -> tuple:
    """Every spelling of ``unit``, grouped under the FAMILY each belongs to.

    One tree per family, not per dimension: W with its prefixes, dBm on its
    own, Vpp with its prefixes -- three trees for one power, because they
    are three ways of writing it and not three sizes of one.  Grouping by
    the dimension's base put dBm under W, which is arithmetic's view and
    nobody's.  The field's own family comes first.

    Empty when there is nothing to choose -- an unknown unit, a dimensionless
    number, or a unit whose ladder is only itself -- which is how a caller
    knows not to mount a picker at all.
    """

    symbol = str(unit).strip()
    if not symbol or symbol == "1":
        return ()
    try:
        choices = DEFAULT_UNITS.display_choices(symbol)
    except UnitError:
        return ()
    if len(choices) < 2:
        return ()
    own = DEFAULT_UNITS.family_of(symbol).symbol
    order: list[str] = [own]
    branches: dict[str, list[tuple[str, str, str]]] = {own: []}
    for spelling in choices:
        trunk = DEFAULT_UNITS.family_of(spelling).symbol
        leaves = branches.get(trunk)
        if leaves is None:
            leaves = []
            branches[trunk] = leaves
            order.append(trunk)
        leaves.append((spelling, spelling, spelling))
    return tuple((trunk, tuple(branches[trunk])) for trunk in order if branches[trunk])


def fluent_unit_picker(unit: str, parent=None) -> "FluentUnitPicker | None":
    """The picker for this unit, or None when it has only one spelling."""

    return FluentUnitPicker(unit, parent) if unit_choice_tree(unit) else None


def _tab_padding_h() -> int:
    """A pivot tab's horizontal padding: from the underline's ends to the
    label."""

    return scaled_px(16, minimum=12)


def _tab_margin_h() -> int:
    """A pivot tab's horizontal margin: from its tab rect to its underline."""

    return scaled_px(3)


def _tab_font() -> QtGui.QFont:
    """The font the tab sheet names for a label."""

    font = QtGui.QFont(FONT)
    font.setPointSizeF(fluent_font_size())
    return font


class _FluentTabCloseButton(QtWidgets.QToolButton):
    """The close affordance for a closable tab: a small "x" glyph on a round
    hover wash -- ACCENT_TINT, SELECTED while pressed, as every hover and
    press is.  Quiet MUTED_INK at rest (a near-white "x" disappears on the
    tab bar), TEXT on the wash -- closing a view is not removing anything,
    so it does not turn red.

    The button paints itself: the disk on whole device pixels, an even
    number of them across, and the x centred on its INK on the disk's
    centre (:func:`paint_centred_ink`).  Drawn as a QToolButton's text, the
    x was centred on its line box and sat a pixel below the middle of its
    disk.  It stays a ``QToolButton``: a ``QAbstractButton`` subclass does not
    composite as a ``QTabBar`` tab button (it rendered blank).  The glyph's
    font is the button's own, never ``self.font()``: a window-level
    ``QWidget { font }`` sheet overrode setFont and drew a larger x in one
    window than another.  Frontend-owned, reused via
    :meth:`FluentTabWidget.add_closable_tab`.

    The x has to end as far before its underline's end as the label's ink
    starts after the underline's start.  The button carries the difference
    as a transparent ``trail`` on its right, which the BAR sets after it has
    laid the tab out (``_FluentTabBar.tabLayoutChange``), from where the
    style actually seated the button and where the glyph's ink actually
    ends -- not from a fitted constant.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        # Even, so the hover disk (radius size / 2) is round.
        self._size = scaled_even_px(18, minimum=14)
        self._glyph = QtGui.QFont(FONT)
        self._glyph.setPixelSize(max(9, int(self._size * 0.62)))
        self._glyph.setBold(True)
        self._trail = -1
        self.setText("✕")                       # heavy multiplication x
        self.setAttribute(QtCore.Qt.WA_Hover, True)
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self.setFocusPolicy(QtCore.Qt.NoFocus)
        self.setToolTip("Close this tab")
        self.set_trail(0)

    def _ink(self) -> QtCore.QRectF:
        path = QtGui.QPainterPath()
        path.addText(0.0, 0.0, self._glyph, self.text())
        return path.boundingRect()

    def glyph_ink_right(self) -> float:
        """Where the x's ink ends, from the left of its disk: the ink is
        centred on the disk."""

        return (self._size + self._ink().width()) / 2.0

    def trail(self) -> int:
        return self._trail

    def set_trail(self, trail: int) -> None:
        trail = max(0, int(trail))
        if trail == self._trail:
            return
        self._trail = trail
        self.setFixedSize(self._size + trail, self._size)
        self.update()

    def enterEvent(self, event) -> None:  # noqa: N802 - Qt naming
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802 - Qt naming
        self.update()
        super().leaveEvent(event)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        del event
        ratio = float(self.devicePixelRatioF())
        across = max(2, int(round(self._size * ratio)) & ~1) / ratio
        disk = QtCore.QRectF(device_snap(self, 0.0), device_snap(self, 0.0, axis="y"), across, across)
        washed = self.isEnabled() and (self.isDown() or self.underMouse())
        painter = QtGui.QPainter(self)
        if washed:
            painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(QtGui.QColor(SELECTED if self.isDown() else ACCENT_TINT))
            painter.drawEllipse(disk)
        paint_centred_ink(painter, self, disk, self._glyph, self.text(), TEXT if washed else MUTED_INK)
        painter.end()


class _FluentTabBar(QtWidgets.QTabBar):
    """Pivot tab bar that fits every tab WITHOUT scroll arrows and WITHOUT clipping a close
    ``x``, while keeping short tabs at full width.

    When the natural tab widths overflow the bar, the WIDEST tabs are capped to a shared
    width (water-filling) and ``ElideRight`` trims only their labels: ``Monitor`` / ``Logic``
    stay full, long Edit titles trim with a trailing ``...``, and every tab's close ``x``
    stays painted because the cap reserves the right-side button slot.  This is the
    squeeze-free replacement for the old ``width//count`` layout that crammed EVERY tab
    (short ones included) to an equal sliver.  ``sizeHint`` still reports the NATURAL total
    so the QTabWidget grants the bar its full window-capped width (``tabSizeHint`` caps to
    that ACTUAL width); reporting the capped sum instead would loop back and collapse the
    bar.  No tab scrolls off, so no scroll arrows appear; the corner ``...`` overflow menu
    lists every FULL title.  Reusable, frontend-owned."""

    def __init__(self, parent=None):
        super().__init__(parent)
        # Pivot tabs carry their own accent UNDERLINE; QTabBar's default "base" -- a line the
        # style strokes across the WHOLE bar width (and so grows with the tab count) -- would
        # show as a thin line on the bar's top edge, above the labels.  Turn it off.
        self.setDrawBase(False)

    def tabLayoutChange(self) -> None:  # noqa: N802 - Qt naming
        """Balance every close x against its label, now that the style has
        placed both (see ``_FluentTabCloseButton``)."""

        super().tabLayoutChange()
        label = QtGui.QFontMetricsF(_tab_font())
        for index in range(self.count()):
            button = self.tabButton(index, QtWidgets.QTabBar.RightSide)
            if not isinstance(button, _FluentTabCloseButton) or not self.tabText(index):
                continue
            rect = self.tabRect(index)
            underline_end = rect.x() + rect.width() - _tab_margin_h()
            # How far inside the underline the style ends the button: its
            # own seat, the same whatever the button's width.
            seat = underline_end - (button.x() + button.width())
            lead = _tab_padding_h() + label.leftBearing(self.tabText(index)[0])
            ink_end = underline_end - seat - button.width() + button.glyph_ink_right()
            # The x's ink should end ``lead`` before the underline does; a
            # wider trail moves the disk left by what it adds (the style
            # keeps the button's right end on its seat).
            gap = underline_end - ink_end
            button.set_trail(int(round(button.trail() + lead - gap)))

    def _natural_widths(self) -> list:
        # Explicit base call (NOT a no-arg super(), which fails inside a comprehension frame).
        base = QtWidgets.QTabBar.tabSizeHint
        return [base(self, i).width() for i in range(self.count())]

    def natural_width(self) -> int:
        """The width at which no tab has to be elided.

        The cap is applied inside ``tabSizeHint``, so asking a laid-out bar how
        wide its tabs are answers with the widths it already settled for.  A
        container that wants to give the bar enough room has to ask what it
        wanted, which is this.
        """

        return sum(self._natural_widths())

    def _shrink_cap(self):
        """Water-filling width cap: the largest ``cap`` such that capping every tab at it
        sums to <= the bar width.  Tabs already narrower than ``cap`` (Monitor / Logic) keep
        their natural width; only the wide Edit tabs are capped + elided.  None when all fit."""
        avail = self.width()
        widths = sorted(self._natural_widths())
        n = len(widths)
        if n == 0 or avail <= 0 or sum(widths) <= avail:
            return None
        # The cap floor is JUST the close-x slot + a sliver of label, NOT a readable-label
        # width: a larger floor would make n*floor exceed a narrow bar, so the rightmost tab
        # (and its close x) would spill past the edge -- the "x cut off" bug.  Floored only
        # this low, the tabs always sum to <= the bar (the cap falls to avail/n first), so a
        # close x is never clipped; the heavily-elided labels stay reachable via the ... menu.
        floor = scaled_px(34, minimum=28)
        prefix = 0
        for k in range(n):
            cap = (avail - prefix) / (n - k)         # share the remainder among the wide tabs
            if cap <= widths[k]:
                return max(int(cap), floor)
            prefix += widths[k]                      # this tab fits under the cap -> keeps natural
        return floor

    def tabSizeHint(self, index):  # noqa: N802 - Qt naming
        base = QtWidgets.QTabBar.tabSizeHint
        hint = base(self, index)
        cap = self._shrink_cap()
        if cap is None or hint.width() <= cap:
            return hint                              # fits (or is a short tab) -> natural width
        return QtCore.QSize(cap, hint.height())

    def sizeHint(self):  # noqa: N802 - Qt naming
        hint = QtWidgets.QTabBar.sizeHint(self)
        widths = self._natural_widths()
        return QtCore.QSize(sum(widths), hint.height()) if widths else hint

    def is_overflowing(self) -> bool:
        """True when the natural tab widths exceed the bar (some labels are being elided) --
        the cue to offer the ``...`` overflow menu of full titles."""
        widths = self._natural_widths()
        avail = self.width()
        return bool(widths) and avail > 0 and sum(widths) > avail


class FluentTabWidget(QtWidgets.QTabWidget):
    """Pivot-underline tab widget that also supports CLOSABLE tabs.

    Permanent tabs are added with :meth:`add_permanent_tab` (no close button)
    and are what every page in this project is: the editor's Edit / Preview /
    Scan / Target and the console's Monitor / Logic all must always exist.  A
    host that does want a page an operator can dismiss uses
    :meth:`add_closable_tab`, which shows the close affordance and emits
    :attr:`tab_close_requested` with the page widget so the host can tear it
    down -- the tab widget never removes a page behind its owner's back."""

    tab_close_requested = QtCore.pyqtSignal(QtWidgets.QWidget)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTabPosition(QtWidgets.QTabWidget.North)
        self.setTabBar(_FluentTabBar(self))     # content-width pivot tabs; elides to fit, no arrows
        self.tabBar().setExpanding(False)
        self.tabBar().setVisible(False)         # until a second page gives it a choice
        # No left/right scroll ARROWS (the native chevrons crowd the last tab's close "x"
        # and read as browser chrome, not Fluent).  With scroll buttons OFF and ElideRight,
        # Qt shrinks the OVERFLOWING tabs by eliding their labels so ALL tabs fit side by
        # side -- short tabs (Monitor / Logic) stay full, long Edit titles trim with "..."
        # and every tab's close "x" stays fully painted (the reported "x cut off" is gone).
        # The corner ``...`` overflow button lists every FULL title for navigation.
        self.setUsesScrollButtons(False)
        self.tabBar().setElideMode(QtCore.Qt.ElideRight)
        # No NATIVE close buttons.  A plain ``addTab()`` tab is PERMANENT (no X),
        # so GUIs that use ``addTab`` directly (e.g. a multi-page editor's
        # Edit / Preview / Scan, which must always exist) never get a close
        # affordance.  ``add_closable_tab`` opts in a CUSTOM, subtle close button
        # (a soft grey "x" that highlights on hover -- not the harsh native X).
        self.setStyleSheet(
            f"""
            QTabWidget {{
                background: transparent;
                margin: 0px;
                padding: 0px;
            }}
            QTabWidget::pane {{
                background: white;
                margin: 0px;
                padding: 0px;
                border: none;
                border-radius: {_radius()}px;
            }}
            QTabWidget QStackedWidget {{
                background: white;
                border: none;
                margin: 0px;
                padding: 0px;
                border-radius: {_radius()}px;
            }}
            QTabWidget::tab-bar {{
                margin: 0px;
                padding: 0px;
            }}
            QTabBar {{
                background: transparent;
            }}
            QTabBar::tab {{
                background: transparent;
                color: {MUTED_INK};
                border: none;
                border-bottom: {scaled_px(2, minimum=2)}px solid transparent;
                height: {scaled_px(32, minimum=26)}px;
                padding: 0px {_tab_padding_h()}px;
                margin: 0px {_tab_margin_h()}px;
                font: {fluent_font_size()}pt "{FONT}";
            }}
            QTabBar::tab:first {{ margin-left: {scaled_px(CARD_PAD)}px; }}
            QTabBar::tab:selected {{
                color: {TEXT};
                border-bottom: {scaled_px(2, minimum=2)}px solid {ACCENT};
            }}
            QTabBar::tab:!selected:hover {{
                color: {TEXT};
            }}
            """
        )
        self._overflow_btn = self._build_overflow_button()
        # The corner widget sits FLUSH at the widget's right edge, so the button's rounded hover
        # background used to be CLIPPED on the right.  Wrap it in a transparent holder whose right
        # margin reserves the clearance, so the full hover pill is visible.
        corner = QtWidgets.QWidget(self)
        corner.setStyleSheet("background: transparent;")
        corner_lay = QtWidgets.QHBoxLayout(corner)
        corner_lay.setContentsMargins(0, 0, scaled_px(6, minimum=4), 0)
        corner_lay.setSpacing(0)
        corner_lay.addWidget(self._overflow_btn)
        self._overflow_corner = corner
        self.setCornerWidget(corner, QtCore.Qt.TopRightCorner)
        corner.hide()
        self.currentChanged.connect(lambda _i: self._update_overflow())

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        # The tab widget is a plain WHITE rounded card with NO border line ANYWHERE -- the pivot tabs sit
        # on the white top, the content flows straight below, and the card is delineated only by its white
        # body against the grey window.  (Any border here read as boxing the tabs at the top, or as a line
        # between the tabs and the content below -- neither is wanted; the tabs + content are ONE surface.)
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QColor("white"))
        painter.drawRoundedRect(QtCore.QRectF(self.rect()), float(_radius()), float(_radius()))
        painter.end()
        super().paintEvent(event)

    def _build_overflow_button(self) -> QtWidgets.QToolButton:
        """The Fluent overflow affordance: a subtle ``...`` button (top-right corner) that
        replaces the native scroll arrows.  Shown only when the tabs overflow the bar;
        clicking it lists EVERY tab and jumps to the picked one (scrolling it into view)."""
        btn = QtWidgets.QToolButton(self)
        btn.setText("⋯")                          # MIDLINE HORIZONTAL ELLIPSIS
        btn.setCursor(QtCore.Qt.PointingHandCursor)
        btn.setFocusPolicy(QtCore.Qt.NoFocus)
        btn.setToolTip("All tabs")
        # The menu is popped from the `clicked` handler (the button owns no QMenu), so the
        # default (DelayedPopup) mode is correct -- InstantPopup would suppress `clicked`.
        size = scaled_px(26, minimum=20)
        btn.setFixedSize(size, size)
        font = btn.font()
        font.setPixelSize(max(11, int(size * 0.72)))
        font.setBold(True)
        btn.setFont(font)
        radius = _radius()
        btn.setStyleSheet(
            f"""
            QToolButton {{ border: none; background: transparent; color: {MUTED_INK};
                           border-radius: {radius}px; padding: 0px; }}
            QToolButton:hover {{ background: {ACCENT_TINT}; color: {TEXT}; }}
            QToolButton:pressed {{ background: {SELECTED}; color: {TEXT}; }}
            QToolButton::menu-indicator {{ image: none; width: 0px; }}
            """
        )
        btn.clicked.connect(self._show_overflow_menu)
        return btn

    def _tabs_overflow(self) -> bool:
        bar = self.tabBar()
        # _FluentTabBar keeps natural tab widths and scrolls on overflow; ask it whether the
        # natural widths exceed the bar (some tabs scrolled off) -- the cue to offer the
        # ``...`` menu of every tab (Qt scrolls the picked one into view).
        if hasattr(bar, "is_overflowing"):
            return bar.is_overflowing()
        return bar.count() > 0 and bar.sizeHint().width() > bar.width() + 1

    def _update_overflow(self) -> None:
        """Show the overflow ``...`` button only while the tabs do not all fit.  Run on a
        0-ms timer after a resize / tab change so the tab bar has been re-laid-out first
        (its width is stale inside the resize/insert event itself).  Visibility toggles the
        HOLDER (the margin-bearing corner widget), so a hidden overflow reserves no corner space."""
        corner = getattr(self, "_overflow_corner", None)
        if corner is not None:
            corner.setVisible(self._tabs_overflow())

    def _schedule_overflow_update(self) -> None:
        QtCore.QTimer.singleShot(0, self._update_overflow)

    def _overflow_menu(self) -> QtWidgets.QMenu:
        """Build (but do not show) the Fluent overflow menu: one action per tab that selects it on
        trigger (Qt scrolls it into view).  Split out from the popup so the click->select wiring is
        testable without a blocking ``exec_``.  The actions are NOT checkable -- a checkable action
        makes Qt reserve an indicator gutter and draw its own (un-Fluent) tick/check box; the current
        tab is already the visibly-active pivot, so no tick is needed on the list.

        The card is rounded the SAME way every other Fluent popup is (the FluentPopup Setting card,
        the FluentComboBox drop-down): a stylesheet ``border-radius`` alone leaves the menu's opaque
        SQUARE window behind the arc, so the corners outside the radius show a square white nub (and
        Windows adds a native shadow) -- reading as a hard-cornered popup next to the antialiased
        round ones.  So this uses :class:`_FluentRoundedMenu`, which is TRANSLUCENT + frameless and
        PAINTS one antialiased rounded rect (white fill + 1 px border) behind its items -- the shared
        rounded-card single source, composited so the labels still draw on top."""
        menu = _FluentRoundedMenu(self)
        for i in range(self.count()):
            act = menu.addAction(self.tabText(i))
            act.triggered.connect(lambda _c, idx=i: self.setCurrentIndex(idx))
        return menu

    def _show_overflow_menu(self) -> None:
        """Pop the overflow list below the ``...`` button -- the single overflow navigation,
        in place of left/right scroll arrows.  It sits the SAME few-px ``popup_gap`` off the button
        that the combo drop-down / Setting popup use below THEIR anchor, so the overflow list reads as
        one of the family instead of flush against the tab strip.  (A ``QMenu.exec_`` position is the
        top-left Qt honours as-is -- unlike a combo popup, it is NOT re-flushed -- so the gap goes
        straight into the anchor point; no Move-event filter is needed here.)

        The menu is built for this click and retired after it: it lists the
        tabs as they are NOW, and one that stayed parented here would be one
        more stale list per click for the life of the tab widget."""
        menu = self._overflow_menu()
        btn = self._overflow_btn
        try:
            menu.exec_(btn.mapToGlobal(QtCore.QPoint(0, btn.height() + popup_gap())))
        finally:
            retire_widget(menu)

    def resizeEvent(self, event):  # noqa: N802 - Qt naming
        super().resizeEvent(event)
        self._schedule_overflow_update()

    def showEvent(self, event):  # noqa: N802 - Qt naming
        super().showEvent(event)
        self._schedule_overflow_update()

    def tabInserted(self, index):  # noqa: N802 - Qt naming
        super().tabInserted(index)
        self._show_bar_when_it_chooses()
        self._schedule_overflow_update()

    def tabRemoved(self, index):  # noqa: N802 - Qt naming
        super().tabRemoved(index)
        self._show_bar_when_it_chooses()
        self._schedule_overflow_update()

    def _show_bar_when_it_chooses(self) -> None:
        """No bar over one permanent page: a single pivot chooses nothing.

        A lone CLOSABLE page keeps its bar, because the bar carries its
        close button.  Asked at insertion, before a closable tab has its
        button, the count already says two whenever a second page arrives.
        """

        bar = self.tabBar()
        lone = self.count() == 1 and bar.tabButton(0, QtWidgets.QTabBar.RightSide) is None
        bar.setVisible(not lone)

    def add_permanent_tab(self, widget: QtWidgets.QWidget, title: str) -> int:
        """Add a fixture tab with no close button (for example, Monitor)."""
        return self.addTab(widget, title)

    def add_closable_tab(self, widget: QtWidgets.QWidget, title: str, *,
                         focus: bool = True) -> int:
        """Add a tab WITH a subtle custom close button.  Clicking it emits
        :attr:`tab_close_requested` with ``widget`` so the host can tear it down."""
        index = self.addTab(widget, title)
        self.tabBar().setTabButton(index, QtWidgets.QTabBar.RightSide,
                                   self._make_close_button(widget))
        if focus:
            self.setCurrentIndex(index)
        return index

    def _make_close_button(self, widget: QtWidgets.QWidget) -> "_FluentTabCloseButton":
        """A custom-PAINTED close affordance whose "x" is guaranteed centred in
        its hover circle (a text "x" in a QToolButton drifts off-centre)."""
        btn = _FluentTabCloseButton(self)
        btn.clicked.connect(lambda: self.tab_close_requested.emit(widget))
        return btn


def _switch_track_size() -> tuple[int, int]:
    """The track every switch draws: one size, whatever its label says.

    Its height is even, and so is the thumb inside it (the height less two
    insets): a round thing on a whole-pixel centre is round."""

    return scaled_px(40, minimum=32), scaled_even_px(20, minimum=16)


def fluent_switch_width(text: str) -> int:
    """The exact width a :class:`FluentSwitch` takes for ``text``.

    The ONE switch-width authority.  A layout that must reserve a column for a
    switch (a form's Auto label column) asks THIS instead of predicting the
    geometry with its own padding constant -- two independent guesses is how a
    switch's painted track ended up wider than the cell it was given and drew
    underneath its neighbour.
    """

    track_w, _track_h = _switch_track_size()
    if not text:
        return track_w + 2
    metrics = QtGui.QFontMetrics(QtGui.QFont(FONT, fluent_font_size()))
    return track_w + scaled_px(8) + fluent_text_width(metrics, str(text)) + 2


class FluentSwitch(QtWidgets.QAbstractButton):
    """An on/off setting: a small track with its label BESIDE it, in TEXT ink.

    Every boolean SETTING is one of these -- Selectors, Plot, Show off rows,
    Remote -- including one that sits in a row of commands: a checkable
    button there read as one more command, and on was told from off by a
    shade.  The label used to be painted across the track, under the thumb,
    which covered a letter of every word it carried.  The track is a solid
    fill with a white thumb and no outline anywhere: off is the neutral
    PLACEHOLDER grey, on the ACCENT; under the pointer the track steps one
    shade deeper; disabled it is the DIVIDER grey with a BG thumb.  Rings
    round the track and rims round the thumb were lines laid round a shape
    its fill already drew.  The widget is a control's height (30 design px)
    with the track centred in it, so a switch lines up with the buttons of
    its row.  The whole width, label included, takes the click.  In a row
    of commands a switch and its label are their own group:
    :func:`add_row_switch` puts the air of two button gaps either side.
    """

    def __init__(self, text: str = "", parent=None):
        super().__init__(parent)
        self.setText(text)
        self.setCheckable(True)
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self.setFont(QtGui.QFont(FONT, fluent_font_size()))
        self.setAttribute(QtCore.Qt.WA_Hover, True)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        # A control's height, never stretched: a row centres it, so its
        # track sits on the same line as the buttons beside it.
        self.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self._offset = float(self._thumb_inset())
        self._animating = False
        self._anim = QtCore.QPropertyAnimation(self, b"offset", self)
        self._anim.setDuration(150)
        self._anim.stateChanged.connect(self._on_anim_state_changed)
        self.toggled.connect(self._start_animation)
        self._apply_metrics()

    @staticmethod
    def _thumb_inset() -> int:
        return scaled_px(4, minimum=3)

    def _apply_metrics(self) -> None:
        # A control's height, the same number a FluentButton takes.
        self.setMinimumSize(fluent_switch_width(self.text()), control_height())

    def setText(self, text: str) -> None:  # noqa: N802 - Qt API
        value = str(text)
        if value == self.text():
            return
        super().setText(value)
        self._apply_metrics()
        self.update()

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(self.minimumWidth(), self.minimumHeight())

    def minimumSizeHint(self) -> QtCore.QSize:
        return self.sizeHint()

    def hitButton(self, pos) -> bool:
        return (
            0 <= int(pos.x()) <= min(self.width(), fluent_switch_width(self.text()))
            and 0 <= int(pos.y()) <= self.height()
        )

    def _track_box(self) -> QtCore.QRect:
        """The track's whole-pixel box, centred in the widget's height."""

        track_w, track_h = _switch_track_size()
        return QtCore.QRect(0, (self.height() - track_h) // 2, track_w, track_h)

    def _track_outer(self) -> QtCore.QRectF:
        """The track on device pixels, an even number of device pixels tall,
        so its thumb is centred."""

        box = self._track_box()
        ratio = float(self.devicePixelRatioF())
        tall = max(2, int(round(box.height() * ratio)))
        tall -= tall & 1
        left = device_snap(self, box.left())
        top = device_snap(self, box.top(), axis="y")
        return QtCore.QRectF(left, top, round(box.width() * ratio) / ratio, tall / ratio)

    def paintEvent(self, event) -> None:
        del event
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        outer = self._track_outer()
        radius = outer.height() / 2.0
        on, enabled = self.isChecked(), self.isEnabled()
        hovered = enabled and (self.underMouse() or self.isDown())
        if not enabled:
            fill, thumb = QtGui.QColor(DIVIDER), QtGui.QColor(BG)
        elif on:
            fill, thumb = QtGui.QColor(ACCENT_HOVER if hovered else ACCENT), QtGui.QColor(SURFACE)
        else:
            fill = _mixed(PLACEHOLDER, TEXT, 0.2) if hovered else QtGui.QColor(PLACEHOLDER)
            thumb = QtGui.QColor(SURFACE)
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(fill)
        painter.drawRoundedRect(outer, radius, radius)

        # The thumb sits the same inset from the track's end as from its top
        # and bottom, an even number of device pixels across, on device
        # pixels -- round, and one gap all round, at any ratio.
        ratio = float(self.devicePixelRatioF())
        gap = round(self._thumb_inset() * ratio) / ratio
        diameter = outer.height() - 2 * gap
        top = outer.top() + gap
        if self._animating:
            left = device_snap(self, self._offset)
        elif self.isChecked():
            left = outer.right() - gap - diameter
        else:
            left = outer.left() + gap
        painter.setBrush(thumb)
        painter.drawEllipse(QtCore.QRectF(left, top, diameter, diameter))

        if self.text():
            track_w, _track_h = _switch_track_size()
            painter.setPen(QtGui.QColor(TEXT if enabled else DISABLED_INK))
            painter.setFont(self.font())
            left = track_w + scaled_px(8)
            # On the row's one line (centred_line_top), with the buttons
            # and fields beside it.
            top = centred_line_top(0, self.height(), self.font())
            painter.drawText(
                QtCore.QRect(left, top, max(0, self.width() - left), self.height() - top),
                QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop,
                self.text(),
            )
        painter.end()

    def mouseReleaseEvent(self, event) -> None:
        # QAbstractButton already gates toggling on the left button + enabled state,
        # so the release is forwarded unconditionally.
        super().mouseReleaseEvent(event)

    def _checked_offset(self) -> float:
        track_w, track_h = _switch_track_size()
        inset = self._thumb_inset()
        diameter = max(2, track_h - 2 * inset)
        return float(track_w - diameter - inset if self.isChecked() else inset)

    def _start_animation(self, checked: bool) -> None:
        del checked
        end_pos = self._checked_offset()
        if not self.isEnabled() or self.width() <= 0 or self.height() <= 0:
            self._anim.stop()
            self._offset = end_pos
            self.update()
            return
        self._anim.stop()
        self._anim.setStartValue(self._offset)
        self._anim.setEndValue(end_pos)
        self._anim.start()

    def _on_anim_state_changed(self, new_state, _old_state) -> None:
        self._animating = new_state == QtCore.QAbstractAnimation.Running
        if not self._animating:
            self._offset = self._checked_offset()
            self.update()

    def getOffset(self) -> float:
        return float(self._offset)

    def setOffset(self, value: float) -> None:
        self._offset = float(value)
        self.update()

    offset = QtCore.pyqtProperty(float, fget=getOffset, fset=setOffset)


def command_gap() -> int:
    """The space between two commands in a row (Control, Log, Close)."""

    return scaled_px(8, minimum=5)


def add_row_switch(
    layout: QtWidgets.QBoxLayout,
    switch: FluentSwitch,
    *,
    lead: bool = True,
    trail: bool = True,
) -> None:
    """Add ``switch`` to a row of commands as a group of its own.

    At the row's own spacing a switch's label ran into the next command --
    "Remote Log", "Plot Start" -- and read as one more word on a button.  A
    second gap on each side that has a neighbour (``lead`` before the
    track, ``trail`` after the label) keeps the setting and its label apart
    from the commands, whatever the row's spacing is.
    """

    gap = layout.spacing()
    if gap < 0:
        gap = command_gap()
    if lead:
        layout.addSpacing(gap)
    layout.addWidget(switch, 0, QtCore.Qt.AlignVCenter)
    if trail:
        layout.addSpacing(gap)


def fluent_spinbox_stylesheet(
    selector: str,
    *,
    edge: str = _FIELD_EDGE_REST,
    focus_edge: str = _FIELD_EDGE_FOCUS,
    fill: str = "white",
    ink: str = TEXT,
) -> str:
    # ``edge``/``fill``/``ink`` are the field's state colours: a bound field
    # (Config, Scan, API) wears its binding's colours on the frame.
    #
    # The steppers are BORDER-origin boxes whose ``margin`` is the edge
    # width: the margin box starts one edge-width left of the cell, a quiet
    # column between the digits and the stepper.  They are transparent and
    # serve geometry and hit testing only -- the tint and the chevrons are
    # painted from the frame by _paint_fluent_spin_buttons.
    return f"""
    {selector} {{
        background: {fill};
        border: {FIELD_BORDER_PX}px solid {edge};
        border-radius: {_radius()}px;
        padding: {scaled_px(PADDING_V)}px {scaled_px(EDIT_PADDING_H)}px;
        font: {fluent_font_size()}pt "{FONT}";
        color: {ink};
    }}
    {selector}:focus {{ border: {FIELD_BORDER_PX}px solid {focus_edge}; }}
    {selector}:disabled {{ background: {BG}; color: {DISABLED_INK}; }}
    {selector}::up-button, {selector}::down-button {{
        subcontrol-origin: border;
        width: {scaled_px(COMBO_WIDTH)}px;
        margin: {FIELD_BORDER_PX}px;
        border: none;
        background: transparent;
    }}
    {selector}::up-button {{ subcontrol-position: top right; }}
    {selector}::down-button {{ subcontrol-position: bottom right; }}
    {selector}::up-arrow, {selector}::down-arrow {{ image: none; }}
    """


def fluent_inner_edit_stylesheet(ink: str = TEXT) -> str:
    """The sheet of a line edit a spin box has adopted: no frame, fill or
    padding of its own -- the box draws those -- only the digits' ink."""

    return (
        f'QLineEdit {{ background: transparent; border: none; padding: 0px; color: {ink}; '
        f'font: {fluent_font_size()}pt "{FONT}"; }}'
        f"QLineEdit:disabled {{ color: {DISABLED_INK if ink == TEXT else ink}; }}"
    )


def _spin_field_chrome() -> int:
    """What a spin box's frame takes beside its digits: both edges, the
    padding, the line edit's own text inset and the stepper."""

    return (
        fluent_stepper_extent() + 2 * FIELD_BORDER_PX
        + scaled_px(2 * EDIT_PADDING_H) + 2 * _QLINEEDIT_HMARGIN
    )


def _spin_stepper_cells(widget: QtWidgets.QAbstractSpinBox) -> tuple[float, QtCore.QRectF, QtCore.QRectF]:
    """``(stepper left, up cell, down cell)`` in logical px, from the frame.

    The stepper begins at the margin column one edge-width left of the
    stepper column (what a part placed beside it, the step chip, keeps
    clear of); the cells tile the room from that column's far side to the
    right edge, split at a device pixel."""

    cell = stepper_column(widget)
    left = device_snap(widget, cell.left() - FIELD_BORDER_PX)
    start = left + frame_border_px(widget)
    right = frame_inner_rect(widget).right()
    mid = device_snap(widget, cell.center().y(), axis="y")
    up = QtCore.QRectF(start, cell.top(), right - start, mid - cell.top())
    down = QtCore.QRectF(start, mid, right - start, cell.bottom() - mid)
    return left, up, down


def field_room_end(widget: QtWidgets.QWidget) -> float:
    """Where the room inside a field's frame ends on the right, in logical
    px: the frame's inner edge, or the left of a spin box's stepper cells.

    A part seated at a field's right end (a binding badge) keeps ONE gap
    from this, so it stands as far from the stepper in a spin box as from
    the edge in a plain field."""

    if (isinstance(widget, QtWidgets.QAbstractSpinBox)
            and widget.buttonSymbols() != QtWidgets.QAbstractSpinBox.NoButtons):
        return _spin_stepper_cells(widget)[1].left()
    return frame_inner_rect(widget).right()


def _paint_fluent_spin_buttons(widget: QtWidgets.QAbstractSpinBox) -> None:
    """Paint the stepper: the hovered cell's tint and two chevrons.

    The field is a two-cell frame, the classic numeric up-down, shown by its
    chevrons and the tint under the pointer -- no line divides it from the
    digits: a divider was one more line in a field already edged.  The tint
    fills exactly the hovered cell to the frame's inner edge, concentric at
    the corner.  The two chevrons are mirror images about the centre of
    the frame's inner rect, each in the middle of its half: placed on the
    cells' own centres, an odd number of rows inside the frame gave one
    cell a row more and put one chevron a pixel further from the middle
    than the other.  A direction the box cannot move in (``stepEnabled``: a
    bound, or no grid point left inside it) has its chevron dimmed and
    takes no tint.  Nothing is offset from a sub-control rectangle.
    """

    if widget.buttonSymbols() == QtWidgets.QAbstractSpinBox.NoButtons:
        return
    option = QtWidgets.QStyleOptionSpinBox()
    widget.initStyleOption(option)
    _left, up_cell, down_cell = _spin_stepper_cells(widget)
    live = widget.isEnabled() and not widget.isReadOnly()
    # Asked of the box itself: the style copies it into the option only
    # where the platform style says spin controls dim at a bound.
    directions = widget.stepEnabled() if live else QtWidgets.QAbstractSpinBox.StepNone
    movable = {
        True: bool(directions & QtWidgets.QAbstractSpinBox.StepUpEnabled),
        False: bool(directions & QtWidgets.QAbstractSpinBox.StepDownEnabled),
    }
    painter = QtGui.QPainter(widget)
    hovered = (
        live and bool(option.state & QtWidgets.QStyle.State_MouseOver)
        and int(option.activeSubControls)
        & int(QtWidgets.QStyle.SC_SpinBoxUp | QtWidgets.QStyle.SC_SpinBoxDown)
    )
    up_hovered = bool(option.activeSubControls & QtWidgets.QStyle.SC_SpinBoxUp)
    if hovered and movable[up_hovered]:
        cell = up_cell if up_hovered else down_cell
        corner = max(0.0, _radius() - FIELD_BORDER_PX)
        inner = QtGui.QPainterPath()
        inner.addRoundedRect(frame_inner_rect(widget), corner, corner)
        tile = QtGui.QPainterPath()
        tile.addRect(cell)
        painter.save()
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        painter.fillPath(inner.intersected(tile), QtGui.QColor(ACCENT_TINT))
        painter.restore()
    inner = frame_inner_rect(widget)
    middle = inner.center().y()
    rise = device_snap(widget, middle - inner.height() / 4.0, axis="y", grid=0.5)
    for up, cell in ((True, up_cell), (False, down_cell)):
        ink = QtGui.QColor(GRAPHITE if movable[up] else PLACEHOLDER)
        centre = QtCore.QPointF(cell.center().x(), rise if up else 2.0 * middle - rise)
        paint_chevron(painter, widget, centre, up=up, colour=ink)
    painter.end()


class _WheelFocusGuardMixin:
    """Scroll-over protection for numeric spin controls: the wheel only changes the value
    AFTER the widget has been clicked into (has keyboard focus).  Without focus the wheel
    event is ignored and bubbles up to the parent scroll area, so scrolling a Setting / Edit
    page that happens to pass over a spinbox scrolls the PAGE instead of silently nudging the
    number -- the误触 the user hit.  Mirrors ``FluentComboBox.wheelEvent`` (ignore -> bubble)
    so every numeric control behaves the same way.  ``StrongFocus`` is what makes a left click
    grab focus (Tab too); the real gate is the ``hasFocus`` check below, not the policy."""

    def _install_wheel_focus_guard(self) -> None:
        self.setFocusPolicy(QtCore.Qt.StrongFocus)

    def wheelEvent(self, event):
        if not self.hasFocus():
            event.ignore()         # not focused -> let the parent scroll area scroll the page
            return
        super().wheelEvent(event)  # focused (clicked into) -> normal value step


class FluentSpinBox(_WheelFocusGuardMixin, QtWidgets.QSpinBox):
    valueNormalized = QtCore.pyqtSignal()
    _normalization_pending = False
    _number_geometry = None

    def sizeHint(self) -> QtCore.QSize:  # noqa: N802
        size = super().sizeHint()
        size.setWidth(self.fontMetrics().horizontalAdvance("100000") + _spin_field_chrome())
        return size

    def minimumSizeHint(self) -> QtCore.QSize:  # noqa: N802
        return QtCore.QSize(scaled_px(48), self.minimumHeight())
    def __init__(self, parent=None):
        super().__init__(parent)
        self._install_wheel_focus_guard()
        # A HALF-TYPED NUMBER IS NOT A VALUE.  With keyboard tracking on,
        # Qt interprets the text on every keystroke: deleting the 1 from
        # "0.1" leaves "0.", which is read as zero and snapped to this
        # box's own minimum before the next character can be typed.  The
        # number is judged when the operator is done with it -- Enter, or
        # the focus leaving -- which is also when the range applies.
        self.setKeyboardTracking(False)
        self.setButtonSymbols(QtWidgets.QAbstractSpinBox.PlusMinus)
        self.setMinimumHeight(control_height())
        # Left-align with zero internal text margins so the text's left inset is
        # exactly the stylesheet ``padding`` (EDIT_PADDING_H) -- identical to a
        # plain FluentLineEdit.  Center alignment made the
        # left gap content-dependent and looked narrower than the line edits.
        self.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.lineEdit().setTextMargins(0, 0, 0, 0)
        self.setStyleSheet(fluent_spinbox_stylesheet("QSpinBox"))
        # The stepper tint is painted from the hovered sub-control, which Qt
        # tracks only for a widget that asks for hover events.
        self.setAttribute(QtCore.Qt.WA_Hover, True)

    def _visible_integer(self, value: int, low: int, high: int) -> int:
        bounded = min(high, max(low, value))
        if self._number_geometry is None or self.isReadOnly() or not self.isEnabled():
            return bounded
        number, _text, _width = _visible_decimal(
            Decimal(bounded), Decimal(low), Decimal(high), True,
            _numeric_text_width(self.lineEdit()), self.lineEdit().fontMetrics(),
        )
        return int(number)

    def _numeric_error(self, error: ValueError | None) -> None:
        message = "" if error is None else str(error)
        changed = str(self.property("numericError") or "") != message
        self.setProperty("numericError", message)
        if changed:
            self._queue_normalization()

    def _queue_normalization(self) -> None:
        if not self._normalization_pending:
            self._normalization_pending = True
            QtCore.QMetaObject.invokeMethod(self, "_deliver_normalization", QtCore.Qt.QueuedConnection)

    @QtCore.pyqtSlot()
    def _deliver_normalization(self) -> None:
        if self._normalization_pending:
            self._normalization_pending = False
            if self.isEnabled() and not self.isReadOnly():
                self.valueNormalized.emit()

    def setValue(self, value: int) -> None:  # noqa: N802
        try:
            visible = self._visible_integer(int(value), self.minimum(), self.maximum())
        except ValueError as error:
            self._numeric_error(error)
            return
        self._numeric_error(None)
        super().setValue(visible)
        self.lineEdit().setText(super().textFromValue(visible))
        if visible != value:
            self._queue_normalization()

    def setRange(self, minimum: int, maximum: int) -> None:  # noqa: N802
        """The owner's range -- applied when it changes.

        Qt's own setRange invalidates the box's geometry whether or not the
        numbers moved, and every reconcile re-declares them: a control
        re-projected every beat relaid out and repainted every beat.
        """

        if (int(minimum), int(maximum)) == (self.minimum(), self.maximum()):
            return
        low, high = int(minimum), max(int(minimum), int(maximum))
        try:
            visible = self._visible_integer(self.value(), low, high)
        except ValueError as error:
            with signals_blocked(self):
                super().setRange(low, high)
            self._numeric_error(error)
            self.lineEdit().clear()
            return
        self._numeric_error(None)
        super().setRange(low, high)
        self.setValue(visible)

    def textFromValue(self, value: int) -> str:  # noqa: N802
        if self.property("numericError") and not self.lineEdit().text():
            return ""
        return super().textFromValue(value)

    def validate(self, text: str, position: int):
        state, text, position = super().validate(text, position)
        if state != QtGui.QValidator.Acceptable or self._number_geometry is None:
            return state, text, position
        try:
            value = self._visible_integer(
                super().valueFromText(text), self.minimum(), self.maximum()
            )
        except ValueError as error:
            self._numeric_error(error)
            return QtGui.QValidator.Invalid, text, position
        self._numeric_error(None)
        fitted = super().textFromValue(value)
        return state, fitted, min(position, len(fitted))

    def valueFromText(self, text: str) -> int:  # noqa: N802
        try:
            value = self._visible_integer(
                super().valueFromText(text), self.minimum(), self.maximum()
            )
        except ValueError as error:
            self._numeric_error(error)
            return self.value()
        self._numeric_error(None)
        return value

    def stepBy(self, steps: int) -> None:  # noqa: N802
        # Let Qt own integer/wrapping/step semantics, but expose only the
        # final representable value, never an intermediate refused step.
        previous = self.value()
        with signals_blocked(self):
            super().stepBy(steps)
            try:
                visible = self._visible_integer(self.value(), self.minimum(), self.maximum())
            except ValueError as error:
                super().setValue(previous)
                self._numeric_error(error)
                return
            super().setValue(visible)
        self._numeric_error(None)
        if self.value() != previous:
            self.valueChanged[int].emit(self.value())
            self.valueChanged[str].emit(self.text())

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        # The box has just laid its edit out; the edit Qt made is a plain
        # QLineEdit, so the box seats its text (see _seat_line_edit_text).
        _seat_line_edit_text(self.lineEdit())
        self._update_number_geometry()

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        _seat_line_edit_text(self.lineEdit())
        self._update_number_geometry()

    def changeEvent(self, event) -> None:  # noqa: N802
        super().changeEvent(event)
        if event.type() in (QtCore.QEvent.FontChange, QtCore.QEvent.StyleChange):
            _seat_line_edit_text(self.lineEdit())
        if event.type() in (
            QtCore.QEvent.FontChange,
            QtCore.QEvent.StyleChange,
            QtCore.QEvent.EnabledChange,
            QtCore.QEvent.ReadOnlyChange,
        ):
            self._number_geometry = None
            if self.isReadOnly() or not self.isEnabled():
                self.setProperty("numericError", "")
                self._normalization_pending = False
                self.lineEdit().setText(super().textFromValue(self.value()))
                self.lineEdit().setCursorPosition(0)
            self._update_number_geometry()

    def _update_number_geometry(self) -> None:
        edit = self.lineEdit()
        if edit is None or not self.isVisible():
            return
        geometry = (_numeric_text_width(edit), edit.font().key())
        if geometry == self._number_geometry:
            return
        self._number_geometry = geometry
        previous = self.value()
        with signals_blocked(self):
            self.setValue(previous)
        if self.value() != previous:
            self._queue_normalization()

    def setMinimum(self, minimum: int) -> None:  # noqa: N802
        self.setRange(int(minimum), self.maximum())

    def setMaximum(self, maximum: int) -> None:  # noqa: N802
        self.setRange(self.minimum(), int(maximum))

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        _paint_fluent_spin_buttons(self)

    def contextMenuEvent(self, event) -> None:  # noqa: N802 - Qt naming
        _apply_fluent_context_menu(self.lineEdit(), event)


class FluentInputDialog(FluentCardDialog):
    """Small Fluent input dialog; the historical numeric API remains intact."""

    def __init__(self, prompt: str, default: float | str, parent=None, *, title: str = ""):
        super().__init__(parent, title=title)
        pad = scaled_px(16, minimum=10)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(pad, pad, pad, pad)
        layout.setSpacing(scaled_px(8, minimum=5))

        # The card has no title bar to carry a title, so a titled dialog
        # renders its own -- the same section title the message box
        # uses, because they are the same kind of thing.
        if title:
            layout.addWidget(FluentSectionLabel(str(title)))
        layout.addWidget(FluentLabel(prompt, self))
        self._edit = FluentLineEdit(str(default), self)
        layout.addWidget(self._edit)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch()
        ok = FluentButton("OK", self, role="primary")
        cancel = FluentButton("Cancel", self)
        ok.clicked.connect(self.accept)
        cancel.clicked.connect(self.reject)
        # The confirm sits last, where the message card puts its own.
        btn_row.addWidget(cancel)
        btn_row.addWidget(ok)
        layout.addLayout(btn_row)

    def text(self) -> str:
        """What was typed, whatever it was.

        ``getValue`` folds a Cancel and an unreadable number into one
        answer, which is right for a caller that only wants a number and
        wrong for one that has to tell a refusal from a typo.
        """

        return self._edit.text()

    def getValue(self):
        if self.exec_() == QtWidgets.QDialog.Accepted:
            try:
                return float(self._edit.text()), True
            except ValueError:
                return None, False
        return None, False

class _PythonHighlighter(QtGui.QSyntaxHighlighter):
    """Python colouring for the code editors this package hands out.

    One highlighter, because there is one code editor: the scan generator, the
    table view and the source popups are all FluentCodeEdit, and a page whose
    whole job is to be read was rendering comments in the same colour as the
    code that they explain.

    Comments first and unconditionally: a generated program leads with the
    sentence that says what its columns MEAN, and that sentence is the part a
    physicist reads.  Everything else is deliberately thin -- this is a code
    field in an instrument window, not an IDE.
    """

    KEYWORDS = (
        "and as assert async await break class continue def del elif else except "
        "finally for from global if import in is lambda nonlocal not or pass raise "
        "return try while with yield None True False"
    ).split()

    def __init__(self, document) -> None:
        super().__init__(document)

        def _format(colour: str, *, italic: bool = False, bold: bool = False):
            fmt = QtGui.QTextCharFormat()
            fmt.setForeground(QtGui.QColor(colour))
            fmt.setFontItalic(italic)
            if bold:
                fmt.setFontWeight(QtGui.QFont.DemiBold)
            return fmt

        self._comment = _format(GREEN, italic=True)
        self._string = _format(ORANGE_DARK)
        self._number = _format(ORANGE)
        self._keyword = _format(ACCENT, bold=True)
        self._rules = (
            (re.compile(r"\b(" + "|".join(self.KEYWORDS) + r")\b"), self._keyword),
            (re.compile(r"\b\d+\.?\d*([eE][-+]?\d+)?\b"), self._number),
            (re.compile(r"'[^'\n]*'|\"[^\"\n]*\""), self._string),
        )

    def highlightBlock(self, text: str) -> None:  # noqa: N802 - Qt naming
        for pattern, fmt in self._rules:
            for match in pattern.finditer(text):
                self.setFormat(match.start(), match.end() - match.start(), fmt)
        # Last, and it wins: a # inside a string is not a comment, but a string
        # inside a comment is still a comment.
        hash_at = _comment_start(text)
        if hash_at is not None:
            self.setFormat(hash_at, len(text) - hash_at, self._comment)


def _comment_start(text: str) -> int | None:
    """Where a Python comment begins on this line, ignoring quoted hashes."""

    quote = ""
    for index, character in enumerate(text):
        if quote:
            if character == quote:
                quote = ""
        elif character in "'\"":
            quote = character
        elif character == "#":
            return index
    return None


class FluentCodeEdit(QtWidgets.QPlainTextEdit):
    """A monospace code/expression editor with the house Fluent chrome (white card, light
    border, Consolas font, Fluent scrollbars).  ONE source for the look so every code editor
    -- a scan-tab generator + table view, or a source-choice popup -- renders
    identically; changing the recipe here changes them all."""

    def __init__(self, text: str = "", parent=None, *, read_only: bool = False):
        super().__init__(parent)
        if text:
            self.setPlainText(text)
        self.setReadOnly(read_only)
        pad = scaled_px(4)
        self.setStyleSheet(
            f'QPlainTextEdit {{ background: white; color: {TEXT}; border: {FIELD_BORDER_PX}px solid {PLACEHOLDER}; '
            f'border-radius: {scaled_px(4)}px; font: {fluent_font_size()}pt "Consolas", "Courier New", '
            f'monospace; padding: {pad}px; }}')
        apply_fluent_scrollbars(self)
        self._highlighter = _PythonHighlighter(self.document())

    def contextMenuEvent(self, event) -> None:  # noqa: N802 - Qt naming
        _apply_fluent_context_menu(self, event)   # Fluent right-click (undo/cut/copy/paste), not the native menu


class _StepChip(QtWidgets.QAbstractButton):
    """The step a decimal box moves by, shown in the box: ``±0.1``.

    A quiet ghost left of the stepper -- MUTED_INK, smaller than the digits
    -- that says what one notch does and is the way to change it: under
    the pointer it becomes a light-blue pill in TEXT ink, and a click asks
    for a new step.  On a box that cannot be edited it is DISABLED_INK and
    takes no pointer.  Its geometry is the box's (see
    ``FluentDoubleSpinBox._place_step_chip``).

    The chip paints and measures with ONE font it builds itself
    (``_STEP_CHIP_FONT_RATIO`` of the field's size), never ``self.font()``:
    a window-level ``QWidget { font }`` sheet overrides ``setFont``, and
    there the step came out the digits' size on their baseline -- ``0.001
    ±1`` read as a value and its uncertainty.

    THE CHIP IS CENTRED ON THE FRAME, like every part inside a field (a
    chevron, a badge).  The pill is centred on the box's inner rect, its
    height of that rect's device-pixel parity, so it stands exactly as far
    from the top edge as from the bottom; the text is centred on its own
    ink inside the pill, at rest and under the pointer alike.  It used to
    stand on the digits' baseline, which put the pill two pixels below the
    middle of the field -- the one thing in a row of centred parts that
    was not.
    """

    def __init__(self, box: "FluentDoubleSpinBox"):
        super().__init__(box)
        self.setFocusPolicy(QtCore.Qt.NoFocus)
        self.setAttribute(QtCore.Qt.WA_Hover, True)
        self._ink_font = _step_chip_font()
        self.set_live(True)

    def ink_font(self) -> QtGui.QFont:
        return self._ink_font

    def refresh_font(self) -> None:
        """The field's size changed (a fluent scale change restyles it)."""

        self._ink_font = _step_chip_font()
        self.update()

    def ink(self, text: str | None = None) -> QtCore.QRectF:
        """The ink box of ``text`` (default: the chip's own) at a baseline
        origin, in logical px."""

        path = QtGui.QPainterPath()
        path.addText(0.0, 0.0, self._ink_font, self.text() if text is None else text)
        return path.boundingRect()

    def pill(self) -> QtCore.QRectF:
        """The hover pill, in this chip: the chip's width, the text's ink
        plus even padding tall -- made the parity of the box's inner rect
        in device pixels and no taller than it -- centred on that rect."""

        ratio = float(self.devicePixelRatioF())
        inner = frame_inner_rect(self.parentWidget())
        room = max(1, int(round(inner.height() * ratio)))
        tall = min(room, int(math.ceil((self.ink().height() + 2 * scaled_px(3, minimum=2)) * ratio)))
        tall += (room - tall) & 1
        top = inner.top() - self.y() + (room - tall) // 2 / ratio
        return QtCore.QRectF(0.0, top, float(self.width()), tall / ratio)

    def set_live(self, live: bool) -> None:
        """Whether the box can be edited: a read-only box shows its step
        without offering to change it (a disabled box's chip is disabled
        with it, whatever this says)."""

        self.setEnabled(bool(live))
        if live:
            self.setCursor(QtCore.Qt.PointingHandCursor)
        else:
            self.unsetCursor()
        self.update()

    def enterEvent(self, event) -> None:  # noqa: N802 - Qt naming
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802 - Qt naming
        self.update()
        super().leaveEvent(event)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        del event
        live = self.isEnabled()
        hovered = live and (self.underMouse() or self.isDown())
        pill = self.pill()
        painter = QtGui.QPainter(self)
        if hovered:
            painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(QtGui.QColor(ACCENT_TINT))
            painter.drawRoundedRect(pill, pill.height() / 2.0, pill.height() / 2.0)
        paint_centred_ink(painter, self, pill, self._ink_font, self.text(),
                          TEXT if hovered else MUTED_INK if live else DISABLED_INK)
        painter.end()


#: The step chip's size against the digits': small enough to read as a note
#: beside the number rather than a second number (at 9 pt digits it is
#: 7.2 pt, two pixels shorter at any ratio).
_STEP_CHIP_FONT_RATIO = 0.8
#: The chip every width rule reserves room for (see "ONE RESERVED ROOM" in
#: FluentDoubleSpinBox): a step of a ten-thousandth, the finest a decimal
#: box usually takes.
_STEP_CHIP_SAMPLE = "±0.0001"


def _step_chip_font() -> QtGui.QFont:
    font = QtGui.QFont(FONT)
    font.setPointSizeF(fluent_font_size() * _STEP_CHIP_FONT_RATIO)
    return font


class FluentDoubleSpinBox(_WheelFocusGuardMixin, QtWidgets.QDoubleSpinBox):
    """One bounded Decimal value, rounded to the digits the operator can see.

    A STEP STAYS ON THE GRID.  The wheel, the arrow keys and the stepper
    move along the multiples of the step in the unit on screen: from a value
    on the grid one notch is one step, from one off it the next grid point
    in that direction.  A notch that would leave the owner's range stops at
    the last grid point inside it and never lands on a bound that is off
    the grid -- down from 1 in steps of 1 with a floor of 0.0001 stays 1,
    it does not become 0.0001 -- so down-then-up returns where it started.
    A notch never moves against its direction, and where no grid point
    inside the range lies that way it does not move at all (that direction
    is shown dimmed).  Only a NAMED value -- typed, or set by the owner --
    is clamped to the bound: that is someone naming a number and the bound
    refusing it.  Prefix changes stay exact decimal shifts.  Display
    normalization updates the draft separately from a user edit, so
    resizing a device control cannot be mistaken for a hardware Apply.
    """

    #: CLASS attributes, not instance ones.  Qt asks this widget for its text
    #: from inside setRange and setDecimals, which run in __init__ before any
    #: assignment could have happened -- and a plain AttributeError raised
    #: inside a Qt slot does not become a traceback, it ends the process.
    #: Declared here, there is no instant in which they are missing.
    valueNormalized = QtCore.pyqtSignal()

    _normalization_pending = False
    _step_chip = None
    #: The right text margin the step chip added: an owner's line edit may
    #: keep a margin of its own (a binding badge), which the chip never takes.
    _chip_margin = 0
    _display_text = ""
    _text_width = 0
    _number_metrics = None
    _number_geometry = None
    _unit = "1"
    _shown_unit = "1"
    _value = Decimal(0)
    _step = Decimal(1)
    #: The owner's bounds, None on a side nobody declared.  Qt cannot say
    #: "no bound": with none declared its double range is the whole double
    #: line, and that placeholder is not a number anyone asked for.  Read
    #: back as one it did no harm as a clamp, but converted into a shown
    #: unit it did: the top of a dBm box read in milliwatts is 10**(DBL_MAX
    #: / 10), which is not finite, and a step that converted both ends before
    #: moving refused to move at all.
    _low: Decimal | None = None
    _high: Decimal | None = None

    def width_for(self, sample: str) -> int:
        """The width at which ``sample`` shows whole.

        Measured the way the box measures a number it is asked to show --
        its frame, buttons, padding and focus inset at this style and
        scale, plus the two pixels the fit rule keeps -- rather than a
        nominal sum rounded once, which at a fractional scale came out
        four pixels short and refused the very number it was sized for.
        The edit is placed by the box's resize, so a box never shown is
        resized in place first; the chrome does not depend on the width.
        """

        QtWidgets.QApplication.sendEvent(
            self, QtGui.QResizeEvent(self.size(), self.size())
        )
        # The chip's RESERVED room counts -- one width whatever the step, so
        # boxes sized for one sample end on one edge and a new step moves
        # nothing; a box sized for a sample shows its step.
        chrome = self.width() - self._digit_room() + self._step_chip_reserve()
        return self.fontMetrics().horizontalAdvance(str(sample)) + 2 + chrome

    def sizeHint(self) -> QtCore.QSize:  # noqa: N802
        size = super().sizeHint()
        size.setWidth(
            self.fontMetrics().horizontalAdvance("-123.456e-12") + _spin_field_chrome() + self._step_chip_reserve()
        )
        return size

    def minimumSizeHint(self) -> QtCore.QSize:  # noqa: N802
        # The stepper and a short number.  Not the step chip: it yields to
        # the digits at a width that cannot hold both.
        digits = self.fontMetrics().horizontalAdvance("0.0") + 2 + _spin_field_chrome()
        return QtCore.QSize(max(scaled_px(48), digits), self.minimumHeight())

    def __init__(self, parent=None):
        super().__init__(parent)
        self._install_wheel_focus_guard()
        # A HALF-TYPED NUMBER IS NOT A VALUE.  With keyboard tracking on,
        # Qt interprets the text on every keystroke: deleting the 1 from
        # "0.1" leaves "0.", which is read as zero and snapped to this
        # box's own minimum before the next character can be typed.  The
        # number is judged when the operator is done with it -- Enter, or
        # the focus leaving -- which is also when the range applies.
        self.setKeyboardTracking(False)
        self.setButtonSymbols(QtWidgets.QAbstractSpinBox.PlusMinus)
        self.setMinimumHeight(control_height())
        # See FluentSpinBox: left-align + zero text margins for left-padding
        # parity with the line edits.
        self.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.lineEdit().setTextMargins(0, 0, 0, 0)
        self.setStyleSheet(fluent_spinbox_stylesheet("QDoubleSpinBox"))
        # The stepper tint is painted from the hovered sub-control, which Qt
        # tracks only for a widget that asks for hover events.
        self.setAttribute(QtCore.Qt.WA_Hover, True)
        self.editingFinished.connect(self._update_number_geometry)
        self._step_chip = _StepChip(self)
        self._step_chip.clicked.connect(self._on_edit_step)
        # The chip yields to what is TYPED as well as to what is shown: an
        # operator's long number takes the chip's room the moment it needs
        # it, and the chip comes back when the number fits beside it again.
        self.lineEdit().textChanged.connect(self._on_text_for_chip)

        # No range of its own: a bound is the owner's, declared through
        # setRange, and until then the box holds whatever it is given.
        super().setRange(-sys.float_info.max, sys.float_info.max)
        # Every decimal digit, so Qt's shadow never rounds the value; an
        # owner that declares a resolution says so with setDecimals.
        self.setDecimals(323)
        self._adopt_step(Decimal(1))
        self._refresh_step_chip()

    # ------------------------------------------------------------ the value

    def setValue(self, value: float) -> None:  # noqa: N802 - Qt API name
        """Project the owner's number through the same bounds/visible precision."""

        try:
            number = _finite_decimal(value)
        except ValueError:
            return
        self._commit(number, notify_normalized=True)

    def _bounded(self, number: Decimal) -> Decimal:
        """``number`` inside the owner's range: a box never holds a value
        outside the bound its owner declared, whoever named it."""

        if self._low is not None and number < self._low:
            return self._low
        if self._high is not None and number > self._high:
            return self._high
        return number

    def setRange(self, minimum: float, maximum: float, *, value: object | None = None,
                 unit: str | None = None, reject_unrepresentable: bool = False) -> bool:  # noqa: N802
        """The owner's range -- applied when it changes, see FluentSpinBox.

        The value it holds is moved inside the new range BEFORE Qt is told:
        Qt clamps its own double inside setRange and emits valueChanged from
        there, and a slot reading ``decimalValue`` during that emission must
        see the same number the signal carries.  Left to Qt alone, the shadow
        was clamped and the decimal was not, so a box showing 5 still stepped
        from a hidden 7 -- the first notch down went nowhere.
        """

        low, high = float(minimum), float(maximum)
        # Qt's reading of an inverted range: the top is raised to the bottom.
        high = max(low, high)
        wanted_unit = self._unit if unit is None else str(unit).strip() or "1"
        if ((low, high) == (self.minimum(), self.maximum())
                and wanted_unit == self._unit and not self.property("numericError")
                and (value is None or _finite_decimal(value) == self._value)):
            return True
        # The whole double line is Qt's way of saying "no bound", and the form
        # says it the same way for a side its owner left open.
        # From what was GIVEN, not from its float: an integer bound past
        # 2**53 read through a double is a different number.
        previous = self._low, self._high, self._unit, self._shown_unit, self._display_text
        previous_value = self._value
        self._low = None if low <= -sys.float_info.max else _finite_decimal(minimum)
        self._high = None if high >= sys.float_info.max else _finite_decimal(maximum)
        # ``unit`` is what the NUMBER is in -- the owner's unit, and its
        # range's; the digits' own spelling is the picker's beside the box.
        # Re-declared, it is not changed: every projection re-configures the
        # box it keeps, and the spelling the operator chose to read it in is
        # theirs -- reset on every draft, "ms" snapped back to "s" the
        # moment the wheel stopped.
        if wanted_unit != self._unit:
            self._unit = self._shown_unit = wanted_unit
        self._display_text = ""
        requested = self._value if value is None else _finite_decimal(value)
        clipped = self._bounded(self._quantized(requested))
        try:
            number = self._visible_value(clipped)
        except (UnitError, ValueError, ArithmeticError) as error:
            self.setProperty("numericError", str(error))
            if reject_unrepresentable:
                self._low, self._high, self._unit, self._shown_unit, self._display_text = previous
            else:
                # Device bounds are facts, not an optional display request.
                # Keep those facts and mark the input unavailable until it
                # fits, rather than continue authoring against an old range.
                self._value = clipped
                self._display_text = ""
                with signals_blocked(self):
                    super().setRange(low, high)
                    super().setValue(float(clipped))
                    self.lineEdit().clear()
            self._queue_normalization()
            return False
        self._value = number
        with signals_blocked(self):
            super().setRange(low, high)
            self._commit(number)
        if number != previous_value:
            self.valueChanged[float].emit(float(number))
            self.valueChanged[str].emit(self.text())
        if number != requested:
            self._queue_normalization()
        return True

    def setMinimum(self, minimum: float) -> None:  # noqa: N802 - Qt API name
        self.setRange(minimum, self.maximum())

    def setMaximum(self, maximum: float) -> None:  # noqa: N802 - Qt API name
        self.setRange(self.minimum(), maximum)

    def setDecimals(self, places: int) -> None:  # noqa: N802 - Qt API name
        """The owner's resolution.  The value is quantized to it before Qt
        re-rounds its shadow, for the reason ``setRange`` gives."""

        places = int(places)
        if places == self.decimals():
            return
        previous_text = self._display_text
        self._display_text = ""
        try:
            value = self._visible_value(self._quantized(self._value, places))
        except ValueError as error:
            self._display_text = previous_text
            self.setProperty("numericError", str(error))
            self._queue_normalization()
            return
        self._value = value
        super().setDecimals(places)
        self._refresh_step_chip()
        self._commit(self._value, notify_normalized=True)

    def setSingleStep(self, step: float) -> None:  # noqa: N802 - Qt API name
        """The owner's (or the operator's) step, shown by the step chip.
        Until one is named a notch is 1, whatever the box holds."""

        try:
            magnitude = abs(_finite_decimal(step))
        except ValueError:
            return
        if magnitude == 0:
            return
        self._adopt_step(magnitude)

    def _adopt_step(self, magnitude: Decimal) -> None:
        changed = magnitude != self._step
        self._step = magnitude
        super().setSingleStep(float(magnitude))
        if changed:
            self._refresh_step_chip()

    def decimalValue(self) -> Decimal:  # noqa: N802 - Qt API style
        """The value as the box holds it -- exact, in the owner's unit."""

        return self._value

    def _commit(self, number: Decimal, *, notify_normalized: bool = False, user_edit: bool = False) -> bool:
        """Make ``number`` the value, in the owner's unit, shadow and all."""

        requested = number
        try:
            number = self._visible_value(self._bounded(self._quantized(number)))
        except ValueError as error:
            self.setProperty("numericError", str(error))
            self._queue_normalization()
            return False
        self.setProperty("numericError", "")
        if user_edit and number != self._value:
            self._normalization_pending = False
        self._value = number
        # Qt compares the double against its own range and last value and
        # emits valueChanged; its rounding to decimals() is why the decimal
        # is quantized to those decimals first, so shadow and value agree.
        super().setValue(float(number))
        # Qt only rewrites the text when the double changed; a decimal that
        # differs in a digit the double cannot hold would otherwise be shown
        # from the float.
        self.lineEdit().setText(self.textFromValue(float(number)))
        # The chip shows only beside a number it leaves room for.
        self._place_step_chip()
        if notify_normalized and number != requested:
            self._queue_normalization()
        return True

    def _visible_value(self, number: Decimal) -> Decimal:
        if number == self._value and self._display_text:
            return number
        shown = self._shown_from_value(number)
        if (self.isReadOnly() or not self.isEnabled()
                or self._number_metrics is None or self._text_width <= 0):
            # This is a projection of another owner's value.  Keep exact
            # text for Qt readback/copy and let QLineEdit handle overflow.
            self._display_text = format_quantity(shown, "1")
            return number
        low, high = self._shown_bound(self._low), self._shown_bound(self._high)
        if low is not None and high is not None and low > high:
            low, high = high, low
        rounded, text, required = _visible_decimal(
            shown, low, high, self.decimals() == 0, self._text_width, self._number_metrics,
        )
        self._display_text = text
        return self._bounded(self._quantized(self._value_from_shown(rounded)))

    def _queue_normalization(self) -> None:
        if not self._normalization_pending:
            self._normalization_pending = True
            QtCore.QMetaObject.invokeMethod(self, "_deliver_normalization", QtCore.Qt.QueuedConnection)

    @QtCore.pyqtSlot()
    def _deliver_normalization(self) -> None:
        if not self._normalization_pending:
            return
        self._normalization_pending = False
        if self.isEnabled() and not self.isReadOnly():
            self.valueNormalized.emit()

    def _quantized(self, number: Decimal, places: int | None = None) -> Decimal:
        """``number`` at the resolution the owner declared, when it did.

        Never signals: Qt asks this box for the text of its range ends, and
        with no bound declared those are the ends of the double line, which
        have more digits than the default decimal context holds.
        """

        places = int(self.decimals()) if places is None else int(places)
        if places >= 323 or not number.is_finite():
            return number
        if places == 0:
            return number.to_integral_value(rounding=ROUND_HALF_EVEN)
        with localcontext() as context:
            context.prec = max(context.prec, number.adjusted() + places + 2)
            return number.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_EVEN)

    # ------------------------------------------------------------- the unit

    def valueUnit(self) -> str:  # noqa: N802 - Qt API name
        return self._unit

    def setShownUnit(self, unit: str) -> None:  # noqa: N802 - Qt API name
        """Convert to the selected unit and accept its visible precision.

        The canonical unit is unchanged. Any width-rounding is converted
        back into the actual value, not retained as a display-only spelling.
        """

        wanted = str(unit).strip() or self._unit
        if wanted == self._shown_unit:
            return
        if wanted != self._unit:
            DEFAULT_UNITS.convert(0.0, self._unit, wanted)
        previous = self._shown_unit, self._display_text
        self._shown_unit = wanted
        self._display_text = ""
        if not self._commit(self._value, user_edit=True):
            self._shown_unit, self._display_text = previous

    def shownUnit(self) -> str:  # noqa: N802 - Qt API name
        return self._shown_unit

    def _shown_from_value(self, number: Decimal) -> Decimal:
        return DEFAULT_UNITS.convert_decimal(number, self._unit, self._shown_unit)

    def _value_from_shown(self, number: Decimal) -> Decimal:
        return DEFAULT_UNITS.convert_decimal(number, self._shown_unit, self._unit)

    def _shown_bound(self, edge: Decimal | None) -> Decimal | None:
        return _bound_in_unit(edge, self._unit, self._shown_unit)

    # ------------------------------------------------------------- stepping

    def _notch(self, steps: int) -> Decimal | None:
        """Where ``steps`` notches would take the value, in the unit on
        screen, or None when they do not move it (see the class: the grid
        inside the range, never against the notch's direction)."""

        try:
            shown = self._shown_from_value(self._value)
            low, high = self._shown_bound(self._low), self._shown_bound(self._high)
            if low is not None and high is not None and low > high:
                low, high = high, low
            moved = _grid_step_inside(shown, self._step, steps, low, high)
        except (UnitError, ValueError, ArithmeticError):
            return None
        return None if moved == shown else moved

    def stepBy(self, steps: int) -> None:  # noqa: N802 - Qt API name
        """One notch is one step of the unit on screen, on that step's grid."""

        steps = int(steps)
        moved = self._notch(steps) if steps else None
        if moved is None:
            return
        try:
            self._commit(self._quantized(self._value_from_shown(moved)), user_edit=True)
        except (UnitError, ValueError, ArithmeticError):
            return
        if self.hasFocus():
            self.lineEdit().selectAll()

    def stepEnabled(self):  # noqa: N802 - Qt API name
        """The directions a notch would move the value in -- asked by Qt's
        wheel, arrow keys and auto-repeating stepper before each step, and
        read by the stepper's paint to dim a direction that cannot move."""

        enabled = QtWidgets.QAbstractSpinBox.StepNone
        if self.isReadOnly():
            return enabled
        if self._notch(1) is not None:
            enabled |= QtWidgets.QAbstractSpinBox.StepUpEnabled
        if self._notch(-1) is not None:
            enabled |= QtWidgets.QAbstractSpinBox.StepDownEnabled
        return enabled

    # ----------------------------------------------------------- the text

    def textFromValue(self, value: float) -> str:  # noqa: N802 - Qt API
        # NEVER raises, for the same reason valueFromText does not: Qt calls
        # this while painting, and an exception out of a Qt slot ends the
        # process.  A unit this registry has never heard of is a defect to fix
        # where it was declared, not a reason for the window to die drawing a
        # number it could otherwise have shown.
        if self.property("numericError") and not self._display_text:
            return ""
        try:
            number = self._value if float(self._value) == float(value) else _finite_decimal(value)
        except ValueError:
            return str(value)
        try:
            if number == self._value and self._display_text:
                return self._display_text
            number = self._quantized(number)
            # DIGITS ONLY.  A box is for the number; which unit that number
            # is read in is said once, beside it, by the row that owns it --
            # printing the symbol in here as well put the same fact in two
            # places and left no room to type in the box it was crowding.
            return format_quantity(self._shown_from_value(number), "1")
        except (UnitError, ValueError, ArithmeticError):
            return str(number)

    def valueFromText(self, text: str) -> float:  # noqa: N802 - Qt API
        """The digits typed, read as a number OF THE UNIT ON SCREEN.

        NEVER raises.  Qt calls this from inside its own event handling, and
        an exception leaving a Qt slot ends the process instead of the edit.
        The fallback may NOT ask the box for its value: value() interprets the
        text, which calls back into here, which would ask again -- one
        unreadable keystroke and the stack is gone.

        Typed numbers and steps share the owner's bounds; visible precision
        is part of the committed value, not just its formatting.
        """

        try:
            typed = Decimal(str(text).strip())
            if not typed.is_finite():
                raise ValueError(text)
            number = self._visible_value(self._bounded(self._quantized(self._value_from_shown(typed))))
        except (UnitError, ValueError, ArithmeticError) as error:
            self.setProperty("numericError", str(error))
            self._queue_normalization()
            return float(self._value)
        self.setProperty("numericError", "")
        if number != self._value:
            self._normalization_pending = False
        self._value = number
        return float(number)

    def validate(self, text: str, position: int):
        """Accept while it is being typed; judge when it could be finished.

        A box holds digits, a sign and a decimal point, and nothing else.  Its
        scale is the picker's to say, so there is no suffix to type past and
        no partially-typed word to keep Intermediate for -- only a number that
        is not finished yet, which "-" and "." and "" are.
        """

        if text == self._display_text and not self.property("numericError"):
            return QtGui.QValidator.Acceptable, text, position
        stripped = str(text).strip()
        if self.decimals() == 0 and "." in stripped:
            # A box at zero decimals holds whole numbers, and a decimal
            # point is not a digit it takes -- the way Qt's own integer
            # box refuses one, rather than rounding what was typed.
            return QtGui.QValidator.Invalid, text, position
        if not stripped or stripped in ("+", "-", ".", "+.", "-."):
            return QtGui.QValidator.Intermediate, text, position
        if self.decimals() != 0 and re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)[eE][+-]?", stripped):
            return QtGui.QValidator.Intermediate, text, position
        try:
            typed = Decimal(stripped)
            if not typed.is_finite():
                return QtGui.QValidator.Invalid, text, position
        except ArithmeticError:
            return QtGui.QValidator.Invalid, text, position
        if self._number_metrics is not None and self._number_metrics.horizontalAdvance(text) + 2 > self._text_width:
            low, high = self._shown_bound(self._low), self._shown_bound(self._high)
            try:
                _value, fitted, width = _visible_decimal(
                    typed, low, high, self.decimals() == 0, self._text_width, self._number_metrics,
                )
            except ValueError as error:
                self.setProperty("numericError", str(error))
                self._queue_normalization()
                return QtGui.QValidator.Intermediate, text, position
            if width <= self._text_width:
                return QtGui.QValidator.Acceptable, fitted, min(position, len(fitted))
        return QtGui.QValidator.Acceptable, text, position

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        _seat_line_edit_text(self.lineEdit())
        self._place_step_chip()
        self._update_number_geometry()

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        _seat_line_edit_text(self.lineEdit())
        self._place_step_chip()
        self._update_number_geometry()

    def changeEvent(self, event) -> None:  # noqa: N802
        super().changeEvent(event)
        if event.type() in (QtCore.QEvent.FontChange, QtCore.QEvent.StyleChange):
            _seat_line_edit_text(self.lineEdit())
        if event.type() in (
            QtCore.QEvent.FontChange,
            QtCore.QEvent.StyleChange,
            QtCore.QEvent.EnabledChange,
            QtCore.QEvent.ReadOnlyChange,
        ):
            if self._step_chip is not None:
                if event.type() in (QtCore.QEvent.FontChange, QtCore.QEvent.StyleChange):
                    self._step_chip.refresh_font()
                self._step_chip.set_live(not self.isReadOnly())
                self._place_step_chip()
            if hasattr(self, "_display_text"):
                self._display_text = ""
                if self.isReadOnly() or not self.isEnabled():
                    with signals_blocked(self):
                        self._commit(self._value)
                    self._normalization_pending = False
                    self.lineEdit().setCursorPosition(0)
            self._update_number_geometry()

    def _update_number_geometry(self) -> None:
        edit = self.lineEdit()
        if edit is None or not self.isVisible():
            return
        # The whole digit room: the chip yields to the digits, never the
        # other way round (see the step chip below).
        geometry = (self._digit_room(), edit.font().key())
        if geometry == self._number_geometry and self._display_text:
            return
        self._number_geometry = geometry
        self._text_width = geometry[0]
        self._number_metrics = QtGui.QFontMetrics(edit.font())
        self._display_text = ""
        if edit.isModified() and edit.hasFocus():
            return
        previous = self._value
        with signals_blocked(self):
            self._commit(previous)
        if self._value != previous:
            self._queue_normalization()

    def offers_step_edit(self) -> bool:
        """Whether this box lets the operator choose its step.

        A whole-number box moves one at a time; choosing a step size means
        nothing there.
        """

        return self.decimals() != 0

    # ------------------------------------------------------------ the step chip
    #
    # A decimal box shows the step one notch moves it by, inside the field
    # just left of the stepper: ``±0.1`` in a quiet ghost that turns into a
    # light-blue pill under the pointer and asks for a new step when
    # clicked.  The step used to be reachable only through a right-click
    # menu -- or a six-pixel "." strip before that -- and an operator who
    # cannot see the step cannot know what the arrows will do.
    #
    # THE CHIP YIELDS TO THE DIGITS.  The value is fitted to the field's
    # whole digit room (``_text_width``, the chip not counted), and the chip
    # is shown only while the text in the edit -- shown or being typed --
    # still fits beside it; the digits then end before it (a right text
    # margin).  So the chip never rounds a value, never covers a digit, and
    # at a width that cannot hold both it is simply not there -- Edit step…
    # in the context menu still is.
    #
    # ONE RESERVED ROOM.  Every width rule that sizes a box for a sample
    # (width_for, sizeHint, the form's number column) counts the room of a
    # canonical chip, ``_STEP_CHIP_SAMPLE``, not the chip shown now: a
    # column whose width followed the step text ended two forms of one card
    # on two edges and jumped when a step was edited.  A wider step simply
    # takes more of the digits' room while the number leaves it.

    def _step_chip_text(self) -> str:
        step = self._step
        shown = _decimal_text(step) if -4 <= step.adjusted() <= 6 else f"{float(step):g}"
        return f"±{shown}"

    def _step_chip_metrics(self, text: str | None = None) -> tuple[int, int, int]:
        """``(chip width, gap to the stepper, gap to the digits)`` in
        logical px for ``text`` (default: the step shown now): the chip is
        the text's ink plus its padding, measured with the chip's own font,
        and as tall as the frame's inner rect (its pill is centred there)."""

        ink = self._step_chip.ink(self._step_chip_text() if text is None else text)
        pad = scaled_px(6, minimum=4)
        width = int(math.ceil(ink.width())) + 2 * pad
        return width, scaled_px(3, minimum=2), scaled_px(4, minimum=2)

    def _offers_step_chip(self) -> bool:
        return (
            self._step_chip is not None
            and self.offers_step_edit()
            and self.buttonSymbols() != QtWidgets.QAbstractSpinBox.NoButtons
        )

    def _step_chip_room(self, text: str | None = None) -> int:
        """How much the chip showing ``text`` (default: the step now) takes
        from the digits: the right text margin.

        The edit field ends at the frame's padding before the stepper; the
        chip sits ``gap`` left of the stepper, and the digits stop ``text
        gap`` before the chip.  None of it depends on the box's width."""

        if not self._offers_step_chip():
            return 0
        width, gap, text_gap = self._step_chip_metrics(text)
        before_stepper = FIELD_BORDER_PX + scaled_px(EDIT_PADDING_H)
        return max(0, width + gap + text_gap - before_stepper)

    def _step_chip_reserve(self) -> int:
        """The chip's share of every width rule: the room of the canonical
        chip, the same for every decimal box at this scale."""

        return self._step_chip_room(_STEP_CHIP_SAMPLE)

    def _digit_room(self) -> int:
        """The width the digits have with no chip beside them."""

        return max(1, _numeric_text_width(self.lineEdit()) + self._chip_margin)

    def set_tone(self, *, edge: str | None = None, fill: str = "white", ink: str = TEXT) -> None:
        """Wear a state's colours on the frame -- a bound field's edge, fill
        and ink -- or, with no ``edge``, the plain field's.  A bound edge
        keeps its colour under focus, as a bound line edit's does."""

        rest, focus = (_FIELD_EDGE_REST, _FIELD_EDGE_FOCUS) if edge is None else (edge, edge)
        self.setStyleSheet(fluent_spinbox_stylesheet(
            "QDoubleSpinBox", edge=rest, focus_edge=focus, fill=fill, ink=ink))

    def _set_chip_margin(self, room: int) -> None:
        """Give the chip ``room`` of the right text margin, on top of the
        edit's own: only the chip's share changes."""

        if room == self._chip_margin:
            return
        edit = self.lineEdit()
        margins = edit.textMargins()
        edit.setTextMargins(margins.left(), margins.top(),
                            margins.right() - self._chip_margin + room, margins.bottom())
        self._chip_margin = room

    def _on_text_for_chip(self, _text: str) -> None:
        self._place_step_chip()

    def setLineEdit(self, edit: QtWidgets.QLineEdit) -> None:  # noqa: N802 - Qt API name
        """Adopt an owner's line edit (a bound field) as the editor inside
        this frame.

        The box draws the frame, so the edit draws none and takes exactly
        the room the frame leaves -- a field's own minimum height would push
        it, its digits and its badge below the frame's centre.  It keeps its
        own text margins (a badge's room); the chip follows what is typed
        into it as it did the box's own edit."""

        edit.setMinimumHeight(0)
        edit.setStyleSheet(fluent_inner_edit_stylesheet())
        super().setLineEdit(edit)
        self._chip_margin = 0
        edit.textChanged.connect(self._on_text_for_chip)
        self._number_geometry = None
        self._refresh_step_chip()

    def edit_field(self) -> QtCore.QRect:
        """Where this box lays its line edit out, asked of its style: the
        edit's own geometry catches up only when a hidden box is shown, and
        a part placed over the edit (a binding badge) is placed before."""

        option = QtWidgets.QStyleOptionSpinBox()
        self.initStyleOption(option)
        return self.style().subControlRect(
            QtWidgets.QStyle.CC_SpinBox, option, QtWidgets.QStyle.SC_SpinBoxEditField, self)

    def _place_step_chip(self) -> None:
        chip = self._step_chip
        if chip is None:
            return
        edit = self.lineEdit()
        room = self._step_chip_room()
        shown = self._offers_step_chip()
        if shown:
            metrics = self._number_metrics or edit.fontMetrics()
            needed = metrics.horizontalAdvance(edit.text()) + 2
            shown = needed <= self._digit_room() - room
        if not shown:
            chip.hide()
            self._set_chip_margin(0)
            return
        width, gap, _text_gap = self._step_chip_metrics()
        stepper, _up, _down = _spin_stepper_cells(self)
        inner = frame_inner_rect(self)
        top = int(math.ceil(inner.top()))
        bottom = int(math.floor(inner.bottom()))
        right = int(math.floor(stepper)) - gap
        text = self._step_chip_text()
        if chip.text() != text:
            chip.setText(text)
        chip.setGeometry(right - width, top, width, max(1, bottom - top))
        tip = (f"Step {text[1:]} · click to change" if not self.isReadOnly() and self.isEnabled()
               else f"Step {text[1:]}")
        if chip.toolTip() != tip:
            chip.setToolTip(tip)
        self._set_chip_margin(room)
        if chip.isHidden():
            chip.show()
            chip.raise_()

    def _refresh_step_chip(self) -> None:
        """The step or the box's kind changed: show it, and re-fit the digits
        to the room the chip leaves them."""

        if self._step_chip is None:
            return
        self._place_step_chip()
        self.updateGeometry()
        self._update_number_geometry()

    def contextMenuEvent(self, event) -> None:  # noqa: N802 - Qt naming
        # "Edit step…" stays here as well, with the box's other commands; the
        # chip is the way that is seen.
        extra = (("Edit step…", self._on_edit_step),) if self.offers_step_edit() else ()
        _apply_fluent_context_menu(self.lineEdit(), event, extra=extra)

    def _on_edit_step(self) -> None:
        current = self.singleStep()
        dialog = FluentInputDialog("Edit step", current, self)
        dialog._edit.setMaxLength(self.lineEdit().maxLength())
        value, ok = dialog.getValue()
        if ok and value is not None:
            self.setSingleStep(value)

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        _paint_fluent_spin_buttons(self)


def _visible_decimal(
    number: Decimal,
    low: Decimal | None,
    high: Decimal | None,
    integral: bool,
    available_width: int,
    metrics: QtGui.QFontMetrics,
    *,
    point: bool = False,
) -> tuple[Decimal, str, int]:
    """The most precise bounded decimal whose complete text fits the editor.

    The returned number, not an unrounded input, is what the control owns.
    Only fractional places are rounded. The same decimal grid is restricted
    to the real bounds; it never crosses them or uses scientific notation.
    ``point`` keeps a decimal point in the text (``4.0``, never ``4``).
    """
    from decimal import ROUND_HALF_UP

    if low is not None:
        number = max(number, low)
    if high is not None:
        number = min(number, high)
    if not integral:
        number = _finite_decimal(float(number))
    if low is not None:
        number = max(number, low)
    if high is not None:
        number = min(number, high)
    digits = len(number.as_tuple().digits)
    with localcontext() as context:
        context.prec = max(28, digits + 2, number.adjusted() + 2)
        places = 0 if integral else max(0, -number.as_tuple().exponent)
        for decimals in range(places, -1, -1):
            quantum = Decimal(1).scaleb(-decimals)
            rounded = number.quantize(quantum, rounding=ROUND_HALF_UP)
            if low is not None and rounded < low:
                rounded = low.quantize(quantum, rounding=ROUND_CEILING)
            if high is not None and rounded > high:
                rounded = high.quantize(quantum, rounding=ROUND_FLOOR)
            if ((low is not None and rounded < low)
                    or (high is not None and rounded > high)):
                continue
            rounded = rounded.normalize() + Decimal(0)
            fixed = _decimal_text(rounded, point=point)
            width = metrics.horizontalAdvance(fixed) + 2
            if width <= available_width:
                return rounded, fixed, width
    raise ValueError("The numeric field is too narrow to display a value within its limits; choose another unit.")


def _decimal_text(number: Decimal, *, point: bool = False) -> str:
    """``number`` as a numeric box spells it: plain digits, never scientific.

    ``point`` keeps a decimal point in the text (``4.0``, never ``4``).
    """

    fixed = format(number.normalize() + Decimal(0), "f")
    if point and "." not in fixed:
        fixed += ".0"
    return fixed


def _finite_decimal(value: object) -> Decimal:
    """The shortest decimal that reads back as ``value``, or ValueError.

    A Qt virtual that raised anything else would end the process, so inside
    the box a number that is not a number is exactly one kind of failure.
    """

    number = decimal_of(value)
    if number is None or not number.is_finite():
        raise ValueError(f"not a finite number: {value!r}")
    return number


def _bound_in_unit(edge: object, unit: str, shown: str) -> Decimal | None:
    """One declared bound in the unit on screen, or None where there is
    none there: 0 W has no dBm to be, so a floor of 0 W is no floor in
    dBm, and an undeclared side has no bound in any unit.

    Both numeric boxes ask this.  Converted without the question, the
    quantity line edit raised out of a resize for a W field read in dBm.
    """

    if edge is None:
        return None
    try:
        return DEFAULT_UNITS.convert_decimal(edge, unit, shown)
    except (UnitError, ValueError, ArithmeticError):
        return None


def _grid_step(value: Decimal, step: Decimal, steps: int) -> Decimal:
    """``steps`` notches from ``value`` along the grid of multiples of ``step``.

    On the grid, a notch is a step.  Off it, the first notch goes to the
    nearest grid point in its direction and the rest are steps from there.
    """

    ratio = value / step
    if ratio == ratio.to_integral_value():
        index = int(ratio) + steps
    elif steps > 0:
        index = int(ratio.to_integral_value(rounding=ROUND_FLOOR)) + steps
    else:
        index = int(ratio.to_integral_value(rounding=ROUND_CEILING)) + steps
    return (Decimal(index) * step).normalize() + Decimal(0)


def _grid_step_inside(
    value: Decimal, step: Decimal, steps: int, low: Decimal | None, high: Decimal | None,
) -> Decimal:
    """``steps`` notches from ``value`` on the grid of multiples of ``step``,
    inside ``[low, high]`` (None: no bound on that side).

    A notch past a bound stops at the last grid point inside it -- never on
    the bound itself when the bound is off the grid.  The result never lies
    against the notches' direction: where no grid point inside the range
    lies that way (a value already at, or below, the lowest one, or a step
    wider than the whole range), the value is returned unchanged.
    """

    moved = _grid_step(value, step, steps)
    if low is not None and moved < low:
        moved = (low / step).to_integral_value(rounding=ROUND_CEILING) * step
    if high is not None and moved > high:
        moved = (high / step).to_integral_value(rounding=ROUND_FLOOR) * step
    moved = moved.normalize() + Decimal(0)
    if (low is not None and moved < low) or (high is not None and moved > high):
        return value
    if (steps < 0 and moved >= value) or (steps > 0 and moved <= value):
        return value
    return moved


def fluent_integer_box(
    *, minimum: int | None = None, maximum: int | None = None, parent=None
) -> FluentDoubleSpinBox:
    """A box for a whole number, bounded only where its owner bounded it.

    Qt's integer spin stops at signed 31 bits and cannot say "no bound"; a
    zero-decimal decimal box holds every Python int exactly and a side left
    None is a side with no bound, so the control's domain IS the owner's.
    """

    box = FluentDoubleSpinBox(parent)
    box.setDecimals(0)
    box.setSingleStep(1)
    box.setRange(
        -sys.float_info.max if minimum is None else int(minimum),
        sys.float_info.max if maximum is None else int(maximum),
    )
    return box


def fluent_count_box(
    *, minimum: int = 0, maximum: int = (1 << 32) - 1, parent=None
) -> FluentDoubleSpinBox:
    """A box for a whole number of things the board counts in 32 bits: the
    integer box with the hardware's own uint32 domain, so the control and
    the count share one.  One factory, because three pages each spelled
    these settings by hand."""

    return fluent_integer_box(minimum=int(minimum), maximum=int(maximum), parent=parent)


class FluentCheckBox(QtWidgets.QCheckBox):
    """A square box with a painted tick.

    It was a circle, and a round box reads as a radio button -- one choice
    of several -- which an independent on/off is not.  The box is a solid
    fill whose edge is that fill, never an outline: unticked it is
    NEUTRAL_FILL, dark enough to read on a white card and on the BG face of
    a Spacer card; ticked it is ACCENT with a white tick.  Under the pointer
    the fill steps one shade toward blue; disabled it is DIVIDER, a ticked
    one with a BG tick.  The tick is centred on the box's device-pixel
    centre.
    """

    def __init__(self, text: str = "", parent=None):
        super().__init__(text, parent)
        self.setMinimumHeight(control_height())
        rest_hover = _mixed(NEUTRAL_FILL, ACCENT, 0.4).name()
        self.setStyleSheet(
            f"""
            QCheckBox {{
                color: {TEXT};
                font: {fluent_font_size()}pt "{FONT}";
                spacing: {scaled_px(6)}px;
                background: transparent;
                outline: none;
            }}
            QCheckBox::indicator {{
                width: {scaled_even_px(16)}px;
                height: {scaled_even_px(16)}px;
                border-radius: {_radius()}px;
                border: {FIELD_BORDER_PX}px solid {NEUTRAL_FILL};
                background: {NEUTRAL_FILL};
            }}
            QCheckBox::indicator:hover {{ border-color: {rest_hover}; background: {rest_hover}; }}
            QCheckBox::indicator:checked {{ border-color: {ACCENT}; background: {ACCENT}; }}
            QCheckBox::indicator:checked:hover {{ border-color: {ACCENT_HOVER}; background: {ACCENT_HOVER}; }}
            QCheckBox:disabled {{ color: {DISABLED_INK}; }}
            QCheckBox::indicator:disabled, QCheckBox::indicator:checked:disabled {{
                border-color: {DIVIDER}; background: {DIVIDER};
            }}
            """
        )

    def _indicator_rect(self) -> QtCore.QRect:
        option = QtWidgets.QStyleOptionButton()
        self.initStyleOption(option)
        return self.style().subElementRect(QtWidgets.QStyle.SE_CheckBoxIndicator, option, self)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        # The sheet draws the box; the tick is drawn here, because a sheet
        # can only fill it.  Its shape is laid out about the box's centre
        # on a device pixel, so the tick sits in the middle of the box at
        # any ratio instead of half a pixel low.
        super().paintEvent(event)
        if not self.isChecked():
            return
        box = QtCore.QRectF(self._indicator_rect())
        cx = device_snap(self, box.center().x())
        cy = device_snap(self, box.center().y(), axis="y")
        w, h = box.width(), box.height()
        # The tick's own extent: 0.26..0.75 of the box across, 0.34..0.68
        # down -- centred on the box, its corners keep those proportions.
        points = ((0.26, 0.52), (0.43, 0.68), (0.75, 0.34))
        dx = cx - (0.26 + 0.75) / 2.0 * w
        dy = cy - (0.34 + 0.68) / 2.0 * h
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        pen = QtGui.QPen(QtGui.QColor(SURFACE if self.isEnabled() else BG))
        pen.setWidthF(max(1.5, w / 8.0))
        pen.setCapStyle(QtCore.Qt.RoundCap)
        pen.setJoinStyle(QtCore.Qt.RoundJoin)
        painter.setPen(pen)
        painter.drawPolyline(QtGui.QPolygonF([
            QtCore.QPointF(dx + x * w, dy + y * h) for x, y in points
        ]))
        painter.end()


class FluentPageBody(QtWidgets.QWidget):
    """A scrolled page body that paints its pane's surface and tells Qt so.

    The scrolled content used to be transparent, showing the same colour
    through from the page.  A scrolled widget Qt can see through is
    repainted whole on every step of the wheel -- an Edit page's plot
    surface, a 10 ms frame at three device pixels per point, forty times a
    scroll -- where one Qt knows to be opaque is moved by blitting the
    backing store and only the exposed strip is painted.  Qt learns opacity
    from ``WA_OpaquePaintEvent``, which also stops it painting any styled
    background, so this widget paints the colour itself.

    The colour is the SURFACE: a page body lives inside a tab pane, and a
    pane is white like every other page in it.  It painted the window's
    page grey once, and the one tab that scrolled was the one tab that was
    grey.
    """

    _REASSERT = (
        QtCore.QEvent.ParentChange,
        QtCore.QEvent.StyleChange,
        QtCore.QEvent.Polish,
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self._promise_opaque()

    def _promise_opaque(self) -> None:
        self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent, True)
        self.setAttribute(QtCore.Qt.WA_StyledBackground, False)

    def event(self, event) -> bool:
        # QScrollArea.setWidget and stylesheet polish both reset the
        # attribute; the promise outlives them.
        handled = super().event(event)
        if event.type() in self._REASSERT:
            self._promise_opaque()
        return handled

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API
        painter = QtGui.QPainter(self)
        painter.fillRect(event.rect(), QtGui.QColor(SURFACE))
        painter.end()


class FluentScrollArea(QtWidgets.QScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)
        #: Width-bounded content reflows to the viewport, so the viewport's
        #: width must not depend on whether the vertical bar happens to be
        #: showing.  It did: the row that first pushed the content past the
        #: viewport made the bar appear and narrowed EVERY control by the
        #: bar's width, in the same instant the operator added that row.
        self._bounded = False
        self._reserving = False
        self.verticalScrollBar().rangeChanged.connect(self._reserve_scrollbar_gutter)
        self.setWidgetResizable(True)
        self.setMinimumWidth(0)
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.setStyleSheet(
            f"""
            QScrollArea {{ background: transparent; border: none; }}
            QScrollArea > QWidget {{ background: transparent; }}
            QScrollArea > QWidget > QWidget {{ background: transparent; }}
            {fluent_scrollbar_stylesheet("QScrollBar")}
            """
        )
    def set_width_bounded_widget(self, widget: QtWidgets.QWidget) -> None:
        """Mount vertical content that must never widen its visible host.

        Logic lists and Edit pages are window-bounded surfaces: long status
        text, paths, signal names, or form size hints must reflow/elide inside
        the viewport while vertical overflow remains reachable.  Monitor boards
        intentionally do *not* use this mode because a single authored panel may
        be wider than the viewport and must then expose the Fluent horizontal
        scrollbar.
        """

        if not isinstance(widget, QtWidgets.QWidget):
            raise TypeError("bounded scroll content must be a QWidget")
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        widget.setMinimumWidth(0)
        widget.setSizePolicy(
            QtWidgets.QSizePolicy.Ignored,
            QtWidgets.QSizePolicy.Preferred,
        )
        self._bounded = True
        self.setWidget(widget)
        self._reserve_scrollbar_gutter()

    @QtCore.pyqtSlot()
    def _reserve_scrollbar_gutter(self) -> None:
        """Hold the vertical bar's width whether or not the bar is there.

        Asked as "is the range non-empty", not "is the bar visible": the
        widget's visibility is not settled yet at the moment its range
        changes, and this must be a function of the content alone.
        """

        if not self._bounded or self._reserving:
            return
        self._reserving = True
        try:
            bar = self.verticalScrollBar()
            needed = bar.maximum() > bar.minimum()
            reserved = 0 if needed else bar.sizeHint().width()
            margins = self.viewportMargins()
            if margins.right() != reserved:
                # The RIGHT gutter is this method's business; the bottom is
                # not.  Zeroing it here undid whatever a linked group had
                # padded the viewport with to keep its rows in step.
                self.setViewportMargins(0, 0, reserved, margins.bottom())
        finally:
            self._reserving = False

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._reserve_scrollbar_gutter()


class _RuledRowHeader(QtWidgets.QHeaderView):
    """A table's row header whose row rules are painted like the cells'.

    The rules continue the viewport's own, one edge-width at each row's
    bottom, and the rule that would lie on the frame's bottom edge is not
    drawn -- as a section border it doubled the edge and squared the
    rounded corner beside it."""

    def __init__(self, parent=None):
        super().__init__(QtCore.Qt.Vertical, parent)
        # What QTableView gives the header it makes for itself.
        self.setSectionsClickable(True)
        self.setHighlightSections(True)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().paintEvent(event)
        viewport = self.viewport()
        edge = frame_border_px(viewport)
        width = float(viewport.width())
        painter = QtGui.QPainter(viewport)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, False)
        divider = QtGui.QColor(DIVIDER)
        first = max(0, self.visualIndexAt(0))
        last = self.visualIndexAt(viewport.height() - 1)
        last = self.count() - 1 if last < 0 else last
        for visual in range(first, last + 1):
            section = self.logicalIndex(visual)
            if self.isSectionHidden(section):
                continue
            bottom = self.sectionViewportPosition(section) + self.sectionSize(section)
            if bottom >= viewport.height():
                continue
            y = device_snap(viewport, bottom, axis="y") - edge
            painter.fillRect(QtCore.QRectF(0.0, y, width, edge), divider)
        painter.end()


class _TableRules(QtWidgets.QWidget):
    """A table's two header rules, painted together over its headers.

    The column header's bottom rule and the row header's right rule cross
    at the corner; drawn as the header widgets' and the corner button's
    own style-sheet borders they were three strokes from three widgets,
    and inside a FluentFrame (whose ``QFrame`` rule a QHeaderView inherits)
    they ended in rounded hooks that left ground at the cross.  Here each
    is ONE filled rectangle placed from the table's frame: the header rule
    runs from the frame's left edge to the end of the viewport (into the
    right edge when nothing stands between), the row-header rule from the
    frame's top edge to the bottom of the viewport, both snapped to device
    pixels, so the cross and both joints with the frame are exact.

    The widget is transparent to the pointer and masked to the two rules,
    so it covers no cell and the viewport still scrolls by blitting.
    """

    def __init__(self, table: "FluentTableView"):
        super().__init__(table)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(QtCore.Qt.WA_NoSystemBackground, True)
        self.setFocusPolicy(QtCore.Qt.NoFocus)
        self._mask_key = None

    def rules(self) -> tuple[QtCore.QRectF, ...]:
        """The header rule and the row-header rule, in table coordinates."""

        table = self.parentWidget()
        inner = frame_inner_rect(table)
        edge = frame_border_px(table)
        viewport = table.viewport().geometry()
        # A rule that meets the frame runs INTO the frame's edge, which is
        # the same colour: from the edge's first whole device pixel to its
        # last, so at a fractional ratio the edge's partial pixel beside the
        # joint is filled rather than left as a light notch.  Where a scroll
        # bar stands between, the rule ends with the viewport.
        outer_left = device_snap(table, 0.0, "ceil")
        outer_top = device_snap(table, 0.0, "ceil", axis="y")
        outer_right = device_snap(table, table.width(), "floor")
        outer_bottom = device_snap(table, table.height(), "floor", axis="y")
        rules = []
        header = table.horizontalHeader()
        if not header.isHidden():
            y = device_snap(table, header.geometry().bottom() + 1, axis="y") - edge
            right = device_snap(table, viewport.right() + 1)
            right = outer_right if right >= inner.right() else right
            rules.append(QtCore.QRectF(outer_left, y, right - outer_left, edge))
        side = table.verticalHeader()
        if not side.isHidden():
            x = device_snap(table, side.geometry().right() + 1) - edge
            bottom = device_snap(table, viewport.bottom() + 1, axis="y")
            bottom = outer_bottom if bottom >= inner.bottom() else bottom
            rules.append(QtCore.QRectF(x, outer_top, edge, bottom - outer_top))
        return tuple(rules)

    def place(self) -> None:
        table = self.parentWidget()
        if self.geometry() != table.rect():
            self.setGeometry(table.rect())
        rules = self.rules()
        key = tuple((r.x(), r.y(), r.width(), r.height()) for r in rules)
        if key != self._mask_key:
            self._mask_key = key
            region = QtGui.QRegion()
            for rule in rules:
                # Whole logical pixels around the snapped rule.
                left, top = int(math.floor(rule.left())), int(math.floor(rule.top()))
                region += QtCore.QRect(
                    left, top,
                    max(1, int(math.ceil(rule.right())) - left),
                    max(1, int(math.ceil(rule.bottom())) - top),
                )
            if region.isEmpty():
                self.hide()
                return
            self.setMask(region)
        if self.isHidden():
            self.show()
        self.raise_()
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        del event
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, False)
        for rule in self.rules():
            painter.fillRect(rule, QtGui.QColor(DIVIDER))
        painter.end()


class FluentTableView(QtWidgets.QTableView):
    """Virtual item view with the same quiet Fluent chrome as the form family.

    The table deliberately owns no model or data interpretation.  A caller can
    mount any ``QAbstractItemModel`` and retain Qt's virtual, visible-cell-only
    painting instead of manufacturing one widget per value.
    """

    _content_rows_limit = 0
    _content_viewport_rows = 8
    _fitting_content_height = False
    _fits_values = False
    _fitted_width = 0
    _fitting_values = False
    _fit_pending = False
    #: The header rules' layer; None until __init__ makes it (Qt resizes
    #: the view while its base constructor runs).
    _rules = None
    #: Rows read per fit, from the top of the view: a fit is a sample of what
    #: is on screen, coalesced to one per turn, never a scan of the model.
    _FIT_SAMPLE_ROWS = 8

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setVerticalHeader(_RuledRowHeader(self))
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        # Rows, not boxes: one hairline under each row, no vertical rules.
        # No banding either -- every other Fluent surface is one white
        # ground separated by hairlines, and a table striped grey/white
        # belongs to a different toolkit.
        self.setShowGrid(False)
        self.setAlternatingRowColors(False)
        self.setWordWrap(False)
        self.setSortingEnabled(False)
        self.setCornerButtonEnabled(False)
        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        # A data table's unit is the ROW: the pointer and a selection light
        # the whole of it.  A single-cell hover band slid along under the
        # pointer and read as the cell being picked.
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self._hover_row = -1
        self.setMouseTracking(True)
        # Model dimensions never participate in page/window size hints.  A
        # million-cell scientific table is still one viewport with its own
        # scrollbars, not a million-row request to the surrounding layout.
        self.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustIgnored)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding,
        )
        self.setEditTriggers(
            QtWidgets.QAbstractItemView.DoubleClicked
            | QtWidgets.QAbstractItemView.EditKeyPressed
            | QtWidgets.QAbstractItemView.SelectedClicked
            | QtWidgets.QAbstractItemView.AnyKeyPressed
        )
        self.horizontalHeader().setSectionsMovable(False)
        self.horizontalHeader().setStretchLastSection(False)
        self.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.Interactive
        )
        self.horizontalHeader().setDefaultSectionSize(scaled_px(80, minimum=64))
        self.verticalHeader().setSectionsMovable(False)
        self.verticalHeader().setDefaultAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.verticalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Fixed)
        self.verticalHeader().setDefaultSectionSize(scaled_px(28, minimum=22))
        self.verticalHeader().setMinimumWidth(scaled_px(58, minimum=46))
        # A cell's text sits the item padding PLUS the style's own item text
        # margin in from its edge; a header section gets only its padding,
        # so the same padding put every heading outboard of its column.
        text_margin = self.style().pixelMetric(QtWidgets.QStyle.PM_FocusFrameHMargin, None, self) + 1
        header_pad = scaled_px(6, minimum=4) + text_margin
        # The header RULES are painted by _TableRules, one pass for both,
        # placed from the frame; the headers and the corner draw no border
        # and no radius (a FluentFrame's ``QFrame`` rule would otherwise
        # round a QHeaderView's ends).  A cell paints no background of its
        # own: the hover and selection washes are the view's (paintEvent).
        self.setStyleSheet(
            f"""
            QTableView {{
                background: white;
                color: {TEXT};
                border: {FIELD_BORDER_PX}px solid {DIVIDER};
                border-radius: {_radius()}px;
                selection-background-color: transparent;
                selection-color: {TEXT};
                font: {fluent_font_size()}pt "{FONT}";
                outline: none;
            }}
            QTableView::item {{
                padding: {scaled_px(3, minimum=2)}px {scaled_px(6, minimum=4)}px;
                border: none;
            }}
            QTableView::item:selected, QTableView::item:hover {{
                background: transparent;
                color: {TEXT};
                border: none;
            }}
            QTableView QLineEdit {{
                background: white;
                color: {TEXT};
                border: {scaled_px(1, minimum=1)}px solid {ACCENT};
                border-radius: {_radius()}px;
                padding: 0px {scaled_px(4, minimum=2)}px;
                font: {fluent_font_size()}pt "{FONT}";
                selection-background-color: {ACCENT_TINT};
                selection-color: {TEXT};
            }}
            QHeaderView {{ background: transparent; border: none; border-radius: 0px; }}
            QHeaderView::section {{
                background: transparent;
                color: {MUTED_INK};
                border: none;
                padding: {scaled_px(4, minimum=3)}px {header_pad}px;
                font: {fluent_font_size()}pt "{FONT}";
            }}
            QHeaderView::section:horizontal {{ font-weight: 600; }}
            QTableCornerButton::section {{ background: transparent; border: none; border-radius: 0px; }}
            """
        )
        apply_fluent_scrollbars(self)
        self._rules = _TableRules(self)
        for bar in (self.horizontalScrollBar(), self.verticalScrollBar()):
            bar.valueChanged.connect(self._schedule_value_fit)

    def setModel(self, model) -> None:  # noqa: N802 - Qt API name
        super().setModel(model)
        if model is None:
            return
        model.modelReset.connect(self._refit_value_columns)
        for name in ("dataChanged", "layoutChanged", "rowsInserted", "columnsInserted", "headerDataChanged"):
            getattr(model, name).connect(self._schedule_value_fit)
        self._refit_value_columns()

    def set_value_columns(self) -> None:
        """Size the columns to the widest value in view, and never elide one.

        A number cut to "2.6749…" hid the digits the operator opened the table
        to read.  Every column takes the width of the widest cell in the rows
        at the top of the view -- visible cells only, the way the table
        paints them -- and widens, never narrows, while the same data is
        shown, so scrolling does not make the columns breathe.  The tooltip
        of a cell is its whole value, for a column narrower than a value.
        """

        self._fits_values = True
        self.setTextElideMode(QtCore.Qt.ElideNone)
        self._refit_value_columns()

    def _refit_value_columns(self, *_args) -> None:
        self._fitted_width = 0
        self._schedule_value_fit()

    def _schedule_value_fit(self, *_args) -> None:
        if self._fits_values and not self._fit_pending:
            self._fit_pending = True
            QtCore.QTimer.singleShot(0, self._fit_value_columns)

    def _fit_value_columns(self) -> None:
        self._fit_pending = False
        model = self.model()
        if not self._fits_values or self._fitting_values or model is None:
            return
        rows, columns = model.rowCount(), model.columnCount()
        if not rows and not columns:
            return
        header, side = self.horizontalHeader(), self.verticalHeader()
        viewport = self.viewport()
        first_row = max(0, side.visualIndexAt(0))
        last_row = side.visualIndexAt(viewport.height() - 1)
        last_row = rows - 1 if last_row < 0 else last_row
        last_row = min(last_row, first_row + self._FIT_SAMPLE_ROWS - 1)
        first_column = max(0, header.visualIndexAt(0))
        last_column = header.visualIndexAt(viewport.width() - 1)
        last_column = columns - 1 if last_column < 0 else last_column
        metrics = self.fontMetrics()
        bold = QtGui.QFont(self.font())
        bold.setWeight(QtGui.QFont.DemiBold)
        heading = QtGui.QFontMetrics(bold)
        widest = 0
        for column in range(first_column, last_column + 1):
            label = model.headerData(column, QtCore.Qt.Horizontal, QtCore.Qt.DisplayRole)
            widest = max(widest, heading.horizontalAdvance(str("" if label is None else label)))
            for row in range(first_row, last_row + 1):
                text = model.index(row, column).data(QtCore.Qt.DisplayRole)
                widest = max(widest, metrics.horizontalAdvance(str("" if text is None else text)))
        wanted = widest + 2 * scaled_px(6, minimum=4) + scaled_px(10, minimum=6)
        if wanted <= max(self._fitted_width, header.minimumSectionSize()):
            return
        self._fitted_width = wanted
        self._fitting_values = True
        try:
            header.setDefaultSectionSize(wanted)
        finally:
            self._fitting_values = False

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API name
        super().resizeEvent(event)
        if self._rules is not None:
            self._rules.place()
        self._schedule_value_fit()

    def set_content_rows_limit(self, limit: int, *, viewport_rows: int = 8) -> None:
        """Lay a table of at most ``limit`` rows out whole; scroll a longer one itself.

        A longer table is a viewport at least ``viewport_rows`` tall that grows
        into whatever room its page has, with its own scrollbar -- so its
        column header stays in view.  Expanding every table of up to two
        thousand rows into the page scrolled the header away with the first
        screenful and gave the page and the table a scrollbar each.  Only
        section geometry is read, never scientific cells.
        """
        if limit < 1 or viewport_rows < 1:
            raise ValueError("row limits must be positive")
        self._content_rows_limit = int(limit)
        self._content_viewport_rows = int(viewport_rows)
        self._fit_content_height()

    def updateGeometries(self) -> None:  # noqa: N802
        super().updateGeometries()
        self._fit_content_height()
        if self._rules is not None:
            self._rules.place()

    def _fit_content_height(self) -> None:
        if not self._content_rows_limit or self._fitting_content_height or self.model() is None:
            return
        self._fitting_content_height = True
        try:
            rows = self.model().rowCount(self.rootIndex())
            expanded = rows <= self._content_rows_limit
            policy = QtCore.Qt.ScrollBarAlwaysOff if expanded else QtCore.Qt.ScrollBarAsNeeded
            if self.verticalScrollBarPolicy() != policy:
                self.setVerticalScrollBarPolicy(policy)
            vertical = self.verticalHeader()
            height = vertical.length() if expanded else sum(
                vertical.sectionSize(index)
                for index in range(min(rows, self._content_viewport_rows))
            )
            if not self.horizontalHeader().isHidden():
                height += self.horizontalHeader().height()
            bar = self.horizontalScrollBar()
            if bar.maximum() > bar.minimum():
                height += bar.sizeHint().height()
            height += 2 * self.frameWidth()
            ceiling = height if expanded else QtWidgets.QWIDGETSIZE_MAX
            if self.minimumHeight() != height or self.maximumHeight() != ceiling:
                self.setMinimumHeight(height)
                self.setMaximumHeight(ceiling)
                self.updateGeometry()
        finally:
            self._fitting_content_height = False

    def sizeHint(self) -> QtCore.QSize:  # noqa: N802 - Qt API name
        hint = super().sizeHint()
        if self._content_rows_limit:
            hint.setHeight(self.minimumHeight())
        return hint

    def _selected_runs(self, low_row: int, high_row: int) -> list[tuple[int, int, int, int]]:
        """``(first row, last row, left, right)`` of every run of selected
        rows between ``low_row`` and ``high_row``: neighbouring shown rows
        whose selected cells span the same columns.

        Only the rows asked about, so the cost follows the viewport, not
        the selection: the selection's RANGES are cut to them -- Select All
        on a camera frame is millions of indexes, and this runs on every
        hover."""

        selection = self.selectionModel()
        if selection is None or not selection.hasSelection():
            return []
        root = self.rootIndex()
        spans: dict[int, tuple[int, int]] = {}
        for block in selection.selection():
            if block.parent() != root:
                continue
            left = self.columnViewportPosition(block.left())
            right = self.columnViewportPosition(block.right()) + self.columnWidth(block.right())
            if right <= left:
                continue
            for row in range(max(block.top(), low_row), min(block.bottom(), high_row) + 1):
                if self.isRowHidden(row):
                    continue
                start, end = spans.get(row, (left, right))
                spans[row] = (min(start, left), max(end, right))
        runs: list[tuple[int, int, int, int]] = []
        previous = None
        for row in range(low_row, high_row + 1):
            if self.isRowHidden(row):
                continue
            span = spans.get(row)
            if span is not None and runs and previous == runs[-1][1] and span == runs[-1][2:]:
                runs[-1] = (runs[-1][0], row, *span)
            elif span is not None:
                runs.append((row, row, *span))
            previous = row
        return runs

    def _row_band(self, first: int, last: int, left: float, right: float) -> QtCore.QRectF:
        """The wash of rows ``first``..``last`` between ``left`` and
        ``right``: from the top of the first to the rule under the last (or
        to the frame, where no rule is drawn), a hair inside the columns'
        ends so its rounded corners stand clear of the frame and the header
        rule."""

        viewport = self.viewport()
        edge = frame_border_px(viewport)
        top = self.rowViewportPosition(first)
        bottom = self.rowViewportPosition(last) + self.rowHeight(last)
        y0 = device_snap(viewport, top, axis="y")
        if bottom >= viewport.height():
            y1 = device_snap(viewport, viewport.height(), axis="y")
        else:
            y1 = device_snap(viewport, bottom, axis="y") - edge
        inset = scaled_px(2, minimum=1)
        x0 = device_snap(viewport, left + inset)
        x1 = device_snap(viewport, right - inset)
        return QtCore.QRectF(x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0))

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """The washes, the cells over them, then the grid's own lines.

        The hovered row is an ACCENT_TINT band; a selected run of rows is ONE
        rounded SELECTED block -- the row under the pointer inside it in
        SELECTED_HOVER -- with no outline: a fill says "picked", and a box
        drawn round it was one more line.  A row's rule is one hairline
        across the WHOLE viewport, frame to frame -- drawn as a cell border
        it stopped where the last column did -- except inside a selected
        run, which it would cut into stripes; a block ends on the rule
        under its last row, so the rules either side stay whole.
        """

        model = self.model()
        if model is None:
            super().paintEvent(event)
            return
        viewport = self.viewport()
        header = self.verticalHeader()
        first = max(0, header.visualIndexAt(0))
        last = header.visualIndexAt(viewport.height() - 1)
        last = model.rowCount(self.rootIndex()) - 1 if last < 0 else last
        # One row past each side of the screen, to know whether a run goes on.
        low_row, high_row = max(0, first - 1), min(last + 1, model.rowCount(self.rootIndex()) - 1)
        runs = self._selected_runs(low_row, high_row)
        radius = float(_radius())
        hover = self._hover_row
        painter = QtGui.QPainter(viewport)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        painter.setPen(QtCore.Qt.NoPen)
        hover_in_run = False
        for top_row, bottom_row, left, right in runs:
            block = QtGui.QPainterPath()
            block.addRoundedRect(self._row_band(top_row, bottom_row, left, right), radius, radius)
            painter.fillPath(block, QtGui.QColor(SELECTED))
            if top_row <= hover <= bottom_row and not self.isRowHidden(hover):
                hover_in_run = True
                band = QtGui.QPainterPath()
                band.addRect(self._row_band(hover, hover, left, right))
                painter.fillPath(block.intersected(band), QtGui.QColor(SELECTED_HOVER))
        if 0 <= hover < model.rowCount(self.rootIndex()) and not hover_in_run and not self.isRowHidden(hover):
            columns = self.horizontalHeader()
            left = columns.sectionViewportPosition(columns.logicalIndex(0))
            wash = QtGui.QPainterPath()
            wash.addRoundedRect(self._row_band(hover, hover, left, left + columns.length()), radius, radius)
            painter.fillPath(wash, QtGui.QColor(ACCENT_TINT))
        painter.end()
        super().paintEvent(event)
        inside = {row for top_row, bottom_row, _left, _right in runs for row in range(top_row, bottom_row)}
        width = float(viewport.width())
        edge = frame_border_px(viewport)
        painter = QtGui.QPainter(viewport)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, False)
        divider = QtGui.QColor(DIVIDER)
        for visual in range(first, last + 1):
            row = header.logicalIndex(visual)
            if header.isSectionHidden(row) or row in inside:
                continue
            bottom = self.rowViewportPosition(row) + self.rowHeight(row)
            if bottom >= viewport.height():
                # A rule separates rows; on the frame's own bottom edge it
                # would only double it (a table fitted to its rows).
                continue
            y = device_snap(viewport, bottom, axis="y") - edge
            painter.fillRect(QtCore.QRectF(0.0, y, width, edge), divider)
        painter.end()

    def viewportEvent(self, event) -> bool:  # noqa: N802 - Qt naming
        kind = event.type()
        if kind in (QtCore.QEvent.HoverEnter, QtCore.QEvent.HoverMove, QtCore.QEvent.MouseMove):
            self._set_hover_row(self.rowAt(event.pos().y()))
        elif kind in (QtCore.QEvent.HoverLeave, QtCore.QEvent.Leave):
            self._set_hover_row(-1)
        elif kind == QtCore.QEvent.Resize and self._rules is not None:
            # A scroll bar that comes or goes moves the viewport's end.
            self._rules.place()
        return super().viewportEvent(event)

    def _set_hover_row(self, row: int) -> None:
        if row != self._hover_row:
            self._hover_row = row
            self.viewport().update()

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """One click opens the cell it lands on.

        A click on a number is a request to see it, and selection paint --
        any selection paint -- covers the very digits being asked about.
        The editor shows the exact stored value in a plain field, so the
        cheapest gesture answers the commonest question and the second
        click has nothing left to add.  Keyboard navigation is deliberately
        NOT an edit trigger: arrow keys must still walk the grid instead of
        being swallowed by an editor at every step.
        """

        super().mousePressEvent(event)
        if event.button() != QtCore.Qt.LeftButton:
            return
        index = self.indexAt(event.pos())
        if not index.isValid():
            return
        if not bool(index.flags() & QtCore.Qt.ItemIsEditable):
            return
        super().edit(index)

    def edit(self, index, trigger, event) -> bool:  # noqa: N802 - Qt naming
        """Open a cell, and watch the editor's keys while it is open.

        The editor's event filter belongs to the DELEGATE, not to the
        view, so a view that wants a say has to ask for one -- and this
        is the one place every editor is born, whichever trigger opened
        it and whichever delegate made it.
        """

        started = bool(super().edit(index, trigger, event))
        if started:
            editor = self.viewport().focusWidget()
            if editor is not None and editor is not self:
                editor.installEventFilter(self)
        return started

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt naming
        """Walking the grid while a cell is open is part of editing it.

        Qt gives the delegate Tab and Backtab -- commit, step sideways,
        keep editing -- and nothing else.  Up and Down appeared to work
        only because a QLineEdit ignores them and the unhandled key
        reaches the view underneath: an accident that lasts exactly as
        long as every editor happens to be a plain line edit, and one
        that no single-row table can even show.  The four directions are
        one contract, so all four are answered here.
        """

        if (
            event.type() == QtCore.QEvent.KeyPress
            and isinstance(watched, QtWidgets.QWidget)
            and event.key() in (QtCore.Qt.Key_Up, QtCore.Qt.Key_Down)
            and not event.modifiers()
        ):
            index = self.currentIndex()
            model = self.model()
            if index.isValid() and model is not None:
                step = -1 if event.key() == QtCore.Qt.Key_Up else 1
                row = index.row() + step
                if 0 <= row < model.rowCount():
                    moved = model.index(row, index.column())
                    self.commitData(watched)
                    self.closeEditor(
                        watched, QtWidgets.QAbstractItemDelegate.NoHint
                    )
                    self.setCurrentIndex(moved)
                    if bool(moved.flags() & QtCore.Qt.ItemIsEditable):
                        super().edit(moved)
                    return True
        return super().eventFilter(watched, event)

    def _selected_bounds(self) -> tuple[tuple[QtCore.QModelIndex, ...], int, int, int, int] | None:
        selected = tuple(self.selectionModel().selectedIndexes())
        if not selected:
            current = self.currentIndex()
            selected = (current,) if current.isValid() else ()
        if not selected:
            return None
        rows = tuple(index.row() for index in selected)
        columns = tuple(index.column() for index in selected)
        return selected, min(rows), max(rows), min(columns), max(columns)

    def copy_selection(self) -> None:
        bounds = self._selected_bounds()
        if bounds is None:
            return
        selected, first_row, last_row, first_column, last_column = bounds
        chosen = {(index.row(), index.column()): index for index in selected}
        lines = []
        for row in range(first_row, last_row + 1):
            fields = []
            for column in range(first_column, last_column + 1):
                index = chosen.get((row, column))
                value = "" if index is None else index.data(QtCore.Qt.DisplayRole)
                fields.append("" if value is None else str(value))
            lines.append("\t".join(fields))
        QtWidgets.QApplication.clipboard().setText("\n".join(lines))

    def paste_clipboard(self) -> None:
        model = self.model()
        if model is None:
            return
        start = self.currentIndex()
        if not start.isValid():
            bounds = self._selected_bounds()
            if bounds is None:
                return
            _selected, first_row, _last_row, first_column, _last_column = bounds
            start = model.index(first_row, first_column)
        text = QtWidgets.QApplication.clipboard().text()
        lines = text.splitlines()
        if not lines:
            lines = [""]
        cells = tuple(tuple(line.split("\t")) for line in lines)
        set_block = getattr(model, "set_block_data", None)
        if callable(set_block):
            set_block(start.row(), start.column(), cells)
            return
        for row_offset, fields in enumerate(cells):
            for column_offset, value in enumerate(fields):
                index = model.index(
                    start.row() + row_offset,
                    start.column() + column_offset,
                )
                if index.isValid():
                    model.setData(index, value, QtCore.Qt.EditRole)

    def clear_selection(self) -> None:
        model = self.model()
        if model is None:
            return
        indexes = tuple(self.selectionModel().selectedIndexes())
        clear = getattr(model, "clear_indices", None)
        if callable(clear):
            clear(tuple((index.row(), index.column()) for index in indexes))
            return
        for index in indexes:
            model.setData(index, "", QtCore.Qt.EditRole)

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt API name
        if event.matches(QtGui.QKeySequence.Copy):
            self.copy_selection()
            event.accept()
            return
        if event.matches(QtGui.QKeySequence.Paste):
            self.paste_clipboard()
            event.accept()
            return
        if event.key() in (QtCore.Qt.Key_Delete, QtCore.Qt.Key_Backspace):
            self.clear_selection()
            event.accept()
            return
        super().keyPressEvent(event)

    def contextMenuEvent(self, event) -> None:  # noqa: N802 - Qt API name
        menu = _FluentRoundedMenu(self)
        copy = menu.addAction("Copy")
        paste = menu.addAction("Paste")
        clear = menu.addAction("Clear")
        copy.triggered.connect(self.copy_selection)
        paste.triggered.connect(self.paste_clipboard)
        clear.triggered.connect(self.clear_selection)
        # One click, one menu: retired once it has been used, so a table that
        # is right-clicked for a session does not keep every menu it showed.
        try:
            menu.exec_(event.globalPos())
        finally:
            retire_widget(menu)
        event.accept()


#: Item-data roles of a selection grid beyond Qt's own.
_GRID_STATE_ROLE = QtCore.Qt.UserRole + 1
_GRID_NOTE_ROLE = QtCore.Qt.UserRole + 2
#: The value a trailing action cell reports; None on every other cell.
_GRID_ACTION_ROLE = QtCore.Qt.UserRole + 3
#: Cell states besides chosen/unchosen: reached by a rule rather than marked
#: (a cell may be chosen AND derived: a pair a rule chose); marked but not
#: honoured; wanted by nobody (an empty card in red dashes, so a name in a
#: cell still means it was chosen); held by another row (an empty dashed
#: outline that cannot be chosen here).
GRID_CELL_STATES = ("derived", "warning", "missing", "held")


def _mixed(first: str, second: str, weight: float) -> QtGui.QColor:
    """``weight`` of ``second`` over ``first``."""

    a, b = QtGui.QColor(first), QtGui.QColor(second)
    return QtGui.QColor(*(
        round(x + (y - x) * weight)
        for x, y in zip((a.red(), a.green(), a.blue()), (b.red(), b.green(), b.blue()))
    ))


class _SelectionGridModel(QtCore.QAbstractTableModel):
    """Read-only labels; only cells given a label can be selected.

    ``actions`` -- ``(value, glyph, tooltip)`` -- add, after a gap column,
    one trailing cell per action to every row.  Action cells are never
    selectable, so neither the selection nor the keyboard lands on them.
    """

    def __init__(self, parent=None, *, actions=(), label_tip: str = ""):
        super().__init__(parent)
        self._actions = tuple(actions)
        self._label_tip = str(label_tip)
        self._rows: tuple[str, ...] = ()
        self._columns: tuple[str, ...] = ()
        self._cells: dict[tuple[int, int], str] = {}
        self._states: dict[tuple[int, int], str] = {}
        self._tips: dict[tuple[int, int], str] = {}
        self._row_notes: tuple[str, ...] = ()
        self._row_tips: tuple[str, ...] = ()

    def reset(self, rows, columns, cells, *, row_notes=(), row_tips=()) -> None:
        self.beginResetModel()
        self._rows, self._columns, self._cells = tuple(rows), tuple(columns), dict(cells)
        self._states, self._tips = {}, {}
        self._row_notes, self._row_tips = tuple(row_notes), tuple(row_tips)
        self.endResetModel()

    def set_states(self, states, tips) -> None:
        if dict(states) == self._states and dict(tips) == self._tips:
            return
        self._states, self._tips = dict(states), dict(tips)
        if self._rows and self._columns:
            self.dataChanged.emit(
                self.index(0, 0), self.index(len(self._rows) - 1, len(self._columns) - 1),
                [_GRID_STATE_ROLE, QtCore.Qt.ToolTipRole],
            )

    def rowCount(self, parent=QtCore.QModelIndex()) -> int:  # noqa: N802 - Qt API
        return 0 if parent.isValid() else len(self._rows)

    def columnCount(self, parent=QtCore.QModelIndex()) -> int:  # noqa: N802 - Qt API
        if parent.isValid():
            return 0
        return len(self._columns) + (1 + len(self._actions) if self._actions else 0)

    def action_columns(self) -> range:
        """The gap before the actions, then one column per action."""

        start = len(self._columns)
        return range(start, start + 1 + len(self._actions)) if self._actions else range(0)

    def data(self, index, role=QtCore.Qt.DisplayRole):
        cell = (index.row(), index.column())
        action = index.column() - len(self._columns) - 1
        if 0 <= action < len(self._actions):
            value, glyph, tip = self._actions[action]
            return {QtCore.Qt.DisplayRole: glyph, QtCore.Qt.ToolTipRole: tip,
                    _GRID_ACTION_ROLE: value}.get(role)
        if role == QtCore.Qt.DisplayRole:
            return self._cells.get(cell)
        if role == QtCore.Qt.ToolTipRole:
            return self._tips.get(cell, self._cells.get(cell))
        if role == _GRID_STATE_ROLE:
            return self._states.get(cell)
        return None

    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):  # noqa: N802 - Qt API
        rows = orientation == QtCore.Qt.Vertical
        if role == QtCore.Qt.DisplayRole:
            labels = self._rows if rows else self._columns
            return labels[section] if 0 <= section < len(labels) else None
        if rows and role == _GRID_NOTE_ROLE and 0 <= section < len(self._row_notes):
            return self._row_notes[section]
        if rows and role == QtCore.Qt.ToolTipRole and 0 <= section < len(self._rows):
            tip = self._row_tips[section] if section < len(self._row_tips) else ""
            return "\n".join(part for part in (tip, self._label_tip) if part) or None
        return None

    def flags(self, index):
        cell = (index.row(), index.column())
        if cell in self._cells and self._states.get(cell) != "held":
            return QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable
        return QtCore.Qt.NoItemFlags


def _paint_grid_card(
    painter: QtGui.QPainter,
    surface: QtWidgets.QWidget,
    cell: QtCore.QRect,
    fill: str | None,
    edge: str | None = None,
) -> QtCore.QRectF:
    """One small card in a grid cell, and the rect its label goes in.

    Every card -- chosen, empty, derived, held -- has the SAME outer extent:
    the cell inset by the gutter, snapped to device pixels, rounded like
    every other face.  The face fills that extent.  An ``edge`` is drawn
    only where it says something, and always DASHED (a rule reached it,
    nobody has it, another row holds it): a whole-device-pixel stroke laid
    INSIDE the extent, concentric with it, so an edged card is exactly as
    large as its plain neighbours and the gutters stay even."""

    inset = scaled_px(2, minimum=1)
    # QRectF(QRect) is exclusive at the right and bottom already: the cell's
    # far edges are right() and bottom(), with no QRect "+ 1".
    raw = QtCore.QRectF(cell).adjusted(inset, inset, -inset, -inset)
    left = device_snap(surface, raw.left())
    top = device_snap(surface, raw.top(), axis="y")
    right = device_snap(surface, raw.right())
    bottom = device_snap(surface, raw.bottom(), axis="y")
    outer = QtCore.QRectF(left, top, right - left, bottom - top)
    radius = float(_radius())
    stroke = frame_border_px(surface)
    painter.setRenderHint(QtGui.QPainter.Antialiasing)
    if fill is not None:
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QColor(fill))
        painter.drawRoundedRect(outer, radius, radius)
    if edge is None:
        return outer
    pen = QtGui.QPen(QtGui.QColor(edge), stroke)
    # Flat caps, 3 on and 3 off: Qt's square cap grew every dash by a pen
    # width into the gap and the edge read as a solid line at 1x.
    pen.setCapStyle(QtCore.Qt.FlatCap)
    pen.setDashPattern([3.0, 3.0])
    painter.setPen(pen)
    painter.setBrush(QtCore.Qt.NoBrush)
    half = stroke / 2.0
    painter.drawRoundedRect(outer.adjusted(half, half, -half, -half), radius - half, radius - half)
    return outer


class _SelectionGridDelegate(QtWidgets.QStyledItemDelegate):
    """A cell is a small card: blank until chosen, named once it is.

    Every state is a FILL; no solid edge outlines a card.  An empty cell is
    a neutral BG card -- somewhere a choice can go; under the pointer it
    takes the ACCENT_TINT hover wash and names itself in MUTED_INK.  A
    chosen cell is SELECTED with its name in TEXT (SELECTED_HOVER under the
    pointer).  A DERIVED cell -- reached by a rule rather than marked, chosen
    or not -- is the lighter ACCENT_TINT wash with a dashed SELECTION_EDGE
    and its name in DERIVED_INK: the dashes say "a rule put this here".  A
    WARNING cell -- chosen but not honoured -- is RED_TINT with its name in
    RED_DARK, chosen or not, so the problem is never hidden under the
    selection; a MISSING cell is an empty card in red dashes and a HELD one
    an empty dashed outline (somebody else's frame).  A disabled grid shows
    every card on the BG face with the chosen names in DISABLED_INK, so
    what is chosen still reads but nothing invites a click.  Labels are
    centred on their ink (:func:`paint_centred_ink`).
    """

    def paint(self, painter, option, index) -> None:
        if index.data(_GRID_ACTION_ROLE) is not None:
            self._paint_action(painter, option, index)
            return
        state = index.data(_GRID_STATE_ROLE)
        held = state == "held"
        if not held and not index.flags() & QtCore.Qt.ItemIsSelectable:
            return
        grid = self.parent()
        live = grid.isEnabled()
        selected = bool(option.state & QtWidgets.QStyle.State_Selected)
        hovered = bool(option.state & QtWidgets.QStyle.State_MouseOver) and not held and live
        derived = state == "derived"
        warning = state == "warning"
        edge = None
        if held:
            fill, edge = None, DIVIDER
        elif not live:
            fill = BG
        elif warning:
            fill = _mixed(RED_TINT, RED, 0.19).name() if hovered else RED_TINT
        elif derived:
            fill, edge = (_mixed(ACCENT_TINT, SELECTED, 0.4).name() if hovered else ACCENT_TINT), SELECTION_EDGE
        elif selected:
            fill = SELECTED_HOVER if hovered else SELECTED
        else:
            fill = ACCENT_TINT if hovered else BG
            if state == "missing":
                edge = RED
        painter.save()
        rect = _paint_grid_card(painter, grid.viewport(), option.rect, fill, edge)
        # A name says the row has the cell; the empty states say nobody, or
        # somebody else, has it -- named only as the preview under the pointer.
        if selected or hovered or derived or warning:
            ink = (
                DISABLED_INK if not live
                else RED_DARK if warning else DERIVED_INK if derived
                else TEXT if selected else MUTED_INK
            )
            # The font the grid sizes its columns with (set_grid).
            font = QtGui.QFont(FONT)
            font.setPointSize(fluent_font_size())
            paint_centred_ink(painter, grid.viewport(), rect, font, str(index.data() or ""), ink)
        painter.restore()

    def _paint_action(self, painter, option, index) -> None:
        """The quiet destructive glyph every remove-row x is -- the same
        object as FluentButton's ghost ×: no face and no edge at rest, its
        glyph in MUTED_INK; under the pointer the RED_TINT face and RED_DARK
        ink, held down one shade deeper -- red at rest belongs to problems.
        As tall as the row's cells, so it lines up with them, and centred on
        its ink."""

        grid = self.parent()
        cell = (index.row(), index.column())
        live = grid.isEnabled()
        hovered = live and grid.hovered_action() == cell
        pressed = live and grid.pressed_action() == cell
        hover_face, pressed_face = _destructive_glyph_faces()
        painter.save()
        rect = _paint_grid_card(
            painter, grid.viewport(), option.rect,
            pressed_face if pressed else hover_face if hovered else None,
        )
        font = QtGui.QFont(FONT)
        font.setPointSize(round(fluent_font_size() * 1.35))
        paint_centred_ink(
            painter, grid.viewport(), rect, font, str(index.data()),
            DISABLED_INK if not live else RED_DARK if hovered or pressed else MUTED_INK,
        )
        painter.restore()


class _SelectionGridRowHeader(QtWidgets.QHeaderView):
    """Row labels, each with an optional muted line beneath it.

    ``activatable`` rows are things the grid's owner can act on (a file, say):
    their label lights up under the pointer and a click emits ``activated``
    instead of selecting the row's cells.  The labels follow the header's
    ``defaultAlignment`` (right unless the owner says left).
    """

    activated = QtCore.pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(QtCore.Qt.Vertical, parent)
        self.activatable = False
        self._hovered = -1
        self.setMouseTracking(True)

    def _fonts(self) -> tuple[QtGui.QFont, QtGui.QFont]:
        label = QtGui.QFont(FONT)
        label.setPointSize(fluent_font_size())
        note = QtGui.QFont(FONT)
        note.setPointSize(max(1, fluent_font_size() - 1))
        return label, note

    def _texts(self, section: int) -> tuple[str, str]:
        model = self.model()
        if model is None:
            return "", ""
        label = model.headerData(section, QtCore.Qt.Vertical, QtCore.Qt.DisplayRole)
        note = model.headerData(section, QtCore.Qt.Vertical, _GRID_NOTE_ROLE)
        return str(label or ""), str(note or "")

    def two_line_height(self) -> int:
        label, note = self._fonts()
        return (QtGui.QFontMetrics(label).height() + QtGui.QFontMetrics(note).height()
                + scaled_px(6, minimum=4))

    def _left(self) -> bool:
        return bool(self.defaultAlignment() & QtCore.Qt.AlignLeft)

    def sectionSizeFromContents(self, section: int) -> QtCore.QSize:  # noqa: N802 - Qt API
        label, note = self._texts(section)
        label_font, note_font = self._fonts()
        width = max(
            QtGui.QFontMetrics(label_font).horizontalAdvance(label),
            QtGui.QFontMetrics(note_font).horizontalAdvance(note),
        )
        pad = scaled_px(8, minimum=5)
        return QtCore.QSize(width + pad * (1 if self._left() else 2), self.defaultSectionSize())

    def paintSection(self, painter, rect, section: int) -> None:  # noqa: N802 - Qt API
        if not rect.isValid():
            return
        label, note = self._texts(section)
        label_font, note_font = self._fonts()
        hovered = section == self._hovered and self.isEnabled()
        # Left-aligned labels start on the grid's own edge, the edge the
        # content above and below it shares; right-aligned ones sit off it.
        pad = scaled_px(8, minimum=5)
        inner = rect.adjusted(0 if self._left() else pad, 0, -pad, 0)
        align = QtCore.Qt.AlignLeft if self._left() else QtCore.Qt.AlignRight
        painter.save()
        if self.activatable and hovered:
            # A label that acts is a row like any other under the pointer:
            # the hover wash behind it, its words still in TEXT.
            # As tall as the row's cards (the same gutter), the width of
            # the label column.
            inset = scaled_px(2, minimum=1)
            surface = self.viewport()
            area = QtCore.QRectF(rect)
            left = device_snap(surface, area.left())
            top = device_snap(surface, area.top() + inset, axis="y")
            band = QtCore.QRectF(
                left, top,
                device_snap(surface, area.right() - inset) - left,
                device_snap(surface, area.bottom() - inset, axis="y") - top,
            )
            painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(QtGui.QColor(ACCENT_TINT))
            painter.drawRoundedRect(band, _radius(), _radius())
        painter.setFont(label_font)
        painter.setPen(QtGui.QColor(
            DISABLED_INK if not self.isEnabled()
            else TEXT if self.activatable or hovered else MUTED_INK
        ))
        if not note:
            painter.drawText(inner, align | QtCore.Qt.AlignVCenter, label)
        else:
            label_height = QtGui.QFontMetrics(label_font).height()
            note_height = QtGui.QFontMetrics(note_font).height()
            top = inner.top() + (inner.height() - label_height - note_height) // 2
            painter.drawText(QtCore.QRect(inner.left(), top, inner.width(), label_height),
                             align | QtCore.Qt.AlignVCenter, label)
            painter.setFont(note_font)
            painter.setPen(QtGui.QColor(MUTED_INK))
            painter.drawText(QtCore.QRect(inner.left(), top + label_height, inner.width(), note_height),
                             align | QtCore.Qt.AlignVCenter, note)
        painter.restore()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt API
        hovered = self.logicalIndexAt(event.pos())
        if hovered != self._hovered:
            self._hovered = hovered
            self.setCursor(QtCore.Qt.PointingHandCursor if self.activatable and hovered >= 0
                           else QtCore.Qt.ArrowCursor)
            self.viewport().update()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802 - Qt API
        if self._hovered != -1:
            self._hovered = -1
            self.viewport().update()
        super().leaveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt API
        section = self.logicalIndexAt(event.pos())
        if self.activatable and event.button() == QtCore.Qt.LeftButton and section >= 0:
            self.activated.emit(section)
            event.accept()
            return
        super().mouseReleaseEvent(event)


class FluentSelectionGrid(QtWidgets.QTableView):
    """A grid of choices whose selection IS the state, not a table of values.

    Selection is the ordinary Qt table gesture -- a click picks one cell, a
    drag a rectangle, Ctrl adds or removes, a row or column header takes the
    whole line -- so nothing new is learned.  What differs from
    :class:`FluentTableView` is the reading: an unchosen cell is blank, a
    chosen one is a named card, cells without a label are no cells at all,
    and the grid is exactly as large as its cells.  ``selection_edited``
    carries the chosen ``(row, column)`` cells after an operator's change;
    :meth:`set_selected` and :meth:`set_grid` never emit it.

    ``toggle=True`` is the grid for marking many cells one by one: a click
    turns one cell on or off and a drag sweeps a run of them, headers select
    nothing, and ``selection_edited`` comes once the gesture is over.  Rows
    may carry a muted second line (:meth:`set_grid` ``row_notes``).
    ``row_actions`` -- ``(value, glyph, tooltip)`` -- end every row with one
    small glyph cell per action, after a narrow gap; ``label_action`` --
    ``(value, tooltip)`` -- makes each row label a link.  Both are reported
    as ``row_action(row, value)`` and never change the selection.
    :meth:`set_cell_states` marks cells a rule reached ("derived"), that
    cannot be honoured ("warning"), that nobody has ("missing") or that
    another row holds ("held", not choosable: the pointer says so, and a
    click shows the cell's tooltip at once).  :meth:`set_label_width` ends
    the row-label column where a form's label column ends, so the first
    card starts on the form's field edge.
    """

    selection_edited = QtCore.pyqtSignal(object)
    row_action = QtCore.pyqtSignal(int, object)

    def __init__(self, parent=None, *, cell_width: int = 52, cell_height: int = 34,
                 toggle: bool = False, row_actions: Sequence[tuple[object, str, str]] = (),
                 label_action: tuple[object, str] | None = None):
        super().__init__(parent)
        self._cell_width = cell_width
        self._cell_height = cell_height
        self._toggle = bool(toggle)
        self._grid_model = _SelectionGridModel(
            self, actions=tuple((value, str(glyph), str(tip)) for value, glyph, tip in row_actions),
            label_tip="" if label_action is None else str(label_action[1]),
        )
        #: The action cell under the pointer, and the one a press began on.
        self._hovered_action: tuple[int, int] | None = None
        self._pressed_action: tuple[int, int] | None = None
        self._cursor = QtCore.Qt.ArrowCursor
        row_header = _SelectionGridRowHeader(self)
        row_header.activatable = label_action is not None
        if label_action is not None:
            row_header.activated.connect(lambda row: self.row_action.emit(row, label_action[0]))
        self.setVerticalHeader(row_header)
        self.setModel(self._grid_model)
        self.setItemDelegate(_SelectionGridDelegate(self))
        self.setShowGrid(False)
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.setCornerButtonEnabled(False)
        self.setWordWrap(False)
        self.setMouseTracking(True)
        self.setSelectionMode(
            QtWidgets.QAbstractItemView.MultiSelection if self._toggle
            else QtWidgets.QAbstractItemView.ExtendedSelection
        )
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectItems)
        self.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        for header, size in ((self.horizontalHeader(), cell_width), (row_header, cell_height)):
            header.setSectionResizeMode(QtWidgets.QHeaderView.Fixed)
            # Qt's own floor is measured from the header font and padding --
            # wider than a gap column, and than a short cell, ever asks for.
            header.setMinimumSectionSize(scaled_px(4, minimum=2))
            header.setDefaultSectionSize(scaled_px(size))
            header.setHighlightSections(False)
            header.setSectionsMovable(False)
            if self._toggle or header is row_header and label_action is not None:
                header.setSectionsClickable(False)
        row_header.setDefaultAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.setStyleSheet(
            f"""
            QTableView {{
                background: transparent;
                border: none;
                outline: none;
                font: {fluent_font_size()}pt "{FONT}";
            }}
            QHeaderView {{ background: transparent; }}
            QTableCornerButton::section {{ background: transparent; border: none; }}
            QHeaderView::section {{
                background: transparent;
                color: {MUTED_INK};
                border: none;
                padding: 0px {scaled_px(6, minimum=4)}px;
                font: {fluent_font_size()}pt "{FONT}";
            }}
            QHeaderView::section:hover {{ color: {TEXT}; }}
            """
        )
        self._programmatic = False
        #: The selection when a toggle gesture began; it is reported once, at its end.
        self._gesture_start: tuple[tuple[int, int], ...] | None = None
        self.selectionModel().selectionChanged.connect(self._selection_changed)

    def set_grid(self, rows: Sequence[str], columns: Sequence[str],
                 cells: "dict[tuple[int, int], str]", *,
                 row_notes: Sequence[str] = (), row_tips: Sequence[str] = (),
                 narrow_columns: Sequence[int] = ()) -> None:
        """Header labels, and the label of every selectable ``(row, column)``.

        ``row_notes`` puts a muted second line under row labels;
        ``narrow_columns`` are gaps between column groups, holding no cells.
        """

        self._programmatic = True
        try:
            self._grid_model.reset(rows, columns, cells, row_notes=row_notes, row_tips=row_tips)
        finally:
            self._programmatic = False
        # Wide enough for the longest label: a frame may be named, not numbered.
        font = QtGui.QFont(FONT)
        font.setPointSize(fluent_font_size())
        metrics = QtGui.QFontMetrics(font)
        widest = max((metrics.horizontalAdvance(str(label)) for label in (*cells.values(), *columns)),
                     default=0)
        horizontal = self.horizontalHeader()
        horizontal.setDefaultSectionSize(
            max(scaled_px(self._cell_width), widest + 2 * scaled_px(10, minimum=6)))
        gap = scaled_px(14, minimum=8)
        for column in narrow_columns:
            horizontal.resizeSection(column, gap)
        trailing = self._grid_model.action_columns()
        for column in trailing:
            horizontal.resizeSection(column, gap if column == trailing.start else scaled_px(36))
        rows_header = self.verticalHeader()
        rows_header.setDefaultSectionSize(max(
            scaled_px(self._cell_height),
            rows_header.two_line_height() if any(row_notes) else 0,
        ))
        self.updateGeometry()

    def label_width_hint(self) -> int:
        """The form label column at which every row label still fits."""

        header = self.verticalHeader()
        widest = max((header.sectionSizeFromContents(section).width()
                      for section in range(header.count())), default=0)
        return max(0, widest - self._label_offset())

    def set_label_width(self, width: int) -> None:
        """End the row-label column where a form's ``width`` label column
        ends, so the first card starts on the form's field edge."""

        self.verticalHeader().setMinimumWidth(int(width) + self._label_offset())
        self.updateGeometries()
        self.updateGeometry()

    @staticmethod
    def _label_offset() -> int:
        # A setting row's label-to-field spacing, less the inset that keeps
        # a card off its cell's edge.
        return scaled_px(6, minimum=4) - scaled_px(2, minimum=1)

    def set_cell_states(self, states: "dict[tuple[int, int], str]",
                        tips: "dict[tuple[int, int], str] | None" = None) -> None:
        """Mark cells with :data:`GRID_CELL_STATES`, with optional tooltips."""

        unknown = set(states.values()) - set(GRID_CELL_STATES)
        if unknown:
            raise ValueError(f"unknown grid cell states {sorted(unknown)!r}")
        self._grid_model.set_states(states, tips or {})

    def set_selected(self, cells) -> None:
        chosen = QtCore.QItemSelection()
        for row, column in cells:
            index = self._grid_model.index(row, column)
            if index.isValid() and index.flags() & QtCore.Qt.ItemIsSelectable:
                chosen.select(index, index)
        self._programmatic = True
        try:
            self.selectionModel().select(chosen, QtCore.QItemSelectionModel.ClearAndSelect)
        finally:
            self._programmatic = False

    def selected_cells(self) -> tuple[tuple[int, int], ...]:
        return tuple(sorted((index.row(), index.column()) for index in self.selectedIndexes()))

    def _selection_changed(self, *_args) -> None:
        if self._programmatic or self._gesture_start is not None:
            return
        self.selection_edited.emit(self.selected_cells())

    def hovered_action(self) -> tuple[int, int] | None:
        return self._hovered_action

    def pressed_action(self) -> tuple[int, int] | None:
        return self._pressed_action

    def _action_cell(self, pos) -> tuple[int, int] | None:
        index = self.indexAt(pos)
        if not index.isValid() or index.data(_GRID_ACTION_ROLE) is None:
            return None
        return index.row(), index.column()

    def _held(self, pos) -> QtCore.QModelIndex | None:
        index = self.indexAt(pos)
        return index if index.isValid() and index.data(_GRID_STATE_ROLE) == "held" else None

    def _set_cursor(self, cursor) -> None:
        if cursor != self._cursor:
            self._cursor = cursor
            self.viewport().setCursor(cursor)

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt API
        # An action cell is a button: it acts on release, and neither starts
        # a selection gesture nor ends one.
        cell = self._action_cell(event.pos())
        if cell is not None:
            if event.button() == QtCore.Qt.LeftButton:
                self._pressed_action = cell
                self.viewport().update()
            event.accept()
            return
        # A held cell cannot be chosen here; the click says why at once
        # instead of being swallowed.
        held = self._held(event.pos())
        if held is not None:
            QtWidgets.QToolTip.showText(event.globalPos(), str(held.data(QtCore.Qt.ToolTipRole) or ""),
                                        self.viewport(), self.visualRect(held))
            event.accept()
            return
        if self._toggle and event.button() == QtCore.Qt.LeftButton:
            self._gesture_start = self.selected_cells()
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802 - Qt API
        # The second click of a double click must not act again: the row
        # under it may already be another one.
        if self._action_cell(event.pos()) is not None:
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt API
        cell = self._action_cell(event.pos())
        self._set_cursor(QtCore.Qt.PointingHandCursor if cell is not None
                         else QtCore.Qt.ForbiddenCursor if self._held(event.pos()) is not None
                         else QtCore.Qt.ArrowCursor)
        if cell != self._hovered_action:
            self._hovered_action = cell
            self.viewport().update()
        if self._pressed_action is None:
            super().mouseMoveEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802 - Qt API
        self._set_cursor(QtCore.Qt.ArrowCursor)
        if self._hovered_action is not None:
            self._hovered_action = None
            self.viewport().update()
        super().leaveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt API
        pressed, self._pressed_action = self._pressed_action, None
        if pressed is not None:
            self.viewport().update()
            if event.button() == QtCore.Qt.LeftButton and self._action_cell(event.pos()) == pressed:
                self.row_action.emit(pressed[0], self._grid_model.index(*pressed).data(_GRID_ACTION_ROLE))
            event.accept()
            return
        super().mouseReleaseEvent(event)
        started, self._gesture_start = self._gesture_start, None
        if started is not None and self.selected_cells() != started:
            self.selection_edited.emit(self.selected_cells())

    def sizeHint(self) -> QtCore.QSize:  # noqa: N802 - Qt API
        horizontal, vertical = self.horizontalHeader(), self.verticalHeader()
        return QtCore.QSize(
            max(vertical.minimumWidth(), vertical.sizeHint().width())
            + horizontal.length() + 2 * self.frameWidth(),
            horizontal.sizeHint().height() + vertical.length() + 2 * self.frameWidth(),
        )

    def minimumSizeHint(self) -> QtCore.QSize:  # noqa: N802 - Qt API
        return self.sizeHint()


class FluentSegmentedButtons(QtWidgets.QWidget):
    """One of a closed set of named choices: ONE control, the current marked.

    For a set the OWNER declares closed -- an enumeration, the modes of a
    setting, a binding's Default / API / Config -- however many it holds, as
    long as the labels fit; a choice among data (frame labels, an axis's
    events) or a set that grows or is filtered is a :class:`FluentComboBox`.

    A DIVIDER track with no edge holds the choices flush, each a
    transparent face in TEXT ink.  The pointer lays the ACCENT_TINT hover
    wash on a choice and a press the SELECTED fill; the current one IS the
    SELECTED fill -- SELECTED_HOVER under the pointer -- in the same weight
    as the others, and no outline: the fill alone says which is chosen.
    Every face is rounded at the field radius and the track at that radius
    plus its rim, so the current face is concentric with the track rather
    than a near-square block in a rounded one.  A row of separate tinted buttons read as a row of
    commands, which these are not.

    A choice the owner WITHHOLDS (``button(value).setEnabled(False)``) keeps
    its place in DISABLED_INK and takes neither the pointer nor Tab.  The
    whole control disabled is a BG track whose current face is the DIVIDER
    grey, every label DISABLED_INK: what is chosen still reads, nothing
    invites a click.  A choice is reached by Tab, never focused by a click,
    and focus paints nothing on it: a click makes it current, and a focus
    face left behind would be one more state on the track.

    ``activated`` carries the value an operator clicked -- the owner decides
    and marks it; :meth:`set_current` marks a value (or none) without
    emitting it, and :meth:`current` says which is marked.
    """

    activated = QtCore.pyqtSignal(object)

    def __init__(self, choices: Sequence[tuple[object, str]], parent=None):
        super().__init__(parent)
        layout = QtWidgets.QHBoxLayout(self)
        rim = scaled_px(2, minimum=2)
        layout.setContentsMargins(rim, rim, rim, rim)
        layout.setSpacing(0)
        self._buttons: dict[object, QtWidgets.QPushButton] = {}
        pad = scaled_px(10, minimum=6)
        metrics = QtGui.QFontMetrics(QtGui.QFont(FONT, fluent_font_size()))
        for value, label in choices:
            button = QtWidgets.QPushButton(label, self)
            button.setCursor(QtCore.Qt.PointingHandCursor)
            button.setFocusPolicy(QtCore.Qt.TabFocus)
            button.setMinimumHeight(control_height() - 2 * rim)
            button.setMinimumWidth(fluent_text_width(metrics, str(label)) + 2 * pad + 2)
            button.setProperty("current", False)
            button.clicked.connect(lambda _checked=False, value=value: self.activated.emit(value))
            button.installEventFilter(self)
            layout.addWidget(button)
            self._buttons[value] = button
        self.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self._rim = rim
        # Every face's edge is its own fill (kept one edge-width wide so a
        # state never moves a label): no state is told by a line.
        self.setStyleSheet(
            f"""
            QPushButton {{
                background: transparent;
                color: {TEXT};
                border: {FIELD_BORDER_PX}px solid transparent;
                border-radius: {_radius()}px;
                padding: 0px {pad - FIELD_BORDER_PX}px;
                font: {fluent_font_size()}pt "{FONT}";
                outline: none;
            }}
            QPushButton:enabled:hover {{ background: {ACCENT_TINT}; border-color: {ACCENT_TINT}; }}
            QPushButton:enabled:pressed {{ background: {SELECTED}; border-color: {SELECTED}; }}
            QPushButton[current="true"] {{
                background: {SELECTED}; border-color: {SELECTED};
            }}
            QPushButton[current="true"]:enabled:hover, QPushButton[current="true"]:enabled:pressed {{
                background: {SELECTED_HOVER}; border-color: {SELECTED_HOVER};
            }}
            QPushButton:disabled {{
                background: transparent; border-color: transparent; color: {DISABLED_INK};
            }}
            QPushButton[current="true"]:disabled {{ background: {DIVIDER}; border-color: {DIVIDER}; }}
            """
        )

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """The track: one neutral rounded face the choices sit in, its
        corners concentric with theirs."""

        del event
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QColor(DIVIDER if self.isEnabled() else BG))
        radius = float(_radius() + self._rim)
        painter.drawRoundedRect(QtCore.QRectF(self.rect()), radius, radius)
        painter.end()

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt naming
        """Paint a choice: its face by the sheet, its label on the line
        every text of the row sits on (:func:`centred_line_top` of the
        control), not the style's centred line box."""

        if event.type() != QtCore.QEvent.Paint or watched not in self._buttons.values():
            return super().eventFilter(watched, event)
        option = QtWidgets.QStyleOptionButton()
        watched.initStyleOption(option)
        painter = QtWidgets.QStylePainter(watched)
        painter.drawControl(QtWidgets.QStyle.CE_PushButtonBevel, option)
        style = watched.style()
        area = style.subElementRect(QtWidgets.QStyle.SE_PushButtonContents, option, watched)
        # The control's middle, not the face's: the track's rim is even
        # above and below, so they are one.
        offset = watched.y()
        area.setTop(centred_line_top(-offset, self.height(), watched.font()))
        style.drawItemText(painter, area, int(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop),
                           option.palette, watched.isEnabled(), option.text, QtGui.QPalette.ButtonText)
        painter.end()
        return True

    def button(self, value) -> QtWidgets.QPushButton:
        return self._buttons[value]

    def current(self):
        """The marked value, or None when none is."""

        return next((value for value, button in self._buttons.items() if button.property("current")), None)

    def set_current(self, value) -> None:
        for choice, button in self._buttons.items():
            current = choice == value
            if button.property("current") != current:
                button.setProperty("current", current)
                button.style().unpolish(button)
                button.style().polish(button)


class LinkedScrollPanes(QtWidgets.QWidget):
    """Side-by-side scroll areas that scroll vertically AS ONE.

    Columns whose rows line up have to move together, and the way to make that
    true is not to synchronize two scrollbars.  Two bars means "is there
    anything below the fold" gets answered twice -- and the two answers
    disagree the moment one column is taller than the other, so whichever bar
    was picked to be the visible one silently decides how far the operator can
    reach.  A column with more rows than its neighbour then has names below the
    fold that no gesture can bring into view.

    Here the panes have no vertical bar of their own.  This widget owns the one
    bar, takes its range from whichever pane currently overflows the most, and
    hides it when none of them do -- so a pane that fits shows no bar and a
    pane that does not is reachable to its last row, with nobody hand-computing
    a height.  Each pane keeps its own horizontal behaviour: a wide timeline
    still gets its own horizontal scrollbar.
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        row = QtWidgets.QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        self._row = row
        self._panes: list[QtWidgets.QAbstractScrollArea] = []
        self._moving = False
        self._equalising = False
        #: What each pane's viewport was shortened by to match the group's,
        #: parallel to ``_panes``.
        self._padding: list[int] = []
        self._bar = QtWidgets.QScrollBar(QtCore.Qt.Vertical, self)
        self._bar.setStyleSheet(fluent_scrollbar_stylesheet("QScrollBar"))
        self._bar.setFixedWidth(fluent_scrollbar_thickness())
        self._bar.valueChanged.connect(self._drive_panes)
        self._bar.hide()
        row.addWidget(self._bar)

    # -- construction ----------------------------------------------------
    def add_pane(self, pane: QtWidgets.QAbstractScrollArea, *, stretch: int = 0) -> None:
        """Adopt one scroll area into the linked group, left to right."""

        if not isinstance(pane, QtWidgets.QAbstractScrollArea):
            raise TypeError("a linked pane must be a scroll area")
        pane.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        bar = pane.verticalScrollBar()
        bar.rangeChanged.connect(self._reconcile_bar)
        bar.valueChanged.connect(self._follow_pane)
        # A wheel anywhere in the group drives the GROUP.  Qt scrolls a pane on
        # the wheel only when that pane's own bar has range, so a pane with
        # nothing of its own to scroll swallowed the gesture -- an empty
        # timeline beside a column of twenty-two channels could not be scrolled
        # by pointing at the timeline, which is most of the surface.
        pane.viewport().installEventFilter(self)
        self._panes.append(pane)
        self._padding.append(0)
        # Keep the shared bar last so it sits against the group's right edge.
        self._row.insertWidget(self._row.count() - 1, pane, stretch)
        self._reconcile_bar()

    @property
    def scrollbar(self) -> QtWidgets.QScrollBar:
        """The single shared vertical bar (visible only while it can move)."""

        return self._bar

    # -- the one answer to "does this overflow?" -------------------------
    def _deepest(self) -> QtWidgets.QScrollBar | None:
        """The pane bar with the most room below the fold, if any pane has room."""

        bars = [pane.verticalScrollBar() for pane in self._panes]
        return max(bars, key=lambda bar: bar.maximum(), default=None)

    def _equalise_reach(self) -> None:
        """Give every pane the same VIEWPORT, so one value means one row.

        A pane's maximum is its content height less its viewport height, and
        the viewports in a group are not the same: a wide timeline gives a
        strip up to its own horizontal scrollbar and the name column beside it
        does not.  Driving every pane to one value then parts them exactly at
        the bottom -- Qt clamps the shallower pane at ITS maximum, so the rows
        line up the whole way down and step out of line on the last screenful.

        Equalising the viewports removes the difference at its source, and it
        is the only quantity here it is safe to equalise.  ``maximum`` is
        CLAMPED AT ZERO, so a pane that already fits cannot report how much it
        fits by; recovering its natural reach by subtracting back the padding
        we gave it is therefore only true of a pane that overflows, and for one
        that does not the clamp hides the padding and the correction counts it
        again every round -- 8, 45, 111, 206, ... until the viewport is gone
        and the column is blank.  Nor is a shorter viewport reach: shrinking
        the window onto a short pane does raise its maximum, but it does so by
        hiding the rows it was supposed to keep in step.

        So the viewports are levelled DOWN to the shallowest, which every pane
        can give, and the padding is bounded by the difference it was measured
        from.  A pane whose content ends sooner than its neighbour's still
        stops sooner; that is a fact about its content, not something a margin
        can or should fix.
        """

        if self._equalising or len(self._panes) < 2:
            return
        natural = [
            pane.viewport().height() + padding
            for pane, padding in zip(self._panes, self._padding)
        ]
        # Nothing to level before the group has been laid out: a viewport with
        # no height yet is not a measurement of anything.
        if any(height <= 0 for height in natural):
            return
        target = min(natural)
        self._equalising = True
        try:
            for index, (pane, height) in enumerate(zip(self._panes, natural)):
                wanted = height - target
                if self._padding[index] == wanted:
                    continue
                self._padding[index] = wanted
                margins = pane.viewportMargins()
                pane.setViewportMargins(
                    margins.left(), margins.top(), margins.right(), wanted
                )
        finally:
            self._equalising = False

    def _reconcile_bar(self, *_range: int) -> None:
        self._equalise_reach()
        deepest = self._deepest()
        maximum = deepest.maximum() if deepest is not None else 0
        self._bar.setRange(0, maximum)
        if deepest is not None:
            self._bar.setPageStep(deepest.pageStep())
            self._bar.setSingleStep(deepest.singleStep())
        self._bar.setVisible(maximum > 0)

    # -- keeping the group together --------------------------------------
    def _drive_panes(self, value: int) -> None:
        if self._moving:
            return
        self._moving = True
        try:
            for pane in self._panes:
                pane.verticalScrollBar().setValue(value)
        finally:
            self._moving = False

    def _follow_pane(self, value: int) -> None:
        """A pane moved on its own (a wheel, a keypress); bring the group along."""

        if self._moving:
            return
        self._moving = True
        try:
            self._bar.setValue(value)
            for pane in self._panes:
                pane.verticalScrollBar().setValue(value)
        finally:
            self._moving = False

    def eventFilter(self, watched, event):  # noqa: N802 - Qt name
        """Turn a wheel over any pane into a move of the shared bar."""

        if event.type() == QtCore.QEvent.Wheel and self._bar.maximum() > 0:
            steps = event.angleDelta().y() / 120.0
            if steps:
                self._bar.setValue(
                    self._bar.value() - int(round(steps * self._bar.singleStep() * 3))
                )
                return True
        return super().eventFilter(watched, event)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt name
        super().resizeEvent(event)
        # A shorter viewport can turn content that fitted into content that
        # overflows; the bar's own range is what decides, so re-ask it.
        self._reconcile_bar()


def apply_fluent_scrollbars(widget: "QtWidgets.QAbstractScrollArea") -> None:
    """Give any scroll-area-based widget (QPlainTextEdit, QTextEdit, QTableView, ...) the
    SAME Fluent scrollbar look that FluentScrollArea uses, sourced from the one shared
    ``fluent_scrollbar_stylesheet``.  Appends to (never clobbers) the widget's existing
    stylesheet, so its own background/border/padding rules are preserved.  Use this when
    the widget cannot be a FluentScrollArea (e.g. a QPlainTextEdit, which IS already a
    scroll area and must not be double-nested)."""

    existing = widget.styleSheet()
    widget.setStyleSheet(
        (existing + "\n" if existing else "")
        + fluent_scrollbar_stylesheet("QScrollBar", frame_width=widget.frameWidth())
    )


class FluentWindow(FramelessWindow):
    """Frameless wrapper for PyQt frontends."""

    hidden = QtCore.pyqtSignal()   # fires on close OR hide/minimize (event-loop quit)
    closed = QtCore.pyqtSignal()   # notification AFTER a committed close; never a close command
    close_requested = QtCore.pyqtSignal()  # fires on the X/close (before hide-or-destroy), NOT on minimize

    def __init__(
        self,
        *,
        widget: QtWidgets.QWidget,
        title: str = "",
        hide_on_close: bool = False,
    ):
        super().__init__()
        self._hide_on_close = bool(hide_on_close)
        self._zlc_close_guard = None
        self._zlc_hide_guard = None
        self._zlc_close_committed = False
        self.setObjectName("FluentWindow")
        self.setStyleSheet(f"QWidget#FluentWindow {{ background: {BG}; }}")

        title_bar = StandardTitleBar(self)
        title_bar.iconLabel.setFixedSize(0, 0)
        title_bar.titleLabel.hide()
        self.setTitleBar(title_bar)
        self._zlc_title_bar = title_bar
        self.setWindowTitle(title)
        # The product mark, left of the words: a window whose title bar wears a
        # different face from its taskbar button reads as two programs.
        self._zlc_icon = QtWidgets.QLabel(title_bar)
        self._zlc_icon.setObjectName("zlcWindowIcon")
        self._zlc_icon.setStyleSheet(
            "QLabel#zlcWindowIcon { background: transparent; }"
        )
        self._zlc_icon.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
        self._zlc_icon.setAlignment(QtCore.Qt.AlignCenter)
        self._zlc_icon.show()
        self._zlc_title = QtWidgets.QLabel(title, title_bar)
        self._zlc_title.setObjectName("zlcWindowTitle")
        self._zlc_title.setStyleSheet(
            f'QLabel#zlcWindowTitle {{ background: transparent; color: {TEXT};'
            f' font: {fluent_font_size()}pt "{FONT}"; }}')
        self._zlc_title.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
        self._zlc_title.setAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft)
        self._zlc_title.show()
        self.windowTitleChanged.connect(self._zlc_title.setText)
        top_margin = frameless_content_top_margin(title_bar)

        if widget is None:
            raise ValueError("FluentWindow needs a constructed widget.")
        self.loaded = widget
        self.loaded.setParent(self)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, top_margin, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.loaded)
        self.titleBar.raise_()
        self.loaded.adjustSize()
        self.resize(max(scaled_px(900, minimum=680), self.loaded.width()), self.loaded.height() + top_margin)

    def _sync_frameless_content_top_margin(self) -> None:
        """Reapply the native title-bar boundary after Qt resolves chrome."""

        layout = self.layout()
        if layout is None:
            return
        margins = layout.contentsMargins()
        top_margin = frameless_content_top_margin(self.titleBar)
        if margins.top() == top_margin:
            return
        layout.setContentsMargins(
            margins.left(), top_margin, margins.right(), margins.bottom()
        )
        self.loaded.adjustSize()

    def _position_window_title(self) -> None:
        """Keep FluentWindow's own title label pinned to the body content column.

        The label spans from ``TITLE_LEFT_INSET`` to just left of the first window
        control button, vertically centred in the 32 px bar.  Being a free child of
        the title bar (NOT in its layout), it is immune to title-set churn and to
        title-bar layout changes -- the title therefore sits on the SAME left
        column as the body content on every GUI and DPR."""
        lab = self._zlc_title
        tb = self._zlc_title_bar
        x = scaled_px(TITLE_LEFT_INSET)
        right = tb.width()
        for child in tb.findChildren(QtWidgets.QAbstractButton):
            if child.isVisible() and child.width() > 0:
                right = min(right, child.x())

        # The MARK takes the content column and the words follow it, so the
        # title block still begins where the body below it does.
        from ..qt import app_icon  # noqa: PLC0415 -- qt.py reads this module

        side = max(1, int(round(tb.height() * TITLE_ICON_RATIO)))
        # ASK IN LOGICAL PIXELS.  ensure_qt_app turns on AA_UseHighDpiPixmaps,
        # so QIcon hands back a pixmap already scaled for the screen and
        # already carrying its device ratio; scaling the request as well made
        # it twice the label, and a QLabel does not shrink a pixmap -- it clips
        # it, so the title bar showed the middle of the mark with its corners
        # cropped away.
        pixmap = app_icon().pixmap(side, side)
        if not pixmap.isNull():
            self._zlc_icon.setPixmap(pixmap)
        self._zlc_icon.setGeometry(
            x, max(0, (tb.height() - side) // 2), side, side
        )
        self._zlc_icon.setVisible(not pixmap.isNull())
        self._zlc_icon.raise_()

        words = x + (side + scaled_px(TITLE_ICON_GAP) if not pixmap.isNull() else 0)
        lab.setGeometry(words, 0, max(0, right - words - scaled_px(8)), tb.height())
        lab.raise_()

    def showEvent(self, event):  # noqa: N802 - Qt naming
        super().showEvent(event)
        self._sync_frameless_content_top_margin()
        self._position_window_title()
        # the title bar / buttons may finish their geometry one event-loop tick
        # after show; re-pin then so the right edge is computed against real button
        # positions
        QtCore.QTimer.singleShot(0, self._position_window_title)
        QtCore.QTimer.singleShot(0, self._sync_frameless_content_top_margin)

    def resizeEvent(self, event):  # noqa: N802 - Qt naming
        super().resizeEvent(event)
        self._position_window_title()

    def set_close_guard(self, guard) -> None:
        """Install an internal lifecycle guard evaluated before a real close.

        The guard returns True only after child workers/resources are confirmed
        terminated.  False or an exception keeps the window and its children alive,
        allowing a later close attempt to retry instead of tearing down live owners.
        """

        if guard is not None and not callable(guard):
            raise TypeError("close guard must be callable or None")
        self._zlc_close_guard = guard

    def set_hide_guard(self, guard) -> None:
        """Install the equivalent owner-termination guard for X-to-hide windows."""

        if guard is not None and not callable(guard):
            raise TypeError("hide guard must be callable or None")
        self._zlc_hide_guard = guard

    def closeEvent(self, event):
        guard = self._zlc_hide_guard if self._hide_on_close else self._zlc_close_guard
        if guard is not None:
            try:
                ready = guard() is True
            except BaseException:
                logging.getLogger(__name__).exception("window close guard failed")
                ready = False
            if not ready:
                event.ignore()
                return
        self.close_requested.emit()   # the X was pressed (NOT a minimize); fires whether we hide or destroy
        if self._hide_on_close:
            event.ignore()
            self.hide()
            return
        super().closeEvent(event)
        if event.isAccepted():
            self._zlc_close_committed = True
            self.closed.emit()   # notification after a genuine committed close

    def hideEvent(self, event):
        self.hidden.emit()
        super().hideEvent(event)


class FluentDialogWindow(FluentWindow):
    """Modal Fluent top-level window with the ordinary accept/reject contract."""

    Accepted = QtWidgets.QDialog.Accepted
    Rejected = QtWidgets.QDialog.Rejected

    def __init__(
        self,
        *,
        widget: QtWidgets.QWidget,
        title: str,
        anchor: QtWidgets.QWidget | None = None,
        window_ratio: float = WINDOW_SCREEN_FRACTION,
    ) -> None:
        ratio = float(window_ratio)
        if not 0 < ratio <= 1:
            raise ValueError("window_ratio must be between 0 and 1")
        super().__init__(widget=widget, title=str(title))
        # Its own top-level FROM CONSTRUCTION, on purpose.  This dialog used
        # to be Qt-parented to the window it was opened from and then given
        # the Window flag afterwards -- and a frameless window realizes its
        # native handle inside __init__, so the late flag re-created the
        # handle under it, the construct that froze and cascaded closes on
        # the bench twice.  ``anchor`` is only WHERE to appear: the dialog
        # centres on it, and application modality does the blocking without
        # any native ownership between windows.
        self.setWindowModality(QtCore.Qt.ApplicationModal)
        self._dialog_anchor = anchor
        self._dialog_ratio = ratio
        self._dialog_result = self.Rejected
        self._dialog_loop = None

    def result(self) -> int:
        return int(self._dialog_result)

    def accept(self) -> None:
        self._dialog_result = self.Accepted
        self.close()

    def reject(self) -> None:
        self._dialog_result = self.Rejected
        self.close()

    def exec_(self) -> int:
        """Show modally through the same Fluent shell and screen-size policy."""

        if self._dialog_loop is not None:
            raise RuntimeError("FluentDialogWindow is already executing")
        self._dialog_result = self.Rejected
        app = ensure_qt_app()
        if QtCore.QThread.currentThread() != app.thread():
            raise RuntimeError("Fluent dialogs must run on the GUI thread")
        self.resize(screen_fit_window_size(self._dialog_ratio))
        anchor = self._dialog_anchor
        if anchor is None:
            center_window_on_primary_screen(self, app)
        else:
            # Centred on the window it came from, and then put on a screen:
            # a window near an edge centres this one past it, and a modal
            # nobody can reach is a bench nobody can carry on using.
            frame = self.frameGeometry()
            frame.moveCenter(anchor.window().frameGeometry().center())
            self.move(
                _placed_on_screen(frame.topLeft(), self.size(), widget=anchor)
            )
        retain_window(self)
        loop = QtCore.QEventLoop(self)
        self._dialog_loop = loop
        self.closed.connect(loop.quit)
        self.show()
        try:
            loop.exec_()
            return self.result()
        finally:
            try:
                self.closed.disconnect(loop.quit)
            except (RuntimeError, TypeError):
                pass
            self._dialog_loop = None
            release_window(self)


def bind_body_close(window: "FluentWindow", body: QtWidgets.QWidget) -> None:
    """Pair a body that guards its own close with the window hosting it.

    Several bodies refuse their own close and raise ``close_requested`` instead,
    so a host can tear down before the window goes.  The host half of that
    handshake existed nowhere: every application had to reinvent it, and one
    that did not produced a window whose X did nothing at all -- the body
    ignored the event and nothing answered the request.

    So it is done here, once, for every window this launcher opens.  Both
    directions are bound: a body asking to close closes the window, and a
    window closing lets the body commit its own close and run its teardown.
    A body without the protocol is left alone.
    """

    request = getattr(body, "close_requested", None)
    if request is not None and hasattr(request, "connect"):
        request.connect(window.close)
    finish = getattr(body, "finish_close", None)
    if callable(finish):
        window.close_requested.connect(finish)


def launch_fluent_window(
    widget_or_factory: QtWidgets.QWidget | Callable[[], QtWidgets.QWidget],
    *,
    title: str,
    hide_on_close: bool = False,
    fixed_size: bool = False,
    size: tuple | None = None,
    window_ratio: float = WINDOW_SCREEN_FRACTION,
    wire=None,
) -> FluentWindow:
    """The ONE top-level-GUI launcher sequence.

    Every window from here is a PLAIN top-level that belongs to nobody.
    There used to be an ``owner`` parameter that Qt-parented a new window
    to the one it was opened from and applied the Window flag afterwards
    -- but a frameless window realizes its native handle inside __init__,
    so the late flag re-created the handle underneath it, and on the
    bench that construct twice cascaded closes the wrong way (one closed
    log window froze its siblings; one closed device control reached the
    console, whose close guard shut the running installation).  A surface
    that genuinely belongs to a page -- the panel Setting frame -- is a
    :class:`FluentOverlayFrame`: an ordinary child widget of that page, so
    stacking, tab visibility, shared minimise and shared close all follow
    from parenthood with no desktop machinery at all.  Window flags are
    never changed after construction, anywhere.

    ``widget_or_factory`` may be an already constructed QWidget or a zero-
    argument factory.  The factory form is the canonical public entry because
    it calls :func:`ensure_qt_app` *before* constructing the body, then wraps
    and shows it through one shared lifecycle.  With the default
    ``fixed_size=False`` the initial window is always the shared human-screen
    size from ``window_ratio``; a caller no longer has to remember a per-demo
    ``fixed_size=False`` switch.

    The fixed-content mode remains available explicitly with
    ``fixed_size=True`` for small embedded/test shells.  It is intentionally
    not the default for a top-level human window.

    The launcher wraps the body in a :class:`FluentWindow`, runs the
    caller's ``wire(window)`` (per-GUI signal wiring between construction and show: close ->
    teardown, title propagation), size it, centre it on the primary screen, show it, and retain it
    on the ONE app registry (:func:`retain_window`).

    EVERY formal window entry uses this same lifecycle.
    routes through here.  It exists because hand-copied launchers drift: one had silently dropped
    ensure_qt_app / the shared scale / centring / retention, the same failure family as the
    duplicated scale rule (commit 57a1d25).

    ``fixed_size=False`` with no explicit ``size`` uses
    ``screen_fit_window_size(window_ratio)``.  The caller may pass an already
    constructed widget, but the factory form is preferred so the QApplication
    and High-DPI/font configuration happen before any QWidget constructor."""
    app = ensure_qt_app()
    widget = widget_or_factory
    if not isinstance(widget, QtWidgets.QWidget):
        if not callable(widget):
            raise TypeError("widget_or_factory must be QWidget or a zero-argument factory")
        widget = widget()
    if not isinstance(widget, QtWidgets.QWidget):
        raise TypeError("window factory must return QWidget")
    window = FluentWindow(
        widget=widget,
        title=title,
        hide_on_close=hide_on_close,
    )
    bind_body_close(window, widget)
    if wire is not None:
        wire(window)
    if fixed_size:
        window.adjustSize()
        window.setFixedSize(window.size())
    elif size is not None:
        # A size may be a policy rather than a number: a callable is asked
        # with the built window, so a body that wants "the content's own
        # height plus the chrome" is sized BEFORE it is shown, instead of
        # appearing at the screen fraction and then jumping to its size.
        wanted = size(window) if callable(size) else size
        window.resize(int(wanted[0]), int(wanted[1]))
    else:
        target = screen_fit_window_size(float(window_ratio))
        window.resize(target)
    center_window_on_primary_screen(window, app)
    window.show()
    retain_window(window)   # the ONE app-level registry; pruned when the window dies
    return window


def open_fluent_window(
    factory: Callable[[], QtWidgets.QWidget],
    *,
    title: str,
    hide_on_close: bool = False,
    window_ratio: float = WINDOW_SCREEN_FRACTION,
    wire=None,
) -> FluentWindow:
    """Open a human-facing frameless window from a zero-argument body factory.

    This is the reusable ``create_window``-style API for Notebook cells,
    examples, and host applications.  It owns the ordering
    ``ensure_qt_app() -> factory() -> shared screen-fit launcher`` so every
    caller gets the same High-DPI/font setup, initial screen size, centring,
    retention, and close wiring.  It deliberately has no ``fixed_size`` or
    ad-hoc resize argument: use ``window_ratio`` when a product intentionally
    needs a different human-screen fraction.
    """

    if not callable(factory):
        raise TypeError("factory must be a zero-argument QWidget factory")
    return launch_fluent_window(
        factory,
        title=str(title),
        hide_on_close=hide_on_close,
        fixed_size=False,
        window_ratio=float(window_ratio),
        wire=wire,
    )


# ---------------------------------------------------------------------------
# Layout primitives -- structural guards against alignment / overlap / cutoff.
#
# Pages should compose these instead of hand-rolling fixed-width QHBoxLayouts.
# Labels elide and gain tooltips instead of overflowing; the label column width
# is computed from the actual text; fields share one row height so rows align.
# ---------------------------------------------------------------------------


class ElidedLabel(QtWidgets.QLabel):
    """A label that elides with ``...`` and exposes the full text as a tooltip."""

    def __init__(self, text: str = "", parent=None, *, mode=QtCore.Qt.ElideRight, align=QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft):
        super().__init__(parent)
        self._full = str(text)
        self._mode = mode
        self.setAlignment(align)
        self.setStyleSheet(f'QLabel {{ color: {TEXT}; font: {fluent_font_size()}pt "{FONT}"; background: transparent; }}')
        self.setMinimumWidth(scaled_px(8))
        # Elision makes the *painted* QLabel text fit the assigned geometry, so
        # the normal Preferred policy remains width-neutral once layout has
        # assigned that geometry.  QSizePolicy.Ignored is not equivalent: Qt
        # then omits even an explicit minimumWidth from the layout allocation
        # while QWidget still enforces that minimum on itself.  The result is
        # overlapping siblings (and right-edge overflow) whenever a caller
        # reserves a real identity/status column.  Keep the label shrinkable,
        # but let the layout honour its declared minimum.
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred,
            QtWidgets.QSizePolicy.Preferred,
        )
        self.setText(str(text))

    def setText(self, text: str) -> None:  # noqa: N802 - Qt API name
        self._full = str(text)
        self.setToolTip(self._full)
        self._elide()

    def text(self) -> str:  # noqa: N802 - Qt API name
        return self._full

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._elide()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        # On the row's one line, as a FluentLabel is.
        if not _paint_label_line(self):
            super().paintEvent(event)

    def _elide(self) -> None:
        metrics = self.fontMetrics()
        available = max(0, self.width() - scaled_px(2))
        # During construction/layout a label can briefly have width zero.  Do
        # not put the full diagnostic/path back into QLabel in that state: its
        # resulting size hint would make the very text being elided author the
        # parent width.  The next resize paints the proper elision.
        shown = (
            metrics.elidedText(self._full, self._mode, available)
            if available > scaled_px(4)
            else ""
        )
        super().setText(shown)


class FluentStatusStrip(FluentFrame):
    """The PERSISTENT one-line status surface a GUI mounts once and feeds forever.

    One severity-coloured dot + one eliding message line (full text in the tooltip -- strip
    text can carry paths/errors, so it must elide, never clip) + an optional right-side action
    button (e.g. the console's "Stop task").  The strip is ALWAYS visible: content switches by
    priority instead of rows appearing/disappearing (the old transient task banner shifted the
    whole layout under the pointer every time a task started/finished).

    Severity is a DATA input mapped to the one palette here, in the one status vocabulary
    (``zlc_ui.status.STATUS_SEVERITIES``): ``idle`` (grey), ``task`` (accent dot, plain ink --
    what just happened or is under way), ``warning`` (amber advisory), ``error`` (red).  Task
    and warning shared the amber, so "added occupancy" read as something wrong.
    The shared precedent is the always-visible last-message-wins Log strip; this deviates
    consciously by adding severity colour + the action slot, which its separate popup/timer
    channels carried instead."""

    #: severity -> (dot colour, text colour); the ONE mapping every consumer shares.  Keyed
    #: by the status vocabulary itself: a word added there without a colour here fails at
    #: import, not in the first Qt slot that shows it.
    SEVERITIES = {
        word: {"idle": (GREY, MUTED_INK), "task": (ACCENT, TEXT),
               "warning": (ORANGE, ORANGE_DARK), "error": (RED, RED_DARK)}[word]
        for word in STATUS_SEVERITIES
    }

    action_clicked = QtCore.pyqtSignal()

    def __init__(self, parent=None, *, action_text: str = ""):
        super().__init__(parent, bordered=False)
        self.setFixedHeight(scaled_px(38, minimum=30))
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(scaled_px(12), scaled_px(4), scaled_px(12), scaled_px(4))
        lay.setSpacing(scaled_px(8))
        self.dot = FluentStatusDot(size=12)
        self.message = ElidedLabel("")
        lay.addWidget(self.dot)
        lay.addWidget(self.message, 1)
        self.action_button = None
        if action_text:
            self.action_button = FluentButton(str(action_text), role="destructive")
            self.action_button.clicked.connect(self.action_clicked.emit)
            self.action_button.setVisible(False)
            lay.addWidget(self.action_button)
        self._severity = None
        self.show_message("", severity="idle")

    @property
    def current_severity(self) -> str:
        return self._severity

    def show_message(self, text: str, *, severity: str = "idle") -> None:
        """Set the strip's line.  Change-gated so the per-tick caller never repolishes Qt
        styles for an unchanged state."""
        if severity not in self.SEVERITIES:
            raise ValueError(f"unknown severity {severity!r}; choose from {tuple(self.SEVERITIES)}.")
        if severity != self._severity:
            self._severity = severity
            dot, colour = self.SEVERITIES[severity]
            self.dot.set_color(dot)
            self.message.setStyleSheet(
                f'QLabel {{ color: {colour}; font: {fluent_font_size()}pt "{FONT}"; background: transparent; }}')
        if str(text) != self.message.text():
            self.message.setText(str(text))

    def set_action_visible(self, visible: bool) -> None:
        """Show/hide the action button (a no-op when the strip was built without one)."""
        if self.action_button is not None:
            self.action_button.setVisible(bool(visible))

    def text(self) -> str:
        """The current message's FULL text (the label elides for display only)."""
        return self.message.text()
