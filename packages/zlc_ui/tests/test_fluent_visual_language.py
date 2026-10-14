"""The Fluent controls speak one visual language, read back from a render.

Hover is the light ACCENT_TINT wash; what is chosen or pressed is the
medium SELECTED fill, rounded, with no outline; ACCENT is left to small
indicators without words; a state is a fill, never a line drawn round a
shape -- only fields keep an edge, and an edge that says something is
dashed.  Rendered once at the platform's own ratio and read pixel by pixel.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def app():
    from zlc_ui.qt import ensure_qt_app

    return ensure_qt_app(["visual-language"])


@pytest.fixture
def host(app):
    from PyQt5 import QtGui, QtWidgets
    from zlc_ui.fluent.style import SURFACE

    widget = QtWidgets.QWidget()
    palette = widget.palette()
    palette.setColor(QtGui.QPalette.Window, QtGui.QColor(SURFACE))
    widget.setPalette(palette)
    widget.setAutoFillBackground(True)
    widget.resize(520, 320)
    yield widget
    widget.close()


def _shown(app, host, *widgets) -> None:
    host.show()
    for widget in widgets:
        widget.show()
    app.processEvents()


def _at(widget, x: float, y: float) -> str:
    """The colour on screen at ``widget``'s logical ``(x, y)``: read from
    its window, so a transparent widget shows what it stands on."""

    from PyQt5 import QtCore

    window = widget.window()
    origin = widget.mapTo(window, QtCore.QPoint(0, 0))
    image = window.grab().toImage()
    ratio = image.devicePixelRatio() or 1.0
    return image.pixelColor(int((origin.x() + x) * ratio), int((origin.y() + y) * ratio)).name().upper()


def _under_pointer(widget, pos=None) -> None:
    """Hover ``widget`` (at ``pos``) without moving the real cursor."""

    from PyQt5 import QtCore, QtGui, QtWidgets

    widget.setAttribute(QtCore.Qt.WA_UnderMouse, True)
    QtWidgets.QApplication.sendEvent(widget, QtCore.QEvent(QtCore.QEvent.Enter))
    if pos is not None:
        point = QtCore.QPointF(pos)
        for kind, old in ((QtCore.QEvent.HoverEnter, QtCore.QPointF(-1, -1)),
                          (QtCore.QEvent.HoverMove, point)):
            QtWidgets.QApplication.sendEvent(widget, QtGui.QHoverEvent(kind, point, old))
    widget.update()


def test_a_chosen_grid_card_is_the_selected_fill_and_nothing_round_it(app, host) -> None:
    from zlc_ui.fluent import FluentSelectionGrid, scaled_px
    from zlc_ui.fluent.style import ACCENT_TINT, BG, RED_TINT, SELECTED, SELECTION_EDGE

    grid = FluentSelectionGrid(host)
    grid.set_grid(("a", "b"), ("0", "1", "2", "3"),
                  {(row, column): str(column) for row in range(2) for column in range(4)})
    grid.set_selected([(0, 0), (0, 1), (1, 2)])
    # (0, 1) is chosen by a rule: derived wins over the selection's fill.
    grid.set_cell_states({(0, 1): "derived", (1, 2): "warning"})
    grid.resize(grid.sizeHint())
    _shown(app, host, grid)
    viewport = grid.viewport()

    def card(row, column):
        return grid.visualRect(grid.model().index(row, column))

    def colour(x, y):
        return _at(viewport, x, y)

    chosen, empty, derived, warned = card(0, 0), card(0, 2), card(0, 1), card(1, 2)
    # The card's top row: the cell inset by the grid's gutter.
    inset = scaled_px(2, minimum=1)
    middle = chosen.center().x()
    assert colour(chosen.left() + 6, chosen.center().y()) == SELECTED.upper()
    assert colour(middle, chosen.top() + inset) == SELECTED.upper(), "no outline on the chosen card"
    assert colour(empty.left() + 6, empty.center().y()) == BG.upper(), "an empty card is a neutral fill"
    assert colour(empty.center().x(), empty.top() + inset) == BG.upper(), "and has no edge"
    assert colour(derived.left() + 6, derived.center().y()) == ACCENT_TINT.upper()
    top_row = {colour(x, derived.top() + inset) for x in range(derived.left() + 6, derived.right() - 6)}
    assert SELECTION_EDGE.upper() in top_row, "a derived card keeps its dashed edge"
    assert ACCENT_TINT.upper() in top_row or len(top_row) > 1, "dashed, not solid"
    assert colour(warned.left() + 6, warned.center().y()) == RED_TINT.upper(), (
        "a warning shows even on a chosen card")
    # One extent for every card: the chosen face is as large as the edged one.
    def extent(rect):
        row = rect.center().y()
        ground = colour(rect.left(), row)
        inked = [x for x in range(rect.left(), rect.right() + 1) if colour(x, row) != ground]
        return inked[0] - rect.left(), rect.right() - inked[-1]
    assert extent(chosen) == extent(derived) == extent(empty)

    # Under the pointer: an empty card takes the hover wash.
    _under_pointer(viewport, empty.center())
    app.processEvents()
    assert colour(empty.left() + 6, empty.center().y()) == ACCENT_TINT.upper()


def test_a_row_remover_in_a_grid_is_the_buttons_ghost_x(app, host) -> None:
    from zlc_ui.fluent import FluentSelectionGrid
    from zlc_ui.fluent.style import RED_TINT, SURFACE

    grid = FluentSelectionGrid(host, row_actions=(("remove", "×", "Remove"),))
    grid.set_grid(("a",), ("0",), {(0, 0): "0"})
    grid.resize(grid.sizeHint())
    _shown(app, host, grid)
    viewport = grid.viewport()
    action = grid.visualRect(grid.model().index(0, 2))
    assert _at(viewport, action.left() + 4, action.top() + 4) == SURFACE.upper(), "no face or edge at rest"
    grid._hovered_action = (0, 2)
    viewport.update()
    app.processEvents()
    assert _at(viewport, action.left() + 4, action.top() + 6) == RED_TINT.upper()


def test_a_selected_run_of_table_rows_is_one_rounded_block(app, host) -> None:
    from PyQt5 import QtCore, QtGui
    from zlc_ui.fluent import FluentTableView
    from zlc_ui.fluent.style import ACCENT_TINT, SELECTED, SELECTED_HOVER

    model = QtGui.QStandardItemModel(6, 2)
    for row in range(6):
        for column in range(2):
            model.setItem(row, column, QtGui.QStandardItem(""))
    table = FluentTableView(host)
    table.setModel(model)
    table.resize(300, 260)
    selection = table.selectionModel()
    for row in (1, 2):
        selection.select(model.index(row, 0),
                         QtCore.QItemSelectionModel.Select | QtCore.QItemSelectionModel.Rows)
    _shown(app, host, table)
    viewport = table.viewport()
    x = table.columnViewportPosition(0) + 20
    first_top = table.rowViewportPosition(1)
    joint = table.rowViewportPosition(2) - 1
    assert _at(viewport, x, first_top + 8) == SELECTED.upper()
    assert _at(viewport, x, joint) == SELECTED.upper(), "no rule cuts a selected run"
    left = table.columnViewportPosition(0)
    corner = _at(viewport, left + 2, first_top)
    assert corner != SELECTED.upper(), "the block's corner is rounded"
    table._hover_row = 4
    viewport.update()
    app.processEvents()
    assert _at(viewport, x, table.rowViewportPosition(4) + 8) == ACCENT_TINT.upper()
    table._hover_row = 2
    viewport.update()
    app.processEvents()
    assert _at(viewport, x, table.rowViewportPosition(2) + 8) == SELECTED_HOVER.upper()


def test_a_stepper_is_chevrons_and_a_tint_inside_the_frame(app, host) -> None:
    from PyQt5 import QtWidgets
    from zlc_ui.fluent import FluentDoubleSpinBox
    from zlc_ui.fluent import fluent as F
    from zlc_ui.fluent.style import ACCENT, ACCENT_TINT, PLACEHOLDER, SURFACE

    box = FluentDoubleSpinBox(host)
    box.setRange(0.0001, 100.0)
    box.setValue(1)
    box.resize(160, box.sizeHint().height())
    _shown(app, host, box)
    left, up, down = F._spin_stepper_cells(box)
    middle = box.height() / 2.0
    assert _at(box, left, middle) == SURFACE.upper(), "no line divides the stepper from the digits"
    _under_pointer(box, up.center().toPoint())
    app.processEvents()
    assert _at(box, up.right() - 1, up.center().y()) == ACCENT_TINT.upper(), "the tint reaches the inner edge"
    edge = _at(box, 6, 0)
    assert edge in (PLACEHOLDER.upper(), ACCENT.upper()), edge
    assert _at(box, box.width() - 1, up.center().y()) == edge, "and leaves the frame's edge whole"
    assert _at(box, up.center().x(), 0) == edge
    # Down from 1 with a floor of 0.0001 cannot move: its chevron is dimmed.
    enabled = box.stepEnabled()
    assert enabled & QtWidgets.QAbstractSpinBox.StepUpEnabled
    assert not enabled & QtWidgets.QAbstractSpinBox.StepDownEnabled

    def darkest(cell):
        image = box.grab().toImage()
        ratio = image.devicePixelRatio() or 1.0
        return min(
            image.pixelColor(int(x * ratio), int(y * ratio)).lightness()
            for x in range(int(cell.left()) + 2, int(cell.right()) - 1)
            for y in range(int(cell.top()) + 1, int(cell.bottom()) - 1)
        )

    assert darkest(down) > darkest(up), "the dead direction's chevron is lighter"


def test_switches_and_ticks_are_solid_fills(app, host) -> None:
    from PyQt5 import QtGui
    from zlc_ui.fluent import FluentCheckBox, FluentSwitch
    from zlc_ui.fluent.style import ACCENT, DIVIDER, NEUTRAL_FILL, PLACEHOLDER

    off = FluentSwitch("Off", host)
    disabled = FluentSwitch("Disabled", host)
    disabled.move(0, 40)
    disabled.setEnabled(False)
    unticked = FluentCheckBox("Unticked", host)
    unticked.move(0, 80)
    ticked = FluentCheckBox("Ticked", host)
    ticked.move(0, 120)
    ticked.setChecked(True)
    _shown(app, host, off, disabled, unticked, ticked)
    track = off._track_box()
    assert _at(off, track.center().x(), track.top()) == PLACEHOLDER.upper(), "no ring round the track"
    assert _at(off, track.right() - 3, track.center().y()) == PLACEHOLDER.upper()
    assert _at(disabled, track.center().x(), track.top()) == DIVIDER.upper()
    box = unticked._indicator_rect()
    assert _at(unticked, box.center().x(), box.top()) == NEUTRAL_FILL.upper(), "its edge is its fill"
    assert _at(unticked, box.left() + 3, box.center().y()) == NEUTRAL_FILL.upper()
    box = ticked._indicator_rect()
    assert _at(ticked, box.center().x(), box.top()) == ACCENT.upper()
    # The white tick is centred on the box.
    image = ticked.grab().toImage()
    ratio = image.devicePixelRatio() or 1.0
    white = [
        (x / ratio, y / ratio)
        for x in range(int(box.left() * ratio), int((box.right() + 1) * ratio))
        for y in range(int(box.top() * ratio), int((box.bottom() + 1) * ratio))
        if QtGui.QColor(image.pixel(x, y)).lightness() > 235
    ]
    assert white
    xs, ys = [p[0] for p in white], [p[1] for p in white]
    centre = QtGui.QVector2D(box.left() + box.width() / 2.0, box.top() + box.height() / 2.0)
    assert abs((min(xs) + max(xs) + 1 / ratio) / 2.0 - centre.x()) <= 1.0
    assert abs((min(ys) + max(ys) + 1 / ratio) / 2.0 - centre.y()) <= 1.0


def test_the_current_segment_and_an_active_cycle_are_the_selected_fill(app, host) -> None:
    from zlc_ui.fluent import FluentCycleComboBox, FluentSegmentedButtons
    from zlc_ui.fluent.style import ACCENT_TINT, PLACEHOLDER, SELECTED

    segments = FluentSegmentedButtons((("a", "Alpha"), ("b", "Beta")), host)
    segments.set_current("a")
    cycle = FluentCycleComboBox(host)
    cycle.addItems(["Scope", "All"])
    cycle.move(0, 60)
    cycle.resize(140, cycle.sizeHint().height())
    cycle.setChecked(True)
    _shown(app, host, segments, cycle)
    current = segments.button("a")
    assert _at(current, 4, current.height() / 2.0) == SELECTED.upper()
    assert _at(current, current.width() / 2.0, 0) == SELECTED.upper(), "no outline on the current choice"
    other = segments.button("b")
    _under_pointer(other)
    app.processEvents()
    assert _at(other, 4, other.height() / 2.0) == ACCENT_TINT.upper(), "hover is the light wash"
    assert _at(cycle, 3, cycle.height() / 2.0) == SELECTED.upper()
    assert _at(cycle, cycle.width() / 2.0, 0) == PLACEHOLDER.upper(), "its edge is the field's own"


def test_a_selected_card_keeps_its_own_edge_and_chips_have_none(app, host) -> None:
    from zlc_ui.fluent import FluentGroupBox, PublishedItemsLegend
    from zlc_ui.fluent.style import ACCENT_TINT, BG, DIVIDER

    card = FluentGroupBox("Period", host)
    card.resize(140, 80)
    card.set_selected(True)
    legend = PublishedItemsLegend(host)
    legend.move(0, 100)
    legend.resize(240, 40)
    legend.set_rows((("counts", "(n,)", ""),))
    _shown(app, host, card, legend)
    assert _at(card, 20, 60) == ACCENT_TINT.upper()
    assert _at(card, 70, 0) == DIVIDER.upper(), "the card's edge, not a second selection edge"
    chip = legend._chips(legend.width())[0][0]
    assert _at(legend, chip.center().x(), chip.top()) == BG.upper(), "a chip has no hairline"


def test_an_info_tree_row_is_washed_whole(app, host) -> None:
    from zlc_ui.fluent.info_pane import InfoTree
    from zlc_ui.fluent.style import ACCENT_TINT, SELECTED

    tree = InfoTree(name_width=120, parent=host)
    tree.set_rows((("name", "value"), ("other", "3")))
    tree.resize(320, 120)
    _shown(app, host, tree)
    first, second = tree.topLevelItem(0), tree.topLevelItem(1)
    tree.setCurrentItem(first)
    app.processEvents()
    viewport = tree.viewport()
    row = tree.visualItemRect(first)
    for x in (12, viewport.width() // 2, viewport.width() - 12):
        assert _at(viewport, x, row.center().y()) == SELECTED.upper(), x
    _under_pointer(viewport, tree.visualItemRect(second).center())
    app.processEvents()
    assert _at(viewport, viewport.width() // 2, tree.visualItemRect(second).center().y()) == ACCENT_TINT.upper()


def test_a_glyph_is_centred_on_its_ink(app) -> None:
    from PyQt5 import QtCore, QtGui, QtWidgets
    from zlc_ui.fluent import paint_centred_ink

    surface = QtWidgets.QWidget()
    surface.resize(40, 40)
    image = QtGui.QImage(40, 40, QtGui.QImage.Format_ARGB32)
    image.fill(QtGui.QColor("white"))
    font = QtGui.QFont("Segoe UI")
    font.setPointSize(16)
    painter = QtGui.QPainter(image)
    paint_centred_ink(painter, surface, QtCore.QRectF(0, 0, 40, 40), font, "×", "black")
    painter.end()
    inked = [(x, y) for x in range(40) for y in range(40)
             if QtGui.QColor(image.pixel(x, y)).lightness() < 200]
    assert inked
    xs, ys = [p[0] for p in inked], [p[1] for p in inked]
    assert abs((min(xs) + max(xs) + 1) / 2.0 - 20) <= 1.0
    assert abs((min(ys) + max(ys) + 1) / 2.0 - 20) <= 1.0


def test_a_button_disabled_or_hidden_with_the_focus_hands_it_to_nobody(app, host) -> None:
    """Disabled while it held the focus, On Pulse passed it to Stop Pulse,
    and the next Space stopped the pulse just started."""

    from PyQt5 import QtWidgets
    from zlc_ui.fluent import FluentButton

    run, stop = FluentButton("On Pulse", host), FluentButton("Stop Pulse", host)
    stop.move(120, 0)
    _shown(app, host, run, stop)
    host.activateWindow()
    QtWidgets.QApplication.setActiveWindow(host)
    for hide in (False, True):
        run.show()
        run.setEnabled(True)
        run.setFocus()
        app.processEvents()
        assert run.hasFocus()
        if hide:
            run.hide()
        else:
            run.setEnabled(False)
        app.processEvents()
        assert not stop.hasFocus(), "hidden" if hide else "disabled"
        assert QtWidgets.QApplication.focusWidget() is None
