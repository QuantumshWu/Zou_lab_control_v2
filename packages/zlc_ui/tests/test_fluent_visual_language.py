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
    from zlc_ui.fluent import FluentSelectionGrid, control_height, row_gap
    from zlc_ui.fluent.fluent import _grid_row_insets
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
    # A row is a form row: a control-high card and a row gap shared above
    # and below it.  The card's top row: the cell inset by its share above.
    assert chosen.height() == control_height() + row_gap()
    inset = _grid_row_insets()[0]
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
    # The grid ends on its last row's cards: no gutter below them adds to
    # the gap the layout puts after the grid.
    faced = [y for y in range(warned.top(), viewport.height())
             if colour(warned.left() + 6, y) == RED_TINT.upper()]
    assert faced and faced[-1] == viewport.height() - 1
    assert warned.height() < chosen.height()

    # Under the pointer: an empty card takes the hover wash.
    _under_pointer(viewport, empty.center())
    app.processEvents()
    assert colour(empty.left() + 6, empty.center().y()) == ACCENT_TINT.upper()


def test_a_row_remover_in_a_grid_is_the_buttons_ghost_x(app, host) -> None:
    from zlc_ui.fluent import FluentSelectionGrid, control_height
    from zlc_ui.fluent.style import BG, RED_TINT, SURFACE

    grid = FluentSelectionGrid(host, row_actions=(("remove", "×", "Remove"),))
    grid.set_grid(("a", "b"), ("0",), {(0, 0): "0", (1, 0): "0"})
    grid.resize(grid.sizeHint())
    _shown(app, host, grid)
    viewport = grid.viewport()
    action = grid.visualRect(grid.model().index(0, 2))
    assert _at(viewport, action.left() + 4, action.top() + 4) == SURFACE.upper(), "no face or edge at rest"
    grid._hovered_action = (0, 2)
    viewport.update()
    app.processEvents()
    assert _at(viewport, action.left() + 4, action.top() + 6) == RED_TINT.upper()
    # Its face is a control-high square on the band of the cards beside it.
    cell = grid.visualRect(grid.model().index(0, 0))

    def band(x, fill):
        return [y for y in range(cell.top(), cell.bottom() + 1) if _at(viewport, x, y) == fill.upper()]

    # Read beside the glyph, and along a row under the rounded corners and above it.
    face = band(action.left() + 4, RED_TINT)
    assert face == band(cell.left() + 4, BG) and len(face) == control_height()
    across = [x for x in range(action.left(), action.right() + 1)
              if _at(viewport, x, face[4]) == RED_TINT.upper()]
    assert len(across) == control_height()


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
    from PyQt5 import QtCore, QtTest, QtWidgets
    from zlc_ui.fluent import FluentDoubleSpinBox
    from zlc_ui.fluent import fluent as F
    from zlc_ui.fluent.style import ACCENT, ACCENT_TINT, PLACEHOLDER, SELECTED, SURFACE

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
    # Held down, the cell is the SELECTED tile every press is.
    QtTest.QTest.mousePress(box, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, up.center().toPoint())
    app.processEvents()
    assert _at(box, up.right() - 1, up.center().y()) == SELECTED.upper()
    QtTest.QTest.mouseRelease(box, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, up.center().toPoint())
    box.setValue(1)
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

    # Disabled (a read-only card), ticked and unticked still read apart: the
    # unticked box keeps NEUTRAL_FILL, the ticked one is DISABLED_INK with
    # the white tick.  With no edge, the unticked fill must read on every
    # face a read-only box sits on -- a white card, the BG page, the
    # ACCENT_TINT body of a selected or expanded block -- and the ticked
    # fill must stay clearly darker than it.
    from zlc_ui.fluent.style import ACCENT_TINT, BG, DISABLED_INK, SURFACE
    off_box = FluentCheckBox("Read-only off", host)
    off_box.move(160, 80)
    on_box = FluentCheckBox("Read-only on", host)
    on_box.move(160, 120)
    on_box.setChecked(True)
    for check in (off_box, on_box):
        check.setEnabled(False)
    _shown(app, host, off_box, on_box)
    box = off_box._indicator_rect()
    unticked_fill = _at(off_box, box.center().x(), box.top() + 1)
    assert unticked_fill == NEUTRAL_FILL.upper()
    box = on_box._indicator_rect()
    ticked_fill = _at(on_box, box.center().x(), box.top() + 1)
    assert ticked_fill == DISABLED_INK.upper()

    def luminance(colour: str) -> float:
        channels = [int(colour[i:i + 2], 16) / 255.0 for i in (1, 3, 5)]
        linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
        return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]

    def contrast(a: str, b: str) -> float:
        high, low = sorted((luminance(a), luminance(b)), reverse=True)
        return (high + 0.05) / (low + 0.05)

    for face in (SURFACE, BG, ACCENT_TINT):
        assert contrast(unticked_fill, face) >= 1.4, (face, contrast(unticked_fill, face))
    assert contrast(unticked_fill, DIVIDER) > 1.2, "not the DIVIDER grey that vanished"
    assert contrast(ticked_fill, unticked_fill) >= 1.8, contrast(ticked_fill, unticked_fill)
    image = on_box.grab().toImage()
    ratio = image.devicePixelRatio() or 1.0
    brightest = max(
        QtGui.QColor(image.pixel(x, y)).lightness()
        for x in range(int(box.left() * ratio), int((box.right() + 1) * ratio))
        for y in range(int(box.top() * ratio), int((box.bottom() + 1) * ratio))
    )
    assert brightest == QtGui.QColor(SURFACE).lightness(), "a white tick, not a grey one"


def test_a_switch_that_says_its_state_keeps_the_width_of_its_widest_state(app, host) -> None:
    from zlc_ui.fluent import FluentSwitch, fluent_switch_width

    switch = FluentSwitch("Auto", host, room_for=("Auto", "Manual"))
    widest = fluent_switch_width("Manual")
    assert fluent_switch_width("Auto") < widest
    assert switch.sizeHint().width() == widest
    switch.setText("Manual")
    assert switch.sizeHint().width() == widest
    switch.setText("Auto")
    assert switch.sizeHint().width() == widest, "what follows it never moves"
    assert FluentSwitch("Auto", host).sizeHint().width() == fluent_switch_width("Auto")


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

    # A toggle command that is ON is held down: the same SELECTED fill, a
    # step deeper under the pointer, over a tinted card's white face too.
    from PyQt5 import QtCore, QtWidgets
    from zlc_ui.fluent import FluentButton, FluentPopup, FluentSettingsPopupAnchor
    from zlc_ui.fluent.style import BUTTON_TINT, DIVIDER, SELECTED_HOVER

    toggle = FluentButton("Collapse", host)
    toggle.move(0, 110)
    toggle.set_active(True)
    _shown(app, host, toggle)

    def face(button):
        app.processEvents()
        return _at(button, 4, button.height() / 2.0)

    assert toggle.is_active() and face(toggle) == SELECTED.upper()
    _under_pointer(toggle)
    assert face(toggle) == SELECTED_HOVER.upper()
    toggle.setAttribute(QtCore.Qt.WA_UnderMouse, False)
    QtWidgets.QApplication.sendEvent(toggle, QtCore.QEvent(QtCore.QEvent.Leave))
    toggle.set_on_tint(True)
    assert face(toggle) == SELECTED.upper(), "an engaged toggle outranks the on-tint face"
    toggle.set_on_tint(False)
    toggle.set_active(False)
    assert face(toggle) == BUTTON_TINT.upper()
    toggle.set_active(True)
    toggle.setEnabled(False)
    assert face(toggle) == DIVIDER.upper(), "disabled hides the state, as a disabled segment does"
    # The button a frame opens from is ON while its frame is open.
    opener = FluentButton("Setting", host)
    opener.move(0, 150)
    _shown(app, host, opener)
    popup = FluentPopup(host)
    FluentSettingsPopupAnchor(popup, opener).toggle(QtWidgets.QLabel("frame", popup))
    assert opener.is_active() and face(opener) == SELECTED.upper()
    popup.hide()
    assert not opener.is_active() and face(opener) == BUTTON_TINT.upper()


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


def test_an_empty_code_view_says_why_in_muted_ink(app, host) -> None:
    """The note is set after the (empty) text, as a page projects it: Qt
    5.15 drew a QPlainTextEdit placeholder only after a text change, and
    the empty scan table stood blank."""

    from PyQt5 import QtGui
    from zlc_ui.fluent import FluentCodeEdit
    from zlc_ui.fluent.style import MUTED_INK

    view = FluentCodeEdit(read_only=True, parent=host)
    view.resize(300, 120)
    view.setPlainText("")
    view.setPlaceholderText("No scan table yet: write a program and press Run.")
    _shown(app, host, view)
    image = view.viewport().grab().toImage()
    lightness = [image.pixelColor(x, y).lightness()
                 for y in range(image.height()) for x in range(image.width())]
    muted = QtGui.QColor(MUTED_INK).lightness()
    assert min(lightness) <= muted + 8, "the empty view shows its note"
    assert min(lightness) >= muted - 8, "in the muted ink of a note, not the text's"


def test_a_blank_choice_reads_as_a_placeholder(app, host) -> None:
    """A combo's "(not selected)" is a missing input, not a value: muted ink
    like an empty field's placeholder, flat or tree, while a row whose empty
    key is a real choice keeps the text's ink."""

    from PyQt5 import QtGui
    from zlc_ui.fluent import FluentComboBox, FluentTreeComboBox
    from zlc_ui.fluent.style import MUTED_INK, TEXT

    flat = FluentComboBox(host)
    flat.add_blank_item("(not selected)")
    flat.addItem("sequencer", "sequencer")
    flat.resize(200, flat.sizeHint().height())
    tree = FluentTreeComboBox(host)
    tree.move(0, 60)
    tree.set_choice_tree((("signals", (("count", "count", "signals · count"),)),),
                         current="", none_label="(not selected)")
    tree.resize(200, tree.sizeHint().height())
    keyed = FluentComboBox(host)
    keyed.move(0, 120)
    keyed.addItem("Independent file", "")
    keyed.resize(200, keyed.sizeHint().height())
    _shown(app, host, flat, tree, keyed)

    def ink(combo):
        image = combo.grab().toImage()
        ratio = image.devicePixelRatio() or 1.0
        rect = combo._collapsed_text_rect()
        return min(
            (QtGui.QColor(image.pixel(x, y)) for x in range(int(rect.left() * ratio), int(rect.right() * ratio))
             for y in range(int(rect.top() * ratio), int(rect.bottom() * ratio))),
            key=lambda colour: colour.lightness(),
        ).name().upper()

    assert flat.current_is_blank() and tree.current_is_blank() and not keyed.current_is_blank()
    assert ink(flat) == MUTED_INK.upper()
    assert ink(tree) == MUTED_INK.upper()
    assert ink(keyed) == TEXT.upper()
    flat.setCurrentIndex(1)
    tree.select_choice_key("count")
    app.processEvents()
    assert ink(flat) == TEXT.upper()
    assert ink(tree) == TEXT.upper()


def test_a_switched_off_editable_combo_writes_in_disabled_ink(app, host) -> None:
    """Its prompt and its typed text alike, as a non-editable combo's text:
    a prompt in reading ink invited typing into a box that refuses it."""

    from PyQt5 import QtGui
    from zlc_ui.fluent import FluentComboBox
    from zlc_ui.fluent.style import DISABLED_INK, MUTED_INK, TEXT

    prompt = FluentComboBox(host)
    prompt.setEditable(True)
    prompt.lineEdit().setPlaceholderText("Choose or type a name")
    typed = FluentComboBox(host)
    typed.move(0, 60)
    typed.setEditable(True)
    typed.setEditText("bias_value")
    for combo in (prompt, typed):
        combo.resize(240, combo.sizeHint().height())
    _shown(app, host, prompt, typed)

    def inks(combo):
        image = combo.grab().toImage()
        return {QtGui.QColor(image.pixel(x, y)).name().upper()
                for x in range(image.width()) for y in range(image.height())}

    assert MUTED_INK.upper() in inks(prompt) and TEXT.upper() in inks(typed)
    for combo in (prompt, typed):
        combo.setEnabled(False)
    app.processEvents()
    for combo in (prompt, typed):
        assert DISABLED_INK.upper() in inks(combo)
        assert not {MUTED_INK.upper(), TEXT.upper()} & inks(combo)


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


def test_a_status_word_is_one_colour_for_a_dot_a_line_and_a_readout(app, host) -> None:
    """One table, ``STATUS_CUES``: a dot, the words beside it, a copyable
    readout and the error line under a form all take a status word's colour
    from it -- read back from the render, not from the sheet."""

    from PyQt5 import QtCore, QtGui
    from zlc_ui.fluent import (
        STATUS_CUES, FluentLabel, FluentReadoutEdit, FluentStatusDot, error_note_label,
        set_status_ink,
    )
    from zlc_ui.fluent.style import FONT, TEXT

    def closest(widget, colour: str) -> int:
        """How far the render's darkest pixel -- a stem fully inked -- is
        from ``colour`` (summed channel distance).  Not the nearest pixel:
        a dark ink's antialiased edges pass through every lighter grey."""

        want = QtGui.QColor(colour)
        window = widget.window()
        image = window.grab(QtCore.QRect(widget.mapTo(window, QtCore.QPoint(0, 0)),
                                         widget.size())).toImage()
        pixel = min(
            (image.pixelColor(x, y) for y in range(image.height()) for x in range(image.width())),
            key=lambda colour: colour.lightness(),
        )
        return (abs(pixel.red() - want.red()) + abs(pixel.green() - want.green())
                + abs(pixel.blue() - want.blue()))

    dot = FluentStatusDot(host)
    label = FluentLabel("IIII WWWW", host)
    label.move(0, 30)
    readout = FluentReadoutEdit("IIII WWWW", host, flush=True)
    readout.setFixedWidth(200)
    readout.move(0, 70)
    note = error_note_label("IIII WWWW broke")
    note.setParent(host)
    note.move(0, 110)
    _shown(app, host, dot, label, readout, note)
    # A readout reports in body ink until it is told a word.
    assert closest(readout, TEXT) <= 24
    for word, (dot_colour, ink) in STATUS_CUES.items():
        dot.set_state(word)
        set_status_ink(label, word)
        set_status_ink(readout, word)
        app.processEvents()
        centre = dot.rect().center()
        assert _at(dot, centre.x(), centre.y()) == QtGui.QColor(dot_colour).name().upper(), word
        assert closest(label, ink) <= 24, (word, ink)
        assert closest(readout, ink) <= 24, (word, ink)
        # The font travels with the ink: a colour sheet that dropped it
        # re-set the words in the window's default face.
        assert FONT in label.styleSheet(), word
    # The same word again writes nothing: a status beat must not repolish.
    last = label.property("status_word")
    sheet = label.styleSheet()
    label.setStyleSheet(sheet + " ")
    set_status_ink(label, last)
    assert label.styleSheet() == sheet + " "
    assert note.property("status_word") == "error"
    assert closest(note, STATUS_CUES["error"][1]) <= 24


def test_only_a_button_told_to_confirm_twice_arms_and_its_second_click_confirms(app, host) -> None:
    """Removing a logic node or a plot panel takes two clicks, and nothing
    else does.  The first click arms the white-on-red confirm, label and
    width unchanged; a second within the double-click interval confirms;
    left alone, disabled or hidden it is back at rest; a double-click is
    two clicks."""

    from PyQt5 import QtCore, QtGui, QtTest, QtWidgets
    from zlc_ui.fluent import FluentButton, arm_interval_ms
    from zlc_ui.fluent.style import RED_DARK, RED_TINT, SURFACE

    side = FluentButton("x").minimumHeight()
    glyph = FluentButton("×", host, role="destructive", confirm_twice="remove this node")
    glyph.setFixedSize(side, side)
    labelled = FluentButton("Remove", host, role="destructive", confirm_twice="remove this node")
    labelled.move(60, 0)
    plain = FluentButton("×", host, role="destructive")
    plain.setFixedSize(side, side)
    plain.move(180, 0)
    _shown(app, host, glyph, labelled, plain)
    confirmed, clicked = [], []
    for button in (glyph, labelled):
        button.confirmed.connect(lambda button=button: confirmed.append(button))
    plain.clicked.connect(lambda: clicked.append(plain))

    def face(button):
        app.processEvents()
        return _at(button, 3, button.height() / 2.0)

    def click(button):
        QtTest.QTest.mouseClick(button, QtCore.Qt.LeftButton)

    # At rest: the ghost x and the labelled red face, each saying it takes two.
    assert face(glyph) == SURFACE.upper(), "a glyph remover at rest is a ghost"
    assert face(labelled) == RED_TINT.upper()
    assert glyph.toolTip() == labelled.toolTip() == "Click twice to remove this node"
    width = labelled.width()
    for button in (glyph, labelled):
        click(button)
        assert confirmed == [] and button.is_armed()
        assert button.role() == "destructive_primary" and face(button) == RED_DARK.upper()
        assert button.toolTip() == "Click again to remove this node"
        assert labelled.width() == width, "arming moves nothing"
        click(button)
        assert confirmed == [button], confirmed
        assert button.role() == "destructive" and not button.is_armed()
        assert button.toolTip() == "Click twice to remove this node"
        confirmed.clear()
    assert face(glyph) == SURFACE.upper() and face(labelled) == RED_TINT.upper()

    # Left alone past the interval, disabled, or hidden: back at rest, and
    # the next click arms again rather than removing.
    click(glyph)
    QtTest.QTest.qWait(arm_interval_ms() + 50)
    assert not glyph.is_armed() and glyph.role() == "destructive"
    click(glyph)
    glyph.setEnabled(False)
    assert not glyph.is_armed()
    glyph.setEnabled(True)
    click(glyph)
    glyph.hide()
    assert not glyph.is_armed()
    glyph.show()
    click(glyph)
    assert confirmed == [] and glyph.is_armed()
    glyph.disarm()

    # A double-click -- press, release, double-click, release -- removes.
    centre = QtCore.QPointF(glyph.rect().center())
    for kind, held in ((QtCore.QEvent.MouseButtonPress, QtCore.Qt.LeftButton),
                       (QtCore.QEvent.MouseButtonRelease, QtCore.Qt.NoButton),
                       (QtCore.QEvent.MouseButtonDblClick, QtCore.Qt.LeftButton),
                       (QtCore.QEvent.MouseButtonRelease, QtCore.Qt.NoButton)):
        QtWidgets.QApplication.sendEvent(glyph, QtGui.QMouseEvent(
            kind, centre, QtCore.Qt.LeftButton, held, QtCore.Qt.NoModifier))
    assert confirmed == [glyph], confirmed

    # Not told to confirm twice: one click acts, and it never arms.
    click(plain)
    assert clicked == [plain] and not plain.is_armed() and plain.role() == "destructive"


def _ink_rows(widget, test, left: float = 0.0, right: float | None = None) -> list[int]:
    """The device rows of ``widget``'s own grab, between logical ``left``
    and ``right``, that hold a pixel ``test`` accepts."""

    from PyQt5 import QtGui

    image = widget.grab().toImage()
    ratio = image.devicePixelRatio() or 1.0
    right = widget.width() if right is None else right
    return [
        y for y in range(image.height())
        if any(test(QtGui.QColor(image.pixel(x, y)))
               for x in range(int(left * ratio), min(image.width(), int(right * ratio))))
    ]


def test_the_step_chip_is_centred_on_the_frame(app, host) -> None:
    """The pill and its text stand as far from the top edge as from the
    bottom -- on the frame's inner rect, not on the digits' baseline."""

    from PyQt5 import QtCore
    from zlc_ui.fluent import FluentDoubleSpinBox, frame_inner_rect
    from zlc_ui.fluent.style import ACCENT_TINT

    box = FluentDoubleSpinBox(host)
    box.setDecimals(3)
    box.setSingleStep(0.1)
    box.setValue(1.5)
    box.resize(170, box.sizeHint().height())
    _shown(app, host, box)
    chip = box._step_chip
    assert chip.isVisible()
    inner = frame_inner_rect(box)
    pill = chip.pill().translated(chip.pos())
    ratio = box.devicePixelRatioF()
    assert abs(pill.center().y() - inner.center().y()) * ratio <= 0.5, (pill, inner)

    chip.setAttribute(QtCore.Qt.WA_UnderMouse, True)
    chip.update()
    app.processEvents()
    tint = ACCENT_TINT.upper()
    washed = _ink_rows(box, lambda c: c.name().upper() == tint, chip.x(), chip.x() + chip.width())
    top, bottom = inner.top() * ratio, inner.bottom() * ratio
    assert washed and abs((washed[0] - top) - (bottom - 1 - washed[-1])) <= 1, (washed, top, bottom)
    inked = _ink_rows(box, lambda c: c.lightness() < 150, chip.x(), chip.x() + chip.width())
    assert abs((inked[0] - washed[0]) - (washed[-1] - inked[-1])) <= 1, "the text is centred in the pill"
    # Held down, the pill is the SELECTED fill every press is.
    from zlc_ui.fluent.style import SELECTED
    chip.setDown(True)
    app.processEvents()
    held = _ink_rows(box, lambda c: c.name().upper() == SELECTED.upper(), chip.x(), chip.x() + chip.width())
    assert held == washed, (held, washed)
    chip.setDown(False)


def _badge_box(frame, button, colours) -> tuple[list[int], list[int]]:
    """The device columns and rows of ``frame``'s grab, inside ``button``,
    painted in one of ``colours``."""

    image = frame.grab().toImage()
    ratio = image.devicePixelRatio() or 1.0
    inked = [
        (x, y)
        for x in range(int(button.x() * ratio), int((button.x() + button.width()) * ratio))
        for y in range(image.height())
        if image.pixelColor(x, y).name().upper() in colours
    ]
    return [p[0] for p in inked], [p[1] for p in inked]


def test_a_binding_badge_keeps_one_gap_in_a_field_and_in_a_repeat_box(app, host) -> None:
    """The badge stands as far from the stepper of a box that adopted the
    field as from the edge of a plain field, centred on the frame, with the
    digits clear of it; two badges stack inside the frame with room above
    and below."""

    from zlc_ui.fluent import field_room_end, fluent_count_box, frame_inner_rect
    from zlc_ui.fluent.style import CONFIG_GREEN_DARK, ORANGE_DARK
    from zlc_ui.pulse.scan_line_edit import FluentScanLineEdit

    plain = FluentScanLineEdit("12", host)
    plain.resize(130, plain.sizeHint().height())
    box = fluent_count_box(minimum=1, parent=host)
    box.setLineEdit(FluentScanLineEdit(""))
    box.move(0, 50)
    box.resize(130, plain.height())
    box.setValue(5)
    for edit in (plain, box.lineEdit()):
        edit.set_field_state(editable=True, source="config")
    _shown(app, host, plain, box)
    green = {CONFIG_GREEN_DARK.upper()}

    def gaps(frame, edit):
        button = edit.binding_button
        assert button.parentWidget() is frame, "seated in the frame that is drawn"
        xs, ys = _badge_box(frame, button, green)
        ratio = frame.devicePixelRatioF()
        inner = frame_inner_rect(frame)
        return (field_room_end(frame) * ratio - (max(xs) + 1),
                min(ys) - inner.top() * ratio, inner.bottom() * ratio - (max(ys) + 1), min(xs))

    plain_gap, plain_top, plain_bottom, _left = gaps(plain, plain)
    box_gap, box_top, box_bottom, box_left = gaps(box, box.lineEdit())
    assert plain_gap == box_gap, (plain_gap, box_gap)
    assert abs(plain_top - plain_bottom) <= 1 and abs(box_top - box_bottom) <= 1
    edit = box.lineEdit()
    ratio = box.devicePixelRatioF()
    text_end = (edit.x() + edit.width() - edit.textMargins().right()) * ratio
    assert text_end <= box_left, "the digits end before the badge"

    edit.set_field_state(editable=True, scan=True, source="config")
    app.processEvents()
    _xs, rows = _badge_box(box, edit.binding_button, green | {ORANGE_DARK.upper()})
    inner = frame_inner_rect(box)
    above, below = rows[0] - inner.top() * ratio, inner.bottom() * ratio - 1 - rows[-1]
    assert above >= 1 and below >= 1 and abs(above - below) <= 1, (above, below)


def test_segments_are_rounded_fills_with_no_outline(app, host) -> None:
    """The track is DIVIDER with no edge, the current face SELECTED with no
    outline, both rounded; a withheld choice is quiet and skipped, and a
    disabled control keeps its current face as a neutral fill."""

    from zlc_ui.fluent import FluentSegmentedButtons
    from zlc_ui.fluent import fluent as F
    from zlc_ui.fluent.style import BG, DIVIDER, SELECTED

    segments = FluentSegmentedButtons((("default", "Default"), ("api", "API"), ("config", "Config")), host)
    segments.set_current("config")
    segments.button("api").setEnabled(False)
    _shown(app, host, segments)
    assert segments.current() == "config"
    current = segments.button("config")
    assert _at(segments, segments.width() / 2.0, 0) == DIVIDER.upper(), "the track has no edge"
    assert _at(segments, 0, 0) != DIVIDER.upper(), "and is rounded"
    assert _at(current, current.width() / 2.0, 0) == SELECTED.upper(), "no outline on the current"
    assert _at(current, 0, 0) != SELECTED.upper(), "its corner is rounded"
    radius = F._radius()
    assert _at(current, radius, radius) == SELECTED.upper(), "at the field radius, not a near-square"
    withheld = segments.button("api")
    assert not withheld.isEnabled()
    assert _at(withheld, 4, withheld.height() / 2.0) == DIVIDER.upper(), "a withheld choice has no face"
    segments.setEnabled(False)
    app.processEvents()
    assert _at(segments, segments.width() / 2.0, 0) == BG.upper()
    assert _at(current, 4, current.height() / 2.0) == DIVIDER.upper()
    segments.set_current(None)
    assert segments.current() is None


def test_a_tab_close_x_stands_on_its_label_line(app, host) -> None:
    """A closable tab's x is centred on its label's capitals, at a scale
    whose label band is even and at one whose band is odd, whichever tab
    is current.  Seated by the style on the middle of the whole tab,
    underline included, it stood 0.8 px under the words at a ratio of 3:
    the button is now the tab's height, seated on its top, and its disk
    is centred where the style puts the label's capitals."""

    from PyQt5 import QtCore, QtWidgets
    from zlc_ui.fluent import FluentTabWidget, set_fluent_scale
    from zlc_ui.fluent import fluent as F

    def ink_middle(image, rect, band, ratio):
        rows = [y for y in range(int(rect.top() * ratio), int((rect.top() + band) * ratio))
                if any(image.pixelColor(x, y).lightness() < 200
                       for x in range(int(rect.left() * ratio), int((rect.right() + 1) * ratio)))]
        return (rows[0] + rows[-1] + 1) / 2.0 / ratio

    shown = F._FLUENT_SCALE
    bands = set()
    try:
        for scale in sorted({shown, 0.82, 0.9}):
            set_fluent_scale(scale)
            tabs = FluentTabWidget(host)
            tabs.add_permanent_tab(QtWidgets.QLabel(""), "Monitor")
            tabs.add_closable_tab(QtWidgets.QLabel(""), "Edit", focus=False)
            tabs.resize(400, 100)
            _shown(app, host, tabs)
            bar = tabs.tabBar()
            close = bar.tabButton(1, QtWidgets.QTabBar.RightSide)
            band = F._tab_label_height()
            bands.add(band % 2)
            image = host.grab().toImage()
            ratio = image.devicePixelRatio() or 1.0
            origin = bar.mapTo(host, QtCore.QPoint())
            # "Monitor": capitals and no descender, above its underline.
            label = ink_middle(image, bar.tabRect(0).translated(origin), band, ratio)
            x = ink_middle(image, QtCore.QRect(close.mapTo(host, QtCore.QPoint()), close.size()), band, ratio)
            assert abs(x - label) <= 0.5, (scale, x, label)
            # Where the style puts the capitals is where the disk is centred.
            assert abs(origin.y() + F._tab_cap_centre() - label) <= 0.5, (scale, label)
            for current in (0, 1):
                tabs.setCurrentIndex(current)
                app.processEvents()
                tab = bar.tabRect(1)
                assert (close.y(), close.height()) == (tab.top(), tab.height()), (scale, current)
                centre = close.y() + close.disk_rect().center().y()
                assert abs(centre - (tab.top() + F._tab_cap_centre())) <= 0.5 / ratio, (scale, centre)
            tabs.setParent(None)
            tabs.deleteLater()
            app.processEvents()
    finally:
        set_fluent_scale(shown)
    assert bands == {0, 1}, "both parities of the label band were measured"

def test_the_tri_switch_is_gone() -> None:
    import zlc_ui.fluent as fluent
    from zlc_ui.fluent import fluent as module

    assert "FluentTriSwitch" not in fluent.__all__
    assert not hasattr(module, "FluentTriSwitch")


def test_parts_inside_a_field_are_centred_on_its_ink_and_frame(app, host) -> None:
    """A tab's close x on its disk, and a spin box's two chevrons about the
    middle of an inner rect of either parity."""

    from PyQt5 import QtCore, QtWidgets
    from zlc_ui.fluent import FluentSpinBox, FluentTabWidget, frame_inner_rect, stepper_column

    tabs = FluentTabWidget(host)
    tabs.add_permanent_tab(QtWidgets.QLabel("a"), "Monitor")
    tabs.add_closable_tab(QtWidgets.QLabel("b"), "Edit")
    tabs.resize(300, 100)
    _shown(app, host, tabs)
    close = tabs.tabBar().tabButton(1, QtWidgets.QTabBar.RightSide)
    close.setAttribute(QtCore.Qt.WA_UnderMouse, True)
    close.update()
    app.processEvents()
    image = close.grab().toImage()
    ratio = image.devicePixelRatio() or 1.0
    dark = [(x, y) for x in range(image.width()) for y in range(image.height())
            if image.pixelColor(x, y).lightness() < 150]
    disk = close.disk_rect()
    left, top, across = disk.left() * ratio, disk.top() * ratio, disk.width() * ratio
    xs, ys = [p[0] for p in dark], [p[1] for p in dark]
    assert abs((min(ys) - top) - (top + across - 1 - max(ys))) <= 1
    assert abs((min(xs) - left) - (left + across - 1 - max(xs))) <= 1
    tabs.hide()

    for height in (30, 29):
        spin = FluentSpinBox(host)
        spin.setRange(0, 10)
        spin.setValue(5)
        spin.setFixedSize(90, height)
        _shown(app, host, spin)
        column = stepper_column(spin)
        inner = frame_inner_rect(spin)
        ratio = spin.devicePixelRatioF()
        top, bottom = inner.top() * ratio, inner.bottom() * ratio - 1
        rows = [y for y in _ink_rows(spin, lambda c: c.lightness() < 190 and abs(c.red() - c.blue()) < 12,
                                     column.left() + 2, column.right() - 2)
                if top <= y <= bottom]
        assert rows[0] - top == bottom - rows[-1], (height, rows[0] - top, bottom - rows[-1])
        spin.hide()


def test_a_checkbox_label_too_wide_for_its_room_ends_in_an_ellipsis(host) -> None:
    """A port name in a narrow card ends in '…' with the whole name in the
    tooltip; it is never cut mid-letter at the card's edge, and a label
    that fits keeps no tooltip."""

    from PyQt5 import QtWidgets
    from zlc_ui.fluent import FluentCheckBox

    layout = QtWidgets.QVBoxLayout(host)
    layout.setContentsMargins(0, 0, 0, 0)
    long_box = FluentCheckBox("pgc_1D_freq_switch", host)
    short_box = FluentCheckBox("cooling", host)
    layout.addWidget(long_box)
    layout.addWidget(short_box)
    host.setFixedWidth(long_box.sizeHint().width() - 20)
    host.show()
    QtWidgets.QApplication.processEvents()
    long_box.grab()
    short_box.grab()
    assert long_box.width() < long_box.sizeHint().width()
    assert long_box.minimumSizeHint().width() < long_box.width()
    assert long_box._shown_option().text.endswith("…")
    assert long_box.toolTip() == "pgc_1D_freq_switch"
    assert long_box.text() == "pgc_1D_freq_switch"
    assert short_box._shown_option().text == "cooling" and short_box.toolTip() == ""


def test_a_plain_container_in_a_white_card_shows_the_card_not_the_window_grey(app) -> None:
    """The window sheet paints BG on the window alone: a plain container in a
    card -- an Auto row's switch-and-field cell, the Save figure name row --
    shows the card's white, where one rule for every QWidget painted it grey."""

    from PyQt5 import QtGui, QtWidgets
    from zlc_ui.fluent import FluentGroupBox, FluentSwitch, fluent_widget_stylesheet
    from zlc_ui.fluent.style import BG, SURFACE

    root = QtWidgets.QWidget()
    root.setObjectName("SheetRoot")
    root.setStyleSheet(fluent_widget_stylesheet(root.objectName()))
    root.resize(400, 200)
    outer = QtWidgets.QVBoxLayout(root)
    outer.setContentsMargins(20, 20, 20, 20)
    card = FluentGroupBox("Card", root)
    outer.addWidget(card)
    inner = QtWidgets.QVBoxLayout(card)
    holder = QtWidgets.QWidget(card)
    holder.setMinimumHeight(40)
    row = QtWidgets.QHBoxLayout(holder)
    row.setContentsMargins(0, 0, 0, 0)
    row.addWidget(FluentSwitch("Auto", holder))
    row.addStretch(1)
    inner.addWidget(holder)
    root.show()
    QtWidgets.QApplication.processEvents()
    image = root.grab().toImage()
    ratio = image.devicePixelRatio()
    corner = holder.mapTo(root, holder.rect().topRight())
    probe = QtGui.QColor(image.pixel(int((corner.x() - 4) * ratio), int((corner.y() + 4) * ratio))).name().upper()
    window = QtGui.QColor(image.pixel(int(5 * ratio), int(5 * ratio))).name().upper()
    root.close()
    assert probe == SURFACE.upper(), probe
    assert window == BG.upper(), window
