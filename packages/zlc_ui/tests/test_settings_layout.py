"""The panel Setting popup lays out from measured truth, not guesses.

Two defects are pinned here, both found by measuring real geometry:

* the Auto switch's painted track was sized by a second, disagreeing width
  formula in the form (a hand padding constant), so it drew and hit-tested
  underneath the editor control beside it;
* the popup's width was guessed with a magic pad while the scroll body was
  pinned to the form's minimum width -- when the guess undershot the real
  chrome (left pad + frame + scrollbar), the body sat wider than the viewport
  with the horizontal bar forced off, clipping every row's right edge.
"""

from __future__ import annotations

import os
import subprocess
import sys

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


#: Every snippet starts here.  Without it the subprocess resolves the
#: layers through whatever the editable install points at -- on this
#: machine, sibling checkouts of the same package names -- so the suite
#: silently tested a DIFFERENT zlc_plot than the one beside it.  The
#: product bootstrap is what puts this checkout's layers on the path,
#: and it is the same one every launcher uses.
_BOOTSTRAP = "import zou_lab_control" + chr(10)


def _run_qt(code: str) -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = (
        "" if environment.get("ZLC_TEST_INSTALLED") == "1"
        else os.pathsep.join((str(REPO_ROOT), str(SRC)))
    )
    environment["QT_QPA_PLATFORM"] = "offscreen"
    completed = subprocess.run(
        [sys.executable, "-c", _BOOTSTRAP + code],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=40,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout


_SURFACE_PROLOGUE = """
import zou_lab_control
from PyQt5 import QtCore
from zlc_ui.qt import ensure_qt_app
from zlc_ui.console.panel_card_view import PanelCardView
app = ensure_qt_app(['settings-layout'])

def parameter_surface(count):
    display = []
    for index in range(count):
        display.append({
            'key': f'param_{index}',
            'label': f'A rather long display parameter label {index}',
            'kind': 'float', 'value': 1.0 * index, 'allow_none': True,
            'choices': (), 'minimum': 0.0, 'maximum': 100.0, 'step': 0.5,
            'automatic': True, 'unavailable_reason': '',
        })
    return {
        'semantic': (), 'display': tuple(display), 'fit': (),
        'semantic_unavailable': '', 'display_unavailable': '',
        'fit_unavailable': '',
    }

card = PanelCardView('panel-1', 'Camera')
card.set_size_choices(('2x2',), '2x2')
card.set_interval_choices((100, 200, 400, 800), 400)
card.set_signal_choices(
    (('camera-1', (('frames  [96x128]', '@logic/camera-1/frames'),)),),
    current='@logic/camera-1/frames',
)
state = {
    'signal': '@logic/camera-1/frames', 'kind': 'image', 'size': '2x2',
    'interval_ms': 100, 'title': 'Camera',
    'semantic': {}, 'display': {}, 'fit': {}, 'overlay_signal': '',
}
card.set_panel_projection(state, parameter_surface(8))
# The Setting frame is a child overlay clipped to the page it lives in --
# the nearest scroll viewport, exactly as in the real console.  A bare
# card would be its own (panel-sized) page, so these tests mount the card
# the way the console does and give the page the room a real panel area
# has: they measure the FORM's layout truth, not the page wall.
from PyQt5 import QtWidgets
area = QtWidgets.QScrollArea()
area.setWidgetResizable(True)
holder = QtWidgets.QWidget()
holder_layout = QtWidgets.QVBoxLayout(holder)
holder_layout.addWidget(card)
holder_layout.addStretch(1)
area.setWidget(holder)
area.resize(920, 760)
area.show()
app.processEvents()
card._open_settings()
app.processEvents()
form = card._settings_form
body = card._settings_body
viewport = card._settings_scroll.viewport()
"""

_SURFACE_EPILOGUE = """
popup = card._settings_popup
card.retire_settings_popup()
if popup is not None:
    popup.close()
card.close()
card.deleteLater()
app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
app.processEvents()
"""


def test_the_switch_track_never_paints_under_its_neighbour() -> None:
    _run_qt(
        _SURFACE_PROLOGUE
        + """
from zlc_ui.fluent import fluent_switch_width
# One width authority: the column reserved for an Auto switch is what the
# switch itself will paint, and the paint never exceeds the widget's cell.
for key, switch in form._auto_switches.items():
    row = form._rows[key]
    control = form._widgets[key]
    assert switch.width() >= fluent_switch_width(switch.text()), (
        key, switch.width(), fluent_switch_width(switch.text()))
    switch_right = switch.mapTo(row, QtCore.QPoint(switch._content_width(), 0)).x()
    control_left = control.mapTo(row, QtCore.QPoint(0, 0)).x()
    assert switch_right <= control_left, (key, switch_right, control_left)
# A deliberately squeezed switch clips its own track instead of overflowing.
squeezed = next(iter(form._auto_switches.values()))
squeezed.setFixedWidth(40)
app.processEvents()
assert squeezed._content_width() <= squeezed.width()
"""
        + _SURFACE_EPILOGUE
    )


def test_the_settings_body_is_never_clipped_by_the_viewport() -> None:
    _run_qt(
        _SURFACE_PROLOGUE
        + """
# The body is a width CONSUMER: nothing pins its minimum, and the popup is
# sized from the form's measured requirement plus the popup's real chrome --
# so the viewport always spans the body and no row loses its right edge.
assert body.minimumWidth() == 0
assert body.width() <= viewport.width(), (body.width(), viewport.width())
margins = body.layout().contentsMargins()
required = form.minimum_content_width() + margins.left() + margins.right()
assert viewport.width() >= required, (viewport.width(), required)
rows = [form._rows[key] for key in form.spec.keys]
rects = [QtCore.QRect(row.mapTo(body, QtCore.QPoint(0, 0)), row.size()) for row in rows]
for index, rect in enumerate(rects):
    assert rect.right() <= body.width(), (form.spec.keys[index], rect)
    for other in rects[index + 1:]:
        assert not rect.intersects(other), (rect, other)
"""
        + _SURFACE_EPILOGUE
    )


def test_reprojection_while_open_reflows_without_clipping() -> None:
    _run_qt(
        _SURFACE_PROLOGUE
        + """
# A live projection replacement while the popup is open (the beat does this)
# re-measures and re-presents through the card's one placement call.
card.set_panel_projection(state, parameter_surface(14))
app.processEvents()
app.processEvents()
form = card._settings_form
body = card._settings_body
viewport = card._settings_scroll.viewport()
assert body.minimumWidth() == 0
assert body.width() <= viewport.width(), (body.width(), viewport.width())
margins = body.layout().contentsMargins()
required = form.minimum_content_width() + margins.left() + margins.right()
assert viewport.width() >= required, (viewport.width(), required)
"""
        + _SURFACE_EPILOGUE
    )


def test_reconcile_replaces_the_enabled_when_dependency_graph() -> None:
    _run_qt(
        """
import zou_lab_control
from zlc_ui.qt import ensure_qt_app
from zlc_ui.form import FormFieldProps, FormSpec
from zlc_ui.form.qt_form import FluentParameterForm
app = ensure_qt_app(['form-dependency-reconcile'])
old = FormSpec((
    FormFieldProps('a', 'bool', 'A', default=True),
    FormFieldProps(
        'value', 'bool', 'Value', default=True,
        enabled_when=('a', (True,)),
    ),
))
form = FluentParameterForm(old, {'a': True, 'value': True})
assert form.widget_for('value').isEnabled()
new = FormSpec((
    FormFieldProps('c', 'bool', 'C', default=False),
    FormFieldProps(
        'value', 'bool', 'Value', default=True,
        enabled_when=('c', (True,)),
    ),
))
form.reconcile(new, {'c': False, 'value': True})
assert not form.widget_for('value').isEnabled()
controller = form.widget_for('c')
controller.setChecked(True)
form.changed.emit('c')
assert form.widget_for('value').isEnabled()
typed = FormSpec((
    FormFieldProps('flag', 'bool', 'Flag', default=True),
    FormFieldProps(
        'typed', 'bool', 'Typed', default=True,
        enabled_when=('flag', (1,)),
    ),
))
form.reconcile(typed, {'flag': True, 'typed': True})
assert not form.widget_for('typed').isEnabled(), 'bool True is not integer 1'
negative = FormSpec((
    FormFieldProps(
        'offset', 'float', 'Offset', default=None,
        maximum=-2.0, automatic=True,
    ),
))
form = FluentParameterForm(negative, {'offset': None})
form.auto_switch_for('offset').setChecked(False)
assert form.read_value('offset') == -2.0, (
    'leaving Auto must choose a value inside a negative-only domain'
)
# Switching a text label back to Auto publishes None against a declaration
# still carrying the typed title as its default.  Whether the field may be
# empty is the field's own declaration; reading it off that default raised
# out of the Qt slot that published the panel, which aborts the process.
titled = FormSpec((
    FormFieldProps('title', 'text', 'Title', default='mot camera', automatic=True),
))
form = FluentParameterForm(titled, {'title': 'mot camera'})
automatic = FormSpec((
    FormFieldProps('title', 'text', 'Title', default=None, automatic=True),
))
assert form.adopt_projection(automatic, {'title': None}) is False
form.reconcile(automatic, {'title': None})
assert form.auto_switch_for('title').isChecked()
"""
    )


def test_compound_choice_only_cycles_its_large_domain_when_focused() -> None:
    """Scope stays one popup fate; its real coordinates belong to the wheel."""

    _run_qt(
        """
import zou_lab_control
from PyQt5 import QtCore, QtGui, QtWidgets
from zlc_ui.qt import ensure_qt_app
from zlc_ui.form import FormChoice, FormFieldProps, FormSpec
from zlc_ui.form.qt_form import FluentParameterForm
from zlc_ui.fluent import FluentCycleComboBox

app = ensure_qt_app(['scope-cycle'])
cycles = tuple((("scope-value", value), str(value)) for value in range(1024))
field = FormFieldProps(
    'fate:y', 'choice', 'spatial-y', default='reduce', required=True,
    choices=(FormChoice('(reduced)', 'reduce'), FormChoice('Y axis', 'y')),
    cycle_choices=cycles, cycle_label='Scope',
)
form = FluentParameterForm(FormSpec((field,)), {'fate:y': 'reduce'})
combo = form.widget_for('fate:y')
assert isinstance(combo, FluentCycleComboBox)
assert combo.count() == 3, '1024 coordinates must not become popup rows'
assert combo.itemText(2) == 'Scope'

body = QtWidgets.QWidget()
layout = QtWidgets.QVBoxLayout(body)
layout.addWidget(form)
layout.addStretch()
body.setMinimumHeight(1200)
scroll = QtWidgets.QScrollArea()
scroll.setWidget(body)
scroll.resize(420, 240)
scroll.show()
app.processEvents()
bar = scroll.verticalScrollBar()
assert bar.maximum() > 0

def send(delta):
    event = QtGui.QWheelEvent(
        QtCore.QPointF(combo.rect().center()),
        QtCore.QPointF(combo.mapToGlobal(combo.rect().center())),
        QtCore.QPoint(), QtCore.QPoint(0, delta),
        QtCore.Qt.NoButton, QtCore.Qt.NoModifier,
        QtCore.Qt.NoScrollPhase, False,
    )
    QtWidgets.QApplication.sendEvent(combo, event)
    app.processEvents()
    return event.isAccepted()

# Even with focus, an ordinary fate never consumes the page wheel.
combo.setFocus(QtCore.Qt.MouseFocusReason)
assert not send(-120)
assert form.read_value('fate:y') == 'reduce'

# Scope begins at the first real coordinate.  Selecting it through the real
# popup commit returns focus to the collapsed control, so the next wheel
# notch can advance it without an extra click that would reopen the popup.
scroll.activateWindow()
app.processEvents()
combo._commit_flat_index(combo.model().index(2, 0))
app.processEvents()
assert combo.currentText() == 'Scope: 0'
assert combo.hasFocus()
seen = []
form.changed.connect(seen.append)
assert send(-120)
assert form.read_value('fate:y') == ('scope-value', 1)
assert combo.currentText() == 'Scope: 1'
assert seen == ['fate:y']

# Choosing Scope changes the plot vocabulary while the combo still owns
# focus: X/Y may disappear from this row when those roles move elsewhere.
# Reconcile must project the accepted Scope value, not let rebuilding the
# ordinary popup rows reset the focused combo to Reduced.
changed_field = FormFieldProps(
    'fate:y', 'choice', 'spatial-y', default=('scope-value', 1), required=True,
    choices=(FormChoice('(reduced)', 'reduce'),),
    cycle_choices=cycles, cycle_label='Scope',
)
form.reconcile(
    FormSpec((changed_field,)),
    {'fate:y': ('scope-value', 1)},
)
assert form.read_value('fate:y') == ('scope-value', 1)
assert combo.currentText() == 'Scope: 1'

# The same wheel without focus is left to the containing page.
scroll.setFocus(QtCore.Qt.MouseFocusReason)
assert not send(-120)
assert form.read_value('fate:y') == ('scope-value', 1)
scroll.close()
"""
    )


def test_every_field_applies_as_it_is_typed_and_none_is_written_over() -> None:
    """One rule for the whole form, and the operator owns their cursor.

    Text was the single commit-on-defocus kind in the registry -- a Y label
    applied when you clicked away while the colour maximum beside it applied
    per keystroke, so one popup answered two rules depending on the row.

    Live editing then needs the other half.  Every keystroke round-trips
    through the owner and comes back as a projection, and reconcile writes
    projections into widgets: written into the box being typed in, that
    installs a value the operator never typed and can disable the box
    mid-word when an emptied optional field flips to Auto.
    """

    _run_qt(
        """
import zou_lab_control
from PyQt5 import QtCore, QtTest, QtWidgets
from zlc_ui.qt import ensure_qt_app
from zlc_ui.form.form import FormFieldProps, FormSpec
from zlc_ui.form.qt_form import FluentParameterForm, FORM_WIDGET_HANDLERS

app = ensure_qt_app(['test'])

# Every kind the panel Setting popup can produce.
LIVE_KINDS = ('text', 'number', 'int', 'bool', 'choice')
for kind in LIVE_KINDS:
    assert kind in FORM_WIDGET_HANDLERS, kind

spec = FormSpec((
    FormFieldProps('label', 'text', 'Y label'),
    FormFieldProps('ceiling', 'number', 'Value maximum'),
))
values = {'label': 'start', 'ceiling': 100.0}
form = FluentParameterForm(spec, values)
window = QtWidgets.QMainWindow()
window.setCentralWidget(form)
window.show()
app.processEvents()

seen = []
form.changed.connect(seen.append)

# TEXT applies as it is typed, exactly as the number beside it does.
label = form._widgets['label']
label.setFocus(QtCore.Qt.MouseFocusReason)
label.selectAll()
QtTest.QTest.keyClicks(label, 'ab')
app.processEvents()
assert seen.count('label') >= 2, seen

# And the owner writing a projection back does not touch the box the
# operator is inside: this is the half-typed value the round trip would
# otherwise normalise away.
seen.clear()
ceiling = form._widgets['ceiling']
ceiling.setFocus(QtCore.Qt.MouseFocusReason)
ceiling.selectAll()
QtTest.QTest.keyClicks(ceiling, '0')
app.processEvents()
assert ceiling.hasFocus()
form.reconcile(spec, {'label': 'ab', 'ceiling': 100.0})
app.processEvents()
assert ceiling.text() == '0', (
    'the form wrote the stored value over the operator: %r' % ceiling.text()
)
# The field they are NOT in still takes the projection.
assert form._widgets['label'].text() == 'ab', form._widgets['label'].text()

# Leave the field, and the projection lands.
form._widgets['label'].setFocus(QtCore.Qt.MouseFocusReason)
app.processEvents()
form.reconcile(spec, {'label': 'ab', 'ceiling': 100.0})
app.processEvents()
assert ceiling.text() == '100.0', ceiling.text()
window.close()
print('ok')
"""
    )


def test_a_form_that_grows_a_row_does_not_scroll_itself() -> None:
    """Adding a control must not move the controls already on screen.

    Choosing a fit model adds one row, and the whole form jumped: every
    label the operator was reading slid down and the combo they had just
    used ended up somewhere else.  Nothing inside the form moved -- the
    popup placement resized the popup to its UNBOUNDED size hint as a way
    of measuring it, and for that instant the scroll viewport was tall
    enough that the vertical range collapsed and Qt clamped the operator's
    scroll position into it.  Restoring the real size restored the range,
    not the position, so every edit walked the form a little further.
    """

    _run_qt(
        _SURFACE_PROLOGUE
        + """
scroll = card._settings_scroll
bar = scroll.verticalScrollBar()

# Fill the popup past its own height, then look at the bottom -- the only
# place from which a lost scroll position is visible.
card.set_panel_projection(state, parameter_surface(14))
app.processEvents()
bar.setValue(bar.maximum())
app.processEvents()
assert bar.maximum() > 0, 'the form must overflow for this to mean anything'

before_value = bar.value()
before_max = bar.maximum()
before_geometry = card._settings_popup.geometry()
before_rows = {
    key: widget.mapTo(form, widget.rect().topLeft()).y()
    for key, widget in form._widgets.items()
}

# APPEND-ONLY, not re-place-everything.  The layout comparison used to ask
# whether the placed rows matched the whole wanted list -- which a row that
# has been built but not yet inserted can never satisfy -- so every append
# pulled all of them out and put them back.
inserted = []
real_insert = form._layout.insertWidget
form._layout.insertWidget = lambda index, widget, *a, **k: (
    inserted.append(index), real_insert(index, widget, *a, **k))[1]

card.set_panel_projection(state, parameter_surface(15))
app.processEvents()
form._layout.insertWidget = real_insert
assert len(inserted) == 1, inserted

after_rows = {
    key: widget.mapTo(form, widget.rect().topLeft()).y()
    for key, widget in form._widgets.items()
}
added = set(after_rows) - set(before_rows)
assert len(added) == 1, added

# The operator's position, untouched.
assert bar.value() == before_value, (bar.value(), before_value)
# ... while the range really did grow, so this is not a no-op test.
assert bar.maximum() > before_max, (bar.maximum(), before_max)

# Every row that was already there is exactly where it was, and the new
# one is below all of them.
for key, y in before_rows.items():
    assert after_rows[key] == y, (key, y, after_rows[key])
new_key = next(iter(added))
assert after_rows[new_key] > max(before_rows.values()), (
    new_key, after_rows[new_key], max(before_rows.values()))
"""
        + _SURFACE_EPILOGUE
    )


def test_a_short_form_keeps_its_pitch_when_it_grows() -> None:
    """A row's position depends on the rows above it and nothing else.

    A form shorter than its host had no trailing slack, so the surplus
    height was shared out along the column and the PITCH depended on the row
    COUNT.  Adding one control moved every control already on screen upward,
    the lowest by the most -- with no scrolling involved at all, so this is a
    second, independent way the same complaint appears.
    """

    _run_qt(
        _SURFACE_PROLOGUE
        + """
# Deliberately far short of the viewport: this is the regime where surplus
# height exists to be mis-shared.
card.set_panel_projection(state, parameter_surface(2))
app.processEvents()
scroll = card._settings_scroll
assert scroll.verticalScrollBar().maximum() == 0, 'must not overflow yet'

before = {
    key: widget.mapTo(form, widget.rect().topLeft()).y()
    for key, widget in form._widgets.items()
}
assert len(before) >= 3, before

card.set_panel_projection(state, parameter_surface(3))
app.processEvents()
after = {
    key: widget.mapTo(form, widget.rect().topLeft()).y()
    for key, widget in form._widgets.items()
}
assert len(after) == len(before) + 1, (len(before), len(after))
for key, y in before.items():
    assert after[key] == y, (key, y, after[key])
"""
        + _SURFACE_EPILOGUE
    )


def test_the_scrollbar_appearing_does_not_narrow_every_control() -> None:
    """Width-bounded content reflows to the viewport, so the viewport is fixed.

    The bar was AsNeeded with no reserved gutter, so the very row that
    pushed the content past the viewport also took the bar's width off every
    control in the form -- one more layout change caused by nothing but a
    control being added.

    Tested on the scroll area itself, at a FIXED outer size: mounted in the
    Setting popup the popup would legitimately re-measure its own width at
    the same moment, and the two effects are not separable there.
    """

    _run_qt(
        """
import zou_lab_control
from PyQt5 import QtWidgets
from zlc_ui.qt import ensure_qt_app
from zlc_ui.fluent import FluentScrollArea, FluentLabel
app = ensure_qt_app(['scroll-gutter'])

area = FluentScrollArea()
area.resize(300, 200)
body = QtWidgets.QWidget()
column = QtWidgets.QVBoxLayout(body)
column.setContentsMargins(0, 0, 0, 0)
rows = []
for index in range(3):
    label = FluentLabel(f'row {index}', body)
    label.setFixedHeight(20)
    column.addWidget(label)
    rows.append(label)
area.set_width_bounded_widget(body)
area.show()
app.processEvents()

assert area.verticalScrollBar().maximum() == 0, 'must not overflow yet'
narrow_viewport = area.viewport().width()
narrow_body = body.width()

# Grow the content until the bar is genuinely needed.  The outer widget is
# never resized, so any viewport change is the bar's doing and nothing else.
for index in range(3, 40):
    label = FluentLabel(f'row {index}', body)
    label.setFixedHeight(20)
    label.show()
    column.addWidget(label)
column.activate()
body.updateGeometry()
app.processEvents()
app.processEvents()
assert area.verticalScrollBar().maximum() > 0, 'the bar must be needed now'

assert area.viewport().width() == narrow_viewport, (
    narrow_viewport, area.viewport().width())
assert body.width() == narrow_body, (narrow_body, body.width())

area.close()
area.deleteLater()
app.processEvents()
"""
    )


def test_a_frame_carried_away_from_its_button_still_does_not_move() -> None:
    """The previous guard only ever tested the frame where it opens.

    Anchored, the frame is already pinned to its card-relative height cap and
    cannot grow, so nothing moved and the fix looked complete.  Carried
    somewhere else it was placed under a DIFFERENT height rule -- no cap at
    all -- and the first content change after the drag grew it by 131 px and
    moved every row 100 px.  A gesture that chooses a POSITION was deciding
    the SIZE.

    Placement now happens once, when the frame opens.  What is on screen
    stays where the operator put it, and a new control goes below.
    """

    _run_qt(
        _SURFACE_PROLOGUE
        + """
from PyQt5 import QtGui
card.set_panel_projection(state, parameter_surface(14))
app.processEvents()
popup = card._settings_popup
scroll = card._settings_scroll
bar = scroll.verticalScrollBar()
assert bar.maximum() > 0, 'the form must overflow for this to mean anything'
anchored = popup.geometry()

# Carry it off, exactly as the drag handle does.
handle = card._settings_drag_handle
start = popup.frameGeometry().topLeft() + QtCore.QPoint(20, 8)
for kind, offset in (
    (QtCore.QEvent.MouseButtonPress, QtCore.QPoint(0, 0)),
    (QtCore.QEvent.MouseMove, QtCore.QPoint(90, -60)),
    (QtCore.QEvent.MouseButtonRelease, QtCore.QPoint(90, -60)),
):
    where = start + offset
    card.eventFilter(handle, QtGui.QMouseEvent(
        kind, handle.mapFromGlobal(where), where,
        QtCore.Qt.LeftButton, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier))
app.processEvents()
carried = popup.geometry()
# The gesture must actually have carried it, or everything below is a
# test of the anchored case wearing a different name.
assert carried.topLeft() != anchored.topLeft(), (anchored, carried)

for position in (bar.maximum(), bar.maximum() // 2, 0):
    bar.setValue(position)
    app.processEvents()
    before_geometry = popup.geometry()
    before = {
        key: (widget.mapToGlobal(widget.rect().topLeft()).x(),
              widget.mapToGlobal(widget.rect().topLeft()).y())
        for key, widget in form._widgets.items()
    }
    before_max = bar.maximum()

    card.set_panel_projection(state, parameter_surface(15))
    app.processEvents()

    after = {
        key: (widget.mapToGlobal(widget.rect().topLeft()).x(),
              widget.mapToGlobal(widget.rect().topLeft()).y())
        for key, widget in form._widgets.items()
    }
    # The frame the operator carried there is untouched -- size AND place.
    assert popup.geometry() == before_geometry, (
        position, before_geometry, popup.geometry())
    assert popup.geometry() == carried, (position, carried, popup.geometry())
    # Nothing that was on screen moved ON SCREEN.
    for key, point in before.items():
        assert after[key] == point, (position, key, point, after[key])
    # ... and the new control really did arrive, below everything.
    added = set(after) - set(before)
    assert len(added) == 1, added
    assert after[next(iter(added))][1] > max(y for _x, y in before.values())
    assert bar.maximum() > before_max, (before_max, bar.maximum())

    card.set_panel_projection(state, parameter_surface(14))
    app.processEvents()
"""
        + _SURFACE_EPILOGUE
    )


def test_a_row_whose_key_left_the_spec_goes_even_under_the_cursor() -> None:
    """The form never holds a widget it cannot answer for.

    Keeping the focused row from being REBUILT is the rule; keeping it
    from being REMOVED is a different thing, and the exemption was written
    once and applied to both.  A focused key still in the spec never
    reaches the deletion loop, so the second exemption could only ever
    retain a row whose key had LEFT the spec -- while _fields and
    _handlers are rebuilt from the new spec alone.  The retained row was
    still laid out, still visible, still wired to the build-time
    changed(key); the next keystroke sent read_value into a KeyError, out
    of a Qt slot, and PyQt aborted the process without a traceback.

    Reachable by ordinary use: flipping paints_images drops
    'overlay_signal', flipping live drops 'interval_ms'.
    """

    _run_qt(
        """
import zou_lab_control
from PyQt5 import QtCore, QtTest, QtWidgets
from zlc_ui.qt import ensure_qt_app
from zlc_ui.form.form import FormFieldProps, FormSpec
from zlc_ui.form.qt_form import FluentParameterForm

app = ensure_qt_app(['test'])

spec = FormSpec((
    FormFieldProps('label', 'text', 'Y label'),
    FormFieldProps('overlay', 'text', 'Overlay signal'),
))
form = FluentParameterForm(spec, {'label': 'start', 'overlay': 'trap'})
window = QtWidgets.QMainWindow()
window.setCentralWidget(form)
window.show()
app.processEvents()

# Their cursor is in the row the owner is about to drop.
overlay = form._widgets['overlay']
overlay.setFocus(QtCore.Qt.MouseFocusReason)
overlay.selectAll()
QtTest.QTest.keyClicks(overlay, 'pr')
app.processEvents()
assert overlay.hasFocus()

smaller = FormSpec((FormFieldProps('label', 'text', 'Y label'),))
form.reconcile(smaller, {'label': 'start'})
app.processEvents()

assert set(form._widgets) == {'label'}, sorted(form._widgets)
assert set(form._rows) == {'label'}, sorted(form._rows)
# The invariant that makes the abort impossible: every widget still held
# is one the form can read.
for key in form._widgets:
    form.read_value(key)
window.close()
print('ok')
"""
    )


def test_a_growing_axis_does_not_rebuild_the_fate_popup() -> None:
    """One more shot is not a different form.

    A Scope cycle carries every coordinate its axis offers, so it grows
    with the data: one more shot, one more entry.  Compared as structure,
    that refused adoption and sent the form through reconcile, which
    refilled the combo -- ``clear()`` and re-add -- once per shot.  An
    open dropdown closed under the operator and the click already on its
    way was eaten, which is what "the facet option flickers, or clicking
    it does nothing" looked like from the outside.
    """

    from PyQt5 import QtWidgets

    from zlc_ui import ensure_qt_app
    from zlc_ui.form import FormChoice, FormFieldProps, FormSpec
    from zlc_ui.form.qt_form import FluentParameterForm

    ensure_qt_app(["zlc-ui-tests"])

    def spec(depth: int) -> FormSpec:
        return FormSpec(
            (
                FormFieldProps(
                    key="fate",
                    kind="choice",
                    label="source index",
                    choices=(
                        FormChoice("Reduce", "reduce"),
                        FormChoice("Facet", "facet"),
                    ),
                    cycle_label="Scope",
                    cycle_choices=tuple(
                        (offset, f"= {offset}")
                        for offset in range(-(depth - 1), 1)
                    ),
                ),
            )
        )

    form = FluentParameterForm(spec(5), {"fate": "facet"})
    widget = form.widget_for("fate")
    cleared = {"count": 0}
    original_clear = widget.clear

    def counted_clear() -> None:
        cleared["count"] += 1
        return original_clear()

    widget.clear = counted_clear

    adopted = form.adopt_projection(spec(6), {"fate": "facet"})
    assert adopted, "one more coordinate must not refuse adoption"
    assert form.widget_for("fate") is widget, "the control must survive"
    assert cleared["count"] == 0, "the popup must not be refilled"
    assert form.read_value("fate") == "facet", form.read_value("fate")

    # Growing it again through the full reconcile path is just as quiet.
    form.reconcile(spec(7), {"fate": "facet"})
    assert form.widget_for("fate") is widget
    assert cleared["count"] == 0, "reconcile refilled a popup that did not change"
    assert form.read_value("fate") == "facet", form.read_value("fate")

    # A genuinely different choice list still refills, exactly as before.
    changed = FormSpec(
        (
            FormFieldProps(
                key="fate",
                kind="choice",
                label="source index",
                choices=(
                    FormChoice("Reduce", "reduce"),
                    FormChoice("Facet", "facet"),
                    FormChoice("X axis", "x"),
                ),
                cycle_label="Scope",
                cycle_choices=((0, "= 0"),),
            ),
        )
    )
    form.reconcile(changed, {"fate": "facet"})
    assert cleared["count"] == 1, "a new choice list must refill the popup"


def test_the_setting_frame_says_the_panels_standing_condition() -> None:
    """What the card's dot says is written where the operator goes to fix it.

    A red dot with a tooltip names the panel; an operator who opens Setting
    to put it right needs the condition in front of them.  It heads the
    frame in the dot's colour, follows every change, and leaves with it.
    """

    _run_qt(_SURFACE_PROLOGUE + """
from zlc_ui.fluent import GREY, RED
card.set_status('its plot surface has been travelling for 37 s', error=True)
card._open_settings()
app.processEvents()
label = card._settings_status
assert label is not None
assert not label.isHidden()
assert label.text() == 'its plot surface has been travelling for 37 s'
assert RED.upper() in label.styleSheet().upper()
card.set_status('waiting for a signal', error=False)
assert label.text() == 'waiting for a signal'
assert GREY.upper() in label.styleSheet().upper()
card.set_status('', error=False)
app.processEvents()
assert label.isHidden() and label.text() == ''
print('setting status ok')
""")


def _counted_domain(size: int):
    """A lazy wheel domain that counts how often it is read."""

    from collections.abc import Sequence

    class Counted(Sequence):
        def __init__(self) -> None:
            self.reads = 0

        def __len__(self) -> int:
            return size

        def __getitem__(self, index):
            if not 0 <= index < size:
                raise IndexError(index)
            self.reads += 1
            return index, str(index)

    return Counted()


def test_a_scope_that_did_not_move_is_read_not_searched_for() -> None:
    """The wheel domain is DATA, read one position at a time.

    Every beat repeated the same coordinate, and every beat the form walked
    the whole axis three times to confirm it -- once to normalize, once to
    write, once to read back -- for a value the widget already held.  What
    the widget shows at its position IS a value of the domain; and a Scope
    action relabelled in place must keep its row rather than raise.
    """

    from zlc_ui import ensure_qt_app
    from zlc_ui.form import FormChoice, FormFieldProps, FormSpec
    from zlc_ui.form.qt_form import FluentParameterForm

    ensure_qt_app(["zlc-ui-tests"])

    def spec(domain, label: str = "Scope") -> FormSpec:
        return FormSpec(
            (
                FormFieldProps(
                    "fate",
                    "choice",
                    "source index",
                    choices=(FormChoice("Reduce", "reduce"),),
                    cycle_label=label,
                    cycle_choices=domain,
                ),
            )
        )

    domain = _counted_domain(1024)
    form = FluentParameterForm(spec(domain), {"fate": 1023})
    widget = form.widget_for("fate")
    domain.reads = 0
    assert form.read_value("fate") == 1023
    assert domain.reads == 1, "the value at the held position needs no confirming"
    domain.reads = 0
    assert form.adopt_projection(spec(domain), {"fate": 1023})
    assert domain.reads == 1, "the same projection is one read, not a walk"
    grown = _counted_domain(1025)
    domain.reads = grown.reads = 0
    assert form.adopt_projection(spec(grown), {"fate": 1023})
    assert grown.reads <= 3 and domain.reads <= 1, (grown.reads, domain.reads)
    grown.reads = 0
    form.populate({"fate": 1023})
    assert grown.reads <= len(grown) + 1, "at most one walk, then the widget answers"
    # A relabelled action keeps its row, its value and its widget.
    form.reconcile(spec(grown, "Slice"), {"fate": 1023})
    assert form.widget_for("fate") is widget
    assert widget.itemText(widget.count() - 1) == "Slice"
    assert widget.currentText() == "Slice: 1023"
    assert form.read_value("fate") == 1023


def test_a_coordinate_the_axis_just_grew_is_judged_by_the_vocabulary_it_came_with() -> None:
    """One projection, one moment: a value that arrives with a longer axis
    is judged against that axis.  Judged against the one already installed
    it was "not one of the typed choices" -- an exception instead of the
    False that hands the projection to reconcile, so the Card never reached
    the reconcile that would have shown it."""

    from dataclasses import replace

    from zlc_ui import ensure_qt_app
    from zlc_ui.form import FormChoice, FormFieldProps, FormSpec
    from zlc_ui.form.qt_form import FluentParameterForm

    ensure_qt_app(["zlc-ui-tests"])
    scoped = FormFieldProps(
        "fate", "choice", "Fate", default="reduce",
        choices=(FormChoice("Reduced", "reduce"),),
        cycle_label="Scope", cycle_choices=(("scope:0", "0"),),
    )
    form = FluentParameterForm(FormSpec((scoped,)), {"fate": "reduce"})
    grown = replace(scoped, cycle_choices=(("scope:0", "0"), ("scope:1", "1")))
    assert form.adopt_projection(FormSpec((grown,)), {"fate": "scope:1"}) is False
    form.reconcile(FormSpec((grown,)), {"fate": "scope:1"})
    assert form.read_value("fate") == "scope:1"
    more = replace(
        scoped, cycle_choices=(("scope:0", "0"), ("scope:1", "1"), ("scope:2", "2"))
    )
    assert form.adopt_projection(FormSpec((more,)), {"fate": "scope:2"}) is False
    assert form.adopt_projection(FormSpec((more,)), {"fate": "scope:1"}) is True
    assert form.read_value("fate") == "scope:1"


def test_a_kept_edit_takes_the_bounds_its_owner_re_declared() -> None:
    """A blank-or-number edit kept across a widened range held the OLD
    validator and clamp: 50 in a 0..100 field was Intermediate and Return
    put it back to 10.  The spelling the operator chose to read a quantity
    in survives the re-declaration; a different unit is a different row."""

    from dataclasses import replace

    from PyQt5 import QtCore, QtGui, QtWidgets

    from zlc_ui import ensure_qt_app
    from zlc_ui.form import FormFieldProps, FormSpec
    from zlc_ui.form.qt_form import FluentParameterForm

    ensure_qt_app(["zlc-ui-tests"])
    optional = FormFieldProps("bound", "float", "Bound", minimum=0.0, maximum=10.0)
    form = FluentParameterForm(FormSpec((optional,)), {"bound": 5.0})
    edit = form.widget_for("bound")
    form.reconcile(FormSpec((replace(optional, maximum=100.0),)), {"bound": 50.0})
    assert form.widget_for("bound") is edit, "same family: the control is kept"
    assert edit.validator().validate("50", 2)[0] == QtGui.QValidator.Acceptable
    QtWidgets.QApplication.sendEvent(
        edit,
        QtGui.QKeyEvent(QtCore.QEvent.KeyPress, QtCore.Qt.Key_Return, QtCore.Qt.NoModifier),
    )
    assert form.read_value("bound") == 50.0

    floor = FormFieldProps("floor", "float", "Floor", unit="dBm", minimum=-120.0, maximum=30.0)
    form = FluentParameterForm(FormSpec((floor,)), {"floor": -3.0})
    power = form.widget_for("floor")
    form._shown_unit_picked("floor", "mW")
    form.reconcile(FormSpec((replace(floor, maximum=20.0),)), {"floor": -3.0})
    assert form.widget_for("floor") is power
    assert form.shown_unit_for("floor") == "mW", "the operator's spelling is theirs"
    assert abs(form.read_value("floor") - -3.0) < 1e-9
    form.reconcile(FormSpec((replace(floor, unit="Hz"),)), {"floor": 5.0})
    assert form.widget_for("floor") is not power, "another unit is another row"
    assert form.unit_picker_for("floor").unit() == "Hz"


def test_an_auto_quantity_returns_to_editable_with_its_picker() -> None:
    """The editor and its picker share a holder, and Auto disabled the
    HOLDER while Manual re-enabled only the editor inside it: the field
    could never be edited again."""

    from zlc_ui import ensure_qt_app
    from zlc_ui.form import FormFieldProps, FormSpec
    from zlc_ui.form.qt_form import FluentParameterForm

    ensure_qt_app(["zlc-ui-tests"])
    auto = FormFieldProps("delay", "float", "Delay", automatic=True, unit="s")
    form = FluentParameterForm(FormSpec((auto,)), {"delay": None})
    assert form.auto_switch_for("delay").isChecked()
    assert not form.widget_for("delay").isEnabled()
    form.auto_switch_for("delay").setChecked(False)
    assert form.widget_for("delay").isEnabled()
    assert form.cell_for("delay").isEnabled()
    assert form.unit_picker_for("delay") is not None
    assert form.read_all() == {"delay": 0.0}


def test_replacing_a_row_keeps_the_picker_it_built() -> None:
    """A same-key row rebuilt in another family recorded its new cell and
    picker, then retired the old row -- by key, erasing both records: the
    picker was on screen and ``unit_picker_for`` said None."""

    from dataclasses import replace

    from PyQt5 import QtCore

    from zlc_ui import ensure_qt_app
    from zlc_ui.form import FormFieldProps, FormSpec
    from zlc_ui.form.qt_form import FluentParameterForm

    app = ensure_qt_app(["zlc-ui-tests"])
    amount = FormFieldProps("amount", "float", "Amount", default=1.0, unit="s")
    form = FluentParameterForm(FormSpec((amount,)), {"amount": 1.0})
    first = form.unit_picker_for("amount")
    form.reconcile(FormSpec((replace(amount, default=None),)), {"amount": None})
    picker = form.unit_picker_for("amount")
    assert picker is not None and picker is not first
    assert form.cell_for("amount") is not form.widget_for("amount")
    assert form.cell_for("amount").isAncestorOf(picker)
    app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
    assert form.unit_picker_for("amount") is picker


def test_the_same_values_enable_a_dependent_the_same_way_through_every_entry() -> None:
    """Construction, populate, refresh, reconcile and a toggle each had a
    formula of their own: the same values disabled a dependent through one
    door and enabled it through the next, a governed field's Auto was
    overwritten by its controller, and a controller mid-keystroke raised
    inside the change slot."""

    from zlc_ui import ensure_qt_app
    from zlc_ui.form import FormFieldProps, FormSpec
    from zlc_ui.form.qt_form import FluentParameterForm

    ensure_qt_app(["zlc-ui-tests"])
    toggle = FormFieldProps("on", "bool", "On", default=False)
    dependent = FormFieldProps(
        "dependent", "text", "Dependent", default="", enabled_when=("on", (True,))
    )
    spec = FormSpec((toggle, dependent))
    form = FluentParameterForm(spec, {"on": False, "dependent": ""})
    gated = form.widget_for("dependent")
    assert not gated.isEnabled()
    form.populate({"on": False, "dependent": ""})
    assert not gated.isEnabled(), "populate re-enabled what its controller forbids"
    form.refresh()
    assert not gated.isEnabled()
    form.populate({"on": True, "dependent": ""})
    assert gated.isEnabled()
    form.reconcile(spec, {"on": False, "dependent": ""})
    assert not gated.isEnabled()

    governed = FormFieldProps(
        "gain", "float", "Gain", automatic=True, enabled_when=("on", (True,))
    )
    form = FluentParameterForm(FormSpec((toggle, governed)), {"on": True, "gain": None})
    assert not form.widget_for("gain").isEnabled(), "Auto holds no value to edit"
    form.auto_switch_for("gain").setChecked(False)
    assert form.widget_for("gain").isEnabled()
    form.populate({"on": False, "gain": 1.0})
    assert not form.widget_for("gain").isEnabled()

    count = FormFieldProps("count", "number", "Count", default=2)
    flag = FormFieldProps("flag", "text", "Flag", default="", enabled_when=("count", (2,)))
    form = FluentParameterForm(FormSpec((count, flag)), {"count": 2, "flag": ""})
    assert form.widget_for("flag").isEnabled()
    form.widget_for("count").setText("-")
    assert not form.widget_for("flag").isEnabled(), "half-typed is not one of the values"
    form.widget_for("count").setText("2")
    assert form.widget_for("flag").isEnabled()


def test_a_form_opens_on_the_values_it_is_given_and_an_int_has_no_width() -> None:
    """A required field with no default could not be built with the value
    it was handed, because the form was built on the default first.  A
    Python int has no width: 2**40 overflowed the control meant to hold it,
    and so did the validator of the edit meant to replace that control."""

    from PyQt5 import QtCore, QtGui, QtWidgets

    from zlc_ui import ensure_qt_app
    from zlc_ui.form import FormFieldProps, FormSpec
    from zlc_ui.form.qt_form import FluentParameterForm

    ensure_qt_app(["zlc-ui-tests"])
    required = FormFieldProps("n", "int", "N", required=True)
    form = FluentParameterForm(FormSpec((required,)), {"n": 10})
    assert form.read_all() == {"n": 10}

    big = FormFieldProps("n", "int", "N", default=2**40)
    form = FluentParameterForm(FormSpec((big,)), {"n": 2**40})
    assert form.read_all() == {"n": 2**40}
    box = form.widget_for("n")
    box.stepBy(1)
    assert form.read_value("n") == 2**40 + 1
    assert box.validate("1.5", 3)[0] == QtGui.QValidator.Invalid, "a whole number"

    wide = FormFieldProps("n", "int", "N", default=10, maximum=2**40)
    form = FluentParameterForm(FormSpec((wide,)), {"n": 10})
    assert form.read_all() == {"n": 10}
    form.populate({"n": 2**40})
    assert form.read_value("n") == 2**40

    optional = FormFieldProps("n", "int", "N", maximum=2**40)
    form = FluentParameterForm(FormSpec((optional,)), {"n": None})
    edit = form.widget_for("n")
    edit.setText(str(2**41))
    QtWidgets.QApplication.sendEvent(
        edit,
        QtGui.QKeyEvent(QtCore.QEvent.KeyPress, QtCore.Qt.Key_Return, QtCore.Qt.NoModifier),
    )
    assert form.read_value("n") == 2**40, "clamped exactly, in integers"


def test_a_rows_field_is_built_one_row_at_a_time() -> None:
    """A field of rows -- a derive's signals -- is edited as rows: Add puts
    an empty row on the form with a box per column and the cursor in its
    first box, × takes a row away, every keystroke applies like every other
    kind, and a projection that carries the same rows back does not rebuild
    the boxes -- nor does one carrying different rows while the operator is
    inside one of them."""

    _run_qt(
        """
import zou_lab_control
from PyQt5 import QtCore, QtTest, QtWidgets
from zlc_ui.qt import ensure_qt_app
from zlc_ui.fluent import FluentLineEdit
from zlc_ui.form.form import FormFieldProps, FormSpec
from zlc_ui.form.qt_form import FluentParameterForm

app = ensure_qt_app(['test'])
columns = (
    FormFieldProps('name', 'text', 'Name', default='', required=True, description='name of the signal'),
    FormFieldProps('expression', 'text', 'Expression', default='', required=True, description='expression over a.<output>'),
)
spec = FormSpec((
    FormFieldProps('expressions', 'rows', 'Signals', default=(), required=True, columns=columns),
))
first = {'name': 'agree', 'expression': 'a.occupied.frame(0) == a.occupied.frame(2)'}
form = FluentParameterForm(spec, {'expressions': (first,)})
window = QtWidgets.QMainWindow(); window.setCentralWidget(form); window.show()
window.activateWindow(); app.processEvents()
editor = form.widget_for('expressions')
assert form.read_all() == {'expressions': (first,)}
seen = []; form.changed.connect(seen.append)

editor.add_button.click(); app.processEvents()
assert seen == ['expressions'], seen
_row, cells, remove = editor._rows[1]
assert isinstance(cells['name'], FluentLineEdit)
assert QtWidgets.QApplication.focusWidget() is cells['name']
assert cells['expression'].placeholderText() == 'expression over a.<output>'
QtTest.QTest.keyClicks(cells['name'], 'counts')
QtTest.QTest.keyClicks(cells['expression'], 'a.counts.frame(1).where(agree)')
app.processEvents()
second = {'name': 'counts', 'expression': 'a.counts.frame(1).where(agree)'}
assert form.read_all() == {'expressions': (first, second)}, form.read_all()
assert seen.count('expressions') > 2

# The projection of the same rows lands without rebuilding a box.
form.reconcile(spec, {'expressions': (first, second)}); app.processEvents()
assert editor._rows[1][1]['expression'] is cells['expression']
# Different rows are not written under the operator's cursor ...
cells['expression'].setFocus(QtCore.Qt.MouseFocusReason); app.processEvents()
assert QtWidgets.QApplication.focusWidget() is cells['expression']
form.reconcile(spec, {'expressions': (first,)}); app.processEvents()
assert form.read_all() == {'expressions': (first, second)}
# ... and land once they have left it.
cells['expression'].clearFocus(); app.processEvents()
form.reconcile(spec, {'expressions': (first,)}); app.processEvents()
assert form.read_all() == {'expressions': (first,)}

# × takes the row away; with no row left, the required field is vacant.
editor._rows[0][2].click(); app.processEvents()
assert form.is_empty('expressions') and form.read_draft() == {'expressions': None}
form.populate({'expressions': (first,)})
assert form.read_all() == {'expressions': (first,)}
window.close()
print('ok')
"""
    )
