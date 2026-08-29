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


def _run_qt(code: str) -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = (
        "" if environment.get("ZLC_TEST_INSTALLED") == "1"
        else os.pathsep.join((str(REPO_ROOT), str(SRC)))
    )
    environment["QT_QPA_PLATFORM"] = "offscreen"
    completed = subprocess.run(
        [sys.executable, "-c", code],
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
card.show()
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
