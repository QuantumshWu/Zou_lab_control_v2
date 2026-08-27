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
