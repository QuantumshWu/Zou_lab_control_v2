from __future__ import annotations


def test_panel_card_construct_and_setters(run_qt) -> None:
    run_qt(
        """
from PyQt5 import QtCore, QtGui, QtTest, QtWidgets
from zlc_ui.qt import ensure_qt_app
from zlc_ui.fluent import FluentComboBox, FluentTreeComboBox
from zlc_ui.console import panel_card_view as tested_module
PanelCardView = tested_module.PanelCardView
app = ensure_qt_app(['test'])
card = PanelCardView('panel-1', 'Card')
card.set_size_choices(('1x2', '2x2', '1x4'), '2x2')
card.set_signal_choices((('source', (('Temperature', 'temperature'),)),))
state = {
    'signal': '', 'kind': 'image', 'size': '2x2', 'interval_ms': 100,
    'title': 'Card', 'semantic': {}, 'display': {}, 'fit': {},
    'overlay_signal': '',
}
surface_projection = {
    'semantic': (), 'display': (), 'fit': (),
}
card.set_panel_projection(state, surface_projection)
class _GatedSurface(QtWidgets.QLabel):
    # What a plot widget is, as far as this card is concerned: something you
    # can stop the operator dragging on.
    interaction = True
    def set_interaction_enabled(self, enabled):
        self.interaction = bool(enabled)
surface = _GatedSurface('fake surface')
card.set_surface(surface)
# The status mark: a dot overlaying the title strip beside Setting, so the
# card body reserves no status row and the message rides the tooltip.
from zlc_ui.fluent import GREY, RED
card.set_status('ready', error=False)
assert not card.status_dot.isHidden(), 'a status did not surface its dot'
assert card.status_dot.toolTip() == 'ready'
assert GREY in card.status_dot.styleSheet()
card.set_status('the renderer refused this frame', error=True)
assert not card.status_dot.isHidden()
assert card.status_dot.toolTip() == 'the renderer refused this frame'
assert RED in card.status_dot.styleSheet()
# Geometry is only true once the layout has run: the dot and the Setting
# button are items in the title strip's row, not free-floating overlays.
card.show(); app.processEvents()
settings_x = card.settings_button.mapTo(card._title_band, QtCore.QPoint()).x()
assert card.status_dot.x() < settings_x, 'the dot must sit beside Setting'
assert card.status_dot.y() >= 0 and card.status_dot.y() < card._title_band.height()
card.set_status('', error=False)
assert card.status_dot.isHidden(), 'clearing the status must clear the mark'
card.set_status('ready', error=False)
assert card.panel_id == 'panel-1'
signal_field = next(field for field in card._form_spec().fields if field.key == 'signal')
assert signal_field.kind == 'keyed_choice'
card._open_settings()
signal_picker = card._settings_form.widget_for('signal')
assert isinstance(signal_picker, FluentTreeComboBox)
assert signal_picker.current_choice_key() == ''
assert signal_picker.model().item(0).text() == 'source'
# A Setting value the card cannot read is said on the card while it stands.
# The next projection writes the record's value over that row -- with its
# signals blocked, so no edit clears it -- and the error goes with the value
# it was about: the presenter's word shows again.
def unreadable(_key):
    raise ValueError('not a panel size')
card._settings_form.read_value = unreadable
card._setting_changed('size')
del card._settings_form.read_value
assert card.status_dot.toolTip() == 'not a panel size'
assert card._settings_popup.isVisible()
card.set_panel_projection(dict(state, title='Renamed'), surface_projection)
assert card.status_dot.toolTip() == 'ready', card.status_dot.toolTip()
card.set_panel_projection(state, surface_projection)
picked = []
card.state_changed.connect(picked.append)
card.show()
signal_picker.showPopup()
source_index = signal_picker.model().index(0, 0)
signal_picker.view().expand(source_index)
app.processEvents()
leaf_index = signal_picker.model().index(0, 0, source_index)
QtTest.QTest.mouseClick(
    signal_picker.view().viewport(), QtCore.Qt.LeftButton,
    pos=signal_picker.view().visualRect(leaf_index).center(),
)
assert {'signal': 'temperature'} in picked
assert surface.parentWidget() is not None
# A bare card is interactive until TaskConsole projects its global switch.
assert surface.interaction is True
card.set_selectors_enabled(True)
assert surface.interaction is True
card.set_selectors_enabled(False)
assert surface.interaction is False
body = QtWidgets.QWidget()
body.setMinimumSize(420, 1200)
card.setParent(body)
card.setGeometry(10, 10, 340, 240)
scroll = QtWidgets.QScrollArea()
scroll.setWidget(body)
scroll.resize(380, 300)
scroll.show()
app.processEvents()
bar = scroll.verticalScrollBar()
assert bar.maximum() > 0
wheel = QtGui.QWheelEvent(
    QtCore.QPointF(surface.rect().center()),
    QtCore.QPointF(surface.mapToGlobal(surface.rect().center())),
    QtCore.QPoint(),
    QtCore.QPoint(0, -240),
    QtCore.Qt.NoButton,
    QtCore.Qt.NoModifier,
    QtCore.Qt.NoScrollPhase,
    False,
)
QtWidgets.QApplication.sendEvent(surface, wheel)
assert bar.value() > 0, 'a wheel the surface ignores must still reach the board scroll'
card.set_surface(None)
# An emptied card says why on its strip's second line; the body is its frame.
assert card._title_label.text().endswith('Choose&nbsp;a&nbsp;signal&nbsp;in&nbsp;Setting</span>'), card._title_label.text()
card.set_panel_projection(dict(state, size='1x4'), surface_projection)
assert card.panel_size == '1x4'
try:
    card.set_panel_projection(dict(state, size='not-a-panel-size'), surface_projection)
except ValueError:
    pass
else:
    raise AssertionError('invalid panel size was accepted')
"""
    )


def test_panel_card_qtest_signal_payloads(run_qt) -> None:
    run_qt(
        """
from zlc_ui.console import panel_card_view as tested_module
from PyQt5 import QtCore, QtGui, QtTest, QtWidgets
from zlc_ui.fluent import GREY, RED, FluentOverlayFrame
from zlc_ui.qt import ensure_qt_app
PanelCardView = tested_module.PanelCardView
app = ensure_qt_app(['test'])
card = PanelCardView('panel-1', 'Card')
card.set_size_choices(('2x2',), '2x2')
card.set_interval_choices((100, 200, 400, 800), 400)
card.set_panel_projection({
    'signal': '', 'kind': 'image', 'size': '2x2', 'interval_ms': 400,
    'title': 'Card', 'semantic': {}, 'display': {}, 'fit': {},
    'overlay_signal': '',
}, {
    'semantic': (), 'display': (), 'fit': (),
    'semantic_unavailable': '', 'display_unavailable': '',
    'fit_unavailable': '',
})
card.resize(340, 180); card.show(); app.processEvents()
events = []
card.remove_requested.connect(lambda: events.append(('remove',)))
card.edit_requested.connect(lambda: events.append(('edit',)))
card.state_changed.connect(lambda patch: events.append(('state', patch)))
# The header × is guarded against a stray click: first click turns it into a
# red confirm, the second click inside the platform double-click interval
# removes the panel.
QtTest.QTest.mouseClick(card.close_button, QtCore.Qt.LeftButton)
assert events == []
assert card.close_button.role() == 'destructive_primary'
QtTest.QTest.mouseClick(card.close_button, QtCore.Qt.LeftButton)
assert events == [('remove',)]
assert card.close_button.role() == 'destructive'
# Edit and the explicit text Remove remain available in Setting as well.
QtTest.QTest.mouseClick(card.settings_button, QtCore.Qt.LeftButton)
app.processEvents()
popup = card._settings_popup
# A frame INSIDE the page, not a window.  A bare card is its own page, so
# the frame is an ordinary child clipped to the card.
assert isinstance(popup, FluentOverlayFrame)
assert not popup.isWindow()
assert popup.parentWidget() is card.window()
assert card._settings_drag_handle.text() == 'Setting · panel-1'
assert card._settings_close_button.toolTip() == 'Close settings'
assert 240 <= popup.width() <= 320
assert popup.height() <= card.height()
assert popup.mapToGlobal(popup.rect().bottomRight()).y() <= card.mapToGlobal(
    card.rect().bottomRight()
).y()
assert card._settings_scroll.geometry().right() == popup.rect().right()
app.processEvents()
# Whatever the form's length, all of it is reachable inside the popup the
# card clamps: the scrollbar's range is exactly the overflow.
scroll = card._settings_scroll
overflow = max(0, scroll.widget().sizeHint().height() - scroll.viewport().height())
assert scroll.verticalScrollBar().maximum() == overflow
title = card._settings_form.widget_for('title')
title.setText('Card changed')
title.editingFinished.emit()
assert card._settings_drag_handle.text() == 'Setting · panel-1'
card._settings_form.widget_for('interval_ms').button(800).click()
app.processEvents()
assert popup.isVisible()
# Closing never delays the next open: no wait between them.
QtTest.QTest.mouseClick(card._settings_close_button, QtCore.Qt.LeftButton)
app.processEvents()
assert not popup.isVisible()
QtTest.QTest.mouseClick(card.settings_button, QtCore.Qt.LeftButton)
app.processEvents()
assert popup.isVisible()
popup.hide()  # The page owner may hide it while switching views.
app.processEvents()
assert not popup.isVisible()
QtTest.QTest.mouseClick(card.settings_button, QtCore.Qt.LeftButton)
app.processEvents()
QtTest.QTest.mouseClick(card.edit_button, QtCore.Qt.LeftButton)
app.processEvents()
QtTest.QTest.mouseClick(card.settings_button, QtCore.Qt.LeftButton)
app.processEvents()
QtTest.QTest.mouseClick(card.remove_button, QtCore.Qt.LeftButton)
assert ('edit',) in events
assert ('remove',) in events
assert any(event[0] == 'state' and event[1]['title'] == 'Card changed' for event in events)
assert ('state', {'interval_ms': 800}) in events
popup.close(); card.close(); card.deleteLater()
app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete); app.processEvents()
"""
    )


def test_the_setting_frame_belongs_to_the_panel_page(run_qt) -> None:
    """The Setting frame lives inside the panel page, full stop.

    It was a Qt.Popup once (dismissed by the next click), then a Qt.Tool
    companion window (round after round of desktop bookkeeping: follow the
    console's moves, hide on minimise, restore with it) -- and the window
    kind still had the two defects ownership cannot fix: a top-level
    survives its page being hidden, so switching the console to another
    tab left the frame floating over the wrong page, and nothing bounded
    a drag to the panel area.  As an ordinary child of the panel page's
    viewport every one of those behaviours is parenthood, not machinery:
    clipped to the page, hidden and restored with the tab, travelling
    with the window, gone with it.
    """

    run_qt(
        """
from zlc_ui.fluent import fluent as tested_module
from PyQt5 import QtCore, QtTest, QtWidgets
from zlc_ui import open_task_console
from zlc_ui.console import PanelCardView
from zlc_ui.qt import ensure_qt_app
FluentOverlayFrame = tested_module.FluentOverlayFrame
app = ensure_qt_app(['test'])
top_levels = set(app.topLevelWidgets())
console = open_task_console(window_ratio=0.4)
console.set_panel_sizes(('2x2',), '2x2')
console.set_panel_intervals((100, 200, 400, 800), 100)
console.add_panel('panel-1', 'Panel')
console.set_panel_projection('panel-1', {
    'signal': '', 'kind': 'image', 'size': '2x2', 'interval_ms': 100,
    'title': 'Panel', 'semantic': {}, 'display': {}, 'fit': {},
    'overlay_signal': '',
}, {
    'semantic': (), 'display': (), 'fit': (),
    'semantic_unavailable': '', 'display_unavailable': '',
    'fit_unavailable': '',
})
app.processEvents()
owner = next(
    widget for widget in set(app.topLevelWidgets()) - top_levels
    if isinstance(widget, tested_module.FluentWindow)
)
card = owner.findChild(PanelCardView)
assert card is not None

QtTest.QTest.mouseClick(card.settings_button, QtCore.Qt.LeftButton)
app.processEvents()
popup = card._settings_popup
assert popup is not None
assert popup.isVisible()
# NOT a window: the console window is the only top-level there is.
assert not popup.isWindow()
assert popup.window() is owner
assert {
    widget for widget in app.topLevelWidgets() if widget.isVisible()
} == top_levels | {owner}
# Its page is the panel area's scroll viewport -- the widget the tab
# shows and hides.
page = popup.parentWidget()
scroll_ancestor = card.parentWidget()
while scroll_ancestor is not None and not isinstance(
    scroll_ancestor, QtWidgets.QAbstractScrollArea
):
    scroll_ancestor = scroll_ancestor.parentWidget()
assert scroll_ancestor is not None
assert page is scroll_ancestor.viewport()

# THE TAB IS THE LAW.  Switching the console to Logic hides the panel
# page, and the frame -- its child -- goes with it; switching back
# brings it back.  This is the exact defect the window kind had: a
# Qt.Tool top-level stayed up, floating over the Logic page.
tabs = owner.findChild(tested_module.FluentTabWidget)
assert tabs is not None and tabs.count() >= 2
tabs.setCurrentIndex(1)
app.processEvents()
assert not popup.isVisible(), 'the frame floated over another tab'
# Hidden page updates stay as metadata until the same popup becomes visible.
form = card._settings_parameter_forms['display']
reconciles = []
original_reconcile = form.reconcile
form.reconcile = lambda *args, **kwargs: (reconciles.append(True), original_reconcile(*args, **kwargs))[1]
changed_state = dict(card._state_projection)
changed_state['display'] = {'colormap': 'magma'}
changed_surface = dict(card._parameter_surface)
changed_surface['display'] = ({
    'key': 'colormap', 'label': 'Colormap', 'kind': 'choice',
    'value': 'magma', 'allow_none': False,
    'choices': (('Viridis', 'viridis'), ('Magma', 'magma')),
    'minimum': None, 'maximum': None, 'step': None,
},)
console.set_panel_projection('panel-1', changed_state, changed_surface)
assert not reconciles and 'colormap' not in form.spec.keys
tabs.setCurrentIndex(0)
app.processEvents()
assert popup.isVisible(), 'the frame did not come back with its tab'
print('restored Setting keys', form.spec.keys)
assert form.read_value('colormap') == 'magma'
assert len(reconciles) == 1
# The rows came with their section: titled, and in the page's one column.
assert card._settings_sections['display'].isVisible()
assert not card._settings_sections['fit'].isVisible()
tabs.setCurrentIndex(1); app.processEvents()
tabs.setCurrentIndex(0); app.processEvents()
assert len(reconciles) == 1, 'an unchanged page return rebuilt the form'

# Runtime keyed choices are not part of FormSpec: retain their invalidation too.
form = card._settings_form
panel_reconciles = []
panel_reconcile = form.reconcile
form.reconcile = lambda *args, **kwargs: (panel_reconciles.append(True), panel_reconcile(*args, **kwargs))[1]
card.set_signal_choices((('source', (('Original', '@logic/source/value'),)),),
                        current='@logic/source/value')
picker = form.widget_for('signal')
QtTest.QTest.mouseClick(card._settings_close_button, QtCore.Qt.LeftButton)
app.processEvents()
before_choices = len(panel_reconciles)
card.set_signal_choices((('source', (('Original', '@logic/source/value'),
                                    ('Derived', '@logic/source/derived'))),),
                        current='@logic/source/value')
assert len(panel_reconciles) == before_choices
assert not picker._find_choice_index('@logic/source/derived').isValid()
QtTest.QTest.mouseClick(card.settings_button, QtCore.Qt.LeftButton)
app.processEvents()
assert form.widget_for('signal') is picker
assert picker._find_choice_index('@logic/source/derived').isValid()
assert len(panel_reconciles) == before_choices + 1

# The console going behind another program cannot separate them: a child
# has no desktop presence to leave behind.
QtWidgets.QApplication.sendEvent(
    owner, QtCore.QEvent(QtCore.QEvent.WindowDeactivate)
)
app.processEvents()
assert popup.isVisible()

# The console MOVING takes the frame with it -- parenthood, not tracking.
before = popup.mapToGlobal(QtCore.QPoint(0, 0))
was = owner.frameGeometry().topLeft()
owner.move(was.x() + 37, was.y() + 23)
app.processEvents()
delta = popup.mapToGlobal(QtCore.QPoint(0, 0)) - before
assert delta == QtCore.QPoint(37, 23), delta

# A DRAG CANNOT LEAVE THE PAGE.  The clamp is the page rect: asked to put
# the frame far outside, the position pins to the page's edge.
low = card._clamped_overlay_position(
    page.mapToGlobal(QtCore.QPoint(-5000, -5000))
)
assert low == QtCore.QPoint(0, 0), low
high = card._clamped_overlay_position(
    page.mapToGlobal(QtCore.QPoint(5000, 5000))
)
assert high == QtCore.QPoint(
    max(0, page.width() - popup.width()),
    max(0, page.height() - popup.height()),
), high

# A page frame has no Popup auto-dismiss. Closing its explicit button
# must not suppress the next genuine Setting click within 250 ms.
QtTest.QTest.mouseClick(card._settings_close_button, QtCore.Qt.LeftButton)
app.processEvents()
assert not popup.isVisible()
QtTest.QTest.qWait(20)
QtTest.QTest.mouseClick(card.settings_button, QtCore.Qt.LeftButton)
app.processEvents()
assert popup.isVisible(), 'an explicit close suppressed the next Setting click'

# A frame the operator CLOSED stays closed across tab changes.
QtTest.QTest.mouseClick(card._settings_close_button, QtCore.Qt.LeftButton)
app.processEvents()
assert not popup.isVisible()
tabs.setCurrentIndex(1)
app.processEvents()
tabs.setCurrentIndex(0)
app.processEvents()
assert not popup.isVisible(), 'a retired frame was resurrected by its tab'

QtTest.QTest.mouseClick(card.settings_button, QtCore.Qt.LeftButton)
app.processEvents()
assert popup.isVisible()
owner.hide()
app.processEvents()
assert not popup.isVisible(), 'a hidden owner left its frame visible'

owner.show()
app.processEvents()
owner.close()
app.processEvents()
assert not popup.isVisible(), 'a closed owner left its frame visible'
console.deleteLater()
owner.deleteLater()
QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
app.processEvents()
"""
    )


def test_public_panel_removal_retires_its_popup_and_card(run_qt) -> None:
    run_qt(
        """
from PyQt5 import QtCore, sip
from zlc_ui import open_task_console
from zlc_ui.qt import ensure_qt_app

app = ensure_qt_app(['panel-remove'])
console = open_task_console(window_ratio=0.4)
console.set_panel_sizes(('2x2',), '2x2')
console.set_panel_intervals((100,), 100)
console.add_panel('panel-1', 'Panel')
console.set_panel_projection('panel-1', {
    'signal': '', 'kind': 'image', 'size': '2x2', 'interval_ms': 100,
    'title': 'Panel', 'semantic': {}, 'display': {}, 'fit': {},
    'overlay_signal': '',
}, {
    'semantic': (), 'display': (), 'fit': (),
    'semantic_unavailable': '', 'display_unavailable': '',
    'fit_unavailable': '',
})
card = console._cards['panel-1']
card._open_settings()
app.processEvents()
popup = card._settings_popup
assert popup is not None and popup.isVisible()

try:
    console.remove_panel('panel-1')
    assert not popup.isVisible(), 'public remove_panel left Setting visible'
    assert console.panel_ids() == ()
    assert not console._view.board._wired_cards
    QtCore.QCoreApplication.sendPostedEvents(
        None, QtCore.QEvent.DeferredDelete
    )
    app.processEvents()
    assert sip.isdeleted(card), 'removed card was not deferred-deleted'
finally:
    console.close()
    app.processEvents()
"""
    )


def test_a_dragged_setting_popup_stays_where_the_operator_put_it(run_qt) -> None:
    """Growing the form must not throw the popup back beside its button.

    Placement had two owners -- the anchor and the operator's drag -- and
    the anchor re-asserted itself on every content resize.  Picking a fit
    model adds the parameter row, so the popup jumped back to the card
    the moment the operator used it.

    That was first answered by remembering the dragged origin and feeding it
    back into the re-placement, which left the re-placement itself in
    charge: it kept the position and changed the SIZE instead, under a cap
    that differed once a drag had happened.  Now placement runs once, when
    the frame opens; nothing re-places a frame that is already on screen.
    """

    run_qt(
        """
from PyQt5 import QtCore
from zlc_ui import open_task_console
from zlc_ui.qt import ensure_qt_app

app = ensure_qt_app(['setting-drag'])
console = open_task_console(window_ratio=0.4)
console.set_panel_sizes(('2x2',), '2x2')
console.set_panel_intervals((100,), 100)
console.add_panel('panel-1', 'Panel')
state = {
    'signal': '', 'kind': 'image', 'size': '2x2', 'interval_ms': 100,
    'title': 'Panel', 'semantic': {}, 'display': {}, 'fit': {},
    'overlay_signal': '',
}
def fit_fields(*keys):
    return tuple(
        {
            'key': key, 'label': key.title(), 'kind': 'text', 'value': '',
            'allow_none': True, 'choices': (), 'minimum': None,
            'maximum': None, 'step': None,
        }
        for key in keys
    )
console.set_panel_projection('panel-1', state, {
    'semantic': (), 'display': (), 'fit': fit_fields('model'),
    'semantic_unavailable': '', 'display_unavailable': '',
    'fit_unavailable': '',
})
card = console._cards['panel-1']
card._open_settings()
app.processEvents()
popup = card._settings_popup
assert popup is not None and popup.isVisible()

handle = card._settings_drag_handle
press = QtCore.QPoint(4, 4)
carried = popup.frameGeometry().topLeft() + QtCore.QPoint(120, 90)
import PyQt5.QtGui as QtGui
app.sendEvent(handle, QtGui.QMouseEvent(
    QtCore.QEvent.MouseButtonPress, QtCore.QPointF(press),
    handle.mapToGlobal(press), QtCore.Qt.LeftButton,
    QtCore.Qt.LeftButton, QtCore.Qt.NoModifier,
))
app.sendEvent(handle, QtGui.QMouseEvent(
    QtCore.QEvent.MouseMove, QtCore.QPointF(press),
    carried + (handle.mapToGlobal(press) - popup.frameGeometry().topLeft()),
    QtCore.Qt.LeftButton, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier,
))
app.sendEvent(handle, QtGui.QMouseEvent(
    QtCore.QEvent.MouseButtonRelease, QtCore.QPointF(press),
    carried, QtCore.Qt.NoButton, QtCore.Qt.NoButton, QtCore.Qt.NoModifier,
))
app.processEvents()
moved = popup.frameGeometry().topLeft()
assert moved != carried - QtCore.QPoint(120, 90), 'the drag did nothing'

try:
    # Picking a fit model: the form grows a row and re-measures.
    console.set_panel_projection('panel-1', state, {
        'semantic': (), 'display': (),
        'fit': fit_fields('model', 'expression'),
        'semantic_unavailable': '', 'display_unavailable': '',
        'fit_unavailable': '',
    })
    app.processEvents()
    assert popup.frameGeometry().topLeft() == moved, (
        popup.frameGeometry().topLeft(), moved
    )
    # Closing ends the operator's placement; the next open anchors again.
    # There is no remembered origin to check -- placement runs once, when
    # the frame opens, so a closed frame carries nothing forward.
    card.retire_settings_popup()
    app.processEvents()
    assert not popup.isVisible()
    specs = []
    original_spec = card._form_spec
    def counted_spec():
        specs.append(True)
        return original_spec()
    card._form_spec = counted_spec
    console.set_panel_projection('panel-1', state, {
        'semantic': (), 'display': (), 'fit': fit_fields('model'),
    })
    app.processEvents()
    assert not specs, 'a hidden Setting rebuilt its controls'
    card._open_settings()
    app.processEvents()
    assert popup.isVisible() and 'expression' not in card._settings_parameter_forms['fit'].keys
finally:
    console.close()
    app.processEvents()
"""
    )


def test_board_constructs_and_packs_from_metrics(run_qt) -> None:
    run_qt(
        """
from zlc_ui.qt import ensure_qt_app
from zlc_ui.board import BoardMetrics
from zlc_ui.console import ConsoleBoardView, PanelCardView
app = ensure_qt_app(['test'])
card = PanelCardView('panel-1')
# The card's own rectangle is the packer's input, so the expectation is
# derived from it -- a number typed here would be a second answer to the
# question this change exists to give one answer to.
# The rectangle the card RESERVED -- the picture its preset plans plus its
# own strip and margins -- is what the packer is given.
hint = card.size()
board = ConsoleBoardView(metrics=BoardMetrics(12))
board.set_cards((card,))
board.resize(hint.width() + 40, hint.height() + 40); board.show(); app.processEvents()
assert card.geometry().getRect() == (12, 12, hint.width(), hint.height())
assert not board.grab().isNull()
"""
    )


def test_board_does_not_capture_ancestor_update_suppression(run_qt) -> None:
    run_qt(
        """
from PyQt5 import QtWidgets
from zlc_ui.qt import ensure_qt_app
from zlc_ui.console import ConsoleBoardView, PanelCardView
app = ensure_qt_app(['ancestor-update-suppression'])

class CountingCard(PanelCardView):
    def __init__(self, panel_id):
        self.paint_count = 0
        super().__init__(panel_id)

    def paintEvent(self, event):
        self.paint_count += 1
        super().paintEvent(event)

host = QtWidgets.QWidget()
board = ConsoleBoardView(host)
card = CountingCard('panel-1')
host.resize(640, 480)
board.setGeometry(host.rect())

# QTabWidget/QScrollArea may suppress their descendants while changing the
# visible page.  Packing during that interval must not record the inherited
# state as a permanent board-local disable.
host.setUpdatesEnabled(False)
board.set_cards((card,))
assert not board.updatesEnabled()
host.setUpdatesEnabled(True)
host.show()
app.processEvents()

assert board.updatesEnabled()
assert card.updatesEnabled()
assert card.paint_count > 0
assert card.geometry().width() > 0 and card.geometry().height() > 0
host.close()
app.processEvents()
"""
    )


def test_board_qtest_drop_uses_the_slot_under_the_pointer_and_keeps_every_place_a_drop_authored(run_qt) -> None:
    """A card carried by its strip lands in the slot under the pointer,
    commits the new order, and keeps every place a drop authored.

    The board packed every card into the first free slot on the board and
    exempted exactly one -- whichever had been dropped last -- so it could
    hold one placement at a time.  A card put below another was released by
    the NEXT drop and flew up into the hole beside it, which is the one thing
    gravity never does: it stops a card at the first thing in its way.
    """

    run_qt(
        """
from PyQt5 import QtCore, QtTest
from zlc_ui.board import BoardMetrics
from zlc_ui.qt import ensure_qt_app
from zlc_ui.console import ConsoleBoardView, PanelCardView
app = ensure_qt_app(['test'])
metrics = BoardMetrics(10)

def new_board(rows):
    cards = tuple(PanelCardView(f'panel-{index}') for index in range(3))
    w, h = cards[0].size().width(), cards[0].size().height()
    board = ConsoleBoardView(metrics=metrics)
    board.resize(3 * w + 4 * 10, rows * h + (rows + 1) * 10)
    board.set_cards(cards); board.show(); app.processEvents()
    events = []
    board.order_committed.connect(events.append)
    return board, cards, events, w, h

def drop_top_left_at(board, card, top_left):
    grab = QtCore.QPoint(18, 18)
    pointer = QtCore.QPoint(top_left[0] + 18, top_left[1] + 18)
    QtTest.QTest.mousePress(card, QtCore.Qt.LeftButton, pos=grab)
    QtTest.QTest.mouseMove(card, card.mapFrom(board, pointer))
    # Released where the pointer is: the card has moved under it since.
    QtTest.QTest.mouseRelease(card, QtCore.Qt.LeftButton, pos=card.mapFrom(board, pointer))
    app.processEvents()

# The nearest anchor, before gravity; each drop commits the board's order.
board, cards, events, w, h = new_board(2)
row = ((10, 10), (20 + w, 10), (30 + 2 * w, 10))
assert tuple(card.geometry().getRect()[:2] for card in cards) == row

drop_top_left_at(board, cards[2], (12, h + 16))
assert events[-1] == ('panel-0', 'panel-1', 'panel-2'), events
second_row = ((10, 10), (20 + w, 10), (10, 20 + h))
assert tuple(card.geometry().getRect()[:2] for card in cards) == second_row

board.resize(w + 20, 4 * h); app.processEvents()
assert all(card.geometry().x() == 10 for card in cards)
board.resize(3 * w + 4 * 10, 2 * h + 3 * 10); app.processEvents()
assert tuple(card.geometry().getRect()[:2] for card in cards) == second_row

# Onto panel-0: the two change places.
drop_top_left_at(board, cards[2], (12, 14))
assert events[-1] == ('panel-2', 'panel-1', 'panel-0'), events
by_id = {card.panel_id: card for card in cards}
assert tuple(by_id[panel_id].geometry().getRect()[:2] for panel_id in events[-1]) == second_row

# Every place a drop authored stands through the next drop.
board.close()
board, cards, events, w, h = new_board(3)
assert tuple(card.geometry().getRect()[:2] for card in cards) == (
    (10, 10), (20 + w, 10), (30 + 2 * w, 10)
)
# panel-1 under panel-0; panel-2 slides left into the place it left.
drop_top_left_at(board, cards[1], (12, h + 16))
assert cards[1].geometry().getRect()[:2] == (10, 20 + h), cards[1].geometry()
assert cards[2].geometry().getRect()[:2] == (20 + w, 10), cards[2].geometry()

# A second drop, and the first placement still stands.
drop_top_left_at(board, cards[2], (12, 2 * h + 40))
assert cards[0].geometry().getRect()[:2] == (10, 10), cards[0].geometry()
assert cards[1].geometry().getRect()[:2] == (10, 20 + h), cards[1].geometry()
assert cards[2].geometry().getRect()[:2] == (10, 30 + 2 * h), cards[2].geometry()
"""
    )


def test_board_qtest_drag_previews_the_drop_while_the_card_follows_the_pointer(run_qt) -> None:
    """While a card is carried it follows the pointer, and the board already
    stands as the drop will leave it: the card it is let go onto in the place
    it left, and the slot drawn where the card will rest."""

    run_qt(
        """
from PyQt5 import QtCore, QtTest
from zlc_ui.board import BoardMetrics
from zlc_ui.qt import ensure_qt_app
from zlc_ui.console import ConsoleBoardView, PanelCardView
app = ensure_qt_app(['test'])
metrics = BoardMetrics(10)
cards = tuple(PanelCardView(f'panel-{index}') for index in range(4))
w, h = cards[0].size().width(), cards[0].size().height()
board = ConsoleBoardView(metrics=metrics)
board.resize(2 * w + 3 * 10, 2 * h + 3 * 10)
board.set_cards(cards); board.show(); app.processEvents()
slots = ((10, 10), (20 + w, 10), (10, 20 + h), (20 + w, 20 + h))

def drag_to(card, board_point):
    QtTest.QTest.mousePress(card, QtCore.Qt.LeftButton, pos=QtCore.QPoint(18, 18))
    QtTest.QTest.mouseMove(card, card.mapFrom(board, QtCore.QPoint(*board_point)))
    app.processEvents()

def drop_at(card, board_point):
    QtTest.QTest.mouseRelease(
        card, QtCore.Qt.LeftButton, pos=card.mapFrom(board, QtCore.QPoint(*board_point))
    )
    app.processEvents()

def places(ids):
    by_id = {card.panel_id: card for card in cards}
    return tuple(by_id[panel_id].geometry().getRect()[:2] for panel_id in ids)

drag_to(cards[0], (w + 60, h + 60))
assert cards[0].pos() == QtCore.QPoint(w + 42, h + 42)
# Over panel-3: panel-3 already stands in the place panel-0 left, and the
# other two have not moved.
assert places(('panel-3', 'panel-1', 'panel-2')) == slots[:3]
assert board._drop_rect == QtCore.QRect(20 + w, 20 + h, w, h), board._drop_rect
drop_at(cards[0], (w + 60, h + 60))
assert board._order == ('panel-3', 'panel-1', 'panel-2', 'panel-0')
assert places(board._order) == slots and board._drop_rect is None

# Over panel-2: the two change places.
drag_to(cards[0], (35, 2 * h))
assert places(('panel-3', 'panel-1', 'panel-2')) == (slots[0], slots[1], slots[3])
assert board._drop_rect == QtCore.QRect(10, 20 + h, w, h), board._drop_rect
drop_at(cards[0], (35, 2 * h))
assert board._order == ('panel-3', 'panel-1', 'panel-0', 'panel-2')
assert places(board._order) == slots
"""
    )


def test_board_drag_raises_the_active_card_above_an_overlapping_sibling(run_qt) -> None:
    run_qt(
        """
from PyQt5 import QtCore, QtTest
from zlc_ui.board import BoardMetrics
from zlc_ui.console import ConsoleBoardView, PanelCardView
from zlc_ui.qt import ensure_qt_app
app = ensure_qt_app(['drag-stack'])
metrics = BoardMetrics(10)
first, second = PanelCardView('first'), PanelCardView('second')
w, h = first.size().width(), first.size().height()
board = ConsoleBoardView(metrics=metrics); board.resize(w + 20, 2 * h + 30)
board.set_cards((first, second)); board.show(); app.processEvents()
first.move(second.pos())
second.raise_()
def top_card_at(point):
    widget = board.childAt(point)
    while widget is not None and widget.parentWidget() is not board:
        widget = widget.parentWidget()
    return widget
assert top_card_at(first.geometry().center()) is second
QtTest.QTest.mousePress(first, QtCore.Qt.LeftButton, pos=QtCore.QPoint(18, 18))
app.processEvents()
assert top_card_at(first.geometry().center()) is first
QtTest.QTest.mouseRelease(first, QtCore.Qt.LeftButton, pos=QtCore.QPoint(18, 18))
"""
    )


def test_board_reuses_cards_and_reflows_on_resize(run_qt) -> None:
    run_qt(
        """
from zlc_ui.qt import ensure_qt_app
from zlc_ui.board import BoardMetrics
from zlc_ui.console import ConsoleBoardView, PanelCardView
app = ensure_qt_app(['test'])
metrics = BoardMetrics(10)
cards = tuple(PanelCardView(f'panel-{index}') for index in range(3))
w, h = cards[0].size().width(), cards[0].size().height()
board = ConsoleBoardView(metrics=metrics); board.resize(2 * w + 3 * 10, 2 * h + 3 * 10)
board.set_cards(cards); board.show(); app.processEvents()
identities = tuple(board._cards.values())
assert cards[0].geometry().x() == 10 and cards[1].geometry().x() == 20 + w
board.set_cards(cards); app.processEvents()
assert tuple(board._cards.values()) == identities
board.resize(w + 20, 4 * h); app.processEvents()
assert all(card.geometry().x() == 10 for card in cards)
"""
    )


def test_logic_row_projects_its_state_and_raises_its_intents(run_qt) -> None:
    run_qt(
        """
from PyQt5 import QtCore, QtTest
from zlc_ui.qt import ensure_qt_app
from zlc_ui.console import LogicRowView
app = ensure_qt_app(['test'])
row = LogicRowView('Processor', 'processor'); row.resize(1000, 48); row.show(); app.processEvents()
start, stop = row.start_button, row.stop_button
assert not start.isEnabled() and start.text() == 'Start'
# Stop is always there, idle or not -- disabled until there is something to stop.
assert stop.isVisible() and not stop.isEnabled() and stop.role() == 'destructive'
# Remove is on the row, not behind a menu.
assert row.remove_button.isVisible() and row.remove_button.isEnabled()
slots = lambda: (start.x(), start.width(), stop.x(), stop.width(), row.edit_button.x())
idle = slots()
row.set_commands(can_start=True, can_stop=True)
row.set_state('running', 'processing fake input')
row.set_publishes((('out', '1', 'fake output'),))
app.processEvents()
assert row.status_label.text() == 'processing fake input'
# Running, Start says what it now does; neither slot moves.
assert start.text() == 'Restart' and start.isEnabled() and stop.isEnabled()
assert slots() == idle, (slots(), idle)
assert 'out' in row.publishes_label.text()
# A start queued behind a device release: Starting…, and Stop cancels it.
row.set_commands(can_start=False, can_stop=True, pending=True)
app.processEvents()
assert start.text() == 'Starting…' and not start.isEnabled() and stop.isEnabled()
assert slots() == idle, (slots(), idle)
# A failed processor that still follows its source has something to stop.
row.set_state('error', "failed; restarts with the source's next run")
row.set_commands(can_start=True, can_stop=True)
assert start.text() == 'Restart' and start.isEnabled() and stop.isEnabled()
events = []
row.start_requested.connect(lambda: events.append('start'))
row.stop_requested.connect(lambda: events.append('stop'))
row.edit_requested.connect(lambda: events.append('edit'))
row.remove_requested.connect(lambda: events.append('remove'))
QtTest.QTest.mouseClick(start, QtCore.Qt.LeftButton)
QtTest.QTest.mouseClick(stop, QtCore.Qt.LeftButton)
# A draft that cannot start is a state of its own, drawn as the node's Edit
# page draws it -- a word the row refused would raise out of a Qt slot.
from zlc_ui.console.logic_row_view import STATE_COLOURS
row.set_state('draft', 'Draft: select a compatible source')
dot, ink = STATE_COLOURS['draft']
assert dot in row.dot.styleSheet() and ink in row.status_label.styleSheet()
row.set_state('idle', 'stopped')
row.set_commands(can_start=True, can_stop=False)
app.processEvents()
assert start.text() == 'Start' and not stop.isEnabled() and stop.isVisible()
assert slots() == idle, (slots(), idle)
QtTest.QTest.mouseClick(start, QtCore.Qt.LeftButton)
QtTest.QTest.mouseClick(row.edit_button, QtCore.Qt.LeftButton)
QtTest.QTest.mouseClick(row.remove_button, QtCore.Qt.LeftButton)
assert events == ['start', 'stop', 'start', 'edit', 'remove']
"""
    )


def test_logic_editor_is_a_live_closable_draft_projection(run_qt) -> None:
    run_qt(
        """
from PyQt5 import QtCore, QtTest
from zlc_ui.console import TaskConsoleHandle, TaskConsoleView
from zlc_ui.form import FormFieldProps, FormSpec
from zlc_ui.qt import ensure_qt_app
app = ensure_qt_app(['logic-editor'])
view = TaskConsoleView()
handle = TaskConsoleHandle(None, view)
handle.set_panel_sizes(('2x2',), '2x2')
handle.add_logic_row('camera-1', 'measurement')
handle.set_panel_publishers((('panel-1', (
    ('center_x', '35', 'held · @logic/panel-1/center_x'),
    ('roi', '4', 'live · @logic/panel-1/roi'),
)),))
projection = {
    'node_id': 'camera-1',
    'api_name': 'camera_measurement',
    'kind': 'measurement',
    'form_spec': FormSpec((FormFieldProps(key='repeat', kind='int', label='Repeat', default=0, minimum=0),)),
    'form_values': {'repeat': 0},
    'artifact_form_spec': FormSpec((
        FormFieldProps(
            key='calibration_path', kind='path', label='Calibration Path',
            required=True, file_filter='JSON files (*.json)', base_dir='C:/data',
        ),
    )),
    'artifact_values': {'calibration_path': 'C:/data/calibration.json'},
    'auto_preview': True,
    'preview_offered': True,
    'artifact_results': ({
        'name': 'artifact_path', 'contract_id': 'calibration.readout',
        'path': 'C:/data/calibration-2.json',
    },),
    'source_required': True,
    'source_signal': '',
    'source_options': ('camera-1.frames',),
    'source_labels': {'camera-1.frames': 'frames  [1 × 1 × (3×96×128)]'},
    'source_groups': {'camera-1.frames': 'camera-1'},
    'device_keys': {'camera': 'camera'},
    'device_options': {'camera': ('camera', 'mot_camera')},
    'device_labels': {'camera': 'Science camera', 'mot_camera': 'MOT monitor'},
    'running': False,
    'pending': False,
    'can_start': False,
    'can_stop': False,
    'issues': ('select a compatible source',),
    'error': '',
}
patches = []
def accept_and_refresh(node_id, patch):
    patches.append((node_id, patch))
    if 'values' in patch:
        projection['form_values'].update(patch['values'])
    if 'artifact_inputs' in patch:
        projection['artifact_values'].update(patch['artifact_inputs'])
    if 'source_signal' in patch:
        projection['source_signal'] = patch['source_signal']
    if 'device_keys' in patch:
        projection['device_keys'].update(patch['device_keys'])
    handle.update_logic_editor(node_id, projection)
handle.logic_draft_changed.connect(accept_and_refresh)
handle.open_logic_editor('camera-1', projection)
editor = handle._logic_editors['camera-1']
# The frame-signal picker is the SAME grouped tree every signal chooser uses:
# a producer header with keyed leaves, not a flat list.
from zlc_ui.fluent import FluentComboBox, FluentTreeComboBox
assert isinstance(editor.source_combo, FluentTreeComboBox)
assert editor.source_combo.model().item(0).text() == '(not selected)'
source_header = editor.source_combo.model().item(1)
assert source_header.text() == 'camera-1'
assert source_header.child(0).text() == 'frames  [1 × 1 × (3×96×128)]'
# The draft refuses without its signal: the row carries the column's required
# marker, and its blank choice reads as a missing input, not a value.
assert editor._selector_rows[editor.source_combo][1] == 'Signal *'
assert editor.source_combo.current_is_blank()
handle.set_logic_commands('camera-1', can_start=False, can_stop=False)
view.show()
app.processEvents()
assert editor.artifact_form.widget_for('calibration_path').mapTo(editor, QtCore.QPoint()).x() == editor.form.widget_for('repeat').mapTo(editor, QtCore.QPoint()).x()
# ONE RHYTHM AND ONE PAGE EDGE.  Selectors, both forms and the results are
# one run of rows a form gap apart, whichever widget holds them; a section
# gap follows the header and precedes the action bar; every control starts
# on one column; and the content sits a window pad inside the page on both
# sides, the scrollbar's gutter held although no bar shows.  The header and
# the action bar stand outside the scrolling rows, on the same edges.
from zlc_ui.fluent import FluentScrollArea, FluentSettingRow, section_gap, window_pad
from zlc_ui.fluent import row_gap
top_of = lambda widget: widget.mapTo(editor, QtCore.QPoint()).y()
left_of = lambda widget: widget.mapTo(editor, QtCore.QPoint()).x()
rows = sorted((row for row in editor.findChildren(FluentSettingRow) if row.isVisible()), key=top_of)
assert len(rows) == 5, [row.layout().itemAt(0).widget().text() for row in rows]
gaps = [top_of(below) - top_of(above) - above.height() for above, below in zip(rows, rows[1:])]
assert gaps == [row_gap()] * 4, gaps
assert top_of(rows[0]) - top_of(editor.status_label) - editor.status_label.height() == section_gap()
assert len({row.layout().itemAt(1).widget().mapTo(editor, QtCore.QPoint()).x() for row in rows}) == 1
left = rows[0].mapTo(editor, QtCore.QPoint()).x()
assert (left, editor.width() - left - rows[0].width()) == (window_pad(), window_pad())
scroll = editor.findChild(FluentScrollArea)
viewport_bottom = scroll.viewport().mapTo(editor, QtCore.QPoint(0, scroll.viewport().height())).y()
assert top_of(editor.remove_button) - viewport_bottom == section_gap()
assert editor.height() - top_of(editor.remove_button) - editor.remove_button.height() == window_pad()
assert left_of(editor.title_label) == left_of(editor.remove_button) == window_pad()
assert editor.width() - left_of(editor.stop_button) - editor.stop_button.width() == window_pad()
assert not scroll.isAncestorOf(editor.status_label) and not scroll.isAncestorOf(editor.start_button)
publisher = handle._panel_publisher_rows['panel-1']
assert tuple(view._logic_rows) == (handle._rows['camera-1'], publisher)
assert not publisher.publishes_label.isHidden()
assert all(button.isHidden() for button in (
    publisher.start_button, publisher.stop_button,
    publisher.preview_switch, publisher.remove_button,
))
assert not publisher.edit_button.isHidden()
# The hidden controls keep their places, so a publisher row's name and Edit
# stand in the node rows' columns.
view.tabs.setCurrentIndex(1)
app.processEvents()
node_row = handle._rows['camera-1']
at = lambda widget: widget.mapTo(view, QtCore.QPoint()).x()
assert at(publisher.edit_button) == at(node_row.edit_button), (
    at(publisher.edit_button), at(node_row.edit_button))
assert at(publisher.name_label) == at(node_row.name_label)
assert at(publisher.kind_label) == at(node_row.kind_label)
publisher_edits = []
handle.panel_publisher_edit_requested.connect(publisher_edits.append)
QtTest.QTest.mouseClick(publisher.edit_button, QtCore.Qt.LeftButton)
assert publisher_edits == ['panel-1']
publisher_projection = {
    'node_id': 'panel-1',
    'api_name': 'Image outputs',
    'kind': 'panel publisher',
    'form_spec': FormSpec((
        FormFieldProps(key='roi_mean', kind='bool', label='Mean', default=True),
        FormFieldProps(key='roi_max', kind='bool', label='Max', default=True),
    )),
    'form_values': {'roi_mean': True, 'roi_max': True},
    'source_required': False,
    'device_options': {}, 'device_keys': {},
    'preview_offered': False, 'auto_preview': False,
    'running': False, 'pending': False,
    'can_start': False, 'can_stop': False,
    'issues': (), 'error': '', 'status': '',
}
publisher_patches = []
handle.panel_publisher_draft_changed.connect(
    lambda panel_id, patch: publisher_patches.append((panel_id, patch))
)
handle.open_panel_publisher_editor('panel-1', publisher_projection)
publisher_editor = handle._panel_publisher_editors['panel-1']
assert publisher_editor.source_combo is None
# Nothing to start or remove on a publisher's page: no action bar at all.
assert not any(button.isVisibleTo(publisher_editor) for button in (
    publisher_editor.start_button, publisher_editor.stop_button, publisher_editor.remove_button,
))
publisher_editor.form.widget_for('roi_max').click()
assert publisher_patches == [('panel-1', {'values': {'roi_max': False}})]
assert view.tabs.indexOf(editor) >= 0
assert view.tabs.indexOf(publisher_editor) >= 0
assert view.tabs.currentWidget() is publisher_editor
assert not editor.start_button.isEnabled()
assert not editor.stop_button.isHidden() and not editor.stop_button.isEnabled()
assert not handle._rows['camera-1'].start_button.isEnabled()
assert not handle._rows['camera-1'].stop_button.isEnabled()
editor.form.widget_for('repeat').setValue(3)
artifact_picker = editor.artifact_form.widget_for('calibration_path')
assert artifact_picker.browse.text() == 'Browse…'
artifact_picker.setText('C:/data/manual.json')
def pick_from_popup(combo, value):
    index = combo.findData(value)
    assert index >= 0
    combo.showPopup()
    app.processEvents()
    model_index = combo.model().index(index, combo.modelColumn())
    rect = combo.view().visualRect(model_index)
    assert rect.isValid() and not rect.isEmpty()
    QtTest.QTest.mouseClick(
        combo.view().viewport(), QtCore.Qt.LeftButton,
        QtCore.Qt.NoModifier, rect.center(),
    )
source_combo = editor.source_combo
source_model_events = []
source_combo.model().rowsRemoved.connect(
    lambda *_args: source_model_events.append('removed')
)
source_combo.model().rowsInserted.connect(
    lambda *_args: source_model_events.append('inserted')
)
source_combo.showPopup()
app.processEvents()
header_index = source_combo.model().index(1, 0)
source_combo.view().expand(header_index)
app.processEvents()
leaf_index = source_combo.model().index(0, 0, header_index)
leaf_rect = source_combo.view().visualRect(leaf_index)
assert leaf_rect.isValid() and not leaf_rect.isEmpty()
QtTest.QTest.mouseClick(
    source_combo.view().viewport(), QtCore.Qt.LeftButton,
    QtCore.Qt.NoModifier, leaf_rect.center(),
)
assert editor.source_combo is source_combo
assert source_combo.current_choice_key() == 'camera-1.frames'
assert not source_model_events
camera_combo = editor._device_combos['camera']
assert camera_combo.currentText() == 'Science camera'
camera_model_events = []
camera_combo.model().rowsRemoved.connect(
    lambda *_args: camera_model_events.append('removed')
)
camera_combo.model().rowsInserted.connect(
    lambda *_args: camera_model_events.append('inserted')
)
pick_from_popup(camera_combo, 'mot_camera')
assert camera_combo.currentText() == 'MOT monitor'
assert editor._device_combos['camera'] is camera_combo
assert not camera_model_events
app.processEvents()
assert ('camera-1', {'values': {'repeat': 3}}) in patches
assert ('camera-1', {'artifact_inputs': {'calibration_path': 'C:/data/manual.json'}}) in patches
assert ('camera-1', {'source_signal': 'camera-1.frames'}) in patches
assert ('camera-1', {'device_keys': {'camera': 'mot_camera'}}) in patches
editor.form.widget_for('repeat').valueNormalized.emit()
assert patches[-1] == ('camera-1', {'values': {'repeat': 3}, 'normalized': True})
# A number the form cannot read yet is no value: the draft keeps its last
# one and the header keeps its own line -- no raw read error in it.
status_line = editor.status_label.text()
repeat_box = editor.form.widget_for('repeat')
repeat_box.setProperty('numericError', 'the number does not fit the box')
sent = len(patches)
editor.form.changed.emit('repeat')
assert len(patches) == sent and editor.status_label.text() == status_line
repeat_box.setProperty('numericError', '')
# An emptied path is the vacancy the draft holds, for Start's admission to
# name in the header as a draft problem.
artifact_picker.setText('')
assert patches[-1] == ('camera-1', {'artifact_inputs': {'calibration_path': ''}}), patches[-1]
artifact_picker.setText('C:/data/manual.json')
readout = editor._artifact_result_readouts['artifact_path']
assert readout.isReadOnly()
assert readout.text() == 'C:/data/calibration-2.json'
# The result path starts on the fields' text x, once: the readout's own
# inset is a field's, and the page adds none of its own.
from PyQt5 import QtWidgets
def text_x(edit):
    option = QtWidgets.QStyleOptionFrame()
    option.initFrom(edit)
    contents = edit.style().subElementRect(QtWidgets.QStyle.SE_LineEditContents, option, edit)
    return edit.mapTo(editor, QtCore.QPoint()).x() + contents.left() + edit.textMargins().left()
assert text_x(readout) == text_x(artifact_picker.edit), (text_x(readout), text_x(artifact_picker.edit))
# A problem the node's own editor shows in red under its grid is counted in
# the header, not said again in grey; one nothing below shows comes first.
claims = {'calibrations': {'label': 'Calibrations', 'problems': ('frame 1 has no file', 'b.json is listed twice')}}
handle.update_logic_editor('camera-1', dict(projection, artifact_claims=claims, issues=claims['calibrations']['problems']))
assert editor.status_label.text() == 'Draft: 2 problems in Calibrations'
handle.update_logic_editor('camera-1', dict(
    projection, artifact_claims=claims, issues=('Frames must be selected', *claims['calibrations']['problems'])))
assert editor.status_label.text() == 'Draft: Frames must be selected'
# So is one a contribution states under the line it is about (Derive's
# typing errors), by the section it names.
from PyQt5 import QtWidgets
typed = ('bright · where: the mask does not fit the value', 'same · combine: booleans do not mix with numbers')
class Typed(QtWidgets.QWidget):
    draft_changed = QtCore.pyqtSignal(object)
    def update_projection(self, projection):
        pass
    def shown_problems(self):
        return {problem: 'Outputs' for problem in typed}
def typed_editor(parent):
    return Typed(parent)
handle.update_logic_editor('camera-1', dict(projection, ui_contributions=(typed_editor,), issues=typed))
assert editor.status_label.text() == 'Draft: 2 problems in Outputs'
handle.update_logic_editor('camera-1', dict(
    projection, ui_contributions=(typed_editor,), issues=('Frames must be selected', *typed)))
assert editor.status_label.text() == 'Draft: Frames must be selected'

patches.clear()
running = dict(
    projection,
    form_values={'repeat': 3},
    running=True,
    can_start=True,
    can_stop=True,
    issues=(),
)
handle.set_logic_commands('camera-1', can_start=True, can_stop=True)
handle.update_logic_editor('camera-1', running)
assert not patches, patches
assert editor.start_button.text() == 'Restart' and editor.start_button.isEnabled()
assert editor.stop_button.isEnabled()
# The row says the same thing about the same node.
assert handle._rows['camera-1'].start_button.text() == 'Restart'
assert handle._rows['camera-1'].stop_button.isEnabled()
# Mode-dependent inputs use this same editor and its ordinary Fluent forms.
from zlc_atom.nodes.slm_feedback.logic_node import LOGIC_NODE as feedback
from zlc_workbench.authoring_form import project_logic_schema, display_value
values = {field.name: display_value(field.default) for field in feedback.authoring_schema.fields}
mode_projection = dict(running,
    form_spec=project_logic_schema(feedback, workspace_root='C:/data', acquisition_enabled=False),
    form_values=values, acquisition_input='acquisition_logic', acquisition_enabled=False,
    source_required=False, device_options={'camera': ('camera',), 'sequencer': ('pulse',), 'slm': ('slm',)},
)
handle.update_logic_editor('camera-1', mode_projection)
assert editor.form.widget_for('pulse_template').isEnabled()
assert editor.form.widget_for('exposure_seconds').isEnabled()
assert not editor._acquisition_combo.isEnabled() and editor.source_combo is None
mode_projection = dict(mode_projection,
    form_spec=project_logic_schema(feedback, workspace_root='C:/data', acquisition_options=('scan',)),
    form_values=dict(values, feedback_mode='ramsey_frequency'),
    acquisition_enabled=True, source_required=True, device_options={'slm': ('slm',)},
)
handle.update_logic_editor('camera-1', mode_projection)
assert not editor.form.widget_for('pulse_template').isEnabled()
assert not editor.form.widget_for('exposure_seconds').isEnabled()
assert editor._acquisition_combo.isEnabled() and editor.source_combo is not None
assert set(editor._device_combos) == {'slm'}
# A device argument reads as words, its acronym kept: not "Slm".
assert editor._selector_rows[editor._device_combos['slm']][1] == 'SLM'
view.editor_close_requested.emit(editor)
app.processEvents()
assert view.tabs.indexOf(editor) < 0
assert view.tabs.indexOf(publisher_editor) >= 0
assert 'camera-1' not in handle._logic_editors
assert tuple(handle._rows) == ('camera-1',)
view.editor_close_requested.emit(publisher_editor)
app.processEvents()
assert 'panel-1' not in handle._panel_publisher_editors
"""
    )


def test_a_logic_node_is_renamed_on_its_edit_page_and_called_by_its_name(run_qt) -> None:
    run_qt(
        """
from PyQt5 import QtCore
from zlc_ui.console import TaskConsoleHandle, TaskConsoleView
from zlc_ui.fluent import FluentSettingRow
from zlc_ui.form import FormFieldProps, FormSpec
from zlc_ui.qt import ensure_qt_app
app = ensure_qt_app(['logic-name'])
view = TaskConsoleView()
handle = TaskConsoleHandle(None, view)
handle.set_panel_sizes(('2x2',), '2x2')
# A row is made blank; what the node is called arrives by one channel.
handle.add_logic_row('occupancy2', 'processor')
row = handle._rows['occupancy2']
assert row.name_label.text() == ''
handle.set_logic_name('occupancy2', 'Loading')
assert row.name_label.text() == 'Loading' and row.name_label.toolTip() == 'Loading'
projection = {
    'node_id': 'occupancy2', 'name': 'Loading', 'name_problem': '',
    'api_name': 'occupancy', 'kind': 'processor',
    'form_spec': FormSpec((FormFieldProps(key='repeat', kind='int', label='Repeat', default=0, minimum=0),)),
    'form_values': {'repeat': 0},
    'source_required': True, 'source_label': 'Frames', 'source_signal': '@logic/camera_measurement/frames',
    'source_options': ('@logic/camera_measurement/frames',),
    'source_labels': {'@logic/camera_measurement/frames': 'frames'},
    'source_groups': {'@logic/camera_measurement/frames': 'Camera A'},
    'device_options': {}, 'device_keys': {},
    'preview_offered': False, 'auto_preview': False,
    'running': True, 'pending': False, 'can_start': False, 'can_stop': True,
    'issues': (), 'error': '', 'status': 'following Camera A/frames', 'state': 'running',
}
sent = []
def answer(node_id, text):
    # The presenter's part: refuse a slash, otherwise take the name.
    sent.append((node_id, text))
    if '/' in text:
        projection['name_problem'] = 'A name is one line of text without “/”.'
    else:
        projection.update(name=text, name_problem='')
    handle.update_logic_editor(node_id, projection)
handle.logic_name_committed.connect(answer)
handle.open_logic_editor('occupancy2', projection)
editor = handle._logic_editors['occupancy2']
view.resize(1000, 700)
view.show()
app.processEvents()
tab = lambda: view.tabs.tabText(view.tabs.indexOf(editor))
assert tab() == 'Edit · Loading'
assert editor.title_label.text() == 'Loading'
assert editor.kind_label.text() == '(occupancy · processor)'
# The Name is the first row, in the one label column and the one control column.
top = lambda widget: widget.mapTo(editor, QtCore.QPoint()).y()
left = lambda widget: widget.mapTo(editor, QtCore.QPoint()).x()
rows = sorted((r for r in editor.findChildren(FluentSettingRow) if r.isVisible()), key=top)
box = editor.name_form.widget_for('name')
assert rows[0].isAncestorOf(box), [r.layout().itemAt(0).widget().text() for r in rows]
assert editor.name_form._labels['name'].text() == 'Name'
assert left(box) == left(editor.source_combo) == left(editor.form.widget_for('repeat'))
assert box.text() == 'Loading' and box.toolTip() == 'Internal ID: occupancy2'
# Finishing the box (Return, then the focus leaving) sends the name once,
# stripped; the page and its tab follow the presenter's answer.
box.setText('  Counts ')
box.editingFinished.emit()
box.editingFinished.emit()
assert sent == [('occupancy2', 'Counts')], sent
assert tab() == 'Edit · Counts' and editor.title_label.text() == 'Counts'
assert box.text() == 'Counts'
# Leaving an unchanged box sends nothing.
box.editingFinished.emit()
assert sent == [('occupancy2', 'Counts')]
# A refusal is one sentence under the line, from the control column; the box
# goes back to the name the node still has, and the sentence stays.
assert editor.name_problem_label.isHidden()
below = top(editor.source_combo)
box.setText('a/b')
box.editingFinished.emit()
app.processEvents()
assert sent[-1] == ('occupancy2', 'a/b')
assert editor.name_problem_label.isVisible()
assert editor.name_problem_label.text() == 'A name is one line of text without “/”.'
assert left(editor.name_problem_label) == left(box)
assert top(box) < top(editor.name_problem_label) < top(editor.source_combo)
assert top(editor.source_combo) > below, 'the sentence takes its own line'
assert box.text() == 'Counts' and tab() == 'Edit · Counts'
# Answering a standing refusal with the name the node has clears it.
box.editingFinished.emit()
app.processEvents()
assert sent[-1] == ('occupancy2', 'Counts')
assert editor.name_problem_label.isHidden() and tab() == 'Edit · Counts'
box.setText('Loading')
box.editingFinished.emit()
app.processEvents()
assert tab() == 'Edit · Loading'
# A Task holding the console leaves the Name editable.
handle.set_task_takeover('another_task')
assert box.isEnabled() and not editor.form.isEnabled()
handle.set_task_takeover('')
# A panel publisher's page has nothing to rename.
handle.set_panel_publishers((('panel-1', (('roi', '4', 'live · panel-1/roi'),)),))
handle.open_panel_publisher_editor('panel-1', {
    'node_id': 'panel-1', 'api_name': 'Camera A/frames outputs', 'kind': 'panel publisher',
    'form_spec': FormSpec((FormFieldProps(key='roi_mean', kind='bool', label='Mean', default=True),)),
    'form_values': {'roi_mean': True}, 'source_required': False,
    'device_options': {}, 'device_keys': {},
    'preview_offered': False, 'auto_preview': False,
    'running': False, 'pending': False, 'can_start': False, 'can_stop': False,
    'issues': (), 'error': '', 'status': '',
})
publisher = handle._panel_publisher_editors['panel-1']
assert not publisher._name_frame.isVisibleTo(publisher)
assert view.tabs.tabText(view.tabs.indexOf(publisher)) == 'Edit · Camera A/frames outputs'
"""
    )


def test_panel_editor_and_setting_are_views_of_the_same_projection(run_qt) -> None:
    run_qt(
        """
from pathlib import Path
from tempfile import TemporaryDirectory
from PyQt5 import QtCore, QtGui, QtTest, QtWidgets
from zlc_ui.console import TaskConsoleHandle, TaskConsoleView
from zlc_ui.fluent import FluentComboBox, FluentSegmentedButtons, FluentTreeComboBox
from zlc_ui.form import FormFieldProps, FormSpec
from zlc_ui.qt import ensure_qt_app
app = ensure_qt_app(['panel-editor'])
temporary = TemporaryDirectory()
save_directory = temporary.name
view = TaskConsoleView()
handle = TaskConsoleHandle(None, view)
handle.set_panel_sizes(('1x2', '2x2', '1x4'), '2x2')
handle.set_panel_intervals((100, 200, 400, 800), 400)
handle.add_panel('panel-1', 'Camera')
groups = (('camera', (('frames', '@logic/cm/frames'),)),)
overlay_groups = (('occupancy', (('sites', '@logic/occ/site_overlay'),)),)
handle.set_panel_signal_choices(
    'panel-1', groups, current='@logic/cm/frames',
    overlay_groups=overlay_groups,
    overlay_current='@logic/occ/site_overlay',
)
state = {
    'signal': '@logic/cm/frames', 'kind': 'image', 'size': '2x2',
    'interval_ms': 100, 'title': 'Camera',
    'semantic': {}, 'display': {}, 'fit': {},
    'overlay_signal': '@logic/occ/site_overlay',
}
surface = {
    'paints_images': True,
    'semantic': ({
        'key': 'x', 'label': 'X axis', 'kind': 'choice', 'value': 'sensor_x',
        'allow_none': False,
        'choices': (('Sensor X', 'sensor_x'), ('Point axis', 'point')),
        'minimum': None, 'maximum': None, 'step': None,
    },),
    'display': (
        {'key': 'title', 'label': 'Title', 'kind': 'text',
         'value': None, 'allow_none': True, 'choices': (),
         'minimum': None, 'maximum': None, 'step': None,
         'automatic': True, 'unavailable_reason': ''},
        {'key': 'x_label', 'label': 'X label', 'kind': 'text',
         'value': None, 'allow_none': True, 'choices': (),
         'minimum': None, 'maximum': None, 'step': None,
         'automatic': True, 'unavailable_reason': ''},
        {'key': 'x_display_unit', 'label': 'X unit', 'kind': 'choice',
         'value': None, 'allow_none': True,
         'choices': (('Pixel', 'pixel'), ('Millimetre', 'mm')),
         'minimum': None, 'maximum': None, 'step': None,
         'automatic': True, 'unavailable_reason': ''},
        # co_edited_with is the workbench's join: a limit pair is validated
        # as a pair, so an editor sends both ends.  Written out here because
        # this surface is hand-built; the real one comes from
        # zlc_workbench.panel_state.control_document, which is where the
        # join is pinned.
        {'key': 'color_min', 'label': 'Color minimum', 'kind': 'number',
         'value': None, 'allow_none': True, 'choices': (),
         'minimum': None, 'maximum': None, 'step': None,
         'automatic': False, 'co_edited_with': 'color_max',
         'unavailable_reason': 'Choose Fixed color limits to edit.'},
        {'key': 'color_max', 'label': 'Color maximum', 'kind': 'number',
         'value': None, 'allow_none': True, 'choices': (),
         'minimum': None, 'maximum': None, 'step': None,
         'automatic': False, 'co_edited_with': 'color_min',
         'unavailable_reason': 'Choose Fixed color limits to edit.'},
        {'key': 'colormap', 'label': 'Colormap', 'kind': 'choice',
         'value': 'viridis', 'allow_none': False,
         'choices': (('Viridis', 'viridis'), ('Magma', 'magma')),
         'minimum': None, 'maximum': None, 'step': None},
        {'key': 'show_colorbar', 'label': 'Colorbar', 'kind': 'boolean',
         'value': True, 'allow_none': False, 'choices': (),
         'minimum': None, 'maximum': None, 'step': None},
    ),
    'fit': ({
        'key': 'model', 'label': 'Fit model', 'kind': 'choice', 'value': None,
        'allow_none': True,
        'choices': (('2-D Gaussian', 'anisotropic_gaussian_center'),),
        'minimum': None, 'maximum': None, 'step': None,
    },),
}
handle.set_panel_projection('panel-1', state, surface)
card = handle._cards['panel-1']
blank_surface = {
    'paints_images': False,
    'semantic': (), 'display': (), 'fit': (),
    'semantic_unavailable': '', 'display_unavailable': '',
    'fit_unavailable': '',
}
handle.set_panel_projection('panel-1', state, blank_surface)
card.resize(340, 180)
authored_size = card.size()
# The Setting frame is a child of the panel page now: like any child it is
# only visible inside a shown window, which is the one an operator ever
# clicks the button in.
view.resize(1400, 900)
view.show()
app.processEvents()
card._open_settings()
assert card._settings_popup.isVisible()
blank_popup_width = card._settings_popup.width()
handle.set_panel_projection('panel-1', state, surface)
app.processEvents()
assert card.size() == authored_size, 'same-size projection reset the card geometry'
assert card._settings_popup.isVisible()
card._settings_body.layout().activate()
display_form = card._settings_parameter_forms['display']
bound_editor = display_form.widget_for('colormap')
bound_editor_right = bound_editor.mapTo(
    card._settings_scroll.viewport(),
    QtCore.QPoint(bound_editor.width() - 1, 0),
).x()
assert bound_editor_right < card._settings_scroll.viewport().width(), (
    'an already-open Setting popup kept its blank-schema width after binding: '
    f'blank={blank_popup_width}, bound={card._settings_popup.width()}, '
    f'editor-right={bound_editor_right}, '
    f'viewport={card._settings_scroll.viewport().width()}'
)
# The Edit page's sections, each under its title a section gap from the
# last, every control on ONE column across them.
setting_rows = lambda: [
    (section, key, form) for section, form in (
        ('panel', card._settings_form), *card._settings_parameter_forms.items())
    for key in form.spec.keys
]
for section, key, form in setting_rows():
    row = form._rows[key]
    assert row.height() >= row.minimumSizeHint().height(), (
        f'Setting row {key!r} was vertically cut off: '
        f'{row.height()} < {row.minimumSizeHint().height()}'
    )
# One control column across the sections: an Auto row's cell (its switch,
# then its editor) starts where every other row's control does, and every
# row's words start at the one label edge.
def control_edges():
    return {
        (section, key): form.cell_for(key).mapTo(card._settings_body, QtCore.QPoint(0, 0)).x()
        for section, key, form in setting_rows()
    }
parameter_left_edges = control_edges()
assert len(set(parameter_left_edges.values())) == 1, parameter_left_edges
assert display_form.cell_for('title').isAncestorOf(display_form.auto_switch_for('title'))
label_left_edges = {
    (section, key): form._labels[key].mapTo(card._settings_body, QtCore.QPoint(0, 0)).x()
    for section, key, form in setting_rows()
}
assert len(set(label_left_edges.values())) == 1, label_left_edges
label_slot_widths = {
    (section, key): form._rows[key].layout().itemAt(0).widget().width()
    for section, key, form in setting_rows()
}
assert len(set(label_slot_widths.values())) == 1, label_slot_widths
from zlc_ui.fluent import control_height, row_gap, section_gap
headings = [card._settings_sections[name] for name in ('panel', 'semantic', 'display', 'fit')]
section_forms = card._settings_forms()
assert [heading.isVisible() for heading in headings] == [True, True, True, True]
assert [heading.text() for heading in headings] == ['Panel', 'Semantic', 'Display', 'Fit']
for heading, form in zip(headings, section_forms):
    assert heading.height() == control_height(), heading.height()
    assert form.y() - heading.y() - heading.height() == row_gap(), (heading.geometry(), form.geometry())
for form, heading in zip(section_forms, headings[1:]):
    assert heading.y() - form.y() - form.height() == section_gap(), (form.geometry(), heading.geometry())
# The Panel section names its size as the Edit page does.
assert card._settings_form._labels['size'].text() == 'Panel size'
def setting_geometry():
    return {
        (section, key): (
            QtCore.QRect(form._rows[key].mapTo(card._settings_body, QtCore.QPoint(0, 0)),
                         form._rows[key].size()),
            form.widget_for(key).geometry(),
        )
        for section, key, form in setting_rows()
    }
stable_geometry = setting_geometry()
card._settings_form.widget_for('interval_ms').button(800).click()
state = dict(state, interval_ms=800)
handle.set_panel_projection('panel-1', state, surface)
app.processEvents()
projected_geometry = setting_geometry()
assert projected_geometry == stable_geometry, {
    key: (stable_geometry[key], projected_geometry[key])
    for key in stable_geometry
    if stable_geometry[key] != projected_geometry[key]
}
# Marking a segment re-polishes it, and the layout requests that follow
# settle over a few turns of the loop; the probe below is about the
# projection, not the click before it.
QtTest.QTest.qWait(50)
class LayoutProbe(QtCore.QObject):
    def __init__(self):
        super().__init__()
        self.requests = 0
    def eventFilter(self, watched, event):
        if event.type() == QtCore.QEvent.LayoutRequest:
            self.requests += 1
        return False
layout_probe = LayoutProbe()
for form in card._settings_forms():
    for watched in (form, *form._rows.values()):
        watched.installEventFilter(layout_probe)
for form in card._settings_forms():
    form.reconcile(form.spec, form.read_all())
app.processEvents()
assert layout_probe.requests == 0, (
    'a value-only Setting projection rebuilt the form layout'
)
compact_surface = dict(
    surface,
    semantic=(),
    display=tuple(
        field for field in surface['display']
        if field['key'] in {'title', 'x_label', 'colormap'}
    ),
    fit=(),
)
handle.set_panel_projection('panel-1', state, compact_surface)
card._settings_popup.hide()
app.processEvents()
card._open_settings()
app.processEvents()
compact_left_edges = control_edges()
assert len(set(compact_left_edges.values())) == 1, compact_left_edges
assert not card._settings_sections['semantic'].isVisible()
assert not card._settings_sections['fit'].isVisible()
handle.set_panel_projection('panel-1', state, surface)
title_widget = card._settings_form.widget_for('title')
natural_title_width = title_widget.minimumWidth()
title_widget.setMinimumWidth(520)
card._settings_popup.hide()
app.processEvents()
card._open_settings()
card._settings_body.layout().activate()
editor_right = title_widget.mapTo(
    card._settings_scroll.viewport(),
    QtCore.QPoint(title_widget.width() - 1, 0),
).x()
assert editor_right < card._settings_scroll.viewport().width(), (
    'Setting cut off the editor: '
    f'popup={card._settings_popup.width()} '
    f'viewport={card._settings_scroll.viewport().width()} '
    f'editor-right={editor_right}'
)
title_widget.setMinimumWidth(natural_title_width)
assert isinstance(card._settings_form.widget_for('signal'), FluentTreeComboBox)
assert isinstance(card._settings_form.widget_for('overlay_signal'), FluentTreeComboBox)
assert card._settings_form.widget_for('signal').current_choice_key() == '@logic/cm/frames'
assert card._settings_form.widget_for('overlay_signal').current_choice_key() == '@logic/occ/site_overlay'
assert card._settings_form.widget_for('signal')._display_text() == 'camera · frames'
assert card._settings_form.widget_for('overlay_signal')._display_text() == 'occupancy · sites'
assert card._settings_form.widget_for('overlay_signal').model().item(0).text() == 'Off'
replacement_groups = (('camera', (('reference', '@logic/cm/reference'),)),)
replacement_state = dict(state, signal='@logic/cm/reference')
handle.set_panel_signal_choices(
    'panel-1', replacement_groups, current='@logic/cm/reference',
    overlay_groups=overlay_groups, overlay_current='@logic/occ/site_overlay',
)
handle.set_panel_projection('panel-1', replacement_state, surface)
assert card._settings_form.widget_for('signal').current_choice_key() == '@logic/cm/reference'
handle.set_panel_signal_choices(
    'panel-1', groups, current='@logic/cm/frames',
    overlay_groups=overlay_groups,
    overlay_current='@logic/occ/site_overlay',
)
handle.set_panel_projection('panel-1', state, surface)
assert 'kind' not in card._settings_form.read_all()
assert display_form.read_all()['show_colorbar'] is True
assert display_form.read_all()['colormap'] == 'viridis'
assert display_form.read_all()['title'] is None
assert display_form.auto_switch_for('title').isChecked()
assert not display_form.widget_for('title').isEnabled()
assert card._settings_parameter_forms['semantic'].read_all()['x'] == 'sensor_x'
assert card._settings_parameter_forms['fit'].widget_for('model').currentText() == '(none)'
assert card._settings_form.read_all()['overlay_signal'] == '@logic/occ/site_overlay'
# The scheduler's refresh policy is a closed set: every interval at once,
# each segment just its number, the unit said once by the row.
interval_choice = card._settings_form.widget_for('interval_ms')
assert isinstance(interval_choice, FluentSegmentedButtons)
assert [interval_choice.button(value).text() for value in (100, 200, 400, 800)] == [
    '100', '200', '400', '800',
]
assert card._settings_form.read_value('interval_ms') == state['interval_ms']
# The cell kind is a grid panel PARAMETER: an enabled choice whose empty
# entry means the data decides, emitting the same cell_kind state patch the
# presenter accepts.
handle.set_grid_cell_kinds(('curve', 'image', 'histogram'))
facet_state = dict(state, kind='facet_grid', cell_kind='curve', title='Site grid')
handle.set_panel_projection('panel-1', facet_state, surface)
cell_combo = card._settings_form.widget_for('cell_kind')
assert isinstance(cell_combo, FluentComboBox)
assert cell_combo.isEnabled()
assert tuple(cell_combo.itemData(index) for index in range(cell_combo.count())) == (
    '', 'curve', 'image', 'histogram',
)
assert card._settings_form.read_all()['cell_kind'] == 'curve'
cell_patches = []
capture_cell_patch = lambda patch: cell_patches.append(dict(patch))
card.state_changed.connect(capture_cell_patch)
cell_combo.setCurrentIndex(cell_combo.findData('image'))
cell_combo.activated.emit(cell_combo.currentIndex())
card.state_changed.disconnect(capture_cell_patch)
assert {'cell_kind': 'image'} in cell_patches
# Nothing took that pick.  A refusal comes back as the record the presenter
# kept -- the same projection -- and the open form shows the record again.
handle.set_panel_projection('panel-1', facet_state, surface)
assert card._settings_form.read_all()['cell_kind'] == 'curve'
handle.set_panel_projection('panel-1', state, surface)
assert 'cell_kind' not in card._settings_form.spec.keys, (
    'only a facet grid offers a cell kind')

projection = {
    'panel_id': 'panel-1', 'state': state, 'signal_options': groups,
    'overlay_signal_options': overlay_groups,
    'interval_choices': (100, 200, 400, 800),
    'parameter_surface': surface,
    'save_directory': save_directory,
    'frozen_signal': '@logic/cm/frames',
    'frozen_publication': object(), 'frozen_snapshot': object(),
    'stale': False, 'producer_node_id': 'cm',
}
events = []
handle.panel_state_changed.connect(
    lambda panel_id, patch: events.append(('state', panel_id, patch))
)
handle.logic_edit_requested.connect(
    lambda node_id: events.append(('edit_logic', node_id))
)
handle.panel_snapshot_refresh_requested.connect(
    lambda panel_id: events.append(('refresh', panel_id))
)
handle.panel_save_figure_requested.connect(
    lambda panel_id, path: events.append(('save', panel_id, path))
)
handle.panel_editor_closed.connect(
    lambda panel_id: events.append(('closed', panel_id))
)
card_semantic = card._settings_parameter_forms['semantic'].widget_for('x')
card_semantic.setCurrentIndex(1)
card_semantic.activated.emit(1)
card_fit = card._settings_parameter_forms['fit'].widget_for('model')
card_fit.setCurrentIndex(1)
card_fit.activated.emit(1)
fit_state = dict(
    state,
    fit={'model': 'anisotropic_gaussian_center'},
)
fit_surface = dict(
    surface,
    fit=(
        dict(
            surface['fit'][0],
            value='anisotropic_gaussian_center',
        ),
        {
            'key': 'expression', 'label': 'Parameters', 'kind': 'text',
            'value': '', 'allow_none': True, 'choices': (),
            'minimum': None, 'maximum': None, 'step': None,
            'description': 'name=value fixes; name=guess(value) initializes',
        },
    ),
)
handle.set_panel_projection('panel-1', fit_state, fit_surface)
fit_expression = card._settings_parameter_forms['fit'].widget_for('expression')
fit_expression.setText('center_x=10, radius_x=guess(4)')
fit_expression.editingFinished.emit()
assert ('state', 'panel-1', {
    'fit': {'expression': 'center_x=10, radius_x=guess(4)'}
}) in events
handle.set_panel_projection('panel-1', state, surface)
handle.open_panel_editor('panel-1', projection)
editor = handle._panel_editors['panel-1']
assert isinstance(editor.panel_form.widget_for('signal'), FluentTreeComboBox)
assert isinstance(editor.panel_form.widget_for('overlay_signal'), FluentTreeComboBox)
assert editor.panel_form.widget_for('signal').current_choice_key() == '@logic/cm/frames'
assert editor.panel_form.widget_for('overlay_signal').current_choice_key() == '@logic/occ/site_overlay'
assert editor.panel_form.widget_for('signal')._display_text() == 'camera · frames'
assert editor.panel_form.widget_for('overlay_signal')._display_text() == 'occupancy · sites'
class _PlotHost:
    def __init__(self, text):
        self.widget = QtWidgets.QLabel(text)
        self.qt_widget_calls = 0
    def qt_widget(self):
        self.qt_widget_calls += 1
        return self.widget
first_host = _PlotHost('frozen plot one')
second_host = _PlotHost('frozen plot two')
handle.show_panel_editor('panel-1', first_host)
assert editor._surface is first_host.widget
assert first_host.qt_widget_calls == 1
assert first_host.widget.parentWidget() is editor.surface_holder
handle.show_panel_editor('panel-1', second_host)
assert editor._surface is second_host.widget
assert first_host.widget.parentWidget() is None
assert second_host.widget.parentWidget() is editor.surface_holder
# An Edit surface's refusals are relayed like the card's, and a replaced
# surface goes quiet.
class _ErrorSurface(QtWidgets.QWidget):
    errorOccurred = QtCore.pyqtSignal(str)
class _ErrorHost:
    def __init__(self):
        self.widget = _ErrorSurface()
    def qt_widget(self):
        return self.widget
reports = []
editor.plot_error.connect(reports.append)
noisy = _ErrorHost()
handle.show_panel_editor('panel-1', noisy)
noisy.widget.errorOccurred.emit('editor refusal')
assert reports == ['editor refusal'], reports
handle.show_panel_editor('panel-1', second_host)
noisy.widget.errorOccurred.emit('stale editor noise')
assert reports == ['editor refusal'], reports
assert editor._surface is second_host.widget
assert view.tabs.count() == 3 and view.tabs.currentWidget() is editor
assert view.tabs.tabText(view.tabs.indexOf(editor)) == 'Edit panel · Camera'
assert editor.kind_label.text() == '(image)'
# The page's name is its heading; every card lays its title out as its
# first row, a control high, as the Figure Viewer's and Device Manager's do.
from zlc_ui.fluent import FluentSectionLabel
assert isinstance(editor.title_label, FluentSectionLabel)
for group in (*editor.parameter_groups.values(), editor.snapshot_group,
              editor.producer_group, editor.save_group):
    assert group._zlc_strip_px == 0, group.title()
    assert group.layout().itemAt(0).layout().itemAt(0).widget().text() == group.title()
assert editor.panel_form._labels['size'].text() == 'Panel size'
facet_projection = dict(projection, state=facet_state)
assert handle.update_panel_editor('panel-1', facet_projection)
assert editor.kind_label.text() == '(facet grid · curve cells)'
assert handle.update_panel_editor('panel-1', projection)
assert editor.producer_summary.text() == 'Logic node: cm'
assert editor.open_producer_button.isEnabled()
# The same in Edit: a refused pick is answered by the same projection, and
# the picker goes back to the overlay the panel draws.
editor.panel_form.widget_for('overlay_signal').select_choice_key('')
assert editor.panel_form.read_value('overlay_signal') == ''
assert handle.update_panel_editor('panel-1', projection)
assert editor.panel_form.widget_for('overlay_signal').current_choice_key() == (
    '@logic/occ/site_overlay'
)
from zlc_atom.nodes.camera_measurement.logic_node import LOGIC_NODE as camera_descriptor
from zlc_atom.nodes.seamless_scan.logic_node import LOGIC_NODE as scan_descriptor
from zlc_workbench.authoring_form import project_logic_schema
def linked_projection(descriptor, values):
    fields = frozenset(name for mapping in descriptor.selection_mappings for name in mapping.draft_fields)
    spec = project_logic_schema(descriptor, workspace_root=save_directory)
    return {'form_spec': FormSpec(tuple(field for field in spec.fields if field.key in fields)),
            'form_values': values, 'read_only_fields': fields,
            'ui_contributions': descriptor.ui_contributions}
linked = linked_projection(camera_descriptor, {'roi_x': 3, 'roi_y': 4, 'roi_width': 24, 'roi_height': 32})
handle.set_panel_producer_projection('panel-1', linked)
assert editor.producer_form.spec.keys == ('roi_x', 'roi_y', 'roi_width', 'roi_height')
roi_widget = editor.producer_form.widget_for('roi_x')
assert not roi_widget.isEnabled() and editor.producer_form.read_value('roi_x') == 3
panel_widget = editor.panel_form.widget_for('title')
handle.set_panel_producer_projection('panel-1', dict(linked, form_values=dict(linked['form_values'], roi_x=11)))
assert editor.producer_form.widget_for('roi_x') is roi_widget
assert editor.producer_form.read_value('roi_x') == 11
assert editor.panel_form.widget_for('title') is panel_widget
import json
plan = {'axes': [
    {'port': 'manual:position', 'values': [1, 2, 3], 'unit': 'mm', 'mode': 'range', 'value_text': ''},
    {'port': 'device:rf:frequency', 'values': [1, 2, 3], 'unit': 'MHz'},
    {'port': 'pulse:param:duration:hold', 'values': [2, 4, 6], 'unit': 'ms', 'mode': 'values', 'value_text': '10, 20'},
]}
linked_scan = linked_projection(scan_descriptor, {'plan': json.dumps(plan)})
handle.set_panel_producer_projection('panel-1', linked_scan)
scan_readout = next(iter(editor._producer_contributions.values()))
assert not scan_readout.isEnabled() and not editor.producer_form.spec.keys
assert scan_readout.values_form.isHidden() and scan_readout.add_button.isHidden()
row = scan_readout._rows[0]
assert row.start_spin.value() == 1 and row.stop_spin.value() == 3
assert scan_readout._rows[1].start_spin.valueUnit() == 'MHz'
assert scan_readout._rows[2].stop_spin.value() == 6
# A Values axis reads as it plays: its list on screen, the choice on Values,
# and no note about a Range bank the run does not use.
assert scan_readout._rows[2].input_stack.currentIndex() == 1
assert scan_readout._rows[2].values_edit.text() == '10, 20'
assert scan_readout._rows[2].mode_choice.button('values').property('current') is True
assert scan_readout._rows[2].custom_label.isHidden()
assert 'not in this pulse' not in scan_readout._rows[1].port_combo.currentText()
plan['axes'][0]['values'] = [4, 5, 6]
handle.set_panel_producer_projection('panel-1', dict(linked_scan, form_values={'plan': json.dumps(plan)}))
assert scan_readout._rows[0] is row and row.start_spin.value() == 4 and row.stop_spin.value() == 6
assert scan_readout._port_read_request is None
# A selection writes canonical float endpoints before any editable Logic view
# chooses their visible precision.  This linked producer is deliberately
# disabled/read-only: its width must affect only the spelling, never turn the
# producer's valid number into a transient business error.
selected = (142.1234567890123, 153.12345678395613, 164.1234567789)
plan['axes'][0]['values'] = list(selected)
handle.set_panel_producer_projection('panel-1', dict(linked_scan, form_values={'plan': json.dumps(plan)}))
assert not row.start_spin.property('numericError') and not row.stop_spin.property('numericError')
assert 'numeric' not in row.custom_label.text().lower(), row.custom_label.text()
app.processEvents()
assert row.start_spin.value() == selected[0] and row.stop_spin.value() == selected[-1]
assert not row.start_spin.property('numericError')
assert not row.stop_spin.property('numericError')
assert 'numeric' not in row.custom_label.text().lower(), row.custom_label.text()
handle.set_panel_producer_projection('panel-1', {})
assert not editor._producer_contributions and not editor.producer_form.spec.keys
assert editor.parameter_forms['semantic'].spec.keys == ('x',)
assert editor.parameter_forms['display'].spec.keys == (
    'title', 'x_label', 'x_display_unit', 'color_min', 'color_max',
    'colormap', 'show_colorbar'
)
assert editor.parameter_forms['fit'].spec.keys == ('model',)
fit_projection = dict(
    projection,
    state=fit_state,
    parameter_surface=fit_surface,
)
assert handle.update_panel_editor('panel-1', fit_projection)
assert editor.parameter_forms['fit'].spec.keys == ('model', 'expression')
assert handle.update_panel_editor('panel-1', projection)
locked_surface = dict(surface, science_locked=True)
handle.set_panel_projection('panel-1', state, locked_surface)
assert handle.update_panel_editor(
    'panel-1', dict(projection, parameter_surface=locked_surface)
)
# Opening the Edit page retires the card's Setting overlay, so ask it the
# way an operator does: open it again and read what it now offers.  A form
# nobody can see has no enabled state worth asserting.
card._open_settings()
app.processEvents()
assert not card._settings_form.widget_for('signal').isEnabled(), (
    f'popup={card._settings_popup.isVisible()} '
    f'locked={card._parameter_surface.get("science_locked")!r} '
    f'groups={bool(card._groups)}'
)
assert not card._settings_form.widget_for('overlay_signal').isEnabled()
assert not card._settings_parameter_forms['semantic'].widget_for('x').isEnabled()
assert card._settings_form.widget_for('size').isEnabled()
assert card._settings_parameter_forms['display'].widget_for('colormap').isEnabled()
assert card._settings_parameter_forms['fit'].widget_for('model').isEnabled()
assert not editor.panel_form.widget_for('signal').isEnabled()
assert not editor.panel_form.widget_for('overlay_signal').isEnabled()
assert not editor.parameter_forms['semantic'].isEnabled()
assert editor.parameter_forms['display'].isEnabled()
assert editor.parameter_forms['fit'].isEnabled()
assert editor.open_producer_button.isEnabled()
handle.set_panel_projection('panel-1', state, surface)
assert handle.update_panel_editor('panel-1', projection)
assert editor.parameter_forms['semantic'].isEnabled()
display_form = editor.parameter_forms['display']
title_auto = display_form.auto_switch_for('title')
title_edit = display_form.widget_for('title')
assert title_auto.isChecked()
assert title_auto.text() == 'Auto'
switch_image = QtGui.QImage(
    title_auto.sizeHint(), QtGui.QImage.Format_ARGB32_Premultiplied
)
switch_image.fill(QtCore.Qt.transparent)
title_auto.resize(title_auto.sizeHint())
title_auto.render(switch_image)
from zlc_ui.fluent import fluent_switch_width
track_right = fluent_switch_width('')
assert any(
    QtGui.qAlpha(switch_image.pixel(x, y)) > 0
    for x in range(track_right + 1, switch_image.width())
    for y in range(switch_image.height())
), 'the Auto/Manual words are painted beside the track, inside the switch'
# The row's words are in the label column; the switch is the first cell of
# the control column, before the editor it governs.
title_row = display_form._rows['title']
title_label = title_row.layout().itemAt(0).widget()
assert title_label is display_form._labels['title'] and title_label.text() == 'Title'
title_cell = title_row.layout().itemAt(1).widget()
assert title_cell is display_form.cell_for('title')
assert title_cell.layout().itemAt(0).widget() is title_auto
assert title_cell.layout().itemAt(1).widget() is title_edit
assert not title_edit.isEnabled()
edit_left = title_edit.x()
title_auto.setChecked(False)
app.processEvents()
assert title_auto.text() == 'Manual'
assert title_edit.x() == edit_left, 'flipping the switch moves nothing after it'
assert title_edit.isEnabled()
title_edit.setText('Camera image')
title_edit.editingFinished.emit()
title_auto.setChecked(True)
assert not title_edit.isEnabled()
x_unit_auto = editor.parameter_forms['display'].auto_switch_for('x_display_unit')
assert x_unit_auto.isChecked()
x_unit_edit = editor.parameter_forms['display'].widget_for('x_display_unit')
assert not x_unit_edit.isEnabled()
x_unit_auto.setChecked(False)
assert x_unit_edit.isEnabled() and x_unit_edit.currentData() == 'pixel'
x_unit_auto.setChecked(True)
color_min_edit = editor.parameter_forms['display'].widget_for('color_min')
try:
    editor.parameter_forms['display'].auto_switch_for('color_min')
except KeyError:
    pass
else:
    raise AssertionError('fixed color limits must not have independent Auto switches')
assert not color_min_edit.isEnabled()
fixed_display = tuple(
    dict(
        field,
        value=0 if field['key'] == 'color_min' else 10,
        unavailable_reason='',
    )
    if field['key'] in {'color_min', 'color_max'} else field
    for field in surface['display']
)
fixed_surface = dict(surface, display=fixed_display)
handle.set_panel_projection('panel-1', state, fixed_surface)
handle.update_panel_editor(
    'panel-1', dict(projection, parameter_surface=fixed_surface)
)
color_min_edit = editor.parameter_forms['display'].widget_for('color_min')
assert color_min_edit.isEnabled() and color_min_edit.text() == '0'
color_max_edit = editor.parameter_forms['display'].widget_for('color_max')
assert color_max_edit.isEnabled() and color_max_edit.text() == '10'
color_min_edit.setText('12')
color_max_edit.setText('20')
assert ('state', 'panel-1', {
    'display': {'color_max': 20, 'color_min': 12}
}) in events
semantic_combo = editor.parameter_forms['semantic'].widget_for('x')
semantic_combo.setCurrentIndex(1)
semantic_combo.activated.emit(1)
display_combo = editor.parameter_forms['display'].widget_for('colormap')
display_combo.setCurrentIndex(1)
display_combo.activated.emit(1)
fit_combo = editor.parameter_forms['fit'].widget_for('model')
fit_combo.setCurrentIndex(1)
fit_combo.activated.emit(1)
editor_interval = editor.panel_form.widget_for('interval_ms')
assert isinstance(editor_interval, FluentSegmentedButtons)
editor_interval.button(800).click()
editor.refresh_button.click()
editor.open_producer_button.click()
assert editor.save_directory.text() == save_directory
assert [editor.save_format.button(extension).text() for extension in ('png', 'pdf', 'svg')] == [
    'PNG', 'PDF', 'SVG',
]
assert editor.save_auto_name.isChecked()
assert Path(editor.save_preview.text()).parent == Path(save_directory)
editor.save_auto_name.setChecked(False)
editor.save_name.setText('manual-name')
editor.save_format.button('svg').click()
editor.save_button.click()
app.processEvents()
assert ('state', 'panel-1', {'interval_ms': 800}) in events
assert ('state', 'panel-1', {'semantic': {'x': 'point'}}) in events
assert ('state', 'panel-1', {
    'fit': {'model': 'anisotropic_gaussian_center'}
}) in events
assert ('state', 'panel-1', {'display': {'colormap': 'magma'}}) in events
assert ('state', 'panel-1', {'display': {'title': 'Camera image'}}) in events
assert ('state', 'panel-1', {'display': {'title': None}}) in events
assert ('state', 'panel-1', {'display': {'x_display_unit': 'pixel'}}) in events
assert ('state', 'panel-1', {'display': {'x_display_unit': None}}) in events
assert ('state', 'panel-1', {
    'fit': {'model': 'anisotropic_gaussian_center'}
}) in events
assert ('refresh', 'panel-1') in events
assert ('edit_logic', 'cm') in events
assert ('save', 'panel-1', str(Path(save_directory) / 'manual-name.svg')) in events

changed = dict(state, title='Retitled', interval_ms=800)
handle.set_panel_projection('panel-1', changed, surface)
handle.update_panel_editor('panel-1', dict(projection, state=changed, stale=True))
assert 'Retitled' in card._title_label.toolTip()
assert editor.panel_form.read_all()['title'] == 'Retitled'
assert not editor.save_button.isEnabled()
view.editor_close_requested.emit(editor)
app.processEvents()
assert view.tabs.count() == 2 and 'panel-1' not in handle._panel_editors
assert ('closed', 'panel-1') in events
assert second_host.widget.parentWidget() is None
temporary.cleanup()
"""
    )


def test_status_strip_shows_the_latest_line_and_falls_back_to_idle(run_qt) -> None:
    run_qt(
        """
from zlc_ui.qt import ensure_qt_app
from zlc_ui.console import StatusStrip
app = ensure_qt_app(['test'])
strip = StatusStrip()
strip.show_status('idle text', 'idle')
assert strip.current_severity == 'idle'
assert strip.text() == 'idle text'
strip.show_status('warning text', 'warning')
assert strip.current_severity == 'warning'
strip.show_status('task text', 'task')
assert strip.current_severity == 'task'
strip.show_status('error text', 'error')
assert strip.current_severity == 'error'
assert strip.text() == 'error text'
# And the next thing that happens is what the operator sees.  An error used to
# latch forever, so every later success was written into a slot the strip
# would not display: fix the problem, press the button, see the old error.
strip.show_status('saved 3 panel(s)', 'task')
assert strip.current_severity == 'task'
assert strip.text() == 'saved 3 panel(s)'
# An empty message is nothing to say: the idle line, not an older one.
strip.show_status('', 'task')
assert strip.current_severity == 'idle'
assert strip.text() == 'idle text'
"""
    )


def test_task_console_construct_and_setters(run_qt) -> None:
    run_qt(
        """
from PyQt5 import QtCore, QtGui
from zlc_ui.qt import ensure_qt_app
from zlc_ui.console import LogicRowView, PanelCardView, TaskConsoleView
from zlc_ui.fluent import ACCENT, row_gap, scaled_px
app = ensure_qt_app(['test'])
view = TaskConsoleView()
card = PanelCardView('panel-1')
row = LogicRowView('Logic')
view.set_cards((card,))
view.set_logic_rows((row,))
view.set_summary('one fake card')
view.show_status('ready', 'idle')
view.set_panel_kinds((('image', 'Image'),))
view.resize(1500, 422); view.show(); app.processEvents()
assert view.summary_label.text() == 'one fake card'
assert view.status_strip.current_severity == 'idle'
assert view.board._cards['panel-1'] is card
assert view.selectors_switch.width() > 0
# The header is one semantic row: identity on the left, telemetry as the
# only flexible middle cell, and the command cluster pinned to the right.
# A wrapping row top-aligned every sizeHint and left the spare width after the
# final button, making both the grey summary and the action cluster visibly
# wrong while every control still existed.
header = view.summary_label.parentWidget()
layout = header.layout()
right_edge = header.rect().right() - layout.contentsMargins().right()
assert view.clear_board_button.geometry().right() == right_edge
center_y = header.rect().center().y()
for widget in (
    view.status_dot,
    view.name_label,
    view.summary_label,
    view.kind_combo,
    view.add_button,
    view.selectors_switch,
    view.pause_switch,
    view.save_screenshot_button,
    view.save_layout_button,
    view.load_layout_button,
    view.clear_board_button,
):
    assert abs(widget.geometry().center().y() - center_y) <= 1
# Cells and commands on the header stand one row gap apart.
assert layout.spacing() == row_gap()

# The strip and every page under it share one left edge and one top: the
# first tab's underline, the Monitor board's first card and the Logic list's
# first row -- switching pages moves nothing.
def at(widget):
    point = widget.mapTo(view.tabs, QtCore.QPoint())
    return point.x(), point.y()
view.tabs.setCurrentIndex(0); app.processEvents()
bar = view.tabs.tabBar()
image = view.tabs.grab().toImage()
ratio = image.devicePixelRatio() or 1.0
strip_left, strip_top = at(bar)
y = int((strip_top + bar.height()) * ratio) - 1
underline = next(x / ratio for x in range(int(strip_left * ratio), image.width())
                 if image.pixelColor(x, y).name() == QtGui.QColor(ACCENT).name())
monitor = at(card)
view.tabs.setCurrentIndex(1); app.processEvents()
logic = at(row)
assert monitor == logic, (monitor, logic)
assert underline == monitor[0], (underline, monitor)

# And the header and the strip above the tabs stand on those same edges: the
# header's dot, the strip's dot, the first underline and the first Logic card
# start on one x, and the header's last command ends where the Logic card
# ends -- one left edge and one right edge for the whole window.
def x_in_view(widget, x=0):
    return widget.mapTo(view, QtCore.QPoint(x, 0)).x()
page_left = x_in_view(view.tabs, monitor[0])
edges = (x_in_view(view.status_dot), x_in_view(view.status_strip.dot), page_left, x_in_view(row))
assert len(set(edges)) == 1, edges
clear_right = x_in_view(view.clear_board_button, view.clear_board_button.width())
row_right = x_in_view(row, row.width())
assert clear_right == row_right, (clear_right, row_right)
"""
    )


def test_task_console_reuses_logic_rows_after_repeated_projection(run_qt) -> None:
    run_qt(
        """
from zlc_ui.qt import ensure_qt_app
from zlc_ui.console import LogicRowView, TaskConsoleView
app = ensure_qt_app(['test'])
view = TaskConsoleView()
row_a = LogicRowView('A')
row_b = LogicRowView('B')
view.set_logic_rows((row_a, row_b))
view.set_logic_rows((row_a, row_b))
app.processEvents()
row_a.set_state('running', 'still usable')
row_b.set_state('idle', 'also usable')
assert row_a.parentWidget() is view.logic_body
assert row_b.parentWidget() is view.logic_body
assert not view.logic_hint.isVisible()
assert row_a.status_label.text() == 'still usable'
assert row_b.status_label.text() == 'also usable'
"""
    )


def test_task_console_qtest_signal_payloads(run_qt) -> None:
    run_qt(
        """
from zlc_ui.console import task_console_view as tested_module
from PyQt5 import QtCore, QtTest
from zlc_ui.qt import ensure_qt_app
from zlc_ui.console import TaskConsoleHandle
app = ensure_qt_app(['test'])
view = tested_module.TaskConsoleView()
handle = TaskConsoleHandle(None, view)
handle.set_panel_sizes(('2x2',), '2x2')
view.show(); app.processEvents()
events = []
handle.set_panel_kinds((('curve', 'Curve'), ('image', '2D image')), 'image')
handle.set_logic_kinds((
    ('calibration', 'task', 'nothing'),
    ('occupancy', 'processor', 'occupied'),
    ('camera_measurement', 'measurement', 'frames'),
))
handle.add_panel_requested.connect(lambda kind: events.append(('panel', kind)))
handle.add_logic_requested.connect(lambda api_name: events.append(('logic', api_name)))
handle.pause_toggled.connect(lambda value: events.append(('pause', value)))
handle.save_screenshot_requested.connect(lambda: events.append(('screenshot',)))
handle.save_layout_requested.connect(lambda: events.append(('layout',)))
handle.clear_board_requested.connect(lambda: events.append(('clear',)))
handle.stop_task_requested.connect(lambda: events.append(('stop-task',)))
assert view.kind_combo.count() == 5
assert view.kind_combo.itemData(0) == ('plot', 'curve')
assert view.kind_combo.itemData(1) == ('plot', 'image')
assert view.kind_combo.itemData(2) == ('logic', 'camera_measurement')
assert view.kind_combo.itemData(3) == ('logic', 'occupancy')
assert view.kind_combo.itemData(4) == ('logic', 'calibration')
assert view.kind_combo.itemText(0) == 'Plot: Curve'
assert view.kind_combo.itemText(1) == 'Plot: 2D image'
assert view.kind_combo.itemText(2) == 'Measurement: camera_measurement'
assert view.kind_combo.itemText(3) == 'Processor: occupancy'
assert view.kind_combo.itemText(4) == 'Task: calibration'
assert view.kind_combo.currentData() == ('plot', 'image')
# The command names what it adds, from the family chosen beside it, in one
# slot: the commands after it stand still when the words change.
assert view.add_button.text() == 'Add panel'
add_slot = lambda: (view.add_button.geometry(), view.selectors_switch.geometry())
plot_slot = add_slot()
QtTest.QTest.mouseClick(view.add_button, QtCore.Qt.LeftButton)
view.kind_combo.setCurrentIndex(2)
app.processEvents()
assert view.add_button.text() == 'Add node'
assert add_slot() == plot_slot, (add_slot(), plot_slot)
assert 'click Add node' in view.logic_hint.text()
QtTest.QTest.mouseClick(view.add_button, QtCore.Qt.LeftButton)
QtTest.QTest.mouseClick(view.pause_switch, QtCore.Qt.LeftButton)
QtTest.QTest.mouseClick(view.save_screenshot_button, QtCore.Qt.LeftButton)
QtTest.QTest.mouseClick(view.save_layout_button, QtCore.Qt.LeftButton)
QtTest.QTest.mouseClick(view.clear_board_button, QtCore.Qt.LeftButton)
assert ('panel', 'image') in events
assert ('logic', 'camera_measurement') in events
assert ('pause', True) in events
assert ('screenshot',) in events
assert ('layout',) in events
assert ('clear',) in events

# A running Task freezes Logic identity, not the monitor window.
handle.add_panel('task-preview', 'Capture preview')
handle.add_logic_row('calibration', 'task')
handle.add_logic_row('camera', 'measurement')
handle.set_logic_commands('calibration', can_start=True, can_stop=True)
handle.set_logic_commands('camera', can_start=True, can_stop=True)
handle.logic_stop_requested.connect(lambda node_id: events.append(('stop', node_id)))
preview = handle._cards['task-preview']
task_row = handle._rows['calibration']
camera_row = handle._rows['camera']
view.tabs.setCurrentIndex(1)
app.processEvents()
slots = lambda row: tuple(
    widget.mapTo(view, QtCore.QPoint()).x()
    for widget in (row.preview_switch, row.start_button, row.stop_button, row.edit_button)
)
before_takeover = slots(task_row)
handle.set_task_takeover('calibration')
app.processEvents()
assert view.status_strip.action_button is not None
assert view.status_strip.action_button.isVisible()
assert view.kind_combo.isEnabled()
assert not view.add_button.isEnabled()
assert view.save_layout_button.isEnabled()
assert not view.load_layout_button.isEnabled()
assert not view.clear_board_button.isEnabled()
assert view.selectors_switch.isEnabled()
assert view.pause_switch.isEnabled()
assert view.save_screenshot_button.isEnabled()
assert preview.settings_button.isEnabled()
for row in (task_row, camera_row):
    assert not row.start_button.isEnabled()
    assert not row.edit_button.isEnabled()
    assert not row.remove_button.isEnabled()
# The running Task's own row keeps its Stop, in its place; every other
# row's Stop waits for the Task like the rest of its commands.
assert task_row.stop_button.isVisible() and task_row.stop_button.isEnabled()
assert slots(task_row) == before_takeover, (slots(task_row), before_takeover)
assert camera_row.stop_button.isVisible() and not camera_row.stop_button.isEnabled()
before = list(events)
QtTest.QTest.mouseClick(view.add_button, QtCore.Qt.LeftButton)
QtTest.QTest.mouseClick(camera_row.start_button, QtCore.Qt.LeftButton)
QtTest.QTest.mouseClick(camera_row.stop_button, QtCore.Qt.LeftButton)
assert events == before
QtTest.QTest.mouseClick(task_row.stop_button, QtCore.Qt.LeftButton)
assert events[-1] == ('stop', 'calibration')
view.kind_combo.setCurrentIndex(0)
assert view.kind_combo.currentData()[0] == 'plot'
assert view.add_button.isEnabled()
QtTest.QTest.mouseClick(view.add_button, QtCore.Qt.LeftButton)
assert events[-1] == ('panel', 'curve')
QtTest.QTest.mouseClick(view.status_strip.action_button, QtCore.Qt.LeftButton)
assert events[-1] == ('stop-task',)

handle.set_task_takeover('')
assert not view.status_strip.action_button.isVisible()
assert view.kind_combo.isEnabled()
assert view.add_button.isEnabled()
assert preview.settings_button.isEnabled()
assert camera_row.start_button.isEnabled()
assert camera_row.stop_button.isEnabled()
assert camera_row.edit_button.isEnabled()
assert camera_row.remove_button.isEnabled()

"""
    )


def test_pause_is_reversible_and_says_which_way_it_goes(run_qt) -> None:
    """A one-way pause is a stopped console with no way back.

    The button carries the command and its label carries the state, so it must
    ask for the state it is NOT in -- and the presenter, which owns the answer,
    is what tells it which that is.
    """

    run_qt(
        """
from PyQt5 import QtCore, QtTest
from zlc_ui.qt import ensure_qt_app
from zlc_ui.console import TaskConsoleView
app = ensure_qt_app(['pause'])
view = TaskConsoleView(); view.show(); app.processEvents()
asked = []
view.pause_toggled.connect(asked.append)
QtTest.QTest.mouseClick(view.pause_switch, QtCore.Qt.LeftButton)
assert asked == [True], asked
assert view.pause_switch.text() == 'Pause', 'the label moved before the presenter agreed'
view.set_paused(True)
assert view.pause_switch.text() == 'Resume'
QtTest.QTest.mouseClick(view.pause_switch, QtCore.Qt.LeftButton)
assert asked == [True, False], asked
"""
    )


def test_the_signal_chooser_lists_rows_under_their_producer(run_qt) -> None:
    """It is asked for at the moment it is needed, so it can be readable.

    A picker squeezed into the header collapsed to an ellipsis and clipped
    the control beside it; a chooser you cannot read is not a chooser.
    """

    run_qt(
        """
from PyQt5 import QtCore
from zlc_ui.qt import ensure_qt_app
from zlc_ui.console import SignalChooser
app = ensure_qt_app(['chooser'])
rows = (
    ('@logic/cm/frames', 'frames', 'live', 'cm', ''),
    ('@logic/panel-1/roi_mean', 'roi_mean', 'finished', 'panel-1', '@logic/cm/frames'),
)
dialog = SignalChooser(rows)
texts = [dialog.list.item(i).text() for i in range(dialog.list.count())]
assert texts == ['cm', '    frames', 'panel-1', '    roi_mean  (finished)'], texts

# A producer heading is a heading, not a choice.
assert not dialog.list.item(0).flags() & QtCore.Qt.ItemIsSelectable
assert dialog.chosen() == '@logic/cm/frames', dialog.chosen()

dialog.list.setCurrentRow(3)
assert dialog.chosen() == '@logic/panel-1/roi_mean'
assert 'cut from @logic/cm/frames' in dialog.list.item(3).toolTip()
assert dialog.accept_button.isEnabled()

# Nothing has published: say so, and refuse to pretend otherwise.
empty = SignalChooser(())
assert empty.chosen() is None
assert not empty.accept_button.isEnabled()
"""
    )


def test_task_console_acceptance_launcher_keeps_the_empty_status_left_anchored(run_qt) -> None:
    run_qt(
        """
from zlc_ui.qt import ensure_qt_app
from zlc_ui.console import TaskConsoleView
from zlc_ui.fluent import launch_fluent_window
app = ensure_qt_app(['acceptance'])
body = TaskConsoleView()
window = launch_fluent_window(body, title='TaskConsole@Zou lab', fixed_size=False)
app.processEvents()
dot_left = body.status_strip.dot.geometry().left()
assert dot_left == body.status_strip.layout().contentsMargins().left(), (dot_left, body.status_strip.layout().contentsMargins().left())
window.close()
"""
    )


def test_a_window_hosting_a_close_guarding_body_can_actually_be_closed(run_qt) -> None:
    """The X did nothing at all.

    Several bodies refuse their own close and raise close_requested so a host
    can tear down first.  The host half of that handshake lived nowhere, so an
    application that did not reinvent it produced a window that could not be
    closed -- the body ignored the event and nothing answered the request.
    """

    run_qt(
        """
from PyQt5 import QtWidgets
from zlc_ui.qt import ensure_qt_app
from zlc_ui.fluent import launch_fluent_window
from zlc_ui.pulse.editor_view import PulseEditorView
app = ensure_qt_app(['close-handshake'])

window = launch_fluent_window(PulseEditorView, title='PulseGUI@Zou lab')
body = window.findChild(PulseEditorView)
assert body is not None and window.isVisible()

# The body asking to close closes the window...
torn = []
body.close_requested.connect(lambda: torn.append('asked'))
body.close_requested.emit()
app.processEvents()
assert not window.isVisible(), 'the body asked to close and the window stayed'

# ...and a window closing lets the body commit its own close.
second = launch_fluent_window(PulseEditorView, title='PulseGUI@Zou lab')
inner = second.findChild(PulseEditorView)
second.close()
app.processEvents()
assert not second.isVisible()
assert inner._closing, 'the body never got to run its own teardown'

# Binding must not invent a close guard for a widget that has none.
plain = launch_fluent_window(lambda: QtWidgets.QLabel('plain'), title='Plain')
assert plain.isVisible()
plain.close()
app.processEvents()
assert not plain.isVisible()
"""
    )


def test_a_schedule_rebuild_keeps_the_operator_where_they_were(run_qt) -> None:
    """Every update tore the cards out of their layout and reset the scroll.

    So pressing On Pulse -- or editing anything at all -- threw the board back
    to the first period and the leftmost time, losing the place of anyone
    working on period nine.
    """

    run_qt(
        """
from zlc_ui.qt import ensure_qt_app
from zlc_ui.pulse.schedule_view import PulseScheduleView
from zlc_ui.pulse.models import FieldVM, PeriodVM, PortRowVM, ScheduleVM
app = ensure_qt_app(['scroll'])
view = PulseScheduleView(); view.resize(900, 500); view.show()
for _ in range(5): app.processEvents()

def vm(revision, name='p'):
    ports = tuple(PortRowVM(key=f'ch{i}', kind='digital', label=f'ch{i}', endpoint_text=f'P{i}') for i in range(20))
    periods = tuple(
        PeriodVM(period_id=f'period{i}', name=f'{name}{i}', duration=FieldVM(text='1'), unit='us',
                 digital=tuple((p.key, False) for p in ports))
        for i in range(12)
    )
    return ScheduleVM(document_generation=0, revision=revision, document_name='probe',
                      clock_text='20 ns/tick', total_text='12 us', total_tooltip='', period_count=12,
                      visible_text='20/20', summary_text='', ports=ports, periods=periods)

view.set_schedule(vm(1))
for _ in range(5): app.processEvents()
bars = [view.timeline_scroll.horizontalScrollBar(), view.timeline_scroll.verticalScrollBar(),
        view.left_scroll.verticalScrollBar()]
for bar in bars: bar.setValue(bar.maximum())
before = [bar.value() for bar in bars]
assert all(before), before

view.set_schedule(vm(2, name='q'))
for _ in range(5): app.processEvents()
assert [bar.value() for bar in bars] == before, [bar.value() for bar in bars]

# A rebuild that shortens the pulse cannot restore past the new end.
def short(revision):
    full = vm(revision)
    return ScheduleVM(document_generation=0, revision=revision, document_name='probe',
                      clock_text=full.clock_text, total_text='1 us', total_tooltip='', period_count=1,
                      visible_text=full.visible_text, summary_text='', ports=full.ports,
                      periods=full.periods[:1])

view.set_schedule(short(3))
for _ in range(5): app.processEvents()
assert all(bar.value() <= bar.maximum() for bar in bars)
"""
    )


def test_every_panel_kind_opens_its_setting_form_with_exact_keys(run_qt) -> None:
    """The Setting form's SPEC and its VALUES are one decision, not two lists.

    ``FluentParameterForm`` demands exact keys, and it is built inside a Qt
    slot -- so a spec that declares a row the values do not supply does not
    raise, it ABORTS the process the moment an operator presses Setting.  The
    only test that can catch it is one that compares the two key sets for
    every panel kind a card can be put in: image, facet grid with each cell
    kind, curve, histogram, rolling and pulse timeline.  The parameter
    surfaces are empty on purpose -- the question is the card's OWN rows.
    """

    run_qt(
        """
from PyQt5 import QtCore, QtWidgets
from zlc_ui.qt import ensure_qt_app
from zlc_ui.console import panel_card_view as tested_module
from zlc_ui.console import panel_editor_view as editor_module

app = ensure_qt_app(['test'])
surface = {'semantic': (), 'display': (), 'fit': ()}
groups = (('camera', (('frames', '@logic/cm/frames'),)),)
overlays = (('occupancy', (('sites', '@logic/occ/occupied'),)),)
owner = QtWidgets.QWidget()
owner.show()
app.processEvents()

try:
    for kind, cell_kind, overlay_signal in (
        ('image', '', '@logic/occ/occupied'),
        ('image', '', ''),
        ('facet_grid', '', '@logic/occ/occupied'),
        ('facet_grid', 'image', '@logic/occ/occupied'),
        ('facet_grid', 'curve', ''),
        ('facet_grid', 'histogram', ''),
        ('curve', '', ''),
        ('histogram', '', ''),
        ('rolling', '', ''),
        ('pulse_timeline', '', ''),
    ):
        card = None
        editor = None
        try:
            state = {
                'signal': '@logic/cm/frames', 'kind': kind,
                'cell_kind': cell_kind, 'size': '2x2',
                'interval_ms': 100, 'title': 'Card', 'semantic': {},
                'display': {}, 'fit': {}, 'overlay_signal': overlay_signal,
            }
            projected_surface = dict(
                surface,
                paints_images=(
                    kind == 'image'
                    or (kind == 'facet_grid' and cell_kind in {'', 'image'})
                ),
            )
            card = tested_module.PanelCardView('panel-1', 'Card', owner)
            card.set_size_choices(('1x2', '2x2', '1x4'), '2x2')
            card.set_cell_kind_choices(('curve', 'image', 'histogram'))
            card.set_signal_choices(
                groups, current=state['signal'],
                overlay_groups=overlays, overlay_current=overlay_signal,
            )
            card.set_panel_projection(state, projected_surface)
            card.show()
            app.processEvents()
            spec_keys = set(card._form_spec().keys)
            value_keys = set(card._form_values())
            assert spec_keys == value_keys, (
                kind, cell_kind, sorted(spec_keys ^ value_keys)
            )
            # The real crash path: this is what pressing Setting runs.
            card._open_settings()
            assert card._settings_form is not None
            assert set(card._settings_form.spec.keys) == value_keys

            editor = editor_module.PanelEditorView('panel-1', {
                'panel_id': 'panel-1', 'state': state,
                'signal_options': groups,
                'overlay_signal_options': overlays,
                'interval_choices': (100, 200, 400, 800),
                'size_choices': ('1x2', '2x2', '1x4'),
                'parameter_surface': projected_surface, 'save_directory': '.',
                'frozen_signal': state['signal'],
                'frozen_publication': None, 'frozen_snapshot': None,
                'stale': False, 'producer_node_id': '',
            }, owner)
            assert ('overlay_signal' in editor.panel_form.spec.keys) == (
                'overlay_signal' in spec_keys
            )
        finally:
            if card is not None:
                card.retire_settings_popup()
                card.deleteLater()
            if editor is not None:
                editor.deleteLater()
            QtCore.QCoreApplication.sendPostedEvents(
                None, QtCore.QEvent.DeferredDelete
            )
            app.processEvents()
finally:
    owner.close()
    owner.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(
        None, QtCore.QEvent.DeferredDelete
    )
    app.processEvents()
"""
    )


def test_the_auto_preview_control_is_a_switch_that_looks_different_when_on(run_qt) -> None:
    """A checkable FluentButton once drew NO checked state, and a later toggle
    chip still read as one more command among Start and Edit.  The setting
    is a FluentSwitch: a track that is visibly on or off, its label beside it.
    """

    run_qt(
        """
from zlc_ui.qt import ensure_qt_app
from zlc_ui.fluent import FluentSwitch
from zlc_ui.console import logic_row_view as tested_module

app = ensure_qt_app(['test'])
row = tested_module.LogicRowView('camera_measurement', 'measurement')
row.resize(760, 48)
row.show()
app.processEvents()

switch = row.preview_switch
assert isinstance(switch, FluentSwitch), type(switch)

seen = []
switch.toggled.connect(seen.append)
row.set_auto_preview(False)
app.processEvents()
off = switch.grab().toImage()
row.set_auto_preview(True)
app.processEvents()
on = switch.grab().toImage()
assert not seen, 'showing the owner state must not re-raise it: %r' % (seen,)
assert on != off, 'the switch must look different when it is on'

order = []
layout = row.layout().itemAt(0).layout()
for index in range(layout.count()):
    widget = layout.itemAt(index).widget()
    if widget is not None and widget in (
        row.start_button, row.stop_button, row.preview_switch,
        row.edit_button, row.remove_button,
    ):
        order.append(widget)
assert order == [
    row.preview_switch, row.start_button, row.stop_button,
    row.edit_button, row.remove_button,
], order
# The switch belongs immediately left of Start and Stop immediately right of
# it, as on the Edit page.

row.set_task_takeover(True)
assert not switch.isEnabled(), 'a row that cannot Start cannot re-aim Start'

# A node that declares nothing to open has no preference to express, so it
# has no switch either -- a control that does nothing is worse than absent.
row.set_preview_offered(False)
app.processEvents()
assert not switch.isVisible(), 'no switch where the node opens nothing'
row.set_preview_offered(True)
app.processEvents()
assert switch.isVisible()
"""
    )


def test_the_editor_shows_the_same_preview_switch_beside_its_own_start(run_qt) -> None:
    """Adding a node opens its Edit tab and focuses it, so THAT Start is the
    one an operator presses.  The switch is a projection of the node's stored
    preference in both places -- the editor keeps no default of its own.
    """

    run_qt(
        """
from PyQt5 import QtCore, QtTest
from zlc_ui.qt import ensure_qt_app
from zlc_ui.fluent import FluentSwitch
from zlc_ui.form import FormSpec
from zlc_ui.console import logic_editor_view as tested_module

app = ensure_qt_app(['test'])
projection = {
    'node_id': 'camera_measurement',
    'api_name': 'camera_measurement',
    'kind': 'measurement',
    'form_spec': FormSpec(()),
    'form_values': {},
    'artifact_form_spec': FormSpec(()),
    'artifact_values': {},
    'artifact_results': (),
    'auto_preview': False,
    'preview_offered': True,
    'can_start': True,
    'can_stop': False,
}
editor = tested_module.LogicEditorView('camera_measurement', projection)
editor.resize(900, 420)
editor.show()
app.processEvents()
switch = editor.preview_switch
assert isinstance(switch, FluentSwitch), type(switch)
assert switch.isChecked() is False, 'the editor shows the stored preference'

# Beside Start, in pixels: same row, immediately to its left.  A switch that
# merely EXISTS somewhere is what an operator could not find.
start, stop = editor.start_button, editor.stop_button
assert switch.isVisible() and start.isVisible()
assert abs(switch.y() - start.y()) <= 2, (switch.geometry(), start.geometry())
gap = start.x() - (switch.x() + switch.width())
assert 0 <= gap <= 40, (switch.geometry(), start.geometry())
assert start.role() == 'primary' and start.text() == 'Start' and start.isEnabled()
# Start then Stop, as on the row and as On Pulse then Stop Pulse; Stop is
# always there, disabled while there is nothing to stop.
assert stop.isVisible() and not stop.isEnabled() and stop.role() == 'destructive'
assert abs(stop.y() - start.y()) <= 2 and start.x() + start.width() < stop.x()
slots = lambda: (switch.x(), start.x(), start.width(), stop.x(), stop.width())
idle = slots()

# Running, Start says Restart and stays this page's primary; nothing on the
# row moves -- no button appears beside it.
editor.update_projection(dict(projection, running=True, can_stop=True))
app.processEvents()
assert start.text() == 'Restart' and start.role() == 'primary', (start.text(), start.role())
assert start.isEnabled() and stop.isEnabled()
assert slots() == idle, (slots(), idle)

# A follower waiting for its source has no host yet, but it IS active: the
# page says what the row says, Restart and an enabled Stop.
from zlc_ui.console.logic_row_view import start_label
editor.update_projection(dict(projection, running=False, can_stop=True))
app.processEvents()
assert start.text() == start_label(pending=False, active=True) == 'Restart'
assert stop.isEnabled()

# A start queued behind a device release can be cancelled from here.
editor.update_projection(dict(projection, pending=True, can_start=False, can_stop=True))
app.processEvents()
assert start.text() == 'Starting…' and not start.isEnabled() and stop.isEnabled()
assert slots() == idle, (slots(), idle)
stops = []
editor.stop_requested.connect(lambda: stops.append('stop'))
QtTest.QTest.mouseClick(stop, QtCore.Qt.LeftButton)
assert stops == ['stop']

# While a Task owns the console only that Task's own Stop stays live.
editor.update_projection(dict(projection, running=True, can_stop=True))
editor.set_mutation_enabled(False, owns_task=True)
assert stop.isEnabled() and not start.isEnabled() and not editor.remove_button.isEnabled()
editor.set_mutation_enabled(False)
assert not stop.isEnabled()
editor.set_mutation_enabled(True)
editor.update_projection(projection)
app.processEvents()
assert start.text() == 'Start' and not stop.isEnabled()
assert slots() == idle, (slots(), idle)

raised = []
switch.toggled.connect(raised.append)
editor.update_projection(dict(projection, auto_preview=True))
assert switch.isChecked() is True
assert not raised, 'projecting the owner state must not raise intent: %r' % (raised,)

intents = []
editor.auto_preview_changed.connect(intents.append)
switch.setChecked(False)
assert intents == [False], intents

editor.set_mutation_enabled(False)
assert not switch.isEnabled(), 'a Task owns the console: nothing here re-aims Start'

editor.update_projection(dict(projection, preview_offered=False))
assert not switch.isVisible(), 'no switch where the node opens nothing'
"""
    )


def test_a_tall_logic_edit_page_keeps_its_header_and_action_bar_and_opens_at_its_top(run_qt) -> None:
    """Only the rows scroll.  On a page taller than its tab the node's name,
    the status line that says why Start is off, and Start, Stop and Remove
    stay on screen; and a newly opened page starts at its first row, even
    when something moved its bar while it opened."""

    run_qt(
        """
from PyQt5 import QtCore, QtGui
from zlc_ui.qt import ensure_qt_app
from zlc_ui.fluent import FluentScrollArea
from zlc_ui.form import FormFieldProps, FormSpec
from zlc_ui.console import LogicEditorView

app = ensure_qt_app(['test'])
fields = tuple(FormFieldProps(key=f'value_{index}', kind='int', label=f'Value {index}', default=index)
               for index in range(30))
projection = {
    'api_name': 'seamless_scan', 'kind': 'measurement',
    'form_spec': FormSpec(fields), 'form_values': {field.key: field.default for field in fields},
    'artifact_form_spec': FormSpec(()), 'artifact_values': {}, 'artifact_results': (),
    'auto_preview': False, 'preview_offered': True,
    'can_start': False, 'can_stop': True, 'running': True,
    'issues': ('Signal must be selected',),
}
editor = LogicEditorView('scan-1', projection)
editor.resize(700, 360)
scroll = editor.findChild(FluentScrollArea)
bar = scroll.verticalScrollBar()
editor.show()
# A wheel turned over the window while the page opened.
viewport = scroll.viewport()
centre = QtCore.QPointF(viewport.width() / 2, viewport.height() / 2)
app.sendEvent(viewport, QtGui.QWheelEvent(
    centre, QtCore.QPointF(viewport.mapToGlobal(centre.toPoint())), QtCore.QPoint(), QtCore.QPoint(0, -120),
    QtCore.Qt.NoButton, QtCore.Qt.NoModifier, QtCore.Qt.NoScrollPhase, False,
))
assert bar.maximum() > 0 and bar.value() > 0, (bar.maximum(), bar.value())
app.processEvents()
assert bar.value() == 0, 'a newly opened page starts at its first row'
inside = lambda widget: QtCore.QRect(QtCore.QPoint(), editor.size()).contains(
    QtCore.QRect(widget.mapTo(editor, QtCore.QPoint()), widget.size()))
# Scrolled to the end, the header and the action bar are still there.
bar.setValue(bar.maximum())
app.processEvents()
for widget in (editor.title_label, editor.status_label, editor.remove_button,
               editor.preview_switch, editor.start_button, editor.stop_button):
    assert widget.isVisible() and inside(widget), widget
assert editor.status_label.text() == 'Draft: Signal must be selected'
assert editor.start_button.text() == 'Restart' and editor.stop_button.isEnabled()
# After that the bar is the operator's: a tab switch keeps it.
editor.hide(); editor.show(); app.processEvents()
assert bar.value() == bar.maximum() > 0
editor.close()
"""
    )


def test_the_card_strip_renders_the_caption_its_presenter_projected(run_qt) -> None:
    """The strip says what the presenter captioned the panel, and only that.

    A signal-bound panel is captioned from its signal's name, whose grammar
    the runtime owns; the presenter reads it there, so the card composes
    nothing.  Display only: the title is also the panel's editable name, and
    the caption must never reach the state, the rename field, or the Setting
    form.  (The caption itself is pinned through the presenter's projection
    in test_console_presenter.)
    """

    run_qt(
        """
from zlc_ui.qt import ensure_qt_app
from zlc_ui.console import panel_card_view as tested_module
PanelCardView = tested_module.PanelCardView
app = ensure_qt_app(['test'])


def project(card, title, signal, caption):
    surface = {'semantic': (), 'display': (), 'fit': ()}
    if caption is not None:
        surface['caption'] = caption
    card.set_panel_projection(
        {
            'signal': signal, 'kind': 'image', 'size': '2x2',
            'interval_ms': 100, 'title': title, 'semantic': {},
            'display': {}, 'fit': {}, 'overlay_signal': '',
        },
        surface,
    )


card = PanelCardView('panel-2', 'Card')
card.set_size_choices(('1x2', '2x2', '1x4'), '2x2')
card.set_signal_choices((('source', (('Frames', '@logic/camera_measurement/frames'),)),))

signal = '@logic/camera_measurement/frames'
project(card, signal, signal, 'panel-2@camera_measurement/frames')
assert card._caption() == 'panel-2@camera_measurement/frames', card._caption()
line = card._band_fragments()[0]
# The card's own id leads in muted ink; the name the operator reads follows it.
assert line[0][:2] == ('panel-2', tested_module.MUTED_INK), line
assert line[0][0] + line[1][0] == 'panel-2@camera_measurement/frames', line

# The name itself is untouched, everywhere it is the NAME and not a caption.
assert card._base_title == signal
assert card._state_projection['title'] == signal
assert card._form_values()['title'] == signal

# A renamed panel is re-captioned by the same projection that renames it.
project(card, 'MOT shot', signal, 'panel-2 MOT shot')
assert card._caption() == 'panel-2 MOT shot', card._caption()
assert card._form_values()['title'] == 'MOT shot'

# A card nothing has captioned is called by its name.
project(card, 'Image 2', '', None)
assert card._caption() == 'Image 2', card._caption()
card.deleteLater()
print('ok')
"""
    )


def test_a_reason_is_never_a_control_in_the_setting_form(run_qt) -> None:
    """Why a section could not be applied is a MESSAGE, not a setting.

    All three reasons were declared as form fields of kind "text", so each
    became a line edit labelled with the bare section name -- the Fit
    section showed a greyed "Fit" box under its "Fit model" combo and read
    as a second, broken input.  The console already has one place for what
    it could not do; the form declares controls and nothing else.
    """

    run_qt(
        """
from zlc_ui.qt import ensure_qt_app
from zlc_ui.console.panel_card_view import PanelCardView
app = ensure_qt_app(['reason'])

card = PanelCardView('panel-1', 'Panel')
card.set_size_choices(('2x2',), '2x2')
card.set_interval_choices((100,), 100)
state = {
    'signal': '', 'kind': 'image', 'size': '2x2', 'interval_ms': 100,
    'title': 'Panel', 'semantic': {}, 'display': {}, 'fit': {},
    'overlay_signal': '',
}
loud = {
    'semantic': (), 'display': (), 'fit': (),
    'semantic_unavailable': 'the semantic reason',
    'display_unavailable': 'the display reason',
    'fit_unavailable': 'the fit reason',
}
card.set_panel_projection(state, loud)
card.show()
app.processEvents()
card._open_settings()
app.processEvents()

declared = {key for form in card._settings_forms() for key in form.spec.keys}
for section in ('semantic', 'display', 'fit'):
    assert f'{section}_unavailable' not in declared, declared
assert not any('unavailable' in key for key in declared), declared

# The surface still carries them -- they are real, the console says them.
assert card._parameter_surface['fit_unavailable'] == 'the fit reason'

card.retire_settings_popup()
card.close()
app.processEvents()
"""
    )


def test_a_click_on_a_card_is_not_a_drop(run_qt) -> None:
    """A drop pins the card as the board's anchor, so a click must not be one.

    The card emitted `dropped` on any left release that followed a left
    press, zero motion included.  ConsoleBoardView then recorded that card
    as the anchor -- kept until the card leaves the board -- so every later
    repack placed it at the absolute (col, row) it happened to hold and
    stacked the others around it.  One click to look at a card, and
    widening the window rearranged everything except that card.
    """

    run_qt(
        """
from PyQt5 import QtCore, QtGui, QtTest, QtWidgets
from zlc_ui.qt import ensure_qt_app
from zlc_ui.console import PanelCardView

app = ensure_qt_app(['card-click-is-not-a-drop'])
card = PanelCardView('panel-1')
card.show()
app.processEvents()

drops = []
card.dropped.connect(drops.append)

at = QtCore.QPoint(40, 12)
QtTest.QTest.mousePress(card, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, at)
QtTest.QTest.mouseRelease(card, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, at)
app.processEvents()
assert drops == [], 'a zero-motion click dropped the card: %r' % (drops,)

# Past the window system's own threshold, it IS a drag.
far = QtCore.QPoint(
    at.x() + QtWidgets.QApplication.startDragDistance() * 3 + 8, at.y() + 6
)
QtTest.QTest.mousePress(card, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, at)
# QTest.mouseMove does not carry the held button, and the threshold is
# only crossed by a move that says the button is down.
app.sendEvent(card, QtGui.QMouseEvent(
    QtCore.QEvent.MouseMove,
    QtCore.QPointF(far),
    QtCore.Qt.NoButton,
    QtCore.Qt.LeftButton,
    QtCore.Qt.NoModifier,
))
QtTest.QTest.mouseRelease(card, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, far)
app.processEvents()
assert len(drops) == 1, 'a real drag did not drop the card: %r' % (drops,)
print('ok')
"""
    )


def test_panel_editor_body_is_opaque_so_scrolling_blits(run_qt) -> None:
    """The Edit page's scrolled body must be a widget Qt cannot see through.

    A scrolled widget Qt can see through is repainted whole on every step
    of the wheel -- the plot surface's full frame, 10 ms at three device
    pixels per point, forty times a scroll -- where an opaque one is moved
    by blitting the backing store and only the exposed strip is painted.
    Qt decides by ``WA_OpaquePaintEvent`` (or an auto-filled opaque brush,
    which a stylesheet background switches off); the stylesheet that paints
    the colour is not consulted, so the promise is made explicitly and the
    painted pixels must honour it.  The colour it paints is the pane's
    SURFACE: an Edit page is one tab among white ones, and it was the one
    grey tab while its body painted the window's page colour.
    """
    run_qt(
        """
from PyQt5 import QtCore, QtGui, QtWidgets
from zlc_ui.qt import ensure_qt_app
from zlc_ui.fluent import SURFACE
from zlc_ui.console import panel_editor_view as editor_module
app = ensure_qt_app(['test'])
owner = QtWidgets.QWidget()
state = {
    'signal': '@logic/cm/frames', 'kind': 'image', 'cell_kind': '',
    'size': '2x2', 'interval_ms': 100, 'title': 'Card', 'semantic': {},
    'display': {}, 'fit': {}, 'overlay_signal': '',
}
editor = editor_module.PanelEditorView('panel-1', {
    'panel_id': 'panel-1', 'state': state,
    'signal_options': (('camera', (('frames', '@logic/cm/frames'),)),),
    'overlay_signal_options': (),
    'interval_choices': (100, 200, 400, 800),
    'size_choices': ('1x2', '2x2', '1x4'),
    'parameter_surface': {'semantic': (), 'display': (), 'fit': (), 'paints_images': True},
    'save_directory': '.', 'frozen_signal': state['signal'],
    'frozen_publication': None, 'frozen_snapshot': None,
    'stale': False, 'producer_node_id': '',
}, owner)
body = editor.scroll.widget()
assert body is not None
assert body.testAttribute(QtCore.Qt.WA_OpaquePaintEvent), 'the scrolled body must promise an opaque paint'
assert not body.testAttribute(QtCore.Qt.WA_TranslucentBackground)
editor.resize(900, 700)
editor.show()
app.processEvents()
picture = body.grab().toImage()
for x, y in ((1, 1), (picture.width() - 2, picture.height() - 2)):
    # pixelColor, not QColor(pixel): a QColor built from a packed pixel
    # reads it as RGB and reports alpha 255 for a transparent pixel.
    colour = picture.pixelColor(x, y)
    assert colour.alpha() == 255 and colour.name().upper() == SURFACE.upper(), (x, y, colour.name(), SURFACE)
editor.deleteLater()
app.processEvents()
print('opaque body ok')
"""
    )


def test_a_card_replaced_under_its_id_leaves_the_board(run_qt) -> None:
    """One panel_id is one card.  A different object arriving under an id
    already on the board was wired and shown beside the object it replaced,
    which stayed a child of the board, still wired, behind it."""

    run_qt(
        """
from PyQt5 import QtCore, sip
from zlc_ui.qt import ensure_qt_app
from zlc_ui.console import ConsoleBoardView, PanelCardView
app = ensure_qt_app(['test'])
board = ConsoleBoardView()
old = PanelCardView('stable', 'old')
new = PanelCardView('stable', 'new')
board.set_cards((old,))
board.set_cards((new,))
assert old.isHidden(), 'the replaced card is retired'
assert old not in board._wired_cards and board._cards == {'stable': new}
app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
assert sip.isdeleted(old)
assert board.findChildren(PanelCardView, options=QtCore.Qt.FindDirectChildrenOnly) == [new]
# The same object under the same id is kept, exactly as before.
board.set_cards((new,))
assert not new.isHidden() and board._cards == {'stable': new}
"""
    )


def test_a_number_the_calibration_form_cannot_read_stays_out_of_its_draft(run_qt) -> None:
    """The Calibration row's own form read every change inside the Qt slot,
    and a box holding a number it cannot show refuses to be read: that
    refusal escaped the slot, which aborts the console.  It is no value yet
    -- the draft keeps its last one -- and a box's own normalization reaches
    the draft as one, not as an operator edit."""

    run_qt(
        """
from zlc_ui.qt import ensure_qt_app
from zlc_ui.form import FormFieldProps, FormSpec
from zlc_atom.nodes.calibration.logic_node import LOGIC_NODE
app = ensure_qt_app(['test'])
(factory,) = LOGIC_NODE.ui_contributions
form = factory()
spec = FormSpec((FormFieldProps(key='repeats', kind='int', label='Samples', default=200, minimum=1),))
form.update_projection({'form_spec': spec, 'form_values': {'repeats': 200}})
patches = []
form.draft_changed.connect(patches.append)
box = form.widget_for('repeats')
box.setProperty('numericError', 'the number does not fit the box')
form.changed.emit('repeats')
form.value_normalized.emit('repeats')
assert patches == [], patches
box.setProperty('numericError', '')
form.value_normalized.emit('repeats')
assert patches == [{'values': {'repeats': 200}, 'normalized': True}], patches
form.deleteLater()
"""
    )


def test_identical_artifact_results_keep_their_readouts(run_qt) -> None:
    """The same results re-projected destroyed every readout and built
    them again on every beat; a readout whose name is still listed is kept
    and re-told its path."""

    run_qt(
        """
from PyQt5 import QtCore
from zlc_ui.qt import ensure_qt_app
from zlc_ui.console import LogicEditorView
from zlc_ui.form import FormSpec
app = ensure_qt_app(['test'])
def projection(path, *names):
    return {'form_spec': FormSpec(()), 'form_values': {},
            'preview_offered': False, 'auto_preview': False,
            'artifact_results': tuple(
                {'name': name, 'path': path, 'contract_id': 'review'} for name in names)}
editor = LogicEditorView('logic', projection('C:/data/report.npz', 'report'))
readout = editor._artifact_result_readouts['report']
destroyed = []
readout.destroyed.connect(lambda *_: destroyed.append(True))
editor.update_projection(projection('C:/data/report.npz', 'report'))
app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
assert editor._artifact_result_readouts['report'] is readout and destroyed == []
editor.update_projection(projection('C:/data/report-2.npz', 'report'))
assert editor._artifact_result_readouts['report'] is readout
assert readout.text() == 'C:/data/report-2.npz', 'a new path is told to the same readout'
editor.update_projection(projection('C:/data/report-2.npz', 'report', 'summary'))
assert editor._artifact_result_readouts['report'] is not readout, 'a new name set rebuilds the rows'
assert list(editor._artifact_result_readouts) == ['report', 'summary']
editor.deleteLater(); app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
"""
    )
