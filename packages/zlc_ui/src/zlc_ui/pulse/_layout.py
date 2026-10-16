"""Shared display-scaled geometry for pulse views.

The editor keeps the suite's one page rhythm -- ``card_pad``, ``row_gap``
and ``section_gap`` from :mod:`zlc_ui.fluent` -- and adds only what is its
own: the port-row table's height and pitch, the card widths, and the label
column of its side panels.
"""

from __future__ import annotations

from PyQt5 import QtCore, QtWidgets

from zlc_ui.fluent import (
    FluentLabel, card_pad, control_height, row_gap, scaled_px, section_gap, setting_label_width,
)
from zlc_ui.fluent.fluent import _numeric_text_width


#: The count every repeat box is sized for: a common count shows whole, and
#: a longer one widens its box as it is shown or typed (:func:`count_box_width`).
COUNT_SAMPLE = "10000"
#: The widest a side panel's label column grows: a longer port name elides
#: there (whole in its tooltip) rather than pushing every field away.
CHANNEL_LABEL_WIDTH = 100
TIME_UNIT_WIDTH = 60
#: Every card and side panel opens with this many header rows above its
#: port rows: a period's Duration caption, value, unit and name.
PANEL_HEADER_ROWS = 4
CHANNEL_ROW_SPACING = 4
PERIOD_CARD_WIDTH = 160
#: A spacer card carries one number and a column of circles: narrow enough
#: to read as a gap between periods rather than as one of them.
SPACER_CARD_WIDTH = 94


def row_spacing() -> int:
    """The pitch of the port rows, the one table of the editor: row N of
    every card and side panel is the same output, read across, so the rows
    are packed closer than a form's to show more outputs at once."""

    return scaled_px(CHANNEL_ROW_SPACING, minimum=3)


def label_column_width(labels) -> int:
    """A side panel's label column: its widest label and the air after it,
    capped at :data:`CHANNEL_LABEL_WIDTH`."""

    return min(scaled_px(CHANNEL_LABEL_WIDTH, minimum=84), setting_label_width(labels, minimum=0))


def config_name_width() -> int:
    """The box a Config name is chosen in, on every page that binds fields
    to names: one width, so those grids read alike."""

    return scaled_px(240, minimum=180)


def bound_field_width(labels) -> int:
    """The column of field labels on a page that binds fields to Config
    names: its widest label and the air after it, so each name's box stands
    beside the field it binds; a longer label elides."""

    return min(scaled_px(320), setting_label_width(labels, minimum=0))


def channel_name_edit_width() -> int:
    return scaled_px(108, minimum=88)


def time_unit_width() -> int:
    return scaled_px(TIME_UNIT_WIDTH, minimum=62)


def panel_top_height() -> int:
    """The header block every card and side panel opens with: its header
    rows at the row gap, and the section gap that parts them from the port
    rows (less the port rows' own spacing, which the card's column adds
    after the block).  One height for all of them, so row N of every card
    and side panel stands on one y."""

    return (PANEL_HEADER_ROWS * control_height() + (PANEL_HEADER_ROWS - 1) * row_gap()
            + section_gap() - row_spacing())


def period_card_width(*, ports=(), analog_mode_choices=()) -> int:
    """Keep the shared compact card wide enough for its actual DAC controls.

    Fonts and binding badges have minimum sizes, so scaling the card alone
    can leave no room for a legal three-digit DAC code on a small display.
    Measure the existing controls, including their real style and margins;
    the Period and closed Component use this same result.
    """
    width = scaled_px(PERIOD_CARD_WIDTH, minimum=112)
    codes = [str(int(bound)) for port in ports if port.kind == "dac"
             for bound in (port.lo, port.hi) if bound is not None]
    if not codes:
        return width
    from zlc_ui.fluent import FluentComboBox
    from .scan_line_edit import FluentScanLineEdit

    combo, edit = FluentComboBox(), FluentScanLineEdit()
    try:
        for choice in analog_mode_choices:
            combo.addItem(choice.label, choice.value)
        combo.ensurePolished()
        edit.ensurePolished()
        code_width = max(edit.fontMetrics().horizontalAdvance(text) + 2 for text in codes)
        return max(width, 2 * card_pad() + combo.sizeHint().width() + row_gap()
                   + edit.width() - _numeric_text_width(edit) + code_width)
    finally:
        combo.deleteLater()
        edit.deleteLater()


def period_control_width(card_width: int) -> int:
    """The one control column of a card: its full inner width, so the
    header controls and the port rows share both edges."""

    return max(scaled_px(76, minimum=68), card_width - 2 * card_pad())


def spacer_card_width() -> int:
    return scaled_px(SPACER_CARD_WIDTH, minimum=68)


def spacer_control_width(card_width: int) -> int:
    """The one control column of a spacer card: its full inner width."""
    return card_width - 2 * card_pad()


def card_caption(text: str) -> FluentLabel:
    """A card's header caption ("Duration", a bracket's "Repeats"): one row
    tall, its words on the card's content edge -- the x of the card's title
    and of the field boxes under it, as a side panel's labels stand on
    theirs -- so a card stacks its words on one left edge."""

    label = FluentLabel(text)
    label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
    label.setFixedHeight(control_height())
    return label


def count_box_width(box: QtWidgets.QAbstractSpinBox, text: str = "") -> int:
    """The width of a box for a whole count: :data:`COUNT_SAMPLE` or
    ``text``, whichever is wider, shown whole.

    One measure for every repeat box of the editor (a Pulse's Repeat, the
    Scan repeats, a bracket's Repeats): the digits' advance and the chrome
    the box and its line edit keep beside them -- frame, padding, stepper
    and the edit's own text margins (a binding badge's room) -- read from
    the style without resizing the box, since a resize refits a half-typed
    number.  A box that shows or takes a longer count is re-sized from it.
    """

    edit = box.lineEdit()
    box.ensurePolished()
    edit.ensurePolished()
    option = QtWidgets.QStyleOptionSpinBox()
    box.initStyleOption(option)
    area = box.style().subControlRect(
        QtWidgets.QStyle.CC_SpinBox, option, QtWidgets.QStyle.SC_SpinBoxEditField, box)
    chrome = box.width() - area.width() + edit.width() - _numeric_text_width(edit)
    metrics = edit.fontMetrics()
    digits = max(metrics.horizontalAdvance(COUNT_SAMPLE), metrics.horizontalAdvance(str(text)))
    return digits + 2 + chrome


__all__ = [
    "bound_field_width", "card_caption", "channel_name_edit_width",
    "config_name_width", "count_box_width",
    "label_column_width", "panel_top_height",
    "period_card_width", "period_control_width",
    "row_spacing", "spacer_card_width",
    "spacer_control_width", "time_unit_width",
]
