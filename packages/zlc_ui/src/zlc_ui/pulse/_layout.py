"""Shared display-scaled geometry for pulse views.

The editor keeps the suite's one page rhythm -- ``card_pad``, ``row_gap``
and ``section_gap`` from :mod:`zlc_ui.fluent` -- and adds only what is its
own: the port-row table's height and pitch, the card widths, and the label
column of its side panels.
"""

from __future__ import annotations

from PyQt5 import QtCore, QtWidgets

from zlc_ui.fluent import (
    card_pad, row_gap,
    FONT, MUTED_LABEL_STYLE, ElidedLabel,
    fluent_font_size, scaled_px, setting_label_width,
)


ROW_HEIGHT = 30
#: The widest a side panel's label column grows: a longer port name elides
#: there (whole in its tooltip) rather than pushing every field away.
CHANNEL_LABEL_WIDTH = 100
TIME_UNIT_WIDTH = 60
PANEL_TOP_HEIGHT = 178
CHANNEL_ROW_SPACING = 4
PERIOD_CARD_WIDTH = 160
#: A spacer card carries one number and a column of circles: narrow enough
#: to read as a gap between periods rather than as one of them.
SPACER_CARD_WIDTH = 94


def px(value: int | float, *, minimum: int = 1) -> int:
    return scaled_px(value, minimum=minimum)


def row_height() -> int:
    return px(ROW_HEIGHT, minimum=22)


def row_spacing() -> int:
    """The pitch of the port rows, the one table of the editor: row N of
    every card and side panel is the same output, read across, so the rows
    are packed closer than a form's to show more outputs at once."""

    return px(CHANNEL_ROW_SPACING, minimum=3)






def channel_label_width() -> int:
    return px(CHANNEL_LABEL_WIDTH, minimum=84)


def label_column_width(labels) -> int:
    """A side panel's label column: its widest label and the air after it,
    capped at :func:`channel_label_width`."""

    return min(channel_label_width(), setting_label_width(labels, minimum=0))


def config_name_width() -> int:
    """The box a Config name is chosen in, on every page that binds fields
    to names: one width, so those grids read alike."""

    return px(240, minimum=180)


def bound_field_width(labels) -> int:
    """The column of field labels on a page that binds fields to Config
    names: its widest label and the air after it, so each name's box stands
    beside the field it binds; a longer label elides."""

    return min(px(320), setting_label_width(labels, minimum=0))


def channel_name_edit_width() -> int:
    return px(108, minimum=88)


def time_unit_width() -> int:
    return px(TIME_UNIT_WIDTH, minimum=62)


def panel_top_height() -> int:
    return px(PANEL_TOP_HEIGHT, minimum=138)


def period_card_width(*, ports=(), analog_mode_choices=()) -> int:
    """Keep the shared compact card wide enough for its actual DAC controls.

    Fonts and binding badges have minimum sizes, so scaling the card alone
    can leave no room for a legal three-digit DAC code on a small display.
    Measure the existing controls, including their real style and margins;
    the Period and closed Component use this same result.
    """
    width = px(PERIOD_CARD_WIDTH, minimum=112)
    codes = [str(int(bound)) for port in ports if port.kind == "dac"
             for bound in (port.lo, port.hi) if bound is not None]
    if not codes:
        return width
    from zlc_ui.fluent import FluentComboBox
    from zlc_ui.fluent.fluent import _numeric_text_width
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

    return max(px(76, minimum=68), card_width - 2 * card_pad())


def spacer_card_width() -> int:
    return px(SPACER_CARD_WIDTH, minimum=68)


def spacer_control_width(card_width: int) -> int:
    """The one control column of a spacer card: its full inner width."""
    return card_width - 2 * card_pad()


def form_label(text: str) -> ElidedLabel:
    """A side panel's row label: left-aligned at the card's content edge,
    centred on the row's height, elided beyond its column -- whose width
    the panel sets on every label from the labels it shows."""

    label = ElidedLabel(text)
    label.setFixedHeight(row_height())
    return label


def quiet(label: QtWidgets.QLabel) -> QtWidgets.QLabel:
    """Draw a label's words in the muted ink of a summary or a status: they
    carry information, in a lighter voice than the values beside them."""

    label.setStyleSheet(f'QLabel {{ {MUTED_LABEL_STYLE} font: {fluent_font_size()}pt "{FONT}"; }}')
    return label


def form_control_cell(widget: QtWidgets.QWidget) -> QtWidgets.QWidget:
    widget.setFixedHeight(row_height())
    cell = QtWidgets.QWidget()
    cell.setStyleSheet("background: transparent;")
    layout = QtWidgets.QHBoxLayout(cell)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)
    fixed_width = widget.minimumWidth() > 0 and widget.maximumWidth() == widget.minimumWidth()
    if fixed_width:
        layout.addWidget(widget, 0, QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        layout.addStretch(1)
    else:
        layout.addWidget(widget, 1)
    return cell


def add_labeled_widget(layout, label_text: str, widget: QtWidgets.QWidget) -> ElidedLabel:
    """One ``label | control`` row of a side panel.  A readout in the control
    column is a :class:`~zlc_ui.fluent.FluentReadoutEdit` at its default
    inset, so its text starts where the fields' text starts."""

    row = QtWidgets.QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(row_gap())
    label = form_label(label_text)
    row.addWidget(label)
    row.addWidget(form_control_cell(widget), 1)
    layout.addLayout(row)
    return label


__all__ = [
    "add_labeled_widget", "bound_field_width", "channel_label_width", "channel_name_edit_width",
    "config_name_width", "form_control_cell", "form_label",
    "label_column_width", "panel_top_height", "quiet",
    "period_card_width", "period_control_width", "px", "row_height",
    "row_spacing", "spacer_card_width",
    "spacer_control_width", "time_unit_width",
]
