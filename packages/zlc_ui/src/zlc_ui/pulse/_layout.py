"""Shared display-scaled geometry for pulse views.

One spacing scale for every page of the editor, in design px scaled through
:func:`px`: a card's content stands :func:`card_pad` inside its edge, cells
of one row stand :func:`inline_gap` apart, rows of a form and cards side by
side stand :func:`row_gap` apart, and a page's sections -- and its content
from the page's own edge -- stand :func:`section_gap` apart.  Each page took
private literals before, so the same page edge jumped from tab to tab and
one strip of cards had two different gaps.
"""

from __future__ import annotations

from PyQt5 import QtCore, QtWidgets

from zlc_ui.fluent import (
    CARD_PAD, FONT, MUTED_INK, MUTED_LABEL_STYLE, ElidedLabel, FluentLabel, FluentReadoutEdit,
    field_text_inset, fluent_font_size, line_edit_text_right, scaled_px,
    setting_label_width, window_pad,
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
INLINE_GAP = 6
ROW_GAP = 6


def px(value: int | float, *, minimum: int = 1) -> int:
    return scaled_px(value, minimum=minimum)


def row_height() -> int:
    return px(ROW_HEIGHT, minimum=22)


def row_spacing() -> int:
    """The pitch of the port rows, the one table of the editor: row N of
    every card and side panel is the same output, read across, so the rows
    are packed closer than a form's to show more outputs at once."""

    return px(CHANNEL_ROW_SPACING, minimum=3)


def card_pad() -> int:
    """A card's content inset, on every side: where FluentGroupBox paints
    its title, so the title, the labels and the controls start on one x."""

    return scaled_px(CARD_PAD)


def inline_gap() -> int:
    """Between two cells of one row: a label and its control, a field and
    its unit, two buttons."""

    return px(INLINE_GAP, minimum=4)


def row_gap() -> int:
    """Between two rows of a form, and between two cards side by side."""

    return px(ROW_GAP, minimum=4)


def section_gap() -> int:
    """Between a page's sections, and from the page's edge to its content:
    every tab's cards start and end on the same two x."""

    return window_pad(1)


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
        return max(width, 2 * card_pad() + combo.sizeHint().width() + inline_gap()
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


def field_column_readout(readout: FluentReadoutEdit) -> FluentReadoutEdit:
    """Start a read-only value's text where a field's text starts.

    A readout has no box, so its text sits at its own left edge -- right
    under a column header, but a field's padding to the left of the typed
    text in the field above or below it.  Standing in a column of fields
    it takes that padding as its left text margin, measured from where its
    text starts now, so the value and the typed text share one x.
    """

    readout.ensurePolished()
    option = QtWidgets.QStyleOptionFrame()
    option.initFrom(readout)
    contents = readout.style().subElementRect(QtWidgets.QStyle.SE_LineEditContents, option, readout)
    margins = readout.textMargins()
    text_left = contents.left() + margins.left() + (contents.right() + 1 - line_edit_text_right(readout))
    readout.setTextMargins(margins.left() + field_text_inset() - text_left,
                           margins.top(), margins.right(), margins.bottom())
    return readout


def quiet(label: QtWidgets.QLabel) -> QtWidgets.QLabel:
    """Draw a label's words in the muted ink of a summary or a status: they
    carry information, in a lighter voice than the values beside them."""

    label.setStyleSheet(f'QLabel {{ {MUTED_LABEL_STYLE} font: {fluent_font_size()}pt "{FONT}"; }}')
    return label


def column_header(text: str = "") -> FluentLabel:
    """A column's header over a grid of rows, in a table header's look:
    muted ink, semibold.  The weight is in the sheet because a FluentLabel's
    sheet sets its font and overrides a bold set with setFont."""

    label = FluentLabel(text)
    label.setStyleSheet(
        f'QLabel {{ color: {MUTED_INK}; font: {fluent_font_size()}pt "{FONT}"; '
        f'font-weight: 600; background: transparent; }}'
    )
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
    """One ``label | control`` row of a side panel; a readout in the control
    column starts its text where the fields' text starts."""

    row = QtWidgets.QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(inline_gap())
    label = form_label(label_text)
    row.addWidget(label)
    if isinstance(widget, FluentReadoutEdit):
        field_column_readout(widget)
    row.addWidget(form_control_cell(widget), 1)
    layout.addLayout(row)
    return label


__all__ = [
    "add_labeled_widget", "bound_field_width", "card_pad", "channel_label_width", "channel_name_edit_width",
    "column_header", "config_name_width", "field_column_readout", "form_control_cell", "form_label",
    "inline_gap", "label_column_width", "panel_top_height", "quiet",
    "period_card_width", "period_control_width", "px", "row_gap", "row_height",
    "row_spacing", "section_gap", "spacer_card_width",
    "spacer_control_width", "time_unit_width",
]
