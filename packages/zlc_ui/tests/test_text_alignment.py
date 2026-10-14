from __future__ import annotations


def test_every_text_in_a_row_sits_on_one_baseline(run_qt) -> None:
    """A row label, a section title, a line edit, a combo, an editable
    combo, a spin box and a button of one height put their text on one
    baseline, and a field's capitals are centred in its frame to a device
    pixel: a row of mixed controls reads as one line."""

    run_qt(
        r"""
from PyQt5 import QtCore, QtGui, QtWidgets
from zlc_ui.qt import ensure_qt_app
from zlc_ui.fluent import (
    FluentButton, FluentComboBox, FluentLabel, FluentLineEdit, FluentSectionLabel, FluentSpinBox,
    control_height, frame_inner_rect,
)

app = ensure_qt_app(['zlc-ui-tests'])
row = QtWidgets.QWidget()
row.setStyleSheet("background: white;")
row.setFocusPolicy(QtCore.Qt.StrongFocus)
layout = QtWidgets.QHBoxLayout(row)
text = "HELM"
label = FluentLabel(text)
section = FluentSectionLabel(text)
line = FluentLineEdit(text)
combo = FluentComboBox()
combo.addItem(text)
editable = FluentComboBox()
editable.setEditable(True)
editable.setEditText(text)
spin = FluentSpinBox()
spin.setValue(44)
button = FluentButton(text)
fields = (line, combo, editable, spin)
for widget in (label, section, *fields, button):
    layout.addWidget(widget)
for widget in (*fields, button):
    widget.setFixedSize(120, control_height())
row.show()
row.setFocus()
for _ in range(10):
    app.processEvents()
assert editable.lineEdit().geometry() == editable._editor_rect(), "the editor fills exactly the text room"
image = row.grab().toImage().convertToFormat(QtGui.QImage.Format_RGB32)
ratio = row.devicePixelRatioF()


def ink_rows(widget):
    left, right = int((widget.x() + 2) * ratio), int((widget.x() + widget.width() - 2) * ratio)
    rows = [y for y in range(int(widget.y() * ratio) + 2, int((widget.y() + widget.height()) * ratio) - 2)
            if any(QtGui.QColor(image.pixel(x, y)).lightness() < 128 for x in range(left, right))]
    return rows[0], rows[-1]


# The last ink row of a capital or a digit is the baseline.
found = {type(widget).__name__ + str(index): ink_rows(widget)[1]
         for index, widget in enumerate((label, section, *fields, button))}
assert len(set(found.values())) == 1, found
for field in (line, combo, editable):
    first, last = ink_rows(field)
    inner = frame_inner_rect(field)
    middle = field.y() + inner.top() + inner.height() / 2.0
    offset = (first + last + 1) / 2.0 / ratio - middle
    assert abs(offset) <= 1.0 / ratio, (type(field).__name__, offset)
row.close()
"""
    )
