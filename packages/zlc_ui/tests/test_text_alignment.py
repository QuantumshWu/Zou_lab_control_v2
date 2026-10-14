from __future__ import annotations


def test_every_field_kind_puts_its_text_on_one_line(run_qt) -> None:
    """A line edit, a combo and an editable combo of one height show their
    text on the same pixel rows: a row of mixed fields reads as one line."""

    run_qt(
        r"""
from PyQt5 import QtCore, QtGui, QtWidgets
from zlc_ui.qt import ensure_qt_app
from zlc_ui.fluent import FluentComboBox, FluentLineEdit

app = ensure_qt_app(['zlc-ui-tests'])
row = QtWidgets.QWidget()
row.setStyleSheet("background: white;")
layout = QtWidgets.QHBoxLayout(row)
text = "HELM"
line = FluentLineEdit(text)
combo = FluentComboBox()
combo.addItem(text)
editable = FluentComboBox()
editable.setEditable(True)
editable.setEditText(text)
fields = (line, combo, editable)
for field in fields:
    field.setFixedWidth(160)
    layout.addWidget(field)
row.show()
for _ in range(10):
    app.processEvents()
assert editable.lineEdit().geometry() == editable._editor_rect(), "the editor fills exactly the text room"
image = row.grab().toImage().convertToFormat(QtGui.QImage.Format_RGB32)
ratio = row.devicePixelRatioF()


def ink_rows(widget):
    # The first letter only: a focused line edit also draws its cursor.
    left = int((widget.x() + 3) * ratio)
    right = int((widget.x() + 18) * ratio)
    rows = [y for y in range(int(widget.y() * ratio) + 2, int((widget.y() + widget.height()) * ratio) - 2)
            if any(QtGui.QColor(image.pixel(x, y)).value() < 110 for x in range(left, right))]
    return rows[0], rows[-1]


found = [ink_rows(field) for field in fields]
assert len(set(found)) == 1, found
row.close()
"""
    )
