"""One real-platform render pass for ``test_frame_joints``.

Run by the test, one process per (device-pixel ratio, fluent scale), on the
REAL platform: the offscreen platform has no device-pixel ratio but 1, and a
joint that is exact at 1 is exactly what hid the spin divider's half-pixel
gap.  The ratio is set with QT_SCALE_FACTOR by the parent; this child checks
it got it, renders every frame-with-parts control, measures the joints in
device pixels and prints one JSON line of verdicts.  Nothing is taken from a
colour constant: the edge, the ground and the tint are read from the render.

    python _joint_child.py '{"dpr": 1.5, "scale": 0.85, "crops": "<dir>"}'
    python _joint_child.py discover
"""

from __future__ import annotations

import json
import math
import os
import sys
import time

import zou_lab_control  # noqa: F401 - this checkout's layers first

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from zlc_ui.qt import ensure_qt_app


def _discover() -> None:
    app = ensure_qt_app()
    screen = app.primaryScreen()
    print(json.dumps({"native": None if screen is None else float(screen.devicePixelRatio())}))


if len(sys.argv) > 1 and sys.argv[1] == "discover":
    _discover()
    sys.exit(0)

ARGS = json.loads(sys.argv[1])
APP = ensure_qt_app()
from zlc_ui.fluent import fluent as F  # noqa: E402 - after the app

F.set_fluent_scale(float(ARGS["scale"]))
DPR = float(ARGS["dpr"])
CROPS = ARGS.get("crops") or ""
INTEGRAL = abs(DPR - round(DPR)) < 1e-6
BD = max(1, round(F.FIELD_BORDER_PX * DPR))
RESULTS: list[dict] = []
FOCUS_OK = True


def pump(seconds: float = 0.05) -> None:
    end = time.monotonic() + seconds
    while True:
        APP.processEvents()
        if time.monotonic() >= end:
            break
        time.sleep(0.005)


class Host(QtWidgets.QWidget):
    def __init__(self, width: int, height: int):
        super().__init__(None, QtCore.Qt.Tool | QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose, False)
        palette = self.palette()
        palette.setColor(QtGui.QPalette.Window, QtGui.QColor(F.BG))
        self.setPalette(palette)
        self.setAutoFillBackground(True)
        self.resize(width, height)
        self.move(APP.primaryScreen().availableGeometry().topLeft())
        self.setFocusPolicy(QtCore.Qt.StrongFocus)


def show(host: Host) -> None:
    host.show()
    pump(0.15)
    host.setFocus(QtCore.Qt.OtherFocusReason)
    pump(0.05)


def grab(host: Host) -> np.ndarray:
    image = host.grab().toImage().convertToFormat(QtGui.QImage.Format_RGB32)
    width, height = image.width(), image.height()
    pointer = image.constBits()
    pointer.setsize(image.byteCount())
    raw = np.frombuffer(pointer, np.uint8).reshape(height, image.bytesPerLine() // 4, 4)
    return raw[:, :width, 2::-1].astype(int).copy()


def check(case: str, name: str, ok: bool, detail: str = "", *, image=None, box=None) -> bool:
    entry = {"case": case, "check": name, "ok": bool(ok), "detail": detail}
    if not ok and CROPS and image is not None and box is not None:
        os.makedirs(CROPS, exist_ok=True)
        x0, y0, x1, y1 = (int(v) for v in box)
        x0, y0 = max(0, x0), max(0, y0)
        part = np.ascontiguousarray(image[y0:y1, x0:x1].astype(np.uint8))
        if part.size:
            qimage = QtGui.QImage(part.data, part.shape[1], part.shape[0], 3 * part.shape[1],
                                  QtGui.QImage.Format_RGB888)
            path = os.path.join(CROPS, f"{case}__{name}__x{DPR:g}.png".replace("/", "_").replace(" ", "_"))
            qimage.scaled(part.shape[1] * 8, part.shape[0] * 8).save(path)
            entry["crop"] = path
    RESULTS.append(entry)
    return ok


def eq(a, b) -> bool:
    return bool(np.array_equal(np.asarray(a), np.asarray(b)))


def hexs(pixel) -> str:
    return "%02X%02X%02X" % tuple(int(v) for v in pixel)


def dev(value: float) -> int:
    """The device pixel a logical coordinate starts."""

    return int(math.floor(value * DPR + 1e-6))


def edge_px(value: float) -> int:
    """The first device pixel whose centre lies at or past a logical edge."""

    return int(math.floor(value * DPR + 0.5))


def ink_box(image, x0, y0, x1, y1, ground, threshold=40):
    """Bounding box (device px, inclusive) of pixels that differ from ground."""

    region = image[y0:y1, x0:x1]
    mask = np.abs(region - np.asarray(ground)).max(axis=2) > threshold
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return x0 + xs.min(), y0 + ys.min(), x0 + xs.max(), y0 + ys.max()


# ------------------------------------------------------------------ spin boxes
def make_spin(kind: str):
    if kind == "spin":
        box = F.FluentSpinBox()
        box.setRange(-1000, 1000)
        box.setValue(5)
    elif kind == "intbox":
        box = F.fluent_integer_box(minimum=0, maximum=100000)
        box.setValue(5)
    else:
        box = F.FluentDoubleSpinBox()
        box.setRange(-1000, 1000)
        box.setValue(1.5)
    return box


def spin_state(box, host, state: str) -> bool:
    global FOCUS_OK
    if state == "disabled":
        box.setEnabled(False)
    elif state == "readonly":
        box.setReadOnly(True)
    elif state == "focus":
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and not box.hasFocus():
            host.activateWindow()
            APP.setActiveWindow(host)
            box.setFocus(QtCore.Qt.TabFocusReason)
            pump(0.05)
        if not box.hasFocus():
            FOCUS_OK = False
            return False
    elif state.startswith("hover"):
        _divider, up, down = F._spin_stepper_cells(box)
        cell = up if state == "hover_up" else down
        pos = cell.center()
        box.setAttribute(QtCore.Qt.WA_UnderMouse, True)
        APP.sendEvent(box, QtGui.QHoverEvent(QtCore.QEvent.HoverEnter, pos, QtCore.QPointF(-1, -1)))
        APP.sendEvent(box, QtGui.QHoverEvent(QtCore.QEvent.HoverMove, pos, pos))
    return True


def spin_case(kind: str, state: str, offset: int) -> None:
    case = f"{kind}/{state}/o{offset}"
    host = Host(220, 80)
    box = make_spin(kind)
    box.setParent(host)
    box.setGeometry(offset, offset, 160, box.sizeHint().height())
    height = box.height()
    show(host)
    if not spin_state(box, host, state):
        RESULTS.append({"case": case, "check": "state", "ok": True, "detail": "FOCUS-UNAVAILABLE", "skip": True})
        host.close()
        return
    pump(0.1)
    box.repaint()
    pump(0.05)
    image = grab(host)
    ox, oy = offset, offset
    width = box.width()
    # The device rows/columns whose centres lie inside the widget.
    x0, y0 = edge_px(ox), edge_px(oy)
    x1, y1 = edge_px(ox + width), edge_px(oy + height)  # exclusive
    divider_x, up_cell, down_cell = F._spin_stepper_cells(box)
    column = F.stepper_column(box)
    ymid = dev(oy + height / 2.0)
    # The edge's own colour: its first column, or -- where a fractional
    # origin leaves no whole pixel of it -- its darkest.
    edge = min((image[ymid, x] for x in range(x0 - 1, x0 + 2 * BD + 1)), key=lambda p: int(p.sum()))
    ground = image[ymid, x0 + 3 * BD + 1]
    zone = range(dev(ox + column.left() - 4), dev(ox + column.left()) + 1)
    lit = [x for x in zone if int(image[ymid, x].sum()) <= int(edge.sum())]
    if lit:
        edge = image[ymid, lit[0]]
    expected_first = edge_px(ox + column.left() - F.FIELD_BORDER_PX)
    crop = (dev(ox + divider_x - 6), y0 - 2, dev(ox + divider_x + 6), y1 + 2)
    contiguous = bool(lit) and lit == list(range(lit[0], lit[0] + len(lit)))
    check(case, "T-S1 divider columns", contiguous and len(lit) == BD and lit[0] == expected_first,
          f"edge {hexs(edge)} cols {lit} expected {BD} from {expected_first}", image=image, box=crop)
    if not lit:
        host.close()
        return
    first, last = lit[0], lit[-1]
    rows_bad = [y for y in range(y0, y1) for x in lit if not eq(image[y, x], edge)]
    check(case, "T-S2 divider meets top and bottom", not rows_bad,
          f"rows not edge-coloured: {sorted(set(rows_bad))[:8]}", image=image, box=crop)
    hovered = state.startswith("hover")
    probe_y = dev(oy + (up_cell if state != "hover_down" else down_cell).center().y()) if hovered else ymid
    left_px, right_px = image[probe_y, first - 1], image[probe_y, last + 1]
    tint = image[dev(oy + (up_cell if state == "hover_up" else down_cell).center().y()), last + 1] if hovered else None
    right_ground = tint if hovered else ground
    check(case, "T-S3 no smear beside divider", eq(left_px, ground) and eq(right_px, right_ground),
          f"left {hexs(left_px)} right {hexs(right_px)} ground {hexs(ground)}", image=image, box=crop)
    if INTEGRAL:
        reference = first - 2 * BD
        edge_rows = list(range(y0, y0 + BD)) + list(range(y1 - BD, y1))
        bad = [y for y in edge_rows if not eq(image[y, reference], edge) or not eq(image[y, first], edge)]
        check(case, "T-S4 top/bottom edge untouched", not bad, f"rows {bad}", image=image, box=crop)
    else:
        bad = []
        for y in list(range(y0, y0 + 2 * BD)) + list(range(y1 - 2 * BD, y1)):
            if image[y, first].sum() > image[y, first - 2 * BD].sum():
                bad.append(y)
        check(case, "T-S4 edge not lightened at divider", not bad, f"rows {bad}", image=image, box=crop)
    option = QtWidgets.QStyleOptionSpinBox()
    box.initStyleOption(option)
    up = QtCore.QRectF(box.style().subControlRect(
        QtWidgets.QStyle.CC_SpinBox, option, QtWidgets.QStyle.SC_SpinBoxUp, box))
    b = F.FIELD_BORDER_PX
    sheet = up.adjusted(b, b, -b, 0)
    line = box.lineEdit().geometry()
    check(case, "T-S5 sheet agrees with stepper column",
          abs(sheet.left() - column.left()) < 1e-6 and abs(sheet.right() - column.right()) < 1e-6
          and line.x() + line.width() <= divider_x,
          f"sheet {sheet.left()}..{sheet.right()} column {column.left()}..{column.right()} "
          f"edit ends {line.x() + line.width()} divider {divider_x}")
    chip = getattr(box, "_step_chip", None)
    if chip is not None and chip.isVisible():
        inner = F.frame_inner_rect(box)
        geometry = QtCore.QRectF(chip.geometry())
        margins = box.lineEdit().textMargins()
        text_end = line.x() + line.width() - margins.right()
        clear = 1.0 / DPR - 1e-6
        check(case, "T-B chip inside the field",
              geometry.left() - text_end >= 0
              and divider_x - geometry.right() >= clear
              and geometry.top() - inner.top() >= clear
              and inner.bottom() - geometry.bottom() >= clear
              and abs((geometry.top() - inner.top()) - (inner.bottom() - geometry.bottom())) <= 1.0 / DPR + 1e-6,
              f"chip {geometry} inner {inner} divider {divider_x} text ends {text_end}")
    if hovered:
        cell = up_cell if state == "hover_up" else down_cell
        row = dev(oy + cell.center().y())
        check(case, "T-H1 tint starts at divider", eq(image[row, last + 1], tint) and not eq(tint, ground),
              f"tint {hexs(tint)} col {last + 1} {hexs(image[row, last + 1])}", image=image, box=crop)
        right_edge = x1 - BD
        if INTEGRAL:
            check(case, "T-H1 tint reaches right edge",
                  eq(image[row, right_edge - 1], tint) and eq(image[row, right_edge], edge),
                  f"{hexs(image[row, right_edge - 1])} | {hexs(image[row, right_edge])}", image=image,
                  box=(right_edge - 8 * BD, y0 - 2, x1 + 2, y1 + 2))
        whites = [x for x in range(last + 1, x1) if eq(image[row, x], ground)]
        check(case, "T-H1 no ground strip in tint row", not whites, f"cols {whites}", image=image,
              box=(last - 4, y0 - 2, x1 + 2, y1 + 2))
        column_x = last + 1
        if state == "hover_up":
            rows = range(y0, dev(oy + cell.center().y()))
        else:
            rows = range(dev(oy + cell.center().y()), y1)
        ground_rows = [y for y in rows if eq(image[y, column_x], ground)]
        check(case, "T-H1 tint reaches top/bottom edge", not ground_rows, f"rows {ground_rows}",
              image=image, box=crop)
        if INTEGRAL:
            if state == "hover_up":
                check(case, "T-H1 tint under top edge", eq(image[y0 + BD, column_x], tint),
                      hexs(image[y0 + BD, column_x]), image=image, box=crop)
            else:
                check(case, "T-H1 tint over bottom edge", eq(image[y1 - BD - 1, column_x], tint),
                      hexs(image[y1 - BD - 1, column_x]), image=image, box=crop)
    host.close()


def chip_case() -> None:
    """The step chip under the pointer: a pill clear of the frame and the
    divider on every side."""

    case = "dspin/chip_hover"
    host = Host(220, 80)
    box = make_spin("dspin")
    box.setParent(host)
    box.setGeometry(16, 16, 160, box.sizeHint().height())
    show(host)
    chip = box._step_chip
    chip.setAttribute(QtCore.Qt.WA_UnderMouse, True)
    chip.repaint()
    pump(0.05)
    image = grab(host)
    geometry = chip.geometry().translated(box.pos())
    row = dev(geometry.y() + geometry.height() / 2.0)
    ground = image[row, dev(box.x()) + 3 * BD + 1]
    pill = image[row, dev(geometry.x() + 2)]
    divider = edge_px(box.x() + F._spin_stepper_cells(box)[0])
    gap = [x for x in range(edge_px(geometry.x() + geometry.width()), divider) if eq(image[row, x], ground)]
    middle = dev(geometry.x() + geometry.width() / 2.0)
    above = image[edge_px(geometry.y()) - 1, middle]
    below = image[edge_px(geometry.y() + geometry.height()), middle]
    crop = (dev(geometry.x()) - 6, dev(box.y()) - 2, divider + 6, dev(box.y() + box.height()) + 2)
    check(case, "T-B hovered chip is a pill clear of frame and divider",
          not eq(pill, ground) and bool(gap) and eq(above, ground) and eq(below, ground),
          f"pill {hexs(pill)} ground {hexs(ground)} gap cols {gap} above {hexs(above)} below {hexs(below)}",
          image=image, box=crop)
    host.close()


# ------------------------------------------------------------------ chevrons
def chevron_rows(oy: float, centre_y: float) -> tuple[int, int]:
    """Device rows that hold a chevron centred on ``centre_y`` and nothing
    of the frame's corner arcs: its half-height, its pen and a pixel."""

    reach = F.scaled_px(F.COMBO_TRI_SIZE) / 4.0 + max(1.0, F.scaled_px(3) / 2.0) / 2.0 + 1.0
    return dev(oy + centre_y - reach), dev(oy + centre_y + reach) + 1


def chevron_case() -> None:
    case = "chevrons"
    host = Host(240, 200)
    widgets = []
    y = 16
    for kind in ("spin", "dspin", "intbox"):
        box = make_spin(kind)
        box.setParent(host)
        box.setGeometry(16, y, 160, box.sizeHint().height())
        widgets.append((kind, box))
        y += box.height() + 8
    combo = F.FluentComboBox(host)
    combo.addItems(["Alpha", "Beta"])
    combo.setGeometry(16, y, 160, combo.sizeHint().height())
    show(host)
    pump(0.1)
    image = grab(host)
    sizes = {}
    centres = {}
    for kind, box in widgets:
        _divider, up_cell, down_cell = F._spin_stepper_cells(box)
        ox, oy = box.x(), box.y()
        ground = image[dev(oy + up_cell.top() + 1), dev(ox + up_cell.left()) + 1]
        for name, cell in (("up", up_cell), ("down", down_cell)):
            x0, x1 = dev(ox + cell.left()), dev(ox + cell.right())
            ya, yb = chevron_rows(oy, cell.center().y())
            found = ink_box(image, x0, ya, x1, yb, ground)
            crop = (x0 - 4, ya - 4, x1 + 4, yb + 4)
            if found is None:
                check(case, f"T-C1 {kind} {name} chevron drawn", False, "no ink", image=image, box=crop)
                continue
            bx0, by0, bx1, by1 = found
            cx = (bx0 + bx1 + 1) / 2.0
            cy = (by0 + by1 + 1) / 2.0
            want_x = (ox + cell.center().x()) * DPR
            want_y = (oy + F.device_snap(box, cell.center().y(), axis="y")) * DPR
            gaps = (bx0 - x0, x1 - 1 - bx1)
            check(case, f"T-C1 {kind} {name} centred",
                  abs(cx - want_x) <= 0.5 + 1e-6 and abs(cy - want_y) <= 0.5 + 1e-6
                  and abs(gaps[0] - gaps[1]) <= 1,
                  f"ink centre {cx},{cy} cell centre {want_x},{want_y} gaps {gaps}", image=image, box=crop)
            sizes[(kind, name)] = (bx1 - bx0, by1 - by0)
            centres[(kind, name)] = cx
    ox, oy = combo.x(), combo.y()
    column = F.stepper_column(combo)
    ground = image[dev(oy + combo.height() / 2.0), dev(ox + column.left()) + 1]
    found = ink_box(image, dev(ox + column.left()), *chevron_rows(oy, combo.height() / 2.0)[:1],
                    dev(ox + column.right()), chevron_rows(oy, combo.height() / 2.0)[1], ground)
    if found is not None and ("dspin", "down") in sizes:
        bx0, by0, bx1, by1 = found
        check(case, "T-C2 combo chevron = spin chevron",
              (bx1 - bx0, by1 - by0) == sizes[("dspin", "down")]
              and abs((bx0 + bx1 + 1) / 2.0 - centres[("dspin", "down")]) <= 0.5 + 1e-6,
              f"combo {(bx1 - bx0, by1 - by0)} at {(bx0 + bx1 + 1) / 2.0} "
              f"spin {sizes[('dspin', 'down')]} at {centres[('dspin', 'down')]}",
              image=image, box=(dev(ox + column.left()) - 4, dev(oy) - 2, dev(ox + combo.width()) + 2,
                                dev(oy + combo.height()) + 2))
    else:
        check(case, "T-C2 combo chevron drawn", False, "no ink")
    host.close()


# ------------------------------------------------------------------ text column
def column_case() -> None:
    case = "column"
    host = Host(240, 160)
    edit = F.FluentLineEdit("1.5", host)
    combo = F.FluentComboBox(host)
    combo.addItems(["1.5"])
    spin = F.FluentDoubleSpinBox(host)
    spin.setValue(1.5)
    height = max(w.sizeHint().height() for w in (edit, combo, spin))
    y = 16
    for widget in (edit, combo, spin):
        widget.setGeometry(16, y, 170, height)
        y += height + 8
    show(host)
    pump(0.1)
    image = grab(host)
    lefts, bottoms = [], []
    for widget in (edit, combo, spin):
        ox, oy = widget.x(), widget.y()
        ground = image[dev(oy + height / 2.0), dev(ox) + 3 * BD]
        found = ink_box(image, dev(ox + 3), dev(oy + 3), dev(ox + 60), dev(oy + height - 3),
                        ground, threshold=80)
        if found is None:
            check(case, f"T-K1 {type(widget).__name__} text drawn", False, "no ink")
            continue
        lefts.append(found[0] - dev(ox))
        bottoms.append(found[3] - dev(oy))
    crop = (dev(16) - 2, dev(16) - 2, dev(16 + 60), dev(y))
    check(case, "T-K1 text starts on one column", len(set(lefts)) == 1, f"left ink columns {lefts}",
          image=image, box=crop)
    if INTEGRAL:
        check(case, "T-K1 one baseline", len(set(bottoms)) == 1, f"bottom ink rows {bottoms}",
              image=image, box=crop)
    host.close()


def form_case() -> None:
    case = "form"
    from zlc_ui.form.form import FormFieldProps, FormSpec
    from zlc_ui.form.qt_form import FluentParameterForm

    spec = FormSpec((
        FormFieldProps("count", "int", "Count", default=3),
        FormFieldProps("level", "float", "Level", default=1.5),
        FormFieldProps("spare", "int", "Spare", default=None, allow_blank=True),
    ))
    host = Host(520, 200)
    form = FluentParameterForm(spec, {"count": 3, "level": 1.5, "spare": None}, parent=host)
    form.setGeometry(16, 16, 480, 160)
    show(host)
    pump(0.2)
    widgets = [form.widget_for(key) for key in ("count", "level", "spare")]
    caps = [w.maximumWidth() for w in widgets]
    rights = [w.mapTo(host, QtCore.QPoint(w.width(), 0)).x() for w in widgets]
    check(case, "T-K2 one cap", len(set(caps)) == 1, f"maximumWidth {caps}")
    image = grab(host)
    edges = []
    for widget in widgets:
        origin = widget.mapTo(host, QtCore.QPoint(0, 0))
        row = dev(origin.y() + widget.height() / 2.0)
        x1 = dev(origin.x() + widget.width())
        ground = image[row, x1 + 2 * BD]
        last = max((x for x in range(x1 - 4 * BD, x1 + 2 * BD) if not eq(image[row, x], ground)), default=None)
        edges.append(last)
    check(case, "T-K2 one right edge", len(set(rights)) == 1 and len(set(edges)) == 1,
          f"logical {rights} rendered {edges}", image=image, box=(0, 0, image.shape[1], image.shape[0]))
    host.close()


def main() -> None:
    probe = Host(40, 40)
    probe.show()
    pump(0.1)
    measured = float(probe.devicePixelRatioF())
    probe.close()
    if abs(measured - DPR) > 1e-6:
        print(json.dumps({"error": f"devicePixelRatioF {measured} != {DPR}"}))
        sys.exit(3)
    for kind in ("spin", "dspin", "intbox"):
        for state in ("rest", "focus", "disabled", "readonly", "hover_up", "hover_down"):
            spin_case(kind, state, 16)
        if not INTEGRAL:
            spin_case(kind, "rest", 17)
    chip_case()
    chevron_case()
    column_case()
    form_case()
    pump(0.1)
    for widget in APP.topLevelWidgets():
        widget.close()
    pump(0.1)
    print("JOINTS " + json.dumps({"dpr": DPR, "scale": ARGS["scale"], "focus": FOCUS_OK, "results": RESULTS}))


if __name__ == "__main__":
    main()
