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
        RESULTS.append({"case": case, "check": "state", "ok": False, "detail": "no keyboard focus",
                        "skip": "FOCUS-UNAVAILABLE"})
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
    # The checks below are judged on their own, at the columns the divider
    # should hold when T-S1 found none: one failure must not hide the rest.
    if not lit:
        lit = list(range(expected_first, expected_first + BD))
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
            corner_case(case, image, box, state, tint, ground, x1, y0, y1)
    host.close()


def corner_case(case, image, box, state, tint, ground, x1, y0, y1) -> None:
    """T-H3: at the rounded corner the tint meets the frame's arc with no
    ground between them (concentric: the field's radius less the edge),
    row by row, and it does not run over the arc."""

    size = int(math.ceil(F._radius() * DPR)) + 2 * BD
    rows = range(y0, y0 + size) if state == "hover_up" else range(y1 - size, y1)
    gaps, over = [], []
    for y in rows:
        line = [image[y, x] for x in range(x1 - 1, x1 - 1 - size, -1)]
        tinted = [i for i, pixel in enumerate(line) if eq(pixel, tint)]
        if not tinted:
            continue
        start = tinted[0]
        # Between the frame's ink and the tint: nothing at the ground colour.
        if any(eq(pixel, ground) for pixel in line[:start]):
            gaps.append(y)
        # The frame's ink is the darkest; the tint never replaces all of it.
        darkest = min(int(pixel.sum()) for pixel in line[:start]) if start else None
        if darkest is None or darkest >= int(tint.sum()):
            over.append(y)
    corner = (x1 - size - 4, rows.start - 4, x1 + 2, rows.stop + 4)
    check(case, "T-H3 tint concentric with the corner (no ground, no overrun)", not gaps and not over,
          f"ground rows {gaps[:6]} overrun rows {over[:6]}", image=image, box=corner)


def _chip_box(host, width: int, *, step: float = 1.0, value: float = 11.0):
    box = F.FluentDoubleSpinBox(host)
    box.setRange(0.0, 100.0)
    box.setSingleStep(step)
    box.setValue(value)
    box.setGeometry(16, 16, width, box.sizeHint().height())
    return box


def chip_case(styled: bool) -> None:
    """The step chip: smaller than the digits in any window (a window-level
    font sheet overrides setFont), its text centred on its INK inside the
    hover pill, the pill clear of the frame and the divider."""

    case = f"dspin/chip/{'styled' if styled else 'plain'}"
    host = Host(260, 80)
    if styled:
        host.setStyleSheet(F.fluent_widget_stylesheet())
    box = _chip_box(host, 180)
    show(host)
    chip = box._step_chip
    pump(0.1)
    if not check(case, "T-B chip shown", chip.isVisible(), f"text {box.text()!r} width {box.width()}"):
        host.close()
        return
    image = grab(host)
    ox, oy = box.x(), box.y()
    inner = F.frame_inner_rect(box)
    ground = image[dev(oy + box.height() / 2.0), dev(ox) + 3 * BD]
    line = box.lineEdit().geometry()
    digits = ink_box(image, dev(ox + line.x()), dev(oy + inner.top()) + 1,
                     dev(ox + chip.x()) - 1, dev(oy + inner.bottom()) - 1, ground, threshold=80)
    chip_rect = chip.geometry().translated(box.pos())
    step = ink_box(image, edge_px(chip_rect.x()), edge_px(chip_rect.y()),
                   edge_px(chip_rect.x() + chip_rect.width()), edge_px(chip_rect.y() + chip_rect.height()),
                   ground, threshold=60)
    crop = (dev(ox), dev(oy) - 2, dev(ox + box.width()) + 2, dev(oy + box.height()) + 2)
    if digits is None or step is None:
        check(case, "T-B digits and step drawn", False, f"digits {digits} step {step}", image=image, box=crop)
    else:
        digit_height = digits[3] - digits[1] + 1
        step_height = step[3] - step[1] + 1
        check(case, "T-B step smaller than the digits", step_height < digit_height,
              f"step ink {step_height} px, digits {digit_height} px (chip font "
              f"{chip.ink_font().pointSizeF():.2f} pt, box {box.font().pointSizeF():.2f} pt)",
              image=image, box=crop)
    chip.setAttribute(QtCore.Qt.WA_UnderMouse, True)
    chip.repaint()
    pump(0.05)
    image = grab(host)
    pill = chip.pill().translated(chip_rect.x(), chip_rect.y())
    cx, cy = dev(pill.center().x()), dev(pill.center().y())
    # The pill's extent, scanned from outside along its centre lines (the
    # text is inside, so the first non-ground pixel is the pill's edge).
    top = next((y for y in range(edge_px(chip_rect.y()) - 2, cy) if not eq(image[y, cx], ground)), None)
    bottom = next((y for y in range(edge_px(chip_rect.y() + chip_rect.height()) + 1, cy, -1)
                   if not eq(image[y, cx], ground)), None)
    left = next((x for x in range(edge_px(chip_rect.x()) - 2, cx) if not eq(image[cy, x], ground)), None)
    right = next((x for x in range(edge_px(chip_rect.x() + chip_rect.width()) + 1, cx, -1)
                  if not eq(image[cy, x], ground)), None)
    divider = edge_px(ox + F._spin_stepper_cells(box)[0])
    hover_crop = (dev(chip_rect.x()) - 8, dev(oy) - 2, divider + 8, dev(oy + box.height()) + 2)
    if None in (top, bottom, left, right):
        check(case, "T-B hovered pill drawn", False, f"{top} {bottom} {left} {right}", image=image, box=hover_crop)
        host.close()
        return
    tint = image[(top + bottom) // 2, left + max(1, (bottom - top) // 2) // 2 + 1]
    clear = (eq(image[top - 1, cx], ground) and eq(image[bottom + 1, cx], ground)
             and right < divider - 1 and top > dev(oy + inner.top()) and bottom < dev(oy + inner.bottom()) - 1)
    check(case, "T-B hovered chip is a pill clear of frame and divider", clear and not eq(tint, ground),
          f"pill rows {top}..{bottom} cols {left}..{right} divider {divider} tint {hexs(tint)}",
          image=image, box=hover_crop)
    text = ink_box(image, left + 1, top + 1, right, bottom, tint, threshold=90)
    if text is None:
        check(case, "T-B hovered text drawn", False, "no ink in the pill", image=image, box=hover_crop)
    else:
        pill_centre = ((left + right + 1) / 2.0, (top + bottom + 1) / 2.0)
        ink_centre = ((text[0] + text[2] + 1) / 2.0, (text[1] + text[3] + 1) / 2.0)
        # Half a device pixel: the text's origin is snapped to one.
        tolerance = 0.5
        check(case, "T-B hovered text centred on its ink",
              abs(ink_centre[0] - pill_centre[0]) <= tolerance and abs(ink_centre[1] - pill_centre[1]) <= tolerance,
              f"ink centre {ink_centre} pill centre {pill_centre} (ink {text}, pill {left},{top},{right},{bottom})",
              image=image, box=hover_crop)
    host.close()


def chip_narrow_case() -> None:
    """The chip yields to the digits: at every width it either fits beside
    the number the box would show without it, or it is not there -- it
    never rounds the value and never reaches past the frame."""

    case = "dspin/chip_narrow"
    host = Host(260, 120)
    widths = (160, 120, 90, 70, 60, 48)
    pairs = []
    for width in widths:
        box = _chip_box(host, width, step=0.001, value=1.234)
        twin = _chip_box(host, width, step=0.001, value=1.234)
        twin.move(16, 60)
        # The same box with no chip to show.
        twin._offers_step_chip = lambda: False
        twin._refresh_step_chip()
        pairs.append((width, box, twin))
    show(host)
    pump(0.1)
    bad = []
    for width, box, twin in pairs:
        twin.setValue(1.234)
        box.setValue(1.234)
        chip = box._step_chip
        inner = F.frame_inner_rect(box)
        line = box.lineEdit().geometry()
        text_end = line.x() + line.width() - box.lineEdit().textMargins().right()
        if box.text() != twin.text():
            bad.append(f"w{width}: shows {box.text()!r}, {twin.text()!r} without the chip")
        if chip.isVisible() and (chip.x() < inner.left() or chip.x() < text_end):
            bad.append(f"w{width}: chip at {chip.x()} past the frame {inner.left()} or the digits {text_end}")
        if chip.isVisible() and box.fontMetrics().horizontalAdvance(box.text()) + 2 > text_end - line.x():
            bad.append(f"w{width}: {box.text()!r} does not fit beside the chip")
        box.hide()
        twin.hide()
    check(case, "T-B chip never rounds the value nor leaves the frame", not bad, "; ".join(bad))
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


# ------------------------------------------------------------------ cards, tables, rules, rings
def corner_masks(image, x0, y0, x1, y1, size, ground):
    """The four corner squares' ink masks, mirrored onto the top-left one."""

    def mask(region):
        # The edge's own coverage: pixels darker than the ground outside.
        return (np.asarray(ground).sum() - region.sum(axis=2)) > 30

    tl = mask(image[y0:y0 + size, x0:x0 + size])
    tr = mask(image[y0:y0 + size, x1 - size:x1])[:, ::-1]
    bl = mask(image[y1 - size:y1, x0:x0 + size])[::-1, :]
    br = mask(image[y1 - size:y1, x1 - size:x1])[::-1, ::-1]
    return tl, tr, bl, br


def mirror_equal(tl, tr, bl, br) -> bool:
    """The four corners carry the same arc.  An opaque square child laid
    over a corner erases its whole inner arc (tens of pixels); the style's
    own antialiased arcs and a different fill beside the edge may differ
    by a pixel or two, which is not a joint."""

    tolerance = 2 * BD + 1
    return all(int(np.logical_xor(tl, other).sum()) <= tolerance for other in (tr, bl, br))


def panel_card_case() -> None:
    case = "panel_card"
    from zlc_ui.console.panel_card_view import PanelCardView

    host = Host(420, 340)
    card = PanelCardView("panel-1", "Camera")
    card.setParent(host)
    card.move(16, 16)
    show(host)
    pump(0.2)
    image = grab(host)
    x0, y0 = edge_px(card.x()), edge_px(card.y())
    x1, y1 = edge_px(card.x() + card.width()), edge_px(card.y() + card.height())
    ymid = dev(card.y() + card.height() / 2.0)
    edge = image[ymid, x0]
    radius = int(math.ceil(F._radius() * DPR)) + BD
    band_bottom = edge_px(card.y() + card._title_band.geometry().bottom() + 1)
    left = [y for y in range(y0 + radius, y1 - radius) if not eq(image[y, x0], edge)]
    right = [y for y in range(y0 + radius, y1 - radius) if not eq(image[y, x1 - 1], edge)]
    top = [x for x in range(x0 + radius, x1 - radius) if not eq(image[y0, x], edge)]
    crop = (x0 - 2, y0 - 2, x0 + 16 * BD, band_bottom + 4)
    check(case, "T-P1 card edge continuous past the title band", not left and not right and not top,
          f"edge {hexs(edge)} left rows {left[:6]} right rows {right[:6]} top cols {top[:6]}",
          image=image, box=crop)
    if INTEGRAL:
        ground = image[y0 - 2, x0 + radius]
        tl, tr, bl, br = corner_masks(image, x0, y0, x1, y1, radius, ground)
        check(case, "T-P1 four corners mirror-equal", mirror_equal(tl, tr, bl, br),
              f"tl {int(tl.sum())} tr {int(tr.sum())} bl {int(bl.sum())} br {int(br.sum())}",
              image=image, box=crop)
    host.close()


def table_case(fitted: bool, host_kind: str = "plain", rows: int = 4) -> None:
    case = f"table/{'fitted' if fitted else 'free'}/{host_kind}/r{rows}"
    host = Host(380, 300)
    if host_kind == "frame":
        holder = F.FluentFrame(host)
    elif host_kind == "group":
        holder = F.FluentGroupBox("Data", host)
    else:
        holder = None
    table = F.FluentTableView(holder or host)
    model = QtGui.QStandardItemModel(rows, 3, host)
    model.setHorizontalHeaderLabels(["A", "B", "C"])
    for r in range(rows):
        for c in range(3):
            model.setItem(r, c, QtGui.QStandardItem(f"{r}{c}"))
    table.setModel(model)
    if holder is not None:
        layout = QtWidgets.QVBoxLayout(holder)
        layout.setContentsMargins(12, 28 if host_kind == "group" else 12, 12, 12)
        layout.addWidget(table)
        holder.setGeometry(8, 8, 340, 264)
    else:
        table.setGeometry(16, 16, 300, 220)
    if fitted:
        table.set_content_rows_limit(10)
        table.resize(300, table.minimumHeight())
    table.setCurrentIndex(model.index(1, 1))
    table.selectionModel().select(model.index(1, 1), QtCore.QItemSelectionModel.ClearAndSelect)
    show(host)
    pump(0.2)
    image = grab(host)
    origin = table.mapTo(host, QtCore.QPoint(0, 0))
    ox, oy = origin.x(), origin.y()
    x0, y0 = edge_px(ox), edge_px(oy)
    x1, y1 = edge_px(ox + table.width()), edge_px(oy + table.height())
    viewport = table.viewport().geometry()
    ymid = dev(oy + viewport.y() + viewport.height() / 2.0)
    edge = image[ymid, x0]
    xmid = dev(ox + viewport.x() + table.columnViewportPosition(2) + 4)
    crop = (x0 - 2, y0 - 2, x1 + 2, y1 + 2)

    def run_from(values, colour):
        count = 0
        for value in values:
            if not eq(value, colour):
                break
            count += 1
        return count

    inner = F.frame_inner_rect(table)
    in_left, in_top = edge_px(ox + inner.left()), edge_px(oy + inner.top())
    inner_right = x1 - BD if INTEGRAL else x1 - BD - 1
    if INTEGRAL:
        # The frame is the style sheet's own border, whole device pixels
        # only at a whole ratio (R1).
        top_edge = run_from([image[y, xmid] for y in range(y0, y1)], edge)
        bottom_edge = run_from([image[y, xmid] for y in range(y1 - 1, y0, -1)], edge)
        check(case, "T-P2 top and bottom edges one edge-width", top_edge == BD and bottom_edge == BD,
              f"top {top_edge} bottom {bottom_edge} want {BD}", image=image, box=crop)
    # The header rules are painted (R2): exact at every ratio.
    header = table.horizontalHeader().geometry()
    side = table.verticalHeader().geometry()
    rule_y = edge_px(oy + F.device_snap(table, header.bottom() + 1, axis="y")) - BD
    rule_x = edge_px(ox + F.device_snap(table, side.right() + 1)) - BD
    rule = image[rule_y, dev(ox + header.x() + header.width() / 2.0)]
    # The rules end with the viewport, or at the frame's inner edge where a
    # fractional ratio lets the viewport run into the edge's partial pixel.
    view_right = min(edge_px(ox + viewport.x() + viewport.width()), edge_px(ox + inner.right()))
    view_bottom = min(edge_px(oy + viewport.y() + viewport.height()), edge_px(oy + inner.bottom()))
    across = [x for x in range(in_left, view_right) for y in range(rule_y, rule_y + BD) if not eq(image[y, x], rule)]
    check(case, "T-P2 header rule runs frame to viewport end through the corner", not across,
          f"rule {hexs(rule)} row {rule_y} off at cols {sorted(set(across))[:8]} ({in_left}..{view_right})",
          image=image, box=(x0 - 2, rule_y - 8 * BD, x0 + 30 * BD, rule_y + 8 * BD))
    down = [y for y in range(in_top, view_bottom) for x in range(rule_x, rule_x + BD) if not eq(image[y, x], rule)]
    check(case, "T-P2 row-header rule runs frame to viewport end through the corner", not down,
          f"col {rule_x} off at rows {sorted(set(down))[:8]} ({in_top}..{view_bottom})",
          image=image, box=(rule_x - 8 * BD, y0 - 2, rule_x + 8 * BD, y0 + 30 * BD))
    # No hook: nothing rule-coloured just above the rule's ends or beside
    # the row-header rule's ends.
    hooks = [(x, y) for x in (in_left, view_right - 1) for y in range(rule_y - 3 * BD, rule_y)
             if eq(image[y, x], rule) and not eq(image[y, x], image[rule_y - 4 * BD, dev(ox + header.x() + 4)])]
    hooks += [(x, y) for y in (view_bottom - 1,) for x in range(rule_x + BD, rule_x + 4 * BD)
              if eq(image[y, x], rule) and y < rule_y]
    check(case, "T-P2 header rule ends square", not hooks, f"hook pixels {hooks[:6]}",
          image=image, box=(view_right - 12 * BD, rule_y - 8 * BD, x1 + 2, rule_y + 8 * BD))
    if not table.verticalScrollBar().isVisible():
        frame_right = edge_px(ox + inner.right())
        check(case, "T-P2 header rule reaches the frame", view_right == frame_right,
              f"rule ends {view_right}, frame inner edge {frame_right}",
              image=image, box=(frame_right - 12 * BD, rule_y - 6 * BD, x1 + 2, rule_y + 6 * BD))
    row_index = min(2, rows - 2)
    row_y = edge_px(oy + viewport.y() + table.rowViewportPosition(row_index) + table.rowHeight(row_index)) - BD
    gaps = [x for x in range(edge_px(ox + viewport.x()), min(inner_right, view_right)) if not eq(image[row_y, x], rule)]
    check(case, "T-P2 row rule reaches the frame", not gaps, f"row {row_y} gaps {gaps[:6]}",
          image=image, box=(inner_right - 12 * BD, row_y - 6 * BD, x1 + 2, row_y + 6 * BD))
    cell = table.visualRect(model.index(1, 1))
    cx = dev(ox + viewport.x() + cell.center().x())
    cy = dev(oy + viewport.y() + cell.center().y())
    top_y = edge_px(oy + viewport.y() + cell.top()) - BD
    outline = image[top_y, cx]
    counts = {
        "top": run_from([image[y, cx] for y in range(top_y, cy)], outline),
        "bottom": run_from([image[y, cx] for y in range(edge_px(oy + viewport.y() + cell.bottom() + 1) - 1, cy, -1)], outline),
        "left": run_from([image[cy, x] for x in range(edge_px(ox + viewport.x() + cell.left()), cx)], outline),
        "right": run_from([image[cy, x] for x in range(edge_px(ox + viewport.x() + cell.right() + 1) - 1, cx, -1)], outline),
    }
    check(case, "T-P2 selection outline one edge-width all round",
          set(counts.values()) == {BD} and not eq(outline, edge),
          f"{counts} outline {hexs(outline)}", image=image,
          box=(dev(ox + viewport.x() + cell.left()) - 6, top_y - 6, dev(ox + viewport.x() + cell.right()) + 6,
               dev(oy + viewport.y() + cell.bottom()) + 6))
    if INTEGRAL and host_kind == "plain":
        ground = image[y0 - 2, x0 + 8 * BD]
        size = int(math.ceil(F._radius() * DPR)) + BD
        tl, tr, bl, br = corner_masks(image, x0, y0, x1, y1, size, ground)
        check(case, "T-P2 four corners mirror-equal", mirror_equal(tl, tr, bl, br),
              f"tl {int(tl.sum())} tr {int(tr.sum())} bl {int(bl.sum())} br {int(br.sum())}",
              image=image, box=crop)
    host.close()


def frame_rule_case(offset: int) -> None:
    """The rule is the bottom edge-width of the item, whole device pixels
    of full ink, placed from the frame -- at a fractional origin too."""

    case = f"frame_rule/o{offset}"
    host = Host(260, 130)
    first = F.FluentFrame(host, rule=True)
    first.setGeometry(offset, offset, 200, 40)
    second = F.FluentFrame(host, bordered=False, rule=False)
    second.setGeometry(offset, offset + 48, 200, 40)
    show(host)
    pump(0.1)
    image = grab(host)
    xmid = dev(offset + 100)
    # The device rows whose top lies inside the item's last logical
    # edge-width: exactly the rule.
    bottom = F.device_snap(first, first.height(), axis="y")
    want = list(range(edge_px(offset + bottom - F.frame_border_px(first)), edge_px(offset + bottom)))
    ground = image[edge_px(offset + 20), xmid]
    lit = [y for y in range(edge_px(offset + 20), edge_px(offset + 40)) if not eq(image[y, xmid], ground)]
    colours = {hexs(image[y, xmid]) for y in lit}
    check(case, "T-P3 rule is the bottom edge-width, full intensity",
          lit == want and len(want) == BD and len(colours) == 1,
          f"rule rows {lit} want {want} colours {colours}",
          image=image, box=(xmid - 8, want[0] - 8 * BD if want else 0, xmid + 8, (want[-1] if want else 0) + 4 * BD))
    # Where the second item's rule would be: its last edge-width, no rule ink.
    bottom2 = F.device_snap(second, second.height(), axis="y")
    rows2 = range(edge_px(offset + 48 + bottom2 - F.frame_border_px(second)), edge_px(offset + 48 + bottom2))
    rule = image[lit[0], xmid] if lit else None
    inked = [y for y in rows2 if rule is not None and eq(image[y, xmid], rule)]
    check(case, "T-P3 an item without a rule draws none", rule is not None and not inked,
          f"rule-coloured rows {inked}", image=image,
          box=(xmid - 8, rows2.start - 8 * BD, xmid + 8, rows2.stop + 4))
    host.close()


def focus_ring_case() -> None:
    global FOCUS_OK
    case = "focus_ring"
    host = Host(240, 120)
    holder = QtWidgets.QWidget(host)
    layout = QtWidgets.QHBoxLayout(holder)
    layout.setContentsMargins(0, 0, 0, 0)
    button = F.FluentButton("Apply")
    layout.addWidget(button)
    holder.setGeometry(24, 24, button.sizeHint().width(), button.sizeHint().height())
    show(host)
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and not button.hasFocus():
        host.activateWindow()
        APP.setActiveWindow(host)
        button.setFocus(QtCore.Qt.TabFocusReason)
        pump(0.05)
    if not button.hasFocus():
        FOCUS_OK = False
        RESULTS.append({"case": case, "check": "state", "ok": False, "detail": "no keyboard focus",
                        "skip": "FOCUS-UNAVAILABLE"})
        host.close()
        return
    # The ring is shown for a keyboard reason once the focus event is
    # delivered; poll for it the way focus itself is polled.
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        pump(0.05)
        ring = getattr(button, "_zlc_focus_ring", None)
        if ring is not None and ring.isVisible():
            break
    pump(0.05)
    ring = getattr(button, "_zlc_focus_ring", None)
    check(case, "T-P4 ring shown outside a zero-margin parent",
          ring is not None and ring.isVisible() and ring.parentWidget() is not holder,
          f"ring parent {None if ring is None else type(ring.parentWidget()).__name__}")
    if ring is None:
        host.close()
        return
    image = grab(host)
    origin = button.mapTo(host, QtCore.QPoint(0, 0))
    bx0, by0 = edge_px(origin.x()), edge_px(origin.y())
    bx1, by1 = edge_px(origin.x() + button.width()), edge_px(origin.y() + button.height())
    cx, cy = (bx0 + bx1) // 2, (by0 + by1) // 2
    ground = image[by0 - 1, cx]
    column = [image[y, cx] for y in range(by0 - 6 * BD, by0)]
    ink = min(column, key=lambda p: int(p.sum()))

    def count(values):
        return sum(1 for value in values if eq(value, ink))

    widths = {
        "top": count(image[y, cx] for y in range(by0 - 6 * BD, by0)),
        "bottom": count(image[y, cx] for y in range(by1, by1 + 6 * BD)),
        "left": count(image[cy, x] for x in range(bx0 - 6 * BD, bx0)),
        "right": count(image[cy, x] for x in range(bx1, bx1 + 6 * BD)),
    }
    stroke = max(1, round(F.FOCUS_RING_PX * DPR))
    check(case, "T-P4 ring is whole device pixels of full ink on every side",
          set(widths.values()) == {stroke} and eq(ground, image[by0 - 1, cx]),
          f"{widths} ink {hexs(ink)}", image=image, box=(bx0 - 8 * BD, by0 - 8 * BD, bx0 + 20 * BD, by1 + 8 * BD))
    host.close()


def tabs_case() -> None:
    """A closable tab is as far from its underline's ends on both sides; the
    first tab stands a card padding inside the card."""

    case = "tabs"
    host = Host(560, 140)
    tabs = F.FluentTabWidget(host)
    tabs.setGeometry(16, 16, 520, 100)
    tabs.addTab(QtWidgets.QLabel("one"), "Monitor")
    tabs.addTab(QtWidgets.QLabel("two"), "Logic")
    tabs.add_closable_tab(QtWidgets.QLabel("three"), "Edit camera")
    tabs.setCurrentIndex(0)
    show(host)
    pump(0.2)
    image = grab(host)
    bar = tabs.tabBar()
    origin = bar.mapTo(host, QtCore.QPoint(0, 0))
    first = bar.tabRect(0).translated(origin)
    row = edge_px(first.y() + first.height()) - 1
    accent = image[row, dev(first.center().x())]
    start = next((x for x in range(edge_px(tabs.x()), edge_px(first.right() + 1)) if eq(image[row, x], accent)), None)
    check(case, "T-K4 first tab inside the card padding",
          start is not None and start >= edge_px(tabs.x() + F.scaled_px(F.CARD_PAD)),
          f"underline starts at {start}, card at {edge_px(tabs.x())}")
    tabs.setCurrentIndex(2)
    pump(0.3)
    image = grab(host)
    rect = bar.tabRect(2).translated(origin)
    underline_row = edge_px(rect.y() + rect.height()) - 1
    ground = image[edge_px(rect.y()) + 2, edge_px(rect.x()) + 2]
    accent = image[underline_row, dev(rect.center().x())]
    under = [x for x in range(edge_px(rect.left()), edge_px(rect.right() + 1)) if eq(image[underline_row, x], accent)]
    region = image[edge_px(rect.y() + 4):edge_px(rect.bottom() - 4), edge_px(rect.left()):edge_px(rect.right() + 1)]
    ink = np.nonzero((np.abs(region - np.asarray(ground)).max(axis=2) > 60).any(axis=0))[0]
    if not under or not len(ink):
        check(case, "T-K4 closable tab drawn", False, "no underline or no ink")
    else:
        left_gap = edge_px(rect.left()) + int(ink.min()) - under[0]
        right_gap = under[-1] - (edge_px(rect.left()) + int(ink.max()))
        check(case, "T-K4 closable tab balanced", abs(left_gap - right_gap) <= math.ceil(DPR),
              f"underline to text {left_gap}, x to underline end {right_gap}", image=image,
              box=(edge_px(rect.left()) - 4, edge_px(rect.top()) - 4, edge_px(rect.right()) + 4,
                   edge_px(rect.bottom()) + 4))
    host.close()


def grid_case() -> None:
    """A chosen card and an empty one have one outer extent on both axes,
    every card stands on even gutters, and the outlines are whole device
    pixels."""

    case = "grid"
    host = Host(360, 200)
    grid = F.FluentSelectionGrid(host)
    frames = ("0", "1", "2", "3")
    grid.set_grid(tuple(f"{f} →" for f in frames[:-1]), tuple(f"→ {f}" for f in frames[1:]),
                  {(a, b - 1): f"{frames[a]}→{frames[b]}" for a in range(3) for b in range(a + 1, 4)})
    grid.set_selected([(0, 0)])
    grid.move(16, 16)
    grid.resize(grid.sizeHint())
    show(host)
    pump(0.2)
    image = grab(host)
    viewport = grid.viewport()
    origin = viewport.mapTo(host, QtCore.QPoint(0, 0))

    def card(row, column):
        """(left, top, right, bottom) device px of the card's ink, scanned
        from the cell's edges inward along its centre lines, and the
        cell's own device box."""

        rect = grid.visualRect(grid.model().index(row, column))
        cx, cy = dev(origin.x() + rect.center().x()), dev(origin.y() + rect.center().y())
        cell = (edge_px(origin.x() + rect.left()), edge_px(origin.y() + rect.top()),
                edge_px(origin.x() + rect.left() + rect.width()), edge_px(origin.y() + rect.top() + rect.height()))
        ground = image[cell[1], cx]
        top = next((y for y in range(cell[1], cy) if not eq(image[y, cx], ground)), None)
        bottom = next((y for y in range(cell[3] - 1, cy, -1) if not eq(image[y, cx], ground)), None)
        left = next((x for x in range(cell[0], cx) if not eq(image[cy, x], ground)), None)
        right = next((x for x in range(cell[2] - 1, cx, -1) if not eq(image[cy, x], ground)), None)
        return (left, top, right, bottom), cell

    chosen, chosen_cell = card(0, 0)
    empty, empty_cell = card(0, 1)
    crop = (chosen_cell[0] - 6, chosen_cell[1] - 6, empty_cell[2] + 6, empty_cell[3] + 6)
    sizes = [None if None in box else (box[2] - box[0], box[3] - box[1]) for box in (chosen, empty)]
    check(case, "T-P5 chosen and empty cards share one extent (both axes)",
          None not in sizes and sizes[0] == sizes[1], f"chosen {chosen} {sizes[0]} empty {empty} {sizes[1]}",
          image=image, box=crop)
    slack = 0 if INTEGRAL else 1
    gutters = []
    for box, cell in ((chosen, chosen_cell), (empty, empty_cell)):
        if None in box:
            continue
        gutters.append((box[0] - cell[0], cell[2] - 1 - box[2], box[1] - cell[1], cell[3] - 1 - box[3]))
    even = all(abs(g[0] - g[1]) <= slack and abs(g[2] - g[3]) <= slack for g in gutters) and len(gutters) == 2
    check(case, "T-P5 even gutters: left = right, top = bottom, same for every card", even
          and len({g for g in gutters}) == 1 if INTEGRAL else even,
          f"gutters (l, r, t, b) {gutters}", image=image, box=crop)
    if INTEGRAL and None not in empty:
        ex = (empty[0] + empty[2]) // 2
        outline = image[empty[1], ex]
        rows = [image[y, ex] for y in range(empty[1], empty[1] + BD + 1)]
        whole = all(eq(row, outline) for row in rows[:BD]) and not eq(rows[BD], outline)
        check(case, "T-P5 outline is whole device pixels", whole,
              f"top rows {[hexs(r) for r in rows]}", image=image, box=crop)
    host.close()


def segmented_case() -> None:
    """The current face is concentric with the track: placed one rim
    inside it, with the track's radius less the rim, so along the corner's
    diagonal the face starts where a concentric arc would."""

    case = "segmented"
    host = Host(300, 80)
    control = F.FluentSegmentedButtons(((1, "One"), (2, "Two"), (3, "Three")), host)
    control.set_current(1)
    control.move(16, 16)
    control.resize(control.sizeHint())
    show(host)
    pump(0.2)
    image = grab(host)
    first = control.button(1)
    rim = first.x()
    check(case, "T-P6 current face one rim inside the track",
          first.x() == first.y() == control.height() - first.y() - first.height() and rim > 0,
          f"face {first.geometry().getRect()} track {control.width()}x{control.height()}")
    x0, y0 = edge_px(control.x()), edge_px(control.y())
    background = image[y0 - 2, x0 + 20]
    track = image[y0 + max(1, BD // 2), x0 + 20 * BD]

    def diagonal_start(differs, start=0, limit=80):
        for step in range(start, limit):
            if differs(image[y0 + step, x0 + step]):
                return step
        return None

    def half(colour_a, colour_b):
        # Past half way from one colour to the other: where an
        # antialiased edge's coverage passes one half.
        span = np.abs(np.asarray(colour_b) - np.asarray(colour_a)).max()
        return lambda pixel: np.abs(np.asarray(pixel) - np.asarray(colour_a)).max() > span / 2.0

    outer = diagonal_start(half(background, track))
    face_edge = image[edge_px(control.y() + first.y() + first.height() / 2.0), edge_px(control.x() + rim)]
    inner = None if outer is None else diagonal_start(half(track, face_edge), outer + 1)
    radius = F._radius() * DPR
    rim_px = rim * DPR
    # Concentric arcs about one centre (radius, radius): the track's at
    # radius R, the face's at R - rim.  Along the diagonal they lie at
    # R(1 - 1/sqrt 2) and R - (R - rim)/sqrt 2 from the corner.
    want_outer = radius * (1.0 - 1.0 / math.sqrt(2.0))
    want_inner = radius - (radius - rim_px) / math.sqrt(2.0)
    ok = (outer is not None and inner is not None
          and abs(outer - want_outer) <= 1.0 and abs(inner - want_inner) <= 1.0)
    check(case, "T-P6 current face concentric with the track", ok,
          f"diagonal: track at {outer} (want {want_outer:.2f}), face at {inner} (want {want_inner:.2f})",
          image=image, box=(x0 - 4, y0 - 4, x0 + 24 * BD, y0 + 24 * BD))
    host.close()


def round_case() -> None:
    """Round things have even sizes."""

    case = "round"
    host = Host(200, 80)
    dot = F.FluentStatusDot(host, size=12)
    dot.move(16, 16)
    closer = F._FluentTabCloseButton(host)
    closer.move(60, 16)
    check(case, "T-P7 status dot even and square", dot.width() == dot.height() and dot.width() % 2 == 0,
          f"{dot.width()}x{dot.height()}")
    check(case, "T-P7 tab close disk even", closer.height() % 2 == 0, f"{closer.width()}x{closer.height()}")
    show(host)
    closer.setAttribute(QtCore.Qt.WA_UnderMouse, True)
    APP.sendEvent(closer, QtCore.QEvent(QtCore.QEvent.Enter))
    closer.style().unpolish(closer)
    closer.style().polish(closer)
    closer.repaint()
    pump(0.1)
    image = grab(host)
    size = closer.height()
    x0, y0 = edge_px(closer.x()), edge_px(closer.y())
    disk = image[y0:edge_px(closer.y() + size), x0:edge_px(closer.x() + size)]
    ground = image[y0 - 2, x0 - 2]
    mask = (np.asarray(ground).sum() - disk.sum(axis=2)) > 6
    # The x itself is symmetric too; the disk's outline must be.
    symmetric = int(np.logical_xor(mask, mask[:, ::-1]).sum()) <= 2 * BD and \
        int(np.logical_xor(mask, mask[::-1, :]).sum()) <= 2 * BD
    # The disk is the style sheet's own fill: whole pixels at a whole ratio (R1).
    check(case, "T-P7 hover disk mirror-symmetric", (symmetric or not INTEGRAL) and mask.any(),
          f"ink {int(mask.sum())}", image=image, box=(x0 - 4, y0 - 4, x0 + 3 * dev(size), y0 + dev(size) + 4))
    host.close()


def guides_case() -> None:
    """Tree guides are one edge-width of full ink; a menu separator is a
    band one edge-width tall across the item inset."""

    case = "guides"
    from zlc_ui.fluent.info_pane import InfoTree

    host = Host(360, 260)
    tree = InfoTree(name_width=120)
    tree.setParent(host)
    tree.set_rows((("alpha", {"beta": {"gamma": 1, "delta": 2}, "eps": 3}), ("zeta", 4)))
    tree.setGeometry(16, 16, 320, 220)
    tree.expandAll()
    show(host)
    pump(0.2)
    image = grab(host)
    viewport = tree.viewport()
    origin = viewport.mapTo(host, QtCore.QPoint(0, 0))
    deepest = None
    iterator = QtWidgets.QTreeWidgetItemIterator(tree)
    while iterator.value():
        item = iterator.value()
        depth, parent = 0, item.parent()
        while parent is not None:
            depth, parent = depth + 1, parent.parent()
        if depth >= 2:
            deepest = item
            break
        iterator += 1
    if deepest is None:
        check(case, "T-P8 tree has nested rows", False, "no depth-2 row")
    else:
        rect = tree.visualItemRect(deepest)
        row = dev(origin.y() + rect.center().y())
        indent = tree.indentation()
        for level in range(2):
            centre = origin.x() + level * indent + indent / 2.0
            zone = range(dev(centre - 4), dev(centre + 4))
            ground = image[row, dev(centre - 5)]
            lit = [x for x in zone if not eq(image[row, x], ground)]
            colours = {hexs(image[row, x]) for x in lit}
            check(case, f"T-P8 guide level {level} one edge-width",
                  len(lit) == BD and len(colours) == 1, f"cols {lit} colours {colours}",
                  image=image, box=(dev(centre) - 12, row - 12, dev(centre) + 12, row + 12))
    edit = F.FluentLineEdit("1.5", host)
    edit.setGeometry(16, 16, 200, edit.sizeHint().height())
    menu = F._FluentRoundedMenu(edit)
    menu.addAction("Copy")
    menu.addSeparator()
    menu.addAction("Edit step…")
    menu.popup(host.mapToGlobal(QtCore.QPoint(30, 60)))
    pump(0.4)
    shot = menu.grab().toImage().convertToFormat(QtGui.QImage.Format_RGB32)
    pointer = shot.constBits()
    pointer.setsize(shot.byteCount())
    menu_image = np.frombuffer(pointer, np.uint8).reshape(shot.height(), shot.bytesPerLine() // 4, 4)[
        :, :shot.width(), 2::-1].astype(int).copy()
    separator = next(action for action in menu.actions() if action.isSeparator())
    band = menu.actionGeometry(separator)
    cx = dev(band.center().x())
    ground = menu_image[dev(band.top()), cx]
    rows = [y for y in range(dev(band.top()), edge_px(band.bottom() + 1)) if not eq(menu_image[y, cx], ground)]
    colours = {hexs(menu_image[y, cx]) for y in rows}
    # The separator is the style sheet's own band: whole device pixels at a
    # whole ratio (R1), one colour at any.
    check(case, "T-P8 menu separator one edge-width band",
          (len(rows) == BD or not INTEGRAL) and len(colours) == 1 and bool(rows),
          f"rows {rows} colours {colours}")
    menu.hide()
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
    cases = []
    for kind in ("spin", "dspin", "intbox"):
        for state in ("rest", "focus", "disabled", "readonly", "hover_up", "hover_down"):
            cases.append((spin_case, (kind, state, 16)))
        if not INTEGRAL:
            cases.append((spin_case, (kind, "rest", 17)))
    cases += [
        (chip_case, (False,)), (chip_case, (True,)), (chip_narrow_case, ()),
        (chevron_case, ()), (column_case, ()), (form_case, ()),
        (panel_card_case, ()), (table_case, (False,)), (table_case, (True,)),
        (table_case, (False, "frame")), (table_case, (False, "group")), (table_case, (False, "frame", 40)),
        (frame_rule_case, (16,)), (frame_rule_case, (17,)), (focus_ring_case, ()), (tabs_case, ()), (grid_case, ()),
        (segmented_case, ()), (round_case, ()), (guides_case, ()),
    ]
    for case, arguments in cases:
        # A case that cannot even be measured is a failed joint, reported
        # with the others rather than hiding them behind one traceback.
        try:
            case(*arguments)
        except Exception as error:  # noqa: BLE001 - reported as a verdict
            import traceback
            RESULTS.append({"case": f"{case.__name__}{arguments}", "check": "measured", "ok": False,
                            "detail": "".join(traceback.format_exception_only(type(error), error)).strip()})
            for widget in APP.topLevelWidgets():
                widget.close()
            pump(0.05)
    pump(0.1)
    for widget in APP.topLevelWidgets():
        widget.close()
    pump(0.1)
    print("JOINTS " + json.dumps({"dpr": DPR, "scale": ARGS["scale"], "focus": FOCUS_OK, "results": RESULTS}))


if __name__ == "__main__":
    main()
