"""Qt review of stable point identities over one existing image plot."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .primitives import ImagePointOverlay, PointStatus
from .raster import RasterPlotHost


def review_image_points(
    host: RasterPlotHost,
    overlay: ImagePointOverlay,
    parent: object = None,
    *,
    title: str = "Review points",
    message: str = "",
    confirm_label: str = "Continue",
    initial_excluded: Sequence[str] = (),
) -> tuple[str, ...] | None:
    """Return excluded point ids, or ``None`` when the operator cancels."""

    if not isinstance(host, RasterPlotHost):
        raise TypeError("point review requires a RasterPlotHost")
    if not isinstance(overlay, ImagePointOverlay) or overlay.point_ids is None:
        raise TypeError("point review requires an overlay with stable point ids")
    point_ids = tuple(overlay.point_ids)
    labels = tuple(
        point_id if label is None else str(label)
        for point_id, label in zip(
            point_ids,
            overlay.labels or (None,) * len(point_ids),
            strict=True,
        )
    )
    excluded = {str(value) for value in initial_excluded}
    unknown = excluded - set(point_ids)
    if unknown:
        raise ValueError(f"initial excluded point ids are unknown: {sorted(unknown)!r}")

    from PyQt5 import QtCore, QtWidgets
    from .backends import _qt5_plot_widget_class

    Qt5PlotWidget = _qt5_plot_widget_class()
    dialog = QtWidgets.QDialog(parent)
    dialog.setWindowTitle(str(title))
    dialog.setModal(True)
    root = QtWidgets.QVBoxLayout(dialog)
    if message:
        explanation = QtWidgets.QLabel(str(message), dialog)
        explanation.setWordWrap(True)
        root.addWidget(explanation)

    body = QtWidgets.QHBoxLayout()
    plot = Qt5PlotWidget(host, dialog)
    body.addWidget(plot, 1)
    side = QtWidgets.QVBoxLayout()
    search = QtWidgets.QLineEdit(dialog)
    search.setPlaceholderText("Filter sites")
    side.addWidget(search)
    points = QtWidgets.QListWidget(dialog)
    points.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
    points.setMinimumWidth(230)
    side.addWidget(points, 1)
    body.addLayout(side)
    root.addLayout(body, 1)

    for index, (point_id, label, coordinate) in enumerate(
        zip(point_ids, labels, overlay.coordinates, strict=True)
    ):
        item = QtWidgets.QListWidgetItem(
            f"{label}  ·  ({float(coordinate[0]):.2f}, {float(coordinate[1]):.2f})"
        )
        item.setData(QtCore.Qt.UserRole, index)
        item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
        item.setCheckState(
            QtCore.Qt.Unchecked if point_id in excluded else QtCore.Qt.Checked
        )
        points.addItem(item)

    status = QtWidgets.QLabel(dialog)
    root.addWidget(status)
    actions = QtWidgets.QHBoxLayout()
    exclude_selected = QtWidgets.QPushButton("Exclude selected", dialog)
    restore_selected = QtWidgets.QPushButton("Restore selected", dialog)
    reset = QtWidgets.QPushButton("Reset", dialog)
    actions.addWidget(exclude_selected)
    actions.addWidget(restore_selected)
    actions.addWidget(reset)
    actions.addStretch(1)
    cancel = QtWidgets.QPushButton("Stop calibration", dialog)
    confirm = QtWidgets.QPushButton(str(confirm_label), dialog)
    confirm.setDefault(True)
    actions.addWidget(cancel)
    actions.addWidget(confirm)
    root.addLayout(actions)

    syncing = False
    selected_indices: set[int] = set()
    overlay_revision = int(overlay.revision)

    def refresh() -> None:
        nonlocal overlay_revision, syncing
        syncing = True
        try:
            for index in range(points.count()):
                item = points.item(index)
                point_id = point_ids[index]
                item.setCheckState(
                    QtCore.Qt.Unchecked
                    if point_id in excluded
                    else QtCore.Qt.Checked
                )
                item.setSelected(index in selected_indices)
        finally:
            syncing = False
        overlay_revision += 1
        statuses = tuple(
            PointStatus.OCCUPIED
            if index in selected_indices
            else PointStatus.INVALID
            if point_id in excluded
            else PointStatus.UNKNOWN
            for index, point_id in enumerate(point_ids)
        )
        host.update_image_overlay(
            ImagePointOverlay(
                overlay_revision,
                overlay.coordinates,
                point_ids=point_ids,
                labels=overlay.labels,
                static_statuses=statuses,
            )
        )
        kept = len(point_ids) - len(excluded)
        status.setText(
            f"Detected {len(point_ids)}  ·  Excluded {len(excluded)}  ·  Final {kept}"
            + (f"  ·  Selected {len(selected_indices)}" if selected_indices else "")
        )
        confirm.setEnabled(kept > 0)

    def item_changed(item: QtWidgets.QListWidgetItem) -> None:
        if syncing:
            return
        index = int(item.data(QtCore.Qt.UserRole))
        point_id = point_ids[index]
        if item.checkState() == QtCore.Qt.Checked:
            excluded.discard(point_id)
        else:
            excluded.add(point_id)
        refresh()

    def list_selection_changed() -> None:
        if syncing:
            return
        selected_indices.clear()
        selected_indices.update(
            int(item.data(QtCore.Qt.UserRole)) for item in points.selectedItems()
        )
        refresh()

    def set_selected(keep: bool) -> None:
        for index in selected_indices:
            if keep:
                excluded.discard(point_ids[index])
            else:
                excluded.add(point_ids[index])
        selected_indices.clear()
        refresh()

    def reset_all() -> None:
        excluded.clear()
        selected_indices.clear()
        refresh()

    def filter_items(text: str) -> None:
        wanted = str(text).casefold().strip()
        for index in range(points.count()):
            item = points.item(index)
            item.setHidden(bool(wanted and wanted not in item.text().casefold()))

    class _PointGesture(QtCore.QObject):
        def __init__(self) -> None:
            super().__init__(plot)
            self.start = None
            self.band = QtWidgets.QRubberBand(QtWidgets.QRubberBand.Rectangle, plot)

        def _canonical(self, position: object) -> tuple[float, float] | None:
            front = plot.presented_front
            if front is None:
                return None
            axis = next(
                (item for item in front.interaction.axes if item.role == "image"),
                None,
            )
            if axis is None:
                return None
            point = axis.canonical_from_normalized(
                float(position.x()) / max(1, plot.width()),
                float(position.y()) / max(1, plot.height()),
            )
            return float(point.x), float(point.y)

        def _nearest(self, position: object) -> int | None:
            center = self._canonical(position)
            if center is None:
                return None
            offset_x = 10 if position.x() + 10 < plot.width() else -10
            offset_y = 10 if position.y() + 10 < plot.height() else -10
            dx = self._canonical(
                QtCore.QPoint(position.x() + offset_x, position.y())
            )
            dy = self._canonical(
                QtCore.QPoint(position.x(), position.y() + offset_y)
            )
            if dx is None or dy is None:
                return None
            scale_x = max(abs(dx[0] - center[0]), np.finfo(float).eps)
            scale_y = max(abs(dy[1] - center[1]), np.finfo(float).eps)
            distance = (
                ((overlay.coordinates[:, 0] - center[0]) / scale_x) ** 2
                + ((overlay.coordinates[:, 1] - center[1]) / scale_y) ** 2
            )
            index = int(np.argmin(distance)) if distance.size else -1
            return index if index >= 0 and float(distance[index]) <= 1.0 else None

        def eventFilter(self, watched: object, event: object) -> bool:
            if watched is not plot:
                return False
            kind = event.type()
            if kind == QtCore.QEvent.MouseButtonPress and event.button() == QtCore.Qt.LeftButton:
                self.start = event.pos()
                self.band.setGeometry(QtCore.QRect(self.start, self.start))
                self.band.show()
                return True
            if kind == QtCore.QEvent.MouseMove and self.start is not None:
                self.band.setGeometry(QtCore.QRect(self.start, event.pos()).normalized())
                return True
            if kind == QtCore.QEvent.MouseButtonRelease and self.start is not None:
                start, self.start = self.start, None
                rectangle = QtCore.QRect(start, event.pos()).normalized()
                self.band.hide()
                if rectangle.width() > 6 or rectangle.height() > 6:
                    first = self._canonical(rectangle.topLeft())
                    second = self._canonical(rectangle.bottomRight())
                    if first is not None and second is not None:
                        low_x, high_x = sorted((first[0], second[0]))
                        low_y, high_y = sorted((first[1], second[1]))
                        selected_indices.clear()
                        selected_indices.update(
                            index
                            for index, (x, y) in enumerate(overlay.coordinates)
                            if low_x <= x <= high_x and low_y <= y <= high_y
                        )
                        refresh()
                else:
                    index = self._nearest(event.pos())
                    if index is not None:
                        point_id = point_ids[index]
                        if point_id in excluded:
                            excluded.remove(point_id)
                        else:
                            excluded.add(point_id)
                        selected_indices.clear()
                        refresh()
                return True
            return False

    gesture = _PointGesture()
    plot.installEventFilter(gesture)
    points.itemChanged.connect(item_changed)
    points.itemSelectionChanged.connect(list_selection_changed)
    search.textChanged.connect(filter_items)
    exclude_selected.clicked.connect(lambda: set_selected(False))
    restore_selected.clicked.connect(lambda: set_selected(True))
    reset.clicked.connect(reset_all)
    cancel.clicked.connect(dialog.reject)
    confirm.clicked.connect(dialog.accept)
    refresh()
    dialog.resize(1040, 720)
    try:
        accepted = dialog.exec_() == QtWidgets.QDialog.Accepted
        return tuple(point_id for point_id in point_ids if point_id in excluded) if accepted else None
    finally:
        plot.removeEventFilter(gesture)
        plot.close_adapter()


__all__ = ["review_image_points"]
