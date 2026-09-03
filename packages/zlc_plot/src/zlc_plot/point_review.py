"""Plot-owned point picking for a caller-owned review view."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from PyQt5 import QtCore, QtWidgets

from .primitives import ImagePointOverlay, PointStatus
class _ImagePointReviewInteraction(QtCore.QObject):
    """Map clicks/rectangles to stable point ids and update the plot overlay."""

    toggle_requested = QtCore.pyqtSignal(str)
    selection_requested = QtCore.pyqtSignal(tuple)

    def __init__(
        self,
        host: object,
        overlay: ImagePointOverlay,
        surface: QtWidgets.QWidget,
    ) -> None:
        if not callable(getattr(host, "update_image_overlay", None)):
            raise TypeError("point review requires a RasterPlotHost")
        if not isinstance(overlay, ImagePointOverlay) or overlay.point_ids is None:
            raise TypeError("point review requires an overlay with stable point ids")
        if not isinstance(surface, QtWidgets.QWidget) or not hasattr(
            surface, "presented_front"
        ):
            raise TypeError("point review surface must expose its presented front")
        super().__init__(surface)
        self._host = host
        self._overlay = overlay
        self._surface = surface
        self._point_ids = tuple(overlay.point_ids)
        self._point_set = set(self._point_ids)
        self._revision = int(overlay.revision)
        self._start = None
        self._closed = False
        self._band = QtWidgets.QRubberBand(
            QtWidgets.QRubberBand.Rectangle, surface
        )
        surface.installEventFilter(self)

    def set_state(
        self,
        excluded_ids: Sequence[str],
        selected_ids: Sequence[str],
    ) -> None:
        excluded = {str(value) for value in excluded_ids}
        selected = {str(value) for value in selected_ids}
        unknown = (excluded | selected) - self._point_set
        if unknown:
            raise ValueError(f"point review state has unknown ids: {sorted(unknown)!r}")
        self._revision += 1
        statuses = tuple(
            PointStatus.OCCUPIED
            if point_id in selected
            else PointStatus.INVALID
            if point_id in excluded
            else PointStatus.UNKNOWN
            for point_id in self._point_ids
        )
        self._host.update_image_overlay(
            ImagePointOverlay(
                self._revision,
                self._overlay.coordinates,
                point_ids=self._point_ids,
                labels=self._overlay.labels,
                static_statuses=statuses,
            )
        )

    def _canonical(self, position: object) -> tuple[float, float] | None:
        front = self._surface.presented_front
        if front is None:
            return None
        axis = next(
            (item for item in front.interaction.axes if item.role == "image"),
            None,
        )
        if axis is None:
            return None
        point = axis.canonical_from_normalized(
            float(position.x()) / max(1, self._surface.width()),
            float(position.y()) / max(1, self._surface.height()),
        )
        return float(point.x), float(point.y)

    def _nearest(self, position: object) -> int | None:
        center = self._canonical(position)
        if center is None:
            return None
        offset_x = 10 if position.x() + 10 < self._surface.width() else -10
        offset_y = 10 if position.y() + 10 < self._surface.height() else -10
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
            ((self._overlay.coordinates[:, 0] - center[0]) / scale_x) ** 2
            + ((self._overlay.coordinates[:, 1] - center[1]) / scale_y) ** 2
        )
        index = int(np.argmin(distance)) if distance.size else -1
        return index if index >= 0 and float(distance[index]) <= 1.0 else None

    def eventFilter(self, watched: object, event: object) -> bool:
        if watched is not self._surface or self._closed:
            return False
        kind = event.type()
        if (
            kind == QtCore.QEvent.MouseButtonPress
            and event.button() == QtCore.Qt.LeftButton
        ):
            self._start = event.pos()
            self._band.setGeometry(QtCore.QRect(self._start, self._start))
            self._band.show()
            return True
        if kind == QtCore.QEvent.MouseMove and self._start is not None:
            self._band.setGeometry(
                QtCore.QRect(self._start, event.pos()).normalized()
            )
            return True
        if kind != QtCore.QEvent.MouseButtonRelease or self._start is None:
            return False
        start, self._start = self._start, None
        rectangle = QtCore.QRect(start, event.pos()).normalized()
        self._band.hide()
        if rectangle.width() > 6 or rectangle.height() > 6:
            first = self._canonical(rectangle.topLeft())
            second = self._canonical(rectangle.bottomRight())
            if first is not None and second is not None:
                low_x, high_x = sorted((first[0], second[0]))
                low_y, high_y = sorted((first[1], second[1]))
                selected = tuple(
                    point_id
                    for point_id, (x, y) in zip(
                        self._point_ids,
                        self._overlay.coordinates,
                        strict=True,
                    )
                    if low_x <= x <= high_x and low_y <= y <= high_y
                )
                self.selection_requested.emit(selected)
        else:
            index = self._nearest(event.pos())
            if index is None:
                self.selection_requested.emit(())
            else:
                self.toggle_requested.emit(self._point_ids[index])
        return True

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._band.hide()
        self._surface.removeEventFilter(self)


class ImagePointReviewSurface(QtWidgets.QWidget):
    """Complete plot-owned surface consumed by the neutral Fluent review view."""

    toggle_requested = QtCore.pyqtSignal(str)
    selection_requested = QtCore.pyqtSignal(tuple)

    def __init__(
        self,
        host: object,
        overlay: ImagePointOverlay,
        parent=None,
    ) -> None:
        super().__init__(parent)
        from .backends import _qt5_plot_widget_class

        Qt5PlotWidget = _qt5_plot_widget_class()
        self._plot = Qt5PlotWidget(host, self)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._plot)
        self._interaction = _ImagePointReviewInteraction(
            host, overlay, self._plot
        )
        self._interaction.toggle_requested.connect(self.toggle_requested.emit)
        self._interaction.selection_requested.connect(
            self.selection_requested.emit
        )

    def set_state(
        self,
        excluded_ids: Sequence[str],
        selected_ids: Sequence[str],
    ) -> None:
        self._interaction.set_state(excluded_ids, selected_ids)

    def close_adapter(self) -> None:
        self._interaction.close()
        self._plot.close_adapter()


__all__ = ["ImagePointReviewSurface"]
