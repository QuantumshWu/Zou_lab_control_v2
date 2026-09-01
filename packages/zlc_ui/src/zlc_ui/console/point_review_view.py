"""Fluent review of stable point identities around one caller-owned surface."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from PyQt5 import QtCore, QtWidgets

from zlc_ui.fluent import (
    ACCENT,
    GREEN,
    GREY,
    ORANGE,
    RED,
    FluentButton,
    FluentCheckBox,
    FluentDialogWindow,
    FluentFrame,
    FluentLabel,
    FluentLineEdit,
    FluentScrollArea,
    FluentStatusStrip,
    scaled_px,
    window_pad,
)


class PointReviewView(FluentFrame):
    """Pure Fluent point-review view over plain rows and one mounted QWidget."""

    state_changed = QtCore.pyqtSignal(tuple, tuple)
    accept_requested = QtCore.pyqtSignal()
    reject_requested = QtCore.pyqtSignal()

    def __init__(
        self,
        surface: QtWidgets.QWidget,
        points: Sequence[tuple[str, str, float, float]],
        *,
        message: str = "",
        confirm_label: str = "Continue",
        initial_excluded: Sequence[str] = (),
        parent=None,
    ) -> None:
        if not isinstance(surface, QtWidgets.QWidget):
            raise TypeError("point review surface must be a QWidget")
        rows = tuple(
            (str(point_id), str(label), float(x), float(y))
            for point_id, label, x, y in points
        )
        point_ids = tuple(row[0] for row in rows)
        if not point_ids or any(not value for value in point_ids):
            raise ValueError("point review requires non-empty point ids")
        if len(set(point_ids)) != len(point_ids):
            raise ValueError("point review point ids must be unique")
        excluded = {str(value) for value in initial_excluded}
        unknown = excluded - set(point_ids)
        if unknown:
            raise ValueError(
                f"initial excluded point ids are unknown: {sorted(unknown)!r}"
            )

        super().__init__(parent, bordered=False)
        self.setMinimumSize(0, 0)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Ignored,
            QtWidgets.QSizePolicy.Ignored,
        )
        self._point_ids = point_ids
        self._point_set = set(point_ids)
        self._initial_excluded = set(excluded)
        self._excluded = set(excluded)
        self._selected: set[str] = set()
        self._syncing = False
        self._checks: dict[str, FluentCheckBox] = {}

        outer = QtWidgets.QVBoxLayout(self)
        margin = window_pad(1)
        outer.setContentsMargins(margin, margin, margin, margin)
        outer.setSpacing(window_pad(0.75))

        explanation = FluentLabel(str(message), self)
        explanation.setWordWrap(True)
        explanation.setVisible(bool(str(message)))
        outer.addWidget(explanation)

        body = QtWidgets.QHBoxLayout()
        body.setSpacing(window_pad(0.75))
        plot_card = FluentFrame(self)
        plot_layout = QtWidgets.QVBoxLayout(plot_card)
        plot_layout.setContentsMargins(
            window_pad(0.5),
            window_pad(0.5),
            window_pad(0.5),
            window_pad(0.5),
        )
        surface.setSizePolicy(
            QtWidgets.QSizePolicy.Ignored,
            QtWidgets.QSizePolicy.Ignored,
        )
        surface.setMinimumSize(0, 0)
        plot_layout.addWidget(surface, 1)
        body.addWidget(plot_card, 1)

        side = FluentFrame(self)
        side.setMinimumWidth(scaled_px(300, minimum=240))
        side.setMaximumWidth(scaled_px(390, minimum=300))
        side_layout = QtWidgets.QVBoxLayout(side)
        side_layout.setContentsMargins(
            window_pad(0.75),
            window_pad(0.75),
            window_pad(0.75),
            window_pad(0.75),
        )
        side_layout.setSpacing(window_pad(0.5))
        self.search = FluentLineEdit("", side)
        self.search.setPlaceholderText("Filter sites")
        side_layout.addWidget(self.search)

        self.point_scroll = FluentScrollArea(side)
        self.point_scroll.setMinimumHeight(0)
        self.point_scroll.setSizeAdjustPolicy(
            QtWidgets.QAbstractScrollArea.AdjustIgnored
        )
        self.point_scroll.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Ignored,
        )
        point_body = QtWidgets.QWidget()
        point_body.setStyleSheet("background: transparent;")
        point_layout = QtWidgets.QVBoxLayout(point_body)
        point_layout.setContentsMargins(0, 0, 0, 0)
        point_layout.setSpacing(0)
        for point_id, label, x, y in rows:
            check = FluentCheckBox(
                f"{label}  ·  ({x:.2f}, {y:.2f})",
                point_body,
            )
            check.setToolTip(point_id)
            check.stateChanged.connect(
                lambda state, identity=point_id: self._check_changed(
                    identity, state
                )
            )
            self._checks[point_id] = check
            point_layout.addWidget(check)
        point_layout.addStretch(1)
        self.point_scroll.set_width_bounded_widget(point_body)
        side_layout.addWidget(self.point_scroll, 1)
        body.addWidget(side)
        outer.addLayout(body, 1)

        self.status = FluentStatusStrip(self)
        outer.addWidget(self.status)

        actions = QtWidgets.QHBoxLayout()
        actions.setSpacing(window_pad(0.5))
        self.exclude_selected_button = FluentButton(
            "Exclude selected", self, color=ORANGE
        )
        self.restore_selected_button = FluentButton(
            "Restore selected", self, color=ACCENT
        )
        self.reset_button = FluentButton("Reset", self, color=GREY)
        actions.addWidget(self.exclude_selected_button)
        actions.addWidget(self.restore_selected_button)
        actions.addWidget(self.reset_button)
        actions.addStretch(1)
        self.stop_button = FluentButton("Stop calibration", self, color=RED)
        self.confirm_button = FluentButton(
            str(confirm_label), self, color=GREEN
        )
        actions.addWidget(self.stop_button)
        actions.addWidget(self.confirm_button)
        outer.addLayout(actions)

        self.search.textChanged.connect(self._filter_rows)
        self.exclude_selected_button.clicked.connect(
            lambda: self._set_selected_retained(False)
        )
        self.restore_selected_button.clicked.connect(
            lambda: self._set_selected_retained(True)
        )
        self.reset_button.clicked.connect(self._reset)
        self.stop_button.clicked.connect(self.reject_requested.emit)
        self.confirm_button.clicked.connect(self.accept_requested.emit)
        self.set_state(self._excluded, ())

    @property
    def excluded_ids(self) -> tuple[str, ...]:
        return tuple(
            point_id for point_id in self._point_ids if point_id in self._excluded
        )

    @property
    def selected_ids(self) -> tuple[str, ...]:
        return tuple(
            point_id for point_id in self._point_ids if point_id in self._selected
        )

    def set_state(
        self,
        excluded_ids: Iterable[str],
        selected_ids: Iterable[str],
    ) -> None:
        excluded = {str(value) for value in excluded_ids}
        selected = {str(value) for value in selected_ids}
        unknown = (excluded | selected) - self._point_set
        if unknown:
            raise ValueError(f"point review state has unknown ids: {sorted(unknown)!r}")
        self._excluded = excluded
        self._selected = selected
        self._syncing = True
        try:
            for point_id, check in self._checks.items():
                check.setChecked(point_id not in excluded)
        finally:
            self._syncing = False
        kept = len(self._point_ids) - len(excluded)
        suffix = f" · Selected {len(selected)}" if selected else ""
        self.status.show_message(
            f"Detected {len(self._point_ids)} · Excluded {len(excluded)} "
            f"· Final {kept}{suffix}",
            severity="task" if selected else "info",
        )
        self.exclude_selected_button.setEnabled(bool(selected))
        self.restore_selected_button.setEnabled(bool(selected))
        self.confirm_button.setEnabled(kept > 0)

    def toggle_point(self, point_id: str) -> None:
        identity = str(point_id)
        if identity not in self._point_set:
            raise ValueError(f"unknown point id {identity!r}")
        excluded = set(self._excluded)
        if identity in excluded:
            excluded.remove(identity)
        else:
            excluded.add(identity)
        self._commit(excluded, ())

    def select_points(self, point_ids: Sequence[str]) -> None:
        self._commit(self._excluded, point_ids)

    def _check_changed(self, point_id: str, state: int) -> None:
        if self._syncing:
            return
        excluded = set(self._excluded)
        if state == QtCore.Qt.Checked:
            excluded.discard(point_id)
        else:
            excluded.add(point_id)
        self._commit(excluded, self._selected)

    def _set_selected_retained(self, retained: bool) -> None:
        excluded = set(self._excluded)
        for point_id in self._selected:
            if retained:
                excluded.discard(point_id)
            else:
                excluded.add(point_id)
        self._commit(excluded, ())

    def _reset(self) -> None:
        self._commit(self._initial_excluded, ())

    def _commit(
        self,
        excluded_ids: Iterable[str],
        selected_ids: Iterable[str],
    ) -> None:
        self.set_state(excluded_ids, selected_ids)
        self.state_changed.emit(self.excluded_ids, self.selected_ids)

    def _filter_rows(self, text: str) -> None:
        wanted = str(text).casefold().strip()
        for check in self._checks.values():
            searchable = f"{check.text()} {check.toolTip()}".casefold()
            check.setVisible(bool(not wanted or wanted in searchable))

    def exec_(self, parent, *, title: str) -> int:
        dialog = FluentDialogWindow(
            widget=self,
            title=str(title),
            anchor=parent,
        )
        self.accept_requested.connect(dialog.accept)
        self.reject_requested.connect(dialog.reject)
        try:
            return dialog.exec_()
        finally:
            self.accept_requested.disconnect(dialog.accept)
            self.reject_requested.disconnect(dialog.reject)


__all__ = ["PointReviewView"]
