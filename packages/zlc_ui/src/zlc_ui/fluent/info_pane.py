"""Generic read-only information surface with an optional path field."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from enum import Enum
from pprint import pformat

from PyQt5 import QtCore, QtGui, QtWidgets

from .fluent import (
    FluentFrame,
    FluentPathEdit,
    FluentReadoutMultiline,
    FluentScrollArea,
    FluentSectionLabel,
    FluentSettingRow,
    FluentStatusStrip,
    FluentTabWidget,
    apply_fluent_scrollbars,
    fluent_font_size,
    scaled_px,
    setting_label_width,
    window_pad,
)
from .style import (
    ACCENT,
    CARD_PAD,
    DIVIDER,
    FONT,
    GRAPHITE,
    GREY,
    ORANGE,
    ORANGE_DARK,
    ORANGE_TINT,
    SURFACE,
    TEXT,
)


InfoRow = tuple[str, object]
InfoTab = tuple[str, tuple[InfoRow, ...]]


class InfoPane(QtWidgets.QWidget):
    """A reusable path header plus injected read-only tabs.

    The pane does not know what the rows represent.  A tab is supplied as
    ``(title, ((label, value), ...))`` and values are formatted only for
    display.  Path and status copy are also injected so the widget can be
    reused by different presenters without importing their data model.
    """

    path_committed = QtCore.pyqtSignal(str)

    def __init__(
        self,
        *,
        label_names: Iterable[str],
        tabs: tuple[InfoTab, ...] = (),
        path_label: str = "Path",
        path_caption: str = "Choose a path",
        file_filter: str = "All files (*)",
        path_base_dir: str = "",
        initial_status: str = "",
        graph_tabs: Iterable[str] = (),
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setStyleSheet("background: transparent;")

        # The declared names set the pane's one stable split.  Archive labels
        # are content, not geometry authority: loading another file must not
        # move the divider or resize the top-level window.
        self._declared_labels = tuple(str(name) for name in label_names)
        self._label_width = setting_label_width(self._declared_labels)
        self._graph_tab_titles = frozenset(str(title) for title in graph_tabs)
        self._graph_tabs: dict[str, QtWidgets.QGraphicsView] = {}
        self._fixed_pane_width: int | None = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(window_pad(1), 0, window_pad(1), 0)
        layout.setSpacing(window_pad(0.5))

        header_frame = FluentFrame(bordered=False)
        header_frame.setFixedHeight(scaled_px(48, minimum=38))
        header = QtWidgets.QHBoxLayout(header_frame)
        header.setContentsMargins(scaled_px(12), scaled_px(6), scaled_px(12), scaled_px(6))
        header.setSpacing(scaled_px(8, minimum=5))
        header.addWidget(FluentSectionLabel(str(path_label)))

        self.path_edit = FluentPathEdit(
            "",
            mode="file",
            caption=str(path_caption),
            file_filter=str(file_filter),
            base_dir=str(path_base_dir),
            refreshable=True,
        )
        self.path_edit.selected.connect(self.path_committed.emit)
        # Refresh is "open this same path again": the file was written or
        # edited elsewhere, and re-picking it in the dialog is the only thing
        # the operator could do about it.
        self.path_edit.refresh_requested.connect(
            lambda: self.path_committed.emit(self.path_edit.text())
        )
        self.path_edit.edit.editingFinished.connect(self._commit_path_draft)
        header.addWidget(self.path_edit, 1)
        layout.addWidget(header_frame)

        self.info_tabs = FluentTabWidget()
        self.info_tabs.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding,
        )
        self._tab_layouts: dict[str, QtWidgets.QVBoxLayout] = {}
        self.set_tabs(tabs)
        layout.addWidget(self.info_tabs, 1)

        self.status = FluentStatusStrip()
        if initial_status:
            self.status.show_message(str(initial_status))
        layout.addWidget(self.status)
        self._apply_pane_width()

    def _apply_pane_width(self) -> None:
        """Wide enough for the label column AND for its own tab bar.

        The pane used to be sized by the settings labels alone.  Five tabs did
        not fit in that width, so the tab bar elided its titles and grew the
        scroll arrows nobody asked for -- a navigation control that has to be
        scrolled to be seen is one an operator does not know exists.
        """

        if self._fixed_pane_width is not None:
            self.setFixedWidth(self._fixed_pane_width)
            return
        label_pane_width = self._label_width + scaled_px(280, minimum=220)
        bar = self.info_tabs.tabBar()
        # The bar caps its own tab widths when they overflow, so its laid-out
        # widths are the ones it already settled for.  Sizing to those would
        # freeze whatever it gave up; ask instead what it wanted.
        natural = getattr(bar, "natural_width", None)
        tabs_width = (
            natural()
            if callable(natural)
            else sum(bar.tabSizeHint(index).width() for index in range(bar.count()))
        )
        corner = self.info_tabs.cornerWidget(QtCore.Qt.TopRightCorner)
        corner_width = 0 if corner is None else corner.sizeHint().width()
        margins = self.layout().contentsMargins()
        needed = tabs_width + corner_width + margins.left() + margins.right()
        self._fixed_pane_width = max(label_pane_width, needed)
        self.setFixedWidth(self._fixed_pane_width)

    @QtCore.pyqtSlot()
    def _commit_path_draft(self) -> None:
        self.path_committed.emit(self.path_edit.text())

    def set_tabs(self, tabs: tuple[InfoTab, ...]) -> None:
        """Replace all tabs from plain ``(title, rows)`` values."""

        normalized = tuple(
            (str(title), tuple((str(label), value) for label, value in rows))
            for title, rows in tabs
        )
        titles = [title for title, _rows in normalized]
        if len(set(titles)) != len(titles):
            raise ValueError("info tab titles must be unique")
        # WHICH TAB THE OPERATOR IS READING is theirs, not the refresh's.
        # Every tab is destroyed and rebuilt below -- the rows change, the
        # titles almost never do -- and the rebuilt stack starts at the
        # first tab, so a refresh threw anyone reading Devices back to Plot.
        showing = self.info_tabs.tabText(self.info_tabs.currentIndex())
        while self.info_tabs.count():
            widget = self.info_tabs.widget(0)
            self.info_tabs.removeTab(0)
            if widget is not None:
                widget.deleteLater()
        self._tab_layouts.clear()
        self._graph_tabs.clear()
        for title, rows in normalized:
            if title in self._graph_tab_titles:
                self._add_graph_tab(title)
            else:
                self._add_rows_tab(title, rows)
        for index in range(self.info_tabs.count()):
            if self.info_tabs.tabText(index) == showing:
                self.info_tabs.setCurrentIndex(index)
                break
        self._apply_pane_width()

    def _add_rows_tab(self, title: str, rows: tuple[InfoRow, ...]) -> None:
        scroll = FluentScrollArea()
        scroll.setWidgetResizable(True)
        body = QtWidgets.QWidget()
        body.setStyleSheet("background: transparent;")
        tab_layout = QtWidgets.QVBoxLayout(body)
        margin = scaled_px(CARD_PAD, minimum=6)
        tab_layout.setContentsMargins(margin, margin, margin, margin)
        tab_layout.setSpacing(scaled_px(3, minimum=2))
        tab_layout.setAlignment(QtCore.Qt.AlignTop)
        self._fill_rows(tab_layout, rows)
        scroll.setWidget(body)
        self.info_tabs.add_permanent_tab(scroll, title)
        self._tab_layouts[title] = tab_layout

    def _fill_rows(self, layout: QtWidgets.QVBoxLayout, rows: tuple[InfoRow, ...]) -> None:
        for key, value in rows:
            text = self._readout_text(value)
            field = FluentReadoutMultiline(text)
            field.setSizePolicy(
                QtWidgets.QSizePolicy.Ignored,
                QtWidgets.QSizePolicy.Fixed,
            )
            layout.addWidget(
                FluentSettingRow(
                    str(key),
                    field,
                    label_width=self._label_width,
                )
            )

    def _add_graph_tab(self, title: str) -> None:
        scene = QtWidgets.QGraphicsScene()
        view = QtWidgets.QGraphicsView(scene)
        view.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        view.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
        view.setRenderHints(
            QtGui.QPainter.Antialiasing | QtGui.QPainter.TextAntialiasing
        )
        view.setFrameShape(QtWidgets.QFrame.NoFrame)
        view.setBackgroundBrush(QtGui.QColor(SURFACE))
        view.setStyleSheet(f"QGraphicsView {{ border-top: 1px solid {DIVIDER}; }}")
        apply_fluent_scrollbars(view)
        self.info_tabs.add_permanent_tab(view, title)
        self._graph_tabs[title] = view

    def set_graph(self, title: str, graph: object) -> None:
        """Replace one node/edge graph from domain-free plain data."""

        key = str(title)
        try:
            view = self._graph_tabs[key]
        except KeyError as error:
            raise KeyError(f"info pane has no graph tab {key!r}") from error
        if not isinstance(graph, Mapping) or set(graph) != {"nodes", "edges"}:
            raise TypeError("info graph must contain nodes and edges")
        raw_nodes, raw_edges = graph["nodes"], graph["edges"]
        if not isinstance(raw_nodes, tuple) or not isinstance(raw_edges, tuple):
            raise TypeError("info graph nodes and edges must be tuples")
        nodes: dict[str, Mapping[str, object]] = {}
        order: list[str] = []
        for raw in raw_nodes:
            if not isinstance(raw, Mapping) or set(raw) != {
                "id", "kind", "title", "subtitle", "root", "tooltip"
            }:
                raise ValueError("info graph node fields differ")
            node_id = str(raw["id"])
            kind = str(raw["kind"])
            if not node_id or node_id in nodes or kind not in {"logic", "device"}:
                raise ValueError("info graph node identity is invalid")
            if type(raw["root"]) is not bool:
                raise TypeError("info graph node root flag must be bool")
            nodes[node_id] = raw
            order.append(node_id)
        edges: list[Mapping[str, object]] = []
        for raw in raw_edges:
            if not isinstance(raw, Mapping) or set(raw) != {
                "source", "target", "kind", "label"
            }:
                raise ValueError("info graph edge fields differ")
            source, target, kind = (
                str(raw["source"]), str(raw["target"]), str(raw["kind"])
            )
            if source not in nodes or target not in nodes or kind not in {
                "causal", "device"
            }:
                raise ValueError("info graph edge is invalid")
            if kind == "causal" and (
                nodes[source]["kind"] != "logic"
                or nodes[target]["kind"] != "logic"
            ):
                raise ValueError("causal graph edges must connect Logic nodes")
            if kind == "device" and (
                nodes[source]["kind"] != "device"
                or nodes[target]["kind"] != "logic"
            ):
                raise ValueError("device graph edges must point into Logic nodes")
            edges.append(raw)

        scene = view.scene()
        assert scene is not None
        scene.clear()
        view._flow_node_rects = {}
        view._flow_edge_paths = ()
        view._flow_edge_count = 0
        if not nodes:
            empty = scene.addSimpleText("No saved Logic flow", QtGui.QFont(FONT, fluent_font_size()))
            empty.setBrush(QtGui.QColor(GREY))
            empty.setPos(scaled_px(18), scaled_px(18))
            scene.setSceneRect(empty.boundingRect().adjusted(-12, -12, 24, 24))
            return

        logic = [node_id for node_id in order if nodes[node_id]["kind"] == "logic"]
        causal = [edge for edge in edges if edge["kind"] == "causal"]
        predecessors = {node_id: [] for node_id in logic}
        successors = {node_id: [] for node_id in logic}
        for edge in causal:
            source, target = str(edge["source"]), str(edge["target"])
            predecessors[target].append(source)
            successors[source].append(target)
        pending = {node_id: len(predecessors[node_id]) for node_id in logic}
        ready = [node_id for node_id in logic if pending[node_id] == 0]
        topological: list[str] = []
        rank = {node_id: 1 for node_id in ready}
        while ready:
            node_id = ready.pop(0)
            topological.append(node_id)
            for target in successors[node_id]:
                rank[target] = max(rank.get(target, 1), rank[node_id] + 1)
                pending[target] -= 1
                if pending[target] == 0:
                    ready.append(target)
        if len(topological) != len(logic):
            raise ValueError("info graph contains a causal cycle")
        for node_id in order:
            if nodes[node_id]["kind"] == "device":
                rank[node_id] = 0

        card_height = scaled_px(62, minimum=50)
        horizontal_gap = scaled_px(16, minimum=12)
        vertical_gap = scaled_px(58, minimum=44)
        margin = scaled_px(18, minimum=14)
        layers: dict[int, list[str]] = {}
        for node_id in order:
            layers.setdefault(rank[node_id], []).append(node_id)
        def normalized_positions() -> dict[str, float]:
            return {
                node_id: (index + 0.5) / len(layer)
                for layer in layers.values()
                for index, node_id in enumerate(layer)
            }

        def reorder(layer_rank: int, *, from_predecessors: bool) -> None:
            layer = layers[layer_rank]
            positions = normalized_positions()
            original = {node_id: index for index, node_id in enumerate(layer)}

            def score(node_id: str) -> tuple[float, int]:
                neighbours = [
                    str(edge["source"] if from_predecessors else edge["target"])
                    for edge in edges
                    if str(edge["target"] if from_predecessors else edge["source"])
                    == node_id
                    and rank[
                        str(edge["source"] if from_predecessors else edge["target"])
                    ]
                    != layer_rank
                ]
                barycenter = (
                    sum(positions[item] for item in neighbours) / len(neighbours)
                    if neighbours
                    else positions[node_id]
                )
                return barycenter, original[node_id]

            layer.sort(key=score)

        layer_ranks = sorted(layers)
        for _sweep in range(2):
            for layer_rank in layer_ranks[1:]:
                reorder(layer_rank, from_predecessors=True)
            for layer_rank in reversed(layer_ranks[:-1]):
                reorder(layer_rank, from_predecessors=False)
        pane_margins = self.layout().contentsMargins()
        settled_view_width = (
            int(view.viewport().width())
            if self._fixed_pane_width is None
            else self._fixed_pane_width
            - pane_margins.left()
            - pane_margins.right()
            - scaled_px(8, minimum=6)
        )
        settled_view_width -= view.verticalScrollBar().sizeHint().width()
        available_width = max(
            scaled_px(220, minimum=190),
            settled_view_width - 2 * margin,
        )
        minimum_card_width = scaled_px(105, minimum=90)
        preferred_card_width = scaled_px(210, minimum=176)
        layer_geometry: dict[int, tuple[float, float]] = {}
        for layer_rank, layer in layers.items():
            card_width = max(
                minimum_card_width,
                min(
                    preferred_card_width,
                    (available_width - (len(layer) - 1) * horizontal_gap)
                    / len(layer),
                ),
            )
            layer_width = (
                len(layer) * card_width
                + max(0, len(layer) - 1) * horizontal_gap
            )
            layer_geometry[layer_rank] = card_width, layer_width
        widest = max(
            float(available_width),
            *(width for _card, width in layer_geometry.values()),
        )
        rects: dict[str, QtCore.QRectF] = {}
        next_y = float(margin)
        for layer_rank, layer in sorted(layers.items()):
            card_width, layer_width = layer_geometry[layer_rank]
            x = margin + (widest - layer_width) / 2.0
            for node_id in layer:
                rects[node_id] = QtCore.QRectF(
                    x, next_y, card_width, card_height
                )
                x += card_width + horizontal_gap
            next_y += card_height + vertical_gap

        long_edges = [
            edge
            for edge in edges
            if rank[str(edge["target"])] - rank[str(edge["source"])] > 1
        ]
        lane_base = margin + widest + scaled_px(22, minimum=16)
        lane_step = scaled_px(12, minimum=9)
        long_lanes = {id(edge): lane_base + index * lane_step for index, edge in enumerate(long_edges)}
        incoming: dict[str, list[int]] = {node_id: [] for node_id in nodes}
        outgoing: dict[str, list[int]] = {node_id: [] for node_id in nodes}
        edge_paths: list[tuple[str, str, QtGui.QPainterPath]] = []
        for edge_index, edge in enumerate(edges):
            outgoing[str(edge["source"])].append(edge_index)
            incoming[str(edge["target"])].append(edge_index)
        for _source, indices in outgoing.items():
            indices.sort(
                key=lambda index: rects[str(edges[index]["target"])].center().x()
            )
        for _target, indices in incoming.items():
            indices.sort(
                key=lambda index: rects[str(edges[index]["source"])].center().x()
            )
        for edge_index, edge in enumerate(edges):
            source, target = str(edge["source"]), str(edge["target"])
            source_rect, target_rect = rects[source], rects[target]
            source_edges = outgoing[source]
            target_edges = incoming[target]
            source_port = (source_edges.index(edge_index) + 1) / (
                len(source_edges) + 1
            )
            target_port = (target_edges.index(edge_index) + 1) / (
                len(target_edges) + 1
            )
            start = QtCore.QPointF(
                source_rect.left() + source_port * source_rect.width(),
                source_rect.bottom(),
            )
            end = QtCore.QPointF(
                target_rect.left() + target_port * target_rect.width(),
                target_rect.top(),
            )
            path = QtGui.QPainterPath(start)
            if id(edge) in long_lanes:
                lane = long_lanes[id(edge)]
                shoulder = scaled_px(16, minimum=12)
                path.lineTo(start.x(), start.y() + shoulder)
                path.lineTo(lane, start.y() + shoulder)
                path.lineTo(lane, end.y() - shoulder)
                path.lineTo(end.x(), end.y() - shoulder)
                path.lineTo(end)
            else:
                middle = (start.y() + end.y()) / 2.0
                path.cubicTo(start.x(), middle, end.x(), middle, end.x(), end.y())
            edge_paths.append((source, target, QtGui.QPainterPath(path)))
            is_device = edge["kind"] == "device"
            pen = QtGui.QPen(QtGui.QColor(ORANGE if is_device else GRAPHITE))
            pen.setWidthF(float(scaled_px(1.4, minimum=1)))
            if is_device:
                pen.setStyle(QtCore.Qt.DashLine)
            item = scene.addPath(path, pen)
            item.setZValue(-2)
            label = str(edge["label"])
            item.setToolTip(label or ("device use" if is_device else "causal input"))
            arrow = scaled_px(6, minimum=5)
            polygon = QtGui.QPolygonF(
                (
                    end,
                    QtCore.QPointF(end.x() - arrow, end.y() - arrow * 1.35),
                    QtCore.QPointF(end.x() + arrow, end.y() - arrow * 1.35),
                )
            )
            head = scene.addPolygon(polygon, pen, QtGui.QBrush(pen.color()))
            head.setZValue(-1)

        for node_id in order:
            node, rect = nodes[node_id], rects[node_id]
            is_device = node["kind"] == "device"
            outline = ORANGE if is_device else ACCENT if node["root"] else DIVIDER
            card_path = QtGui.QPainterPath()
            radius = scaled_px(5, minimum=4)
            card_path.addRoundedRect(rect, radius, radius)
            card = scene.addPath(
                card_path,
                QtGui.QPen(QtGui.QColor(outline), scaled_px(2 if node["root"] else 1)),
                QtGui.QBrush(QtGui.QColor(ORANGE_TINT if is_device else SURFACE)),
            )
            card.setToolTip(str(node["tooltip"]))
            inset = scaled_px(10, minimum=8)
            badge_font = QtGui.QFont(FONT, max(6, fluent_font_size() - 4))
            badge_font.setBold(True)
            badge = scene.addSimpleText("DEVICE" if is_device else "LOGIC", badge_font)
            badge.setAcceptedMouseButtons(QtCore.Qt.NoButton)
            badge.setBrush(QtGui.QColor(ORANGE_DARK if is_device else GREY))
            badge.setPos(rect.left() + inset, rect.top() + scaled_px(5, minimum=4))
            title_font = QtGui.QFont(FONT, max(8, fluent_font_size() - 1))
            title_font.setBold(True)
            metrics = QtGui.QFontMetrics(title_font)
            title = metrics.elidedText(
                str(node["title"]),
                QtCore.Qt.ElideMiddle,
                int(rect.width()) - 2 * inset,
            )
            title_item = scene.addSimpleText(title, title_font)
            title_item.setAcceptedMouseButtons(QtCore.Qt.NoButton)
            title_item.setBrush(QtGui.QColor(TEXT))
            title_item.setPos(rect.left() + inset, rect.top() + scaled_px(19, minimum=15))
            detail_font = QtGui.QFont(FONT, max(7, fluent_font_size() - 3))
            detail_metrics = QtGui.QFontMetrics(detail_font)
            detail = detail_metrics.elidedText(
                str(node["subtitle"]),
                QtCore.Qt.ElideRight,
                int(rect.width()) - 2 * inset,
            )
            detail_item = scene.addSimpleText(detail, detail_font)
            detail_item.setAcceptedMouseButtons(QtCore.Qt.NoButton)
            detail_item.setBrush(QtGui.QColor(GREY))
            detail_item.setPos(rect.left() + inset, rect.top() + scaled_px(40, minimum=32))

        scene_width = widest + 2 * margin
        if long_lanes:
            scene_width = max(scene_width, max(long_lanes.values()) + margin)
        scene_height = max(rect.bottom() for rect in rects.values()) + margin
        scene.setSceneRect(0, 0, scene_width, scene_height)
        view._flow_node_rects = dict(rects)
        view._flow_edge_paths = tuple(edge_paths)
        view._flow_edge_count = len(edges)

    def set_status(self, text: str) -> None:
        self.status.show_message(str(text))

    @staticmethod
    def _readout_text(value: object) -> str:
        if isinstance(value, Enum):
            value = value.value
        if isinstance(value, Mapping):
            return pformat(dict(value), sort_dicts=False, width=80)
        if isinstance(value, (tuple, list)):
            return ", ".join(
                str(item.value if isinstance(item, Enum) else item)
                for item in value
            )
        return str(value)


__all__ = ["InfoPane", "InfoRow", "InfoTab"]
