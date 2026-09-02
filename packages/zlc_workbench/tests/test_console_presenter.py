"""The console presenter wires views to a session, and decides nothing itself.

Headless on purpose.  The presenter never imports Qt: it receives already-built
views and talks to them through their declared setters and signals, which is
what lets the same code path serve a notebook -- the session below it does not
know a window exists.

The views here are stand-ins with the real signatures.  Substituting the widget
layer rather than the presenter is the point: what is under test is the wiring.
"""

from __future__ import annotations

from concurrent.futures import Future
import os
from threading import Event
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from zlc_ui import STATUS_SEVERITIES
from zlc_plot import SelectorKind

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

from zlc_atom.nodes.camera_measurement.measurement import (
    CameraMeasurementNode,
    CameraMeasurementRequest,
)
from zlc_workbench.console import ConsolePresenter
from zlc_workbench.console_layout import LayoutDocument
from zlc_workbench.panel_catalog import task_console_fitting_spec
from zlc_workbench.session import ExperimentSession, Workspace
from zlc_workbench.topology import format_signal_shape
from pulse_fixtures import CAMERA_WINDOWS, PULSE_NAME, write_ordinary_pulse


def _completed_surface_update() -> object:
    future = Future()
    future.set_result(None)
    return SimpleNamespace(future=future)


def _operation_value(operation):
    resolved = operation.result() if hasattr(operation, "result") else operation
    return getattr(resolved, "value", resolved)


def _accepted(port, field):
    surface = port.accepted_surface()
    return None if surface is None else getattr(surface, field)


class _Signal:
    """A Qt signal's shape, without Qt."""

    def __init__(self) -> None:
        self._listeners: list = []

    def connect(self, listener) -> None:
        self._listeners.append(listener)

    def emit(self, *args) -> None:
        for listener in list(self._listeners):
            listener(*args)


class _CardView:
    """A panel card's whole contract, with Qt taken out."""

    def __init__(self, panel_id: str, title: str = "Panel") -> None:
        self.panel_id = str(panel_id)
        self.title = title
        self.surface = None
        self.remove_requested = _Signal()
        self.edit_requested = _Signal()
        self.dropped = _Signal()
        self.drag_started = _Signal()
        self.drag_moved = _Signal()
        self.choices: tuple = ()
        self.chosen = ""
        self.size = ""
        self.selectors_enabled = True
        self.status: tuple = ("", False)

    def set_signal_choices(
        self,
        groups,
        *,
        current: str = "",
        overlay_groups=(),
        overlay_current: str = "",
    ) -> None:
        self.choices = tuple(groups)
        self.chosen = str(current or self.chosen)
        self.overlay_choices = tuple(overlay_groups)
        self.overlay_chosen = str(overlay_current)

    def set_selectors_enabled(self, enabled: bool) -> None:
        self.selectors_enabled = bool(enabled)

    def set_status(self, text: str, *, error: bool) -> None:
        self.status = (str(text), bool(error))

    def set_surface(self, widget) -> None:
        self.surface = widget

    def setParent(self, _parent) -> None:
        self.surface = None


class _LogicRowView:
    """A logic row's whole contract, with Qt taken out."""

    def __init__(self, title: str = "Logic", kind: str = "logic") -> None:
        self.title = str(title)
        self.kind = str(kind)
        self.start_requested = _Signal()
        self.auto_preview_changed = _Signal()
        self.stop_requested = _Signal()
        self.edit_requested = _Signal()
        self.remove_requested = _Signal()
        self.state = ("idle", "")
        self.commands = (False, False)
        self.publishes: tuple = ()
        self.auto_preview = True
        self.preview_offered = True

    def set_preview_offered(self, offered: bool) -> None:
        self.preview_offered = bool(offered)

    def set_auto_preview(self, enabled: bool) -> None:
        self.auto_preview = bool(enabled)

    def set_state(self, state: str, status_text: str = "") -> None:
        self.state = (str(state), str(status_text))

    def set_publishes(self, rows) -> None:
        self.publishes = tuple(rows)

    def set_commands(self, *, can_start: bool, can_stop: bool) -> None:
        self.commands = (bool(can_start), bool(can_stop))


class _ConsoleView:
    """The console HANDLE's whole contract, with Qt taken out.

    Every signal the real handle emits is here, because a presenter that only
    answers the signals its double happens to declare will silently stop
    answering the ones the real window offers.  The cards and rows are made
    HERE, as the window makes them: a double that took them from outside would
    let the presenter go on building widgets that the real seam forbids.
    """

    _SIGNALS = (
        "close_requested", "add_panel_requested", "add_logic_requested",
        "pause_toggled", "selectors_toggled", "save_layout_requested",
        "load_layout_requested", "save_screenshot_requested",
        "stop_task_requested",
        "panel_order_committed",
        "panel_remove_requested",
        "panel_edit_requested", "panel_plot_error",
        "panel_publisher_edit_requested", "panel_publisher_draft_changed",
        "logic_start_requested", "logic_auto_preview_changed",
        "logic_stop_requested",
        "logic_edit_requested", "logic_remove_requested", "logic_draft_changed",
        "panel_state_changed", "panel_snapshot_refresh_requested",
        "panel_editor_closed",
    )

    def __init__(self) -> None:
        for name in self._SIGNALS:
            setattr(self, name, _Signal())
        self._cards: dict[str, _CardView] = {}
        self._rows: dict[str, _LogicRowView] = {}
        self.summary = ""
        self.paused = False
        self.selectors = False
        self.status: list[tuple[str, str]] = []
        self.kinds: tuple = ()
        self.logic_kinds: tuple = ()
        #: What a chooser would answer, and what it was offered.
        self.chooser_answer: str | None = None
        #: What a file dialog would answer; "" is the operator cancelling.
        self.open_answer = ""
        self.save_answer = ""
        self.screenshot_path = ""
        self.offered: tuple = ()
        self.logic_editors: dict[str, dict] = {}
        self.focused_logic_editor = ""
        self.panel_states: dict[str, object] = {}
        self.panel_state_updates: list[tuple[str, object]] = []
        self.panel_parameter_surfaces: dict[str, object] = {}
        self.panel_intervals: tuple[int, ...] = ()
        self.panel_sizes: tuple[str, ...] = ()
        self.panel_editors: dict[str, dict] = {}
        self.panel_editor_update_count: dict[str, int] = {}
        self.panel_editor_surfaces: dict[str, object] = {}
        self.focused_panel_editor = ""
        self.task_takeover = False
        self.panel_mutation_enabled: dict[str, bool] = {}
        self.panel_publishers: dict[str, tuple] = {}
        self.panel_publisher_editors: dict[str, dict] = {}
        #: Every front the presenter put on a staged panel widget, in order.
        self.presented_fronts: list[tuple[str, object]] = []

    # -- what a test reads ------------------------------------------------

    @property
    def cards(self) -> tuple:
        return tuple(self._cards.values())

    @property
    def logic_rows(self) -> tuple:
        return tuple(self._rows.values())

    # -- the board --------------------------------------------------------

    def set_panel_kinds(self, kinds, current: str = "") -> None:
        self.kinds = tuple(kinds)

    def set_grid_cell_kinds(self, kinds) -> None:
        self.grid_cell_kinds = tuple(str(value) for value in kinds)

    def set_panel_intervals(self, intervals, default_interval) -> None:
        self.panel_intervals = tuple(int(value) for value in intervals)
        self.panel_default_interval = int(default_interval)

    def set_panel_sizes(self, sizes, default_size) -> None:
        self.panel_sizes = tuple(str(value) for value in sizes)
        self.panel_default_size = str(default_size)

    def set_logic_kinds(self, kinds) -> None:
        self.logic_kinds = tuple(kinds)

    def set_summary(self, text: str) -> None:
        self.summary = text

    def set_paused(self, paused: bool) -> None:
        self.paused = bool(paused)

    def set_selectors(self, enabled: bool) -> None:
        self.selectors = bool(enabled)

    def show_status(self, text: str, severity: str) -> None:
        # The double answers for the REAL strip, vocabulary included.  It
        # used to take any word at all, so a severity the shipped view
        # rejects sailed through every test here and only failed in the
        # operator's console -- where the rejection came out of a Qt slot
        # and killed the whole application.
        if str(severity) not in STATUS_SEVERITIES:
            raise ValueError(
                f"severity {severity!r} is not one of {STATUS_SEVERITIES}"
            )
        self.status.append((str(severity), str(text)))

    def set_task_takeover(self, active: bool) -> None:
        self.task_takeover = bool(active)

    def choose_signal(self, rows) -> str | None:
        self.offered = tuple(rows)
        return self.chooser_answer

    def show_warning(self, title: str, text: str) -> None:
        self.status.append(("error", str(text)))

    def ask_open_path(self, caption: str, start: str, filter: str) -> str:
        return self.open_answer

    def ask_save_path(self, caption: str, suggested: str, filter: str) -> str:
        return self.save_answer

    def save_screenshot(self, path: str) -> str:
        Path(path).write_bytes(b"plain TaskConsole screenshot")
        self.screenshot_path = str(path)
        return str(path)

    def review_points(
        self,
        surface,
        points,
        *,
        title: str,
        message: str = "",
        confirm_label: str = "Continue",
        initial_excluded=(),
    ):
        del surface, points, title, message, confirm_label
        return tuple(initial_excluded)

    # -- panels -----------------------------------------------------------

    def add_panel(self, panel_id: str, title: str) -> None:
        key = str(panel_id)
        if key not in self._cards:
            card = _CardView(key, str(title))
            card.remove_requested.connect(
                lambda _=None, pid=key: self.panel_remove_requested.emit(pid)
            )
            card.edit_requested.connect(
                lambda _=None, pid=key: self.panel_edit_requested.emit(pid)
            )
            self._cards[key] = card

    def remove_panel(self, panel_id: str) -> None:
        self._cards.pop(str(panel_id), None)

    def panel_ids(self) -> tuple:
        return tuple(self._cards)

    def set_panel_order(self, order) -> None:
        wanted = [str(pid) for pid in order if str(pid) in self._cards]
        wanted += [key for key in self._cards if key not in wanted]
        self._cards = {key: self._cards[key] for key in wanted}

    def show_panel(self, panel_id: str, host) -> None:
        self._cards[str(panel_id)].set_surface(host)

    def present_panel_front(self, panel_id: str, front) -> bool:
        self.presented_fronts.append((str(panel_id), front))
        return True

    def set_panel_signal_choices(self, panel_id: str, *args, **kwargs) -> None:
        self._cards[str(panel_id)].set_signal_choices(*args, **kwargs)

    def set_panel_projection(
        self,
        panel_id: str,
        state: object,
        parameter_surface: object,
    ) -> None:
        key = str(panel_id)
        self.panel_states[key] = state
        self.panel_state_updates.append((key, state))
        self.panel_parameter_surfaces[key] = parameter_surface
        self._cards[key].title = state.title
        self._cards[key].size = state.size

    def set_panel_status(self, panel_id: str, text: str, *, error: bool) -> None:
        self._cards[str(panel_id)].set_status(text, error=error)

    def set_panel_mutation_enabled(self, panel_id: str, enabled: bool) -> None:
        self.panel_mutation_enabled[str(panel_id)] = bool(enabled)

    def set_panel_selectors_enabled(self, panel_id: str, enabled: bool) -> None:
        self._cards[str(panel_id)].set_selectors_enabled(enabled)

    def set_panel_publishers(self, publishers) -> None:
        self.panel_publishers = {
            str(panel_id): tuple(rows)
            for panel_id, rows in publishers
        }

    def open_panel_publisher_editor(self, panel_id: str, projection) -> None:
        self.panel_publisher_editors[str(panel_id)] = dict(projection)

    def update_panel_publisher_editor(self, panel_id: str, projection) -> bool:
        key = str(panel_id)
        if key not in self.panel_publisher_editors:
            return False
        self.panel_publisher_editors[key] = dict(projection)
        return True

    def has_panel_publisher_editor(self, panel_id: str) -> bool:
        return str(panel_id) in self.panel_publisher_editors

    def focus_panel_publisher_editor(self, panel_id: str) -> bool:
        return str(panel_id) in self.panel_publisher_editors

    def close_panel_publisher_editor(self, panel_id: str) -> bool:
        return self.panel_publisher_editors.pop(str(panel_id), None) is not None

    def open_panel_editor(self, panel_id: str, projection) -> None:
        self.panel_editors[str(panel_id)] = dict(projection)

    def update_panel_editor(self, panel_id: str, projection) -> None:
        key = str(panel_id)
        if key in self.panel_editors:
            self.panel_editors[key] = dict(projection)
            self.panel_editor_update_count[key] = (
                self.panel_editor_update_count.get(key, 0) + 1
            )

    def show_panel_editor(self, panel_id: str, host) -> None:
        key = str(panel_id)
        if host is None:
            self.panel_editor_surfaces.pop(key, None)
        else:
            self.panel_editor_surfaces[key] = host

    def focus_panel_editor(self, panel_id: str) -> None:
        self.focused_panel_editor = str(panel_id)

    def close_panel_editor(self, panel_id: str) -> None:
        self.panel_editors.pop(str(panel_id), None)
        self.panel_editor_surfaces.pop(str(panel_id), None)
        if self.focused_panel_editor == str(panel_id):
            self.focused_panel_editor = ""

    # -- logic rows -------------------------------------------------------

    def add_logic_row(
        self, node_id: str, kind: str, offers_preview: bool = True
    ) -> None:
        key = str(node_id)
        if key not in self._rows:
            row = _LogicRowView(key, str(kind))
            row.set_preview_offered(offers_preview)
            row.start_requested.connect(
                lambda _=None, nid=key: self.logic_start_requested.emit(nid)
            )
            row.auto_preview_changed.connect(
                lambda enabled, nid=key: (
                    self.logic_auto_preview_changed.emit(nid, bool(enabled))
                )
            )
            row.stop_requested.connect(
                lambda _=None, nid=key: self.logic_stop_requested.emit(nid)
            )
            row.edit_requested.connect(
                lambda _=None, nid=key: self.logic_edit_requested.emit(nid)
            )
            row.remove_requested.connect(
                lambda _=None, nid=key: self.logic_remove_requested.emit(nid)
            )
            self._rows[key] = row

    def remove_logic_row(self, node_id: str) -> None:
        self._rows.pop(str(node_id), None)

    def logic_row_ids(self) -> tuple:
        return tuple(self._rows)

    def set_logic_state(self, node_id: str, state: str, status_text: str = "") -> None:
        self._rows[str(node_id)].set_state(state, status_text)

    def set_logic_commands(
        self,
        node_id: str,
        *,
        can_start: bool,
        can_stop: bool,
    ) -> None:
        self._rows[str(node_id)].set_commands(
            can_start=can_start,
            can_stop=can_stop,
        )

    def set_logic_auto_preview(self, node_id: str, enabled: bool) -> None:
        self._rows[str(node_id)].set_auto_preview(bool(enabled))

    def set_logic_publishes(self, node_id: str, rows) -> None:
        self._rows[str(node_id)].set_publishes(rows)

    def open_logic_editor(self, node_id: str, projection) -> None:
        self.logic_editors[str(node_id)] = dict(projection)

    def update_logic_editor(self, node_id: str, projection) -> None:
        if str(node_id) in self.logic_editors:
            self.logic_editors[str(node_id)] = dict(projection)

    def has_logic_editor(self, node_id: str) -> bool:
        return str(node_id) in self.logic_editors

    def focus_logic_editor(self, node_id: str) -> None:
        self.focused_logic_editor = str(node_id)

    def close_logic_editor(self, node_id: str) -> None:
        self.logic_editors.pop(str(node_id), None)
        if self.focused_logic_editor == str(node_id):
            self.focused_logic_editor = ""


@pytest.fixture
def session(tmp_path):
    write_ordinary_pulse(tmp_path)
    session = ExperimentSession.open(tmp_path, template="virtual")
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def presenter(session):
    pytest.importorskip("zlc_plot")
    # The REAL mount path, not a copy of it.  The copy this fixture used to
    # carry drifted from the app exactly once -- and that once was a shipped
    # bug the suite could not see.
    from zlc_workbench.apps.task_console import build_panel_host

    def spec_for(snapshot, kind="", cell_kind=""):
        return task_console_fitting_spec(snapshot.block.schema, kind, cell_kind)

    presenter = ConsolePresenter(
        session,
        _ConsoleView(),
        make_host=build_panel_host,
        spec_for=spec_for,
    )
    try:
        yield presenter
    finally:
        deadline = time.monotonic() + 10.0
        while not presenter.close() and time.monotonic() < deadline:
            presenter.beat()
            time.sleep(0.005)
        assert presenter.close(), "Console test owner did not retire"


def _one_shot(session, producer: str = "cm"):
    session.load_pulse(PULSE_NAME)
    node = CameraMeasurementNode(
        camera=session.camera,
        request=CameraMeasurementRequest("camera", 0.02, None, 1, CAMERA_WINDOWS),
        signal_plane=session.signal_plane,
        producer=producer,
    )
    capture = node.prepare()
    session.fire(shots=1)
    result = capture.collect()
    session.nodes = [node]
    return node, result.publication.value(node.signal_key("frames")).snapshot


def _settle_panel_hosts(presenter, predicate=lambda: True) -> None:
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        presenter.beat()
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("panel hosts did not settle")


def test_camera_restart_drains_the_old_generation_before_replacement(
    presenter, session, monkeypatch
) -> None:
    """Restart must not retire a publication still travelling to its Panel."""

    session.load_pulse(PULSE_NAME)
    node_id = presenter.add_logic(
        "camera_measurement",
        values={
            "exposure_seconds": 0.013,
            "repeat": 0,
            "frames_per_cycle": CAMERA_WINDOWS,
        },
    )
    assert presenter.start_logic(node_id) is True
    old_host = presenter.logic[node_id].host
    assert old_host is not None and old_host.running
    signal = old_host.signal_key("frames")

    _settle_panel_hosts(presenter, lambda: bool(session.camera.capture_state()))
    session.fire(shots=1)
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        presenter.beat()
        if (
            len(presenter.panels) == 1
            and (panel := next(iter(presenter.panels.values()))).port is not None
            and _accepted(panel.port, "publication") is not None
        ):
            break
        time.sleep(0.005)
    assert len(presenter.panels) == 1, (
        presenter.logic[node_id].host.observation,
        presenter.view.status,
        session.signal_plane.latest_publication(signal),
    )
    panel = next(iter(presenter.panels.values()))
    shown = _accepted(panel.port, "publication")
    assert shown is not None
    crosshair = panel.host.set_crosshair_selector(2.0, 3.0).result().value
    crosshair_description = panel.host.describe_display().result()
    assert crosshair in crosshair_description.value.selectors
    assert panel.port.accept_configuration(
        crosshair_description,
        panel.state,
    ) is not None
    configured_selectors: list[tuple[object, ...]] = []
    make_plot_host = presenter._make_host

    def recording_host(plot_input, state):
        host = make_plot_host(plot_input, state)
        configure = host.configure

        def record_configuration(**configuration):
            if "selectors" in configuration:
                configured_selectors.append(tuple(configuration["selectors"]))
            return configure(**configuration)

        monkeypatch.setattr(host, "configure", record_configuration)
        return host

    monkeypatch.setattr(presenter, "_make_host", recording_host)

    # Hold the board's sole projection worker, then put one old-generation
    # publication behind it.  This makes the real race deterministic: the
    # Runtime replacement used to retire the publication before the Panel's
    # queued current_dataset(publication) call could consume it.
    release_projection = Event()
    blocker = presenter.board.submit_projection(
        lambda: release_projection.wait(5.0)
    )
    session.fire(shots=1)
    deadline = time.monotonic() + 5.0
    latest = shown
    while latest is shown and time.monotonic() < deadline:
        latest = session.signal_plane.latest_publication(signal)
        time.sleep(0.002)
    assert latest is not shown
    presenter.board.tick()
    presenter.board.commit()
    assert panel.port.surface_busy
    assert (
        old_host.instance_id,
        old_host.generation,
    ) in {
        (root.stream_id.value, root.generation)
        for root in session.signal_plane.publication_roots(latest)
    }

    assert presenter.update_logic_draft(
        node_id, values={"exposure_seconds": 0.012}
    )
    assert presenter.start_logic(node_id) is True
    presenter.poll_logic()
    assert panel.port.surface_busy
    assert (
        session.signal_plane.latest_publication(signal) is latest
    ), old_host.observation
    assert presenter.logic[node_id].host is old_host

    release_projection.set()
    blocker.result(5.0)
    _settle_panel_hosts(
        presenter,
        lambda: presenter.logic[node_id].host is not old_host,
    )
    replacement = presenter.logic[node_id].host
    assert replacement is not None and replacement.running
    assert replacement.generation != old_host.generation

    session.fire(shots=1)
    _settle_panel_hosts(
        presenter,
        lambda: (
            (publication := _accepted(panel.port, "publication")) is not None
            and publication.event_ref.generation == replacement.generation
        ),
    )
    assert panel.port.last_error is None
    # The stronger property arrived with stable hosts: a same-geometry
    # replacement run keeps the panel host, so the crosshair never needs
    # a configure replay -- it simply SURVIVES in place.
    assert not configured_selectors
    restored_crosshair = panel.host.selector_state(
        SelectorKind.CROSSHAIR
    ).result().value
    assert restored_crosshair.value == crosshair.value
    assert not any(
        "another signal generation" in text
        for _severity, text in presenter.view.status
    )


def _wait_for_panel_save(presenter, path: Path) -> None:
    deadline = time.monotonic() + 20.0
    while (presenter._saving_panels or not path.exists()) and time.monotonic() < deadline:
        presenter.beat()
        time.sleep(0.005)
    assert path.exists() and not presenter._saving_panels


def _commit_area(
    host,
    *,
    lower_fraction: float = 0.25,
    upper_fraction: float = 0.75,
) -> None:
    """Commit one real Area gesture through the mounted raster surface."""

    front = host.wait_for_front(5.0)
    axes = front.interaction.axes[0]
    left, bottom, right, top = axes.bounds
    start = (
        left + lower_fraction * (right - left),
        bottom + lower_fraction * (top - bottom),
    )
    end = (
        left + upper_fraction * (right - left),
        bottom + upper_fraction * (top - bottom),
    )
    for action, point in (("press", start), ("move", end), ("release", end)):
        host._pointer_event(
            action,
            point[0],
            point[1],
            button=1,
            identity=front.identity,
            axes=axes,
            interaction=front.interaction,
        ).result()


def _zoom(host, step: float) -> None:
    """Zoom through the real Qt wheel route, not the private host seam."""

    from PyQt5 import QtCore, QtGui, QtWidgets
    from zlc_ui.qt import ensure_qt_app

    app = ensure_qt_app(["panel-wheel"])
    widget = host.qt_widget()
    before = host.describe_display().result().value.viewport
    center = widget.rect().center()
    wheel = QtGui.QWheelEvent(
        QtCore.QPointF(center),
        QtCore.QPointF(widget.mapToGlobal(center)),
        QtCore.QPoint(),
        QtCore.QPoint(0, 120 if step > 0 else -120),
        QtCore.Qt.NoButton,
        QtCore.Qt.NoModifier,
        QtCore.Qt.NoScrollPhase,
        False,
    )
    QtWidgets.QApplication.sendEvent(widget, wheel)
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        app.processEvents()
        if host.describe_display().result().value.viewport != before:
            return
        time.sleep(0.005)
    raise AssertionError("the real Qt wheel event did not change the plot viewport")


def _zoom_in(host) -> None:
    _zoom(host, -1.0)


def _zoom_out(host) -> None:
    _zoom(host, 1.0)


def test_adding_a_panel_shows_a_card_and_reports_it(presenter, session) -> None:
    """The card appears at once; it shows the host when the host has drawn.

    A card takes its surface from the front that lands on it -- which is what
    keeps a panel being re-specified from going blank for the length of the
    new render -- so a brand new panel is carded immediately and filled a
    beat later.
    """

    node, snapshot = _one_shot(session)
    binding = presenter.add_panel(node.signal_key("frames"), snapshot, title="frames")

    assert presenter.view.cards, "the view was never given a card"
    assert "1 panel" in presenter.view.summary
    _settle_panel_hosts(
        presenter, lambda: presenter.view.cards[0].surface is not None
    )
    assert presenter.view.cards[0].surface is binding.host


def test_removing_a_panel_takes_the_card_away_and_closes_its_host(presenter, session) -> None:
    node, snapshot = _one_shot(session)
    binding = presenter.add_panel(node.signal_key("frames"), snapshot)
    assert presenter.edit_panel(binding.panel_id)
    _settle_panel_hosts(
        presenter, lambda: binding.editor_selections is not None
    )
    live_host = binding.host
    editor_host = binding.editor_host
    editor_selections = binding.editor_selections
    assert editor_host is not None and editor_selections is not None

    presenter.view.cards[0].remove_requested.emit()
    assert presenter.view.cards == ()
    assert presenter.panels == {}
    assert live_host._closing and editor_host._closing
    assert editor_selections._releases == []
    assert "0 panel" in presenter.view.summary


def test_pausing_freezes_the_picture_and_keeps_the_bench_running(
    presenter, session
) -> None:
    """Pause is a display state, not an idle bench.

    The beat used to skip board.tick() entirely while paused.  The tick is
    the only periodic caller of the plane's freeze, and that freeze is not a
    read: it routes new publications into the latest-only processor lane and
    drains finished work back out.  So pausing the picture also stopped the
    display clock and stopped every selection- and fit-derived signal from
    being computed -- which is why a test elsewhere in this file had to
    freeze the plane by hand inside a wait loop to make a paused board make
    progress at all.
    """

    node, snapshot = _one_shot(session)
    presenter.add_panel(node.signal_key("frames"), snapshot)
    assert presenter.view.paused is False

    # Freezing the plane cannot be the observable: other steps of the beat
    # freeze it too (a panel schema lookup does).  What the rule is ABOUT is
    # the wiring -- the beat must still tick, and withhold only staging.
    ticks: list[bool] = []
    original = presenter.board.tick

    def recorded(*, stage: bool = True):
        ticks.append(stage)
        return original(stage=stage)

    presenter.board.tick = recorded
    try:
        presenter.view.pause_toggled.emit(True)
        assert presenter.view.paused is True
        assert "paused" in presenter.view.summary

        ticks.clear()
        presenter.beat()
        assert ticks == [False], (
            "a paused console must still tick with staging withheld -- the "
            "tick is the only periodic pump of the derived-signal lane and "
            "the only thing that advances the display clock"
        )

        presenter.view.pause_toggled.emit(False)
        assert presenter.view.paused is False
        assert "running" in presenter.view.summary
        ticks.clear()
        presenter.beat()
        assert ticks == [True]
    finally:
        presenter.board.tick = original


def test_header_save_layout_writes_no_panel_dataset(
    presenter, session, tmp_path
) -> None:
    """Header layout persistence is wiring, not a whole-board archive."""

    node, snapshot = _one_shot(session)
    presenter.add_panel(node.signal_key("frames"), snapshot, title="frames")
    path = tmp_path / "layout.json"
    presenter.view.save_answer = str(path)

    presenter.view.save_layout_requested.emit()

    import json

    document = json.loads(path.read_text(encoding="utf-8"))
    assert presenter.LAYOUT_FORMAT == "zlc.console-board"
    assert document["format"] == presenter.LAYOUT_FORMAT
    assert document["panels"][0]["signal"] == node.signal_key("frames")
    assert not tuple(tmp_path.glob("*.npz"))


def test_the_presenter_never_imports_qt() -> None:
    """Qt lives in the view layer and in one thread-hopping shim, nowhere else."""

    import ast

    source = Path(__import__("zlc_workbench.console", fromlist=["console"]).__file__)
    tree = ast.parse(source.read_text(encoding="utf-8"))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and not node.level:
            roots.add((node.module or "").split(".")[0])
    assert "PyQt5" not in roots


def test_add_panel_puts_a_blank_fixed_kind_panel_on_the_board(presenter) -> None:
    """Panel authoring is independent of whether a producer has run yet."""

    # ONE naming scheme: the plot kind's own name, nothing invented beside
    # it.  The cell kind is NOT a menu row -- it is a panel parameter.
    assert presenter.view.kinds == (
        ("image", "image"),
        ("curve", "curve"),
        ("rolling", "rolling"),
        ("histogram", "histogram"),
        ("facet_grid", "facet_grid"),
    )
    presenter.view.add_panel_requested.emit("image")
    assert len(presenter.panels) == 1
    binding = next(iter(presenter.panels.values()))
    assert binding.signal == ""
    assert binding.kind == "image"
    assert binding.host is None
    assert binding.port is None
    assert presenter.view.cards
    assert presenter.view.panel_intervals == (100, 200, 400, 800)
    assert presenter.view.panel_default_interval == 100
    from zlc_plot import DEFAULTS

    assert presenter.view.panel_sizes == DEFAULTS.layout.size_names
    assert presenter.view.panel_default_size == DEFAULTS.layout.default_preset
    assert binding.state.interval_ms == 100
    display = {
        str(field["key"]): field
        for field in binding.parameter_surface["display"]
    }
    assert {
        "colormap",
        "color_min",
        "color_max",
        "show_colorbar",
    }.issubset(display)
    assert display["title"]["automatic"] is True
    for key in ("color_min", "color_max"):
        assert display[key]["automatic"] is False
        assert display[key]["unavailable_reason"]
    assert binding.state.overlay_signal == ""
    original_state = binding.state
    assert presenter.update_panel_state(binding.panel_id, {"interval_ms": 500}) is False
    assert binding.state is original_state


def test_facet_grid_cell_kind_is_a_panel_parameter(presenter) -> None:
    """The cell kind is chosen in panel settings; empty means the data decides."""

    # The presenter projects the one cell vocabulary to the view seam.
    assert presenter.view.grid_cell_kinds == ("curve", "image", "histogram")

    binding = presenter.add_selected_panel("facet_grid")
    assert binding is not None
    assert binding.state.kind == "facet_grid"
    assert binding.state.cell_kind == ""
    # No cell yet means no display contract yet -- an authored state, not an
    # error: the surface says a signal will resolve it.
    assert "signal" in binding.parameter_surface["display_unavailable"]
    assert binding.parameter_surface["display"] == ()

    # The operator may fix any legal grid cell, before or after a signal --
    # this is the exact patch the settings control emits.
    assert presenter.update_panel_state(
        binding.panel_id, {"cell_kind": "image"}
    ) is True
    assert binding.state.cell_kind == "image"
    assert presenter.update_panel_state(
        binding.panel_id, {"cell_kind": "rolling"}
    ) is False
    assert binding.state.cell_kind == "image"
    # And back to automatic: the data decides again.
    assert presenter.update_panel_state(
        binding.panel_id, {"cell_kind": ""}
    ) is True
    assert binding.state.cell_kind == ""


def test_changing_the_cell_kind_rebuilds_the_plot_host(
    presenter, session
) -> None:
    """A changed cell kind is SPEC identity: the host rebuilds, not shrugs.

    The state used to accept the patch while the plot kept drawing the old
    spec -- a control that looked live and did nothing.
    """

    session.load_pulse(PULSE_NAME)
    node = CameraMeasurementNode(
        camera=session.camera,
        request=CameraMeasurementRequest("camera", 0.02, None, 2, CAMERA_WINDOWS),
        signal_plane=session.signal_plane,
        producer="cm",
    )
    capture = node.prepare()
    session.fire(shots=2)
    capture.collect()
    session.nodes = [node]
    signal = node.signal_key("frames")

    binding = presenter.add_selected_panel("facet_grid")
    assert presenter.update_panel_state(binding.panel_id, {"signal": signal}) is True
    _settle_panel_hosts(presenter, lambda: binding.host is not None)
    assert binding.host is not None
    first_host = binding.host

    # Settle: the described FULL vocabulary of the current cells is written
    # back into state.display.  This is what the running app always does, and
    # it is what makes the next switch dangerous -- the bag now carries names
    # the next vocabulary does not declare.
    _settle_panel_hosts(presenter, lambda: bool(binding.state.display))
    assert binding.state.display, "the settle never described the panel"

    from zlc_plot import DEFAULTS
    from zlc_plot.specs import parameter_schema_for_kind

    def _switch(cell_kind: str, previous_host) -> None:
        # Non-vacuous: the bag being carried INTO this switch must hold at
        # least one name the target vocabulary does not declare, or the
        # switch proves nothing about crossing vocabularies.
        target_names = set(
            parameter_schema_for_kind(
                "facet_grid", style=DEFAULTS.style, facet_cell_kind=cell_kind
            ).names
        )
        assert set(binding.state.display) - target_names, (
            f"the stored appearance is a subset of {cell_kind} cells'; "
            "this switch cannot turn red"
        )
        assert presenter.update_panel_state(
            binding.panel_id, {"cell_kind": cell_kind}
        ) is True
        assert binding.state.cell_kind == cell_kind
        _settle_panel_hosts(
            presenter,
            lambda: binding.host is not None and binding.host is not previous_host,
        )
        assert binding.host is not None
        assert binding.host is not previous_host, (
            "the plot host must be rebuilt for the new cell kind"
        )
        # The rebuilt host must actually COME UP with the description returned
        # by its accepted configure operation.  Console no longer repopulates
        # state later from the host's initial-metadata cache.
        _settle_panel_hosts(
            presenter,
            lambda: binding.accepted_display is not None
            and bool(binding.state.display),
        )
        assert not binding.reported_condition, binding.reported_condition
        assert binding.state.display, (
            f"the {cell_kind} host never settled; it likely failed to start"
        )

    # Both directions of the vocabulary change: whatever the data decided
    # first, each switch crosses into names the other cells do not declare.
    _switch("histogram", first_host)
    histogram_host = binding.host
    assert histogram_host is not None
    assert presenter.update_panel_state(
        binding.panel_id,
        {"display": {"threshold_classifier": True}},
    )
    _settle_panel_hosts(
        presenter,
        lambda: binding.configuration is None
        and binding.state.display.get("threshold_classifier") is True,
    )
    threshold_events = []
    release_threshold = histogram_host.subscribe_selection(
        threshold_events.append
    ).result().value
    histogram_host.set_threshold_selector(1.0).result()
    presenter.commit_surfaces()
    assert binding.state.classifier_thresholds
    assert threshold_events

    _switch("image", histogram_host)
    assert binding.state.classifier_thresholds == (), (
        "histogram interaction leaked into image cells"
    )
    # A callback already queued by the retired histogram host is stale; it
    # cannot repopulate state after the image host has been accepted.
    presenter._settle_panel_threshold(
        binding.panel_id, histogram_host, threshold_events[-1]
    )
    assert binding.state.classifier_thresholds == ()
    release_threshold()
    assert presenter.edit_panel(binding.panel_id)
    _settle_panel_hosts(presenter, lambda: binding.editor_host is not None)
    assert binding.editor_host.startup_failure is None


def test_a_limit_does_not_follow_a_panel_across_cell_kinds(
    presenter, session
) -> None:
    """Appearance crosses a cell-kind change; a LIMIT does not.

    A curve cell that inherited an image cell's TIGHT colour re-fit
    re-fitted its y axis on every shot: the value survived only because the
    two vocabularies happened to share a name.  A limit describes the
    quantity the old kind plotted, the spec that declares it says so, and
    the record drops it with the projection and the fit.
    """

    from zlc_plot import DEFAULTS, PlotKind
    from zlc_plot.specs import (
        non_portable_display_names,
        parameter_schema_for_kind,
    )

    session.load_pulse(PULSE_NAME)
    node = CameraMeasurementNode(
        camera=session.camera,
        request=CameraMeasurementRequest("camera", 0.02, None, 2, CAMERA_WINDOWS),
        signal_plane=session.signal_plane,
        producer="cm",
    )
    capture = node.prepare()
    session.fire(shots=2)
    capture.collect()
    session.nodes = [node]

    binding = presenter.add_selected_panel("facet_grid")
    assert presenter.update_panel_state(
        binding.panel_id, {"signal": node.signal_key("frames")}
    ) is True
    _settle_panel_hosts(
        presenter,
        lambda: binding.accepted_display is not None
        and bool(binding.state.display),
    )
    # The data decided image cells (a picture is the densest structure);
    # the record keeps the empty "data decides" cell kind.
    assert binding.accepted_display.spec.cell.kind is PlotKind.IMAGE
    image_cells = parameter_schema_for_kind(
        "facet_grid", style=DEFAULTS.style, facet_cell_kind="image"
    )
    curve_cells = parameter_schema_for_kind(
        "facet_grid", style=DEFAULTS.style, facet_cell_kind="curve"
    )
    # Non-vacuous: both vocabularies declare relim_mode, and the value set
    # here is not what curve cells start from.
    foreign = next(
        value
        for value in image_cells["relim_mode"].choices
        if value != curve_cells["relim_mode"].default
    )
    assert presenter.update_panel_state(
        binding.panel_id,
        {"display": {"relim_mode": foreign, "title": "kept"}},
    ) is True
    _settle_panel_hosts(
        presenter,
        lambda: binding.configuration is None
        and binding.state.display.get("relim_mode") == foreign,
    )

    assert presenter.update_panel_state(
        binding.panel_id, {"cell_kind": "curve"}
    ) is True
    # The record itself, before any host has described the new cells.
    assert binding.state.display.get("title") == "kept"
    assert "relim_mode" not in binding.state.display
    assert not set(binding.state.display) & non_portable_display_names()
    _settle_panel_hosts(
        presenter,
        lambda: binding.accepted_display is not None
        and binding.state.display.get("relim_mode") is not None,
    )
    assert not binding.reported_condition, binding.reported_condition
    assert binding.state.display["relim_mode"] == curve_cells["relim_mode"].default
    assert binding.state.display["title"] == "kept"


def test_a_blank_panel_can_be_wired_after_a_signal_publishes(
    presenter, session, monkeypatch
) -> None:
    """Only a new signal/schema replaces a host; plot settings configure it."""

    presenter.view.add_panel_requested.emit("image")
    binding = next(iter(presenter.panels.values()))
    assert presenter.edit_panel(binding.panel_id) is True
    assert binding.editor_host is None
    node, _snapshot = _one_shot(session)
    signal = node.signal_key("frames")
    assert presenter.update_panel_state(binding.panel_id, {"signal": signal}) is True
    assert binding.signal == signal
    assert binding.state.title == signal
    _settle_panel_hosts(presenter, lambda: binding.host is not None)
    assert binding.host is not None
    assert binding.port is not None
    _settle_panel_hosts(
        presenter,
        lambda: bool(binding.parameter_surface["fit"]),
    )
    _settle_panel_hosts(presenter, lambda: binding.editor_host is not None)
    first_host = binding.host
    first_editor_host = binding.editor_host
    assert first_editor_host is not None
    configurations = []
    configure = first_host.configure

    def record_configuration(**values):
        configurations.append(values)
        return configure(**values)

    monkeypatch.setattr(first_host, "configure", record_configuration)

    def unexpected_close(*, timeout=None):
        raise AssertionError(f"a parameter edit closed the live host ({timeout=})")

    monkeypatch.setattr(first_host, "close", unexpected_close)
    described = _operation_value(binding.host.describe_display())
    assert described.display_state.values["title"] is None

    assert presenter.update_panel_state(
        binding.panel_id, {"title": "Renamed panel"}
    )
    assert binding.host is first_host
    assert binding.editor_host is first_editor_host
    described = _operation_value(binding.host.describe_display())
    assert described.display_state.values["title"] is None
    assert binding.state.title == "Renamed panel"
    assert configurations == [], "a card title must not submit plot work"

    assert presenter.update_panel_state(
        binding.panel_id, {"display": {"title": "Camera image"}}
    )
    _settle_panel_hosts(
        presenter,
        lambda: (
            binding.configuration is None
            and binding.editor_configuration is None
        ),
    )
    described = _operation_value(binding.host.describe_display())
    assert described.display_state.values["title"] == "Camera image"
    editor_description = _operation_value(
        binding.editor_host.describe_display()
    )
    assert editor_description.display_state.values["title"] == "Camera image"
    assert binding.state.title == "Renamed panel"

    assert presenter.update_panel_state(
        binding.panel_id, {"display": {"title": None}}
    )
    described = _operation_value(binding.host.describe_display())
    assert described.display_state.values["title"] is None
    fit_model = next(
        value
        for _label, value in binding.parameter_surface["fit"][0]["choices"]
    )
    assert presenter.update_panel_state(
        binding.panel_id, {"fit": {"model": fit_model}}
    )
    # Authored plot state crosses in one operation.  Interaction fields are
    # deliberately omitted for a standing host: its committed Area/viewport/
    # focus is newer than the asynchronously acknowledged PanelState mirror.
    assert configurations[-1] == {
        "semantic": dict(binding.state.semantic),
        "parameters": dict(binding.state.display),
        "size": binding.state.size,
        "fit": dict(binding.state.fit),
        "fit_live": True,
    }
    assert presenter.view.panel_editors[binding.panel_id]["state"]["signal"] == signal

    # A rejected authored target may remain editable while the old accepted
    # host keeps presenting data.  Refresh must freeze that accepted target,
    # not relabel its old pixels with the rejected draft.
    surface = binding.accepted_surface
    assert surface is not None
    binding.editor_open = False
    binding.state = replace(
        binding.state,
        display={**binding.state.display, "title": "not accepted"},
    )
    binding.refresh_requested = True
    presenter._panel_presented(binding, surface)
    assert binding.frozen_data is not None
    assert binding.frozen_data.target is surface.target
    assert binding.frozen_configuration_incompatible


def test_arming_a_fit_from_setting_reaches_the_panels_pixels(
    presenter, session
) -> None:
    """A fit is an analysis with a completion, not a display parameter.

    It used to ride ``configure(fit_model=...)``, whose completion describes
    only the configure.  The panel's widget stages its fronts (same-shot
    batching) rather than auto-presenting them, so the fit was computed,
    painted on the worker canvas and dropped.  Live data hid it -- the next
    data front carried the fit along -- and a fully published dataset, which
    has no next front, showed nothing at all.
    """

    node, snapshot = _one_shot(session)
    binding = presenter.add_panel(
        node.signal_key("frames"), snapshot, kind="image"
    )
    _settle_panel_hosts(
        presenter,
        lambda: bool(binding.parameter_surface.get("fit"))
        and bool(presenter.view.presented_fronts),
    )
    fit_model = next(
        value for _label, value in binding.parameter_surface["fit"][0]["choices"]
    )

    before = np.array(presenter.view.presented_fronts[-1][1].buffer.as_rgba(), copy=True)
    assert presenter.update_panel_state(
        binding.panel_id, {"fit": {"model": fit_model}}
    )

    changed = False
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline:
        presenter.beat()
        latest = presenter.view.presented_fronts[-1][1].buffer.as_rgba()
        if latest.shape != before.shape or not np.array_equal(latest, before):
            changed = True
            break
        time.sleep(0.005)
    assert changed, "the fit never reached the panel's presented pixels"

    _settle_panel_hosts(
        presenter,
        lambda: any(
            field["key"] == "expression"
            for field in binding.parameter_surface["fit"]
        ),
    )
    # In the model's own printed symbols -- resolved from the model this
    # test picked, since it takes whichever one the panel offers first.
    from zlc_plot.fit import builtin_fit_models

    chosen = next(
        model
        for model in builtin_fit_models()
        if str(model.model_id) == str(fit_model)
    )
    symbol_of = {
        str(parameter.name): str(parameter.symbol)
        for parameter in chosen.parameters
    }
    assert presenter.update_panel_state(
        binding.panel_id,
        {
            "fit": {
                "expression": (
                    f"{symbol_of['offset']}=0, "
                    f"{symbol_of['amplitude']}=guess(1)"
                )
            }
        },
    )
    _settle_panel_hosts(
        presenter,
        lambda: binding.state.fit.get("fixed") == {"offset": 0.0}
        and binding.state.fit.get("initial") == {"amplitude": 1.0},
    )

    assert presenter.update_panel_state(
        binding.panel_id,
        {"fit": {"expression": "missing=1"}},
    )
    _settle_panel_hosts(
        presenter,
        lambda: set(binding.state.fit) == {"model"}
        and "automatic fit is active"
        in binding.parameter_surface["fit_refused"],
    )
    expression = next(
        field
        for field in binding.parameter_surface["fit"]
        if field["key"] == "expression"
    )
    assert expression["value"] == "missing=1"


def test_a_saved_figure_contains_the_fit_it_was_saved_with(
    presenter, session, tmp_path
) -> None:
    """Save Fig writes the analysis, not the picture that preceded it.

    The saved host used to receive its fit through ``configure(fit_model=)``
    and the file was written immediately after that call returned -- so the
    PNG was whatever had been drawn BEFORE the fit landed.
    """

    node, snapshot = _one_shot(session)
    binding = presenter.add_panel(
        node.signal_key("frames"), snapshot, kind="image"
    )
    assert presenter.edit_panel(binding.panel_id)
    _settle_panel_hosts(
        presenter,
        lambda: bool(binding.parameter_surface.get("fit"))
        and binding.editor_host is not None
        and not binding.frozen_configuration_incompatible,
    )
    fit_model = next(
        value for _label, value in binding.parameter_surface["fit"][0]["choices"]
    )

    plain = tmp_path / "plain.png"
    assert presenter.save_panel_figure(binding.panel_id, str(plain)) is True
    _wait_for_panel_save(presenter, plain)
    plain_bytes = plain.read_bytes()

    assert presenter.update_panel_state(
        binding.panel_id, {"fit": {"model": fit_model}}
    )
    _settle_panel_hosts(
        presenter,
        lambda: binding.configuration is None
        and binding.editor_configuration is None
        and not binding.frozen_configuration_incompatible,
    )
    fitted = tmp_path / "fitted.png"
    assert presenter.save_panel_figure(binding.panel_id, str(fitted)) is True
    _wait_for_panel_save(presenter, fitted)

    assert fitted.read_bytes() != plain_bytes, (
        "the saved figure is identical with and without a fit"
    )


def test_an_invalid_overlay_choice_is_rejected_without_mutating_the_panel(
    presenter, session
) -> None:
    node, snapshot = _one_shot(session)
    binding = presenter.add_panel(
        node.signal_key("frames"), snapshot, kind="image"
    )
    original = binding.state

    assert presenter.update_panel_state(
        binding.panel_id,
        {"overlay_signal": "@logic/other/site_overlay"},
    ) is False
    assert binding.state is original
    # Not "no RUNNING node": a stopped run's data stays on the bench, so what
    # makes this choice invalid is that no node here publishes that signal.
    assert any(
        "this console publishes no" in text
        for severity, text in presenter.view.status
        if severity == "error"
    )


def test_a_contradictory_display_state_is_refused_at_the_write(
    presenter, session
) -> None:
    """What no host could ever accept must never be STORED, and said once.

    An inverted limit pair used to pass through update_panel_state, fail
    the host at its next start, and lock the operator out of every surface
    that could repair it.  It is refused at the write with the contract's
    own sentence -- while an INCOMPLETE state still passes, because fixed
    limits materialize on the next configure.

    The sentence is a WARNING and a standing condition, not an error, and
    not one per keystroke.  Every Setting field applies as it is typed, so
    the road to a valid number runs through invalid ones: typing 0.5 into a
    colour maximum whose minimum is 0 passes through "0", and that used to
    report "color_min must be smaller than color_max" in red, twice, on the
    way to a value the schema was always going to take.  A value that is
    not in effect YET is the shape a vacant semantic role already has.
    """

    node, snapshot = _one_shot(session)
    binding = presenter.add_panel(
        node.signal_key("frames"), snapshot, kind="image"
    )
    _settle_panel_hosts(presenter, lambda: binding.host is not None)
    before = binding.state

    assert presenter.update_panel_state(
        binding.panel_id,
        {"display": {"relim_mode": "fixed", "color_min": 5.0, "color_max": 1.0}},
    ) is False
    assert binding.state is before, "a refused write must store nothing"
    said = [
        (severity, text)
        for severity, text in presenter.view.status
        if "must be smaller" in text
    ]
    assert said, presenter.view.status
    assert all(severity == "warning" for severity, _text in said), said
    assert binding.unapplied_display, "the reason must stand, not scroll away"
    assert "must be smaller" in binding.unapplied_display

    # The next keystroke on the way to a valid number carries the SAME
    # refusal, and the operator is not told twice.
    before_repeat = len(presenter.view.status)
    assert presenter.update_panel_state(
        binding.panel_id,
        {"display": {"relim_mode": "fixed", "color_min": 5.0, "color_max": 2.0}},
    ) is False
    assert len(presenter.view.status) == before_repeat, presenter.view.status[
        before_repeat:
    ]

    # Incomplete is not contradictory: the mode alone is a legal write, and
    # a write that lands clears the condition.
    assert presenter.update_panel_state(
        binding.panel_id, {"display": {"relim_mode": "fixed"}}
    ) is True
    assert binding.unapplied_display == ""


def test_a_wedged_display_state_cannot_lock_the_editor_that_repairs_it(
    presenter, session
) -> None:
    """A refused plot surface is a STATE of the editor, not a closed window.

    A stored contradictory pair (a legacy board, a state written before the
    write-time contract) makes the Edit host fail at start.  The editor
    still OPENS -- its parameter form is the one tool that can repair the
    state the host refused -- the refusal is reported, and the accepted
    repair mounts the surface without reopening anything.
    """

    node, snapshot = _one_shot(session)
    binding = presenter.add_panel(
        node.signal_key("frames"), snapshot, kind="image"
    )
    assert presenter.edit_panel(binding.panel_id) is True
    _settle_panel_hosts(
        presenter, lambda: binding.editor_selections is not None
    )
    presenter.close_panel_editor(binding.panel_id)
    binding.state = replace(
        binding.state,
        display={
            **binding.state.display,
            "relim_mode": "fixed",
            "color_min": 5.0,
            "color_max": 1.0,
        },
    )

    assert presenter.edit_panel(binding.panel_id) is True, (
        "a dead plot surface closed the editor whose form repairs it"
    )
    assert binding.editor_open is True
    # ``editor_host is None`` is also the ordinary state BEFORE the async
    # mount worker starts, so it cannot prove a refusal has settled.  Wait for
    # the owner-visible refusal itself; only then inspect whether the mount
    # refused synchronously or left a host carrying startup_failure.
    def mount_refused() -> bool:
        return any(
            "must be smaller" in text
            for severity, text in presenter.view.status
            if severity == "error"
        )

    _settle_panel_hosts(presenter, mount_refused)
    assert binding.editor_host is None or (
        binding.editor_host.startup_failure is not None
    )

    assert presenter.update_panel_state(
        binding.panel_id,
        {"display": {"color_min": 0.0, "color_max": 10.0}},
    ) is True
    _settle_panel_hosts(presenter, lambda: binding.editor_host is not None)
    assert binding.editor_host is not None, (
        "the accepted repair must mount Edit's plot surface"
    )
    described = _operation_value(binding.editor_host.describe_display())
    assert described.display_state.values["color_min"] == 0.0


def test_plot_materialized_fixed_limits_become_the_panel_state(
    presenter, session
) -> None:
    node, snapshot = _one_shot(session)
    binding = presenter.add_panel(
        node.signal_key("frames"), snapshot, kind="image"
    )
    _settle_panel_hosts(
        presenter,
        lambda: binding.host is not None
        and not binding.parameter_surface["display_unavailable"],
    )

    assert presenter.update_panel_state(
        binding.panel_id,
        {"display": {"relim_mode": "fixed"}},
    )
    _settle_panel_hosts(presenter, lambda: binding.configuration is None)

    assert binding.state.display["relim_mode"] == "fixed"
    assert binding.state.display["color_min"] is not None
    assert binding.state.display["color_max"] is not None

    natural = _operation_value(binding.host.resolved_color_limits())
    huge = float(natural.high) + max(1.0, abs(float(natural.high))) * 1000.0
    assert presenter.edit_panel(binding.panel_id)
    _settle_panel_hosts(
        presenter,
        lambda: binding.editor_host is not None
        and binding.editor_configuration is None,
    )
    for mode in ("tight", "normal"):
        presented_before = len(presenter.view.presented_fronts)
        assert presenter.update_panel_state(
            binding.panel_id,
            {
                "display": {
                    "relim_mode": "fixed",
                    "color_min": float(natural.low),
                    "color_max": huge,
                }
            },
        )
        _settle_panel_hosts(
            presenter,
            lambda: binding.configuration is None
            and binding.editor_configuration is None
            and len(presenter.view.presented_fronts) > presented_before,
        )
        fixed = _operation_value(binding.host.resolved_color_limits())
        assert float(fixed.high) == huge
        frozen_fixed = _operation_value(
            binding.editor_host.resolved_color_limits()
        )
        assert frozen_fixed == fixed

        assert presenter.update_panel_state(
            binding.panel_id,
            {"display": {"relim_mode": mode}},
        )
        _settle_panel_hosts(presenter, lambda: binding.configuration is None)
        assert binding.state.display["relim_mode"] == mode
        assert binding.state.display["color_min"] is None
        assert binding.state.display["color_max"] is None
        automatic = _operation_value(binding.host.resolved_color_limits())
        assert float(automatic.high) < huge

    colormap = next(
        entry
        for entry in binding.parameter_surface["display"]
        if entry["key"] == "colormap"
    )
    alternate = next(
        value
        for _label, value in colormap["choices"]
        if value != binding.state.display["colormap"]
    )
    assert presenter.update_panel_state(
        binding.panel_id,
        {
            "display": {
                "relim_mode": "fixed",
                "color_min": float(natural.low),
                "color_max": huge,
            }
        },
    )
    _settle_panel_hosts(presenter, lambda: binding.configuration is None)
    # Two coalesced edits: the complete target keeps the first edit while the
    # second edit's delta remains the authority for Fixed -> Tight cleanup.
    assert presenter.update_panel_state(
        binding.panel_id,
        {"display": {"colormap": alternate}},
    )
    assert presenter.update_panel_state(
        binding.panel_id,
        {"display": {"relim_mode": "tight"}},
    )
    _settle_panel_hosts(presenter, lambda: binding.configuration is None)
    assert binding.state.display["colormap"] == alternate
    assert binding.state.display["relim_mode"] == "tight"
    assert binding.state.display["color_min"] is None
    assert binding.state.display["color_max"] is None


def test_selector_interaction_does_not_disconnect_panel_signals(
    presenter, session
) -> None:
    """The UI interaction gate is not the ROI/fit publication lifecycle."""

    node, snapshot = _one_shot(session)
    binding = presenter.add_panel(node.signal_key("frames"), snapshot)
    _settle_panel_hosts(presenter, lambda: binding.bridge is not None)
    bridge = binding.bridge
    assert bridge is not None
    assert presenter.view.selectors is False

    presenter.view.selectors_toggled.emit(True)
    assert binding.bridge is bridge

    presenter.view.selectors_toggled.emit(False)
    assert binding.bridge is bridge
    assert presenter.view.selectors is False


def test_first_visible_panel_host_already_owns_its_derivation_bridge(
    presenter,
    session,
) -> None:
    """No owner turn may expose interactive pixels before their Bridge."""

    node, snapshot = _one_shot(session)
    binding = presenter.add_panel(
        node.signal_key("frames"),
        snapshot,
        kind="image",
    )
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        presenter.beat()
        if binding.host is not None:
            assert binding.accepted_display is not None
            assert binding.bridge is not None
            assert binding.selections is not None
            break
        time.sleep(0.005)
    else:
        raise AssertionError("panel host did not become visible")


def test_panel_publisher_edit_owns_stable_output_selection(
    presenter, session
) -> None:
    node, snapshot = _one_shot(session)
    binding = presenter.add_panel(node.signal_key("frames"), snapshot, kind="image")
    _settle_panel_hosts(presenter, lambda: binding.bridge is not None)
    presenter.beat()
    assert presenter.view.panel_publishers[binding.panel_id] == ()

    assert presenter.edit_panel_publisher(binding.panel_id)
    projection = presenter.view.panel_publisher_editors[binding.panel_id]
    assert projection["source_required"] is False
    assert "source_signal" not in projection
    assert set(projection["form_spec"].keys) == {
        "roi_frame", "roi_mean", "roi_sum", "roi_min", "roi_max",
        "roi_min_10_mean", "roi_max_10_mean",
    }
    assert all(field.kind == "bool" for field in projection["form_spec"].fields)

    fit_model = next(
        value
        for _label, value in binding.parameter_surface["fit"][0]["choices"]
    )
    assert presenter.update_panel_state(
        binding.panel_id,
        {"fit": {"model": fit_model}},
    )
    _settle_panel_hosts(
        presenter,
        lambda: bool(binding.parameter_surface.get("fit_outputs")),
    )
    projection = presenter.view.panel_publisher_editors[binding.panel_id]
    assert set(binding.parameter_surface["fit_outputs"]).issubset(
        {
            (field.key, field.label)
            for field in projection["form_spec"].fields
        }
    )

    presenter.view.panel_publisher_draft_changed.emit(
        binding.panel_id,
        {"values": {"roi_frame": False, "roi_max": False}},
    )
    _commit_area(binding.host)
    roi_mean = f"@logic/{binding.panel_id}/roi_mean"
    roi_frame = f"@logic/{binding.panel_id}/roi_frame"
    roi_max = f"@logic/{binding.panel_id}/roi_max"
    _settle_panel_hosts(
        presenter,
        lambda: session.signal_plane.freeze().value(roi_mean) is not None,
    )
    assert session.signal_plane.freeze().value(roi_frame) is None
    assert session.signal_plane.freeze().value(roi_max) is None

    presenter.view.panel_publisher_draft_changed.emit(
        binding.panel_id,
        {"values": {"roi_max": True}},
    )
    assert session.signal_plane.freeze().value(roi_max) is not None

    tree = LayoutDocument((binding.state,), ()).to_tree()
    restored = LayoutDocument.from_tree(tree).panels[0]
    assert restored.published_outputs == binding.state.published_outputs

    histogram = presenter.add_panel(
        node.signal_key("frames"), snapshot, kind="histogram"
    )
    _settle_panel_hosts(
        presenter, lambda: histogram.accepted_display is not None
    )
    assert [
        name for name, _label in presenter._panel_publisher_fields(histogram)
    ] == ["roi_frame", "roi_mean"], (
        "a Histogram range is a band on the measured value: it cuts no axis, "
        "but it decides which cells count, so the cut signal is offered"
    )


def test_camera_area_fit_owner_wake_and_failed_revision_reach_rolling_gap(
    presenter, session, monkeypatch
) -> None:
    """The real panel chain settles Area before Fit and preserves one gap."""

    from zlc_plot.fit import FitEngine
    from zlc_workbench.logic import stable_signal_key

    camera_id = presenter.add_logic(
        "camera_measurement",
        node_id="gap-monitor",
        values={
            "exposure_seconds": 0.002,
            "repeat": 0,
            "frames_per_cycle": 1,
        },
        device_keys={"camera": "camera"},
        open_editor=False,
    )
    session.load_pulse(PULSE_NAME)
    assert presenter.start_logic(camera_id)
    camera_signal = stable_signal_key(camera_id, "frames")

    deadline = time.monotonic() + 10.0
    camera_publication = None
    while camera_publication is None and time.monotonic() < deadline:
        session.fire(shots=1)
        presenter.beat()
        camera_publication = session.signal_plane.latest_publication(camera_signal)
        time.sleep(0.005)
    assert camera_publication is not None

    main = presenter.add_panel(
        camera_signal,
        camera_publication.value(camera_signal).snapshot,
        kind="image",
    )
    _settle_panel_hosts(
        presenter,
        lambda: (
            main.host is not None
            and main.bridge is not None
            and main.port is not None
            and _accepted(main.port, "publication") is not None
            and bool(main.parameter_surface.get("fit"))
        ),
    )
    # Area release is a plot-worker callback.  Its owner wake must settle the
    # canonical PanelState before a Fit click can atomically configure from it;
    # no periodic display beat is allowed in between these two operations.
    presenter.board.wake.take()
    presenter.set_deriving(True)
    _commit_area(main.host, lower_fraction=0.4, upper_fraction=0.6)
    assert not main.state.selector, "the Workbench mirror has not taken its owner turn"

    # Fit may be clicked before that owner turn.  An ordinary fit/display
    # update must preserve the host-native committed Area instead of replaying
    # the stale empty Workbench interaction record over it.
    assert presenter.update_panel_state(
        main.panel_id,
        {"fit": {"model": "radial_gaussian_center"}},
    )
    presenter.commit_surfaces()
    assert main.state.selector
    _settle_panel_hosts(
        presenter,
        lambda: bool(main.parameter_surface.get("fit_outputs")),
    )

    fit_signal = f"@logic/{main.panel_id}/amplitude"
    armed_on = session.signal_plane.latest_publication(camera_signal)
    assert armed_on is not None
    session.fire(shots=1)
    _settle_panel_hosts(
        presenter,
        lambda: (
            (current := session.signal_plane.latest_publication(camera_signal))
            is not None
            and current.event_ref != armed_on.event_ref
            and session.signal_plane.latest_publication(fit_signal) is not None
        ),
    )
    fit_publication = session.signal_plane.latest_publication(fit_signal)
    assert fit_publication is not None
    before_history = session.signal_plane.current_dataset(fit_signal)
    assert all(
        str(column.coordinate_id) != "zlc_data.primary-index"
        for column in before_history.block.schema.point_table.columns
    )
    accepted = main.host._session._accepted_fit
    assert accepted is not None and accepted.selection is not None
    assert accepted.selection.selector_kind is SelectorKind.AREA
    # The Image display frame and its cells are square.  A centred 20% x 20%
    # selector therefore covers the same whole-cell span on both axes even
    # though the 96x128 source footprint is letterboxed in that frame.
    assert accepted.selection.sample_count == 26 * 26
    rolling = presenter.add_panel(
        fit_signal,
        fit_publication.value(fit_signal).snapshot,
        kind="rolling",
    )
    _settle_panel_hosts(
        presenter,
        lambda: (
            rolling.host is not None
            and rolling.port is not None
            and _accepted(rolling.port, "publication") is not None
        ),
    )
    assert rolling.history_lease is not None

    original_fit = FitEngine.fit
    fail_once = [True]

    def controlled_failure(engine, *args, **kwargs):
        if fail_once[0]:
            fail_once[0] = False
            raise RuntimeError("controlled fit gap")
        return original_fit(engine, *args, **kwargs)

    monkeypatch.setattr(FitEngine, "fit", controlled_failure)
    previous = session.signal_plane.latest_publication(camera_signal)
    assert previous is not None
    session.fire(shots=1)
    _settle_panel_hosts(
        presenter,
        lambda: (
            (current := session.signal_plane.latest_publication(camera_signal))
            is not None
            and current.event_ref != previous.event_ref
        ),
    )
    failed_source = session.signal_plane.latest_publication(camera_signal)
    assert failed_source is not None

    def failed_fit_publication():
        candidate = session.signal_plane.latest_publication(fit_signal)
        if candidate is None:
            return None
        roots = session.signal_plane.publication_roots(candidate)
        return candidate if failed_source.event_ref in roots else None

    _settle_panel_hosts(presenter, lambda: failed_fit_publication() is not None)
    failed_fit = failed_fit_publication()
    assert failed_fit is not None
    failed_value = failed_fit.value(fit_signal)
    np.testing.assert_array_equal(
        failed_value.snapshot.block.validity.mask,
        np.asarray([[False]]),
    )
    assert np.isnan(failed_value.snapshot.block.values).all()
    shown_main = _accepted(main.port, "publication")
    assert shown_main is not None
    assert failed_source.event_ref not in session.signal_plane.publication_roots(
        shown_main
    ), "a fit failure presented its unpaired camera data"

    monkeypatch.setattr(FitEngine, "fit", original_fit)
    session.fire(shots=1)
    _settle_panel_hosts(
        presenter,
        lambda: (
            (current := session.signal_plane.latest_publication(camera_signal))
            is not None
            and current.event_ref != failed_source.event_ref
        ),
    )
    recovered_source = session.signal_plane.latest_publication(camera_signal)
    assert recovered_source is not None
    _settle_panel_hosts(
        presenter,
        lambda: (
            (shown := _accepted(rolling.port, "publication")) is not None
            and recovered_source.event_ref
            in session.signal_plane.publication_roots(shown)
        ),
    )

    rolling_session = rolling.host._session
    validity = np.asarray(rolling_session._payload.series[0].valid)
    def rolling_offsets() -> tuple[int, ...]:
        return tuple(
            int(value)
            for value in rolling.host._session._payload.series[0].x.canonical
        )

    # Rolling x counts back from the newest shot at zero, so a shot's place
    # in the trace is its distance from the newest one.
    offsets = rolling_offsets()
    assert offsets == tuple(range(offsets[0], 1))
    newest = max(
        failed_source.event_ref.sequence, recovered_source.event_ref.sequence
    )

    def place_of(sequence: int) -> int:
        return offsets.index(sequence - newest)

    gap_index = place_of(failed_source.event_ref.sequence)
    recovered_index = place_of(recovered_source.event_ref.sequence)
    assert not validity[gap_index]
    assert validity[recovered_index]
    previous_valid = int(np.flatnonzero(validity[:gap_index])[-1])
    assert not np.any(validity[previous_valid + 1 : recovered_index])
    frozen = rolling.frozen_data
    assert frozen is not None
    primary = next(
        column
        for column in frozen.snapshot.block.schema.point_table.columns
        if str(column.coordinate_id) == "zlc_data.primary-index"
    )
    assert tuple(dict.fromkeys(primary.values))
    saved_truth = session.signal_plane.current_dataset(
        fit_signal,
        frozen.publication,
    )
    assert frozen.snapshot.ref == saved_truth.ref
    np.testing.assert_array_equal(
        frozen.snapshot.expanded_validity(),
        saved_truth.expanded_validity(),
    )
    line = rolling_session._renderer._artists["rolling:history"][0]
    plotted = np.asarray(line.get_ydata(), dtype=float)
    assert np.isfinite(plotted[[previous_valid, recovered_index]]).all()
    assert np.isnan(plotted[previous_valid + 1 : recovered_index]).all()

    # Runtime is the only history owner. Shrinking the active maximum releases
    # older excess; widening cannot resurrect data Runtime deliberately freed.
    shown_before_edit = _accepted(rolling.port, "publication")
    frozen_before_edit = rolling.frozen_data
    assert presenter.update_panel_state(
        rolling.panel_id,
        {"display": {"window": 2}},
    )
    assert _accepted(rolling.port, "publication") is shown_before_edit
    assert rolling.frozen_data is frozen_before_edit
    assert not rolling.port.presentation_current, (
        "the old complete surface stays visible, but a changed representation "
        "is display debt until its replacement is accepted"
    )
    _settle_panel_hosts(
        presenter,
        lambda: (
            rolling.configuration is None
            and rolling.port.presentation_current
            and rolling_offsets() == offsets[-2:]
        ),
    )
    retained_offsets = offsets[-2:]
    assert _accepted(rolling.port, "publication") is shown_before_edit
    assert presenter.update_panel_state(
        rolling.panel_id,
        {"display": {"window": 100}},
    )
    _settle_panel_hosts(
        presenter,
        lambda: (
            rolling.configuration is None
            and rolling.port.presentation_current
            and rolling_offsets() == retained_offsets
        ),
    )
    from zlc_plot import AxisRef
    from zlc_plot.semantics import fate_field_name, scope_fate

    primary_fate = fate_field_name(
        AxisRef.point("zlc_data.primary-index")
    )
    assert presenter.update_panel_state(
        rolling.panel_id,
        # History exposes ordinary relative coordinates.  Zero is the latest
        # retained event; Plot no longer needs a history-only Latest sentinel.
        {"semantic": {primary_fate: scope_fate(0)}},
    )
    _settle_panel_hosts(
        presenter,
        lambda: (
            rolling.configuration is None
            and rolling.port.presentation_current
            and rolling_offsets() == (0,)
        ),
    )
    assert presenter.update_panel_state(
        rolling.panel_id,
        {"semantic": {primary_fate: "reduce"}},
    )
    _settle_panel_hosts(
        presenter,
        lambda: (
            rolling.configuration is None
            and rolling.port.presentation_current
            and rolling_offsets() == retained_offsets
        ),
    )
    lease = rolling.history_lease
    presenter.remove_panel(rolling.panel_id)
    assert lease is not None and lease.closed
    latest_only = session.signal_plane.current_dataset(fit_signal)
    assert all(
        str(column.coordinate_id) != "zlc_data.primary-index"
        for column in latest_only.block.schema.point_table.columns
    )


def test_a_second_region_reaches_the_bridge_on_a_live_panel(
    presenter, session
) -> None:
    """Moving the region must move what the region publishes.

    The host renders every revision it is handed while the bookkeeping that
    ACCEPTS a surface runs on the board's beat, so a hand always draws on a
    picture at least one revision ahead of the accepted one.  Three places
    demanded the exact publication, and on a live panel that is never the
    one the operator drew on: measured, every committed region after the
    first was refused, the ROI kept the first box's shape for ever, and the
    remembered old region was re-applied over the new mark.
    """

    from zlc_workbench.logic import stable_signal_key

    camera_id = presenter.add_logic(
        "camera_measurement",
        values={"exposure_seconds": 0.002, "repeat": 0, "frames_per_cycle": 1},
        device_keys={"camera": "camera"},
        open_editor=False,
    )
    session.load_pulse(PULSE_NAME)
    assert presenter.start_logic(camera_id)
    camera_signal = stable_signal_key(camera_id, "frames")
    deadline = time.monotonic() + 10.0
    publication = None
    while publication is None and time.monotonic() < deadline:
        session.fire(shots=1)
        presenter.beat()
        publication = session.signal_plane.latest_publication(camera_signal)
        time.sleep(0.005)
    assert publication is not None

    image = presenter.add_panel(
        camera_signal,
        publication.value(camera_signal).snapshot,
        kind="image",
    )
    _settle_panel_hosts(
        presenter,
        lambda: image.host is not None and image.bridge is not None,
    )
    presenter.set_deriving(True)
    roi_signal = f"@logic/{image.panel_id}/roi_frame"

    def region(lower, upper):
        _commit_area(image.host, lower_fraction=lower, upper_fraction=upper)
        presenter.commit_surfaces()
        seen = None
        end = time.monotonic() + 12.0
        while time.monotonic() < end:
            session.fire(shots=1)
            presenter.beat()
            found = session.signal_plane.latest_publication(roi_signal)
            if found is not None:
                seen = found
                break
            time.sleep(0.005)
        assert seen is not None, "the region published nothing at all"
        return session.signal_plane.current_dataset(roi_signal).block.values.shape

    wide = region(0.15, 0.85)
    # Drawn well outside the first box, so this is a NEW region rather than
    # a grab of the old one -- and small enough that its shape cannot be
    # confused with the first.
    narrow = region(0.55, 0.70)
    assert narrow != wide, (
        "the second region published the first region's shape: %s" % (wide,)
    )
    assert narrow[-1] * narrow[-2] < wide[-1] * wide[-2], (narrow, wide)


def test_history_transition_is_immediate_and_interactions_follow_indexed_front(
    presenter,
    session,
) -> None:
    """Lease transitions and current-host interactions share one truth."""

    from zlc_workbench.logic import stable_signal_key

    camera_id = presenter.add_logic(
        "camera_measurement",
        node_id="roi-monitor",
        values={
            "exposure_seconds": 0.002,
            "repeat": 0,
            "frames_per_cycle": 3,
        },
        device_keys={"camera": "camera"},
        open_editor=False,
    )
    session.load_pulse(PULSE_NAME)
    assert presenter.start_logic(camera_id)
    camera_signal = stable_signal_key(camera_id, "frames")
    deadline = time.monotonic() + 10.0
    camera_publication = None
    while camera_publication is None and time.monotonic() < deadline:
        session.fire(shots=1)
        presenter.beat()
        camera_publication = session.signal_plane.latest_publication(camera_signal)
        time.sleep(0.005)
    assert camera_publication is not None

    image = presenter.add_panel(
        camera_signal,
        camera_publication.value(camera_signal).snapshot,
        kind="image",
    )
    _settle_panel_hosts(
        presenter,
        lambda: image.host is not None and image.bridge is not None,
    )
    presenter.set_deriving(True)
    _commit_area(image.host, lower_fraction=0.35, upper_fraction=0.65)
    presenter.commit_surfaces()
    roi_signal = f"@logic/{image.panel_id}/roi_frame"
    _settle_panel_hosts(
        presenter,
        lambda: session.signal_plane.latest_publication(roi_signal) is not None,
    )
    first_roi = session.signal_plane.latest_publication(roi_signal)
    assert first_roi is not None
    for _index in range(3):
        previous_roi = session.signal_plane.latest_publication(roi_signal)
        assert previous_roi is not None
        session.fire(shots=1)
        _settle_panel_hosts(
            presenter,
            lambda: (
                (candidate := session.signal_plane.latest_publication(roi_signal))
                is not None
                and candidate.event_ref != previous_roi.event_ref
            ),
        )
    first_roi = session.signal_plane.latest_publication(roi_signal)
    assert first_roi is not None
    before_panel = session.signal_plane.current_dataset(roi_signal)
    assert all(
        str(column.coordinate_id) != "zlc_data.primary-index"
        for column in before_panel.block.schema.point_table.columns
    )

    histogram = presenter.add_panel(
        roi_signal,
        first_roi.value(roi_signal).snapshot,
        kind="histogram",
        display={"window": 1},
    )
    _settle_panel_hosts(
        presenter,
        lambda: (
            histogram.host is not None
            and histogram.port is not None
            and _accepted(histogram.port, "publication") is not None
        ),
    )
    shown = _accepted(histogram.port, "publication")
    one_shot_port = histogram.port
    assert shown is not None
    assert histogram.history_lease is None, (
        "a one-shot Histogram consumes the current event, not history"
    )
    retained = session.signal_plane.current_dataset(roi_signal)
    assert all(
        str(column.coordinate_id) != "zlc_data.primary-index"
        for column in retained.block.schema.point_table.columns
    )

    # Re-committing the Area intentionally replaces the selection-derived
    # generation.  Pause prevents the board from swapping the Histogram host,
    # leaving the exact old publication on screen while the new one exists.
    presenter.set_paused(True)
    _commit_area(image.host, lower_fraction=0.25, upper_fraction=0.75)
    presenter.commit_surfaces()
    deadline = time.monotonic() + 5.0
    current_roi = first_roi
    while (
        current_roi.event_ref.generation == first_roi.event_ref.generation
        and time.monotonic() < deadline
    ):
        # THE BEAT PUMPS IT, not this loop.  A paused console used to skip
        # the tick, and the tick is the only periodic caller of the plane's
        # freeze -- which is what routes a new publication into the
        # latest-only processor lane and drains the finished work back out.
        # So this loop had to freeze the plane by hand to make a
        # selection-derived signal advance at all, which is exactly the
        # defect: Pause was stopping the bench, not the picture.
        presenter.beat()
        time.sleep(0.005)
        candidate = session.signal_plane.latest_publication(roi_signal)
        if candidate is not None:
            current_roi = candidate
    assert current_roi.event_ref.generation != first_roi.event_ref.generation
    assert _accepted(histogram.port, "publication") is shown

    assert presenter.update_panel_state(
        histogram.panel_id,
        {"display": {"window": 100}},
    )
    _settle_panel_hosts(
        presenter,
        lambda: (
            histogram.configuration is None
            and histogram.state.display["window"] == 100
        ),
    )
    assert histogram.port is one_shot_port
    assert _accepted(histogram.port, "publication") is shown
    assert not histogram.port.presentation_current
    assert histogram.history_lease is not None
    assert histogram.history_lease.window == 100
    assert histogram.port.last_error is None
    assert not any(
        "another signal generation" in text
        for _severity, text in presenter.view.status
    )

    presenter.set_paused(False)
    _settle_panel_hosts(
        presenter,
        lambda: (
            (accepted := _accepted(histogram.port, "publication")) is not None
            and accepted.event_ref.generation == current_roi.event_ref.generation
        ),
    )
    assert histogram.port.last_error is None

    # History is signal-global but never owned by an ordinary Image.  It may
    # consume the current indexed representation while Histogram is alive;
    # releasing the last real history demand must invalidate and rebuild this
    # second panel too, instead of feeding indexed data into its event host.
    companion = presenter.add_panel(
        roi_signal,
        current_roi.value(roi_signal).snapshot,
        kind="image",
    )
    _settle_panel_hosts(
        presenter,
        lambda: companion.port is not None
        and _accepted(companion.port, "publication") is not None
        and companion.host is not None,
    )
    assert companion.history_lease is None

    grid = presenter.add_panel(
        roi_signal,
        current_roi.value(roi_signal).snapshot,
        kind="facet_grid",
    )
    _settle_panel_hosts(
        presenter,
        lambda: grid.host is not None
        and grid.port is not None
        and _accepted(grid.port, "publication") is not None
        and grid.bridge is not None
        and grid.selections is not None,
    )
    grid.host.focus_facet(0).result(timeout=10)
    _commit_area(grid.host, lower_fraction=0.25, upper_fraction=0.75)
    grid.host.set_crosshair_selector(1.0, 1.0).result(timeout=10)
    presenter.commit_surfaces()
    _settle_panel_hosts(
        presenter,
        lambda: (
            grid.state.focused_cell == 0
            and bool(grid.state.selector)
            and bool(grid.state.crosshair)
        ),
    )
    retained_interaction = (
        grid.state.focused_cell,
        dict(grid.state.selector),
        dict(grid.state.crosshair),
    )
    before_interaction_shot = session.signal_plane.latest_publication(roi_signal)
    assert before_interaction_shot is not None
    session.fire(shots=1)
    _settle_panel_hosts(
        presenter,
        lambda: (
            (latest_roi := session.signal_plane.latest_publication(roi_signal))
            is not None
            and latest_roi.event_ref != before_interaction_shot.event_ref
            and (
                grid.state.focused_cell,
                dict(grid.state.selector),
                dict(grid.state.crosshair),
            )
            == retained_interaction
        ),
    )
    presenter.remove_panel(grid.panel_id)

    indexed_port = histogram.port
    assert presenter.update_panel_state(
        histogram.panel_id,
        {"display": {"window": 1}},
    )
    assert histogram.history_lease is None
    immediate = session.signal_plane.current_dataset(roi_signal)
    assert all(
        str(column.coordinate_id) != "zlc_data.primary-index"
        for column in immediate.block.schema.point_table.columns
    )
    _settle_panel_hosts(
        presenter,
        lambda: all(
            panel.port.presentation_current
            and _accepted(panel.port, "publication") is not None
            for panel in (histogram, companion)
        ),
    )
    assert histogram.port is indexed_port
    assert histogram.history_lease is None
    assert histogram.history_lease is None
    latest = session.signal_plane.current_dataset(roi_signal)
    assert all(
        str(column.coordinate_id) != "zlc_data.primary-index"
        for column in latest.block.schema.point_table.columns
    )
    for panel in (histogram, companion):
        accepted_input = _accepted(panel.port, "plot_input")
        accepted_snapshot = getattr(
            accepted_input,
            "snapshot",
            accepted_input,
        )
        assert all(
            str(column.coordinate_id) != "zlc_data.primary-index"
            for column in accepted_snapshot.block.schema.point_table.columns
        )
    assert all(
        str(entry["label"]) != "source index"
        for entry in histogram.parameter_surface["semantic"]
    )


def test_one_failing_panel_interaction_is_a_line_not_the_instrument(
    presenter,
) -> None:
    """The drain names no exception classes, because the guess was wrong.

    A box drawn on a run whose data had been retired raised LookupError
    from the plane.  The drain caught TypeError and ValueError, so it
    passed through, left the beat, and left the Qt timer slot -- where
    PyQt ends the process.  Whatever an interaction raises, it is one
    error line and the next interaction still runs.
    """

    ran: list[str] = []

    def exploding() -> None:
        raise LookupError("signal 'x' is not retained")

    presenter._enqueue_panel_interaction(exploding)
    presenter._enqueue_panel_interaction(lambda: ran.append("after"))
    presenter._drain_panel_interactions()

    assert ran == ["after"], "one failure must not eat the queue behind it"
    errors = [text for severity, text in presenter.view.status if severity == "error"]
    assert any("cannot apply panel interaction" in text for text in errors), errors
    assert any("not retained" in text for text in errors), errors


def test_committed_selection_outputs_enter_the_real_occupancy_input(
    presenter, session, tmp_path
) -> None:
    from zlc_atom.nodes.calibration import (
        FrameContract,
        ReadoutModel,
        ReadoutModelKind,
        SiteMap,
        TrapCalibration,
    )

    producer_id = presenter.add_logic(
        "camera_measurement", node_id="cm", open_editor=False
    )
    node, snapshot = _one_shot(session, producer=producer_id)
    binding = presenter.add_panel(
        node.signal_key("frames"), snapshot, kind="image"
    )
    _settle_panel_hosts(
        presenter,
        lambda: binding.bridge is not None
        and binding.selections is not None
        and bool(binding.parameter_surface["fit"]),
    )
    consumer_id = presenter.add_logic("occupancy")

    fit_model = next(
        value
        for _label, value in binding.parameter_surface["fit"][0]["choices"]
    )
    binding.host.fit(fit_model).result()
    fit_signals = tuple(
        row
        for row in session.signal_plane.describe_signals()
        if row.contract_id == "zlc.selection.fit.parameter"
    )
    assert fit_signals, (binding.bridge.last_error, binding.selections.last_error)
    assert all(
        not row.name.rsplit("/", 1)[-1].startswith("fit_")
        for row in fit_signals
    )
    # ONE signal per parameter, carrying its own error.  There used to be a
    # second, "center_x_err", published beside it under its own contract --
    # two signals nothing related to each other, so a panel reducing the
    # parameter never saw the error at all.
    assert not any(
        row.name.rsplit("/", 1)[-1].endswith("_err")
        for row in session.signal_plane.describe_signals()
    )
    assert all(
        row.contract_id != "zlc.selection.fit.error"
        for row in session.signal_plane.describe_signals()
    )
    projection = presenter.logic_editor_projection(consumer_id)
    assert projection is not None
    assert any(
        name in projection["source_options"]
        for name in (row.name for row in fit_signals)
    )
    for row in fit_signals:
        if row.name in projection["source_options"]:
            leaf = row.name.rsplit("/", 1)[-1]
            assert projection["source_labels"][row.name] == (
                f"{leaf}  [{format_signal_shape(row.shape)}]"
            )

    presenter.set_deriving(True)
    _commit_area(binding.host)
    roi_signal = f"@logic/{binding.panel_id}/roi_frame"
    _settle_panel_hosts(
        presenter,
        lambda: session.signal_plane.freeze().value(roi_signal) is not None,
    )

    projection = presenter.view.logic_editors[consumer_id]
    assert roi_signal in projection["source_options"]
    roi_description = next(
        row
        for row in session.signal_plane.describe_signals()
        if row.name == roi_signal
    )
    assert projection["source_labels"][roi_signal] == (
        f"roi_frame  [{format_signal_shape(roi_description.shape)}]"
    )
    presenter.beat()
    shown_rows = presenter.view._rows[producer_id].publishes
    shown_in_logic_tab = {name for name, _shape, _detail in shown_rows}
    expected_leaves = {
        row.name.rsplit("/", 1)[-1] for row in fit_signals
    }.union({"roi_frame"})
    assert expected_leaves.isdisjoint(shown_in_logic_tab)
    assert "frames" in shown_in_logic_tab
    panel_group = dict(presenter.signal_groups())[binding.panel_id]
    panel_signal_names = {name for _label, name in panel_group}
    assert {row.name for row in fit_signals}.union({roi_signal}) <= panel_signal_names
    logic_tab_panel_names = {
        detail.rsplit(" · ", 1)[-1]
        for _name, _shape, detail in presenter.view.panel_publishers[
            binding.panel_id
        ]
    }
    assert {row.name for row in fit_signals}.union({roi_signal}) <= (
        logic_tab_panel_names
    )

    source = session.signal_plane.freeze().value(roi_signal)
    assert source is not None
    height, width = source.values.shape[-2:]
    site_ids = ("site_0001",)
    calibration = TrapCalibration(
        SiteMap(
            site_ids,
            np.asarray(((width / 2.0, height / 2.0),)),
            np.asarray((True,)),
            np.asarray((1.0,)),
        ),
        (
            ReadoutModel(
                site_ids,
                np.asarray((0.0,)),
                np.zeros(1),
                np.ones(1),
                np.asarray((True,)),
                np.asarray((1.0,)),
            ),
        ),
        ReadoutModelKind.BOX,
        FrameContract((height, width)),
    )
    artifact = calibration.save(tmp_path / "roi-calibration.json")
    assert presenter.update_logic_draft(
        consumer_id,
        source_signal=roi_signal,
        artifact_inputs={"calibration_path": str(artifact)},
    )
    assert presenter.start_logic(consumer_id)
    _settle_panel_hosts(
        presenter,
        lambda: presenter.logic[consumer_id].host is not None
        and presenter.logic[consumer_id].host.observation.terminal,
    )
    host = presenter.logic[consumer_id].host
    assert host is not None and host.observation.phase == "done"
    assert presenter.logic[consumer_id].draft.source_signal == roi_signal



def test_a_card_shows_whether_its_selectors_are_live(presenter, session) -> None:
    """The card says whether plot interaction, not publication, is enabled."""

    node, snapshot = _one_shot(session)
    binding = presenter.add_panel(node.signal_key("frames"), snapshot)
    _settle_panel_hosts(presenter, lambda: binding.host is not None)
    card = presenter.view.cards[0]
    assert card.selectors_enabled is False

    presenter.set_deriving(True)
    assert card.selectors_enabled is True

    presenter.set_deriving(False)
    assert card.selectors_enabled is False, "the card must show interaction is off"


def test_a_mounted_plot_widgets_error_lands_one_warning_on_the_status_strip(
    presenter, session
) -> None:
    """The plot widget's errorOccurred channel reaches the operator.

    Pointer currency-guard refusals ("the painted pointer front is no longer
    layout-compatible") arrive only on that channel, and nothing in the
    console connected it -- so a panel whose gestures were being refused was
    indistinguishable from a panel that works.
    """

    node, snapshot = _one_shot(session)
    binding = presenter.add_panel(
        node.signal_key("frames"), snapshot, title="camera"
    )
    presenter.view.status.clear()
    presenter.view.panel_plot_error.emit(
        binding.panel_id,
        "the painted pointer front is no longer layout-compatible",
    )
    assert presenter.view.status == [
        (
            "warning",
            "camera: the painted pointer front is no longer layout-compatible",
        )
    ]


def test_a_bridge_side_derivation_failure_is_reported_once(
    presenter, session
) -> None:
    """The selection bridge's own recorded refusal reaches the operator.

    ``_report_panel_errors`` read the selection sources and the port but
    never ``binding.bridge.last_error``, so a bridge-side derivation failure
    left the derived signal silently absent.  Same de-dup discipline as the
    other panel errors: one refusal is reported once.
    """

    node, snapshot = _one_shot(session)
    binding = presenter.add_panel(node.signal_key("frames"), snapshot)
    _settle_panel_hosts(presenter, lambda: binding.bridge is not None)

    class _FailedBridge:
        last_error = RuntimeError("the derivation refused this selection")

        def close(self) -> None:
            pass

    binding.bridge = _FailedBridge()
    presenter.view.status.clear()
    presenter.beat()
    reported = [item for item in presenter.view.status if item[0] == "error"]
    assert reported == [
        ("error", f"{binding.title}: the derivation refused this selection")
    ]
    assert presenter.view.cards[0].status == (
        "the derivation refused this selection",
        True,
    )

    presenter.view.status.clear()
    presenter.beat()
    assert not [item for item in presenter.view.status if item[0] == "error"]


def test_show_panel_mounts_after_the_async_canonical_front_is_ready(
    session,
) -> None:
    """The card remains ready until its canonical host can be accepted."""

    pytest.importorskip("zlc_plot")
    from zlc_workbench.apps.task_console import build_panel_host

    def spec_for(snapshot, kind="", cell_kind=""):
        return task_console_fitting_spec(snapshot.block.schema, kind, cell_kind)

    def ready_host(plot_input, state):
        host = build_panel_host(plot_input, state)
        host.wait_for_front(10.0)
        return host

    presenter = ConsolePresenter(
        session, _ConsoleView(), make_host=ready_host, spec_for=spec_for
    )
    try:
        node, snapshot = _one_shot(session)
        binding = presenter.add_panel(node.signal_key("frames"), snapshot)
        assert binding.host is None
        _settle_panel_hosts(
            presenter,
            lambda: binding.accepted_surface is not None,
        )
        presented = [
            front
            for panel_id, front in presenter.view.presented_fronts
            if panel_id == binding.panel_id
        ]
        assert presented, "show_panel left the staged widget empty until a beat"
        assert presented[0] is not None
    finally:
        presenter.close()


def test_header_save_screenshot_writes_one_plain_gui_image(
    presenter, session, tmp_path
) -> None:
    """The header screenshot does not loop over panels or write an archive."""

    node, snapshot = _one_shot(session)
    presenter.add_panel(node.signal_key("frames"), snapshot)
    path = tmp_path / "task-console.png"
    presenter.view.save_answer = str(path)

    presenter.view.save_screenshot_requested.emit()

    assert presenter.view.screenshot_path == str(path)
    assert path.read_bytes() == b"plain TaskConsole screenshot"
    assert not tuple(tmp_path.glob("*.npz"))


def test_every_control_on_a_card_is_answered(presenter, session) -> None:
    """A card looked configurable and was not: eight of its nine signals were
    raised and dropped.  Each is a decision about THAT panel."""

    node, snapshot = _one_shot(session)
    binding = presenter.add_panel(node.signal_key("frames"), snapshot, title="camera")
    card = presenter.view.cards[0]

    # It offers its own signal among the choices, or it cannot name itself.
    # Grouped under the producer that publishes it, which is the shape the real
    # card's combo renders.
    offered = {
        key: display
        for _producer, leaves in card.choices
        for display, key in leaves
    }
    assert binding.signal in offered
    shape = snapshot.block.values.shape
    shape_text = f"{shape[0]} × {shape[1]} × ({'×'.join(map(str, shape[2:]))})"
    assert offered[binding.signal] == f"frames  [{shape_text}]"
    assert card.chosen == binding.signal

    card.edit_requested.emit()
    assert presenter.view.focused_panel_editor == binding.panel_id
    assert presenter.view.panel_editors[binding.panel_id]["state"] == (
        binding.state.document()
    )

    presenter.view.panel_state_updates.clear()
    presenter.update_panel_state(binding.panel_id, {"title": "MOT"})
    assert presenter.panels[binding.panel_id].title == "MOT"
    assert card.title == "MOT"
    assert presenter.view.panel_state_updates == [
        (binding.panel_id, presenter.panels[binding.panel_id].state)
    ]
    assert presenter.view.panel_editors[binding.panel_id]["state"]["title"] == "MOT"

    presenter.update_panel_state(binding.panel_id, {"interval_ms": 100})
    assert binding.port.display_interval_ms == 100

    presenter.update_panel_state(binding.panel_id, {"size": "4x4"})
    assert card.size == "4x4"


def test_retargeting_a_panel_keeps_its_place_and_releases_the_old_host(
    presenter, session
) -> None:
    """A plotting host is built around the shape of what it draws, so a new
    signal gets a new host rather than a frame of pixels discovering that an
    image arrived where a curve was."""

    node, snapshot = _one_shot(session)
    first = presenter.add_panel(node.signal_key("frames"), snapshot)
    _settle_panel_hosts(presenter, lambda: first.host is not None)
    card = presenter.view.cards[0]
    old_host = first.host
    card.edit_requested.emit()
    frozen = presenter.view.panel_editors[first.panel_id]["frozen_snapshot"]
    assert frozen is not snapshot
    assert frozen.block.schema == snapshot.block.schema
    _settle_panel_hosts(presenter, lambda: first.editor_host is not None)
    old_editor_host = first.editor_host
    assert old_editor_host is not None and old_editor_host is not old_host

    # A second producer, so there is something else to point at.
    other, other_snapshot = _one_shot(session, producer="cm2")
    presenter.update_panel_state(
        first.panel_id, {"signal": other.signal_key("frames")}
    )

    binding = presenter.panels[first.panel_id]
    assert binding.signal == other.signal_key("frames")
    if binding.host is old_host:
        _settle_panel_hosts(
            presenter,
            lambda: binding.host is not None and binding.host is not old_host,
        )
    assert binding.host is not old_host
    assert presenter.view.cards[0] is card, "the card lost its place on the board"
    editor = presenter.view.panel_editors[first.panel_id]
    assert editor["stale"] is True
    assert editor["frozen_snapshot"] is frozen
    assert binding.editor_host is old_editor_host
    assert not old_editor_host._closing

    presenter.view.panel_snapshot_refresh_requested.emit(first.panel_id)
    _settle_panel_hosts(
        presenter,
        lambda: binding.frozen_data is not None
        and binding.frozen_data.signal == other.signal_key("frames")
        and binding.editor_configuration is None
        and binding.frozen_data.description is not None,
    )
    editor = presenter.view.panel_editors[first.panel_id]
    assert editor["stale"] is False
    assert editor["frozen_signal"] == other.signal_key("frames")
    assert editor["frozen_snapshot"] is not other_snapshot
    assert editor["frozen_snapshot"].block.schema == other_snapshot.block.schema
    replacement_editor_host = binding.editor_host
    assert replacement_editor_host is not None
    assert replacement_editor_host is not old_editor_host
    assert old_editor_host._closing
    assert (
        presenter.view.panel_editor_surfaces[first.panel_id]
        is replacement_editor_host
    )

    presenter.view.panel_editor_closed.emit(first.panel_id)
    assert binding.editor_host is None and binding.editor_selections is None
    assert replacement_editor_host._closing


def test_panel_editor_selection_uses_only_its_current_frozen_publication(
    presenter, session
) -> None:
    """A stale frozen image cannot patch the producer form for a new signal."""

    first_id = presenter.add_logic(
        "camera_measurement", node_id="camera_first", open_editor=False
    )
    second_id = presenter.add_logic(
        "camera_measurement", node_id="camera_second", open_editor=False
    )
    first_node, first_snapshot = _one_shot(session, producer=first_id)
    second_node, _second_snapshot = _one_shot(session, producer=second_id)
    panel = presenter.add_panel(
        first_node.signal_key("frames"), first_snapshot, kind="image"
    )
    _settle_panel_hosts(presenter, lambda: panel.host is not None)
    assert presenter.edit_panel(panel.panel_id)
    _settle_panel_hosts(presenter, lambda: panel.editor_host is not None)
    first_editor_host = panel.editor_host
    assert first_editor_host is not None
    _settle_panel_hosts(
        presenter,
        lambda: panel.editor_selections is not None
        and panel.frozen_data is not None
        and panel.frozen_data.description is not None,
    )
    assert presenter.view.selectors is False
    from zlc_ui.qt import ensure_qt_app

    app = ensure_qt_app(["panel-editor-selection"])
    live_widget = panel.host.qt_widget()
    editor_widget = first_editor_host.qt_widget()
    deadline = time.monotonic() + 2.0
    while live_widget.presented_front is None and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.005)
    assert live_widget.presented_front is not None
    live_sequence = live_widget.presented_front.identity.sequence

    first_before = dict(presenter.logic[first_id].draft.values)
    second_before = dict(presenter.logic[second_id].draft.values)
    _commit_area(first_editor_host)
    assert presenter.logic[first_id].draft.values == first_before
    presenter.beat()
    first_selected = dict(presenter.logic[first_id].draft.values)
    assert first_selected != first_before
    assert presenter.logic[second_id].draft.values == second_before
    editor_selector = first_editor_host.selector_state(SelectorKind.AREA).result().value
    live_selector = panel.host.selector_state(SelectorKind.AREA).result().value
    assert live_selector.value == editor_selector.value
    deadline = time.monotonic() + 2.0
    while (
        live_widget.presented_front.identity.sequence <= live_sequence
        and time.monotonic() < deadline
    ):
        app.processEvents()
        time.sleep(0.005)
    assert live_widget.presented_front.identity.sequence > live_sequence

    editor_sequence = editor_widget.presented_front.identity.sequence
    _commit_area(
        panel.host,
        lower_fraction=0.10,
        upper_fraction=0.60,
    )
    presenter.beat()
    editor_selector = first_editor_host.selector_state(SelectorKind.AREA).result().value
    live_selector = panel.host.selector_state(SelectorKind.AREA).result().value
    assert editor_selector.value == live_selector.value
    deadline = time.monotonic() + 2.0
    while (
        editor_widget.presented_front.identity.sequence <= editor_sequence
        and time.monotonic() < deadline
    ):
        app.processEvents()
        time.sleep(0.005)
    assert editor_widget.presented_front.identity.sequence > editor_sequence
    first_current = dict(presenter.logic[first_id].draft.values)
    assert first_current != first_selected

    previous_configuration = panel.editor_configuration
    assert presenter.refresh_panel_snapshot(panel.panel_id)
    assert panel.editor_configuration is previous_configuration
    assert panel.editor_host is first_editor_host
    assert panel.editor_host.qt_widget() is editor_widget
    assert not first_editor_host._closing

    assert presenter.update_panel_state(
        panel.panel_id, {"signal": second_node.signal_key("frames")}
    )
    assert (
        panel.frozen_configuration_incompatible
        and panel.editor_host is first_editor_host
    )
    _commit_area(first_editor_host)
    presenter.beat()
    assert presenter.logic[first_id].draft.values == first_current
    assert presenter.logic[second_id].draft.values == second_before

    assert presenter.refresh_panel_snapshot(panel.panel_id)
    _settle_panel_hosts(
        presenter,
        lambda: panel.frozen_data is not None
        and panel.frozen_data.signal == second_node.signal_key("frames")
        and panel.editor_host is not first_editor_host,
    )
    second_editor_host = panel.editor_host
    assert second_editor_host is not None and second_editor_host is not first_editor_host
    _settle_panel_hosts(
        presenter,
        lambda: panel.editor_selections is not None
        and panel.frozen_data is not None
        and panel.frozen_data.description is not None,
    )
    assert panel.state.selector == {}
    with pytest.raises(KeyError):
        second_editor_host.selector_state(SelectorKind.AREA).result()
    with pytest.raises(KeyError):
        panel.host.selector_state(SelectorKind.AREA).result()
    _commit_area(second_editor_host)
    presenter.beat()
    assert presenter.logic[first_id].draft.values == first_current
    assert presenter.logic[second_id].draft.values != second_before


    area_draft = dict(presenter.logic[second_id].draft.values)
    _zoom_in(second_editor_host)
    presenter.beat()
    assert presenter.logic[second_id].draft.values == area_draft
    _zoom_out(second_editor_host)
    presenter.beat()
    assert presenter.logic[second_id].draft.values == area_draft

    second_editor_host.remove_selector(SelectorKind.AREA).result()
    _settle_panel_hosts(presenter, lambda: panel.state.selector == {})
    # The panel's own record holds it now, so a removed selector is a
    # panel that has none written down -- and a saved board has none either.
    assert panel.state.selector == {}

    # Zoom and pan are how an operator looks, not what they ask for: with no
    # region drawn, moving the view leaves the producer exactly where it was.
    before_zoom = dict(presenter.logic[second_id].draft.values)
    _zoom_in(second_editor_host)
    for _ in range(20):
        presenter.beat()
        time.sleep(0.005)
    assert panel.editor_selections.last_error is None
    assert presenter.logic[second_id].draft.values == before_zoom
    _zoom_out(second_editor_host)
    for _ in range(20):
        presenter.beat()
        time.sleep(0.005)
    assert presenter.logic[second_id].draft.values == before_zoom
    # The view itself is still shared by both surfaces.
    editor_viewport = second_editor_host.describe_display().result().value.viewport
    live_viewport = panel.host.describe_display().result().value.viewport
    assert editor_viewport is not None and live_viewport == editor_viewport


def test_panel_editor_selection_survives_an_expired_history_parent(
    presenter,
    session,
    monkeypatch,
) -> None:
    """Frozen analysis remains valid after Runtime evicts its publication."""

    node_id = presenter.add_logic(
        "camera_measurement",
        node_id="expired-editor-camera",
        open_editor=False,
    )
    node, snapshot = _one_shot(session, producer=node_id)
    panel = presenter.add_panel(node.signal_key("frames"), snapshot, kind="image")
    _settle_panel_hosts(presenter, lambda: panel.host is not None)
    assert presenter.edit_panel(panel.panel_id)
    _settle_panel_hosts(
        presenter,
        lambda: panel.editor_host is not None
        and panel.frozen_data is not None
        and panel.bridge is not None,
    )
    frozen = panel.frozen_data
    assert frozen is not None and frozen.publication is not None
    draft_before = dict(presenter.logic[node_id].draft.values)
    retained = session.signal_plane.retains

    def expires_selected(signal, publication=None):
        if publication is frozen.publication:
            return False
        return retained(signal, publication)

    monkeypatch.setattr(session.signal_plane, "retains", expires_selected)
    _commit_area(panel.editor_host)
    presenter.beat()

    assert panel.state.selector
    assert presenter.logic[node_id].draft.values != draft_before
    assert panel.bridge.last_error is None
    assert session.signal_plane.freeze().value(
        f"@logic/{panel.panel_id}/roi_frame"
    ) is None


def test_pointing_a_panel_at_a_signal_that_never_published_is_refused(
    presenter, session
) -> None:
    node, snapshot = _one_shot(session)
    binding = presenter.add_panel(node.signal_key("frames"), snapshot)
    card = presenter.view.cards[0]

    presenter.update_panel_state(
        binding.panel_id, {"signal": "@logic/nobody/frames"}
    )

    assert presenter.panels[binding.panel_id].signal == node.signal_key("frames")
    assert any("has not published" in text for _severity, text in presenter.view.status)


def test_the_order_the_operator_dragged_the_cards_into_is_the_panel_order(
    presenter, session
) -> None:
    """Where the cards are IS the order.

    It decides what a saved figure contains and in what sequence, so a board
    that rearranges itself back on the next redraw ignores the operator.
    """

    node, snapshot = _one_shot(session)
    first = presenter.add_panel(node.signal_key("frames"), snapshot, title="one")
    second = presenter.add_panel(node.signal_key("frames"), snapshot, title="two")
    assert list(presenter.panels) == [first.panel_id, second.panel_id]

    presenter.view.panel_order_committed.emit((second.panel_id, first.panel_id))

    assert list(presenter.panels) == [second.panel_id, first.panel_id]
    assert list(presenter.view.panel_ids()) == [second.panel_id, first.panel_id]


def test_an_order_naming_a_panel_that_left_keeps_every_panel(presenter, session) -> None:
    """A drop that raced a Remove must not drop the panel it did not name."""

    node, snapshot = _one_shot(session)
    first = presenter.add_panel(node.signal_key("frames"), snapshot)
    second = presenter.add_panel(node.signal_key("frames"), snapshot)

    presenter.view.panel_order_committed.emit(("panel-gone", second.panel_id))

    assert set(presenter.panels) == {first.panel_id, second.panel_id}


def test_a_panel_that_could_not_draw_says_so_on_its_own_card(presenter, session) -> None:
    """A still panel means two different things.

    A render that failed was delivered to a reject() that did nothing, and the
    card's status line was never written to by anything -- so a panel that had
    stopped drawing looked exactly like a panel whose data had stopped
    arriving, and neither said which.
    """

    node, snapshot = _one_shot(session)
    binding = presenter.add_panel(node.signal_key("frames"), snapshot, title="camera")

    binding.port.reject(
        _completed_surface_update(),
        RuntimeError("the renderer refused this frame"),
    )
    presenter.beat()

    card = presenter.view.cards[0]
    assert "refused this frame" in card.status[0]
    assert card.status[1] is True
    assert any("refused this frame" in text for _severity, text in presenter.view.status)


def test_an_anonymous_render_error_is_named_after_its_class(
    presenter, session
) -> None:
    """CancelledError, a bare assert and TimeoutError all stringify to nothing.

    The strip then showed ``camera: `` with nothing after the colon -- a red
    line that named the panel and refused to say what happened.
    """

    node, snapshot = _one_shot(session)
    binding = presenter.add_panel(
        node.signal_key("frames"), snapshot, title="camera"
    )

    binding.port.reject(_completed_surface_update(), AssertionError())
    presenter.beat()

    card = presenter.view.cards[0]
    assert card.status == ("AssertionError", True)
    assert ("error", "camera: AssertionError") in presenter.view.status


def test_a_superseded_render_is_not_reported_at_all(presenter, session) -> None:
    """A cancelled render means a newer frame is already queued behind it.

    Reported red, a camera merely outpacing the render worker looked like a
    camera failing -- once per coalesced frame, with an empty message.
    """

    node, snapshot = _one_shot(session)
    binding = presenter.add_panel(
        node.signal_key("frames"), snapshot, title="camera"
    )
    presenter.view.status.clear()

    from concurrent.futures import CancelledError

    binding.port.reject(_completed_surface_update(), CancelledError())
    presenter.beat()

    assert binding.port.last_error is None
    assert presenter.view.cards[0].status == ("", False)
    assert not any(
        severity == "error" for severity, _text in presenter.view.status
    )


def test_add_panel_adds_a_panel_of_the_kind_chosen_beside_the_button(
    presenter, session
) -> None:
    """That is what the control says it does.

    It ignored the kind entirely and opened a modal signal chooser instead, so
    the combo beside Add Panel described a choice the button did not make --
    and a board where every signal was already shown opened a blank list.
    Which signal a panel shows is a per-panel decision the card's own picker
    already owns, so asking for it up front asked twice.
    """

    before = len(presenter.panels)

    binding = presenter.add_selected_panel("curve")
    assert binding is not None and binding.kind == "curve"
    assert binding.signal == "" and binding.host is None
    assert len(presenter.panels) == before + 1


def test_add_panel_before_anything_publishes_still_creates_the_panel(presenter) -> None:
    """No publication is a normal stopped-pipeline state, not an Add error."""

    binding = presenter.add_selected_panel("image")
    assert binding is not None
    assert binding.signal == ""
    assert binding.host is None


def test_a_panel_keeps_the_kind_it_was_added_as(presenter, session) -> None:
    """An operator who asked for a curve did not ask for it to become an image
    the moment they point the card somewhere else."""

    node, snapshot = _one_shot(session)
    binding = presenter.add_selected_panel("curve")
    assert binding is not None and binding.kind == "curve"

    original_state = binding.state
    assert presenter.update_panel_state(binding.panel_id, {"kind": "image"}) is False
    assert binding.state is original_state

    other, _other_snapshot = _one_shot(session, producer="cm2")
    assert presenter.update_panel_state(
        binding.panel_id, {"signal": other.signal_key("frames")}
    ) is True
    assert presenter.panels[binding.panel_id].kind == "curve"


def test_panel_edit_surface_comes_from_the_current_plot_host(presenter, session) -> None:
    """The editor sees plot-declared choices/bounds even with no authored overrides."""

    node, snapshot = _one_shot(session)
    binding = presenter.add_panel(
        node.signal_key("frames"), snapshot, kind="image"
    )
    _settle_panel_hosts(
        presenter,
        lambda: not binding.parameter_surface.get("semantic_provisional")
        and not binding.parameter_surface["semantic_unavailable"],
    )

    surface = binding.parameter_surface
    assert presenter.view.panel_parameter_surfaces[binding.panel_id] is surface
    semantic = {field["key"]: field for field in surface["semantic"]}
    display = {field["key"]: field for field in surface["display"]}
    assert "kind" not in semantic, "plot kind is fixed at Add Panel"
    # The edit surface is the fate table: one row per axis of the dataset,
    # plus how the axes nobody drew along are collapsed.
    assert "reduction" in semantic
    fate_rows = {key for key in semantic if key.startswith("fate:")}
    assert fate_rows, semantic
    assert {"x", "y"} <= {
        value for key in fate_rows for _label, value in semantic[key]["choices"]
    }
    assert all(
        semantic[key]["choices"] and semantic[key]["value"] is not None
        for key in fate_rows
    )
    assert display["colormap"]["kind"] == "choice"
    assert display["colormap"]["choices"]
    assert display["title"]["allow_none"] is True
    assert display["title"]["value"] is None
    assert display["x_label"]["allow_none"] is True
    assert display["x_label"]["value"] is None
    assert display["color_min"]["allow_none"] is True
    assert display["color_min"]["value"] is None
    assert display["show_colorbar"]["kind"] == "boolean"
    assert "site_overlay" not in display
    assert "site_overlay" not in surface
    fit_choices = dict(surface["fit"][0]["choices"])
    assert "anisotropic_gaussian_center" in fit_choices.values()

    presenter.edit_panel(binding.panel_id)
    _settle_panel_hosts(presenter, lambda: binding.editor_host is not None)
    projection = presenter.view.panel_editors[binding.panel_id]
    assert projection["parameter_surface"] is binding.parameter_surface
    assert binding.editor_host is not None
    assert binding.editor_host is not binding.host
    assert presenter.view.panel_editor_surfaces[binding.panel_id] is binding.editor_host

    colormap = next(
        value
        for _label, value in display["colormap"]["choices"]
        if value != display["colormap"]["value"]
    )
    assert presenter.update_panel_state(
        binding.panel_id, {"display": {"colormap": colormap}}
    )
    _settle_panel_hosts(
        presenter,
        lambda: (
            binding.configuration is None
            and binding.editor_configuration is None
        ),
    )
    assert binding.state.display["colormap"] == colormap
    description = _operation_value(binding.host.describe_display())
    assert description.display_state.values["colormap"] == colormap
    editor_description = _operation_value(
        binding.editor_host.describe_display()
    )
    assert editor_description.display_state.values["colormap"] == colormap

    reduction = next(
        value
        for _label, value in semantic["reduction"]["choices"]
        if value != semantic["reduction"]["value"]
    )
    assert presenter.update_panel_state(
        binding.panel_id, {"semantic": {"reduction": reduction}}
    )
    _settle_panel_hosts(
        presenter,
        lambda: binding.configuration is None
        and next(
            field["value"]
            for field in binding.parameter_surface["semantic"]
            if field["key"] == "reduction"
        )
        is reduction,
    )
    assert binding.state.semantic["reduction"] is reduction
    refreshed = {
        field["key"]: field for field in binding.parameter_surface["semantic"]
    }
    assert refreshed["reduction"]["value"] is reduction


def test_a_board_can_be_written_down_and_put_back(presenter, session, tmp_path) -> None:
    """An arrangement that took an afternoon has to survive the window.

    Saving DATA already had a button and is a different act: it writes the
    numbers on the board.  What an operator actually builds is the board --
    which signals, drawn as what, how big, redrawing how often -- and nothing
    here could say what that was, so it went when the window did.
    """

    node, snapshot = _one_shot(session)
    signal = node.signal_key("frames")
    first = presenter.add_panel(signal, snapshot, title="camera", kind="image")
    _settle_panel_hosts(
        presenter,
        lambda: not first.parameter_surface.get("semantic_provisional")
        and not first.parameter_surface["semantic_unavailable"],
    )
    # One row per axis, and the row that says an axis is drawn along y is as
    # much of the board as a title is.
    semantic_field = next(
        field
        for field in first.parameter_surface["semantic"]
        if str(field["key"]).startswith("fate:") and field["value"] == "y"
    )
    semantic_key = semantic_field["key"]
    semantic_value = semantic_field["value"]
    presenter.update_panel_state(first.panel_id, {"size": "4x4"})
    presenter.update_panel_state(first.panel_id, {"interval_ms": 800})
    presenter.update_panel_state(
        first.panel_id,
        {
            "semantic": {semantic_key: semantic_value},
            "display": {"show_colorbar": False},
            "fit": {"model": "radial_gaussian_center"},
            "overlay_signal": "",
        },
    )
    second = presenter.add_panel(signal, snapshot, title="again", kind="image")
    logic_id = presenter.add_logic(
        "camera_measurement",
        values={"exposure_seconds": 0.037, "repeat": 0},
        device_keys={"camera": "camera"},
    )
    occupancy_id = presenter.add_logic(
        "occupancy",
        source_signal=f"@logic/{logic_id}/frames",
        artifact_inputs={"calibration_path": str(tmp_path / "chosen.json")},
    )
    assert presenter.logic[logic_id].host is None

    import json

    # An arranged board includes which rows open a plot when they start.
    presenter.set_logic_auto_preview(occupancy_id, False)
    authored_display = dict(first.state.display)
    assert authored_display["show_colorbar"] is False
    document = json.loads(json.dumps(presenter.layout()))
    assert document["format"] == presenter.LAYOUT_FORMAT
    assert [panel["title"] for panel in document["panels"]] == ["camera", "again"]
    assert document["panels"][0] == {
        "signal": signal, "title": "camera", "kind": "image",
        "cell_kind": "",
        "size": "4x4", "interval_ms": 800,
        "semantic": first.state.document()["semantic"],
        "display": authored_display,
        "fit": {"model": "radial_gaussian_center"}, "overlay_signal": "",
        "published_outputs": {},
        "selector": {},
        "crosshair": {},
        "classifier_thresholds": [],
        "focused_cell": None,
    }
    # Nothing of this session's bookkeeping: ids are minted fresh on the way in.
    assert not any("panel_id" in panel for panel in document["panels"])
    assert document["logic"] == [
        {
            "node_id": logic_id,
            "api_name": "camera_measurement",
            "values": presenter.logic[logic_id].draft.values,
            "source_signal": "",
            "device_keys": {"camera": "camera"},
            "artifact_inputs": {},
            "auto_preview": True,
        },
        {
            "node_id": occupancy_id,
            "api_name": "occupancy",
            "values": {"model_kind": "default"},
            "source_signal": f"@logic/{logic_id}/frames",
            "device_keys": {},
            "artifact_inputs": {
                "calibration_path": str(tmp_path / "chosen.json")
            },
            "auto_preview": False,
        }
    ]

    presenter.remove_panel(first.panel_id)
    presenter.remove_panel(second.panel_id)
    assert presenter.panels == {}

    presenter.view.presented_fronts.clear()
    assert presenter.apply_layout(document) is True
    restored = list(presenter.panels.values())
    deadline = time.monotonic() + 10.0
    while (
        not all(binding.host is not None for binding in restored)
        and time.monotonic() < deadline
    ):
        presenter.board.tick()
        presenter.commit_surfaces()
        time.sleep(0.005)
    assert {
        panel_id for panel_id, _front in presenter.view.presented_fronts
    } == {binding.panel_id for binding in restored}, (
        "layout panels must present inside their accepted cohorts, not on a later beat"
    )
    _settle_panel_hosts(
        presenter,
        lambda: all(
            not binding.parameter_surface["semantic_unavailable"]
            for binding in restored
        ),
    )
    assert [binding.title for binding in restored] == ["camera", "again"]
    assert restored[0].kind == "image"
    assert restored[0].size == "4x4"
    assert restored[0].port.display_interval_ms == 800
    assert (
        restored[0].state.document()["semantic"]
        == document["panels"][0]["semantic"]
    )
    assert restored[0].state.display == authored_display
    assert restored[0].state.fit == {"model": "radial_gaussian_center"}
    assert restored[0].state.overlay_signal == ""
    assert restored[0].panel_id != first.panel_id, "an id is never handed out twice"
    restored_logic = presenter.logic[logic_id]
    assert restored_logic.host is None and restored_logic.node is None
    assert restored_logic.draft.values["exposure_seconds"] == 0.037
    assert restored_logic.draft.device_keys == {"camera": "camera"}
    assert presenter.logic[occupancy_id].draft.artifact_inputs == {
        "calibration_path": str(tmp_path / "chosen.json")
    }
    assert restored_logic.auto_preview is True
    assert presenter.logic[occupancy_id].auto_preview is False


def test_a_bad_late_layout_entry_leaves_the_current_board_exactly_unchanged(
    presenter, session
) -> None:
    """The whole saved board is accepted before any current object is retired."""

    node, snapshot = _one_shot(session)
    panel = presenter.add_panel(
        node.signal_key("frames"), snapshot, title="current", kind="image"
    )
    logic_id = presenter.add_logic(
        "camera_measurement",
        values={"exposure_seconds": 0.031, "repeat": 0},
        device_keys={"camera": "camera"},
        open_editor=False,
    )
    old_logic = presenter.logic[logic_id]
    old_panel = presenter.panels[panel.panel_id]
    old_card = presenter.view.cards[0]
    old_row = presenter.view.logic_rows[0]
    old_host, old_port = old_panel.host, old_panel.port
    old_layout = presenter.layout()

    document = presenter.layout()
    document["logic"].append(
        {
            "node_id": "bad-last",
            "api_name": "not_a_registered_logic_node",
            "values": {},
            "source_signal": "",
            "device_keys": {},
            "artifact_inputs": {},
            "auto_preview": True,
        }
    )

    assert presenter.apply_layout(document) is False
    assert presenter.layout() == old_layout
    assert tuple(presenter.logic) == (logic_id,)
    assert presenter.logic[logic_id] is old_logic
    assert tuple(presenter.panels) == (panel.panel_id,)
    assert presenter.panels[panel.panel_id] is old_panel
    assert old_panel.host is old_host and old_panel.port is old_port
    assert presenter.view.logic_rows == (old_row,)
    assert presenter.view.cards == (old_card,)


def test_task_console_layout_rejects_a_non_catalog_facet_cell(presenter) -> None:
    """A cell kind must be something a grid cell CAN be; image now is.

    The catalog's curve is a default, not a constraint -- pinning it drew a
    scan of camera frames as million-point polylines -- so a saved board may
    carry image cells.  What stays refused is a kind no grid cell can host.
    """

    document = presenter.layout()
    document["panels"].append(
        {
            "signal": "",
            "title": "Report-only image facets",
            "kind": "facet_grid",
            "cell_kind": "rolling",
            "size": "4x4",
            "interval_ms": 400,
            "semantic": {},
            "display": {},
            "fit": {},
            "overlay_signal": "",
            "published_outputs": {},
            "selector": {},
            "crosshair": {},
            "classifier_thresholds": [],
            "focused_cell": None,
        }
    )

    assert presenter.apply_layout(document) is False
    assert presenter.panels == {}
    from zlc_plot import GRID_CELL_KINDS

    expected = "cell kind must be one of " + ", ".join(
        kind.value for kind in GRID_CELL_KINDS
    )
    assert any(expected in text for _severity, text in presenter.view.status)


def test_a_board_naming_a_signal_nobody_publishes_keeps_the_blank_panel(
    presenter, session
) -> None:
    """An unresolved wire is editable layout state, not a reason to drop it."""

    node, snapshot = _one_shot(session)
    signal = node.signal_key("frames")
    presenter.add_panel(signal, snapshot, title="here", kind="image")
    document = presenter.layout()
    document["panels"].append(
        {"signal": "nobody.publishes.this", "title": "gone", "kind": "image",
         "cell_kind": "", "size": "",
         "interval_ms": 200, "semantic": {}, "display": {}, "fit": {},
         "overlay_signal": "", "published_outputs": {},
         "selector": {}, "crosshair": {}, "classifier_thresholds": [], "focused_cell": None}
    )

    assert presenter.apply_layout(document) is True
    assert [binding.title for binding in presenter.panels.values()] == ["here", "gone"]
    unresolved = tuple(presenter.panels.values())[-1]
    assert unresolved.signal == "nobody.publishes.this"
    assert unresolved.host is None and unresolved.port is None
    assert any("nobody.publishes.this" in text for _severity, text in presenter.view.status)


def test_a_file_that_is_not_a_board_is_refused_by_name(presenter) -> None:
    assert presenter.apply_layout({"format": "zlc.figure"}) is False
    assert any("not a saved board" in text for _severity, text in presenter.view.status)

    invalid_layout = presenter.layout()
    invalid_layout["format"] = "not-a-console-board"
    assert presenter.apply_layout(invalid_layout) is False


def test_panel_edit_projects_the_direct_producer_link_and_ages(
    presenter, session, monkeypatch
) -> None:
    """Edit knows whose data it shows, and when that run is over.

    Its picture is one frozen revision; a later RUN of the same signal leaves
    it describing an experiment the bench no longer holds -- its fit solved
    against gone data, its Save still armed -- so the projection says so
    until Refresh.
    """

    node_id = presenter.add_logic("camera_measurement")
    node, snapshot = _one_shot(session, producer=node_id)
    panel = presenter.add_panel(node.signal_key("frames"), snapshot, kind="image")
    _settle_panel_hosts(
        presenter,
        lambda: panel.accepted_surface is not None
        and panel.parameter_surface.get("semantic_provisional") is False,
    )

    assert presenter.edit_panel(panel.panel_id) is True
    _settle_panel_hosts(presenter, lambda: panel.frozen_data is not None)
    projection = presenter.view.panel_editors[panel.panel_id]
    assert presenter.view.panel_editor_update_count.get(panel.panel_id, 0) == 0
    assert projection["producer_node_id"] == node_id
    assert projection["stale"] is False
    assert projection["data_advanced"] is False

    previous = panel.frozen_data
    assert previous is not None
    _one_shot(session, producer=node_id)
    latest = session.signal_plane.latest_publication(panel.state.signal)
    assert latest is not None and latest is not previous.publication
    current = latest.value(panel.state.signal)
    assert current is not None
    latest_front = SimpleNamespace(
        value=lambda _name: current,
        publication=lambda _name: latest,
    )
    with monkeypatch.context() as blocked:
        blocked.setattr(presenter.board, "tick", lambda: None)
        blocked.setattr(session.signal_plane, "freeze", lambda: latest_front)
        assert presenter.refresh_panel_snapshot(panel.panel_id)
        assert panel.frozen_data is previous, (
            "Refresh must keep the last accepted Edit/Save snapshot until its "
            "replacement is accepted"
        )
    _settle_panel_hosts(
        presenter,
        lambda: panel.frozen_data is not previous
        and presenter.view.panel_editors[panel.panel_id]["stale"] is False
        and presenter.view.panel_editors[panel.panel_id]["data_advanced"] is False,
    )
    assert presenter.view.panel_editors[panel.panel_id]["stale"] is False
    assert presenter.view.panel_editors[panel.panel_id]["data_advanced"] is False

    presenter.view.logic_edit_requested.emit(node_id)
    assert presenter.view.focused_logic_editor == node_id


def test_a_running_task_freezes_logic_identity_but_not_panels(
    presenter, monkeypatch
) -> None:
    """What a Task refuses is what it actually owns.

    It holds its devices exclusively, so nothing else may START, and its run
    is the draft it started from, so that row may not be re-drafted.  The
    window is not its property: an operator watching a run must be able to
    open a panel for it, and used to be told to stop the run first.
    """

    from zlc_runtime import DatasetOutputDeclaration
    from zlc_runtime.host import LogicNodeObservation, NodeProgress
    from zlc_atom.nodes import NodePreviewSpec, calibration_pulse_template_bytes
    from zlc_workbench.logic import LogicCandidate, stable_signal_key

    (presenter.session.workspace.pulses / "imaging_template.json").write_bytes(
        calibration_pulse_template_bytes()
    )

    class TaskHost:
        def __init__(self) -> None:
            self.instance_id = "calibration"
            self.generation = "calibration-test-generation"
            self.dataset_output_declarations = ()
            self.running = False
            self.terminal = False
            self.final_result_resolved = False
            self.cancelled = False
            self.fail = False
            self.observation = LogicNodeObservation(False, False, "starting")

        @property
        def cancel_requested(self) -> bool:
            return self.cancelled

        def start(self) -> None:
            self.cancelled = False
            self.fail = False
            self.running = True
            self.observation = LogicNodeObservation(
                True,
                False,
                "running",
                progress=NodeProgress("Capturing", 2, 5),
            )

        def cancel(self, _reason: str) -> None:
            self.cancelled = True
            self.observation = LogicNodeObservation(
                True,
                False,
                "stopping",
                progress=NodeProgress("Capturing", 2, 5),
            )

        def poll(self):
            if self.fail:
                self.running = False
                self.terminal = True
                self.observation = LogicNodeObservation(
                    False, True, "failed", error="camera fault"
                )
            elif self.cancelled:
                self.running = False
                self.terminal = True
                self.observation = LogicNodeObservation(
                    False, True, "cancelled"
                )
            return self.observation

        def published_signals(self) -> tuple[str, ...]:
            return ("@logic/calibration/capture_preview",)

        @property
        def operator_request(self):
            return None

        def shutdown(self) -> None:
            self.running = False

    task_id = presenter.add_logic("calibration", open_editor=False)
    other_id = presenter.add_logic("camera_measurement", open_editor=False)
    panel = presenter.add_blank_panel("image")
    assert task_id and other_id and panel is not None
    host = TaskHost()
    monkeypatch.setattr(
        presenter,
        "_build_logic_candidate",
        lambda binding, _finalization: LogicCandidate(
            object(),
            host,
            tuple(binding.descriptor.node_previews),
        ),
    )

    presenter.view.logic_start_requested.emit(task_id)
    assert presenter.view.task_takeover is True
    assert presenter._active_task_id == task_id
    assert presenter.view.status[-1] == ("task", "calibration: Capturing 2/5")
    primary = DatasetOutputDeclaration("frames", "camera.frames")
    overlay = DatasetOutputDeclaration("occupied", "atom.occupied")
    presenter.logic[task_id].preview_specs = (
        NodePreviewSpec(
            primary,
            "image",
            producer="camera",
            overlay=overlay,
        ),
    )
    protected = presenter._task_protected_signals()
    assert stable_signal_key(f"{task_id}/camera", primary.name) in protected
    assert stable_signal_key(f"{task_id}/camera", overlay.name) in protected

    previous_repeat = presenter.logic[other_id].draft.values["repeat"]
    # Panel monitoring remains usable, while the Logic graph and every draft
    # keep the identity with which the Task took the bench.
    presenter.view.add_panel_requested.emit("curve")
    presenter.view.add_logic_requested.emit("occupancy")
    presenter.view.panel_remove_requested.emit(panel.panel_id)
    presenter.view.logic_draft_changed.emit(other_id, {"values": {"repeat": 3}})
    assert panel.panel_id not in presenter.panels, "a panel could not be closed"
    assert len(presenter.panels) == 1, "a panel could not be opened"
    assert "occupancy" not in " ".join(presenter.logic)
    assert presenter.logic[other_id].draft.values["repeat"] == previous_repeat

    preview = presenter.add_blank_panel(
        "facet_grid",
        signal="@logic/calibration/capture_preview",
    )
    assert preview is not None
    assert presenter.update_panel_state(
        preview.panel_id,
        {"title": "Still monitorable", "display": {"colormap": "viridis"}},
    )
    assert not presenter.update_panel_state(
        preview.panel_id,
        {"semantic": {"fate:repeat": "reduce"}},
    )
    assert not presenter.update_panel_state(
        preview.panel_id,
        {"signal": "another/signal"},
    )
    assert not presenter.update_panel_published_outputs(
        preview.panel_id,
        {"roi": True},
    )
    presenter.set_deriving(True)
    assert presenter.view._cards[preview.panel_id].selectors_enabled is True
    viewport = object()
    presenter._synchronize_panel_interaction(
        preview,
        None,
        None,
        viewport,
    )
    assert preview.interaction_viewport[1] is viewport
    preview.port = None

    # The bench does not: nothing else may take the devices, and the Task's
    # own draft is the run it is already performing.
    presenter.view.logic_start_requested.emit(other_id)
    assert presenter.logic[other_id].host is None
    assert (
        presenter.update_logic_draft(task_id, values={"repeats": 99}) is False
    ), "the running Task's own draft is the run it started"

    presenter.view.stop_task_requested.emit()
    assert host.cancelled is True
    assert presenter.view.task_takeover is True
    presenter.set_paused(True)
    presenter.beat()
    assert presenter._active_task_id is None
    assert presenter.view.task_takeover is False
    assert presenter.view.status[-1] == ("task", "calibration: cancelled")

    presenter.view.logic_start_requested.emit(task_id)
    assert presenter.view.task_takeover is True
    host.fail = True
    presenter.beat()
    assert presenter._active_task_id is None
    assert presenter.view.task_takeover is False
    assert presenter.view.status[-1] == ("error", "calibration: camera fault")
    assert presenter.logic_editor_projection(task_id)["error"] == "camera fault"


def test_running_row_waits_for_first_publication_and_terminal_phase_wins(
    presenter,
) -> None:
    from types import SimpleNamespace
    from zlc_runtime.host import LogicNodeObservation, NodeProgress
    from zlc_workbench.logic import stable_signal_key

    node_id = presenter.add_logic("camera_measurement", open_editor=False)
    binding = presenter.logic[node_id]
    signal = stable_signal_key(node_id, "frames")
    binding.host = SimpleNamespace(
        running=False,
        observation=LogicNodeObservation(True, False, "running"),
        published_signals=lambda: (signal,),
        dataset_output_declarations=binding.descriptor.outputs,
    )
    binding.shown = ()
    try:
        presenter._show_logic(binding)
    finally:
        binding.host = None
    row = next(row for row in presenter.view.logic_rows if row.title == node_id)
    assert row.publishes == (("frames", "—", f"waiting · {signal}"),)

    terminal = LogicNodeObservation(
        False,
        True,
        "done",
        progress=NodeProgress("Saving stale progress"),
    )
    assert presenter._observation_status(terminal) == "done"


def test_incompatible_preview_reports_once_and_is_never_marked_successful(
    presenter,
    session,
) -> None:
    from types import SimpleNamespace
    from zlc_atom.nodes import NodePreviewSpec
    from zlc_atom.nodes.camera_measurement.measurement import CAMERA_FRAMES_OUTPUT

    node_id = "preview-task"
    presenter.add_logic(
        "camera_measurement",
        node_id=node_id,
        open_editor=False,
    )
    binding = presenter.logic[node_id]
    binding.host = SimpleNamespace(running=True)
    binding.preview_specs = (
        NodePreviewSpec(CAMERA_FRAMES_OUTPUT, "pulse_timeline"),
    )
    _one_shot(session, producer=node_id)

    presenter._ensure_node_previews(binding)
    presenter._ensure_node_previews(binding)
    binding.host = None
    errors = [
        text
        for severity, text in presenter.view.status
        if severity == "error" and "incompatible" in text
    ]
    assert len(errors) == 1
    assert binding.previewed == ()
    assert not presenter.panels


def test_finite_repeat_mount_and_axis_change_never_materialize_on_owner(
    presenter,
    session,
    monkeypatch,
) -> None:
    from threading import Event, Thread, get_ident

    session.load_pulse(PULSE_NAME)
    node = CameraMeasurementNode(
        camera=session.camera,
        request=CameraMeasurementRequest(
            "camera",
            0.02,
            None,
            30,
            CAMERA_WINDOWS,
        ),
        signal_plane=session.signal_plane,
        producer="repeat-camera",
    )
    first_commit = Event()
    capture_release = Event()
    original_commit = session.signal_plane.commit_live
    camera_commits = 0

    def gated_commit(producer, outputs):
        nonlocal camera_commits
        committed = original_commit(producer, outputs)
        if producer is node:
            camera_commits += 1
            if camera_commits == 1:
                first_commit.set()
                assert capture_release.wait(10.0)
        return committed

    monkeypatch.setattr(session.signal_plane, "commit_live", gated_commit)
    capture = node.prepare()
    session.fire(shots=30)
    collected: list[object] = []
    capture_thread = Thread(target=lambda: collected.append(capture.collect()))
    capture_thread.start()
    assert first_commit.wait(5.0)
    signal = node.signal_key("frames")
    publication = session.signal_plane.latest_publication(signal)
    assert publication is not None
    event = publication.value(signal)
    assert event is not None
    assert event.snapshot.block.values.shape[0] == 1
    assert event.canonical_schema is not None
    assert event.canonical_schema.repeat_axis.size == 30
    description = next(
        item
        for item in session.signal_plane.describe_signals()
        if item.name == signal
    )
    assert description.shape[:2] == (30, CAMERA_WINDOWS)

    original = session.signal_plane.current_dataset_view
    entered = Event()
    projection_release = Event()
    projection_threads: list[int] = []

    def blocked_current(name, publication=None):
        projection_threads.append(get_ident())
        entered.set()
        assert projection_release.wait(10.0)
        return original(name, publication)

    monkeypatch.setattr(
        session.signal_plane,
        "current_dataset_view",
        blocked_current,
    )
    binding = presenter.add_blank_panel("facet_grid")
    assert presenter.update_panel_state(binding.panel_id, {"signal": signal})
    presenter.beat()

    assert entered.wait(5.0)
    assert binding.host is None
    assert projection_threads and projection_threads[0] != get_ident()
    projection_release.set()
    _settle_panel_hosts(
        presenter,
        lambda: binding.host is not None
        and bool(binding.parameter_surface.get("semantic")),
    )
    shown = _accepted(binding.port, "plot_input")
    snapshot = getattr(shown, "snapshot", shown)
    assert snapshot.block.values.shape[:2] == (30, CAMERA_WINDOWS)
    assert np.all(snapshot.expanded_validity()[:1])
    assert not np.any(snapshot.expanded_validity()[1:])

    repeat_row = next(
        field
        for field in binding.parameter_surface["semantic"]
        if str(field["key"]) == "fate:repeat"
    )
    next_fate = next(
        value
        for _label, value in repeat_row["choices"]
        if value != repeat_row["value"]
    )
    publication = binding.display_publication
    host = binding.host
    assert presenter.update_panel_state(
        binding.panel_id,
        {"semantic": {"fate:repeat": next_fate}},
    )
    _settle_panel_hosts(
        presenter,
        lambda: binding.configuration is None
        and binding.state.semantic.get("fate:repeat") == next_fate,
    )
    assert binding.host is host
    assert binding.display_publication is publication
    assert _accepted(binding.port, "plot_input") is shown

    capture_release.set()
    capture_thread.join(10.0)
    assert not capture_thread.is_alive() and collected
    latest = session.signal_plane.latest_publication(signal)
    _settle_panel_hosts(
        presenter,
        lambda: _accepted(binding.port, "publication") is latest,
    )
    finished = _accepted(binding.port, "plot_input")
    finished_snapshot = getattr(finished, "snapshot", finished)
    assert finished_snapshot.block.values.shape[:2] == (30, CAMERA_WINDOWS)
    assert np.all(finished_snapshot.expanded_validity())


def test_partial_grid_points_mount_and_reproject_one_canonical_snapshot(
    presenter,
    session,
    tmp_path,
) -> None:
    from types import SimpleNamespace
    from zlc_data import (
        REPEAT,
        SCAN_POINT,
        AxisId,
        AxisSpec,
        BlockId,
        CellValidity,
        DataBlock,
        DatasetRevision,
        DatasetSchema,
        GridTopology,
        OwnedSnapshot,
        PointColumn,
        PointTable,
        StreamGenerationId,
        ValueSchema,
    )
    from zlc_data.figure_archive import read_archive, read_dataset
    from zlc_runtime.dataset import DatasetCoverage
    from zlc_runtime.dataset_output import (
        DatasetOutputDeclaration,
        LiveDatasetOutput,
    )

    declaration = DatasetOutputDeclaration("frame", "test.grid.frame")
    signal = "grid-scan/frame"
    node = SimpleNamespace(
        instance_id="grid-scan",
        dataset_output_declarations=(declaration,),
        signal_key=lambda name: f"grid-scan/{name}",
    )
    repeat = AxisSpec(AxisId("grid.repeat"), "repeat", REPEAT, 1, (0,))
    cell = ValueSchema.scalar(np.dtype("float64"), None)
    x_id = AxisId("grid.x")
    y_id = AxisId("grid.y")
    canonical = DatasetSchema(
        repeat,
        PointTable(
            4,
            (
                PointColumn(
                    x_id,
                    "x",
                    SCAN_POINT,
                    PointColumn.NUMERIC,
                    (0.0, 1.0, 0.0, 1.0),
                ),
                PointColumn(
                    y_id,
                    "y",
                    SCAN_POINT,
                    PointColumn.NUMERIC,
                    (0.0, 0.0, 1.0, 1.0),
                ),
            ),
        ),
        GridTopology(
            (x_id, y_id),
            ((0.0, 1.0), (0.0, 1.0)),
            ((0, 0), (1, 0), (0, 1), (1, 1)),
        ),
        cell,
    )
    session.signal_plane.begin_generation(node)

    def commit_point(index: int, measured: float) -> None:
        point = PointColumn(
            AxisId("event.point"),
            "point",
            SCAN_POINT,
            PointColumn.NUMERIC,
            (0.0,),
        )
        event_schema = DatasetSchema(
            repeat,
            PointTable(1, (point,)),
            None,
            cell,
        )
        block = DataBlock(
            BlockId(f"grid-event-{index}"),
            DatasetRevision(index + 1),
            np.asarray([[[measured]]], dtype=np.float64),
            CellValidity(np.ones((1, 1), dtype=np.bool_)),
            event_schema,
        )
        event = OwnedSnapshot(
            block.ref(StreamGenerationId("grid-plugin")),
            block,
        )
        session.signal_plane.commit_live(
            node,
            {
                "frame": LiveDatasetOutput(
                    declaration,
                    event,
                    DatasetCoverage(index + 1, 4),
                    canonical_schema=canonical,
                    cell_origin=(0, index),
                )
            },
        )

    commit_point(0, 10.0)
    front = session.signal_plane.freeze()
    value = front.value(signal)
    assert value is not None
    assert value.snapshot.block.schema.grid_topology is None
    assert value.snapshot.block.schema.point_table.row_count == 1

    binding = presenter.add_blank_panel("image")
    assert presenter.update_panel_state(binding.panel_id, {"signal": signal})
    _settle_panel_hosts(
        presenter,
        lambda: binding.host is not None
        and bool(binding.parameter_surface.get("semantic")),
    )
    shown = _accepted(binding.port, "plot_input")
    snapshot = getattr(shown, "snapshot", shown)
    assert snapshot.block.schema.grid_topology == canonical.grid_topology
    assert snapshot.block.values.shape == (1, 4, 1)
    validity = snapshot.expanded_validity()
    assert np.all(validity[:, :1])
    assert not np.any(validity[:, 1:])
    assert presenter._shown_snapshot(binding) is snapshot
    assert binding.parameter_surface["data_structure"]

    fate = {
        field["value"]: field
        for field in binding.parameter_surface["semantic"]
        if str(field["key"]).startswith("fate:")
        and field["value"] in {"x", "y"}
    }
    assert set(fate) == {"x", "y"}
    publication = binding.display_publication
    host = binding.host
    assert presenter.update_panel_state(
        binding.panel_id,
        {
            "semantic": {
                str(fate["x"]["key"]): "y",
                str(fate["y"]["key"]): "x",
            }
        },
    )
    _settle_panel_hosts(
        presenter,
        lambda: binding.configuration is None
        and binding.state.semantic.get(str(fate["x"]["key"])) == "y"
        and binding.state.semantic.get(str(fate["y"]["key"])) == "x",
    )
    assert binding.host is host
    assert binding.display_publication is publication
    assert _accepted(binding.port, "plot_input") is shown

    commit_point(1, 20.0)
    latest = session.signal_plane.latest_publication(signal)
    _settle_panel_hosts(
        presenter,
        lambda: _accepted(binding.port, "publication") is latest,
    )
    updated = _accepted(binding.port, "plot_input")
    updated_snapshot = getattr(updated, "snapshot", updated)
    assert binding.host is host
    assert updated_snapshot.block.schema.grid_topology == canonical.grid_topology
    assert np.all(updated_snapshot.expanded_validity()[:, :2])
    assert not np.any(updated_snapshot.expanded_validity()[:, 2:])

    assert presenter.edit_panel(binding.panel_id)
    assert presenter.refresh_panel_snapshot(binding.panel_id)
    assert binding.frozen_data is not None
    assert binding.frozen_data.snapshot is updated_snapshot
    assert presenter.save_panel_figure(
        binding.panel_id,
        str(tmp_path / "partial-grid.png"),
    ) is True
    archive = tmp_path / "partial-grid.npz"
    _wait_for_panel_save(presenter, archive)
    info, arrays = read_archive(archive)
    saved = read_dataset(info, arrays, "data")
    assert saved.block.schema == updated_snapshot.block.schema
    np.testing.assert_array_equal(saved.block.values, updated_snapshot.block.values)
    np.testing.assert_array_equal(
        saved.expanded_validity(),
        updated_snapshot.expanded_validity(),
    )


def test_task_terminal_removes_only_its_auto_previews(
    presenter,
    session,
) -> None:
    node, snapshot = _one_shot(session, producer="sealed-preview")
    sealed_signal = node.signal_key("frames")
    retained = presenter.add_panel(sealed_signal, snapshot)
    missing = presenter.add_blank_panel(
        "image",
        signal="@logic/task/retired-preview",
    )
    assert missing is not None
    manual = presenter.add_blank_panel(
        "image",
        signal="@logic/task/manual-panel",
    )
    assert manual is not None
    task_id = presenter.add_logic("calibration", open_editor=False)
    task = presenter.logic[task_id]
    presenter._auto_task_previews[task_id] = {
        retained.panel_id: sealed_signal,
        missing.panel_id: "@logic/task/retired-preview",
    }

    presenter._reconcile_task_previews(task)
    assert retained.panel_id not in presenter.panels
    assert missing.panel_id not in presenter.panels
    assert manual.panel_id in presenter.panels


def test_a_facet_grid_panel_of_frames_carries_the_occupancy_overlay(
    presenter, session, tmp_path, monkeypatch
) -> None:
    """The kind gates test the SEMANTIC surface, not the outer plot kind.

    A FacetGrid is a layout whose CELLS are the images, so a grid of a
    camera cycle's frames must be able to select and draw the site overlay.
    Gating on ``kind == "image"`` is what made the only picture the overlay
    exists for -- frame_0 | frame_1 side by side -- the one that could not
    have it.
    """

    import numpy as np
    from zlc_atom.nodes.calibration import (
        FrameContract,
        ReadoutModel,
        ReadoutModelKind,
        SiteMap,
        TrapCalibration,
    )
    from zlc_atom.nodes import NodePreviewSpec
    from zlc_plot.primitives import ImageFrame
    from zlc_workbench.logic import stable_signal_key

    camera_node, _snapshot = _one_shot(session, producer="camera_measurement")
    frames_signal = camera_node.signal_key("frames")
    site_ids = ("site-0", "site-1")
    calibration_path = tmp_path / "facet-calibration.json"
    TrapCalibration(
        SiteMap(
            site_ids,
            np.asarray(((12.0, 10.0), (30.0, 20.0))),
            np.asarray((True, False)),
            np.asarray((1.0, 0.0)),
        ),
        (
            ReadoutModel(
                site_ids,
                np.asarray((-1.0e20, 0.0)),
                np.zeros(2),
                np.ones(2),
                np.asarray((True, True)),
                np.asarray((1.0, 1.0)),
            ),
        ),
        ReadoutModelKind.BOX,
        FrameContract((96, 128)),
    ).save(calibration_path)

    occupancy_id = presenter.add_logic(
        "occupancy",
        node_id="occupancy",
        artifact_inputs={"calibration_path": str(calibration_path)},
        source_signal=frames_signal,
        open_editor=False,
    )
    assert presenter.start_logic(occupancy_id) is True
    deadline = time.monotonic() + 10.0
    while presenter.logic[occupancy_id].host.running and time.monotonic() < deadline:
        presenter.poll_logic()
        time.sleep(0.005)
    presenter.poll_logic()

    judged_signal = stable_signal_key("occupancy", "frame_judged")
    status_signal = stable_signal_key("occupancy", "occupied")
    front = session.signal_plane.freeze()
    judged = front.value(judged_signal)
    publication = front.publication(judged_signal)
    status_publication = front.publication(status_signal)
    assert (
        judged is not None
        and publication is not None
        and status_publication is not None
    ), presenter.logic[occupancy_id].host.observation

    other_camera, _ = _one_shot(session, producer="other-camera")
    other_id = presenter.add_logic(
        "occupancy",
        node_id="other-occupancy",
        artifact_inputs={"calibration_path": str(calibration_path)},
        source_signal=other_camera.signal_key("frames"),
        open_editor=False,
    )
    assert presenter.start_logic(other_id)
    deadline = time.monotonic() + 10.0
    while presenter.logic[other_id].host.running and time.monotonic() < deadline:
        presenter.poll_logic()
        time.sleep(0.005)
    presenter.poll_logic()
    wrong_status = stable_signal_key("other-occupancy", "occupied")
    offered = {
        name
        for _producer, leaves in presenter.overlay_signal_groups(
            judged_signal, publication
        )
        for _label, name in leaves
    }
    assert status_signal in offered
    assert wrong_status not in offered
    wrong_publication = session.signal_plane.latest_publication(wrong_status)
    assert wrong_publication is not None
    wrong_front = SimpleNamespace(
        publication=lambda name: (
            wrong_publication if name == wrong_status else publication
        )
    )
    with pytest.raises(ValueError, match="not from the image shot"):
        presenter._image_point_overlay(
            wrong_front,
            publication,
            wrong_status,
            judged.snapshot,
            1,
        )

    logic_binding = presenter.logic[occupancy_id]
    outputs = {output.name: output for output in logic_binding.descriptor.outputs}
    logic_binding.preview_specs = (
        NodePreviewSpec(
            outputs["frame_judged"],
            "facet_grid",
            overlay=outputs["occupied"],
        ),
    )
    original_freeze = session.signal_plane.freeze
    primary_only = SimpleNamespace(
        value=lambda name: None if name == status_signal else front.value(name),
        publication=front.publication,
    )
    monkeypatch.setattr(session.signal_plane, "freeze", lambda: primary_only)
    presenter._ensure_node_previews(logic_binding)
    assert not presenter.panels, "auto-preview waits for its declared overlay"

    monkeypatch.setattr(session.signal_plane, "freeze", lambda: front)
    presenter._ensure_node_previews(logic_binding)
    binding = next(iter(presenter.panels.values()))
    monkeypatch.setattr(session.signal_plane, "freeze", original_freeze)
    _settle_panel_hosts(presenter, lambda: binding.frozen_data is not None)

    assert binding.state.kind == "facet_grid"
    assert binding.state.overlay_signal == status_signal
    frame = binding.frozen_data.plot_input
    assert isinstance(frame, ImageFrame), frame
    spec = task_console_fitting_spec(
        frame.snapshot.block.schema,
        "facet_grid",
        "",
    )
    assert all(
        frame.overlay.statuses_for(spec, float(value)) is not None
        for value in range(CAMERA_WINDOWS)
    )
    assert frame.overlay.point_ids == site_ids
    assert binding.frozen_data.overlay == {"overlay_signal": status_signal}
    lineage = binding.frozen_data.lineage
    nodes = {node["id"]: node for node in lineage["nodes"]}
    saved_events = {
        (
            node["event"]["stream"],
            node["event"]["generation"],
            node["event"]["sequence"],
        )
        for node in nodes.values()
    }

    def event_identity(selected) -> tuple[str, str, int]:
        ref = selected.event_ref
        return ref.stream_id.value, ref.generation.value, ref.sequence

    assert event_identity(publication) in saved_events
    assert event_identity(status_publication) in saved_events
    root_event = nodes[lineage["root"]]["event"]
    assert (
        root_event["stream"],
        root_event["generation"],
        root_event["sequence"],
    ) == event_identity(status_publication)


def test_a_card_stops_wearing_an_error_once_the_panel_has_drawn_again(
    presenter, session
) -> None:
    """The dot is the panel's condition now, not a log of what once happened.

    A batch abandoned for a sibling's sake, a refused gesture, a rebuilt
    host: the panel draws again on the next shot, and the mark stayed on
    forever -- an operator reading a red card had no way to tell a broken
    panel from one that stumbled once ten minutes ago.
    """

    node, snapshot = _one_shot(session)
    binding = presenter.add_panel(node.signal_key("frames"), snapshot)
    _settle_panel_hosts(presenter, lambda: binding.host is not None)
    card = presenter.view.cards[0]

    binding.port.last_error = RuntimeError("the renderer refused this frame")
    presenter.beat()
    assert card.status == ("the renderer refused this frame", True)

    binding.port.last_error = None
    presenter.beat()
    assert card.status == ("", False), "a healed panel keeps no mark"


@pytest.mark.parametrize("wanted", (True, False))
def test_a_started_row_opens_its_declared_preview_only_when_asked_to(
    presenter, session, wanted
) -> None:
    """WHICH panel is the node's declaration; WHETHER is the operator's.

    A measurement owns the shot it takes, so it names the output an operator
    started it to watch.  The switch beside Start says whether that happens
    at all -- a preference about this board, which is why it lives on the row
    and in the layout and not in the authoring schema a notebook also drives.
    """

    from zlc_workbench.logic import stable_signal_key

    camera_id = presenter.add_logic(
        "camera_measurement",
        node_id="monitor",
        values={
            "exposure_seconds": 0.02,
            "repeat": 0,
            "frames_per_cycle": CAMERA_WINDOWS,
        },
        device_keys={"camera": "camera"},
        open_editor=False,
    )
    presenter.set_logic_auto_preview(camera_id, wanted)
    row = next(
        row for row in presenter.view.logic_rows if row.title == camera_id
    )
    assert row.auto_preview is wanted, "the row shows the stored preference"

    binding = presenter.logic[camera_id]
    declared = tuple(
        (spec.output.name, spec.plot_kind)
        for spec in binding.descriptor.node_previews
    )
    assert declared == (("frames", "facet_grid"),), (
        "a measurement names what Start shows, and how: three frames per "
        "cycle are three pictures, never one average of them"
    )

    session.load_pulse(PULSE_NAME)
    assert presenter.start_logic(camera_id) is True
    frames_signal = stable_signal_key(camera_id, "frames")
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline:
        presenter.beat()
        if session.signal_plane.freeze().value(frames_signal) is not None:
            break
        session.fire(shots=1)
        time.sleep(0.01)
    presenter.poll_logic()

    opened = [
        binding
        for binding in presenter.panels.values()
        if binding.state.signal == frames_signal
    ]
    if wanted:
        assert len(opened) == 1, "the declared preview is the panel that opened"
        _settle_panel_hosts(presenter, lambda: opened[0].host is not None)
        assert opened[0].port is not None, "and it is wired to that signal"
    else:
        assert not presenter.panels, "nothing opens itself when the row says no"


def _semantic_choice(binding, name: str):
    """One offered value for a semantic field, as the Setting form offers it."""

    for entry in binding.parameter_surface["semantic"]:
        if str(entry["key"]) != name:
            continue
        for _label, value in tuple(entry["choices"]):
            if value is not None:
                return value
    return None


def _fate_row_offering(binding, fate: str) -> str | None:
    """The name of an axis row that can be given this fate."""

    for entry in binding.parameter_surface["semantic"]:
        key = str(entry["key"])
        if not key.startswith("fate:"):
            continue
        if any(value == fate for _label, value in tuple(entry["choices"])):
            return key
    return None


def test_a_cell_kind_change_is_not_refused_by_the_previous_kinds_assignments(
    presenter, session
) -> None:
    """The reported gesture: curve, assign Group, then switch to histogram.

    A panel's records are the complete assignment of whatever vocabulary it
    last settled under.  Crossing vocabularies is legal -- that is what the
    cell kind control is for -- but the whole bag was handed to the new one,
    so a curve cell's ``group`` reached a histogram cell that has no such
    field, ``updated_spec`` raised ``KeyError('group')``, and the change was
    refused with nothing on screen to say why.
    """

    node, snapshot = _one_shot(session, producer="camera_measurement")
    binding = presenter.add_panel(
        node.signal_key("frames"), snapshot, title="field scan", kind="facet_grid"
    )
    _settle_panel_hosts(presenter, lambda: binding.host is not None)

    assert presenter.update_panel_state(binding.panel_id, {"cell_kind": "curve"})
    _settle_panel_hosts(
        presenter,
        lambda: binding.state.cell_kind == "curve"
        and bool(binding.parameter_surface.get("semantic")),
    )
    row = _fate_row_offering(binding, "group")
    assert row is not None, (
        binding.state.cell_kind,
        binding.parameter_surface.get("semantic_unavailable"),
        tuple(str(entry["key"]) for entry in binding.parameter_surface["semantic"]),
    )
    assert presenter.update_panel_state(binding.panel_id, {"semantic": {row: "group"}})
    _settle_panel_hosts(presenter, lambda: row in binding.state.semantic)

    assert presenter.update_panel_state(binding.panel_id, {"cell_kind": "histogram"}), (
        "an assignment authored under curve must not veto the histogram cell"
    )
    _settle_panel_hosts(
        presenter,
        lambda: binding.host is not None
        and binding.configuration is None
        and not binding.parameter_surface.get("semantic_provisional", True),
    )
    settled = next(
        entry["value"]
        for entry in binding.parameter_surface["semantic"]
        if str(entry["key"]) == row
    )
    assert settled != "group"
    assert binding.state.semantic.get(row) == settled, dict(
        binding.state.semantic
    )
    assert binding.state.cell_kind == "histogram"
    # The outer grid may legitimately own this row as its facet.  Whatever the
    # resolved histogram vocabulary says is the one current record; the old
    # curve-only Group answer cannot remain authoritative.
    assert presenter.update_panel_state(binding.panel_id, {"cell_kind": "curve"})
    _settle_panel_hosts(
        presenter,
        lambda: binding.state.cell_kind == "curve"
        and not binding.parameter_surface.get("semantic_provisional", True),
    )
    curve_settled = next(
        entry["value"]
        for entry in binding.parameter_surface["semantic"]
        if str(entry["key"]) == row
    )
    assert binding.state.semantic.get(row) == curve_settled
    assert not binding.reported_condition


def test_a_cell_kind_change_survives_a_shared_name_it_cannot_honour(
    presenter, session
) -> None:
    """The other half of the same gesture: the name crosses, the value cannot.

    Both cell kinds declare ``x``.  The curve cell offers the y pixel axis
    there; the image cell paints that axis up its own side, so taking it as
    x collides -- "ImagePlot x and y must be different axes" -- and the whole
    record was refused as one edit.  The operator asked for an image cell, so
    the image cell wins: only the legal overlap crosses the transaction and
    the accepted image description replaces the old vocabulary.
    """

    node, snapshot = _one_shot(session, producer="camera_measurement")
    binding = presenter.add_panel(
        node.signal_key("frames"), snapshot, title="frames", kind="facet_grid"
    )
    _settle_panel_hosts(presenter, lambda: binding.host is not None)

    assert presenter.update_panel_state(binding.panel_id, {"cell_kind": "curve"})
    _settle_panel_hosts(
        presenter,
        lambda: binding.state.cell_kind == "curve"
        and bool(binding.parameter_surface.get("semantic")),
    )
    vertical = next(
        (
            str(entry["key"])
            for entry in binding.parameter_surface["semantic"]
            if str(entry["label"]) == "spatial-y"
            and any(value == "x" for _label, value in tuple(entry["choices"]))
        ),
        None,
    )
    assert vertical is not None, "the curve cell must offer the y pixel axis as x"
    assert presenter.update_panel_state(
        binding.panel_id, {"semantic": {vertical: "x"}}
    )
    _settle_panel_hosts(presenter, lambda: binding.state.semantic.get(vertical) == "x")

    assert presenter.update_panel_state(binding.panel_id, {"cell_kind": "image"}), (
        "an axis authored under curve must not veto the image cell"
    )
    _settle_panel_hosts(presenter, lambda: binding.state.cell_kind == "image")
    assert binding.state.cell_kind == "image"
    assert not binding.reported_condition


def test_a_panel_that_crossed_vocabularies_still_configures_and_saves(
    presenter, session, tmp_path
) -> None:
    """Two doors used to open onto the strict gate with the old vocabulary.

    Right after a cell kind change and before the panel settles, the record
    still holds the previous vocabulary's names.  Anything that re-sent the
    whole record -- an unrelated edit, Save Fig -- was refused with "unknown
    display parameter(s)", which is not a thing the operator did.
    """

    node, snapshot = _one_shot(session, producer="camera_measurement")
    binding = presenter.add_panel(
        node.signal_key("frames"), snapshot, title="frames", kind="facet_grid"
    )
    _settle_panel_hosts(presenter, lambda: binding.host is not None)
    assert presenter.update_panel_state(binding.panel_id, {"cell_kind": "image"})
    _settle_panel_hosts(
        presenter,
        lambda: binding.state.cell_kind == "image"
        and bool(binding.parameter_surface.get("display")),
    )
    assert presenter.update_panel_state(
        binding.panel_id, {"display": {"show_colorbar": False}}
    )
    _settle_panel_hosts(
        presenter, lambda: binding.state.display.get("show_colorbar") is False
    )

    # Cross to a vocabulary that declares none of those names, then edit
    # something unrelated in the same beat.
    assert presenter.update_panel_state(binding.panel_id, {"cell_kind": "curve"})
    unrelated = "1x2" if binding.state.size != "1x2" else "4x4"
    assert presenter.update_panel_state(binding.panel_id, {"size": unrelated}), (
        "an unrelated edit must not carry the previous vocabulary to the host",
        presenter.view.status[-3:],
    )
    assert not any(
        "unknown display parameter" in str(text)
        for _severity, text in presenter.view.status
    ), presenter.view.status

    _settle_panel_hosts(
        presenter,
        lambda: binding.configuration is None and binding.host is not None,
    )
    assert presenter.refresh_panel_snapshot(binding.panel_id)
    _settle_panel_hosts(
        presenter,
        lambda: not binding.frozen_configuration_incompatible,
    )
    assert presenter.save_panel_figure(
        binding.panel_id, str(tmp_path / "crossed")
    ) is True, presenter.view.status
    _wait_for_panel_save(presenter, tmp_path / "crossed.png")

    # Settling the new vocabulary deletes the previous kind's hidden display
    # bag.  Returning to Image uses its current defaults; there is no per-kind
    # appearance owner waiting behind the visible Curve state.
    _settle_panel_hosts(
        presenter,
        lambda: binding.state.cell_kind == "curve"
        and bool(binding.parameter_surface.get("display")),
    )
    assert presenter.update_panel_state(binding.panel_id, {"cell_kind": "image"})
    _settle_panel_hosts(
        presenter,
        lambda: binding.state.cell_kind == "image"
        and binding.state.display.get("show_colorbar") is True,
    )
    assert binding.state.display.get("show_colorbar") is True


def test_a_panel_says_what_kind_of_data_it_is_drawing(presenter, session) -> None:
    """Under the signal's name: the dataset's shape, and what this panel pinned.

    The identifier says WHERE the numbers came from.  What they ARE -- the
    axes and their sizes -- and which coordinate this panel is pinned to were
    computed everywhere and stated nowhere, so an operator had to open the
    semantic table to find out what they were looking at.  Both halves are
    projected as FACTS: the strip lays them out, and nobody formats a
    sentence here that the strip would have to take apart again.
    """

    from zlc_data import LATEST_COORDINATE
    from zlc_plot.semantics import (
        is_scope_fate,
        schema_structure,
        scope_coordinate_from_fate,
    )

    node, snapshot = _one_shot(session)
    signal = node.signal_key("frames")
    binding = presenter.add_panel(signal, snapshot, title="camera", kind="image")
    _settle_panel_hosts(
        presenter,
        lambda: any(
            str(field["key"]).startswith("fate:")
            for field in tuple(binding.parameter_surface.get("semantic", ()))
        ),
    )
    assert binding.parameter_surface["data_structure"] == schema_structure(
        snapshot.block.schema
    )
    assert binding.parameter_surface["data_scope"] == ()

    # Pinning an axis to one of its coordinates is a fate the operator gives
    # it, and the strip has to follow that edit.
    fate, pinned = next(
        (field, value)
        for field in binding.parameter_surface["semantic"]
        if str(field["key"]).startswith("fate:")
        for value, _label in tuple(field["cycle_choices"] or ())
        if is_scope_fate(value)
        and scope_coordinate_from_fate(value) is not LATEST_COORDINATE
    )
    assert presenter.update_panel_state(
        binding.panel_id, {"semantic": {str(fate["key"]): pinned}}
    )
    projected = next(
        field
        for field in binding.parameter_surface["semantic"]
        if str(field["key"]) == str(fate["key"])
    )
    assert projected["value"] == pinned, (
        "the authored cycle value must not be replaced by the older accepted "
        "surface while its configuration is in flight"
    )
    number = float(scope_coordinate_from_fate(pinned))
    pinned_text = (
        str(int(number)) if number.is_integer() else f"{number:g}"
    )
    # The strip already says "Latest" for the default scope, so settling
    # on any scope proves nothing: wait for the PINNED one.
    _settle_panel_hosts(
        presenter,
        lambda: binding.parameter_surface.get("data_scope")
        == ((str(fate["label"]), pinned_text),),
    )
    assert binding.parameter_surface["data_scope"] == (
        (str(fate["label"]), pinned_text),
    )


def test_restored_live_selector_answers_displayed_shot_before_plane_latest(
    presenter,
    session,
    monkeypatch,
) -> None:
    """Bridge attachment must not replace the screen's first causal parent."""

    from zlc_runtime import SelectionRange, SelectionState
    from zlc_runtime.selection_bridge import FacetCondition
    from zlc_workbench.selection import panel_selection_document

    session.load_pulse(PULSE_NAME)
    node = CameraMeasurementNode(
        camera=session.camera,
        request=CameraMeasurementRequest(
            "camera",
            0.02,
            None,
            0,
            CAMERA_WINDOWS,
        ),
        signal_plane=session.signal_plane,
        producer="restored-live-camera",
    )
    monitor = node.monitor()
    signal = node.signal_key("frames")

    def next_publication(previous=None):
        deadline = time.monotonic() + 5.0
        session.fire(shots=1)
        while time.monotonic() < deadline:
            monitor.poll()
            publication = session.signal_plane.latest_publication(signal)
            if publication is not None and publication is not previous:
                return publication
            time.sleep(0.002)
        raise AssertionError("camera monitor did not publish its next cycle")

    try:
        displayed = next_publication()
        value = displayed.value(signal)
        assert value is not None

        # Hold bridge installation while a real layout restore accepts N.
        apply_deriving = presenter._apply_deriving
        monkeypatch.setattr(presenter, "_apply_deriving", lambda _binding: None)
        original = presenter.add_panel(
            signal,
            value.snapshot,
            kind="image",
            initial_publication=displayed,
        )
        _settle_panel_hosts(
            presenter,
            lambda: (
                original.host is not None
                and original.port is not None
                and _accepted(original.port, "publication") is displayed
                and original.host.initial_state[0] is not None
            ),
        )
        y_axis, x_axis = value.snapshot.block.schema.cell_schema.data_axes
        # A selection names the exact surface it was drawn on, scope
        # included: the image shows the latest frame, and the subject
        # freezes that into the frame coordinate a restore must carry.
        subject = original.accepted_display.selection_subject
        selection = SelectionState(
            "image",
            "area",
            (
                SelectionRange(
                    str(x_axis.axis_id),
                    float(x_axis.coordinate_at(4)),
                    float(x_axis.coordinate_at(10)),
                    domain="data",
                    coordinate_frame=(
                        None
                        if x_axis.coordinate_frame is None
                        else str(x_axis.coordinate_frame)
                    ),
                ),
                SelectionRange(
                    str(y_axis.axis_id),
                    float(y_axis.coordinate_at(3)),
                    float(y_axis.coordinate_at(8)),
                    domain="data",
                    coordinate_frame=(
                        None
                        if y_axis.coordinate_frame is None
                        else str(y_axis.coordinate_frame)
                    ),
                ),
            ),
            facets=tuple(
                FacetCondition(str(ref.axis_id), coordinate, ref.domain.value)
                for ref, coordinate in subject.scope
            ),
            repeat_index=subject.repeat_index,
        )
        restored_state = replace(
            original.state,
            selector=panel_selection_document(selection),
            published_outputs={"roi_mean": True},
        )
        assert presenter.apply_layout(
            LayoutDocument((restored_state,), ())
        ) is True
        (binding,) = tuple(presenter.panels.values())
        _settle_panel_hosts(
            presenter,
            lambda: (
                binding.host is not None
                and binding.port is not None
                and _accepted(binding.port, "publication") is displayed
                and binding.host.initial_state[0] is not None
            ),
        )

        latest = next_publication(displayed)
        assert _accepted(binding.port, "publication") is displayed
        assert session.signal_plane.latest_publication(signal) is latest

        monkeypatch.setattr(presenter, "_apply_deriving", apply_deriving)
        presenter._apply_deriving(binding)
        derived_name = f"@logic/{binding.panel_id}/roi_mean"
        first = session.signal_plane.latest_publication(derived_name)
        assert first is not None
        assert session.signal_plane.direct_parent_publications(first) == (displayed,)

        deadline = time.monotonic() + 5.0
        caught_up = first
        while caught_up is first and time.monotonic() < deadline:
            session.signal_plane.freeze()
            time.sleep(0.001)
            caught_up = session.signal_plane.latest_publication(derived_name)
        assert caught_up is not None and caught_up is not first
        assert session.signal_plane.direct_parent_publications(caught_up) == (latest,)
        assert caught_up.event_ref.generation == first.event_ref.generation

        # A later reactive answer completes on Runtime's Processor worker.
        # It must wake the Workbench owner immediately; a periodic display
        # tick is neither causal nor prompt enough to stage the owed pair.
        presenter.board.wake.take()
        newest = next_publication(latest)
        presenter.board.wake.take()
        before = session.signal_plane.latest_publication(derived_name)
        deadline = time.monotonic() + 5.0
        after = before
        while after is before and time.monotonic() < deadline:
            session.signal_plane.freeze()
            time.sleep(0.001)
            after = session.signal_plane.latest_publication(derived_name)
        assert after is not None and after is not before
        assert presenter.board.wake.take()
        assert session.signal_plane.latest_publication(signal) is newest
    finally:
        monitor.close()


def _follow_calibration_artifact(session, tmp_path):
    """A one-site calibration matched to the virtual camera's frames."""

    from zlc_atom.nodes.calibration import (
        FrameContract,
        ReadoutModel,
        ReadoutModelKind,
        SiteMap,
        TrapCalibration,
    )

    # One warm shot only to learn the frame geometry for the calibration.
    _node, warm = _one_shot(session, producer="warm")
    height, width = np.asarray(warm.block.values).shape[-2:]
    site_ids = ("site_0001",)
    calibration = TrapCalibration(
        SiteMap(
            site_ids,
            np.asarray(((width / 2.0, height / 2.0),)),
            np.asarray((True,)),
            np.asarray((1.0,)),
        ),
        (
            ReadoutModel(
                site_ids,
                np.asarray((0.0,)),
                np.zeros(1),
                np.ones(1),
                np.asarray((True,)),
                np.asarray((1.0,)),
            ),
        ),
        ReadoutModelKind.BOX,
        FrameContract((height, width)),
    )
    return calibration.save(tmp_path / "follow-calibration.json")


def test_a_started_processor_follows_its_source_across_absence_and_stop(
    presenter, session, tmp_path
) -> None:
    """Start a processor before its camera exists: the Start is accepted as
    a standing follow, the poll beat activates it the moment the source
    publishes, and the operator's own Stop is the one thing that ends the
    following."""

    artifact = _follow_calibration_artifact(session, tmp_path)

    from zlc_workbench.logic import stable_signal_key

    occupancy_id = presenter.add_logic("occupancy", open_editor=False)
    live_signal = stable_signal_key("cm-live", "frames")
    assert presenter.update_logic_draft(
        occupancy_id,
        source_signal=live_signal,
        artifact_inputs={"calibration_path": str(artifact)},
    )

    # The source does not exist yet: Start is accepted as an intent.
    started = presenter.start_logic(occupancy_id)
    assert started is True, presenter.logic[occupancy_id].draft_error
    binding = presenter.logic[occupancy_id]
    assert binding.following
    assert binding.host is None
    state, status = presenter._logic_state(binding)
    assert state == "running"
    assert "following" in status and live_signal in status

    # The camera arrives; the beat completes the standing Start.
    camera_id = presenter.add_logic(
        "camera_measurement",
        node_id="cm-live",
        values={
            "exposure_seconds": 0.002,
            "repeat": 0,
            "frames_per_cycle": 1,
        },
        device_keys={"camera": "camera"},
        open_editor=False,
    )
    session.load_pulse(PULSE_NAME)
    assert presenter.start_logic(camera_id)

    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        session.fire(shots=1)
        presenter.beat()
        if binding.host is not None and binding.host.running:
            break
        time.sleep(0.005)
    assert binding.host is not None and binding.host.running
    assert binding.following

    # The operator's Stop is a decision: the follow ends with it.
    assert presenter.stop_logic(occupancy_id)
    assert not binding.following
    _settle_panel_hosts(
        presenter,
        lambda: binding.host is None or not binding.host.running,
    )
    stopped_host = binding.host
    for _ in range(3):
        session.fire(shots=1)
        presenter.beat()
    assert binding.host is stopped_host
    assert not binding.following


def test_a_following_processor_survives_its_camera_stop_and_restart(
    presenter, session, tmp_path
) -> None:
    """Stop the camera, On Pulse again: the processor restarts BY ITSELF.

    The promise of the standing follow is exactly this sequence.  It used
    to break in two places: the source's death mid-follow ended the
    processor host as FAILED ("generation retired", "not committed"), and
    the follower read any failure as the operator's to fix -- following
    cleared, processor parked until a manual restart.  A source that ends
    or moves on under a follower is its lifecycle, not the processor's
    failure: the host ends cancelled, the follow survives, and the beat
    after the camera's next start completes it."""

    artifact = _follow_calibration_artifact(session, tmp_path)

    from zlc_workbench.logic import stable_signal_key

    occupancy_id = presenter.add_logic("occupancy", open_editor=False)
    live_signal = stable_signal_key("cm-live", "frames")
    assert presenter.update_logic_draft(
        occupancy_id,
        source_signal=live_signal,
        artifact_inputs={"calibration_path": str(artifact)},
    )
    camera_id = presenter.add_logic(
        "camera_measurement",
        node_id="cm-live",
        values={
            "exposure_seconds": 0.002,
            "repeat": 0,
            "frames_per_cycle": 1,
        },
        device_keys={"camera": "camera"},
        open_editor=False,
    )
    session.load_pulse(PULSE_NAME)
    assert presenter.start_logic(camera_id)
    assert presenter.start_logic(occupancy_id)
    binding = presenter.logic[occupancy_id]

    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        session.fire(shots=1)
        presenter.beat()
        if binding.host is not None and binding.host.running:
            break
        time.sleep(0.005)
    assert binding.host is not None and binding.host.running
    assert binding.following

    # The operator stops the CAMERA -- not the processor.
    assert presenter.stop_logic(camera_id)
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        presenter.beat()
        if binding.host is None or not binding.host.running:
            break
        time.sleep(0.005)
    assert binding.host is None or not binding.host.running
    # The source's death is its lifecycle, not this processor's failure,
    # and the follow is still standing.
    if binding.host is not None:
        assert binding.host.observation.phase != "failed", (
            binding.host.observation.error
        )
    assert binding.following, (
        "the camera stop ended the follow the operator never cancelled"
    )

    # On Pulse: the camera starts a new generation; the processor follows
    # with no operator action at all.
    assert presenter.start_logic(camera_id)
    restarted = None
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        session.fire(shots=1)
        presenter.beat()
        restarted = binding.host
        if restarted is not None and restarted.running:
            break
        time.sleep(0.005)
    assert restarted is not None and restarted.running, (
        f"the processor never refollowed: phase="
        f"{None if binding.host is None else binding.host.observation.phase} "
        f"error={None if binding.host is None else binding.host.observation.error} "
        f"following={binding.following}"
    )
    assert binding.following


def test_bound_rolling_panel_offers_the_uncertainty_switch(
    presenter, session
) -> None:
    """The display contract of a live rolling panel carries the band and
    cumulative switches -- the vocabulary the operator flips on the bench."""

    node, snapshot = _one_shot(session)
    binding = presenter.add_panel(
        node.signal_key("frames"), snapshot, kind="rolling"
    )
    _settle_panel_hosts(
        presenter,
        lambda: not binding.parameter_surface.get("display_unavailable", "")
        and bool(binding.parameter_surface.get("display")),
    )
    names = {
        str(entry["key"]) for entry in binding.parameter_surface["display"]
    }
    assert "uncertainty" in names, sorted(names)
    assert "trailing" in names, sorted(names)


def test_the_semantic_form_appears_the_moment_a_signal_connects(
    presenter, session
) -> None:
    """The fate rows are a light schema projection, not a render's reward.

    Deriving them needs only schema + spec, so they are on the surface the
    moment add_panel or a signal connect holds a snapshot -- BEFORE any
    host settles, marked provisional until the host's description replaces
    them with the same complete vocabulary plus accepted fit models.
    """

    node, snapshot = _one_shot(session)
    binding = presenter.add_panel(
        node.signal_key("frames"), snapshot, kind="image"
    )
    # No settle: the rows exist at birth.
    surface = binding.parameter_surface
    assert surface["semantic"], "fate rows must not wait for a render"
    assert surface.get("semantic_provisional") is True
    assert any(
        str(entry["key"]).startswith("fate:") for entry in surface["semantic"]
    )
    assert surface["data_structure"], "the shape strip speaks at birth too"

    _settle_panel_hosts(
        presenter,
        lambda: not binding.parameter_surface.get("semantic_provisional"),
    )
    assert binding.parameter_surface["semantic"], (
        "the described surface keeps the rows"
    )


def test_exact_scan_panels_keep_axes_in_titles_and_refused_settings(
    presenter, session
) -> None:
    """A display reads the canonical scan, never its final one-point chunk."""

    from zlc_data import (
        READOUT_EVENT,
        SITE,
        AxisId,
        AxisSpec,
        DatasetSchema,
        PointColumn,
        PointTable,
        REPEAT,
        ValidityContract,
        ValueSchema,
        owned_snapshot_from_arrays,
    )
    from zlc_atom.nodes.scan.dataset import scan_dataset_schema
    from zlc_plot.semantics import (
        is_scope_fate,
        schema_structure,
        scope_coordinate_from_fate,
    )
    from zlc_runtime import (
        DatasetCoverage,
        DatasetOutputDeclaration,
        LiveDatasetOutput,
    )

    dimensions = (65, 2, 2)  # A real scan axis exceeds the 64-cell cap.
    names = ("field.x", "field.y", "field.z")
    cells = tuple(np.ndindex(*dimensions))
    repeat_id = AxisId("survival.repeat")
    event_repeat = AxisSpec(repeat_id, "repeat", REPEAT, 1, (0,))
    # Frame survival publishes its pairs as a READOUT_EVENT point column --
    # a choice of sub-measurement, as a camera's frames are -- so the scan
    # folds them in as its outermost dimension, ahead of the plan's axes.
    pair = PointColumn(
        AxisId("survival.pair"),
        "pair",
        READOUT_EVENT,
        PointColumn.NUMERIC,
        (0, 1, 2),
        coordinate_labels=("0-1", "0-2", "1-2"),
    )
    site = AxisSpec(AxisId("survival.site"), "site", SITE, 5, tuple(range(5)))
    cell_schema = ValueSchema(
        (site,),
        ValidityContract.components(site.axis_id),
        np.dtype("<f8"),
        "1",
    )
    event_schema = DatasetSchema(
        event_repeat, PointTable(3, (pair,)), None, cell_schema
    )
    canonical = scan_dataset_schema(
        event_schema,
        tuple(tuple(float(value) for value in cell) for cell in cells),
        tuple((name, "") for name in names),
        visits=2,
    )
    event = owned_snapshot_from_arrays(
        event_schema,
        np.ones((1, 3, site.size)),
        1,
        stream_generation="exact-scan-panel",
    )
    declaration = DatasetOutputDeclaration(
        "survival", "frame_survival.survival"
    )
    signal = "scan-survival/survival"
    node = SimpleNamespace(
        instance_id="scan-survival",
        dataset_output_declarations=(declaration,),
        signal_key=lambda name: f"scan-survival/{name}",
    )
    session.signal_plane.begin_generation(node)
    session.signal_plane.commit_live(
        node,
        {
            "survival": LiveDatasetOutput(
                declaration,
                event,
                DatasetCoverage(
                    event_schema.point_table.row_count,
                    canonical.repeat_axis.size * canonical.point_table.row_count,
                ),
                canonical_schema=canonical,
                cell_origin=(0, 0),
            )
        },
    )
    publication = session.signal_plane.freeze().publication(signal)
    expected = schema_structure(canonical)
    assert expected == (
        (("repeat", 2),),
        (
            ("survival.pair", 3),
            ("scan.field.x", 65),
            ("scan.field.y", 2),
            ("scan.field.z", 2),
        ),
        (("site", 5),),
    )

    refused = presenter.add_panel(
        signal,
        event,
        title="refused survival",
        kind="facet_grid",
        initial_publication=publication,
    )
    field_scan = presenter.add_panel(
        signal,
        event,
        title="field scan",
        kind="curve",
        initial_publication=publication,
    )
    assert refused.parameter_surface["data_structure"] == expected
    assert field_scan.parameter_surface["data_structure"] == expected

    _settle_panel_hosts(
        presenter,
        lambda: getattr(refused.port, "last_error", None) is not None,
    )
    presenter._report_panel_errors()
    surface = refused.parameter_surface
    fates = {str(entry["label"]) for entry in surface["semantic"]}
    assert {"field.x", "field.y", "field.z"} <= fates
    assert "point" not in fates
    # The refusal belongs to the PANEL's status, where a refusal goes; the
    # Fit column says why fit is missing, which is that nothing mounted.
    status, marked = presenter.view._cards[refused.panel_id].status
    assert "exceeds the fixed layout" in status and marked
    assert surface["fit_unavailable"] == presenter._RESOLVING_REASON
    assert presenter.view.panel_parameter_surfaces[
        refused.panel_id
    ]["data_structure"] == expected

    # The real large-map geometry that exposed the second half of this bug:
    # the 35 sites and the three pair events are deliberately moved off the
    # image axes so the three scan dimensions become facet/y/x.  The whole fate table must land
    # atomically and a terminal Frozen host must reproduce the same limits.
    map_dimensions = (10, 10, 10)
    map_cells = tuple(np.ndindex(*map_dimensions))
    map_site = AxisSpec(
        AxisId("map.site"), "site", SITE, 35, tuple(range(35))
    )
    map_cell_schema = ValueSchema(
        (map_site,),
        ValidityContract.components(map_site.axis_id),
        np.dtype("<f8"),
        "1",
    )
    map_event_schema = DatasetSchema(
        event_repeat, PointTable(3, (pair,)), None, map_cell_schema
    )
    map_canonical = scan_dataset_schema(
        map_event_schema,
        tuple(
            tuple(float(value) for value in cell) for cell in map_cells
        ),
        tuple((name, "") for name in names),
        visits=20,
    )
    map_event = owned_snapshot_from_arrays(
        map_event_schema,
        np.ones((1, 3, map_site.size)),
        1,
        stream_generation="exact-map-panel",
    )
    map_signal = "scan-map/survival"
    map_node = SimpleNamespace(
        instance_id="scan-map",
        dataset_output_declarations=(declaration,),
        signal_key=lambda name: f"scan-map/{name}",
    )
    session.signal_plane.begin_generation(map_node)
    session.signal_plane.commit_live(
        map_node,
        {
            "survival": LiveDatasetOutput(
                declaration,
                map_event,
                DatasetCoverage(
                    map_event_schema.point_table.row_count,
                    map_canonical.repeat_axis.size
                    * map_canonical.point_table.row_count,
                ),
                canonical_schema=map_canonical,
                cell_origin=(0, 0),
            )
        },
    )
    map_publication = session.signal_plane.freeze().publication(map_signal)
    mapped = presenter.add_panel(
        map_signal,
        map_event,
        title="mapped survival",
        kind="facet_grid",
        initial_publication=map_publication,
    )
    _settle_panel_hosts(
        presenter,
        lambda: mapped.host is not None
        and mapped.frozen_data is not None
        and not mapped.parameter_surface.get("semantic_provisional", True),
    )
    from zlc_plot import AxisRef, NumericRange
    from zlc_plot.selectors import RectangleRange

    from zlc_plot.semantics import fate_field_name

    scan_fates = {
        fate_field_name(AxisRef.point_dimension("scan.field.y")): "y",
        fate_field_name(AxisRef.point_dimension("scan.field.z")): "x",
        fate_field_name(AxisRef.point_dimension("survival.pair")): "reduce",
        fate_field_name(AxisRef.data("map.site")): "reduce",
    }
    offered_fates = {
        str(entry["key"]): tuple(
            value for _label, value in entry["choices"]
        )
        for entry in mapped.parameter_surface["semantic"]
    }
    assert all(
        value in offered_fates.get(name, ())
        for name, value in scan_fates.items()
    ), offered_fates
    mapped.interaction_viewport = (
        presenter._panel_view_identity(mapped),
        RectangleRange(
            NumericRange(0.0, 1.0),
            NumericRange(0.0, 1.0),
        ),
    )
    from zlc_runtime import SelectionRange, SelectionState
    from zlc_workbench.selection import panel_selection_document

    mapped.state = replace(
        mapped.state,
        selector=panel_selection_document(SelectionState(
            plot_kind="image",
            selector_kind="area",
            ranges=(
                SelectionRange("map.site", 0.0, 10.0, domain="data"),
                SelectionRange(
                    "survival.pair", 0.0, 2.0, domain="point_dimension"
                ),
            ),
        )),
    )
    assert presenter.update_panel_state(
        mapped.panel_id, {"semantic": scan_fates}
    )
    scan_configuration = mapped.configuration
    assert scan_configuration is not None
    _settle_panel_hosts(
        presenter,
        lambda: mapped.configuration is None
        and mapped.accepted_display is not None,
    )
    scan_description = mapped.accepted_display
    assert scan_description.semantics.x == AxisRef.point_dimension(
        "scan.field.z"
    )
    assert scan_description.semantics.y == AxisRef.point_dimension(
        "scan.field.y"
    )
    assert all(
        mapped.state.semantic.get(name) == value
        for name, value in scan_fates.items()
    ), dict(mapped.state.semantic)
    assert mapped.interaction_viewport is None, (
        "a viewport measured on pair/site was retained on field.z/field.y"
    )
    assert mapped.state.selector == {}, (
        "a selector measured on pair/site was retained on field.z/field.y"
    )
    live_snapshot = _accepted(mapped.port, "plot_input")
    assert live_snapshot.block.schema.fingerprint == map_canonical.fingerprint
    assert live_snapshot.block.values.shape == (20, 3000, 35)
    assert (
        mapped.frozen_data.snapshot.block.schema.fingerprint
        == map_canonical.fingerprint
    )
    live_description = _operation_value(mapped.host.describe_display())
    assert live_description.semantics.facet == AxisRef.point_dimension(
        "scan.field.x"
    )
    assert live_description.semantics.x == AxisRef.point_dimension(
        "scan.field.z"
    )
    assert live_description.semantics.y == AxisRef.point_dimension(
        "scan.field.y"
    )
    assert (
        live_description.limits.x.low,
        live_description.limits.x.high,
        live_description.limits.y.low,
        live_description.limits.y.high,
    ) == (-0.5, 9.5, -0.5, 9.5)

    assert presenter.edit_panel(mapped.panel_id)
    _settle_panel_hosts(
        presenter,
        lambda: mapped.editor_host is not None
        and mapped.editor_configuration is None,
    )
    frozen_description = _operation_value(
        mapped.editor_host.describe_display()
    )
    assert frozen_description.semantics == live_description.semantics
    assert frozen_description.limits == live_description.limits

    assert session.signal_plane.seal_committed(map_node, cut_short=True)
    assert presenter.close_panel_editor(mapped.panel_id)
    _settle_panel_hosts(presenter)
    assert presenter.edit_panel(mapped.panel_id)
    _settle_panel_hosts(
        presenter,
        lambda: mapped.editor_host is not None
        and mapped.editor_configuration is None,
    )
    terminal_description = _operation_value(
        mapped.editor_host.describe_display()
    )
    assert (
        mapped.frozen_data.snapshot.block.schema.fingerprint
        == map_canonical.fingerprint
    )
    assert terminal_description.semantics == live_description.semantics
    assert terminal_description.limits == live_description.limits


def test_display_state_synchronizes_both_panel_surfaces(
    presenter,
    session,
) -> None:
    """Appearance is ONE truth over two data moments: any display value
    committed on either surface -- a camera orbit, a colour-limit drag --
    lands in the panel record and mirrors to the sibling.  The blacklist
    is empty, and the sync itself never reads as staleness."""

    import time as _time

    node, snap = _one_shot(session)
    panel = presenter.add_panel(node.signal_key("frames"), snap, kind="image")
    _settle_panel_hosts(presenter, lambda: panel.host is not None)
    assert presenter.update_panel_state(
        panel.panel_id, {"display": {"presentation": "height_bars"}}
    )
    _settle_panel_hosts(
        presenter,
        lambda: panel.configuration is None
        and panel.port.presentation_current,
    )
    assert presenter.edit_panel(panel.panel_id)
    _settle_panel_hosts(
        presenter,
        lambda: panel.editor_host is not None
        and panel.frozen_data is not None
        and panel.frozen_data.description is not None,
    )

    def beat(condition, timeout=6.0):
        deadline = _time.monotonic() + timeout
        while _time.monotonic() < deadline:
            presenter.beat()
            if condition():
                return True
            _time.sleep(0.01)
        return condition()

    def live_display():
        return panel.host.describe_display().result(
            timeout=10
        ).value.display_state.values

    def edit_display():
        return panel.editor_host.describe_display().result(
            timeout=10
        ).value.display_state.values

    panel.editor_host.set_parameters(
        {"color_min": 10.0, "color_max": 200.0}
    ).result(timeout=10)
    assert beat(lambda: live_display().get("color_max") == 200.0)
    assert panel.frozen_configuration_incompatible is False

    panel.editor_host.set_parameters({"camera_azimuth": 200.0}).result(
        timeout=10
    )
    assert beat(lambda: live_display().get("camera_azimuth") == 200.0)

    panel.host.set_parameters({"color_max": 150.0}).result(timeout=10)
    assert beat(lambda: edit_display().get("color_max") == 150.0)
    assert panel.state.display.get("color_max") == 150.0


def test_the_operator_viewport_survives_a_same_geometry_run(
    presenter,
    session,
) -> None:
    """A viewport is measured on GEOMETRY, and the same instrument fired
    again puts the same axes under the same numeric rectangle.  Keying
    the remembered viewport on the stream generation -- or on a units
    representation that differed between the description and the record
    -- threw the operator's zoom away at every shot boundary and mounted
    each run at the autoscaled home."""

    import time as _time

    from zlc_plot import NumericRange

    node, snap = _one_shot(session)
    panel = presenter.add_panel(node.signal_key("frames"), snap, kind="image")
    _settle_panel_hosts(
        presenter,
        lambda: panel.host is not None
        and panel.accepted_surface is not None
        and panel.selections is not None,
    )
    first_host = panel.host
    first_host.set_viewport(
        NumericRange(30.0, 60.0), NumericRange(20.0, 50.0)
    ).result(timeout=10)
    deadline = _time.monotonic() + 6.0
    while _time.monotonic() < deadline and panel.interaction_viewport is None:
        presenter.beat()
        _time.sleep(0.01)
    assert panel.interaction_viewport is not None
    # What the panel actually holds, not what was asked for: an image
    # viewport lands on the source pixel grid, so a request that cuts
    # pixels in half is widened to the edges around them.  What this
    # measures is that the rectangle SURVIVES the run, whatever it is.
    before = panel.host.describe_display().result(timeout=10).value.viewport
    assert before is not None
    assert (float(before.x.low), float(before.x.high)) == (29.5, 60.5)
    assert (float(before.y.low), float(before.y.high)) == (19.5, 50.5)

    _one_shot(session, producer="cm")
    _settle_panel_hosts(
        presenter,
        lambda: panel.host is not None and panel.configuration is None,
    )
    # The stronger property landed on the way: a SAME-GEOMETRY run keeps
    # its host entirely, so the view (and any in-flight gesture) never
    # even needs replaying.
    assert panel.host is first_host
    viewport = panel.host.describe_display().result(
        timeout=10
    ).value.viewport
    assert viewport is not None
    assert (float(viewport.x.low), float(viewport.x.high)) == (
        float(before.x.low),
        float(before.x.high),
    )
    assert (float(viewport.y.low), float(viewport.y.high)) == (
        float(before.y.low),
        float(before.y.high),
    )


def test_a_gesture_survives_a_shot_landing_mid_drag(presenter, session) -> None:
    """The reported "drag goes dead ~30% of the time" in one assertion.

    A shot that lands mid-gesture used to retire the host whose widget
    had the mouse grabbed, so the drag stopped answering until release.
    A same-geometry run keeps the host: the gesture continues and its
    release commits.
    """

    node, snap = _one_shot(session)
    panel = presenter.add_panel(node.signal_key("frames"), snap, kind="image")
    _settle_panel_hosts(
        presenter,
        lambda: panel.host is not None and panel.accepted_surface is not None,
    )
    host = panel.host
    front = host.front
    axes = next(t for t in front.interaction.axes if t.role == "image")
    left, top, right, bottom = axes.bounds
    cx, cy = (left + right) / 2, (top + bottom) / 2

    host._pointer_event(
        "press", cx, cy, button=2,
        identity=front.identity, axes=axes, interaction=front.interaction,
    ).result(timeout=10)

    _one_shot(session, producer="cm")
    for _ in range(40):
        presenter.beat()
        time.sleep(0.005)

    assert panel.host is host, "the shot retired the host mid-gesture"
    moved = host._pointer_event(
        "move", cx + 40.0, cy + 20.0, button=2
    ).result(timeout=10)
    assert moved is not None
    host._pointer_event(
        "release", cx + 40.0, cy + 20.0, button=2
    ).result(timeout=10)
    for _ in range(40):
        presenter.beat()
        time.sleep(0.005)
    assert host.describe_display().result(timeout=10).value.viewport is not None


def test_the_display_mirror_goes_quiet_after_a_burst(presenter, session) -> None:
    """Rapid edits must converge, not ping-pong.

    Revision ordering alone cannot break the cross-host loop: a quick
    second edit makes the sibling's echo of the FIRST arrive as a newer
    revision carrying the older value.  Echoes are recognized by value
    and consumed, so the mirror falls silent and both surfaces agree.
    """

    node, snap = _one_shot(session)
    panel = presenter.add_panel(node.signal_key("frames"), snap, kind="image")
    _settle_panel_hosts(
        presenter,
        lambda: panel.host is not None and panel.accepted_surface is not None,
    )
    assert presenter.update_panel_state(
        panel.panel_id, {"display": {"presentation": "height_bars"}}
    )
    _settle_panel_hosts(
        presenter,
        lambda: panel.configuration is None
        and panel.port.presentation_current,
    )
    assert presenter.edit_panel(panel.panel_id)
    _settle_panel_hosts(
        presenter,
        lambda: panel.editor_host is not None
        and panel.frozen_data is not None
        and panel.frozen_data.description is not None,
    )

    pushes: list[dict] = []
    for host in (panel.host, panel.editor_host):
        original = host.set_parameters

        def spy(mapping, _original=original):
            pushes.append(dict(mapping))
            return _original(mapping)

        host.set_parameters = spy

    panel.editor_host.set_parameters({"camera_zoom": 1.2})
    panel.editor_host.set_parameters({"camera_zoom": 1.44})
    for _ in range(80):
        presenter.beat()
        time.sleep(0.005)
    settled = len(pushes)
    for _ in range(60):
        presenter.beat()
        time.sleep(0.005)

    assert len(pushes) == settled, f"the mirror kept talking: {pushes[settled:]}"
    live = panel.host.describe_display().result(
        timeout=10
    ).value.display_state.values
    edit = panel.editor_host.describe_display().result(
        timeout=10
    ).value.display_state.values
    assert live["camera_zoom"] == edit["camera_zoom"] == 1.44
    assert panel.state.display["camera_zoom"] == 1.44


def test_refresh_adopts_the_card_when_the_derived_signal_retired(
    presenter,
    session,
) -> None:
    """The operator's chain: camera image -> ROI -> a second panel.

    A derived signal is retired with the run that produced it.  Refresh
    asked the PLANE what to freeze, so once that run ended it answered
    "has not published yet" and refused -- while the card was still
    showing a perfectly good picture the Edit tab had not caught up to.
    The panel's accepted surface is the authority on what the operator
    sees, so Refresh adopts it.
    """

    from zlc_workbench.logic import stable_signal_key

    camera_id = presenter.add_logic(
        "camera_measurement",
        node_id="roi-refresh",
        values={"exposure_seconds": 0.002, "repeat": 0, "frames_per_cycle": 1},
        device_keys={"camera": "camera"},
        open_editor=False,
    )
    session.load_pulse(PULSE_NAME)
    assert presenter.start_logic(camera_id)
    camera_signal = stable_signal_key(camera_id, "frames")
    deadline = time.monotonic() + 10.0
    publication = None
    while publication is None and time.monotonic() < deadline:
        session.fire(shots=1)
        presenter.beat()
        publication = session.signal_plane.latest_publication(camera_signal)
        time.sleep(0.005)
    assert publication is not None

    image = presenter.add_panel(
        camera_signal,
        publication.value(camera_signal).snapshot,
        kind="image",
    )
    _settle_panel_hosts(
        presenter,
        lambda: image.host is not None and image.bridge is not None,
    )
    presenter.set_deriving(True)
    _commit_area(image.host, lower_fraction=0.3, upper_fraction=0.7)
    presenter.commit_surfaces()
    roi_signal = f"@logic/{image.panel_id}/roi_frame"
    _settle_panel_hosts(
        presenter,
        lambda: session.signal_plane.latest_publication(roi_signal) is not None,
    )

    derived = presenter.add_panel(
        roi_signal,
        session.signal_plane.current_dataset(roi_signal),
        kind="image",
    )
    _settle_panel_hosts(
        presenter,
        lambda: derived.host is not None
        and derived.accepted_surface is not None,
    )
    assert presenter.edit_panel(derived.panel_id)
    _settle_panel_hosts(
        presenter,
        lambda: derived.editor_host is not None
        and derived.frozen_data is not None
        and derived.frozen_data.description is not None,
    )

    opened = derived.frozen_data
    for _ in range(3):
        session.fire(shots=1)
        for _beat in range(20):
            presenter.beat()
            time.sleep(0.005)
    card = derived.accepted_surface
    assert card is not None and card.publication is not opened.publication, (
        "the card must move ahead of Edit for this to mean anything"
    )

    # The producing run ends.  STOP ENDS PRODUCTION, NEVER THE DATA: the
    # derived publication is retained -- the operator keeps deriving from
    # a stopped panel -- so ending the run is no longer what retires it.
    assert presenter.stop_logic(camera_id)
    for _beat in range(40):
        presenter.beat()
        presenter.poll_logic()
        time.sleep(0.005)
    assert session.signal_plane.latest_publication(roi_signal) is not None

    # What genuinely retires the derived signal is its OWNER going away:
    # removing the source panel closes its bridge, which withdraws the
    # derived route from the plane.
    presenter.remove_panel(image.panel_id)
    for _beat in range(10):
        presenter.beat()
        time.sleep(0.005)
    assert session.signal_plane.latest_publication(roi_signal) is None

    assert presenter.refresh_panel_snapshot(derived.panel_id) is True
    assert derived.frozen_data is not opened, (
        "Refresh must adopt the picture the card is showing"
    )
    assert derived.frozen_data.publication is card.publication
    presenter.remove_logic(camera_id)
    for _beat in range(40):
        presenter.beat()
        presenter.poll_logic()
        time.sleep(0.005)


def test_refresh_adopts_a_card_that_is_ahead_even_with_a_newer_shot_pending(
    presenter,
    session,
) -> None:
    """Adoption and waiting are not exclusive.

    When the card has moved ahead of Edit, showing that picture is what
    the operator asked for -- immediately.  Making it wait for the
    pending publication meant a press did nothing visible until the next
    shot rendered, and nothing at all if the bench stopped between.
    """

    node, snap = _one_shot(session)
    panel = presenter.add_panel(node.signal_key("frames"), snap, kind="image")
    _settle_panel_hosts(
        presenter,
        lambda: panel.host is not None and panel.accepted_surface is not None,
    )
    assert presenter.edit_panel(panel.panel_id)
    _settle_panel_hosts(
        presenter,
        lambda: panel.editor_host is not None
        and panel.frozen_data is not None
        and panel.frozen_data.description is not None,
    )
    opened = panel.frozen_data

    # the card accepts a newer shot; Edit stays where it was
    _one_shot(session, producer="cm")
    _settle_panel_hosts(
        presenter,
        lambda: panel.accepted_surface is not None
        and panel.accepted_surface.publication is not opened.publication,
    )
    card = panel.accepted_surface

    # ... and one more publication is already waiting in the plane
    _one_shot(session, producer="cm")

    assert presenter.refresh_panel_snapshot(panel.panel_id) is True
    assert panel.frozen_data is not opened, (
        "a card ahead of Edit must be adopted at once"
    )
    assert panel.frozen_data.publication is card.publication


def test_a_replacement_host_mounts_the_view_the_operator_just_committed(
    presenter,
    session,
) -> None:
    """A projection in flight must not freeze the operator's view.

    The target a replacement is staged from was captured when the
    projection was submitted.  A drag that finishes after that -- which
    is every drag on a live panel -- is written into the panel record but
    not into that target, so the new generation mounted the pre-drag
    camera and the scene jumped back to where the drag began.  Identity
    still comes from the target; what the operator authors comes from the
    record as it stands when the host is built.
    """

    from dataclasses import replace as _replace

    node, snap = _one_shot(session)
    panel = presenter.add_panel(node.signal_key("frames"), snap, kind="image")
    _settle_panel_hosts(
        presenter,
        lambda: panel.host is not None and panel.accepted_surface is not None,
    )

    stale_target = panel.state
    # the operator's drag lands after that target was captured
    panel.state = _replace(
        panel.state,
        display={**dict(panel.state.display), "camera_azimuth": 133.0},
    )

    seen: list[object] = []
    original = presenter._make_host

    def spy(plot_input, state):
        seen.append(state)
        return original(plot_input, state)

    presenter._make_host = spy
    surface = panel.accepted_surface
    assert surface is not None
    host, _operation = presenter._stage_panel_host(
        panel,
        surface.plot_input,
        None,
        surface.publication,
        state=stale_target,
    )
    try:
        assert seen, "the replacement was never built"
        mounted = seen[-1]
        assert mounted.signal == stale_target.signal
        assert mounted.display.get("camera_azimuth") == 133.0, (
            "the replacement mounted a view the operator had already left"
        )
    finally:
        presenter._make_host = original
        presenter._retire_plot_host(host)


def test_the_console_answers_the_manual_axis_question_the_engine_asks(
    session,
) -> None:
    """One vocabulary, two packages, and a test that keeps them equal.

    A request kind cannot be imported across this wall -- the workbench
    may not reach into a node leaf -- so it is a literal at both ends.
    What keeps two literals one word is this: the kind the scan engine
    raises is fed to the console, and the console has to route it.
    """

    from zlc_atom.nodes.scan import MANUAL_AXIS_REQUEST
    from zlc_runtime import OperatorInputRequest
    from zlc_workbench.apps.task_console import build_panel_host
    from zlc_workbench.logic import LogicBinding

    class _Host:
        def __init__(self, request):
            self.operator_request = request
            self.answered = []
            self.cancelled = ""
            self.running = True

        def submit_operator_input(self, request_id, response):
            self.answered.append((request_id, dict(response)))

        def cancel(self, reason):
            self.cancelled = str(reason)

    def request(identity, value):
        return OperatorInputRequest(
            identity,
            MANUAL_AXIS_REQUEST,
            "Set power",
            f"Set power to {value:g}, then continue.",
            {"axis": "power", "value": value, "point": 1, "points": 2},
        )

    asked = []
    answers = [{}, None]
    view = _ConsoleView()
    presenter = ConsolePresenter(
        session,
        view,
        make_host=build_panel_host,
        spec_for=lambda s, kind="", cell_kind="": task_console_fitting_spec(
            s.block.schema, kind, cell_kind
        ),
        manual_axis=lambda incoming: (
            asked.append(incoming) or answers[len(asked) - 1]
        ),
    )
    try:
        host = _Host(request("req-1", 1.0))
        binding = LogicBinding(node_id="scan", descriptor=None, host=host)
        presenter._handle_operator_request(binding)
        assert [incoming.kind for incoming in asked] == [MANUAL_AXIS_REQUEST]
        assert host.answered == [("req-1", {})]
        assert not host.cancelled

        # Declining is an answer too: the operator is the loop here, and
        # the run stops rather than waiting on a hand that has left.
        host.operator_request = request("req-2", 2.0)
        presenter._handle_operator_request(binding)
        assert host.answered == [("req-1", {})]
        assert "stopped the manual scan" in host.cancelled
    finally:
        presenter.close()


def test_a_refused_parameter_expression_is_said_where_messages_are_said(
    presenter, session
) -> None:
    """A reason reaches the header and the card, not a widget in the form.

    It used to be pushed into the Setting form as a field, so the operator
    got a control-shaped thing that was not a control, holding a sentence
    wider than the box it was in.  The console has one place for what it
    could not do.
    """

    node, snapshot = _one_shot(session)
    binding = presenter.add_panel(
        node.signal_key("frames"), snapshot, kind="image"
    )
    _settle_panel_hosts(
        presenter, lambda: bool(binding.parameter_surface.get("fit"))
    )
    fit_model = next(
        value for _label, value in binding.parameter_surface["fit"][0]["choices"]
    )
    assert presenter.update_panel_state(
        binding.panel_id, {"fit": {"model": fit_model}}
    )
    _settle_panel_hosts(
        presenter,
        lambda: any(
            field["key"] == "expression"
            for field in binding.parameter_surface["fit"]
        ),
    )

    said: list[tuple[str, str]] = []
    original = presenter._report

    def report(message, severity="task", **kwargs):
        said.append((severity, str(message)))
        return original(message, severity=severity, **kwargs)

    presenter._report = report

    marks: list[tuple[str, str, bool]] = []
    original_status = presenter.view.set_panel_status

    def status(panel_id, text, error=False):
        marks.append((str(panel_id), str(text), bool(error)))
        return original_status(panel_id, text, error=error)

    presenter.view.set_panel_status = status

    assert presenter.update_panel_state(
        binding.panel_id,
        {"fit": {"model": fit_model, "expression": "not_a_parameter=1"}},
    )
    _settle_panel_hosts(
        presenter,
        lambda: bool(binding.parameter_surface.get("fit_refused")),
    )
    reason = str(binding.parameter_surface["fit_refused"])
    assert "not a parameter" in reason, reason

    # THE HEADER, once, naming the panel it belongs to; and THE CARD, for as
    # long as it stays true.
    _settle_panel_hosts(
        presenter,
        lambda: any(
            binding.panel_id in text and "not a parameter" in text
            for _severity, text in said
        )
        and any(
            identifier == binding.panel_id and error and "not a parameter" in text
            for identifier, text, error in marks
        ),
    )


def _setting_field_sets(surface) -> dict:
    """The form's field IDENTITY, section by section -- what may not move."""

    return {
        section: tuple(str(row["key"]) for row in surface[section])
        for section in ("semantic", "display", "fit")
    }


def test_a_reported_refusal_never_edits_the_setting_field_set(
    presenter, session
) -> None:
    """THE FIELD SET IS A DECLARATION; a refusal may only annotate it.

    The tenth-odd instance of one disease: some transient non-error --
    here the bridge's answers after a run stops -- reached
    ``_report_panel_errors``, whose degrade swapped the panel onto the
    schema projection, whose fit section is empty by design.  With the
    run stopped there is no next present to write the description back,
    so the fit controls were simply GONE from the Setting form.  The law
    was already written (ARCHITECTURE_DESIGN.md:101/103); this is the
    mechanical form of it: whatever a bridge reports, the key set of
    every Setting section stays exactly what the accepted description
    declared.
    """

    node, snapshot = _one_shot(session)
    binding = presenter.add_panel(
        node.signal_key("frames"), snapshot, kind="image"
    )
    _settle_panel_hosts(
        presenter,
        lambda: bool(binding.parameter_surface.get("fit"))
        and binding.accepted_surface is not None,
    )
    declared = _setting_field_sets(binding.parameter_surface)
    assert declared["fit"], "the fixture must describe a fit vocabulary"

    class _ErroredBridge:
        last_error = RuntimeError("a genuine bridge defect")
        last_condition = ""

        def close(self) -> None:
            pass

    binding.bridge = _ErroredBridge()
    presenter.beat()
    assert _setting_field_sets(binding.parameter_surface) == declared

    class _WaitingBridge:
        last_error = None
        last_condition = (
            "this run is no longer held, so its fit derives nothing"
        )

        def close(self) -> None:
            pass

    binding.bridge = _WaitingBridge()
    presenter.view.status.clear()
    presenter.beat()
    assert _setting_field_sets(binding.parameter_surface) == declared
    # A condition is the panel's own state: said verbatim on the card --
    # it is not about the operator's settings -- and never at error
    # severity on the board line.
    status, _marked = presenter.view.cards[0].status
    assert status == _WaitingBridge.last_condition
    assert not [item for item in presenter.view.status if item[0] == "error"]

    # The level clears itself: a bridge that can answer again ends the
    # condition, and the card follows.
    class _HealthyBridge:
        last_error = None
        last_condition = ""

        def close(self) -> None:
            pass

    binding.bridge = _HealthyBridge()
    presenter.beat()
    assert presenter.view.cards[0].status == ("", False)
    assert _setting_field_sets(binding.parameter_surface) == declared


def test_an_unbound_refused_value_keeps_the_kind_vocabulary(
    presenter,
) -> None:
    """The unbound twin of the schema-projection law.

    Which controls a panel has is a fact about its KIND; what they hold
    is the authored bag.  The schema-projected path already answers a
    refused value with the full vocabulary plus a reason; this pins the
    unbound path to the same law -- it used to return an empty control
    set, a different FIELD SET, sending the form down its replacement
    path.
    """

    binding = presenter.add_selected_panel("histogram")
    assert binding is not None
    clean = _setting_field_sets(binding.parameter_surface)
    assert clean["display"], "a histogram declares display controls unbound"

    poisoned = replace(
        binding.state,
        display={**binding.state.display, "bin_count": "not-a-count"},
    )
    surface = presenter._unbound_panel_parameters(poisoned)
    assert _setting_field_sets(surface)["display"] == clean["display"]
    assert surface["display_unavailable"]


def test_opening_edit_names_todays_folder_without_making_it(
    presenter, session
) -> None:
    """Every Edit projection used to create the day folder and flush its
    directory entry to disk -- on the GUI thread, once per publication while
    the tab was open.  Naming is a question; making it is the save's job."""

    node, snapshot = _one_shot(session)
    binding = presenter.add_panel(node.signal_key("frames"), snapshot, kind="image")
    _settle_panel_hosts(presenter, lambda: binding.host is not None)
    assert presenter.edit_panel(binding.panel_id)
    projection = presenter.panel_editor_projection(binding.panel_id)
    named = Path(projection["save_directory"])
    assert named == session.day_folder_path()
    assert not named.exists(), "naming the folder made it"
    assert session.day_folder() == named and named.is_dir()


def test_refresh_advances_the_editors_own_host(presenter, session) -> None:
    """Refresh shows the newer freeze on the host Edit already has.

    It used to build a second host over the whole frozen history -- a
    session, a projection, a fit and a first paint -- for a picture the card
    had just drawn incrementally.  The editor host takes the data through
    the live pair and keeps its artists; only a moved target replaces it.
    """

    node, snap = _one_shot(session)
    panel = presenter.add_panel(node.signal_key("frames"), snap, kind="image")
    _settle_panel_hosts(
        presenter,
        lambda: panel.host is not None and panel.accepted_surface is not None,
    )
    assert presenter.edit_panel(panel.panel_id)
    _settle_panel_hosts(
        presenter,
        lambda: panel.editor_host is not None
        and panel.editor_configuration is None
        and panel.frozen_data is not None
        and panel.frozen_data.description is not None,
    )
    host = panel.editor_host
    opened = panel.frozen_data
    _one_shot(session, producer="cm")
    _settle_panel_hosts(
        presenter,
        lambda: panel.accepted_surface is not None
        and panel.accepted_surface.publication is not opened.publication,
    )
    card = panel.accepted_surface
    assert presenter.refresh_panel_snapshot(panel.panel_id) is True
    assert panel.frozen_data is not opened
    assert panel.frozen_data.publication is card.publication
    _settle_panel_hosts(
        presenter,
        lambda: panel.editor_configuration is None
        and panel.frozen_data.description is not opened.description,
    )
    assert panel.editor_host is host, "Refresh rebuilt the editor host"
    assert not host.closing, "settling the advanced host retired it"
    shown = getattr(card.plot_input, "snapshot", card.plot_input)
    assert panel.frozen_data.snapshot.ref == shown.ref
    assert panel.editor_selections is not None
    assert not [
        text for severity, text in presenter.view.status if severity == "error"
    ]


def test_refreshes_in_flight_supersede_on_the_same_host(
    presenter, session
) -> None:
    """A second Refresh before the first has painted supersedes it silently.

    With a live source every owed presentation arrives while the previous
    adoption is still on its way.  The host retains only the latest waiting
    frame and cancels the rest -- flow control, which must neither rebuild
    the surface nor surface as a panel error.
    """

    node, snap = _one_shot(session)
    panel = presenter.add_panel(node.signal_key("frames"), snap, kind="image")
    _settle_panel_hosts(
        presenter,
        lambda: panel.host is not None and panel.accepted_surface is not None,
    )
    assert presenter.edit_panel(panel.panel_id)
    _settle_panel_hosts(
        presenter,
        lambda: panel.editor_host is not None
        and panel.editor_configuration is None
        and panel.frozen_data is not None
        and panel.frozen_data.description is not None,
    )
    host = panel.editor_host
    opened = panel.frozen_data
    for _ in range(2):
        before = panel.frozen_data.publication
        _one_shot(session, producer="cm")
        _settle_panel_hosts(
            presenter,
            lambda: panel.accepted_surface is not None
            and panel.accepted_surface.publication is not before,
        )
        assert presenter.refresh_panel_snapshot(panel.panel_id) is True
    latest = panel.accepted_surface
    assert panel.frozen_data.publication is latest.publication
    _settle_panel_hosts(
        presenter,
        lambda: panel.editor_configuration is None
        and panel.frozen_data.description is not opened.description,
    )
    assert panel.editor_host is host and not host.closing
    assert panel.frozen_data.publication is latest.publication
    assert not [
        text for severity, text in presenter.view.status if severity == "error"
    ]


def test_save_renders_through_the_settled_editor_host(
    presenter, session, tmp_path, monkeypatch
) -> None:
    """Save Fig draws the export on the Edit surface it already has.

    A fresh host per save cost a session, a projection, a fit and a first
    paint over the frozen history before the one draw the file needed.
    """

    import zlc_plot.figure_artifact as artifact

    node, snapshot = _one_shot(session)
    binding = presenter.add_panel(node.signal_key("frames"), snapshot, kind="image")
    assert presenter.edit_panel(binding.panel_id)
    _settle_panel_hosts(
        presenter,
        lambda: binding.editor_host is not None
        and binding.editor_configuration is None
        and not binding.frozen_configuration_incompatible,
    )
    host = binding.editor_host

    def refused(*_args, **_kwargs):
        raise AssertionError("Save built a second host beside the settled editor")

    monkeypatch.setattr(artifact, "build_figure_host", refused)
    target = tmp_path / "through-editor.png"
    assert presenter.save_panel_figure(binding.panel_id, str(target)) is True
    _wait_for_panel_save(presenter, target)
    assert target.with_suffix(".npz").exists()
    assert binding.editor_host is host and not host.closing, (
        "the editor's host must survive the save it rendered"
    )


def test_an_editor_hosts_finished_operation_wakes_the_owner(
    presenter, session
) -> None:
    """Edit's pending configuration and its in-place freeze both wake the owner.

    Only the live host used to ask for the wake when its configuration
    finished.  The editor's settle -- mount, gesture subscription, clearing
    the pending entry -- is owner work too, and without the wake it waited
    for the next shot or display beat, once per hop of opening Edit and of
    Refresh.
    """

    import time

    node, snap = _one_shot(session)
    panel = presenter.add_panel(node.signal_key("frames"), snap, kind="image")
    _settle_panel_hosts(
        presenter,
        lambda: panel.host is not None and panel.accepted_surface is not None,
    )
    assert presenter.edit_panel(panel.panel_id)
    entry = panel.editor_configuration
    assert entry is not None
    presenter.board.wake.take()
    deadline = time.monotonic() + 10.0
    while not entry[1].done() and time.monotonic() < deadline:
        time.sleep(0.005)
    assert entry[1].done()
    assert presenter.board.wake.take(), "Edit's finished configuration did not wake the owner"
    _settle_panel_hosts(
        presenter,
        lambda: panel.editor_host is not None
        and panel.editor_configuration is None
        and panel.frozen_data is not None
        and panel.frozen_data.description is not None,
    )
    before = panel.frozen_data.publication
    _one_shot(session, producer="cm")
    _settle_panel_hosts(
        presenter,
        lambda: panel.accepted_surface is not None
        and panel.accepted_surface.publication is not before,
    )
    assert presenter.refresh_panel_snapshot(panel.panel_id) is True
    entry = panel.editor_configuration
    assert entry is not None and entry[0] is panel.editor_host
    presenter.board.wake.take()
    deadline = time.monotonic() + 10.0
    while not entry[1].done() and time.monotonic() < deadline:
        time.sleep(0.005)
    assert entry[1].done()
    assert presenter.board.wake.take(), "the in-place freeze did not wake the owner"


def test_opening_edit_under_a_newer_shot_stages_one_host(
    presenter, session, monkeypatch
) -> None:
    """A shot pending when Edit opens does not build a second host: the
    staged host takes the newer freeze through its data pipeline, queued
    behind its own configuration.

    Under a live source this was every Edit-open: Refresh at open found a
    newer publication pending, the owed presentation adopted it, and the
    first host -- still building its session -- was retired before it ran
    a single operation, and a second was staged in its place.
    """

    import time

    from zlc_plot.session import PlotSession

    node, snap = _one_shot(session)
    panel = presenter.add_panel(node.signal_key("frames"), snap, kind="image")
    _settle_panel_hosts(
        presenter,
        lambda: panel.host is not None and panel.accepted_surface is not None,
    )
    built: list[object] = []
    original_make = presenter._make_host

    def counting(plot_input, state):
        host = original_make(plot_input, state)
        built.append(host)
        return host

    monkeypatch.setattr(presenter, "_make_host", counting)
    # The staged host's session build is held, so its worker runs nothing
    # -- not even its configuration -- until the newer shot is adopted.
    hold_until = [0.0]
    original_init = PlotSession.__init__

    def slow_init(self, *args, **kwargs):
        time.sleep(max(0.0, hold_until[0] - time.monotonic()))
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(PlotSession, "__init__", slow_init)
    # A shot published but not yet presented: Refresh at open finds it
    # pending and asks for the owed presentation.
    _one_shot(session, producer="cm")
    hold_until[0] = time.monotonic() + 1.5
    assert presenter.edit_panel(panel.panel_id)
    assert len(built) == 1
    staged = built[0]
    assert panel.editor_host is None and panel.editor_configuration[0] is staged
    assert panel.refresh_requested, "the pending shot was not owed to Edit"
    opened = panel.frozen_data
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and panel.frozen_data is opened:
        presenter.beat()
        time.sleep(0.005)
    assert panel.frozen_data is not opened, "the newer shot was not adopted"
    assert panel.frozen_data.publication is panel.accepted_surface.publication
    assert len(built) == 1, "a second host was built for the newer freeze"
    assert panel.editor_configuration is not None
    assert panel.editor_configuration[0] is staged
    hold_until[0] = 0.0
    _settle_panel_hosts(
        presenter,
        lambda: panel.editor_host is not None and panel.editor_configuration is None,
    )
    assert panel.editor_host is staged and not staged.closing
    assert panel.frozen_data.publication is panel.accepted_surface.publication
    assert not [
        text for severity, text in presenter.view.status if severity == "error"
    ]
