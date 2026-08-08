"""The console presenter wires views to a session, and decides nothing itself.

Headless on purpose.  The presenter never imports Qt: it receives already-built
views and talks to them through their declared setters and signals, which is
what lets the same code path serve a notebook -- the session below it does not
know a window exists.

The views here are stand-ins with the real signatures.  Substituting the widget
layer rather than the presenter is the point: what is under test is the wiring.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

from zlc_atom.nodes.camera_measurement.measurement import (
    CameraMeasurementNode,
    CameraMeasurementRequest,
)
from zlc_workbench.console import ConsolePresenter
from zlc_workbench.session import ExperimentSession, Workspace
from pulse_fixtures import CAMERA_WINDOWS, PULSE_NAME, write_ordinary_pulse


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
        self.signal_picked = _Signal()
        self.size_picked = _Signal()
        self.update_ms_picked = _Signal()
        self.title_committed = _Signal()
        self.edit_requested = _Signal()
        self.dropped = _Signal()
        self.drag_started = _Signal()
        self.drag_moved = _Signal()
        self.choices: tuple = ()
        self.chosen = ""
        self.size = ""
        self.update_ms = 0
        self.selectors_enabled = True
        self.status: tuple = ("", False)

    def set_signal_choices(self, groups, *, current: str = "") -> None:
        self.choices = tuple(groups)
        self.chosen = str(current or self.chosen)

    def set_panel_size(self, size: str) -> None:
        self.size = str(size)

    def set_title(self, title: str) -> None:
        self.title = str(title)

    def set_update_ms(self, interval_ms: int) -> None:
        self.update_ms = int(interval_ms)

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
        self.stop_requested = _Signal()
        self.edit_requested = _Signal()
        self.remove_requested = _Signal()
        self.state = ("idle", "")
        self.publishes: tuple = ()

    def set_state(self, state: str, status_text: str = "") -> None:
        self.state = (str(state), str(status_text))

    def set_publishes(self, rows) -> None:
        self.publishes = tuple(rows)


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
        "panel_order_committed",
        "panel_signal_picked", "panel_size_picked", "panel_update_ms_picked",
        "panel_title_committed", "panel_remove_requested",
        "panel_edit_requested", "logic_start_requested", "logic_stop_requested",
        "logic_edit_requested", "logic_remove_requested", "logic_draft_changed",
        "panel_state_changed", "panel_snapshot_refresh_requested",
        "panel_producer_apply_requested", "panel_editor_closed",
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
        self.panel_editors: dict[str, dict] = {}
        self.panel_editor_surfaces: dict[str, object] = {}
        self.focused_panel_editor = ""

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

    def set_summary(self, text: str) -> None:
        self.summary = text

    def set_paused(self, paused: bool) -> None:
        self.paused = bool(paused)

    def set_selectors(self, enabled: bool) -> None:
        self.selectors = bool(enabled)

    def show_status(self, text: str, severity: str) -> None:
        self.status.append((str(severity), str(text)))

    def choose_signal(self, rows) -> str | None:
        self.offered = tuple(rows)
        return self.chooser_answer

    def edit_values(self, spec, values, *, title: str):
        return dict(values)

    def show_warning(self, title: str, text: str) -> None:
        self.status.append(("error", str(text)))

    def ask_open_path(self, caption: str, start_dir: str, filter: str) -> str:
        return self.open_answer

    def ask_save_path(self, caption: str, start_dir: str, filter: str) -> str:
        return self.save_answer

    def save_screenshot(self, path: str) -> str:
        Path(path).write_bytes(b"plain TaskConsole screenshot")
        self.screenshot_path = str(path)
        return str(path)

    def run_host_dialog(self, opener, host, *, title: str):
        return opener(host, None, title=title)

    # -- panels -----------------------------------------------------------

    def add_panel(self, panel_id: str, title: str) -> None:
        key = str(panel_id)
        if key not in self._cards:
            card = _CardView(key, str(title))
            card.remove_requested.connect(
                lambda _=None, pid=key: self.panel_remove_requested.emit(pid)
            )
            card.signal_picked.connect(
                lambda name, pid=key: self.panel_signal_picked.emit(pid, str(name))
            )
            card.size_picked.connect(
                lambda size, pid=key: self.panel_size_picked.emit(pid, str(size))
            )
            card.update_ms_picked.connect(
                lambda ms, pid=key: self.panel_update_ms_picked.emit(pid, int(ms))
            )
            card.title_committed.connect(
                lambda text, pid=key: self.panel_title_committed.emit(pid, str(text))
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

    def set_panel_signal_choices(self, panel_id: str, *args, **kwargs) -> None:
        self._cards[str(panel_id)].set_signal_choices(*args, **kwargs)

    def set_panel_update_ms(self, panel_id: str, interval_ms: int) -> None:
        self._cards[str(panel_id)].set_update_ms(interval_ms)

    def set_panel_size(self, panel_id: str, size: str) -> None:
        self._cards[str(panel_id)].set_panel_size(size)

    def set_panel_title(self, panel_id: str, title: str) -> None:
        self._cards[str(panel_id)].set_title(title)

    def set_panel_state(self, panel_id: str, state) -> None:
        key = str(panel_id)
        self.panel_states[key] = state
        self.panel_state_updates.append((key, state))

    def set_panel_parameter_surface(self, panel_id: str, surface) -> None:
        self.panel_parameter_surfaces[str(panel_id)] = surface

    def set_panel_status(self, panel_id: str, text: str, *, error: bool) -> None:
        self._cards[str(panel_id)].set_status(text, error=error)

    def set_panel_selectors_enabled(self, panel_id: str, enabled: bool) -> None:
        self._cards[str(panel_id)].set_selectors_enabled(enabled)

    def open_panel_editor(self, panel_id: str, projection) -> None:
        self.panel_editors[str(panel_id)] = dict(projection)

    def update_panel_editor(self, panel_id: str, projection) -> None:
        if str(panel_id) in self.panel_editors:
            self.panel_editors[str(panel_id)] = dict(projection)

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

    def add_logic_row(self, node_id: str, kind: str) -> None:
        key = str(node_id)
        if key not in self._rows:
            row = _LogicRowView(key, str(kind))
            row.start_requested.connect(
                lambda _=None, nid=key: self.logic_start_requested.emit(nid)
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

    def set_logic_publishes(self, node_id: str, rows) -> None:
        self._rows[str(node_id)].set_publishes(rows)

    def open_logic_editor(self, node_id: str, projection) -> None:
        self.logic_editors[str(node_id)] = dict(projection)

    def update_logic_editor(self, node_id: str, projection) -> None:
        if str(node_id) in self.logic_editors:
            self.logic_editors[str(node_id)] = dict(projection)

    def focus_logic_editor(self, node_id: str) -> None:
        self.focused_logic_editor = str(node_id)

    def close_logic_editor(self, node_id: str) -> None:
        self.logic_editors.pop(str(node_id), None)
        if self.focused_logic_editor == str(node_id):
            self.focused_logic_editor = ""


class _Chooser:
    """The window's question, without the window.

    It records what it was offered, because what an operator can choose from is
    as much a part of the behaviour as what happens once they choose.
    """

    def __init__(self) -> None:
        self.offered: tuple = ()
        self.answer: str | None = None

    def __call__(self, rows):
        self.offered = tuple(rows)
        return self.answer


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
    plot = pytest.importorskip("zlc_plot")

    def kind_of(name):
        return next((item for item in plot.PlotKind if item.value == str(name)), None)

    def spec_for(snapshot, kind=""):
        return plot.fitting_spec(snapshot.block.schema, kind_of(kind))

    def make_host(initial, _signal, kind=""):
        # The same rule the real composition root uses.  A double that builds
        # its hosts a different way is a double that stops being evidence.
        return plot.RasterPlotHost.from_plot(initial, spec_for(initial, kind))

    chooser = _Chooser()
    presenter = ConsolePresenter(
        session,
        _ConsoleView(),
        make_host=make_host,
        panel_kinds=plot.panel_kinds,
        spec_for=spec_for,
        choose_signal=chooser,
    )
    presenter.chooser = chooser
    try:
        yield presenter
    finally:
        presenter.close()


def _one_shot(session, producer: str = "cm"):
    session.load_pulse(PULSE_NAME)
    node = CameraMeasurementNode(
        camera=session.camera,
        request=CameraMeasurementRequest(
            "camera", 0.02, None, 1, CAMERA_WINDOWS, 2.0
        ),
        signal_plane=session.signal_plane,
        producer=producer,
    )
    capture = node.prepare()
    session.fire(shots=1)
    result = capture.collect()
    session.nodes = [node]
    return node, result.publication.value(node.signal_key("frames")).snapshot


def _commit_area(host) -> None:
    """Commit one real Area gesture through the mounted raster surface."""

    front = host.wait_for_front(5.0)
    axes = front.interaction.axes[0]
    left, bottom, right, top = axes.bounds
    start = (left + 0.25 * (right - left), bottom + 0.25 * (top - bottom))
    end = (left + 0.75 * (right - left), bottom + 0.75 * (top - bottom))
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


def test_adding_a_panel_shows_a_card_and_reports_it(presenter, session) -> None:
    node, snapshot = _one_shot(session)
    binding = presenter.add_panel(node.signal_key("frames"), snapshot, title="frames")

    assert presenter.view.cards, "the view was never given a card"
    assert presenter.view.cards[0].surface is binding.host
    assert "1 panel" in presenter.view.summary


def test_removing_a_panel_takes_the_card_away_and_closes_its_host(presenter, session) -> None:
    node, snapshot = _one_shot(session)
    binding = presenter.add_panel(node.signal_key("frames"), snapshot)
    assert presenter.edit_panel(binding.panel_id)
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


def test_pausing_stops_the_beat_without_tearing_anything_down(presenter, session) -> None:
    node, snapshot = _one_shot(session)
    presenter.add_panel(node.signal_key("frames"), snapshot)

    presenter.view.pause_toggled.emit(True)
    assert "paused" in presenter.view.summary
    presenter.beat()  # must be a no-op rather than an error

    presenter.view.pause_toggled.emit(False)
    assert "running" in presenter.view.summary
    presenter.beat()


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


def test_add_panel_puts_the_chosen_kind_on_the_board(presenter, session) -> None:
    """The button that was connected to nothing, then to the wrong question.

    Its whole flow now exists: the operator picks a KIND beside the button, the
    presenter finds a published signal that kind can draw, and the card's own
    picker retargets it afterwards.  It used to open a modal signal chooser
    instead, which asked for a per-panel decision the card already owns and
    ignored the kind combo entirely.
    """

    node, _snapshot = _one_shot(session)
    signal = node.signal_key("frames")

    presenter.view.add_panel_requested.emit("image")
    assert len(presenter.panels) == 1
    binding = next(iter(presenter.panels.values()))
    assert binding.signal == signal and binding.kind == "image"


def test_a_signal_already_on_screen_can_still_take_another_panel(
    presenter, session
) -> None:
    """Two views of one signal is a thing an operator asks for -- the frames as
    an image beside their histogram -- so the second is not refused."""

    node, snapshot = _one_shot(session)
    signal = node.signal_key("frames")
    presenter.add_panel(signal, snapshot, kind="image")

    presenter.view.add_panel_requested.emit("histogram")
    kinds = sorted(binding.kind for binding in presenter.panels.values())
    assert kinds == ["histogram", "image"]
    assert {binding.signal for binding in presenter.panels.values()} == {signal}


def test_pausing_is_reversible_and_the_window_is_told(presenter, session) -> None:
    """A one-way pause is a stopped console with no way back."""

    _one_shot(session)
    assert presenter.view.paused is False

    presenter.view.pause_toggled.emit(True)
    assert presenter.view.paused is True

    presenter.view.pause_toggled.emit(False)
    assert presenter.view.paused is False


def test_turning_selectors_off_stops_panels_deriving(presenter, session) -> None:
    """Off must actually stop the work, not merely stop showing it.

    And it must be reversible, which is why off releases the bridge rather than
    quieting it: closing is final by design, so on builds a new one.
    """

    node, snapshot = _one_shot(session)
    binding = presenter.add_panel(node.signal_key("frames"), snapshot)
    assert binding.bridge is not None and binding.bridge.started

    presenter.view.selectors_toggled.emit(False)
    assert binding.bridge is None, "a panel kept deriving after selectors were turned off"
    assert presenter.view.selectors is False

    presenter.view.selectors_toggled.emit(True)
    assert binding.bridge is not None and binding.bridge.started



def test_a_card_shows_whether_its_selectors_are_live(presenter, session) -> None:
    """The control on the card and the bridge behind it must agree.

    Turning selectors off closes every bridge, so a box drawn afterwards
    derives nothing.  The cards went on showing their selector control live,
    which is a control that looks like it works and does not.
    """

    node, snapshot = _one_shot(session)
    presenter.add_panel(node.signal_key("frames"), snapshot)
    card = presenter.view.cards[0]
    assert card.selectors_enabled is True

    presenter.set_deriving(False)
    assert card.selectors_enabled is False, "the card must say the bridge is gone"

    presenter.set_deriving(True)
    assert card.selectors_enabled is True

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
    offered = {key for _producer, leaves in card.choices for _display, key in leaves}
    assert binding.signal in offered
    assert card.chosen == binding.signal

    card.edit_requested.emit()
    assert presenter.view.focused_panel_editor == binding.panel_id
    assert presenter.view.panel_editors[binding.panel_id]["state"] == (
        binding.state.document()
    )

    presenter.view.panel_state_updates.clear()
    card.title_committed.emit("MOT")
    assert presenter.panels[binding.panel_id].title == "MOT"
    assert card.title == "MOT"
    assert presenter.view.panel_state_updates == [
        (binding.panel_id, presenter.panels[binding.panel_id].state)
    ]
    assert presenter.view.panel_editors[binding.panel_id]["state"]["title"] == "MOT"

    card.update_ms_picked.emit(100)
    assert binding.port.display_interval_ms == 100

    card.size_picked.emit("4x4")
    assert card.size == "4x4"


def test_retargeting_a_panel_keeps_its_place_and_releases_the_old_host(
    presenter, session
) -> None:
    """A plotting host is built around the shape of what it draws, so a new
    signal gets a new host rather than a frame of pixels discovering that an
    image arrived where a curve was."""

    node, snapshot = _one_shot(session)
    first = presenter.add_panel(node.signal_key("frames"), snapshot)
    card = presenter.view.cards[0]
    old_host = first.host
    card.edit_requested.emit()
    assert presenter.view.panel_editors[first.panel_id]["frozen_snapshot"] is snapshot
    old_editor_host = first.editor_host
    assert old_editor_host is not None and old_editor_host is not old_host

    # A second producer, so there is something else to point at.
    other, other_snapshot = _one_shot(session, producer="cm2")
    card.signal_picked.emit(other.signal_key("frames"))

    binding = presenter.panels[first.panel_id]
    assert binding.signal == other.signal_key("frames")
    assert binding.host is not old_host
    assert presenter.view.cards[0] is card, "the card lost its place on the board"
    editor = presenter.view.panel_editors[first.panel_id]
    assert editor["stale"] is True
    assert editor["frozen_snapshot"] is snapshot
    assert binding.editor_host is old_editor_host
    assert not old_editor_host._closing

    presenter.view.panel_snapshot_refresh_requested.emit(first.panel_id)
    editor = presenter.view.panel_editors[first.panel_id]
    assert editor["stale"] is False
    assert editor["frozen_signal"] == other.signal_key("frames")
    assert editor["frozen_snapshot"] is other_snapshot
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
    assert presenter.edit_panel(panel.panel_id)
    first_editor_host = panel.editor_host
    assert first_editor_host is not None

    first_before = dict(presenter.logic[first_id].draft.values)
    second_before = dict(presenter.logic[second_id].draft.values)
    _commit_area(first_editor_host)
    first_selected = dict(presenter.logic[first_id].draft.values)
    assert first_selected != first_before
    assert presenter.logic[second_id].draft.values == second_before

    assert presenter.retarget_panel(
        panel.panel_id, second_node.signal_key("frames")
    )
    assert panel.frozen_stale and panel.editor_host is first_editor_host
    _commit_area(first_editor_host)
    assert presenter.logic[first_id].draft.values == first_selected
    assert presenter.logic[second_id].draft.values == second_before

    assert presenter.refresh_panel_snapshot(panel.panel_id)
    second_editor_host = panel.editor_host
    assert second_editor_host is not None and second_editor_host is not first_editor_host
    _commit_area(second_editor_host)
    assert presenter.logic[first_id].draft.values == first_selected
    assert presenter.logic[second_id].draft.values != second_before


def test_pointing_a_panel_at_a_signal_that_never_published_is_refused(
    presenter, session
) -> None:
    node, snapshot = _one_shot(session)
    binding = presenter.add_panel(node.signal_key("frames"), snapshot)
    card = presenter.view.cards[0]

    card.signal_picked.emit("@logic/nobody/frames")

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

    binding.port.reject(object(), RuntimeError("the renderer refused this frame"))
    presenter.beat()

    card = presenter.view.cards[0]
    assert "refused this frame" in card.status[0]
    assert card.status[1] is True
    assert any("refused this frame" in text for _severity, text in presenter.view.status)


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

    asked: list = []
    presenter._choose_signal = lambda rows: asked.append(rows) or None

    _node, snapshot = _one_shot(session)
    before = len(presenter.panels)

    binding = presenter.add_selected_panel("curve")
    assert binding is not None and binding.kind == "curve"
    assert len(presenter.panels) == before + 1
    assert not asked, "the kind is chosen beside the button, not in a dialog"

    # A kind this data cannot be drawn as is refused with the reason.
    assert presenter.add_selected_panel("pulse_timeline") is None
    assert "pulse timeline" in presenter.view.status[-1][1]


def test_add_panel_before_anything_publishes_says_so(presenter) -> None:
    """A board with no data cannot open a panel onto nothing."""

    assert presenter.add_selected_panel("image") is None
    assert "published" in presenter.view.status[-1][1]


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
    assert presenter.retarget_panel(binding.panel_id, other.signal_key("frames")) is True
    assert presenter.panels[binding.panel_id].kind == "curve"


def test_panel_edit_surface_comes_from_the_current_plot_host(presenter, session) -> None:
    """The editor sees plot-declared choices/bounds even with no authored overrides."""

    node, snapshot = _one_shot(session)
    binding = presenter.add_panel(
        node.signal_key("frames"), snapshot, kind="image"
    )

    surface = binding.parameter_surface
    assert presenter.view.panel_parameter_surfaces[binding.panel_id] is surface
    semantic = {field["key"]: field for field in surface["semantic"]}
    display = {field["key"]: field for field in surface["display"]}
    assert "kind" not in semantic, "plot kind is fixed at Add Panel"
    assert {"x", "y", "reduction"}.issubset(semantic)
    assert semantic["x"]["choices"] and semantic["x"]["value"] is not None
    assert display["colormap"]["kind"] == "choice"
    assert display["colormap"]["choices"]
    assert display["color_min"]["allow_none"] is True
    assert display["show_colorbar"]["kind"] == "boolean"
    assert display["colormap"]["quick"] is True
    assert display["interpolation"]["quick"] is False
    assert "site_overlay" not in display
    assert surface["site_overlay"]["choices"] == (
        ("Off", "off"),
        ("Centers", "centers"),
        ("Occupancy", "occupancy"),
    )
    fit_choices = dict(surface["fit"][0]["choices"])
    assert "anisotropic_gaussian_center" in fit_choices.values()

    presenter.edit_panel(binding.panel_id)
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
    assert binding.state.display["colormap"] == colormap
    description = presenter._plot_operation_value(binding.host.describe_display())
    assert description.display_state.values["colormap"] == colormap
    editor_description = presenter._plot_operation_value(
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
    semantic_field = next(
        field
        for field in first.parameter_surface["semantic"]
        if not isinstance(field["value"], (str, bool, int, float, type(None)))
    )
    semantic_key = semantic_field["key"]
    semantic_value = semantic_field["value"]
    presenter.resize_panel(first.panel_id, "4x4")
    presenter.set_panel_interval(first.panel_id, 800)
    presenter.update_panel_state(
        first.panel_id,
        {
            "semantic": {semantic_key: semantic_value},
            "display": {"show_colorbar": False},
            "fit": {"model": "gaussian"},
            "site_overlay": "centers",
        },
    )
    second = presenter.add_panel(signal, snapshot, title="again")
    logic_id = presenter.add_logic(
        "camera_measurement",
        values={"exposure_seconds": 0.037, "repeat": 0},
        device_keys={"camera": "camera"},
    )
    assert presenter.logic[logic_id].host is None

    import json

    document = json.loads(json.dumps(presenter.layout()))
    assert document["format"] == presenter.LAYOUT_FORMAT
    assert [panel["title"] for panel in document["panels"]] == ["camera", "again"]
    assert document["panels"][0] == {
        "signal": signal, "title": "camera", "kind": "image",
        "size": "4x4", "interval_ms": 800,
        "semantic": {semantic_key: str(semantic_value)},
        "display": {"show_colorbar": False},
        "fit": {"model": "gaussian"}, "site_overlay": "centers",
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
        }
    ]

    presenter.remove_panel(first.panel_id)
    presenter.remove_panel(second.panel_id)
    assert presenter.panels == {}

    assert presenter.apply_layout(document) is True
    restored = list(presenter.panels.values())
    assert [binding.title for binding in restored] == ["camera", "again"]
    assert restored[0].kind == "image"
    assert restored[0].size == "4x4"
    assert restored[0].port.display_interval_ms == 800
    assert restored[0].state.semantic == {semantic_key: semantic_value}
    assert restored[0].state.display == {"show_colorbar": False}
    assert restored[0].state.fit == {"model": "gaussian"}
    assert restored[0].state.site_overlay == "centers"
    assert restored[0].panel_id != first.panel_id, "an id is never handed out twice"
    restored_logic = presenter.logic[logic_id]
    assert restored_logic.host is None and restored_logic.node is None
    assert restored_logic.draft.values["exposure_seconds"] == 0.037
    assert restored_logic.draft.device_keys == {"camera": "camera"}


def test_a_board_naming_a_signal_nobody_publishes_comes_back_in_part(
    presenter, session
) -> None:
    """Three quarters of an afternoon's work is worth more than a refusal."""

    node, snapshot = _one_shot(session)
    signal = node.signal_key("frames")
    presenter.add_panel(signal, snapshot, title="here")
    document = presenter.layout()
    document["panels"].append(
        {"signal": "nobody.publishes.this", "title": "gone", "kind": "", "size": "",
         "interval_ms": 200}
    )

    assert presenter.apply_layout(document) is True
    assert [binding.title for binding in presenter.panels.values()] == ["here"]
    assert any("nobody.publishes.this" in text for _severity, text in presenter.view.status)


def test_a_file_that_is_not_a_board_is_refused_by_name(presenter) -> None:
    assert presenter.apply_layout({"format": "zlc.figure/v1"}) is False
    assert any("not a saved board" in text for _severity, text in presenter.view.status)


def test_panel_edit_projects_the_direct_producer_and_apply_uses_start(
    presenter, session, monkeypatch
) -> None:
    node_id = presenter.add_logic("camera_measurement")
    node, snapshot = _one_shot(session, producer=node_id)
    panel = presenter.add_panel(node.signal_key("frames"), snapshot, kind="image")

    assert presenter.edit_panel(panel.panel_id) is True
    projection = presenter.view.panel_editors[panel.panel_id]
    assert projection["producer_node_id"] == node_id
    assert projection["producer_logic"] == presenter.logic_editor_projection(node_id)

    started: list[str] = []
    monkeypatch.setattr(
        presenter,
        "start_logic",
        lambda selected: started.append(str(selected)) or True,
    )
    presenter.view.panel_producer_apply_requested.emit(panel.panel_id)
    assert started == [node_id]
