"""A saved figure can be opened again and read.

The archive always carried everything.  What was missing was anyone able to
open it: read_archive returned a nested JSON document, and answering "what was
the apparatus doing" meant reading that document by eye.

Driven end to end -- a real run, saved through the real session, reopened in a
fresh reader -- because the failure this guards against is precisely that the
writer and the reader agree with each other and not with the file.
"""

from __future__ import annotations

import os
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

from zlc_atom.nodes.camera_measurement.measurement import (
    CameraMeasurementNode,
    CameraMeasurementRequest,
)
from zlc_data.figure_archive import read_archive, read_dataset
from zlc_workbench.panel_save import capture_run_chain, save_panel_figure
from zlc_workbench.apps.task_console import build_panel_host
from zlc_workbench.panel_state import (
    PanelFrozenData,
    PanelState,
    panel_state_from_description,
)
from zlc_workbench.session import ExperimentSession
from zlc_workbench.viewer import FigureViewerPresenter, describe_archive
from zlc_data import (
    AxisId,
    AxisSpec,
    DatasetSchema,
    SITE,
    ValidityContract,
    ValueSchema,
    owned_snapshot_from_arrays,
)
from zlc_plot import AxisRef, NumericRange, SelectorKind
from zlc_plot.primitives import ImageFrame, ImagePointOverlay
from zlc_plot.selectors import RectangleRange, SelectorState
from pulse_fixtures import CAMERA_WINDOWS, PULSE_NAME, write_ordinary_pulse


def _frozen_surface(
    state: PanelState,
    plot_input: object,
    *,
    publication: object | None = None,
    lineage: object = None,
    overlay: object = None,
    viewport: RectangleRange | None = None,
    selectors: object = (),
) -> PanelFrozenData:
    host = build_panel_host(plot_input, state)
    try:
        operation = host.configure(
            viewport=viewport,
            classifier_thresholds=state.classifier_thresholds,
            facet_focus=state.focused_cell,
            selectors=selectors,
            fit=state.fit,
            fit_live=False,
        ).result()
        description = operation.value
        target = panel_state_from_description(state, description)
    finally:
        host.close()
    return PanelFrozenData(
        publication,
        plot_input,
        target,
        description,
        {} if lineage is None else lineage,
        {} if overlay is None else overlay,
    )


@pytest.mark.parametrize("cell_kind", ("curve", "image", "histogram"))
def test_saved_panel_state_keeps_every_public_facet_cell_kind(cell_kind) -> None:
    state = PanelState(
        signal="report/distribution",
        kind="facet_grid",
        cell_kind=cell_kind,
        size="4x4",
        interval_ms=400,
        title="Calibration report",
        published_outputs={"roi_mean": True},
        focused_cell=1,
    )
    restored = PanelState.from_document(state.document())

    assert restored == state


def test_panel_state_rejects_incomplete_or_historical_documents() -> None:
    with pytest.raises(ValueError, match="panel state fields differ"):
        PanelState.from_document({"signal": "frame", "site_overlay": "off"})


class _Signal:
    def __init__(self) -> None:
        self._listeners: list = []

    def connect(self, listener) -> None:
        self._listeners.append(listener)

    def emit(self, *args) -> None:
        for listener in list(self._listeners):
            listener(*args)


class _ViewerView:
    """The viewer HANDLE's contract, with Qt taken out."""

    def __init__(self) -> None:
        self.path_committed = _Signal()
        self.add_panel_requested = _Signal()
        self.panel_state_changed = _Signal()
        self.panel_remove_requested = _Signal()
        self.panel_edit_requested = _Signal()
        self.panel_order_committed = _Signal()
        self.panel_editor_closed = _Signal()
        self.save_image_requested = _Signal()
        self.close_requested = _Signal()
        self.tabs: tuple = ()
        self.flow: object = {"nodes": (), "edges": ()}
        self.surface = None
        self.title = ""
        self.size = ""
        self.path = ""
        self.status: list[tuple[str, bool]] = []
        self.panel_sizes: tuple[str, ...] = ()
        self.panel_intervals: tuple[int, ...] = ()
        self.panel_kinds: tuple = ()
        self.grid_cell_kinds: tuple = ()
        self.panels: dict[str, dict] = {}
        self.editors: dict[str, object] = {}
        self.dpr = 1.0

    def device_pixel_ratio(self) -> float:
        return float(self.dpr)

    def has_panel_editor(self, panel_id: str) -> bool:
        return str(panel_id) in self.editors

    def set_panel_sizes(self, sizes, default_size) -> None:
        self.panel_sizes = tuple(str(value) for value in sizes)
        self.panel_default_size = str(default_size)

    def set_panel_kinds(self, kinds, _default="") -> None:
        self.panel_kinds = tuple(kinds)

    def set_panel_intervals(self, intervals, default) -> None:
        self.panel_intervals = tuple(intervals)
        self.panel_default_interval = int(default)

    def set_grid_cell_kinds(self, kinds) -> None:
        self.grid_cell_kinds = tuple(kinds)

    def add_panel(self, panel_id, title) -> None:
        self.panels[str(panel_id)] = {"title": str(title)}

    def remove_panel(self, panel_id) -> None:
        self.panels.pop(str(panel_id), None)

    def set_panel_order(self, order) -> None:
        self.panel_order = tuple(order)

    def set_panel_signal_choices(self, panel_id, groups, **values) -> None:
        self.panels[str(panel_id)].update(
            signal_groups=tuple(groups),
            **values,
        )

    def set_panel_publishers(self, _publishers) -> None:
        pass

    def panel_ids(self) -> tuple[str, ...]:
        return tuple(self.panels)

    def set_panel_selectors_enabled(self, panel_id, enabled) -> None:
        self.panels[str(panel_id)]["selectors_enabled"] = bool(enabled)

    def set_panel_mutation_enabled(self, panel_id, enabled) -> None:
        self.panels[str(panel_id)]["mutation_enabled"] = bool(enabled)

    def present_panel_front(self, _panel_id, _front) -> bool:
        return True

    def set_panel_projection(self, panel_id, state, surface) -> None:
        self.panels[str(panel_id)].update(state=state, surface=surface)

    def set_panel_status(self, panel_id, text, *, error=False) -> None:
        self.panels[str(panel_id)]["status"] = (str(text), bool(error))

    def show_panel(self, panel_id, host) -> None:
        self.panels[str(panel_id)]["host"] = host
        self.surface = host

    def open_panel_editor(self, panel_id, editor, *, title="") -> None:
        del title
        self.editors[str(panel_id)] = editor

    def show_panel_editor(self, _panel_id, _host) -> None:
        pass

    def focus_panel_editor(self, panel_id) -> bool:
        return str(panel_id) in self.editors

    def close_panel_editor(self, panel_id) -> bool:
        return self.editors.pop(str(panel_id), None) is not None

    def update_panel_editor(self, panel_id, projection) -> bool:
        if str(panel_id) not in self.editors:
            return False
        self.editors[str(panel_id)] = projection
        return True

    def set_archive_info(self, tabs, graph) -> None:
        self.tabs = tuple(tabs)
        self.flow = graph

    def set_title(self, text: str) -> None:
        self.title = str(text)

    def set_path(self, path: str) -> None:
        self.path = str(path)

    def set_status(self, text: str, *, error: bool = False) -> None:
        self.status.append((str(text), bool(error)))


def _wait_until(predicate, *, timeout: float = 10.0) -> None:
    from zlc_ui.qt import ensure_qt_app

    application = ensure_qt_app(["figure-viewer-test"])
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        application.processEvents()
        time.sleep(0.005)
    assert predicate(), "timed out waiting for the FigureViewer owner turn"


def _display_description(plot_input, recipe):
    import zlc_plot

    probe = zlc_plot.open_figure_host(plot_input, recipe)
    try:
        return probe.describe_display().result().value
    finally:
        probe.close(timeout=10)


def _built_presenter(view) -> FigureViewerPresenter:
    from zlc_workbench.apps.figure_viewer import build
    from zlc_workbench.board import attach_qt_worker
    from zlc_ui.qt import ensure_qt_app

    ensure_qt_app(["test-built-figure-viewer"])
    run_off_thread, close_worker = attach_qt_worker("test-built-figure-viewer")
    return build(
        view,
        run_off_thread=run_off_thread,
        close_worker=close_worker,
        request_close=lambda: None,
    )


def _close_presenter(presenter: FigureViewerPresenter) -> None:
    _wait_until(presenter.close)


def _active_record(presenter: FigureViewerPresenter) -> dict[str, object]:
    record = presenter.panels[presenter._active_panel_id]
    return {
        "host": record.host,
        "state": record.state,
        "surface": record.parameter_surface,
        "plot_input": (
            None if record.accepted_surface is None else record.accepted_surface.plot_input
        ),
    }


def _formal_viewer_window(saved, monkeypatch, host_factory):
    pytest.importorskip("PyQt5")
    from PyQt5 import QtCore, QtWidgets
    from zlc_ui.qt import ensure_qt_app
    from zlc_workbench.apps.figure_viewer import create_window
    import zlc_plot

    application = ensure_qt_app(["formal-figure-viewer"])
    host = host_factory(application, QtCore, QtWidgets)
    monkeypatch.setattr(
        zlc_plot.RasterPlotHost,
        "from_plot",
        staticmethod(lambda *_args, **_kwargs: host),
    )
    path, _snapshot = saved
    window = create_window(path=path, window_ratio=0.25)
    owner_turns: list[bool] = []
    timer = QtCore.QTimer()
    timer.setInterval(10)
    timer.timeout.connect(lambda: owner_turns.append(True))
    timer.start()
    return application, QtCore, window, owner_turns, timer


@pytest.fixture
def saved(tmp_path):
    """One real typed run in the formal figure archive."""

    write_ordinary_pulse(tmp_path)
    session = ExperimentSession.open(tmp_path, template="virtual")
    try:
        pulse = session.load_pulse(PULSE_NAME)
        node = CameraMeasurementNode(
            camera=session.camera,
            request=CameraMeasurementRequest(
                "camera", 0.02, None, 1, CAMERA_WINDOWS, photoelectrons=False
            ),
            signal_plane=session.signal_plane,
            producer="cm",
        )
        capture = node.prepare()
        session.fire(shots=1)
        result = capture.collect()
        signal = node.signal_key("frames")
        snapshot = result.publication.value(signal).snapshot
        state = PanelState(signal, "image", "2x2", 400, "camera")
        frozen = _frozen_surface(
            state,
            snapshot,
            publication=result.publication,
            lineage=capture_run_chain(session.signal_plane, result.publication),
        )
        written = save_panel_figure(
            tmp_path / "run.png", state=state, frozen=frozen,
        )
        yield written.archive, snapshot
    finally:
        session.close()


@pytest.fixture
def presenter():
    view = _ViewerView()
    presenter = _built_presenter(view)
    try:
        yield presenter
    finally:
        _close_presenter(presenter)


def test_a_saved_dataset_comes_back_with_its_axes(saved) -> None:
    """The point of recording identity: what returns is the dataset, not numbers.

    A figure that can only be re-read as an array cannot be replotted, refitted
    or compared with a later run, which is most of why it was saved.
    """

    path, original = saved
    info, arrays = read_archive(path)
    restored = read_dataset(info, arrays, "data")

    np.testing.assert_array_equal(
        np.asarray(restored.block.values), np.asarray(original.block.values)
    )
    assert restored.block.schema == original.block.schema
    assert restored.ref.revision == original.ref.revision
    assert [axis.axis_id.value for axis in restored.block.schema.cell_schema.data_axes] == [
        axis.axis_id.value for axis in original.block.schema.cell_schema.data_axes
    ]


def test_the_description_reports_only_facts_saved_in_the_archive(saved) -> None:
    path, _snapshot = saved
    info, arrays = read_archive(path)
    description = describe_archive(info, arrays)
    tabs = dict(description.tabs)
    assert tuple(tabs) == ("Plot", "Logic", "Devices", "Flow", "Raw")

    logic = dict(tabs["Logic"])
    assert tuple(logic) == ("cm",)
    assert logic["cm"]["outputs"] == ["frames"]
    devices = dict(tabs["Devices"])
    assert len(devices) == 1
    camera = next(iter(devices.values()))
    camera_snapshot = camera["snapshots"][0]
    assert camera_snapshot["logic"] == "cm"
    assert camera_snapshot["scope"] == "run"
    assert isinstance(camera_snapshot["sequence"], int)
    working_point = camera_snapshot["snapshot"]
    assert working_point["exposure_seconds"] == pytest.approx(0.02)
    assert working_point["roi_shape_yx"] == [96, 128]

    plot_rows = dict(tabs["Plot"])
    assert "data" in plot_rows and "uint16" in plot_rows["data"]
    assert plot_rows["plot data"].startswith("image")

    # Task-generated report Figures have no Runtime publication to invent a
    # DAG from.  Their frozen source run record still projects as one real
    # Logic node with its actual devices, rather than an empty Viewer.
    task_info = {
        **info,
        "sections": {
            **info["sections"],
            "lineage": {"root": None, "nodes": [], "device_settings": []},
            "source": {
                "task": "calibration",
                "report": "site_map",
                "run_record": {
                    "request": {"repeats": 12},
                    "actual_devices": {
                        "camera": dict(working_point),
                        "sequencer": {"clock_hz": 50_000_000.0},
                    },
                },
            },
        },
    }
    task = describe_archive(task_info, arrays)
    assert tuple(dict(dict(task.tabs)["Logic"])) == ("calibration",)
    assert tuple(dict(dict(task.tabs)["Devices"])) == ("camera", "sequencer")
    assert [node["title"] for node in task.flow["nodes"]] == [
        "calibration", "camera", "sequencer"
    ]
    assert len(task.flow["edges"]) == 2


def test_the_flow_projection_is_the_saved_exact_node_edge_graph(saved) -> None:
    path, _snapshot = saved
    description = describe_archive(*read_archive(path))
    nodes = {node["id"]: node for node in description.flow["nodes"]}
    edges = description.flow["edges"]
    assert {node["kind"] for node in nodes.values()} == {"logic", "device"}
    logic = next(node for node in nodes.values() if node["kind"] == "logic")
    camera = next(node for node in nodes.values() if node["kind"] == "device")
    assert logic["title"] == "cm" and "frames" in logic["subtitle"]
    assert camera["title"] == "camera"
    assert any(
        edge["source"] == camera["id"] and edge["target"] == logic["id"]
        for edge in edges
    )

    # A convergent DAG keeps its shared event and shared device unique.
    info, arrays = read_archive(path)
    camera_record = next(
        node["record"]
        for node in info["sections"]["lineage"]["nodes"]
        if node["record"].get("node") == "cm"
    )
    raw_nodes = [
        {
            "id": "source",
            "event": {"stream": "@logic/cm/frames", "generation": "g", "sequence": 1},
            "parents": [],
            "signals": ["@logic/cm/frames"],
            "record": camera_record,
            "event_record": {"device_snapshots": {"camera": {"gain": 2.0}}},
        },
        *(
            {
                "id": name,
                "event": {"stream": f"@logic/{name}/value", "generation": "g", "sequence": sequence},
                "parents": ["source"],
                "signals": [f"@logic/{name}/value"],
                "record": {"node": name, "parameters": {}},
                "event_record": {},
            }
            for name, sequence in (("left", 2), ("right", 3))
        ),
        {
            "id": "merge",
            "event": {"stream": "@logic/merge/value", "generation": "g", "sequence": 4},
            "parents": ["left", "right"],
            "signals": ["@logic/merge/value"],
            "record": {"node": "merge", "parameters": {}},
            "event_record": {},
        },
    ]
    diamond_info = {
        **info,
        "sections": {
            **info["sections"],
            "lineage": {"root": "merge", "nodes": raw_nodes, "device_settings": []},
        },
    }
    diamond = describe_archive(diamond_info, arrays).flow
    assert sum(node["kind"] == "logic" for node in diamond["nodes"]) == 4
    assert sum(node["kind"] == "device" for node in diamond["nodes"]) == 1
    assert sum(edge["kind"] == "causal" for edge in diamond["edges"]) == 4
    diamond_devices = dict(describe_archive(diamond_info, arrays).tabs)["Devices"]
    camera_snapshots = dict(diamond_devices)["camera"]["snapshots"]
    assert [item["scope"] for item in camera_snapshots] == ["run", "event"]


def test_the_raw_tab_is_the_typed_document_not_a_node_probe(saved) -> None:
    """Every projected tab is a reading; this is the document itself."""

    path, _snapshot = saved
    info, arrays = read_archive(path)
    raw = dict(describe_archive(info, arrays).tabs)["Raw"]
    labels = {label for label, _value in raw}
    assert "source.signal" in labels
    assert any(label.startswith("lineage.nodes") for label in labels)
    # The dataset manifest is part of the document too, however verbose.
    assert any(label.startswith("dataset.data.") for label in labels)


def test_opening_shows_the_figure_and_its_record(presenter, saved) -> None:
    path, _snapshot = saved
    presenter.view.path_committed.emit(str(path))
    _wait_until(lambda: not presenter._busy)

    assert presenter.description is not None, presenter.view.status
    assert presenter.view.title == "run.png"
    assert presenter.view.path == str(path), "the File field cannot stay empty"
    assert dict(presenter.view.tabs)["Logic"]
    _wait_until(
        lambda: (
            presenter.beat()
            or _active_record(presenter)["host"] is not None
        )
    )
    assert presenter.view.surface is not None, presenter.view.status
    assert presenter.view.status[-1] == ("showing @figure/1/data", False)
    assert presenter.view.flow["nodes"]

    panel_id = presenter._active_panel_id
    presenter.resize_panel(panel_id, "4x4")
    _wait_until(
        lambda: (
            presenter.beat()
            or _active_record(presenter)["state"].size == "4x4"
        )
    )
    assert _active_record(presenter)["host"].logical_size is not None
    presenter.view.panel_edit_requested.emit(panel_id)
    editor = presenter.view.editors[panel_id]
    assert editor["state"]["signal"] == "@figure/1/data"
    assert editor["live"] is False
    boolean = next(
        field
        for field in presenter.panels[panel_id].parameter_surface["display"]
        if field["kind"] == "boolean"
    )
    presenter.view.panel_state_changed.emit(
        panel_id,
        {"display": {boolean["key"]: not boolean["value"]}},
    )
    _wait_until(
        lambda: (
            presenter.beat()
            or presenter.panels[panel_id].state.display.get(boolean["key"])
            is not boolean["value"]
        )
    )
    assert presenter.panels[panel_id].state.display[boolean["key"]] is not boolean["value"]
    assert presenter.view.editors[panel_id]["state"]["display"][boolean["key"]] is not boolean["value"]

    presenter.view.add_panel_requested.emit("curve")
    assert len(presenter.panels) == 2
    added = next(key for key in presenter.panels if key != panel_id)
    assert presenter.panels[added].state.kind == "curve"
    assert presenter.panels[added].state.signal == ""
    assert presenter.panels[added].host is None
    presenter.view.panel_remove_requested.emit(added)
    assert tuple(presenter.panels) == (panel_id,)


def test_a_file_that_cannot_be_read_is_answered_not_raised(presenter, tmp_path) -> None:
    """An operator types paths.  Most of what they type is not an archive."""

    stray = tmp_path / "notes.txt"
    stray.write_text("not an archive", encoding="utf-8")
    presenter.view.path_committed.emit(str(stray))
    _wait_until(lambda: not presenter._busy)

    assert presenter.description is None
    assert presenter.view.status[-1][1] is True
    assert "notes.txt" in presenter.view.status[-1][0]


def test_formal_window_slow_failed_open_keeps_turning_and_retains_the_last_figure(
    saved,
    monkeypatch,
) -> None:
    import zlc_workbench.viewer as viewer_module
    from zlc_plot import read_figure_plot

    path, _snapshot = saved
    info, arrays = read_archive(path)
    plot_input, recipe = read_figure_plot(info, arrays, "data")
    description = _display_description(plot_input, recipe)

    def host_factory(_application, _QtCore, QtWidgets):
        surface = QtWidgets.QLabel("last accepted figure")
        return SimpleNamespace(
            configure=lambda **_kwargs: None,
            describe_display=lambda: SimpleNamespace(value=description),
            qt_widget=lambda: surface,
            close=lambda: None,
        )

    _application, _QtCore, window, owner_turns, timer = _formal_viewer_window(
        saved, monkeypatch, host_factory
    )
    try:
        _wait_until(lambda: window.presenter.description is not None)
        accepted = (
            window.presenter.path,
            window.presenter.description,
            _active_record(window.presenter)["host"],
        )
        original_read = viewer_module.read_archive

        def slow_failed_read(candidate):
            if Path(candidate).name == "broken.npz":
                time.sleep(0.25)
                raise OSError("slow unreadable archive")
            return original_read(candidate)

        monkeypatch.setattr(viewer_module, "read_archive", slow_failed_read)
        started = time.monotonic()
        window.path_committed.emit(str(path.with_name("broken.npz")))
        submitted_in = time.monotonic() - started
        _wait_until(lambda: not window.presenter._busy)
        timer.stop()

        assert submitted_in < 0.05, "the File commit performed archive I/O"
        assert len(owner_turns) >= 5, "Qt stopped turning during archive I/O"
        assert (
            window.presenter.path,
            window.presenter.description,
            _active_record(window.presenter)["host"],
        ) == accepted
    finally:
        timer.stop()
        window.close()
        _wait_until(lambda: not window.is_visible())


def test_formal_window_waits_for_guarded_host_work_without_blocking_or_hiding(
    saved,
    monkeypatch,
) -> None:
    from threading import Event

    configured = Event()
    release_configure = Event()
    closing = Event()
    release_close = Event()

    def host_factory(application, QtCore, QtWidgets):
        surface = QtWidgets.QLabel("guarded figure")

        def wait_off_owner(started: Event, release: Event):
            def result(*_args, **_kwargs):
                assert QtCore.QThread.currentThread() != application.thread()
                started.set()
                assert release.wait(5.0), "test never released guarded host work"
                return None

            return SimpleNamespace(result=result)

        return SimpleNamespace(
            configure=lambda **_kwargs: wait_off_owner(configured, release_configure),
            qt_widget=lambda: surface,
            close=lambda: wait_off_owner(closing, release_close).result(),
        )

    application, _QtCore, window, owner_turns, timer = _formal_viewer_window(
        saved, monkeypatch, host_factory
    )
    try:
        _wait_until(configured.is_set)
        _wait_until(lambda: len(owner_turns) >= 3)
        assert window.is_visible()
        assert window.presenter.description is None

        window.close()
        application.processEvents()
        assert window.is_visible(), "pending host work was reported closed"
        release_configure.set()
        _wait_until(closing.is_set)
        assert window.is_visible(), "the host had not actually retired"
        _wait_until(lambda: len(owner_turns) >= 6)
        release_close.set()
        _wait_until(lambda: not window.is_visible())
    finally:
        timer.stop()
        release_configure.set()
        release_close.set()
        if window.is_visible():
            window.close()
            _wait_until(lambda: not window.is_visible())


def test_the_projection_needs_no_session_and_no_qt() -> None:
    """It answers in a notebook too, which is where most reading happens."""

    import ast

    import zlc_workbench.viewer as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    imported = {
        node.module
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert not any(name.startswith(("PyQt5", "zlc_atom")) for name in imported), imported


def test_saving_an_image_works_however_the_archive_was_spelled(presenter, saved) -> None:
    """A relative Open spelling still establishes one absolute archive home."""

    path, _snapshot = saved
    here = os.getcwd()
    os.chdir(path.parent)
    try:
        presenter.open(path.name)
        _wait_until(lambda: not presenter._busy)
        assert presenter.path.is_absolute(), "an archive's location is absolute"
        _wait_until(
            lambda: (
                presenter.beat()
                or _active_record(presenter)["host"] is not None
            )
        )
        presenter.save_image()
        _wait_until(lambda: not presenter._busy)
    finally:
        os.chdir(here)

    written = next(path.parent.glob("run-data*.png"))
    assert Path(written).is_file()
    assert written.parent == path.parent


def test_panel_save_reopens_fixed_kind_state_fit_and_typed_image_overlay(
    saved,
    tmp_path,
) -> None:
    """The archive is the redraw input; calibration is not reopened."""

    _old_path, snapshot = saved
    state = PanelState(
        signal="@logic/occupancy/frame_judged",
        kind="image",
        size="4x4",
        interval_ms=800,
        title="site occupancy",
        semantic={"reduction": "mean"},
        display={"show_colorbar": False},
        fit={
            "model": "anisotropic_gaussian_center",
            "fixed": {"center_x": 1.0},
            "initial": {"radius_x": 10.0, "radius_y": 10.0},
        },
        overlay_signal="@logic/occupancy/occupied",
    )
    source_schema = snapshot.block.schema
    site_axis = AxisSpec(AxisId("site"), "site", SITE, 2, (0, 1))
    status_schema = DatasetSchema(
        source_schema.repeat_axis,
        source_schema.point_table,
        source_schema.grid_topology,
        ValueSchema(
            (site_axis,),
            ValidityContract.value(),
            np.dtype(np.bool_),
            "1",
        ),
    )
    status_shape = status_schema.physical_shape
    occupied = owned_snapshot_from_arrays(
        status_schema,
        np.broadcast_to(np.asarray([False, True]), status_shape),
        snapshot.ref.revision,
        validity=np.ones(status_shape, dtype=np.bool_),
    )
    overlay = ImagePointOverlay(
        7,
        np.asarray(((2.5, 3.5), (7.5, 9.5))),
        ("site-0", "site-1"),
        ("0", "1"),
        None,
        occupied,
    )
    frozen = _frozen_surface(
        state,
        ImageFrame(snapshot, overlay),
        publication=None,
        overlay={"overlay_signal": state.overlay_signal},
        selectors=(
            SelectorState(
                SelectorKind.AREA,
                RectangleRange(
                    NumericRange(20.0, 60.0),
                    NumericRange(15.0, 55.0),
                ),
            ),
        ),
    )

    written = save_panel_figure(
        tmp_path / "panel",
        state=state,
        frozen=frozen,
    )
    with np.load(written.archive, allow_pickle=False) as payload:
        assert "data.overlay.coordinates" in payload.files
        assert "data.overlay.status" in payload.files

    real_view = _ViewerView()
    real_view.dpr = 1.75
    real_presenter = _built_presenter(real_view)
    try:
        real_presenter.open(str(written.archive))
        _wait_until(lambda: not real_presenter._busy)
        assert real_presenter.description is not None, real_view.status
        _wait_until(
            lambda: (
                real_presenter.beat()
                or _active_record(real_presenter)["host"] is not None
            )
        )
        active = _active_record(real_presenter)
        host = active["host"]
        assert host is not None, real_view.status
        restored_frame = active["plot_input"]
        assert isinstance(restored_frame, ImageFrame)
        np.testing.assert_array_equal(
            restored_frame.overlay.coordinates,
            overlay.coordinates,
        )
        assert (
            restored_frame.overlay.status.block.schema
            == overlay.status.block.schema
        )
        np.testing.assert_array_equal(
            restored_frame.overlay.status.block.values,
            overlay.status.block.values,
        )
        np.testing.assert_array_equal(
            restored_frame.overlay.status.block.validity,
            overlay.status.block.validity,
        )
        assert host.describe_display().result().value.spec.kind.value == "image"
        assert active["state"].selector
        assert host.selector_state(SelectorKind.AREA).result().value.value == (
            RectangleRange(
                NumericRange(20.0, 60.0),
                NumericRange(15.0, 55.0),
            )
        )
        assert host.wait_for_front(timeout=5.0).device_pixel_ratio == 1.75
        assert host._session._renderer.primary_axes.get_title() == ""
        # And the authored appearance really is on the built host.
        described = host.describe_display().result().value
        assert described.display_state.values["show_colorbar"] is False
        source_signal = active["state"].signal
        source_panel_id = real_presenter._active_panel_id
        _wait_until(
            lambda: (
                real_presenter.beat()
                or any(
                    row.source_name == source_signal
                    for row in real_presenter._signal_plane.describe_signals()
                )
            )
        )
        derived_roi = next(
            row.name
            for row in real_presenter._signal_plane.describe_signals()
            if row.source_name == source_signal and row.name.endswith("/roi_frame")
        )

        # Add Panel authors an empty fixed-kind card first.  It cannot reject
        # the archive Dataset before the operator has selected a signal/fates.
        existing = set(real_presenter.panels)
        real_view.add_panel_requested.emit("image")
        added_id = next(iter(set(real_presenter.panels) - existing))
        added = real_presenter.panels[added_id]
        assert added.state.signal == ""
        assert added.host is None
        assert any(
            source_signal == signal
            for _group, leaves in real_view.panels[added_id]["signal_groups"]
            for _label, signal in leaves
        )
        real_view.panel_state_changed.emit(added_id, {"signal": derived_roi})
        _wait_until(
            lambda: (
                real_presenter.beat()
                or (
                    real_presenter.panels[added_id].state.signal == derived_roi
                    and real_presenter.panels[added_id].host is not None
                )
            )
        )

        panel_id = source_panel_id
        center_x = float(state.fit["fixed"]["center_x"])
        # A fit edit configures the accepted common Panel host in place; it
        # must not rebuild the host merely because the window DPR snapshot
        # changed meanwhile.
        real_view.dpr = 2.25
        real_presenter.update_panel(
            panel_id,
            # x_0 is what the formula prints for center_x; the stored
            # target below still keys on the internal name.
            {"fit": {"expression": f"x_0=guess({center_x})"}},
        )
        _wait_until(
            lambda: (
                real_presenter.beat()
                or _active_record(real_presenter)["state"].fit
                == {
                    "model": "anisotropic_gaussian_center",
                    "initial": {"center_x": center_x},
                }
            )
        )
        active = _active_record(real_presenter)
        assert active["state"].fit == {
            "model": "anisotropic_gaussian_center",
            "initial": {"center_x": center_x},
        }
        assert active["host"].wait_for_front(timeout=5.0).device_pixel_ratio == 1.75

    finally:
        _close_presenter(real_presenter)


def test_panel_save_thresholds_and_viewport_reopen_in_canonical_units(tmp_path) -> None:
    """Saved V thresholds and the exact view reopen without display-unit drift."""

    from data_factory import Axis, DatasetSchema, DatasetSnapshot, PointTable
    samples = np.linspace(-3.0, 3.0, 80)
    values = np.column_stack((samples - 1.0, samples + 1.0))
    schema = DatasetSchema.create(
        Axis.create("repeat", size=len(samples)),
        PointTable.from_columns({"site": (0.0, 1.0)}),
        dtype=np.float64,
        canonical_unit="V",
        generation="annotation-unit-roundtrip",
    )
    snapshot = DatasetSnapshot(schema, values, revision=0)
    state = PanelState(
        signal="report/distribution",
        kind="facet_grid",
        cell_kind="histogram",
        size="4x4",
        interval_ms=400,
        title="unit report",
        semantic={"fate:point_coordinate:site": "facet"},
        display={"value_display_unit": "mV", "threshold_classifier": True},
        classifier_thresholds=(
            {
                "value": 1.0,
                "scope": (
                    {
                        "domain": "point_coordinate",
                        "axis_id": "site",
                        "coordinate": 0,
                    },
                ),
                "repeat_index": None,
            },
            {
                "value": 2.0,
                "scope": (
                    {
                        "domain": "point_coordinate",
                        "axis_id": "site",
                        "coordinate": 1,
                    },
                ),
                "repeat_index": None,
            },
        ),
    )
    viewport = RectangleRange(
        NumericRange(-2.0, 2.0),
        NumericRange(0.0, 40.0),
    )
    frozen = _frozen_surface(
        state,
        snapshot,
        viewport=viewport,
    )

    written = save_panel_figure(
        tmp_path / "unit-report",
        state=state,
        frozen=frozen,
    )

    view = _ViewerView()
    presenter = _built_presenter(view)
    try:
        presenter.open(str(written.archive))
        _wait_until(lambda: not presenter._busy)
        assert presenter.description is not None, view.status
        _wait_until(
            lambda: (
                presenter.beat()
                or _active_record(presenter)["host"] is not None
            )
        )
        host = _active_record(presenter)["host"]
        assert host is not None
        # Had the Viewer treated archived canonical values as display values,
        # these would be 0.001 V and 0.002 V after its mV projection.
        description = host.describe_display().result(timeout=10).value
        assert description.viewport == viewport
        assert str(description.display_state.values["value_display_unit"]) == "mV"
        host.focus_facet(0).result(timeout=10)
        assert host.selector_state(
            SelectorKind.THRESHOLD,
            display=False,
        ).result(timeout=10).value.value == 1.0
        host.focus_facet(1).result(timeout=10)
        assert host.selector_state(
            SelectorKind.THRESHOLD,
            display=False,
        ).result(timeout=10).value.value == 2.0
    finally:
        _close_presenter(presenter)


def test_viewer_reenabling_facet_fit_solves_every_cell(tmp_path) -> None:
    from data_factory import Axis, DatasetSchema, DatasetSnapshot, PointTable
    from zlc_plot.fit import FacetFitBatchResult

    x = np.linspace(-3.0, 3.0, 40)
    facets = np.repeat((0.0, 1.0), 20)
    schema = DatasetSchema.create(
        Axis.create("repeat", size=1),
        PointTable.from_columns({"x": x, "facet": facets}),
        dtype=np.float64,
        generation="viewer-facet-fit",
    )
    values = (
        2.0 * np.exp(-0.5 * ((x - 0.15) / 0.9) ** 2) + 0.2
    )[None, :]
    snapshot = DatasetSnapshot(schema, values, revision=0)
    state = PanelState(
        signal="saved/facet-curve",
        kind="facet_grid",
        cell_kind="curve",
        size="4x4",
        interval_ms=400,
        title="facet curve",
        semantic={
            "fate:point_coordinate:facet": "facet",
            "fate:point_coordinate:x": "x",
            "fate:repeat": "reduce",
            "reduction": "mean",
        },
        fit={"model": "gaussian_offset", "fit_all_facets": True},
    )
    frozen = _frozen_surface(state, snapshot)
    written = save_panel_figure(
        tmp_path / "facet-fit",
        state=state,
        frozen=frozen,
    )
    view = _ViewerView()
    presenter = _built_presenter(view)
    try:
        presenter.open(str(written.archive))
        _wait_until(lambda: not presenter._busy)
        _wait_until(
            lambda: (
                presenter.beat()
                or _active_record(presenter)["host"] is not None
            )
        )
        panel_id = presenter._active_panel_id
        host = _active_record(presenter)["host"]
        initial = host._session.last_fit
        assert isinstance(initial, FacetFitBatchResult)
        assert len(initial.results) == 2

        presenter.update_panel(panel_id, {"fit": {"model": None}})
        _wait_until(
            lambda: (
                presenter.beat()
                or not _active_record(presenter)["state"].fit
            )
        )
        presenter.update_panel(
            panel_id,
            {"fit": {"model": "gaussian_offset"}},
        )

        def all_fitted() -> bool:
            presenter.beat()
            result = _active_record(presenter)["host"]._session.last_fit
            return isinstance(result, FacetFitBatchResult) and len(result.results) == 2

        _wait_until(all_fitted)
        restored = _active_record(presenter)["host"]._session.last_fit
        assert isinstance(restored, FacetFitBatchResult)
        assert len(restored.overlays) == 2
    finally:
        _close_presenter(presenter)


def test_panel_save_reports_that_the_archive_survived_an_image_failure(
    saved,
    tmp_path,
    monkeypatch,
) -> None:
    import zlc_plot.figure_artifact as figure_module
    _old_path, snapshot = saved
    state = PanelState("camera", "image", "2x2", 400, "camera")
    frozen = _frozen_surface(state, snapshot)

    def fail_image(_path) -> None:
        raise OSError("renderer failed")

    monkeypatch.setattr(
        figure_module,
        "build_figure_host",
        lambda *_args, **_kwargs: SimpleNamespace(
            configure=lambda **_kwargs: SimpleNamespace(
                result=lambda: SimpleNamespace(value=frozen.description)
            ),
            save=fail_image,
            close=lambda: None,
        ),
    )

    with pytest.raises(RuntimeError, match="archive.*saved.*image") as failure:
        save_panel_figure(
            tmp_path / "failed-image",
            state=state,
            frozen=frozen,
        )

    archive = tmp_path / "failed-image.npz"
    assert archive.exists()
    assert str(archive) in str(failure.value)


def test_panel_save_does_not_render_when_the_archive_fails(
    saved,
    tmp_path,
    monkeypatch,
) -> None:
    import zlc_plot.figure_artifact as figure_module

    _old_path, snapshot = saved
    state = PanelState("camera", "image", "2x2", 400, "camera")
    frozen = _frozen_surface(state, snapshot)

    def fail_archive(*_args, **_kwargs):
        raise OSError("archive disk full")

    monkeypatch.setattr(figure_module, "atomic_write_file", fail_archive)
    with pytest.raises(OSError, match="archive disk full"):
        save_panel_figure(
            tmp_path / "failed-archive",
            state=state,
            frozen=frozen,
        )

    assert not (tmp_path / "failed-archive.png").exists()
