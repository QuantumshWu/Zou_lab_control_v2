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
from zlc_workbench.panel_state import PanelFrozenData, PanelState
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
from zlc_plot.selectors import RectangleRange
from pulse_fixtures import CAMERA_WINDOWS, PULSE_NAME, write_ordinary_pulse


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
        selector={"kind": "area", "x": (0.1, 0.9)},
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
        self.lineage: tuple = ()
        self.surface = None
        self.title = ""
        self.size = ""
        self.path = ""
        self.status: list[tuple[str, bool]] = []
        self.panel_sizes: tuple[str, ...] = ()
        self.panel_kinds: tuple = ()
        self.grid_cell_kinds: tuple = ()
        self.panels: dict[str, dict] = {}
        self.editors: dict[str, object] = {}

    def set_panel_sizes(self, sizes, default_size) -> None:
        self.panel_sizes = tuple(str(value) for value in sizes)
        self.panel_default_size = str(default_size)

    def set_panel_kinds(self, kinds) -> None:
        self.panel_kinds = tuple(kinds)

    def set_grid_cell_kinds(self, kinds) -> None:
        self.grid_cell_kinds = tuple(kinds)

    def add_panel(self, panel_id, title) -> None:
        self.panels[str(panel_id)] = {"title": str(title)}

    def remove_panel(self, panel_id) -> None:
        self.panels.pop(str(panel_id), None)

    def set_panel_order(self, order) -> None:
        self.panel_order = tuple(order)

    def set_panel_datasets(self, panel_id, datasets, current="") -> None:
        self.panels[str(panel_id)].update(
            datasets=tuple(datasets), current=str(current)
        )

    def set_panel_projection(self, panel_id, state, surface) -> None:
        self.panels[str(panel_id)].update(state=state, surface=surface)

    def set_panel_status(self, panel_id, text, *, error=False) -> None:
        self.panels[str(panel_id)]["status"] = (str(text), bool(error))

    def show_panel(self, panel_id, host) -> None:
        self.panels[str(panel_id)]["host"] = host
        self.surface = host

    def open_panel_editor(self, panel_id, editor, *, title) -> None:
        self.editors[str(panel_id)] = editor

    def close_panel_editor(self, panel_id) -> bool:
        return self.editors.pop(str(panel_id), None) is not None

    def update_panel_editor(self, panel_id, projection) -> bool:
        if str(panel_id) not in self.editors:
            return False
        self.editors[str(panel_id)] = projection
        return True

    def set_info(self, tabs) -> None:
        self.tabs = tuple(tabs)

    def set_lineage_tree(self, tree) -> None:
        self.lineage = tuple(tree)

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


def _presenter(view, *, make_host) -> FigureViewerPresenter:
    from zlc_workbench.board import attach_qt_worker
    from zlc_ui.qt import ensure_qt_app

    ensure_qt_app(["test-figure-viewer"])
    run_off_thread, close_worker = attach_qt_worker("test-figure-viewer")
    return FigureViewerPresenter(
        view,
        make_host=make_host,
        run_off_thread=run_off_thread,
        close_worker=close_worker,
        request_close=lambda: None,
    )


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
    return presenter.panels[presenter._active_panel_id]


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
        frozen = PanelFrozenData(
            signal, result.publication, snapshot,
            lineage=capture_run_chain(session.signal_plane, result.publication),
        )
        written = save_panel_figure(
            tmp_path / "run.png", state=state, frozen=frozen, viewport=None,
        )
        yield written.archive, snapshot
    finally:
        session.close()


@pytest.fixture
def presenter():
    plot = pytest.importorskip("zlc_plot")
    view = _ViewerView()

    def make_host(plot_input, _name, recipe):
        return plot.open_figure_host(plot_input, recipe)

    presenter = _presenter(view, make_host=make_host)
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
    description = describe_archive(*read_archive(path))
    tabs = dict(description.tabs)
    assert tuple(tabs) == ("Plot", "Measurement", "Device", "Flow", "Raw")

    measurement = dict(tabs["Measurement"])
    assert measurement
    assert dict(tabs["Device"]) == {}

    plot_rows = dict(tabs["Plot"])
    assert "data" in plot_rows and "uint16" in plot_rows["data"]
    assert plot_rows["plot data"].startswith("image")


def test_the_flow_projection_is_the_saved_exact_lineage_tree(saved) -> None:
    path, _snapshot = saved
    description = describe_archive(*read_archive(path))
    assert description.lineage
    assert "@logic/cm/frames" in description.lineage[0][0]


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
    assert dict(presenter.view.tabs)["Measurement"]
    assert presenter.view.surface is not None, presenter.view.status
    assert presenter.view.status[-1] == ("showing data", False)
    assert presenter.view.lineage

    panel_id = presenter._active_panel_id
    presenter.resize_panel(panel_id, "4x4")
    _wait_until(lambda: not presenter._busy)
    assert _active_record(presenter)["host"].logical_size is not None
    presenter.view.panel_edit_requested.emit(panel_id)
    editor = presenter.view.editors[panel_id]
    assert editor["state"]["signal"] == "data"
    assert editor["live"] is False
    boolean = next(
        field
        for field in presenter.panels[panel_id]["surface"]["display"]
        if field["kind"] == "boolean"
    )
    presenter.view.panel_state_changed.emit(
        panel_id,
        {"display": {boolean["key"]: not boolean["value"]}},
    )
    _wait_until(lambda: not presenter._busy)
    assert presenter.panels[panel_id]["state"].display[boolean["key"]] is not boolean["value"]
    assert presenter.view.editors[panel_id]["state"]["display"][boolean["key"]] is not boolean["value"]

    presenter.view.add_panel_requested.emit("curve")
    _wait_until(lambda: not presenter._busy)
    assert len(presenter.panels) == 2
    added = presenter._active_panel_id
    assert presenter.panels[added]["recipe"]["spec"].kind.value == "curve"
    presenter.view.panel_remove_requested.emit(added)
    _wait_until(lambda: not presenter._busy)
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


def test_candidate_mount_is_atomic_and_old_host_retires_after_swap(saved) -> None:
    path, _snapshot = saved
    second_path = path.with_name("second.npz")
    rejected_path = path.with_name("rejected.npz")
    second_path.write_bytes(path.read_bytes())
    rejected_path.write_bytes(path.read_bytes())
    view = _ViewerView()
    hosts: list[object] = []

    class Host:
        def __init__(self, description) -> None:
            self.closed = False
            self.surface_at_close = None
            self.description = description

        def describe_display(self):
            return SimpleNamespace(value=self.description)

        @staticmethod
        def configure(**_kwargs) -> None:
            return None

        def close(self) -> None:
            self.surface_at_close = view.surface
            self.closed = True

    def make_host(plot_input, _name, recipe):
        host = Host(_display_description(plot_input, recipe))
        hosts.append(host)
        return host

    presenter = _presenter(view, make_host=make_host)
    try:
        presenter.open(str(path))
        _wait_until(lambda: not presenter._busy)
        first = _active_record(presenter)["host"]
        assert first is hosts[0]

        presenter.open(str(second_path))
        _wait_until(lambda: not presenter._busy)
        second = _active_record(presenter)["host"]
        _wait_until(lambda: first.closed)
        assert second is hosts[1]
        assert first.surface_at_close is second
        original_show = view.show_panel

        def reject_candidate(panel_id, host) -> None:
            if host is not None and host is not second:
                raise RuntimeError("mount refused")
            original_show(panel_id, host)

        view.show_panel = reject_candidate
        presenter.open(str(rejected_path))
        _wait_until(lambda: not presenter._busy)
        rejected = hosts[2]
        _wait_until(lambda: rejected.closed)

        assert presenter.path == second_path.resolve()
        assert _active_record(presenter)["host"] is second
        assert view.surface is second
        assert view.path == str(second_path.resolve())
        assert view.title == "run.png"
        assert second.closed is False
        assert "mount refused" in view.status[-1][0]
    finally:
        _close_presenter(presenter)


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
        fit={"model": "anisotropic_gaussian_center", "live": False},
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
    frozen = PanelFrozenData(
        signal=state.signal,
        publication=None,
        snapshot=snapshot,
        plot_input=ImageFrame(snapshot, overlay),
        overlay={"overlay_signal": state.overlay_signal},
    )

    written = save_panel_figure(
        tmp_path / "panel",
        state=state,
        frozen=frozen,
        viewport=None,
    )
    with np.load(written.archive, allow_pickle=False) as payload:
        assert "data.overlay.coordinates" in payload.files
        assert "data.overlay.status" in payload.files

    class _RestoredHost:
        def __init__(self, description) -> None:
            self.display = {}
            self.size = ""
            self.fitted = None
            self.closed = False
            self.viewport = None
            self.thresholds = ()
            self.configure_calls = 0
            self.description = description

        def describe_display(self):
            return SimpleNamespace(value=self.description)

        def configure(
            self,
            *,
            parameters=None,
            size=None,
            viewport=None,
            classifier_thresholds=(),
            fit=None,
            fit_live=True,
        ) -> None:
            self.configure_calls += 1
            self.display = dict(parameters or {})
            self.size = str(size or "")
            self.viewport = viewport
            self.thresholds = tuple(classifier_thresholds)
            target = dict(fit or {})
            model = target.pop("model", None)
            target.pop("live", None)
            self.fitted = (
                None
                if model is None
                else (str(model), {**target, "live": bool(fit_live)})
            )

        @staticmethod
        def fit(*_args, **_kwargs) -> None:
            raise AssertionError("Viewer replayed fit outside atomic configure")

        def close(self) -> None:
            self.closed = True

    seen = {}

    def make_host(plot_input, label, recipe):
        seen.update(
            plot_input=plot_input,
            label=label,
            recipe=recipe,
        )
        host = _RestoredHost(_display_description(plot_input, recipe))
        seen["host"] = host
        return host

    view = _ViewerView()
    presenter = _presenter(view, make_host=make_host)
    try:
        presenter.open(str(written.archive))
        _wait_until(lambda: not presenter._busy)
        assert presenter.description is not None, view.status
        assert seen["recipe"]["spec"].kind.value == "image", (
            "the saved kind must not be inferred anew"
        )
        assert seen["label"] == f"site occupancy — {state.signal}"
        frame = seen["plot_input"]
        assert isinstance(frame, ImageFrame)
        np.testing.assert_array_equal(frame.overlay.coordinates, overlay.coordinates)
        assert frame.overlay.point_ids == overlay.point_ids
        assert frame.overlay.labels == overlay.labels
        assert frame.overlay.static_statuses is None
        assert frame.overlay.status.exactly_equals(overlay.status)

        assert seen["recipe"]["parameters"]["show_colorbar"] is False
        assert seen["recipe"]["fit"]["model"] == "anisotropic_gaussian_center"
        assert seen["recipe"]["size"] == "4x4"
        assert next(iter(view.panels.values()))["state"].size == "4x4"
        assert _active_record(presenter)["recipe"] == seen["recipe"]
    finally:
        _close_presenter(presenter)

    real_view = _ViewerView()
    real_presenter = _built_presenter(real_view)
    try:
        real_presenter.open(str(written.archive))
        _wait_until(lambda: not real_presenter._busy)
        assert real_presenter.description is not None
        host = _active_record(real_presenter)["host"]
        recipe = _active_record(real_presenter)["recipe"]
        assert host is not None, real_view.status
        assert recipe["spec"].kind.value == "image"
        host.wait_for_front(timeout=5.0)
        assert host._session._renderer.primary_axes.get_title() == ""
        # And the authored appearance really is on the built host.
        described = host.describe_display().result().value
        assert described.display_state.values["show_colorbar"] is False
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
        semantic={"fate:site": "facet"},
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
    frozen = PanelFrozenData(state.signal, None, snapshot)
    viewport = RectangleRange(
        NumericRange(-2.0, 2.0),
        NumericRange(0.0, 40.0),
    )

    written = save_panel_figure(
        tmp_path / "unit-report",
        state=state,
        frozen=frozen,
        viewport=viewport,
    )

    view = _ViewerView()
    presenter = _built_presenter(view)
    try:
        presenter.open(str(written.archive))
        _wait_until(lambda: not presenter._busy)
        assert presenter.description is not None, view.status
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


def test_panel_save_reports_that_the_archive_survived_an_image_failure(
    saved,
    tmp_path,
    monkeypatch,
) -> None:
    import zlc_plot.figure_artifact as figure_module
    _old_path, snapshot = saved
    state = PanelState("camera", "image", "2x2", 400, "camera")
    frozen = PanelFrozenData(state.signal, None, snapshot)

    def fail_image(_path) -> None:
        raise OSError("renderer failed")

    monkeypatch.setattr(
        figure_module,
        "build_figure_host",
        lambda *_args, **_kwargs: SimpleNamespace(
            configure=lambda **_kwargs: None,
            save=fail_image,
            close=lambda: None,
        ),
    )

    with pytest.raises(RuntimeError, match="archive.*saved.*image") as failure:
        save_panel_figure(
            tmp_path / "failed-image",
            state=state,
            frozen=frozen,
            viewport=None,
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
    frozen = PanelFrozenData(state.signal, None, snapshot)

    def fail_archive(*_args, **_kwargs):
        raise OSError("archive disk full")

    monkeypatch.setattr(figure_module, "atomic_write_file", fail_archive)
    with pytest.raises(OSError, match="archive disk full"):
        save_panel_figure(
            tmp_path / "failed-archive",
            state=state,
            frozen=frozen,
            viewport=None,
        )

    assert not (tmp_path / "failed-archive.png").exists()
