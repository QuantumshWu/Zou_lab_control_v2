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

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

from zlc_atom.nodes.camera_measurement.measurement import (
    CameraMeasurementNode,
    CameraMeasurementRequest,
)
from zlc_workbench.archive import read_archive, read_dataset
from zlc_workbench.panel_save import save_panel_figure
from zlc_workbench.panel_state import PanelFrozenData, PanelState
from zlc_workbench.plot_annotations import PanelPlotAnnotations
from zlc_workbench.session import ExperimentSession
from zlc_workbench.viewer import FigureViewerPresenter, _panel_state, describe_archive
from zlc_plot import AxisRef
from zlc_plot.primitives import ImageFrame, ImagePointOverlay, PointStatus
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
    )
    restored = _panel_state(
        {"panel": {"dataset": "data", "state": state.document()}},
        "data",
    )

    assert restored == state


def test_generic_saved_state_does_not_use_the_task_console_menu_as_plot_policy() -> None:
    state = PanelState(
        signal="pulse/preview",
        kind="pulse_timeline",
        size="4x2",
        interval_ms=400,
        title="Pulse",
    )

    assert state.document()["kind"] == "pulse_timeline"


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
        self.dataset_picked = _Signal()
        self.save_image_requested = _Signal()
        self.close_requested = _Signal()
        self.figure_size_picked = _Signal()
        self.figure_title_committed = _Signal()
        self.figure_edit_requested = _Signal()
        self.datasets: tuple = ()
        self.current_dataset = ""
        self.tabs: tuple = ()
        self.surface = None
        self.title = ""
        self.size = ""
        self.path = ""
        self.status: list[tuple[str, bool]] = []

    def set_info(self, tabs) -> None:
        self.tabs = tuple(tabs)

    def set_figure_title(self, text: str) -> None:
        self.title = str(text)

    def set_figure_size(self, size: str) -> None:
        self.size = str(size)

    def run_host_dialog(self, opener, host, *, title: str):
        return opener(host, None, title=title)

    def show_figure(self, host) -> None:
        # The host, not its widget: this side of the wall never holds one.
        self.surface = host

    def set_title(self, text: str) -> None:
        self.title = str(text)

    def set_path(self, path: str) -> None:
        self.path = str(path)

    def set_datasets(self, datasets, current: str = "") -> None:
        self.datasets = tuple(datasets)
        self.current_dataset = str(current)

    def set_status(self, text: str, *, error: bool = False) -> None:
        self.status.append((str(text), bool(error)))


@pytest.fixture
def saved(tmp_path):
    """One real run, saved the way the console saves it."""

    write_ordinary_pulse(tmp_path)
    session = ExperimentSession.open(tmp_path, template="virtual")
    try:
        session.load_pulse(PULSE_NAME)
        node = CameraMeasurementNode(
            camera=session.camera,
            request=CameraMeasurementRequest("camera", 0.02, None, 1, CAMERA_WINDOWS),
            signal_plane=session.signal_plane,
            producer="cm",
        )
        capture = node.prepare()
        session.fire(shots=1)
        result = capture.collect()
        signal = node.signal_key("frame_0")
        snapshot = result.publication.value(signal).snapshot
        path = session.save_figure(
            "run",
            arrays={"panel-1": snapshot},
            nodes=(node,),
            panel={"panel-1": {"signal": signal, "title": "camera"}},
        )
        yield path, snapshot
    finally:
        session.close()


@pytest.fixture
def presenter():
    plot = pytest.importorskip("zlc_plot")
    view = _ViewerView()

    def make_host(snapshot, name, _state):
        return plot.RasterPlotHost.from_plot(
            snapshot,
            plot.ImagePlot(
                plot.AxisRef.data("spatial-x"),
                plot.AxisRef.data("spatial-y"),
                labels=plot.PlotLabels(name, "x", "y"),
            ),
        )

    presenter = FigureViewerPresenter(
        view, make_host=make_host
    )
    try:
        yield presenter
    finally:
        presenter.close()


def test_a_saved_dataset_comes_back_with_its_axes(saved) -> None:
    """The point of recording identity: what returns is the dataset, not numbers.

    A figure that can only be re-read as an array cannot be replotted, refitted
    or compared with a later run, which is most of why it was saved.
    """

    path, original = saved
    info, arrays = read_archive(path)
    restored = read_dataset(info, arrays, "panel-1")

    np.testing.assert_array_equal(
        np.asarray(restored.block.values), np.asarray(original.block.values)
    )
    assert restored.block.schema == original.block.schema
    assert restored.ref.revision == original.ref.revision
    assert [axis.axis_id.value for axis in restored.block.schema.cell_schema.data_axes] == [
        axis.axis_id.value for axis in original.block.schema.cell_schema.data_axes
    ]


def test_the_description_answers_what_the_apparatus_was_doing(saved) -> None:
    path, _snapshot = saved
    description = describe_archive(*read_archive(path))
    tabs = dict(description.tabs)
    assert tuple(tabs) == ("Plot", "Measurement", "Device", "Flow", "Raw")

    device = dict(tabs["Device"])
    assert any("exposure_seconds" in label for label in device)
    assert any(label.endswith("cm.camera") for label in device)

    measurement = dict(tabs["Measurement"])
    assert measurement["cm.repeat"] == "1"
    assert measurement["cm.frames_per_cycle"] == "3"
    assert measurement["pulse.name"] == PULSE_NAME

    plot_rows = dict(tabs["Plot"])
    assert "panel-1" in plot_rows and "uint16" in plot_rows["panel-1"]
    assert plot_rows["panel panel-1"].startswith("camera")


def test_the_flow_tab_says_where_a_number_came_from(saved) -> None:
    """The question a saved figure is least able to answer on its own."""

    path, _snapshot = saved
    tabs = dict(describe_archive(*read_archive(path)).tabs)
    flow = dict(tabs["Flow"])
    assert "cm" in flow
    assert "camera" in flow["cm"]
    assert "@logic/cm/frame_0" in flow["cm"]


def test_the_raw_tab_hides_nothing(saved) -> None:
    """Every projected tab is a reading; this is the document itself."""

    path, _snapshot = saved
    info, arrays = read_archive(path)
    raw = dict(describe_archive(info, arrays).tabs)["Raw"]
    labels = {label for label, _value in raw}
    assert "pulse.name" in labels
    assert "provenance.cm.devices.camera.exposure_seconds" in labels
    # The dataset manifest is part of the document too, however verbose.
    assert any(label.startswith("dataset.panel-1.") for label in labels)


def test_opening_shows_the_figure_and_its_record(presenter, saved) -> None:
    path, _snapshot = saved
    presenter.view.path_committed.emit(str(path))

    assert presenter.description is not None
    assert presenter.view.title == "run"
    assert presenter.view.path == str(path), "the File field cannot stay empty"
    assert dict(presenter.view.tabs)["Measurement"]
    assert presenter.view.surface is not None, presenter.view.status
    assert presenter.view.status[-1] == ("showing panel-1", False)


def test_a_file_that_cannot_be_read_is_answered_not_raised(presenter, tmp_path) -> None:
    """An operator types paths.  Most of what they type is not an archive."""

    stray = tmp_path / "notes.txt"
    stray.write_text("not an archive", encoding="utf-8")
    presenter.view.path_committed.emit(str(stray))

    assert presenter.description is None
    assert presenter.view.status[-1][1] is True
    assert "notes.txt" in presenter.view.status[-1][0]


def test_reopening_releases_the_previous_figure(presenter, saved) -> None:
    """A viewer used all afternoon must not accumulate plot workers."""

    path, _snapshot = saved
    presenter.view.path_committed.emit(str(path))
    first = presenter._host
    presenter.view.path_committed.emit(str(path))

    assert presenter._host is not first
    assert first.closed if hasattr(first, "closed") else True


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


@pytest.fixture
def saved_pair(tmp_path):
    """An archive with two datasets, which is what a two-panel console saves."""

    write_ordinary_pulse(tmp_path)
    session = ExperimentSession.open(tmp_path, template="virtual")
    try:
        session.load_pulse(PULSE_NAME)
        arrays = {}
        for name in ("panel-1", "panel-2"):
            node = CameraMeasurementNode(
                camera=session.camera,
                request=CameraMeasurementRequest("camera", 0.02, None, 1, CAMERA_WINDOWS),
                signal_plane=session.signal_plane,
                producer=name.replace("-", ""),
            )
            capture = node.prepare()
            session.fire(shots=1)
            result = capture.collect()
            arrays[name] = result.publication.value(node.signal_key("frame_0")).snapshot
        yield session.save_figure(
            "two",
            arrays=arrays,
            # What the console records beside each dataset: what the panel was
            # called and which signal it was showing.
            panel={
                "panel-1": {"title": "before", "signal": "@logic/panel1/frame_0"},
                "panel-2": {"title": "after", "signal": "@logic/panel2/frame_0"},
            },
        )
    finally:
        session.close()


def test_every_dataset_in_an_archive_can_be_reached(presenter, saved_pair) -> None:
    """A viewer that only draws the first hides the rest.

    Which reads exactly like an archive that kept only one -- so the operator
    cannot tell a partial save from a partial viewer.
    """

    presenter.view.path_committed.emit(str(saved_pair))

    # (key, label): the archive's own name for it, and what it IS, taken from
    # the panel record saved beside it.  A saved console figure used to offer
    # "panel-1" and "panel-2" and nothing else, so an operator had to guess
    # which of them was the camera.
    assert presenter.view.datasets == (
        ("panel-1", "before — @logic/panel1/frame_0"),
        ("panel-2", "after — @logic/panel2/frame_0"),
    )
    assert presenter.dataset == "panel-1"
    first = presenter._host

    presenter.view.dataset_picked.emit("panel-2")

    assert presenter.dataset == "panel-2"
    assert presenter._host is not first
    assert "panel-2" in presenter.view.status[-1][0]


def test_saving_the_figure_writes_it_beside_its_archive(presenter, saved_pair) -> None:
    class _Savable:
        def save(self, path):
            path.write_bytes(b"png")

        def close(self) -> None:
            return None

    presenter.view.path_committed.emit(str(saved_pair))
    presenter._host = _Savable()

    written = presenter.save_image()

    assert written and Path(written).exists()
    assert Path(written).parent == saved_pair.parent
    assert "panel-1" in Path(written).name


def test_a_dataset_with_no_panel_record_keeps_the_archive_s_own_name(
    presenter, saved
) -> None:
    """A label is what the archive KNOWS, never something invented.

    Arrays saved from a notebook carry no panel record, so there is nothing to
    call them but the name they were saved under -- and that is what the
    chooser must show.
    """

    presenter.view.path_committed.emit(str(saved))

    assert all(key == label for key, label in presenter.view.datasets), (
        presenter.view.datasets
    )


def test_saving_an_image_works_however_the_archive_was_spelled(presenter, saved) -> None:
    """Where an archive IS is a fact; how it was written down is not.

    The path was kept exactly as handed over, so opening one by a relative
    name -- which is what a shell, a launcher and the console's own Open all
    do -- left every path-derived action holding a relative root.  unique_path
    rightly refuses one, and the refusal left a Qt slot: Save image closed the
    window instead of writing a file, saying nothing on the way out.
    """

    path, _snapshot = saved
    here = os.getcwd()
    os.chdir(path.parent)
    try:
        assert presenter.open(path.name) is not None
        assert presenter.path.is_absolute(), "an archive's location is absolute"
        written = presenter.save_image()
    finally:
        os.chdir(here)

    assert written, presenter.view.status
    assert Path(written).is_file()


def test_the_figure_is_a_panel_with_its_own_decisions(presenter, saved) -> None:
    """A saved figure is a CARD now, not a bare picture.

    It carries the picker, the size and the way into the plot's own controls --
    the same card the console uses, in the mode that admits a file will not
    deliver again.  A bare picture could only ever be looked at.

    Renaming is answered by REMEMBERING, not by writing the name back: the card
    already shows what was typed, and echoing it made the card re-raise the
    commit.  That loop ended the process with no traceback, which is what a Qt
    signal cycle looks like from outside.
    """

    path, _snapshot = saved
    assert presenter.open(path) is not None

    presenter.view.figure_title_committed.emit("reopened run")
    assert presenter.figure_title == "reopened run"

    assert presenter.resize_figure("4x4") is True, presenter.view.status
    assert presenter._host.logical_size is not None

    seen = []
    presenter._edit_figure = lambda host, title: seen.append((host, title))
    presenter.view.figure_edit_requested.emit()
    assert seen and seen[0][0] is presenter._host


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
        overlay_signal="@logic/occupancy/site_overlay",
    )
    overlay = ImagePointOverlay(
        7,
        np.asarray(((2.5, 3.5), (7.5, 9.5))),
        ("site-0", "site-1"),
        ("0", "1"),
        (PointStatus.EMPTY, PointStatus.OCCUPIED),
    )
    frozen = PanelFrozenData(
        signal=state.signal,
        publication=None,
        snapshot=snapshot,
        plot_input=ImageFrame(snapshot, overlay),
        overlay={"overlay_signal": state.overlay_signal},
    )

    class _SavingHost:
        def save(self, path) -> None:
            Path(path).write_bytes(b"png")

        def close(self) -> None:
            return None

    written = save_panel_figure(
        tmp_path / "panel",
        state=state,
        frozen=frozen,
        make_host=lambda _input, _signal, _kind, _cell_kind: _SavingHost(),
        configure_host=lambda _host, _state, _overlay: None,
    )
    with np.load(written.archive, allow_pickle=False) as payload:
        assert "overlay.coordinates" in payload.files

    class _RestoredHost:
        def __init__(self) -> None:
            self.display = {}
            self.size = ""
            self.fitted = None
            self.closed = False

        def configure(self, *, parameters=None, size=None) -> None:
            self.display = dict(parameters or {})
            self.size = str(size or "")

        def fit(self, model, **options) -> None:
            self.fitted = (str(model), dict(options))

        def close(self) -> None:
            self.closed = True

    seen = {}

    def make_host(plot_input, label, panel_state):
        seen.update(
            plot_input=plot_input,
            label=label,
            state=panel_state,
        )
        host = _RestoredHost()
        seen["host"] = host
        return host

    view = _ViewerView()
    presenter = FigureViewerPresenter(view, make_host=make_host)
    try:
        assert presenter.open(str(written.archive)) is not None, view.status
        assert seen["state"].kind == "image", (
            "the saved kind must not be inferred anew"
        )
        assert seen["state"].cell_kind == ""
        assert seen["label"] == f"site occupancy — {state.signal}"
        frame = seen["plot_input"]
        assert isinstance(frame, ImageFrame)
        np.testing.assert_array_equal(frame.overlay.coordinates, overlay.coordinates)
        assert frame.overlay.point_ids == overlay.point_ids
        assert frame.overlay.labels == overlay.labels
        assert frame.overlay.statuses == overlay.statuses

        host = seen["host"]
        assert seen["state"].semantic == {"reduction": "mean"}
        assert host.display == {"show_colorbar": False}
        assert host.size == "4x4"
        assert host.fitted == (
            "anisotropic_gaussian_center",
            {"live": False},
        )
        assert presenter.panel_state == state
    finally:
        presenter.close()

    from zlc_workbench.apps.figure_viewer import build

    real_view = _ViewerView()
    real_presenter = build(real_view)
    try:
        assert real_presenter.open(str(written.archive)) is not None
        assert real_presenter._host is not None, real_view.status
        assert real_presenter.panel_state == state
    finally:
        real_presenter.close()


def test_panel_save_annotation_roundtrip_keeps_canonical_units(tmp_path) -> None:
    """A saved V threshold stays V after reopening a panel displayed in mV."""

    from data_factory import Axis, DatasetSchema, DatasetSnapshot, PointTable
    from zlc_workbench.apps.figure_viewer import build

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
        semantic={"facet": AxisRef.point("site")},
        display={"value_display_unit": "mV"},
    )
    frozen = PanelFrozenData(state.signal, None, snapshot)

    class _SavingHost:
        def save(self, path) -> None:
            Path(path).write_bytes(b"png")

        def close(self) -> None:
            return None

    written = save_panel_figure(
        tmp_path / "unit-report",
        state=state,
        frozen=frozen,
        make_host=lambda *_args: _SavingHost(),
        configure_host=lambda _host, _state, _overlay: None,
        annotations=PanelPlotAnnotations((1.0, 2.0)),
    )

    view = _ViewerView()
    presenter = build(view)
    try:
        assert presenter.open(str(written.archive)) is not None, view.status
        assert presenter.plot_annotations.classifier_thresholds == (1.0, 2.0)
        host = presenter._host
        assert host is not None
        # Had the Viewer treated archived canonical values as display values,
        # these would be 0.001 V and 0.002 V after its mV projection.
        assert tuple(host._session._classifier_thresholds) == (1.0, 2.0)
        description = host.describe_display().result(timeout=10).value
        assert str(description.display_state.values["value_display_unit"]) == "mV"
    finally:
        presenter.close()
