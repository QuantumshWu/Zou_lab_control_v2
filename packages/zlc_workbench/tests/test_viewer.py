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
from zlc_data.figure_archive import read_archive
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
    DomainSpec,
    SITE,
    ValidityContract,
    ValueSchema,
    owned_snapshot_from_arrays,
)
from zlc_plot import (
    AxisRef,
    NumericRange,
    SelectorKind,
    build_figure_host,
    save_figure_artifact,
)
from zlc_plot.primitives import ImageFrame, ImagePointOverlay
from zlc_plot.selectors import RectangleRange, SelectorState
from pulse_fixtures import (
    CAMERA_WINDOWS,
    PULSE_NAME,
    ordinary_imaging_sequence,
    write_ordinary_pulse,
)

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
    host = build_panel_host(
        plot_input,
        state,
        build_host=build_figure_host,
    )
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
        self.new_data_requested = _Signal()
        self.edit_data_requested = _Signal()
        self.data_editor_intent = _Signal()
        self.data_editor_closed = _Signal()
        self.add_panel_requested = _Signal()
        self.panel_state_changed = _Signal()
        self.panel_remove_requested = _Signal()
        self.panel_edit_requested = _Signal()
        self.panel_order_committed = _Signal()
        self.panel_editor_closed = _Signal()
        self.panel_snapshot_refresh_requested = _Signal()
        self.panel_save_figure_requested = _Signal()
        self.panel_plot_error = _Signal()
        self.save_image_requested = _Signal()
        self.info_action_requested = _Signal()
        self.pulse_tab_closed = _Signal()
        self.pulse_include_off_toggled = _Signal()
        self.pulse_selectors_toggled = _Signal()
        self.pulse_size_committed = _Signal()
        self.pulse_save_requested = _Signal()
        self.close_requested = _Signal()
        self.pulse_tabs: dict[str, dict] = {}
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
        self.data_editors: dict[str, object] = {}
        self.editable_data: tuple = ()
        self.dpr = 1.0

    def device_pixel_ratio(self) -> float:
        return float(self.dpr)

    def has_panel_editor(self, panel_id: str) -> bool:
        return str(panel_id) in self.editors

    def set_panel_sizes(self, sizes, default_size) -> None:
        self.panel_sizes = tuple(str(value) for value in sizes)
        self.panel_default_size = str(default_size)

    def set_panel_kinds(self, kinds: object, default_kind: str = "") -> None:
        del default_kind
        self.panel_kinds = tuple(kinds)

    def set_panel_intervals(
        self, intervals: object, default_interval: int
    ) -> None:
        self.panel_intervals = tuple(intervals)
        self.panel_default_interval = int(default_interval)

    def set_grid_cell_kinds(self, kinds) -> None:
        self.grid_cell_kinds = tuple(kinds)

    def set_editable_data_choices(self, choices, *, current="") -> None:
        self.editable_data = tuple(choices)
        self.current_editable_data = str(current)

    def open_data_editor(self, editor_id, projection, *, title="") -> None:
        self.data_editors[str(editor_id)] = {
            "projection": projection,
            "title": str(title),
        }

    def update_data_editor(self, editor_id, projection) -> bool:
        if str(editor_id) not in self.data_editors:
            return False
        self.data_editors[str(editor_id)]["projection"] = projection
        return True

    def close_data_editor(self, editor_id) -> bool:
        return self.data_editors.pop(str(editor_id), None) is not None

    def focus_data_editor(self, editor_id) -> bool:
        return str(editor_id) in self.data_editors

    def has_data_editor(self, editor_id) -> bool:
        return str(editor_id) in self.data_editors

    def add_panel(self, panel_id, title) -> None:
        self.panels[str(panel_id)] = {"title": str(title)}

    def remove_panel(self, panel_id) -> None:
        self.panels.pop(str(panel_id), None)

    def set_panel_order(self, order) -> None:
        self.panel_order = tuple(order)

    def set_panel_signal_choices(self, panel_id: str, *args, **kwargs) -> None:
        groups = args[0]
        self.panels[str(panel_id)].update(
            signal_groups=tuple(groups),
            **kwargs,
        )

    def set_panel_publishers(self, publishers: object) -> None:
        del publishers

    def panel_ids(self) -> tuple[str, ...]:
        return tuple(self.panels)

    def set_panel_selectors_enabled(self, panel_id, enabled) -> None:
        self.panels[str(panel_id)]["selectors_enabled"] = bool(enabled)

    def set_panel_mutation_enabled(self, panel_id, enabled) -> None:
        self.panels[str(panel_id)]["mutation_enabled"] = bool(enabled)

    def present_panel_front(self, panel_id: str, front: object) -> bool:
        del panel_id, front
        return True

    def set_panel_projection(self, panel_id, state, surface) -> None:
        self.panels[str(panel_id)].update(state=state, surface=surface)

    def set_panel_status(self, panel_id, text, *, error=False) -> None:
        self.panels[str(panel_id)]["status"] = (str(text), bool(error))

    def show_panel(self, panel_id, host) -> None:
        self.panels[str(panel_id)]["host"] = host
        self.surface = host

    def open_panel_editor(
        self, panel_id: str, projection: Any, *, title: str = ""
    ) -> None:
        del title
        self.editors[str(panel_id)] = projection

    def show_panel_editor(self, panel_id: str, host: Any | None) -> None:
        del panel_id, host

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

    def open_pulse_tab(self, key, title) -> None:
        self.pulse_tabs.setdefault(
            str(key),
            {"title": str(title), "host": None, "placeholder": "", "mounts": 0,
             "size_names": (), "size": "", "status": ""},
        )

    def set_pulse_size_names(self, key, names) -> bool:
        tab = self.pulse_tabs.get(str(key))
        if tab is None:
            return False
        tab["size_names"] = tuple(names)
        return True

    def set_pulse_size(self, key, size) -> bool:
        tab = self.pulse_tabs.get(str(key))
        if tab is None:
            return False
        tab["size"] = str(size)
        return True

    def set_pulse_status(self, key, text) -> bool:
        tab = self.pulse_tabs.get(str(key))
        if tab is None:
            return False
        tab["status"] = str(text)
        return True

    def has_pulse_tab(self, key) -> bool:
        return str(key) in self.pulse_tabs

    def show_pulse(self, key, host) -> bool:
        tab = self.pulse_tabs.get(str(key))
        if tab is None:
            return False
        tab["host"] = host
        tab["mounts"] += 1
        return True

    def show_pulse_placeholder(self, key, text) -> bool:
        tab = self.pulse_tabs.get(str(key))
        if tab is None:
            return False
        tab["placeholder"] = str(text)
        return True

    def close_pulse_tab(self, key) -> bool:
        return self.pulse_tabs.pop(str(key), None) is not None

    def set_title(self, text: str) -> None:
        self.title = str(text)

    def set_path(self, path: str) -> None:
        self.path = str(path)

    def set_status(self, text: str, *, error: bool = False) -> None:
        self.status.append((str(text), bool(error)))

    def show_status(self, text: str, severity: str) -> None:
        self.status.append((str(text), str(severity) == "error"))

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
    from zlc_workbench.board import attach_qt_owner_turn, attach_qt_worker
    from zlc_ui.qt import ensure_qt_app
    from test_console_presenter import _async_writer

    ensure_qt_app(["test-built-figure-viewer"])
    run_off_thread, close_worker = attach_qt_worker("test-built-figure-viewer")
    monitor_render = SimpleNamespace(build_host=build_figure_host)

    def save_front(path, front):
        front.buffer.save(path)
        return Path(path)

    editor_render = SimpleNamespace(
        build_host=build_figure_host,
        save_figure_artifact=_async_writer(save_figure_artifact),
        save_front=_async_writer(save_front),
    )
    presenter = build(
        view,
        run_off_thread=run_off_thread,
        close_worker=close_worker,
        request_close=lambda: None,
        monitor_render=monitor_render,
        editor_render=editor_render,
        close_render_processes=lambda: True,
    )
    presenter._panel_presenter.board.wake.set_notify(attach_qt_owner_turn(presenter.commit_surfaces))
    return presenter

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

def _formal_viewer_window(saved, build_host):
    """The product window over ``saved``, drawing with ``build_host``."""

    pytest.importorskip("PyQt5")
    from PyQt5 import QtCore
    from zlc_ui.qt import ensure_qt_app
    from zlc_workbench.apps.figure_viewer import create_window

    application = ensure_qt_app(["formal-figure-viewer"])
    render = SimpleNamespace(
        build_host=build_host,
        save_figure_artifact=lambda *_args, **_kwargs: None,
        save_front=lambda *_args, **_kwargs: None,
        retain=lambda: None,
        release=lambda *, timeout=0.0: True,
        close=lambda *, timeout=0.0: True,
    )
    path, _snapshot = saved
    window = create_window(
        path=path,
        window_ratio=0.25,
        monitor_render=render,
        editor_render=render,
    )
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
            lineage=capture_run_chain(session.signal_plane, result.publication)[0],
        )
        written = save_panel_figure(
            tmp_path / "run.png",
            state=state,
            frozen=frozen,
            writer=save_figure_artifact,
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
    info, arrays, datasets = read_archive(path)
    restored = datasets["data"]

    np.testing.assert_array_equal(
        np.asarray(restored.block.values), np.asarray(original.block.values)
    )
    assert restored.block.schema == original.block.schema
    assert restored.ref.revision == original.ref.revision
    assert [axis.axis_id.value for axis in restored.block.schema.cell_domain.axes] == [
        axis.axis_id.value for axis in original.block.schema.cell_domain.axes
    ]

def test_manual_data_uses_runtime_panel_and_the_one_figure_writer(tmp_path) -> None:
    view = _ViewerView()
    presenter = _built_presenter(view)
    try:
        view.new_data_requested.emit()
        editor_id, draft = next(iter(presenter._data_drafts.items()))
        axis_id = "manual.x"
        projection = view.data_editors[editor_id]["projection"]
        assert tuple(axis["domain"] for axis in projection["axes"]) == (
            "repeat",
            "point",
        )
        assert projection["axis_values"]["shape"] == (1, 16)

        view.data_editor_intent.emit(
            editor_id,
            {
                "op": "set_axis_values",
                "axis_id": axis_id,
                "cells": ((0, 8, "8.5"),),
            },
        )
        view.data_editor_intent.emit(
            editor_id,
            {
                "op": "edit_axis",
                "axis_id": axis_id,
                "name": "detuning",
                "length": 16,
                "unit": "MHz",
                "domain": "point",
            },
        )
        view.data_editor_intent.emit(
            editor_id,
            {
                "op": "add_axis",
                "name": "shot",
                "length": 2,
                "unit": "",
                "domain": "repeat",
            },
        )
        shot_id = str(draft["selected_axis"])
        view.data_editor_intent.emit(
            editor_id,
            {
                "op": "set_axis_values",
                "axis_id": shot_id,
                "cells": ((0, 0, "10"), (0, 1, "20")),
            },
        )
        view.data_editor_intent.emit(
            editor_id,
            {"op": "set_scope", "axis_id": shot_id, "index": 1},
        )
        assert draft["scopes"][shot_id] == 1
        view.data_editor_intent.emit(
            editor_id,
            {
                "op": "add_axis",
                "name": "temporary",
                "length": 4,
                "unit": "",
                "domain": "cell_data",
            },
        )
        temporary = str(draft["selected_axis"])
        view.data_editor_intent.emit(
            editor_id,
            {"op": "delete_axis", "axis_id": temporary},
        )
        assert all(axis.name != "temporary" for axis in draft["cell_axes"])
        assert "kept coordinate 0" in str(draft["message"])
        view.data_editor_intent.emit(
            editor_id,
            {"op": "set_table_axis", "axis_id": shot_id, "mode": "rows"},
        )
        view.data_editor_intent.emit(
            editor_id,
            {"op": "set_table_axis", "axis_id": axis_id, "mode": "columns"},
        )
        projection = view.data_editors[editor_id]["projection"]
        assert projection["table"]["shape"] == (2, 16)
        assert tuple(
            (axis["name"], axis["mode"]) for axis in projection["table"]["axes"]
        ) == (("repeat", "scope"), ("shot", "rows"), ("detuning", "columns"))

        view.data_editor_intent.emit(
            editor_id,
            {
                "op": "set_cells",
                "component": "values",
                "cells": ((0, 3, "7.25"),),
            },
        )
        projected = view.data_editors[editor_id]["projection"]
        assert projected["table"]["changed_cells"] == ((0, 3),)
        assert projected["axis_values"]["changed_cells"] == ()
        view.data_editor_intent.emit(
            editor_id,
            {"op": "apply_preview", "note": "manual Figure check"},
        )
        _wait_until(
            lambda: (
                presenter.beat()
                or (
                    presenter.panels[str(draft["panel_id"])].frozen_data is not None
                    and presenter.panels[
                        str(draft["panel_id"])
                    ].frozen_data.publication
                    is draft["publication"]
                )
            )
        )

        target = tmp_path / "manual-data.npz"
        view.data_editor_intent.emit(
            editor_id,
            {
                "op": "save_as",
                "path": str(target),
                "note": "manual Figure check",
            },
        )
        _wait_until(lambda: target.is_file() and not presenter._busy)

        info, arrays, datasets = read_archive(target)
        restored = datasets["data"]
        assert restored.block.values.shape == (2, 16, 1)
        assert restored.block.values[0, 3, 0] == 7.25
        assert restored.block.schema.repeat_domain.axes[-1].coordinates == (10, 20)
        assert restored.block.schema.point_domain.axes[0].name == "detuning"
        assert restored.block.schema.point_domain.axes[0].coordinates[8] == 8.5
        lineage = info["sections"]["lineage"]
        assert lineage["root"] == lineage["nodes"][0]["id"]
        assert lineage["nodes"][0]["parents"] == []
        assert lineage["nodes"][0]["record"]["operation"] == "manual-create"
    finally:
        _close_presenter(presenter)

def test_manual_axis_metadata_edit_preserves_its_existing_scientific_role(saved) -> None:
    import zlc_workbench.viewer as viewer_module

    _path, snapshot = saved
    draft = viewer_module._draft_from_snapshot(
        snapshot,
        editor_id="role-check",
        producer_serial=1,
        name="camera",
        note="",
        source_text="camera",
        source_path=None,
        source_dataset="data",
        recipe=None,
        overlay=None,
    )
    axis = next(
        item for item in draft["cell_axes"] if str(item.role) == "spatial-x"
    )
    viewer_module._edit_axis(
        draft,
        str(axis.axis_id),
        name="camera x",
        length=axis.size,
        unit="pixel",
        domain="cell_data",
    )
    edited = next(
        item for item in draft["cell_axes"] if item.axis_id == axis.axis_id
    )
    assert edited.role == axis.role
    repeat_id = str(draft["repeat_axes"][0].axis_id)
    viewer_module._delete_axis(draft, repeat_id)
    restored = viewer_module._manual_snapshot(draft)
    assert restored.block.schema.repeat_domain.axes == ()
    assert restored.block.schema.repeat_domain.shape == (1,)

def _manual_draft(editor_id: str = "draft") -> dict:
    import zlc_workbench.viewer as viewer_module

    return viewer_module._draft_from_snapshot(
        viewer_module._new_manual_snapshot(),
        editor_id=editor_id,
        producer_serial=1,
        name="manual",
        note="",
        source_text="manual",
        source_path=None,
        source_dataset="",
        recipe=None,
        overlay=None,
    )

def test_two_open_working_copies_publish_as_two_producers() -> None:
    """Two Data copies opened before either is applied are two producers.

    A copy's Runtime identity is its own from the moment it opens.  Read
    from a shared "latest copy" counter at Apply instead, both copies
    applied under one owner id and one data signal, the second Apply
    replaced the first copy's data on the plane, and nothing reported it.
    """

    view = _ViewerView()
    # Both copies stay applied and unsaved; closing asks, and this answers.
    view.confirm_discard = lambda _text: True
    presenter = _built_presenter(view)
    try:
        view.new_data_requested.emit()
        view.new_data_requested.emit()
        first, second = tuple(presenter._data_drafts)
        for editor_id, amount in ((first, "11"), (second, "22")):
            view.data_editor_intent.emit(
                editor_id,
                {
                    "op": "set_cells",
                    "component": "values",
                    "cells": ((0, 0, amount),),
                },
            )
            view.data_editor_intent.emit(
                editor_id, {"op": "apply_preview", "note": editor_id}
            )
        producers = {
            editor_id: presenter._data_drafts[editor_id]["producer"]
            for editor_id in (first, second)
        }
        assert producers[first].data_signal != producers[second].data_signal
        assert producers[first].instance_id != producers[second].instance_id
        for editor_id, amount in ((first, 11.0), (second, 22.0)):
            signal = producers[editor_id].data_signal
            publication = presenter._signal_plane.latest_publication(signal)
            assert publication is not None
            values = publication.value(signal).snapshot.block.values
            assert values.reshape(-1)[0] == amount
        assert not any(error for _text, error in view.status), view.status
    finally:
        _close_presenter(presenter)

def test_moving_an_axis_between_repeat_and_point_keeps_the_scalar_carrier(
    monkeypatch,
) -> None:
    """A named Repeat axis moved to Point reorders the arrays it holds.

    The scalar Cell-data carrier is one of the stored dimensions and stays
    one across that move; asking the arrays for an order without it refused
    the transpose after the axis lists had already moved, and the copy was
    left with axes and values that disagreed until Discard.  An edit the
    arrays refuse must leave the copy exactly as it was.
    """

    import zlc_workbench.viewer as viewer_module

    draft = _manual_draft("scalar-move")
    viewer_module._edit_axis(
        draft, "manual.repeat", name="repeat", length=2, unit="", domain="repeat"
    )
    assert viewer_module._edit_axis(
        draft, "manual.repeat", name="repeat", length=2, unit="", domain="point"
    )
    assert [str(axis.axis_id) for axis in draft["repeat_axes"]] == []
    assert [str(axis.axis_id) for axis in draft["point_axes"]] == [
        "manual.x",
        "manual.repeat",
    ]
    assert draft["values"].shape == viewer_module._logical_shape(draft) == (16, 2, 1)
    assert draft["validity"].shape == (16, 2, 1)
    restored = viewer_module._manual_snapshot(draft)
    assert restored.block.schema.point_domain.shape == (32,)

    # The last Cell-data axis leaving its domain brings the carrier back.
    viewer_module._add_axis(draft, name="site", length=3, unit="", domain="cell_data")
    site = str(draft["selected_axis"])
    assert draft["values"].shape == (16, 2, 3)
    assert viewer_module._edit_axis(
        draft, site, name="site", length=3, unit="", domain="point"
    )
    assert draft["values"].shape == viewer_module._logical_shape(draft) == (16, 2, 3, 1)

    before = {
        key: draft[key] for key in ("repeat_axes", "point_axes", "cell_axes", "values", "validity")
    }
    monkeypatch.setattr(
        viewer_module,
        "_normalize_table_axes",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("refused late")),
    )
    with pytest.raises(RuntimeError, match="refused late"):
        viewer_module._edit_axis(
            draft, site, name="site", length=3, unit="", domain="cell_data"
        )
    for key, value in before.items():
        assert draft[key] is value, f"a refused edit changed {key}"

def test_renaming_an_implicit_axis_keeps_its_coordinate_origin() -> None:
    """An axis that counts from 40 still counts from 40 after a rename.

    ``index_origin`` is where an implicit axis's coordinates start, and the
    Data contract reads it as a coordinate; rebuilding the axis with origin
    zero rebound the same values to 0, 1, 2.  A longer axis extends from
    the same start.
    """

    import zlc_workbench.viewer as viewer_module
    from zlc_data import REPEAT, SCALAR_DOMAIN, SCAN_POINT

    repeat = AxisSpec(AxisId("review.repeat"), "repeat", REPEAT, 1)
    point = AxisSpec(AxisId("review.x"), "x", SCAN_POINT, 3, index_origin=40)
    schema = DatasetSchema(
        DomainSpec((1,), (repeat,), ((0,),)),
        DomainSpec((3,), (point,), ((0, 1, 2),)),
        SCALAR_DOMAIN,
        ValueSchema.scalar(np.dtype("<f8")),
    )
    snapshot = owned_snapshot_from_arrays(schema, np.arange(3.0).reshape(1, 3, 1), 0)
    draft = viewer_module._draft_from_snapshot(
        snapshot,
        editor_id="origin",
        producer_serial=1,
        name="origin",
        note="",
        source_text="origin",
        source_path=None,
        source_dataset="data",
        recipe=None,
        overlay=None,
    )
    viewer_module._edit_axis(
        draft, "review.x", name="renamed", length=3, unit="", domain="point"
    )
    renamed = draft["point_axes"][0]
    assert renamed.name == "renamed"
    assert renamed.coordinates is None and renamed.index_origin == 40
    assert tuple(renamed.coordinate_at(index) for index in range(3)) == (40, 41, 42)
    viewer_module._edit_axis(
        draft, "review.x", name="renamed", length=4, unit="", domain="point"
    )
    assert draft["point_axes"][0].coordinate_at(3) == 43
    restored = viewer_module._manual_snapshot(draft)
    assert restored.block.schema.point_domain.axes[0].index_origin == 40

def test_extending_a_numeric_axis_advances_by_its_step_or_refuses() -> None:
    """Growing an axis continues it by its last step, and never spins.

    ``1e20`` and ``5e19`` are ints on an axis, ``0.5`` is not; the step is
    then a float and the candidate ``1e20`` already exists, and a retry by
    one leaves ``1e20`` exactly where it is.  The next coordinate is one
    more step along.  A step that cannot move the candidate is refused
    with the axis named, not looped on.
    """

    import zlc_workbench.viewer as viewer_module

    draft = _manual_draft("growth")
    viewer_module._edit_axis(
        draft, "manual.x", name="x", length=3, unit="", domain="point"
    )
    viewer_module._set_axis_values(
        draft, "manual.x", ((0, 0, "1e20"), (0, 1, "0.5"), (0, 2, "5e19"))
    )
    viewer_module._edit_axis(
        draft, "manual.x", name="x", length=4, unit="", domain="point"
    )
    grown = draft["point_axes"][0].coordinates
    assert grown == (10**20, 0.5, 5 * 10**19, 15 * 10**19)
    assert len(set(grown)) == 4

    viewer_module._edit_axis(
        draft, "manual.x", name="x", length=2, unit="", domain="point"
    )
    viewer_module._set_axis_values(
        draft, "manual.x", ((0, 0, "1.9999999999999998"), (0, 1, "2"))
    )
    with pytest.raises(ValueError, match="cannot continue the axis"):
        viewer_module._edit_axis(
            draft, "manual.x", name="x", length=3, unit="", domain="point"
        )
    assert draft["point_axes"][0].size == 2, "a refused growth changed the axis"

def test_manual_interaction_projection_does_not_rebuild_domains(monkeypatch) -> None:
    import zlc_workbench.viewer as viewer_module

    snapshot = viewer_module._new_manual_snapshot()
    draft = viewer_module._draft_from_snapshot(
        snapshot,
        editor_id="projection-check",
        producer_serial=1,
        name="manual",
        note="",
        source_text="manual",
        source_path=None,
        source_dataset="",
        recipe=None,
        overlay=None,
    )
    point = draft["point_axes"][0]
    draft["point_axes"][0] = AxisSpec(
        point.axis_id, point.name, point.role, point.size
    )
    monkeypatch.setattr(
        viewer_module,
        "_mapped_domain",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("ordinary editor projection rebuilt Domain codes")
        ),
    )
    projection = viewer_module._data_projection(draft)
    assert isinstance(projection["axis_values"]["values"][0], range)

def test_manual_value_edit_preserves_a_sparse_serpentine_domain() -> None:
    import zlc_workbench.viewer as viewer_module
    from zlc_data import REPEAT, SCALAR_DOMAIN, SCAN_POINT

    repeat = AxisSpec(AxisId("manual.shot"), "shot", REPEAT, 2, (10, 20))
    scan_x = AxisSpec(AxisId("manual.x"), "x", SCAN_POINT, 3, (0, 1, 2))
    scan_y = AxisSpec(AxisId("manual.y"), "y", SCAN_POINT, 2, (5, 6))
    repeat_domain = DomainSpec((2,), (repeat,), ((0, 1),))
    point_domain = DomainSpec(
        (4,),
        (scan_x, scan_y),
        ((0, 1, 2, 1), (0, 0, 0, 1)),
    )
    schema = DatasetSchema(
        repeat_domain,
        point_domain,
        SCALAR_DOMAIN,
        ValueSchema.scalar(np.dtype("<f8")),
    )
    source_values = np.arange(8, dtype=np.float64).reshape((2, 4, 1))
    snapshot = owned_snapshot_from_arrays(schema, source_values, 0)
    draft = viewer_module._draft_from_snapshot(
        snapshot,
        editor_id="mapped-edit",
        producer_serial=1,
        name="mapped",
        note="",
        source_text="mapped",
        source_path=None,
        source_dataset="data",
        recipe=None,
        overlay=None,
    )

    # Rename and change coordinates without changing the carrier topology.
    viewer_module._edit_axis(
        draft,
        str(scan_x.axis_id),
        name="detuning",
        length=scan_x.size,
        unit="MHz",
        domain="point",
    )
    draft["values"][1, 1, 1, 0] = 123.0
    restored = viewer_module._manual_snapshot(draft)

    assert restored.block.schema.repeat_domain.shape == repeat_domain.shape
    assert restored.block.schema.repeat_domain.axis_codes == repeat_domain.axis_codes
    assert restored.block.schema.point_domain.shape == point_domain.shape
    assert restored.block.schema.point_domain.axis_codes == point_domain.axis_codes
    assert restored.block.schema.point_domain.axes[0].name == "detuning"
    assert restored.block.values.shape == source_values.shape
    expected = source_values.copy()
    expected[1, 3, 0] = 123.0
    np.testing.assert_array_equal(restored.block.values, expected)

@pytest.mark.parametrize("source_only", (False, True))
def test_existing_archive_manual_edit_saves_reopens_and_keeps_lineage(
    saved, tmp_path, source_only
) -> None:
    from zlc_data.figure_archive import write_figure_archive

    path, original = saved
    original_info, original_arrays, original_datasets = read_archive(path)
    duplicate = {
        **original_info,
        "sections": {**original_info["sections"], "source": {
            **original_info["sections"]["source"],
            "run_record": original_info["sections"]["lineage"]["nodes"][-1]["record"],
        }},
    }
    actual = describe_archive(original_info, original_arrays)
    repeated = describe_archive(duplicate, original_arrays)
    assert (repeated.flow, repeated.pulses) == (actual.flow, actual.pulses)
    for tab in ("Logic", "Devices"):
        assert dict(repeated.tabs)[tab] == dict(actual.tabs)[tab]
    if source_only:
        sections = original_info["sections"]
        sections["source"]["run_record"] = sections["lineage"]["nodes"][-1]["record"]
        sections["lineage"] = {"root": None, "nodes": [], "device_settings": []}
        path = tmp_path / "source-record-only.npz"
        with path.open("wb") as stream:
            write_figure_archive(
                stream, original_info["name"], arrays=original_arrays, sections=sections
            )
    else:
        sections = {key: value for key, value in original_info["sections"].items() if key != "dataset"}
        sections["plot"] = {**sections["plot"], "other": sections["plot"]["data"]}
        path = tmp_path / "multiple-datasets.npz"
        with path.open("wb") as stream:
            write_figure_archive(
                stream, original_info["name"],
                arrays={"data": original, "other": original}, sections=sections,
            )
    original_lineage = original_info["sections"]["lineage"]
    original_source = original_info["sections"]["source"]
    view = _ViewerView()
    presenter = _built_presenter(view)
    try:
        presenter.open(str(path))
        _wait_until(lambda: not presenter._busy)
        monitor_build = presenter._panel_presenter._make_monitor_host
        builds = []
        def build_monitor(*args, **kwargs):
            builds.append(args)
            return monitor_build(*args, **kwargs)
        presenter._panel_presenter._make_monitor_host = build_monitor
        previous_builder = presenter._build_figure_host
        presenter._build_figure_host = lambda *args, **kwargs: pytest.fail("data editing built a throwaway C Host")
        view.edit_data_requested.emit("archive:data" if source_only else "archive:other")
        editor_id, draft = next(iter(presenter._data_drafts.items()))
        assert not presenter._busy and not builds
        presenter._build_figure_host = previous_builder
        view.data_editor_intent.emit(
            editor_id, {"op": "set_cells", "component": "values", "cells": ((0, 0, "invalid"),)},
        )
        assert draft["message"] and editor_id in view.data_editors
        assert not builds
        view.data_editor_intent.emit(
            editor_id,
            {
                "op": "set_cells",
                "component": "values",
                "cells": ((0, 0, "123"),),
            },
        )
        view.data_editor_intent.emit(
            editor_id,
            {"op": "apply_preview", "note": "existing data correction"},
        )
        _wait_until(
            lambda: (
                presenter.beat()
                or draft["publication"] is not None
                and presenter.panels[str(draft["panel_id"])].frozen_data is not None
            )
        )
        assert len(builds) == 1
        target = tmp_path / "edited-existing.npz"
        view.data_editor_intent.emit(
            editor_id,
            {
                "op": "save_as",
                "path": str(target),
                "note": "existing data correction",
            },
        )
        _wait_until(lambda: target.is_file() and not presenter._busy)
        info, arrays, datasets = read_archive(target)
        restored = datasets["data"]
        assert restored.block.values.shape == original.block.values.shape
        assert restored.block.values.reshape(-1)[0] == 123
        assert restored.block.schema == original.block.schema
        lineage = info["sections"]["lineage"]
        assert lineage["nodes"][-1]["record"]["operation"] == "manual-edit"
        assert lineage["nodes"][:-1] == original_lineage["nodes"]
        assert lineage["nodes"][-1]["parents"] == (
            [] if source_only else [original_lineage["root"]]
        )
        assert lineage["device_settings"] == original_lineage["device_settings"]
        for key, value in original_source.items():
            if key not in {"signal", "title", "overlay_signal"}:
                assert info["sections"]["source"][key] == value

        # Ordinary Panel Save must use the same frozen provenance as Data Save.
        copied = tmp_path / "manual-panel-copy.npz"
        view.panel_save_figure_requested.emit(str(draft["panel_id"]), str(copied.with_suffix(".png")))
        _wait_until(lambda: copied.is_file() and not presenter._busy)
        copied_info, _, _datasets = read_archive(copied)
        assert copied_info["sections"]["source"] == info["sections"]["source"]
        assert copied_info["sections"]["lineage"] == lineage

        previous = draft["publication"]
        view.data_editor_intent.emit(
            editor_id, {"op": "apply_preview", "note": "second edit"}
        )
        assert presenter._signal_plane.direct_parent_publications(draft["publication"]) == (previous,)
        updated, source = capture_run_chain(presenter._signal_plane, draft["publication"])
        assert updated["nodes"][:-1] == lineage["nodes"]
        assert updated["nodes"][-1]["parents"] == [lineage["root"]]
        assert source == original_source
        view.data_editor_intent.emit(editor_id, {"op": "discard"})
    finally:
        _close_presenter(presenter)

def test_a_played_pulse_is_named_on_the_device_tab_not_dumped() -> None:
    """The Device tab says which pulse played and how many periods it has;
    the document itself -- every period, slot and bracket -- and the scan
    table are read where a pulse is drawn, not as hundreds of rows here."""

    from zlc_workbench.viewer import _device_tab_snapshot

    snapshot = {
        "description": {"clock_hz": 5e7},
        "program": {"digest": "abc", "duration_seconds": 0.5, "rows": [[1, 2], [3, 4]]},
        "pulse": {"name": "scan", "periods": [{"name": "p1"}, {"name": "p2"}], "slots": []},
    }
    shown = _device_tab_snapshot(snapshot)
    assert shown["pulse"] == {"name": "scan", "periods": 2}
    assert shown["program"] == {"digest": "abc", "duration_seconds": 0.5, "rows": 2}
    assert shown["description"] == {"clock_hz": 5e7}
    assert snapshot["program"]["rows"] == [[1, 2], [3, 4]], "the record itself is untouched"


def test_a_played_pulse_is_offered_on_the_device_tab_and_drawn_on_its_own(saved) -> None:
    """The Devices page names the pulse a run played and offers to draw it;
    the tab draws the recorded document through the editor's own preview
    builder and takes its host down with it.

    The document itself is hundreds of rows of timing: the Devices page
    says which pulse and how many periods, the picture is where a pulse is
    read.
    """

    from zlc_pulse import sequence_to_tree

    path, _snapshot = saved
    info, arrays, datasets = read_archive(path)
    node = info["sections"]["lineage"]["nodes"][0]
    played_sequence = ordinary_imaging_sequence()
    record = dict(node["record"])
    record["named_devices"] = {**dict(record.get("named_devices", {})), "sequencer": "sequencer"}
    record["pulse"] = {"name": "imaging", "path": "C:/bench/pulses/imaging.json"}
    record["device_snapshots"] = {
        **dict(record.get("device_snapshots", {})),
        "sequencer": {
            "program": {"digest": "abc", "rows": [[1, 2], [3, 4]]},
            "pulse": sequence_to_tree(played_sequence),
        },
    }
    node["record"] = record
    description = describe_archive(info, arrays)
    (played,) = description.pulses
    assert (played.name, played.device_key) == ("imaging", "sequencer")
    devices = dict(dict(description.tabs)["Devices"])
    assert devices[f"sequencer pulse {played.sequence}"] == {
        "text": "imaging",
        "action": f"pulse:{played.key}",
    }
    shown = devices["sequencer"]["snapshots"][0]["snapshot"]
    assert shown["pulse"] == {
        "name": played_sequence.name,
        "periods": len(played_sequence.periods),
    }
    assert shown["program"] == {"digest": "abc", "rows": 2}

    view = _ViewerView()
    presenter = _built_presenter(view)
    try:
        built: list[tuple[object, str]] = []
        resized: list[tuple[object, str]] = []

        class _Host:
            closed = False
            interaction = None
            saved: list[Path] = []
            logical_size = (300, 200)

            def close(self) -> None:
                self.closed = True

            def set_interaction_enabled(self, enabled: bool) -> None:
                self.interaction = bool(enabled)

            def save(self, target) -> None:
                Path(target).write_bytes(b"png")
                self.saved.append(Path(target))

        def make(timeline, *, size):
            built.append((timeline, size))
            return _Host()

        def resize(host, timeline, *, size):
            resized.append((timeline, size))
            return host.logical_size

        presenter._make_pulse_preview = make
        presenter._resize_pulse_preview = resize
        presenter.description = description
        presenter.path = path
        presenter.info_action(f"pulse:{played.key}")
        tab = view.pulse_tabs[played.key]
        assert tab["title"] == "Pulse · imaging"
        assert tab["size_names"], "the tab offers the same sizes the editor's preview does"
        _wait_until(lambda: tab["host"] is not None and not presenter._busy)
        ((timeline, size),) = built
        assert timeline.total_duration > 0 and size
        assert [channel.label for channel in timeline.channels]
        assert [mark.name for mark in timeline.periods] == [
            period.name or period.period_id for period in played_sequence.periods
        ]
        host = tab["host"]
        assert (tab["size"], tab["mounts"]) == (size, 1)
        assert host.interaction is False
        presenter.info_action(f"pulse:{played.key}")
        assert built == [(timeline, size)], "a second open focuses the tab, it does not redraw"

        # The controls act on the drawing: size and off rows redraw the
        # standing host, selectors gate its interaction, Save writes it.
        presenter.set_pulse_size(played.key, "4x4")
        _wait_until(lambda: tab["mounts"] == 2 and not presenter._busy)
        assert resized[-1][1] == "4x4" and tab["size"] == "4x4"
        presenter.set_pulse_include_off(played.key, True)
        _wait_until(lambda: tab["mounts"] == 3 and not presenter._busy)
        assert len(resized[-1][0].channels) >= len(timeline.channels)
        presenter.set_pulse_selectors(played.key, True)
        assert host.interaction is True
        presenter.save_pulse_image(played.key)
        _wait_until(lambda: host.saved and not presenter._busy)
        assert host.saved[0].parent == path.parent.resolve() and host.saved[0].suffix == ".png"
        saved_name = next(
            text[len("saved "):] for text, _error in view.status if text.startswith("saved ")
        )
        assert saved_name == f"{path.stem}-pulse-imaging.png"
        assert (path.parent / saved_name).read_bytes() == b"png"

        assert presenter.close_pulse_tab(played.key)
        assert host.closed and played.key not in view.pulse_tabs
        presenter.info_action("pulse:nobody")
        assert any("played no pulse" in text for text, _error in view.status)
    finally:
        _close_presenter(presenter)


def test_the_description_reports_only_facts_saved_in_the_archive(saved) -> None:
    path, _snapshot = saved
    info, arrays, datasets = read_archive(path)
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

def test_describing_an_archive_reads_recipes_without_rebuilding_datasets(
    saved, monkeypatch
) -> None:
    """What a panel showed is a fact about its recipe, not its Dataset.

    ``describe_archive`` used to read each recipe through
    ``read_figure_plot``, which validates and materialises the Dataset
    beside it -- a full copy of every array, thrown away -- before the
    open path built the same Dataset again to keep.
    """

    import zlc_workbench.viewer as viewer_module

    path, _snapshot = saved
    info, arrays, datasets = read_archive(path)

    def rebuilt(*_args, **_kwargs):
        raise AssertionError("describe_archive rebuilt a Dataset")

    monkeypatch.setattr(viewer_module, "read_figure_plot", rebuilt)
    description = describe_archive(info, arrays)
    plot_rows = dict(dict(description.tabs)["Plot"])
    assert plot_rows["plot data"].startswith("image")

def test_the_flow_projection_is_the_saved_exact_node_edge_graph(saved) -> None:
    path, _snapshot = saved
    description = describe_archive(*read_archive(path)[:2])
    nodes = {node["id"]: node for node in description.flow["nodes"]}
    edges = description.flow["edges"]
    assert {node["kind"] for node in nodes.values()} == {"logic", "device"}
    logic = next(node for node in nodes.values() if node["kind"] == "logic")
    camera = next(node for node in nodes.values() if node["kind"] == "device")
    assert logic["title"] == "cm" and "frames" in logic["subtitle"]
    assert camera["title"] == "camera"
    # Every card names the row it stands for, so the picture is a map of
    # the Logic and Devices tabs -- and the row it names exists there.
    tabs = dict(description.tabs)
    assert logic["row"] == ("Logic", "cm") and "cm" in dict(tabs["Logic"])
    assert camera["row"] == ("Devices", "camera") and "camera" in dict(tabs["Devices"])
    assert any(
        edge["source"] == camera["id"] and edge["target"] == logic["id"]
        for edge in edges
    )

    # A convergent DAG keeps its shared event and shared device unique.
    info, arrays, datasets = read_archive(path)
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
    """Every projected tab is a reading; this is the document itself, one
    row per section, nested as the file nests it -- not flattened into
    hundreds of dotted paths."""

    path, _snapshot = saved
    info, arrays, datasets = read_archive(path)
    raw = dict(dict(describe_archive(info, arrays).tabs)["Raw"])
    assert tuple(raw) == ("dataset", "plot", "lineage", "source")
    assert raw["source"] is info["sections"]["source"]
    assert raw["source"]["signal"] == "@logic/cm/frames"
    assert [node["id"] for node in raw["lineage"]["nodes"]] == ["event-1"]
    # The dataset manifest is part of the document too, however verbose.
    assert "data" in raw["dataset"]

def test_opening_shows_the_figure_and_its_record(presenter, saved, tmp_path, monkeypatch) -> None:
    path, _snapshot = saved
    builds = []
    make_monitor = presenter._panel_presenter._make_monitor_host

    def monitor(*args, **kwargs):
        builds.append(kwargs)
        return make_monitor(*args, **kwargs)

    with monkeypatch.context() as first_open:
        first_open.setattr(presenter._panel_presenter, "_make_monitor_host", monitor)
        first_open.setattr(presenter, "_build_figure_host", lambda *_args, **_kwargs: pytest.fail("Open built an unused edit/save host"))
        presenter.view.path_committed.emit(str(path))
        _wait_until(lambda: not presenter._busy)
    assert len(builds) == 1
    assert builds[0]["initial_spec"] is not None

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
    assert "live" not in editor
    assert editor["frozen_snapshot"] is not None
    assert editor["save_directory"]
    copied_image = tmp_path / "viewer-copy.png"
    copied = copied_image.with_suffix(".npz")
    presenter.view.panel_save_figure_requested.emit(panel_id, str(copied_image))
    _wait_until(lambda: copied.is_file())
    original_info, _original_arrays, _original_datasets = read_archive(path)
    copied_info, _copied_arrays, _copied_datasets = read_archive(copied)
    assert copied_info["sections"]["lineage"] == original_info["sections"]["lineage"]
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

    previous = dict(presenter.panels)
    with monkeypatch.context() as refused:
        def refuse_monitor(*_args, **_kwargs):
            assert all(key in presenter.panels for key in previous)
            raise ValueError("initial figure cannot be drawn")
        refused.setattr(presenter._panel_presenter, "_make_monitor_host", refuse_monitor)
        presenter.open(str(path))
        _wait_until(lambda: not presenter._busy)
    assert presenter.panels == previous
    assert presenter.path == path
    assert "cannot be drawn" in presenter.view.status[-1][0]

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
    import zlc_workbench.apps.task_console as console_app
    import zlc_workbench.viewer as viewer_module

    path, _snapshot = saved
    # The viewer's panels are a console board and mount the console's
    # staging widget: a host's own default widget presents each render as
    # it lands, before the one Console that owns both boards accepted it.
    staging = console_app.staged_panel_surface
    staged: list[object] = []

    def recorded_staging(host):
        widget = staging(host)
        staged.append(widget)
        return widget

    monkeypatch.setattr(console_app, "staged_panel_surface", recorded_staging)
    _application, _QtCore, window, owner_turns, timer = _formal_viewer_window(
        saved, build_figure_host
    )
    try:
        _wait_until(lambda: window.presenter.description is not None)
        _wait_until(
            lambda: bool(staged)
            and bool(window.panel_ids())
            and window._view.panel_surface(window.presenter._active_panel_id)
            is staged[0]
        )
        accepted = (
            window.presenter.path,
            window.presenter.description,
            _active_record(window.presenter)["host"],
        )
        assert accepted[2] is staged[0].host
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
) -> None:
    from concurrent.futures import Future
    from threading import Event, Thread

    from PyQt5 import QtCore
    from zlc_ui.qt import ensure_qt_app

    application = ensure_qt_app(["formal-figure-viewer"])
    configured = Event()
    release_configure = Event()
    closing = Event()
    release_close = Event()

    class GuardedHost:
        """The initial Monitor host can be cancelled before its front is ready."""

        host_id = "guarded-host"
        startup_failure = None
        closing = False
        front = None

        def configure(self, **_kwargs):
            future = Future()

            def work() -> None:
                try:
                    assert QtCore.QThread.currentThread() != application.thread()
                    configured.set()
                    assert release_configure.wait(5.0), (
                        "test never released guarded host work"
                    )
                except BaseException as error:
                    if not future.done():
                        future.set_exception(error)
                else:
                    if not future.done():
                        future.set_result(None)

            Thread(target=work, daemon=True).start()
            return future

        def subscribe_front(self, _callback):
            return lambda: None

        def describe_display(self):
            return self.configure()

        def set_device_pixel_ratio(self, _ratio):
            done = Future()
            done.set_result(None)
            return done

        def pointer_event(self, *_args, **_kwargs):
            return None

        def close(self, *, timeout=0.0):
            assert timeout == 0.0
            closing.set()
            return release_close.is_set()

    guarded_host = GuardedHost()
    application, _QtCore, window, owner_turns, timer = _formal_viewer_window(
        saved, lambda *_args, **_kwargs: guarded_host
    )
    try:
        _wait_until(configured.is_set)
        _wait_until(lambda: len(owner_turns) >= 3)
        assert window.is_visible()
        assert window.presenter._opening_archive is not None
        assert all(binding.host is None for binding in window.presenter.panels.values())

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
        source_schema.repeat_domain,
        source_schema.point_domain,
        DomainSpec((2,), (site_axis,)),
        ValueSchema(
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
        writer=save_figure_artifact,
    )
    archive = written.archive
    with np.load(archive, allow_pickle=False) as payload:
        assert "data.overlay.coordinates" in payload.files
        assert "data.overlay.status" in payload.files

    real_view = _ViewerView()
    real_view.dpr = 1.75
    real_presenter = _built_presenter(real_view)
    try:
        real_presenter.open(str(archive))
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
        _wait_until(
            lambda: (
                real_presenter.beat()
                or any(
                    row.source_name == source_signal
                    and row.name.endswith("/center_x")
                    for row in real_presenter._signal_plane.describe_signals()
                )
            )
        )
        fit_center = next(
            row.name
            for row in real_presenter._signal_plane.describe_signals()
            if row.source_name == source_signal and row.name.endswith("/center_x")
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
        assert any(
            fit_center == signal
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

        # Initial archive fit is synchronously primed and remains live. A
        # later data+overlay publication must fit again without a Fit UI edit.
        assert host._session._live_fit_request is not None
        from zlc_workbench.viewer import _ArchiveDatasetProducer
        from test_selection import _draw_area

        plane = real_presenter._signal_plane
        offset_signal = fit_center.rsplit("/", 1)[0] + "/offset"
        before_offset = plane.current_dataset(offset_signal).block.values.item()
        previous = plane.latest_publication(source_signal)
        shifted = owned_snapshot_from_arrays(
            snapshot.block.schema,
            snapshot.block.values + np.asarray(7, dtype=snapshot.block.values.dtype),
            2,
            validity=snapshot.block.validity,
        )
        producer = _ArchiveDatasetProducer(
            1, 0, "data", ImageFrame(shifted, overlay), archive,
            owner_id=real_presenter._archive_producers[0].instance_id,
            data_signal=source_signal, run_record={"operation": "manual-edit"},
        )
        updated = producer.publish(
            plane, source_publication=(source_signal, previous),
        )
        real_presenter._archive_producers = (producer,)
        front = plane.freeze()
        assert front.publication(source_signal) is updated
        assert front.publication(active["state"].overlay_signal) is updated

        def refitted():
            real_presenter.beat()
            publication = plane.latest_publication(offset_signal)
            return (
                publication is not None
                and publication.direct_parent_refs == (updated.event_ref,)
                and real_presenter.panels[source_panel_id].display_publication is updated
            )

        _wait_until(refitted)
        assert real_presenter.panels[source_panel_id].host is host
        after_offset = plane.current_dataset(offset_signal).block.values.item()
        assert after_offset == pytest.approx(before_offset + 7.0, abs=1e-3)
        assert real_presenter.panels[source_panel_id].state.selector
        roi_publication = plane.latest_publication(derived_roi)
        assert roi_publication is not None
        assert roi_publication.direct_parent_refs == (updated.event_ref,)

        # A real gesture has a nonzero owner revision; the persisted region
        # document deliberately does not carry that lifecycle counter.
        _draw_area(host, span=(0.1, 0.1, 0.8, 0.8))
        binding = real_presenter.panels[source_panel_id]
        _wait_until(lambda: real_presenter.beat() or (
            binding.selection_revision > 0 and binding.configuration is None
        ))
        region_revision = binding.selection_revision
        updated = producer.publish(plane, source_publication=(source_signal, updated))
        _wait_until(refitted)
        assert binding.host is host and binding.selection_revision == region_revision
        assert binding.bridge.last_error is None
        assert plane.latest_publication(derived_roi).direct_parent_refs == (updated.event_ref,)

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

    from data_factory import (
        axis,
        make_dataset_schema,
        make_snapshot,
        mapped_domain_from_columns,
        repeat_domain,
    )

    from zlc_data import DatasetSchema
    samples = np.linspace(-3.0, 3.0, 80)
    values = np.column_stack((samples - 1.0, samples + 1.0))
    schema = make_dataset_schema(
        repeat_domain(size=len(samples)),
        mapped_domain_from_columns({"site": (0.0, 1.0)}),
        dtype=np.float64,
        value_unit="V",
    )
    snapshot = make_snapshot(schema, values, revision=0)
    state = PanelState(
        signal="report/distribution",
        kind="facet_grid",
        cell_kind="histogram",
        size="4x4",
        interval_ms=400,
        title="unit report",
        semantic={"fate:point:site": "facet"},
        display={"value_display_unit": "mV", "threshold_classifier": True},
        classifier_thresholds=(
            {
                "value": 1.0,
                "scope": (
                    {
                        "domain": "point",
                        "axis_id": "site",
                        "coordinate": 0,
                    },
                ),
            },
            {
                "value": 2.0,
                "scope": (
                    {
                        "domain": "point",
                        "axis_id": "site",
                        "coordinate": 1,
                    },
                ),
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
        writer=save_figure_artifact,
    )
    archive = written.archive

    view = _ViewerView()
    presenter = _built_presenter(view)
    try:
        presenter.open(str(archive))
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
    from data_factory import (
        axis,
        make_dataset_schema,
        make_snapshot,
        mapped_domain_from_columns,
        repeat_domain,
    )
    from zlc_data import DatasetSchema
    from zlc_plot.fit import FacetFitBatchResult

    x = np.linspace(-3.0, 3.0, 40)
    facets = np.repeat((0.0, 1.0), 20)
    schema = make_dataset_schema(
        repeat_domain(size=1),
        mapped_domain_from_columns({"x": x, "facet": facets}),
        dtype=np.float64,
    )
    values = (
        2.0 * np.exp(-0.5 * ((x - 0.15) / 0.9) ** 2) + 0.2
    )[None, :]
    snapshot = make_snapshot(schema, values, revision=0)
    state = PanelState(
        signal="saved/facet-curve",
        kind="facet_grid",
        cell_kind="curve",
        size="4x4",
        interval_ms=400,
        title="facet curve",
        semantic={
            "fate:point:facet": "facet",
            "fate:point:x": "x",
            f"fate:repeat:{schema.repeat_domain.axes[0].axis_id}": "reduce",
            "reduction": "mean",
        },
        fit={"model": "gaussian_offset", "fit_all_facets": True},
    )
    frozen = _frozen_surface(state, snapshot)
    written = save_panel_figure(
        tmp_path / "facet-fit",
        state=state,
        frozen=frozen,
        writer=save_figure_artifact,
    )
    archive = written.archive
    view = _ViewerView()
    presenter = _built_presenter(view)
    try:
        presenter.open(str(archive))
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

def test_viewer_restores_facet_cell_kind_before_its_semantic_vocabulary(
    tmp_path,
) -> None:
    """A saved Histogram vocabulary must not be applied to an inferred Image."""

    from data_factory import (
        axis,
        make_dataset_schema,
        make_snapshot,
        mapped_domain_from_columns,
        repeat_domain,
    )

    from zlc_data import DatasetSchema
    from zlc_plot import HistogramPlot

    schema = make_dataset_schema(
        repeat_domain(size=8),
        mapped_domain_from_columns({"source_index": (0.0, 1.0)}),
        cell_axes=(
            axis("frame", size=3),
            axis("site", size=4),
        ),
        dtype=np.float64,
    )
    snapshot = make_snapshot(
        schema,
        np.arange(np.prod(schema.physical_shape), dtype=np.float64).reshape(schema.physical_shape),
        revision=0,
    )
    state = PanelState(
        signal="saved/site-histograms",
        kind="facet_grid",
        cell_kind="histogram",
        size="2x2",
        interval_ms=400,
        title="site histograms",
        semantic={
            "fate:repeat:repeat": "pool",
            "fate:point:source_index": "pool",
            "fate:cell_data:frame": ["scope-value", 1],
            "fate:cell_data:site": "facet",
            "reduction": "mean",
        },
    )
    written = save_panel_figure(
        tmp_path / "site-histograms",
        state=state,
        frozen=_frozen_surface(state, snapshot),
        writer=save_figure_artifact,
    )
    archive = written.archive
    view = _ViewerView()
    presenter = _built_presenter(view)
    try:
        presenter.open(str(archive))
        _wait_until(lambda: not presenter._busy)
        assert presenter.description is not None, view.status
        _wait_until(
            lambda: (
                presenter.beat()
                or _active_record(presenter)["host"] is not None
            )
        )
        active = _active_record(presenter)
        assert active["state"].cell_kind == "histogram"
        described = active["host"].describe_display().result().value
        assert isinstance(described.spec.cell, HistogramPlot)
        assert described.semantics.values["fate:repeat:repeat"] == "pool"
        assert (
            described.semantics.values[
                "fate:point:source_index"
            ]
            == "pool"
        )
        assert described.semantics.values["fate:cell_data:frame"] == (
            "scope-value",
            1,
        )
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

    from zlc_plot.rendering import MatplotlibRenderer

    def fail_image(self, _path, **_kwargs) -> None:
        raise OSError("renderer failed")

    monkeypatch.setattr(MatplotlibRenderer, "save", fail_image)

    with pytest.raises(RuntimeError, match="archive.*saved.*image") as failure:
        save_panel_figure(
            tmp_path / "failed-image",
            state=state,
            frozen=frozen,
            writer=figure_module.save_figure_artifact,
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
            writer=figure_module.save_figure_artifact,
        )

    assert not (tmp_path / "failed-archive.png").exists()


def test_unsaved_edits_are_asked_about_rather_than_locked_in() -> None:
    """A door that only opens from the inside is not a safeguard.

    Closing with an open working copy used to be refused outright: the
    operator was told to Save or Discard and the window simply would not
    go.  The work is theirs, so they are asked once and the answer is
    honoured -- and with nobody to ask (a headless host, a test), nothing
    is discarded and the old refusal stands.
    """

    view = _ViewerView()
    asked: list[str] = []
    answer = {"value": True}
    view.confirm_discard = lambda text: asked.append(text) or answer["value"]
    presenter = _built_presenter(view)
    try:
        presenter._data_drafts["data-1"] = {
            "name": "Manual data 1",
            "modified": True,
            "unsaved": False,
            "message": "",
            "producer": None,
            "publication": None,
        }

        # Declining keeps the working copy and says so.
        answer["value"] = False
        assert presenter.close() is False
        assert asked and "Manual data 1" in asked[-1], asked
        assert "Save or discard" in view.status[-1][0], view.status[-1]
        assert "data-1" in presenter._data_drafts

        # Agreeing closes: the refusal is not repeated.
        answer["value"] = True
        before = len(asked)
        presenter.close()
        assert len(asked) == before + 1, asked
        assert "Save or discard" not in view.status[-1][0], view.status[-1]
    finally:
        presenter._data_drafts.clear()
        _close_presenter(presenter)


def test_a_presenter_with_nobody_to_ask_never_discards_on_its_own() -> None:
    """No confirm hook means the old refusal, not a silent loss."""

    view = _ViewerView()
    presenter = _built_presenter(view)
    try:
        assert presenter._confirm_discard is None
        presenter._data_drafts["data-1"] = {
            "name": "Manual data 1",
            "modified": True,
            "unsaved": False,
            "message": "",
            "producer": None,
            "publication": None,
        }
        assert presenter.close() is False
        assert "Save or discard" in view.status[-1][0], view.status[-1]
    finally:
        presenter._data_drafts.clear()
        _close_presenter(presenter)


def test_the_close_question_is_asked_once_per_gesture() -> None:
    """Closing takes several passes; the decision is not one of them.

    ``close`` is a retry loop -- panels first, then the IO worker, each
    pass returning False and asking to be called again -- and the dirty
    check sat inside it, so the operator was asked once per PASS.  A
    first close wanted two passes and asked twice; after declining, the
    pass already spent made the next close want one, and it asked once.
    A decision about losing work belongs to the gesture, not to the
    plumbing that carries it out.
    """

    view = _ViewerView()
    asked: list[str] = []
    view.confirm_discard = lambda text: asked.append(text) or True
    presenter = _built_presenter(view)
    try:
        presenter._data_drafts["data-1"] = {
            "name": "Manual data 1",
            "modified": True,
            "unsaved": False,
            "message": "",
            "producer": None,
            "publication": None,
        }
        passes = {"count": 0}
        finished = presenter._panel_presenter.close

        def staged_close() -> bool:
            passes["count"] += 1
            return bool(passes["count"] > 1 and finished())

        presenter._panel_presenter.close = staged_close

        presenter.close()
        presenter.close()
        assert passes["count"] >= 2, "the test needs a close that takes two passes"
        assert len(asked) == 1, asked
    finally:
        presenter._data_drafts.clear()
        _close_presenter(presenter)
