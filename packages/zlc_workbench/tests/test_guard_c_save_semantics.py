import zou_lab_control_v2

import json
import time
from pathlib import Path

import numpy as np
import pytest

import zlc_workbench.console as tested_module


print(tested_module.__file__)

from zlc_atom.nodes.calibration import (
    FrameContract,
    ReadoutModel,
    ReadoutModelKind,
    SiteMap,
    TrapCalibration,
)
from zlc_runtime import NodeHost
from zlc_workbench.archive import read_archive, read_dataset
from zlc_workbench.console import ConsolePresenter
from zlc_workbench.logic import stable_signal_key
from zlc_workbench.panel_catalog import task_console_fitting_spec
from zlc_workbench.panel_state import project_panel_state
from zlc_workbench.session import ExperimentSession

from test_console_presenter import _ConsoleView, _Signal, _one_shot
from pulse_fixtures import write_ordinary_pulse


CALIBRATION_SENTINEL = "CALIBRATION_BODY_MUST_NOT_BE_EMBEDDED"


def _wait_terminal(host: NodeHost, *, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while host.running and time.monotonic() < deadline:
        host.poll()
        time.sleep(0.005)
    observation = host.poll()
    assert observation.terminal and observation.error is None, observation


def _calibration(path: Path) -> TrapCalibration:
    site_ids = ("site-0", "site-1")
    calibration = TrapCalibration(
        SiteMap(
            site_ids,
            np.asarray(((12.0, 10.0), (30.0, 20.0))),
            np.asarray((True, True)),
            np.asarray((1.0, 1.0)),
        ),
        (
            ReadoutModel(
                site_ids,
                np.asarray((100.0, 100.0)),
                np.zeros(2),
                np.ones(2),
                np.asarray((True, True)),
                np.asarray((1.0, 1.0)),
            ),
        ),
        ReadoutModelKind.BOX,
        FrameContract(
            (96, 128),
            sensor_shape=(96, 128),
            roi_xywh=(0, 0, 128, 96),
            exposure_seconds=0.02,
            camera_id="camera",
            readout_mode="virtual-external-trigger",
        ),
        report={"test_marker": CALIBRATION_SENTINEL},
    )
    calibration.save(path)
    return calibration


def _records(value) -> list[dict]:
    found: list[dict] = []
    if isinstance(value, dict):
        if isinstance(value.get("node"), str):
            found.append(value)
        for child in value.values():
            found.extend(_records(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(_records(child))
    return found


def test_guard_c_header_saves_and_single_panel_save_have_distinct_semantics(
    tmp_path, monkeypatch
) -> None:
    plot = pytest.importorskip("zlc_plot")
    session = ExperimentSession.open(tmp_path, template="virtual")
    write_ordinary_pulse(tmp_path)
    view = _ConsoleView()
    view.panel_save_figure_requested = _Signal()

    def spec_for(snapshot, kind="", cell_kind=""):
        return task_console_fitting_spec(snapshot.block.schema, kind, cell_kind)

    def make_host(plot_input, state):
        initial = getattr(plot_input, "snapshot", plot_input)
        spec, _semantic, parameters = project_panel_state(
            initial.block.schema,
            spec_for(initial, state.kind, state.cell_kind),
            state,
        )
        parameters = dict(parameters)
        parameters["title"] = state.title
        return plot.RasterPlotHost.from_plot(
            plot_input,
            spec,
            size=state.size,
            parameters=parameters,
        )

    presenter = ConsolePresenter(
        session,
        view,
        make_host=make_host,
        spec_for=spec_for,
    )
    try:
        camera_node, camera_snapshot = _one_shot(
            session, producer="camera_measurement"
        )
        frames_signal = camera_node.signal_key("frames")
        calibration_path = tmp_path / "calibration.json"
        _calibration(calibration_path)

        camera_id = presenter.add_logic(
            "camera_measurement",
            node_id="camera_measurement",
            values={
                "exposure_seconds": 0.02,
                "repeat": 1,
                "frames_per_cycle": 3,
            },
            device_keys={"camera": "camera"},
            open_editor=False,
        )
        occupancy_id = presenter.add_logic(
            "occupancy",
            node_id="occupancy",
            artifact_inputs={"calibration_path": str(calibration_path)},
            source_signal=frames_signal,
            open_editor=False,
        )
        # Occupancy runs THROUGH the console: the overlay is assembled by the
        # node that judged these sites, and a console that is not running it
        # has no calibration to speak for.
        assert presenter.start_logic(occupancy_id) is True
        _wait_terminal(presenter.logic[occupancy_id].host)
        presenter.poll_logic()
        judged_signal = stable_signal_key("occupancy", "frame_judged")
        status_signal = stable_signal_key("occupancy", "occupied")
        front = session.signal_plane.freeze()
        judged = front.value(judged_signal)
        judged_publication = front.publication(judged_signal)
        assert judged is not None and judged_publication is not None
        assert session.signal_plane.direct_parent_publications(judged_publication)

        target = presenter.add_panel(
            judged_signal,
            judged.snapshot,
            title="occupancy image",
            kind="image",
            fit={"model": "anisotropic_gaussian_center"},
            overlay_signal=status_signal,
            initial_publication=judged_publication,
        )
        assert {
            key
            for _producer, leaves in view._cards[target.panel_id].overlay_choices
            for _label, key in leaves
        } == {status_signal}
        other = presenter.add_panel(
            frames_signal,
            camera_snapshot,
            title="camera histogram",
            kind="histogram",
        )

        # Header Save Layout is stopped authoring/wiring only.
        layout_path = tmp_path / "task-console-layout.json"
        view.save_answer = str(layout_path)
        view.save_layout_requested.emit()
        document = json.loads(layout_path.read_text(encoding="utf-8"))
        encoded_layout = json.dumps(document)
        assert "dataset" not in encoded_layout and not tuple(tmp_path.glob("*.npz"))

        def no_device_call(*_args, **_kwargs):
            raise AssertionError("layout/panel save touched the current camera")

        monkeypatch.setattr(session.camera, "set_exposure_seconds", no_device_call)
        monkeypatch.setattr(session.camera, "set_roi", no_device_call)
        monkeypatch.setattr(session.camera, "working_point", no_device_call)
        view.open_answer = str(layout_path)
        view.load_layout_requested.emit()
        assert set(presenter.logic) == {camera_id, occupancy_id}
        assert all(binding.host is None for binding in presenter.logic.values())
        assert presenter.logic[occupancy_id].draft.source_signal == frames_signal
        # Two panels, both added here: occupancy is a processor and opens
        # nothing of its own -- it answers a signal it does not own, and which
        # of its answers belongs on this board is the operator's decision.
        assert {
            (binding.state.signal, binding.state.kind)
            for binding in presenter.panels.values()
        } == {(judged_signal, "image"), (frames_signal, "histogram")}
        assert not session.camera.capture_state()

        # The retained publication is the whole overlay artifact.  Remove the
        # stopped Logic row itself, then rebuild the panel from Runtime: no
        # node/binding survives to lend it a SiteMap.
        removed_occupancy = presenter.logic.pop(occupancy_id)
        assert removed_occupancy.host is None and removed_occupancy.node is None
        retained_panel = next(
            binding
            for binding in presenter.panels.values()
            if binding.state.signal == judged_signal
        )
        assert presenter.refresh_panel_snapshot(retained_panel.panel_id)

        # Header Save Screenshot is one GUI image and creates no archive/layout.
        screenshot_path = tmp_path / "task-console.png"
        before_screenshot = set(tmp_path.rglob("*"))
        view.save_answer = str(screenshot_path)
        view.save_screenshot_requested.emit()
        created_by_screenshot = {
            path for path in tmp_path.rglob("*") if path not in before_screenshot
        }
        assert created_by_screenshot == {screenshot_path}
        assert screenshot_path.read_bytes() == b"plain TaskConsole screenshot"

        # Panel Save Fig must use the frozen Edit snapshot and its run-time chain.
        target = next(
            binding
            for binding in presenter.panels.values()
            if binding.state.signal == judged_signal
        )
        deadline = time.monotonic() + 10.0
        while target.frozen_data is None and time.monotonic() < deadline:
            presenter.beat()
            time.sleep(0.005)
        assert presenter.edit_panel(target.panel_id) is True
        frozen = target.frozen_data
        assert frozen is not None and frozen.publication is not None
        frozen_values = np.array(frozen.snapshot.block.values, copy=True)
        other_panel_id = next(
            binding.panel_id
            for binding in presenter.panels.values()
            if binding.state.signal == frames_signal
        )

        def no_latest_recapture():
            raise AssertionError("Panel Save Fig re-froze latest instead of Edit data")

        monkeypatch.setattr(session.signal_plane, "freeze", no_latest_recapture)
        before_panel_save = {
            path.resolve() for path in tmp_path.rglob("*") if path.is_file()
        }
        view.panel_save_figure_requested.emit(
            target.panel_id, str(tmp_path / "occupancy-panel.svg")
        )
        created = {
            path.resolve()
            for path in tmp_path.rglob("*")
            if path.is_file() and path.resolve() not in before_panel_save
        }
        archives = [path for path in created if path.suffix == ".npz"]
        images = [path for path in created if path.suffix == ".svg"]
        assert len(archives) == 1 and len(images) == 1, (
            "Panel Edit Save Fig has no formal image + data archive intent seam; "
            f"status={view.status!r}"
        )

        with np.load(archives[0], allow_pickle=False) as payload:
            assert "info" in payload.files
        info, arrays = read_archive(archives[0])
        sections = info["sections"]
        datasets = sections["dataset"]
        assert len(datasets) == 1 and other_panel_id not in datasets
        dataset_name = next(iter(datasets))
        restored = read_dataset(info, arrays, dataset_name)
        np.testing.assert_array_equal(restored.block.values, frozen_values)

        panel_section = sections["panel"]
        panel_state = panel_section.get("state", panel_section)
        assert panel_state["kind"] == "image"
        assert panel_state["fit"] == {"model": "anisotropic_gaussian_center"}
        assert panel_state["overlay_signal"] == status_signal
        assert sections["overlay"]["overlay_signal"] == status_signal
        assert "calibration_path" not in sections["overlay"]
        # The rings themselves are IN the archive: geometry once, beside the
        # typed status Dataset view it projects.  No flattened
        # repeat/point status table is a second truth.
        assert set(arrays) >= {
            "overlay.coordinates",
            "overlay.status",
        }

        records = {record["node"]: record for record in _records(sections["run_chain"])}
        assert set(records) >= {"camera_measurement", "occupancy"}
        assert records["camera_measurement"]["named_devices"] == {"camera": "camera"}
        assert records["camera_measurement"]["device_snapshots"]["camera"][
            "exposure_seconds"
        ] == pytest.approx(0.02)
        assert records["occupancy"]["parameters"]["calibration_path"] == str(
            calibration_path.resolve()
        )
        encoded_archive = json.dumps(info).lower()
        assert CALIBRATION_SENTINEL.lower() not in encoded_archive
        assert all(word not in encoded_archive for word in ("fingerprint", "sha256", "hash"))
        assert images[0].stat().st_size > 0
    finally:
        presenter.close()
        session.close()
