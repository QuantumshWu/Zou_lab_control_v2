"""Starting the things that publish signals, from the window that shows them.

A panel shows a signal; a logic node is what produces one.  The console had
panels and no way to start any of it, so every signal on screen had to come from
a notebook running beside the window.
"""

from __future__ import annotations

import ast
from copy import deepcopy
import json
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from zlc_runtime import SignalDataPlane, stable_signal_key
from zlc_workbench.device_use import DeviceUseCoordinator
from zlc_workbench.logic import (
    LogicCatalog,
    LogicDraft,
    build_arguments,
    device_key_options,
    finalize_logic_draft,
    make_host,
    source_label,
)

from test_console_presenter import (  # noqa: F401 -- fixtures
    _LogicRowView,
    _box_calibration,
    _console_over,
    _one_shot,
    _settle_panel_hosts,
    _started_camera,
    _settle_logic,
    presenter,
    session,
)
from pulse_fixtures import PULSE_NAME, ordinary_imaging_sequence


def test_workbench_never_imports_a_concrete_logic_node_leaf() -> None:
    """Leaf add/remove stays closed inside zlc_atom discovery and contracts."""

    root = Path(__file__).parents[1] / "src" / "zlc_workbench"
    violations: list[tuple[str, str]] = []
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if node.level == 0 and module.startswith("zlc_atom.nodes."):
                    violations.append((str(path.relative_to(root)), module))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("zlc_atom.nodes."):
                        violations.append(
                            (str(path.relative_to(root)), alias.name)
                        )
    assert not violations, violations


def test_a_source_row_names_a_bundle_as_a_bundle_and_an_input_as_words() -> None:
    """A bundle input is chosen as a whole producer's bundle: "Signal
    bundle".  Any other input reads like the form labels beside it:
    sentence case, the lab's acronyms in capitals -- never str.title()."""

    from zlc_atom.nodes import DatasetInputSpec

    assert source_label(DatasetInputSpec("bundle", None, "exact", select_bundle=True)) == "Signal bundle"
    assert source_label(DatasetInputSpec("frames", None, "exact")) == "Frames"
    assert source_label(DatasetInputSpec("roi_frame", None, "exact")) == "ROI frame"


#: Cameras, so they answer what a camera answers: a draft's units are decided
#: from the bound device, and a bare object() states nothing.
_DEFAULT_CAMERA = SimpleNamespace(photoelectron_conversion=(200.0, 0.107))
_MOT_CAMERA = SimpleNamespace(photoelectron_conversion=(200.0, 0.107))
_SEQUENCER = object()


class _BenchInstallation:
    """Three installed devices, in memory: what composition reads of a bench.

    Enough for catalog, drafts, device-key options, finalization and build
    arguments; a test about starting, claiming or publishing needs the
    virtual apparatus instead.
    """

    devices = {
        "mot_camera": SimpleNamespace(
            capabilities={"camera.adapter": _MOT_CAMERA}, device=_MOT_CAMERA
        ),
        "sequencer": SimpleNamespace(
            capabilities={"sequencer.streamer": _SEQUENCER}, device=_SEQUENCER
        ),
        "camera": SimpleNamespace(
            capabilities={"camera.adapter": _DEFAULT_CAMERA}, device=_DEFAULT_CAMERA
        ),
    }

    def capability(self, token, *, key=None):
        return self.devices[key].capabilities[token]


@pytest.fixture
def bench(tmp_path):
    """A session with a bare plane and the in-memory bench: no apparatus."""

    plane = SignalDataPlane()
    try:
        yield SimpleNamespace(
            signal_plane=plane,
            installation=_BenchInstallation(),
            workspace=SimpleNamespace(root=tmp_path, data=tmp_path),
            day_folder_path=lambda: str(tmp_path),
            # No installation config, so no accepted roles -- what the real
            # session answers in exactly that case.
            device_labels={},
            nodes=(),
            device_use=DeviceUseCoordinator(),
        )
    finally:
        plane.close()


@pytest.fixture
def bench_presenter(bench):
    with _console_over(bench) as presenter:
        yield presenter


def test_the_node_types_offered_are_the_ones_that_exist(bench_presenter) -> None:
    """Not a menu the console keeps.

    A second catalog drifts, and the way it shows up is an operator picking
    something that then refuses to be built.
    """

    from zlc_atom.nodes import discover_logic_nodes

    offered = {name for name, _kind, _publishes in bench_presenter.catalog.rows()}
    assert offered == {item.api_name for item in discover_logic_nodes()}
    assert "camera_measurement" in offered


def test_scan_estimate_uses_the_read_worker_without_starting_a_node(session, monkeypatch) -> None:
    from concurrent.futures import ThreadPoolExecutor
    from threading import get_ident

    caller = get_ident()
    reads = []
    describe = session.sequencer.describe
    def observed_description():
        reads.append(get_ident())
        return describe()

    with ThreadPoolExecutor(max_workers=1) as worker:
        def run(work, finished, failed):
            pending = worker.submit(work)
            def deliver(done):
                try:
                    result = done.result()
                except Exception as error:
                    failed(error)
                else:
                    finished(result)
            pending.add_done_callback(deliver)
        with _console_over(session, run_device_read=run) as control:
            node_id = control.add_logic("seamless_scan")
            control.update_logic_draft(node_id, values={
                "pulse_template": PULSE_NAME + ".json",
                "plan": json.dumps({"axes": [{"port": "manual:power", "values": [1, 2]}]}),
                "repeats": 3, "shots_per_point": 2,
            })
            monkeypatch.setattr(session.sequencer, "describe", observed_description)
            before = session.sequencer.applied()
            answer = control.control_estimate(node_id).result(timeout=10)
            assert answer["status"] == "estimated" and answer["total_shots"] == 12, answer
            assert reads and all(thread != caller for thread in reads)
            assert answer["total_pulse_seconds"] == pytest.approx(3 * answer["first_sweep_pulse_seconds"])
            assert answer["draft_revision"] == control.logic[node_id].draft_revision
            assert control.logic[node_id].host is None and control.logic[node_id].node is None
            assert session.sequencer.applied() is before


def test_a_parked_scan_axis_blocks_nothing_and_a_scanned_one_waits_at_the_draft(session) -> None:
    """An axis the pulse or the bench cannot reach is a draft issue the Edit
    page says while it is scanned -- Start waits instead of the build
    refusing -- and nothing at all once it is switched off (parked)."""

    with _console_over(session) as control:
        node_id = control.add_logic("seamless_scan")

        def issues(*axes) -> tuple[str, ...]:
            control.update_logic_draft(node_id, values={
                "pulse_template": PULSE_NAME + ".json", "plan": json.dumps({"axes": list(axes)}),
            })
            return tuple(control.logic_editor_projection(node_id)["issues"])

        hand = {"port": "manual:power", "values": [1.0, 2.0]}
        gone = {"port": "pulse:param:duration:gone", "values": [1.0, 2.0]}
        absent = {"port": "device:absent:frequency", "values": [1.0]}
        empty = {"port": "manual:angle", "values": [], "mode": "values", "value_text": ""}
        assert any("offers no scan port named 'pulse:param:duration:gone'" in issue
                   and "switch that axis off" in issue for issue in issues(hand, gone))
        assert any("no installed device 'absent'" in issue for issue in issues(hand, absent))
        assert any("Values is empty" in issue for issue in issues(hand, empty))
        parked = issues(hand, *({**axis, "enabled": False} for axis in (gone, absent, empty)))
        assert not [issue for issue in parked if any(
            word in issue for word in ("gone", "absent", "Values", "parked"))], parked
        assert any("every scan axis is parked" in issue for issue in issues({**hand, "enabled": False}))


def test_adding_a_node_creates_only_a_stopped_draft_and_opens_edit(bench_presenter) -> None:
    """Add is authoring, so it cannot build or acquire before Start.

    Editing a stopped row is authoring too: its draft keeps every field of
    the schema with the authored patch on top, and nothing is built.
    """

    presenter = bench_presenter
    added = presenter.control_action("logic.add", {"api_name": "camera_measurement"})
    assert added["accepted"] and added["status"] == "accepted"
    node_id = added["node_id"]
    assert added["state"]["logic"][0]["draft_revision"] == 0
    assert added["state"]["logic"][0]["generation"] is None
    assert added["state"]["reports"][-1]["message"] == f"added {node_id}"

    assert node_id == "camera_measurement"
    assert presenter.view.logic_rows, "the window was never given a row"
    row = presenter.view._rows[node_id]
    assert row.state[0] == "idle"
    assert presenter.logic[node_id].host is None
    assert presenter.logic[node_id].node is None
    assert [name for name, _by, _state in row.publishes] == ["frames"]
    assert row.publishes[0][2] == f"waiting · {node_id}/frames"
    assert presenter.view.focused_logic_editor == node_id
    projection = presenter.view.logic_editors[node_id]
    assert projection["form_spec"].keys == tuple(
        field.name for field in presenter.logic[node_id].descriptor.authoring_schema.fields
    )
    assert projection["form_values"]["repeat"] == 0
    # Blank exposure and ROI keep the camera's current settings.
    assert projection["form_values"]["exposure_seconds"] is None
    assert projection["device_keys"]["camera"] == "camera"

    updated = presenter.control_action(
        "logic.update", {"node_id": node_id, "values": {"repeat": 9, "frames_per_cycle": 2}},
        expected_draft_revision=0,
    )
    assert updated["accepted"]
    assert updated["state"]["logic"][0]["draft_revision"] == 1
    stale = presenter.control_action("logic.start", {"node_id": node_id}, expected_draft_revision=0)
    assert not stale["accepted"] and "draft changed" in stale["error"]
    missing = presenter.control_action("logic.start", {"node_id": "missing"})
    assert missing["error"] == "unknown logic node 'missing'"
    # A refusal without an error of its own answers with the most severe
    # report since the request, not a progress line a refresh added after it.
    def refused_start(selected):
        presenter._report(f"{selected}: Frames must be selected", severity="error")
        presenter._report("calibration: running", severity="task")
        return False
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(presenter, "start_logic", refused_start)
        refused = presenter.control_action("logic.start", {"node_id": node_id})
    assert not refused["accepted"] and refused["error"] == f"{node_id}: Frames must be selected"
    draft = presenter.logic[node_id].draft
    assert set(draft.values) == set(
        presenter.logic[node_id].descriptor.authoring_schema.field_names
    )
    assert draft.values["repeat"] == 9
    assert draft.values["frames_per_cycle"] == 2
    assert "timeout_seconds" not in draft.values
    assert presenter.logic[node_id].host is None
    catalog = presenter.control_catalog(node_id=node_id)
    assert catalog["node"]["form_values"]["repeat"] == 9
    assert catalog["node"]["device_keys"] == {"camera": "camera"}
    assert [item["api_name"] for item in catalog["logic"]] == ["camera_measurement"]
    listing = presenter.control_catalog()
    assert {item["api_name"] for item in listing["logic"]} == set(presenter.catalog.by_name)
    assert all(item["description"] and "dynamic_outputs" in item and "dataset_inputs" in item for item in listing["logic"])
    assert projection["description"] == presenter.catalog.get("camera_measurement").description
    from zlc_workbench.console import control_capabilities
    cold = control_capabilities(presenter.catalog, presenter.session.workspace)
    assert cold == listing
    assert "exponential_decay" in {model["model_id"] for model in cold["plot"]["fit_models"]}
    assert "facet_grid" in cold["panel_kinds"]
    assert all("parameters" not in model for model in cold["plot"]["fit_models"])
    detailed = control_capabilities(presenter.catalog, presenter.session.workspace, details=True)
    assert all(model["parameters"] for model in detailed["plot"]["fit_models"])
    occupancy = control_capabilities(presenter.catalog, presenter.session.workspace, api_name="occupancy")
    artifact = occupancy["logic"][0]["artifact_inputs"][0]
    assert artifact["relative_to"] == str(presenter.session.workspace.data)
    assert "do not prepend" in artifact["path_rule"]
    assert presenter.control_catalog(api_name="camera_measurement")["logic"][0]["fields"]
    # A control client sends values: no field, column or form field says
    # how a GUI control draws it (side-by-side choices, a wheel notch).
    def keys(fields):
        for field in fields:
            yield from field
            yield from keys(field.get("columns", ()))
    for fields in (catalog["logic"][0]["fields"], catalog["logic"][0]["form_spec"]["fields"],
                   catalog["node"]["form_spec"]["fields"], occupancy["logic"][0]["fields"],
                   occupancy["logic"][0]["form_spec"]["fields"]):
        assert fields and {"name", "key"} & set(keys(fields))
        assert not {"segmented", "step"} & set(keys(fields))
    presenter.add_logic("camera_measurement", node_id="other_camera")
    selected = presenter.control_state(node_id=node_id)
    assert [item["node_id"] for item in selected["logic"]] == [node_id]
    assert "draft" not in selected["logic"][0]
    assert presenter.control_state(node_id=node_id, details=True)["logic"][0]["draft"]["values"]["repeat"] == 9
    updated = presenter.control_action("logic.update", {"node_id": node_id, "values": {"repeat": 10}})
    assert [item["node_id"] for item in updated["state"]["logic"]] == [node_id]
    assert presenter.logic["other_camera"].draft.values["repeat"] == 0
    revision = presenter.logic[node_id].draft_revision
    assert presenter.update_logic_draft(node_id, values={"repeat": 10}, normalized=True)
    assert presenter.logic[node_id].draft_revision == revision
    assert presenter.update_logic_draft(node_id, values={"repeat": 9}, normalized=True)
    assert presenter.logic[node_id].draft_revision == revision + 1
    assert presenter.logic[node_id].draft.values["repeat"] == 9
    json.dumps(catalog, allow_nan=False)
    json.dumps(stale, allow_nan=False)


def test_starting_a_node_runs_it_and_the_row_says_so(presenter, session) -> None:
    session.load_pulse(PULSE_NAME)
    node_id = presenter.add_logic(
        "camera_measurement",
        values={"repeat": 1, "frames_per_cycle": 3, "exposure_seconds": 0.005},
    )
    row = presenter.view._rows[node_id]

    started = presenter.control_action("logic.start", {"node_id": node_id}, expected_draft_revision=0)
    assert started["accepted"]
    assert started["state"]["logic"][0]["observation"]["running"]
    assert row.state[0] == "running"
    assert "1/1 node running" in presenter.view.summary
    declarations = presenter.logic[node_id].host.dataset_output_declarations
    assert tuple(value.name for value in declarations) == ("frames",)
    assert {value.contract_id for value in declarations} == {"camera.frames"}

    session.fire(shots=1)
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline and presenter.logic[node_id].host.running:
        presenter.poll_logic()
        time.sleep(0.01)

    assert not presenter.logic[node_id].host.running
    assert row.state[0] != "error", row.state
    assert "0/1 node running" in presenter.view.summary
    # The signal it declared is on the plane, ready for a panel.
    frames_signal = presenter.logic[node_id].host.signal_key("frames")
    value = session.signal_plane.freeze().value(frames_signal)
    assert value is not None
    shape = value.shape
    # (repeat, frame, y, x): the cycle's frames ARE the point axis now.
    assert shape == (1, 3, 96, 128)
    # One bracket per DOMAIN -- repeat, point, cell -- which is what
    # format_signal_shape writes and what the row and the editor read.
    expected = "(1) × (3) × (96 × 128)"
    assert tuple(value[1] for value in row.publishes) == (expected,)
    state = presenter.control_state()
    assert state["logic"][0]["generation"] == presenter.logic[node_id].host.generation.value
    assert state["logic"][0]["observation"]["terminal"]
    assert state["signals"][0]["publication"] is not None
    json.dumps(state, allow_nan=False)
    occupancy_id = presenter.add_logic("occupancy")
    # The source picker names the output; its shape is the editor's to show.
    assert presenter.logic_editor_projection(occupancy_id)["source_labels"] == {
        frames_signal: "frames"
    }


def test_stop_reaches_a_running_node(presenter, session) -> None:
    session.load_pulse(PULSE_NAME)
    node_id = presenter.add_logic("camera_measurement", values={"repeat": 50})
    presenter.start_logic(node_id)

    generation = presenter.logic[node_id].host.generation.value
    refused = presenter.control_action("logic.stop", {"node_id": node_id}, expected_generation="previous-run")
    assert not refused["accepted"] and "run changed" in refused["error"]
    assert not presenter.logic[node_id].host.cancel_requested
    stopped = presenter.control_action("logic.stop", {"node_id": node_id}, expected_generation=generation)
    assert stopped["accepted"]
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline and presenter.logic[node_id].host.running:
        presenter.poll_logic()
        time.sleep(0.01)

    assert not presenter.logic[node_id].host.running


def test_close_keeps_a_row_when_its_worker_has_not_released(presenter) -> None:
    """Close initiates once, then the ordinary beat reaps each real owner."""

    node_id = presenter.add_logic("camera_measurement")
    shutdown: list[bool] = []
    host = SimpleNamespace(
        running=True,
        cancel=lambda _reason: None,
        poll=lambda: None,
        shutdown=lambda: shutdown.append(True),
        observation=SimpleNamespace(
            error=None,
            running=True,
            phase="stopping",
            terminal=False,
            warnings=(),
        ),
        published_signals=lambda: (),
        dataset_output_declarations=(),
        instance_id=node_id,
        signal_key=lambda name: stable_signal_key(node_id, name),
    )
    presenter.logic[node_id].host = host

    started = time.monotonic()
    assert presenter.close() is False
    assert time.monotonic() - started < 0.25

    assert presenter.logic[node_id].host is host
    assert shutdown == []
    presenter.CLOSE_REPORT_SECONDS = 0.0
    presenter.beat()
    assert node_id in presenter.logic
    assert any(
        "close is still waiting" in text
        for severity, text in presenter.view.status
        if severity == "error"
    )

    host.running = False
    host.observation.running = False
    def fail_shutdown() -> None:
        raise RuntimeError("host has not reaped its worker")

    host.shutdown = fail_shutdown
    presenter.beat()
    assert node_id in presenter.logic

    host.shutdown = lambda: shutdown.append(True)
    presenter.beat()
    assert node_id not in presenter.logic
    assert shutdown == [True]
    assert presenter.close() is True


def test_removing_a_node_takes_its_row_and_shuts_it_down(presenter) -> None:
    node_id = presenter.add_logic("camera_measurement")

    presenter.view._rows[node_id].remove_requested.emit()

    assert presenter.logic == {}
    assert presenter.view.logic_rows == ()
    assert node_id not in presenter.view.logic_editors


def test_two_nodes_of_one_type_get_their_own_names(presenter) -> None:
    """They publish under those names, so sharing one would hide the first."""

    first = presenter.add_logic("camera_measurement")
    second = presenter.add_logic("camera_measurement")

    assert (first, second) == ("camera_measurement", "camera_measurement2")
    keys = {stable_signal_key(name, "frames") for name in (first, second)}
    assert len(keys) == 2


def test_a_missing_device_is_a_repairable_draft_until_start(presenter) -> None:

    class _Bare:
        # The console's full installation surface: capability lookup and the
        # devices mapping (the projection offers tunable devices from it).
        devices: dict = {}

        def capability(self, token, *, key=None):
            raise KeyError(f"no {token}")

    presenter.session.installation, real = _Bare(), presenter.session.installation
    try:
        node_id = presenter.add_logic("camera_measurement")
        assert node_id == "camera_measurement"
        assert presenter.logic[node_id].draft.device_keys == {"camera": ""}
        projection = presenter.logic_editor_projection(node_id)
        assert projection["can_start"] is False
        assert any("camera.adapter" in issue for issue in projection["issues"])
        assert presenter.start_logic(node_id) is False
    finally:
        presenter.session.installation = real

    assert presenter.view.status[-1][0] == "error"
    assert "camera.adapter" in presenter.view.status[-1][1]


def test_editing_a_running_row_changes_only_its_shared_draft(presenter, session) -> None:
    """The run keeps the configuration it started with and the draft is the
    next run's.  While the two differ the row, the page and lab_state say so,
    and Revert puts the run's configuration back without touching the run."""

    session.load_pulse(PULSE_NAME)
    node_id = presenter.add_logic("camera_measurement", values={"repeat": 0})
    presenter.start_logic(node_id)
    started = deepcopy(presenter.logic[node_id].draft)
    current_host = presenter.logic[node_id].host
    current_node = presenter.logic[node_id].node
    row = presenter.view._rows[node_id]

    def edited():
        return (
            presenter.view.logic_editors[node_id]["edited"],
            row.state[1].endswith(" · edited"),
            presenter.control_state(node_id=node_id)["logic"][0]["edited_since_start"],
        )

    assert edited() == (False, False, False)
    # A value its field refuses never reaches the draft: no Edit page could show it.
    below = presenter.control_action("logic.update", {"node_id": node_id, "values": {"repeat": -1}})
    assert not below["accepted"] and below["error"] == "Repeat must be at least 0", below
    assert edited() == (False, False, False) and presenter.edit_logic(node_id) is True
    assert presenter.update_logic_draft(node_id, values={"repeat": 9}) is True
    assert presenter.edit_logic(node_id) is True
    assert presenter.logic[node_id].draft.values["repeat"] == 9
    assert presenter.logic[node_id].host is current_host
    assert presenter.logic[node_id].node is current_node
    assert edited() == (True, True, True)
    # Typed back to the value the run uses, the draft is the run's again.
    presenter.view.logic_draft_changed.emit(node_id, {"values": {"repeat": 0}})
    assert edited() == (False, False, False)
    refused = presenter.control_action("logic.revert", {"node_id": node_id})
    assert not refused["accepted"] and "nothing to revert" in refused["error"], refused

    presenter.update_logic_draft(
        node_id, values={"exposure_seconds": 0.02}, device_keys={"camera": ""}
    )
    assert presenter.start_logic(node_id) is False
    assert presenter.logic[node_id].host is current_host
    assert current_host.running, "invalid Restart stopped the valid current run"
    # A refused Restart left the run -- and the configuration it uses -- alone,
    # and says why where it was pressed: on the row, not only in the log.
    assert presenter.view.logic_editors[node_id]["edited"] is True
    # Amber: the run goes on, but not as the operator asked.
    assert row.state[0] == "warning" and " · edited; Restart refused: " in row.state[1], row.state
    reverted = presenter.control_action("logic.revert", {"node_id": node_id})
    assert reverted["accepted"], reverted
    assert presenter.logic[node_id].draft == started
    assert presenter.view.logic_editors[node_id]["form_values"]["repeat"] == 0
    assert edited() == (False, False, False)
    assert "Restart refused" not in row.state[1], "Revert is an edit: the refusal was the old draft's"
    assert row.state[0] == "running", row.state
    assert presenter.logic[node_id].host is current_host and current_host.running
    # The run keeps declaring what IT shows.  A half-typed draft is not even a
    # valid request -- asking it what to plot raised out of the console's poll.
    presenter.beat()
    presenter.poll_logic()
    assert tuple(
        (spec.output.name, spec.plot_kind)
        for spec in presenter.logic[node_id].preview_specs
    ) == (("frames", "facet_grid"),), "the running node's declaration is the run's"
    presenter.stop_logic(node_id)


def test_a_follow_restart_runs_the_configuration_the_follow_started_with(
    presenter, session, tmp_path
) -> None:
    """A following processor restarts by itself when its source does -- from
    the configuration it was started with.  An edit made meanwhile and never
    restarted stays an edit: the follow does not run it behind the operator's
    back (it used to, and the cue cleared with nobody pressing anything), a
    half-typed one with issues neither fails the follow nor ends it, and
    "· edited", Restart* and Revert stay until the operator decides."""

    import shutil

    from test_console_presenter import _follow_calibration_artifact

    first = Path(_follow_calibration_artifact(session, tmp_path)).resolve()
    second = tmp_path / "second-calibration.json"
    shutil.copyfile(first, second)
    camera_id = presenter.add_logic(
        "camera_measurement", node_id="cm-live",
        values={"exposure_seconds": 0.002, "repeat": 0, "frames_per_cycle": 1},
        device_keys={"camera": "camera"}, open_editor=False,
    )
    source = stable_signal_key(camera_id, "frames")
    occupancy_id = presenter.add_logic("occupancy")
    assert presenter.update_logic_draft(
        occupancy_id, source_signal=source,
        artifact_inputs={"calibrations": [{"path": str(first), "frames": ["rest"]}]},
    )
    binding = presenter.logic[occupancy_id]
    session.load_pulse(PULSE_NAME)
    assert presenter.start_logic(occupancy_id), binding.draft_error
    started = deepcopy(binding.draft)

    def source_runs_again() -> None:
        """Start the camera; the follower restarts on the new generation."""

        before = binding.host
        assert presenter.start_logic(camera_id)
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            session.fire(shots=1)
            presenter.beat()
            if binding.host is not before and binding.host is not None and binding.host.running:
                break
            time.sleep(0.005)
        assert binding.host is not before and binding.host.running, (
            binding.draft_error, None if binding.host is None else binding.host.observation
        )

    def source_stops() -> None:
        assert presenter.stop_logic(camera_id)
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and binding.host.running:
            presenter.beat()
            time.sleep(0.005)
        assert not binding.host.running

    source_runs_again()
    for calibration, issues in ((second, False), (tmp_path / "missing.json", True)):
        presenter.view.logic_draft_changed.emit(occupancy_id, {"artifact_inputs": {
            "calibrations": [{"path": str(calibration), "frames": ["rest"]}],
        }})
        assert bool(presenter.view.logic_editors[occupancy_id]["issues"]) is issues
        source_stops()
        # Its source stopped, the follower says it follows -- an edit's
        # issues are the next run's -- and that it is edited.
        assert presenter.view._rows[occupancy_id].state == (
            "running", f"following {presenter._signal_text(source)} (source stopped) · edited"
        )
        source_runs_again()
        assert binding.following, "the edit ended the follow"
        assert [reading.path for reading in binding.node.readings] == [first], (
            "the follow ran an edit nobody restarted"
        )
        projection = presenter.view.logic_editors[occupancy_id]
        assert projection["edited"] is True, "the edit's cue cleared by itself"
        assert projection["status"].endswith(" · edited"), projection["status"]

    presenter.view.logic_revert_requested.emit(occupancy_id)
    assert binding.draft == started
    assert presenter.view.logic_editors[occupancy_id]["edited"] is False
    presenter.stop_logic(occupancy_id)
    presenter.stop_logic(camera_id)


def test_revert_restores_a_following_processors_source_and_calibration_rows(
    presenter, tmp_path
) -> None:
    """Revert puts back the WHOLE configuration a node was started with -- an
    Occupancy's source and calibration rows as well as its values, and
    nothing an edit added -- and a Restart makes the edited draft the
    configuration it runs.  A value name the schema does not declare never
    gets into a draft: logic.add, logic.update and the Edit page are refused
    with the names it has."""

    from test_console_presenter import _box_calibration

    def calibration(name: str) -> str:
        return str(_box_calibration(
            ("site_0001",), ((64.0, 48.0),), (0.0,), (96, 128)
        ).save(tmp_path / name))

    first, second = calibration("first.json"), calibration("second.json")
    names = ", ".join(presenter.catalog.get("occupancy").authoring_schema.field_names)
    refused = presenter.control_action("logic.add", {"api_name": "occupancy", "values": {"nope": 1}})
    assert not refused["accepted"] and refused["node_id"] is None, refused
    assert refused["error"] == f"occupancy has no field 'nope'; its fields are {names}"
    assert not presenter.logic and not presenter.view._rows, "a refused add left a row"
    node_id = presenter.add_logic("occupancy")
    presenter.update_logic_draft(
        node_id,
        source_signal=stable_signal_key("camera-a", "frames"),
        artifact_inputs={"calibrations": [{"path": first, "frames": ["rest"]}]},
    )
    refused = presenter.control_action("logic.revert", {"node_id": node_id})
    assert not refused["accepted"] and "no running configuration" in refused["error"], refused

    # No camera yet: the Start is a standing follow, which is a current run.
    assert presenter.start_logic(node_id), presenter.logic[node_id].draft_error
    binding = presenter.logic[node_id]
    assert binding.following and binding.host is None
    started = deepcopy(binding.draft)
    assert presenter.view.logic_editors[node_id]["edited"] is False
    refused = presenter.control_action("logic.update", {"node_id": node_id, "values": {"nope": 1}})
    assert refused["error"] == f"{node_id} has no field 'nope'; its fields are {names}", refused
    presenter.view.logic_draft_changed.emit(node_id, {"values": {"nope": 1}})
    assert presenter.view.status[-1] == ("error", f"{node_id} has no field 'nope'; its fields are {names}")
    assert binding.draft == started

    presenter.view.logic_draft_changed.emit(node_id, {"values": {"model_kind": "box"}})
    presenter.view.logic_draft_changed.emit(node_id, {
        "source_signal": stable_signal_key("camera-b", "frames"), "device_keys": {"spare": "mot_camera"},
    })
    assert binding.draft.device_keys["spare"] == "mot_camera"
    presenter.view.logic_draft_changed.emit(node_id, {"artifact_inputs": {"calibrations": [
        {"path": first, "frames": ["first"]}, {"path": second, "frames": ["rest"]},
    ]}})
    projection = presenter.view.logic_editors[node_id]
    assert projection["edited"] is True and projection["status"].endswith(" · edited"), projection["status"]
    row = presenter.control_state(node_id=node_id, details=True)["logic"][0]
    assert row["edited_since_start"] and row["draft"]["values"]["model_kind"] == "box", row
    assert row["running_draft"]["values"]["model_kind"] == "default", row["running_draft"]

    presenter.view.logic_revert_requested.emit(node_id)
    assert binding.draft == started
    projection = presenter.view.logic_editors[node_id]
    assert projection["edited"] is False
    assert projection["source_signal"] == stable_signal_key("camera-a", "frames")
    assert projection["form_values"]["model_kind"] == "default"
    assert projection["artifact_claims"]["calibrations"]["entries"] == (
        {"path": first, "frames": ("rest",)},
    )
    assert binding.following and binding.host is None, "Revert restarted nothing"
    again = presenter.control_action("logic.revert", {"node_id": node_id})
    assert not again["accepted"] and "nothing to revert" in again["error"], again

    presenter.view.logic_draft_changed.emit(node_id, {"values": {"model_kind": "box"}})
    assert presenter.view.logic_editors[node_id]["edited"] is True
    assert presenter.start_logic(node_id)
    assert presenter.view.logic_editors[node_id]["edited"] is False
    assert binding.started_draft.values["model_kind"] == "box"
    presenter.stop_logic(node_id)
    presenter.view.logic_draft_changed.emit(node_id, {"values": {"model_kind": "default"}})
    assert presenter.view.logic_editors[node_id]["edited"] is False, "nothing runs"
    assert presenter.control_state(node_id=node_id, details=True)["logic"][0]["running_draft"] is None


def test_a_build_is_handed_only_what_it_asks_for(bench) -> None:
    """Passing every fact and hoping fails on the first build without **values.

    Which is most of them, and it fails naming a keyword rather than the bench
    fact behind it.
    """

    catalog = LogicCatalog()
    descriptor = catalog.get("camera_measurement")
    finalization = finalize_logic_draft(
        descriptor,
        LogicDraft(values={"repeat": 2}, device_keys={"camera": "camera"}),
        installation=bench.installation,
        signal_plane=bench.signal_plane,
        workspace=bench.workspace,
    )
    assert finalization.can_start
    arguments = build_arguments(
        descriptor,
        signal_plane=bench.signal_plane,
        finalization=finalization,
    )

    assert set(arguments) >= {"camera", "signal_plane", "repeat"}
    assert arguments["repeat"] == 2

    figure_writer = lambda *_args, **_kwargs: None

    def task_build(*, camera, signal_plane, save_figure_artifact):
        return camera, signal_plane, save_figure_artifact

    task_descriptor = replace(descriptor, build=task_build)
    task_arguments = build_arguments(
        task_descriptor,
        signal_plane=bench.signal_plane,
        finalization=finalization,
        extras={
            "save_figure_artifact": figure_writer,
            "unrequested_bench_fact": object(),
        },
    )
    assert task_arguments == {
        "camera": _DEFAULT_CAMERA,
        "signal_plane": bench.signal_plane,
        "save_figure_artifact": figure_writer,
    }
    with pytest.raises(ValueError, match="extras collide"):
        build_arguments(
            descriptor,
            signal_plane=bench.signal_plane,
            finalization=finalization,
            extras={"camera": object()},
        )


def test_named_device_options_and_build_resolution_use_compatible_instances() -> None:
    from zlc_atom.nodes.camera_measurement.logic_node import LOGIC_NODE

    descriptor = LOGIC_NODE
    assert device_key_options(descriptor, installation=_BenchInstallation()) == {
        "camera": ("camera", "mot_camera")
    }

    def build(*, camera, camera_key, signal_plane):
        return camera, camera_key, signal_plane

    keyed_descriptor = replace(descriptor, build=build)
    workspace = SimpleNamespace(root=Path.cwd(), data=Path.cwd())
    plane = SimpleNamespace(latest_publication=lambda _name: None)
    def finalized(key: str):
        return finalize_logic_draft(
            keyed_descriptor,
            LogicDraft(device_keys={"camera": key}),
            installation=_BenchInstallation(),
            signal_plane=plane,
            workspace=workspace,
        )

    default = build_arguments(
        keyed_descriptor,
        signal_plane="plane",
        finalization=finalized("camera"),
    )
    selected = build_arguments(
        keyed_descriptor,
        signal_plane="plane",
        finalization=finalized("mot_camera"),
    )

    assert default["camera"] is _DEFAULT_CAMERA
    assert default["camera_key"] == "camera"
    assert selected["camera"] is _MOT_CAMERA
    assert selected["camera_key"] == "mot_camera"
    invalid = finalized("sequencer")
    assert not invalid.can_start
    with pytest.raises(ValueError, match="not startable"):
        build_arguments(
            keyed_descriptor,
            signal_plane="plane",
            finalization=invalid,
        )

    # A mode that delegates acquisition must not resolve or claim the
    # inactive camera/pulse, nor pass their unfinished draft values to build.
    from zlc_atom.authoring import AuthoringField, AuthoringSchema
    from zlc_atom.nodes import DatasetInputSpec, WorkspaceResourceSpec, LogicNodeDescriptor, NodeKind
    from zlc_runtime import FIT_PARAMETER_CONTRACT
    from zlc_workbench.authoring_form import project_logic_schema

    camera_mode = ("mode", ("camera",))
    external_mode = ("mode", ("external",))
    conditional = LogicNodeDescriptor(
        "conditional", NodeKind.MEASUREMENT,
        description="Acquire in camera mode or follow an external signal in external mode.",
        authoring_schema=AuthoringSchema((
            AuthoringField("mode", "str", "Mode", "camera"),
            AuthoringField("pulse", "resource", "Pulse", "", required=True, enabled_when=camera_mode),
            AuthoringField("exposure", "float", "Exposure", 0.1, minimum=0.0, enabled_when=camera_mode),
            AuthoringField("acquisition_logic", "str", "Acquisition logic", "", required=True, enabled_when=external_mode),
        )),
        device_requirements=(replace(descriptor.device_requirements[0], enabled_when=camera_mode),),
        input_specs=(DatasetInputSpec("signal", FIT_PARAMETER_CONTRACT, "latest", enabled_when=external_mode),),
        workspace_resources=(WorkspaceResourceSpec(
            "pulse", "test.pulse", "pulses", (".json",),
            lambda _path: pytest.fail("inactive Pulse was decoded"),
            argument_name="pulse_resource",
        ),),
        acquisition_input="acquisition_logic",
        acquisition_completion="terminal",
        build=lambda *, signal_plane, source_signal="", **values: values,
    )
    draft = LogicDraft(
        values={"mode": "external", "exposure": "unfinished", "pulse": "missing.json", "acquisition_logic": "scan"},
        source_signal="@logic/panel-1/frequency",
        device_keys={"camera": "sequencer"},
    )
    final = finalize_logic_draft(
        conditional, draft, installation=_BenchInstallation(), signal_plane=plane,
        workspace=workspace, source_options=(draft.source_signal,), acquisition_options=("scan",),
    )
    assert final.can_start, final.issues
    assert not final.devices and not final.resources
    assert conditional.active_dataset_inputs(final.values)
    assert not conditional.active_device_requirements(final.values)
    args = build_arguments(conditional, signal_plane=plane, finalization=final)
    assert args == {"mode": "external", "acquisition_logic": "scan", "signal_plane": plane, "source_signal": draft.source_signal}
    form = project_logic_schema(conditional, workspace_root=str(Path.cwd()), acquisition_options=("scan",))
    pulse_field = next(field for field in form.fields if field.key == "pulse")
    assert pulse_field.enabled_when == camera_mode
    draft.values["mode"] = "camera"
    final = finalize_logic_draft(
        conditional, draft, installation=_BenchInstallation(), signal_plane=plane,
        workspace=workspace, source_options=(draft.source_signal,),
    )
    assert not final.can_start
    assert not final.source_signal
    assert any("exposure" in issue.lower() or "could not convert" in issue for issue in final.issues)
    assert draft.values["exposure"] == "unfinished"


def _claim_descriptor(
    api_name: str,
    *capabilities: str,
    protected_fields: tuple[str, ...] = (),
):
    from zlc_atom.authoring import AuthoringSchema
    from zlc_atom.nodes import (
        DeviceRequirement,
        LogicNodeDescriptor,
        NodeKind,
    )

    class _Hold:
        def execute(self, context):
            while not context.cancel_requested():
                time.sleep(0.001)

    def build(*, device_0, device_1=None):
        return _Hold()

    requested = capabilities or ("camera.adapter",)
    return LogicNodeDescriptor(
        api_name,
        # These tests exercise device-use arbitration between ordinary
        # concurrent nodes.  A Task intentionally owns the whole console, so
        # using Task here would test (and violate) the Task admission rule
        # before the device-use behavior under test is reached.
        NodeKind.MEASUREMENT,
        AuthoringSchema(),
        description="Hold selected device claims until cancelled.",
        device_requirements=tuple(
            DeviceRequirement(
                token,
                f"device_{index}",
                protected_fields if index == 0 else (),
            )
            for index, token in enumerate(requested)
        ),
        build=build,
    )


def test_logic_claim_carries_descriptor_protected_fields(presenter) -> None:
    descriptor = _claim_descriptor(
        "protected",
        protected_fields=("exposure", "roi_x"),
    )
    presenter.catalog = LogicCatalog((descriptor,))
    node_id = presenter.add_logic("protected")

    assert presenter.start_logic(node_id) is True
    revision, owners, policy = presenter.session.device_use.field_policy(
        "camera",
        ("exposure", "gain", "roi_width"),
        dependency_groups=(("roi_x", "roi_width"),),
    )
    assert revision > 0
    assert owners == (node_id,)
    assert policy == {
        "exposure": (node_id,),
        "gain": (),
        "roi_width": (node_id,),
    }


def test_device_setting_history_records_only_worker_verified_active_changes(
    session,
) -> None:
    from zlc_workbench.device_use import DeviceClaim

    device = session.installation.device("camera")
    lease = session.device_use.prepare_logic(
        object(),
        "camera measurement",
        (DeviceClaim("camera", "camera", device, ("exposure",)),),
        stop=lambda _reason: None,
        superseded=lambda: None,
    ).commit()
    session.record_device_tune(
        device_key="camera",
        field="gain",
        requested=8.0,
        previous_effective=6.0,
        new_effective=8.0,
        before_provenance={
            "device_session_id": "camera-session",
            "settings_epoch": 0,
        },
        after_provenance={
            "device_session_id": "camera-session",
            "settings_epoch": 1,
        },
        previous_values={"gain": 6.0},
        current_values={"gain": 8.0},
        active_logic_owners=("camera measurement",),
    )
    event = {
        "device_settings": {
            "camera": {
                "device_session_id": "camera-session",
                "epoch_ranges": [[0, 1]],
                "mixed": True,
            }
        }
    }
    records = session.resolve_device_setting_records((event,))
    assert [record["settings_epoch"] for record in records] == [0, 1]
    assert records[1]["requested"] == 8.0

    lease.release()
    assert session.record_device_tune(
        device_key="camera",
        field="gain",
        requested=10.0,
        previous_effective=8.0,
        new_effective=10.0,
        before_provenance={
            "device_session_id": "camera-session",
            "settings_epoch": 1,
        },
        after_provenance={
            "device_session_id": "camera-session",
            "settings_epoch": 2,
        },
        previous_values={"gain": 8.0},
        current_values={"gain": 10.0},
        active_logic_owners=(),
    ) is None
    assert session.resolve_device_setting_records((event,)) == records


def test_pending_logic_reserves_every_device_before_old_logic_stops(presenter) -> None:
    """A Pulse command cannot enter through the candidate's currently-free device.

    A claim on a device another row holds queues, stops the old row -- which
    keeps its place on the board -- and then starts.
    """

    from zlc_workbench.device_use import DeviceClaim, DeviceUseBusy

    old = _claim_descriptor("old", "camera.adapter")
    replacement = _claim_descriptor(
        "replacement",
        "camera.adapter",
        "sequencer.streamer",
    )
    presenter.catalog = LogicCatalog((old, replacement))
    old_id = presenter.add_logic("old")
    replacement_id = presenter.add_logic("replacement")
    assert presenter.start_logic(old_id) is True
    old_host = presenter.logic[old_id].host
    assert presenter.start_logic(old_id)
    deadline = time.monotonic() + 2.0
    while presenter.logic[old_id].pending is not None and time.monotonic() < deadline:
        presenter.poll_logic()
        time.sleep(.001)
    assert presenter.logic[old_id].host is not old_host
    old_host = presenter.logic[old_id].host
    assert old_host.running
    assert presenter.start_logic(replacement_id) is True
    assert presenter.logic[replacement_id].pending is not None
    assert presenter.logic[replacement_id].host is None

    pulse_owner = object()
    with pytest.raises(DeviceUseBusy, match="replacement"):
        presenter.session.device_use.acquire_command(
            pulse_owner,
            "PulseGUI",
            (
                DeviceClaim(
                    "sequencer",
                    "sequencer",
                    presenter.session.sequencer,
                ),
            ),
        )

    deadline = time.monotonic() + 2.0
    while (
        time.monotonic() < deadline
        and presenter.logic[replacement_id].pending is not None
    ):
        presenter.poll_logic()
        time.sleep(0.001)
    assert old_id in presenter.logic, "the conflicting draft row was removed"
    assert old_host is not None and not old_host.running
    assert presenter.logic[replacement_id].host is not None
    assert presenter.logic[replacement_id].host.running


def _pulse_presenter_on_session(presenter):
    from test_pulse_editor import _EditorView, PulseEditorPresenter

    view = _EditorView()
    pulse = PulseEditorPresenter(
        view,
        ordinary_imaging_sequence(),
        sequencer=presenter.session.sequencer,
        device_use=presenter.session.device_use,
    )
    return view, pulse


def test_running_sequencer_logic_rejects_pulse_without_touching_device(
    presenter,
    monkeypatch,
) -> None:
    import zlc_pulse

    from zlc_workbench.device_use import DeviceUseBusy

    descriptor = _claim_descriptor(
        "sequencer_task",
        "sequencer.streamer",
    )
    presenter.catalog = LogicCatalog((descriptor,))
    node_id = presenter.add_logic("sequencer_task")
    assert presenter.start_logic(node_id) is True

    events: list[str] = []
    sequencer = presenter.session.sequencer
    for name in ("safe", "load", "fire"):
        original = getattr(sequencer, name)

        def recorded(*args, _name=name, _original=original, **kwargs):
            events.append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(sequencer, name, recorded)

    view, pulse = _pulse_presenter_on_session(presenter)
    try:
        before = tuple(events)
        assert pulse.fire() is False
        assert tuple(events) == before
        assert node_id in view.warnings[-1]
        monkeypatch.setattr(
            zlc_pulse,
            "load_streamer_config",
            lambda: (_ for _ in ()).throw(
                AssertionError("session load consulted process-global board config")
            ),
        )
        with pytest.raises(DeviceUseBusy, match=node_id):
            presenter.session.load_pulse(PULSE_NAME)
        assert tuple(events) == before
        pulse.close()
        assert tuple(events) == before
        assert presenter.logic[node_id].host.running
    finally:
        pulse.close()


def test_notebook_fire_holds_the_session_lease_through_wait_done(
    session,
    monkeypatch,
) -> None:
    from zlc_workbench.device_use import DeviceClaim, DeviceUseBusy

    session.load_pulse(PULSE_NAME)
    original = session.sequencer.wait_done
    observed: list[bool] = []
    reentrant_blocked: list[bool] = []

    def wait_done(timeout):
        try:
            session.fire(shots=1)
        except DeviceUseBusy as error:
            reentrant_blocked.append("ExperimentSession" in str(error))
        else:
            reentrant_blocked.append(False)
        try:
            intruder = session.device_use.acquire_command(
                object(),
                "other driver",
                (
                    DeviceClaim(
                        "sequencer",
                        "sequencer",
                        session.sequencer,
                    ),
                ),
            )
        except DeviceUseBusy as error:
            observed.append("ExperimentSession" in str(error))
        else:
            intruder.release()
            observed.append(False)
        return original(timeout)

    monkeypatch.setattr(session.sequencer, "wait_done", wait_done)
    session.fire(shots=1)
    assert reentrant_blocked == [True]
    assert observed == [True]
    session.device_use.assert_idle()


def test_pulse_drive_rejects_whole_logic_candidate_before_any_logic_is_stopped(
    presenter,
) -> None:
    camera_owner = _claim_descriptor(
        "camera_owner",
        "camera.adapter",
    )
    calibration = _claim_descriptor(
        "calibration_like",
        "camera.adapter",
        "sequencer.streamer",
    )
    presenter.catalog = LogicCatalog((camera_owner, calibration))
    camera_id = presenter.add_logic("camera_owner")
    candidate_id = presenter.add_logic("calibration_like")
    assert presenter.start_logic(camera_id) is True
    camera_host = presenter.logic[camera_id].host
    assert camera_host is not None and camera_host.running

    _view, pulse = _pulse_presenter_on_session(presenter)
    try:
        assert pulse.fire() is True
        assert pulse.running is True
        assert presenter.start_logic(candidate_id) is False
        assert pulse.running is True
        assert camera_host.running, "a rejected candidate cancelled an existing Logic"
        assert presenter.logic[candidate_id].pending is None
        assert presenter.logic[candidate_id].host is None
    finally:
        pulse.stop()
        pulse.close()


def test_restart_is_queued_and_keeps_the_stable_signal_key(presenter, session) -> None:
    from zlc_atom.install import create_installation

    previous_installation = session.installation
    session.installation = create_installation(
        [
            {"key": "camera", "type_id": "camera.virtual", "config": {}},
            {"key": "mot_camera", "type_id": "camera.virtual", "config": {}},
        ]
    )
    previous_installation.close()
    node_id = presenter.add_logic("camera_measurement", values={"repeat": 0})
    assert presenter.start_logic(node_id) is True
    old_host = presenter.logic[node_id].host
    assert old_host is not None
    old_key = old_host.signal_key("frames")
    old_generation = old_host.generation
    started = deepcopy(presenter.logic[node_id].draft)
    edit = {"values": {"exposure_seconds": 0.031}, "device_keys": {"camera": "mot_camera"}}
    presenter.update_logic_draft(node_id, **edit)
    # Revert puts the run's camera back as well as its numbers.
    assert presenter.revert_logic_draft(node_id)
    assert presenter.logic[node_id].draft == started
    presenter.update_logic_draft(node_id, **edit)
    assert presenter.logic_editor_projection(node_id)["edited"] is True

    assert presenter.start_logic(node_id) is True
    assert presenter.logic[node_id].host is old_host
    assert presenter.logic[node_id].pending is not None
    # Queued, the Restart's draft is already the configuration the node runs.
    assert presenter.logic_editor_projection(node_id)["edited"] is False

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and presenter.logic[node_id].host is old_host:
        presenter.poll_logic()
        time.sleep(0.002)
    replacement = presenter.logic[node_id].host
    assert replacement is not None and replacement is not old_host
    assert presenter.logic[node_id].node.camera is session.installation.capability(
        "camera.adapter", key="mot_camera"
    )
    assert replacement.signal_key("frames") == old_key
    assert replacement.generation != old_generation

@pytest.mark.parametrize("ending", ("done", "cancelled", "failed", "wrong-source"))
def test_task_acquisition_waits_for_its_own_scan_and_stops_nested_ready_source(presenter, ending) -> None:
    from threading import Event
    from zlc_atom.authoring import AuthoringField, AuthoringSchema
    from zlc_atom.data import snapshot_from_array
    from zlc_atom.nodes import DatasetInputSpec, LogicNodeDescriptor, NodeKind
    from zlc_runtime import DatasetOutputDeclaration, LiveDatasetOutput, MonitorCoverage

    declaration = DatasetOutputDeclaration("result", "test.scan")
    entered, releases, completed, camera_stops = [], [], [], []

    def camera_build():
        def execute(context):
            context.report_ready()
            while not context.cancel_requested():
                time.sleep(.001)
            camera_stops.append(context.generation)
        return SimpleNamespace(execute=execute)

    def scan_build(*, acquisition_logic, restart_logic):
        def execute(context):
            restart_logic(acquisition_logic, context)
            gate = Event()
            releases.append(gate)
            entered.append(context.generation)
            while not gate.wait(.005):
                if context.cancel_requested():
                    raise InterruptedError("scan stopped")
            if ending == "failed":
                raise RuntimeError("scan hardware failed")
            snapshot = snapshot_from_array([float(len(entered))], producer=context.instance_id,
                signal="result", generation=context.generation.value, revision=1)
            context.commit_live({"result": LiveDatasetOutput(declaration, snapshot, MonitorCoverage(1, 1))})
        return SimpleNamespace(execute=execute)

    def task_build(*, acquisition_logic, restart_logic, source_signal):
        def execute(context):
            context.report_progress("Acquiring complete scans")
            for _ in range(2):
                completed.append(restart_logic(acquisition_logic, context, completion="terminal"))
        return SimpleNamespace(execute=execute)

    acquisition = AuthoringSchema((AuthoringField("acquisition_logic", "str", "Acquisition", ""),))
    camera = LogicNodeDescriptor("ready_source", NodeKind.MEASUREMENT, AuthoringSchema(),
        description="Report camera readiness and hold until cancelled.",
        build=camera_build, reports_ready=True)
    scan = LogicNodeDescriptor("finite_scan", NodeKind.MEASUREMENT, acquisition,
        description="Acquire from the selected ready source and publish a finite scan.",
        build=scan_build, acquisition_input="acquisition_logic", outputs=(declaration,))
    task = LogicNodeDescriptor("feedback", NodeKind.TASK, acquisition, build=task_build,
        description="Acquire two complete finite scans for feedback.",
        acquisition_input="acquisition_logic", acquisition_completion="terminal", node_previews=(),
        input_specs=(DatasetInputSpec("signal", "test.scan", "exact"),))
    presenter.catalog = LogicCatalog((camera, scan, task))
    camera_id = presenter.add_logic("ready_source")
    other_id = presenter.add_logic("ready_source", node_id="unrelated")
    scan_id = presenter.add_logic("finite_scan", values={"acquisition_logic": camera_id})
    source_id = (presenter.add_logic("finite_scan", node_id="another_scan",
                 values={"acquisition_logic": camera_id}) if ending == "wrong-source" else scan_id)
    task_id = presenter.add_logic("feedback", values={"acquisition_logic": scan_id},
                                  source_signal=stable_signal_key(source_id, "result"))

    def advance_until(predicate):
        deadline = time.monotonic() + 5
        while not predicate() and time.monotonic() < deadline:
            presenter.beat()
            time.sleep(.002)
        assert predicate(), {key: None if row.host is None else row.host.observation for key, row in presenter.logic.items()}

    assert presenter.start_logic(camera_id)
    old_camera = presenter.logic[camera_id].host
    assert presenter.start_logic(task_id)
    if ending == "wrong-source":
        advance_until(lambda: presenter.logic[task_id].host.terminal)
        observation = presenter.logic[task_id].host.observation
        assert observation.phase == "failed" and "does not derive from" in observation.error
        assert presenter.logic[scan_id].host is None
        assert presenter.logic[camera_id].host is old_camera and old_camera.running
        assert not completed and not entered
        return
    advance_until(lambda: bool(entered))
    assert presenter.logic[camera_id].host is not old_camera
    assert not completed and presenter.logic[task_id].host.running
    assert not presenter.start_logic(other_id), "Task admission must not allow unrelated nodes"
    if ending == "cancelled":
        presenter.stop_logic(task_id)
    else:
        releases[0].set()
        if ending == "done":
            advance_until(lambda: len(entered) == 2)
            assert len(completed) == 1 and entered[0] != entered[1]
            first = completed[0][stable_signal_key(scan_id, "result")]
            assert first.event_ref.generation == entered[0]
            releases[1].set()
    advance_until(lambda: all(row.host is None or row.host.terminal for row in presenter.logic.values()))
    assert presenter.logic[task_id].host.observation.phase == ending
    assert presenter.logic[camera_id].acquisition_run is None
    assert presenter.logic[scan_id].acquisition_run is None
    assert presenter.logic[task_id].acquisition_run is None
    assert len(completed) == (2 if ending == "done" else 0)
    assert camera_stops
    if ending == "failed":
        assert "scan hardware failed" in presenter.logic[task_id].host.observation.error


def test_a_task_restarting_its_acquisition_runs_the_configuration_that_ran(presenter) -> None:
    """A restart the operator did not press -- a Task restarting the
    Measurement it acquires -- runs that Measurement's current configuration;
    an edit nobody restarted stays an edit, as for a follow restart."""
    from threading import Event
    from zlc_atom.authoring import AuthoringField, AuthoringSchema
    from zlc_atom.nodes import LogicNodeDescriptor, NodeKind

    built, acquired = [], Event()

    def camera_build(*, level):
        def execute(context):
            built.append(level)
            context.report_ready()
            while not context.cancel_requested():
                time.sleep(.001)
        return SimpleNamespace(execute=execute)

    def task_build(*, acquisition_logic, restart_logic):
        def execute(context):
            restart_logic(acquisition_logic, context)
            acquired.set()
            while not context.cancel_requested():
                time.sleep(.001)
        return SimpleNamespace(execute=execute)

    camera = LogicNodeDescriptor("ready_source", NodeKind.MEASUREMENT,
        AuthoringSchema((AuthoringField("level", "int", "Level", 1),)),
        description="Report readiness at a level and hold until cancelled.",
        build=camera_build, reports_ready=True)
    task = LogicNodeDescriptor("feedback", NodeKind.TASK,
        AuthoringSchema((AuthoringField("acquisition_logic", "str", "Acquisition", ""),)),
        description="Acquire the selected ready source once.", build=task_build,
        acquisition_input="acquisition_logic", node_previews=())
    presenter.catalog = LogicCatalog((camera, task))
    camera_id = presenter.add_logic("ready_source")
    task_id = presenter.add_logic("feedback", values={"acquisition_logic": camera_id})
    assert presenter.start_logic(camera_id)
    deadline = time.monotonic() + 5
    while built != [1] and time.monotonic() < deadline:
        presenter.beat()
        time.sleep(.002)
    assert presenter.update_logic_draft(camera_id, values={"level": 7})
    assert presenter.control_state(node_id=camera_id)["logic"][0]["edited_since_start"]

    assert presenter.start_logic(task_id)
    deadline = time.monotonic() + 5
    while not acquired.is_set() and time.monotonic() < deadline:
        presenter.beat()
        time.sleep(.002)
    assert acquired.is_set() and built == [1, 1], built
    row = presenter.control_state(node_id=camera_id)["logic"][0]
    assert row["edited_since_start"], "the Task's restart applied an edit nobody restarted"
    presenter.stop_logic(task_id)
    deadline = time.monotonic() + 5
    while any(b.host is not None and b.host.running for b in presenter.logic.values()) and time.monotonic() < deadline:
        presenter.beat()
        presenter.stop_logic(camera_id)
        time.sleep(.002)


def test_saved_artifact_paths_are_visible_and_seed_matching_input_drafts(
    presenter,
) -> None:
    from zlc_atom.authoring import AuthoringSchema
    from zlc_atom.nodes import (
        ArtifactCodec,
        ArtifactInputSpec,
        ArtifactOutputSpec,
        DatasetInputSpec,
        LogicNodeDescriptor,
        NodeKind,
    )
    from zlc_runtime import DatasetOutputDeclaration

    produced_paths: list[Path] = []

    class _Task:
        def execute(self, context):
            produced = context.run_directory / "calibration.json"
            produced.write_text("{}", encoding="utf-8")
            context.register_artifact(
                "artifact_path",
                produced,
                role="final",
                contract_id="calibration.readout",
            )
            produced_paths.append(produced)
            context.report_progress("artifact saved")
            return SimpleNamespace(artifact_path=produced)

    builds: list[bool] = []

    def _build():
        builds.append(True)
        return _Task()

    descriptor = LogicNodeDescriptor(
        "artifact_task",
        NodeKind.TASK,
        AuthoringSchema(),
        description="Produce a calibration artifact without a live preview.",
        artifact_outputs=(
            ArtifactOutputSpec("artifact_path", "calibration.readout"),
        ),
        node_previews=(),
        build=_build,
    )
    consumer = LogicNodeDescriptor(
        "artifact_consumer",
        NodeKind.PROCESSOR,
        AuthoringSchema(),
        description="Consume camera frames with a saved calibration artifact.",
        input_specs=(
            DatasetInputSpec("frames", "camera.frames", "exact"),
            ArtifactInputSpec(
                "calibration_path",
                "Calibration artifact",
                ArtifactCodec(
                    "calibration.readout",
                    "Calibration artifacts (*.json)",
                    (".json",),
                    lambda path: path.read_text(encoding="utf-8"),
                ),
                argument_name="calibration",
            ),
        ),
        outputs=(DatasetOutputDeclaration("judged", "judged"),),
        build=lambda *, calibration, source_signal, signal_plane: object(),
    )
    presenter.catalog = LogicCatalog((descriptor, consumer))
    node_id = presenter.add_logic("artifact_task")
    assert builds == [], "Add read the artifact workspace and built the task"
    assert presenter.logic_editor_projection(node_id)["artifact_results"] == ()
    assert presenter.start_logic(node_id) is True
    assert builds == [True]
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and presenter.logic[node_id].host.running:
        presenter.poll_logic()
        time.sleep(0.001)

    projection = presenter.logic_editor_projection(node_id)
    produced = produced_paths[0].resolve()
    run_document = json.loads((produced.parent / "run.json").read_text())
    assert run_document["task"] == {
        "api_name": "artifact_task",
        "instance_id": node_id,
    }
    assert run_document["input"] == {
        "authored": {},
        "source_signal": None,
        "devices": {},
        "artifacts": {},
        "resources": {},
    }
    assert projection["artifact_results"] == (
        {
            "name": "artifact_path",
            "contract_id": "calibration.readout",
            "path": str(produced),
            "role": "final",
        },
        {
            "name": "run_directory",
            "contract_id": "zlc.task-run",
            "path": str(produced.parent),
            "role": "run",
        },
    )

    consumer_id = presenter.add_logic("artifact_consumer")
    assert presenter.logic[consumer_id].draft.artifact_inputs == {
        "calibration_path": str(produced)
    }
    assert presenter.logic_editor_projection(consumer_id)["artifact_values"] == {
        "calibration_path": str(produced)
    }

    newer_task = presenter.add_logic("artifact_task")
    assert presenter.start_logic(newer_task) is True
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and presenter.logic[newer_task].host.running:
        presenter.poll_logic()
        time.sleep(0.001)
    newer = produced_paths[1].resolve()
    assert newer.parent.name == "artifact_task-2"
    newer_consumer = presenter.add_logic("artifact_consumer")
    assert presenter.logic[newer_consumer].draft.artifact_inputs == {
        "calibration_path": str(newer.resolve())
    }


def test_a_processor_adds_with_an_unresolved_source_and_no_modal(
    presenter, monkeypatch
) -> None:
    monkeypatch.setattr(
        presenter.session,
        "day_folder",
        lambda: pytest.fail("opening Logic Edit created an artifact directory"),
    )
    node_id = presenter.add_logic("occupancy")

    assert node_id == "occupancy"
    assert presenter.logic[node_id].draft.source_signal == ""
    assert presenter.logic[node_id].draft.artifact_inputs == {"calibrations": ()}
    assert presenter.logic[node_id].draft.values == {"model_kind": "default"}
    # Calibrations are rows of files and frames, edited by Occupancy's own
    # editor rather than the path form.
    assert presenter.view.logic_editors[node_id]["artifact_form_spec"].keys == ()
    claims = presenter.view.logic_editors[node_id]["artifact_claims"]["calibrations"]
    assert claims["base_dir"] == str(presenter.session.workspace.data)
    assert claims["entries"] == () and claims["reads"] is None
    assert presenter.view.logic_editors[node_id]["source_required"] is True
    assert presenter.view.logic_editors[node_id]["source_options"] == ()
    # With no source, Start is disabled before the click and refused at it.
    projection = presenter.logic_editor_projection(node_id)
    assert projection["can_start"] is False
    # Said by the row's label, as the operator sees it, not the draft's key.
    assert "Frames must be selected" in projection["issues"]
    assert presenter.start_logic(node_id) is False
    assert presenter.logic[node_id].host is None
    assert presenter.logic[node_id].draft_error == "Frames must be selected"

    camera_id = presenter.add_logic("camera_measurement")
    assert presenter.view.logic_editors[node_id]["source_options"] == (
        stable_signal_key(camera_id, "frames"),
    )
    # Declared and not yet published, it is named in the words it will
    # keep once it publishes: its producer, then its output.
    assert presenter.view.logic_editors[node_id]["source_labels"] == {
        stable_signal_key(camera_id, "frames"): "frames"
    }
    assert presenter.view.logic_editors[node_id]["source_groups"] == {
        stable_signal_key(camera_id, "frames"): camera_id
    }
    # Closed, the picker reads it as a status reads it.
    assert presenter.view.logic_editors[node_id]["source_texts"] == {
        stable_signal_key(camera_id, "frames"): f"{presenter.logic[camera_id].name}/frames"
    }

    presenter.view.logic_draft_changed.emit(
        node_id,
        {"artifact_inputs": {"calibrations": [{"path": "manually-picked.json", "frames": ["rest"]}]}},
    )
    picked = ({"path": "manually-picked.json", "frames": ("rest",)},)
    assert presenter.logic[node_id].draft.artifact_inputs == {"calibrations": picked}
    assert presenter.logic_editor_projection(node_id)["artifact_claims"]["calibrations"]["entries"] == picked
    derived_id = presenter.add_logic("derive")
    projection = presenter.logic_editor_projection(derived_id)
    assert projection["source_bundle"] is True
    assert projection["source_label"] == "Signal bundle"
    bundle_members = [
        name for name in projection["source_options"]
        if name.startswith(f"@logic/{node_id}/")
    ]
    assert bundle_members == [stable_signal_key(node_id, "counts")]
    # A bundle reads as its members in the order Derive's Inputs list
    # them -- the producer's declared outputs -- the first spelled as a
    # status spells it.
    declared = ("counts", "occupied", "frame_judged")
    assert projection["source_labels"][bundle_members[0]] == (
        f"{presenter.logic[node_id].name}/counts, occupied, frame_judged"
    )
    presenter.view.logic_draft_changed.emit(
        derived_id, {"source_signal": stable_signal_key(node_id, "occupied")}
    )
    projection = presenter.logic_editor_projection(derived_id)
    assert stable_signal_key(node_id, "occupied") in projection["source_options"]
    assert stable_signal_key(node_id, "counts") not in projection["source_options"]
    # Whichever member the draft names keys the choice; the words keep
    # the Inputs' order.
    assert projection["source_labels"][stable_signal_key(node_id, "occupied")] == (
        f"{presenter.logic[node_id].name}/counts, occupied, frame_judged"
    )
    assert tuple(name for name, _schema, _snapshot in projection["input_bundle"]) == declared


def test_an_occupancys_frame_claims_survive_its_camera_changing_frame_count(presenter) -> None:
    """The camera's draft says how many frames a cycle has -- before anything
    is published -- and the claims are resolved against it without ever being
    rewritten: Last follows the count, a middle number keeps its place, and a
    middle number the cycle no longer has between its ends is noted, kept and
    stops nothing.  The same claims read it again once the count that suits
    them returns."""

    camera_id = presenter.add_logic("camera_measurement", values={"frames_per_cycle": 3})
    node_id = presenter.add_logic("occupancy")
    assert presenter.view.logic_editors[node_id]["source_frames"] is None
    presenter.view.logic_draft_changed.emit(
        node_id, {"source_signal": stable_signal_key(camera_id, "frames")}
    )
    assert presenter.view.logic_editors[node_id]["source_frames"] == ("0", "1", "2")
    survival_id = presenter.add_logic("frame_survival", source_signal=stable_signal_key(node_id, "occupied"))
    assert presenter.view.logic_editors[survival_id]["source_frames"] == ("0", "1", "2")
    selection = {"selection": "from_first", "pairs": ({"start": "0", "end": "last", "selected": False},),
                 "custom_pairs": ()}
    presenter.view.logic_draft_changed.emit(survival_id, {"values": selection})
    claims = (
        {"path": "load.json", "frames": (2,)},
        {"path": "readout.json", "frames": ("last",)},
        {"path": "rest.json", "frames": ("rest",)},
    )
    presenter.view.logic_draft_changed.emit(node_id, {"artifact_inputs": {"calibrations": claims}})
    editor = presenter.view.logic_editors[node_id]["artifact_claims"]["calibrations"]
    assert editor["problems"] == ()
    assert editor["notes"] == (
        "load's middle frame 2 is not in this source (frames 0–2); it is kept for a cycle that has it",)
    assert editor["reads"] == ((), (2,), (0, 1))
    presenter.view.logic_draft_changed.emit(camera_id, {"values": {"frames_per_cycle": 2}})
    assert presenter.view.logic_editors[survival_id]["source_frames"] == ("0", "1")
    editor = presenter.view.logic_editors[node_id]["artifact_claims"]["calibrations"]
    assert editor["problems"] == ()
    assert editor["notes"] == (
        "load's middle frame 2 is not in this source (frames 0–1); it is kept for a cycle that has it",)
    assert editor["reads"] == ((), (1,), (0,))
    assert presenter.logic[node_id].draft.artifact_inputs == {"calibrations": claims}, (
        "a frame count never edits the draft"
    )
    issues = presenter.logic_editor_projection(node_id)["issues"]
    assert not any("middle frame 2" in issue for issue in issues), "a note never refuses Start"
    presenter.view.logic_draft_changed.emit(camera_id, {"values": {"frames_per_cycle": 4}})
    editor = presenter.view.logic_editors[node_id]["artifact_claims"]["calibrations"]
    assert editor["problems"] == () and editor["notes"] == ()
    assert editor["reads"] == ((2,), (3,), (0, 1))
    assert presenter.logic[node_id].draft.artifact_inputs == {"calibrations": claims}
    assert presenter.view.logic_editors[survival_id]["source_frames"] == ("0", "1", "2", "3")
    for count in (22, 25, 1, 22):
        presenter.view.logic_draft_changed.emit(camera_id, {"values": {"frames_per_cycle": count}})
        expected = tuple(map(str, range(count)))
        assert presenter.view.logic_editors[node_id]["source_frames"] == expected
        assert presenter.view.logic_editors[survival_id]["source_frames"] == expected
        assert presenter.logic[survival_id].draft.values == selection
    assert all(binding.host is None for binding in presenter.logic.values())
    path = presenter.session.workspace.data / "frame-choices.json"
    assert presenter.save_layout(str(path))
    assert presenter.load_layout(str(path))
    assert presenter.edit_logic(survival_id)
    assert presenter.logic[survival_id].draft.values == selection
    assert presenter.view.logic_editors[survival_id]["source_frames"] == tuple(map(str, range(22)))
    assert presenter.logic[node_id].draft.artifact_inputs == {"calibrations": claims}


def test_a_frame_survival_after_derive_is_edited_before_anything_starts(presenter, session) -> None:
    """The user's chain with nothing started: a camera's three frames reach
    Frame Survival through Occupancy and Derive's draft (``same`` internal,
    ``alive`` -- the verdicts where frames 0 and 2 agree -- published), so
    its pairs are marked at once, and a camera edit re-projects them
    through Derive.  Derive's own editor offers the frame pick before data.
    A variable picking one frame passes one frame, and Frame Survival says
    why there is no pair.  Then the whole chain runs, Frame Survival too;
    afterwards, while Derive's draft cannot say, what it published answers.
    A Derive over Frame Survival is told the pair axis by its name, and
    passes on what each of its variables keeps of it."""

    from PyQt5 import QtCore, QtTest
    from zlc_ui.qt import ensure_qt_app

    app = ensure_qt_app(["zlc-derive-frames-test"])
    camera_id = presenter.add_logic(
        "camera_measurement", values={"repeat": 1, "frames_per_cycle": 3, "exposure_seconds": 0.005})
    occupancy_id = presenter.add_logic("occupancy", source_signal=stable_signal_key(camera_id, "frames"))
    same = {"name": "same", "publish": False, "terms": (
        {"of": "occupied", "at": "frame", "labels": ("0",)},
        {"op": "==", "of": "occupied", "at": "frame", "labels": ("2",)})}
    alive = {"name": "alive", "terms": ({"of": "occupied"}, {"op": "where", "of": "same"})}
    derive_id = presenter.add_logic(
        "derive", source_signal=stable_signal_key(occupancy_id, "occupied"), values={"variables": (same, alive)})
    survival_id = presenter.add_logic("frame_survival", source_signal=stable_signal_key(derive_id, "alive"))
    editors = presenter.view.logic_editors
    assert editors[survival_id]["source_frames"] == ("0", "1", "2")

    derive = presenter.logic_editor_projection(derive_id)
    assert derive["source_frames"] == ("0", "1", "2"), "every occupancy output carries the camera's frames"
    assert derive["input_bundle"] == (("counts", None, None), ("occupied", None, None), ("frame_judged", None, None))
    derive_editor = presenter.logic[derive_id].descriptor.ui_contributions[0]()
    form = presenter.logic[survival_id].descriptor.ui_contributions[0]()
    form.draft_changed.connect(lambda patch: presenter.view.logic_draft_changed.emit(survival_id, patch))

    def texts(combo):
        return [combo.itemText(row) for row in range(combo.count())]

    def show(widget, projection):
        widget.update_projection(projection)
        app.processEvents()

    try:
        # Derive's own editor, untyped: the members by name, and on each the
        # frame axis its source's draft declares, with those labels.
        show(derive_editor, derive)
        head = derive_editor._cards[1].head
        assert texts(head["operand"])[:4] == ["Inputs", "    counts", "    occupied", "    frame_judged"]
        assert texts(head["axis"]) == ["All", "Point", "    frame"]
        assert texts(derive_editor._cards[0].head["labels"]) == ["All", "first", "last", "frame", "0", "1", "2"]
        assert not derive_editor._cards[1].terms_ui[1]["axis"].isVisibleTo(derive_editor), "same keeps no frame"

        # Frame Survival's pairs, marked by a click before anything runs.
        form.resize(520, 260)
        form.show()
        show(form, editors[survival_id])
        assert form.table.isEnabled() and form.summary.text() == "All · 3 of 3 pairs"
        view = form.table.viewport()
        QtTest.QTest.mouseClick(view, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier,
                                form.table.visualRect(form.model.index(0, 0)).center())
        app.processEvents()
        selection = presenter.logic[survival_id].draft.values
        assert selection["selection"] == "all" and sorted(
            (row["start"], row["end"], row["selected"]) for row in selection["pairs"]) == [
            ("0", "last", False), ("1", "last", False)]
        show(form, editors[survival_id])
        assert form.summary.text() == "All · 1 of 3 pairs"

        # A camera edit reaches Frame Survival through Derive's draft.
        presenter.view.logic_draft_changed.emit(camera_id, {"values": {"frames_per_cycle": 4}})
        assert editors[survival_id]["source_frames"] == ("0", "1", "2", "3")
        assert editors[derive_id]["source_frames"] == ("0", "1", "2", "3")
        show(form, editors[survival_id])
        assert form.summary.text() == "All · 4 of 6 pairs"

        # One frame picked: one frame passed on, and why there is no pair.
        one = {"name": "alive", "terms": ({"of": "occupied", "at": "frame", "labels": ("1",)},
                                          {"op": "where", "of": "same"})}
        presenter.view.logic_draft_changed.emit(derive_id, {"values": {"variables": (same, one)}})
        assert editors[survival_id]["source_frames"] == ("0",)
        show(form, editors[survival_id])
        assert not form.table.isEnabled()
        assert form.summary.text() == "One frame per cycle: no pairs"
        assert presenter.logic[survival_id].draft.values == selection, "a frame count never edits the draft"
        assert all(binding.host is None for binding in presenter.logic.values())

        # Run it, Frame Survival on Derive's verdicts too.  Afterwards the
        # draft says what it can; where it cannot -- a formula being edited
        # -- what Derive published answers.
        calibration = presenter.session.workspace.data / "box-calibration.json"
        _box_calibration(("site-0", "site-1"), ((12.0, 10.0), (30.0, 20.0)), (100.0, 100.0), (96, 128)).save(
            calibration)
        presenter.view.logic_draft_changed.emit(occupancy_id, {"artifact_inputs": {"calibrations": (
            {"path": str(calibration), "frames": ("rest",)},)}})
        presenter.view.logic_draft_changed.emit(camera_id, {"values": {"frames_per_cycle": 3}})
        presenter.view.logic_draft_changed.emit(derive_id, {"values": {"variables": (same, alive)}})
        presenter.view.logic_draft_changed.emit(survival_id, {"values": {
            "selection": "all", "pairs": (), "custom_pairs": ()}})
        session.load_pulse(PULSE_NAME)
        assert presenter.start_logic(camera_id)
        session.fire(shots=1)
        for node_id in (camera_id, occupancy_id, derive_id, survival_id):
            if node_id != camera_id:
                assert presenter.start_logic(node_id), presenter.view.status[-3:]
            _settle_logic(presenter, node_id)
            assert presenter.logic[node_id].host.observation.error is None, node_id
        assert session.signal_plane.latest_publication(stable_signal_key(derive_id, "alive")) is not None
        assert session.signal_plane.latest_publication(stable_signal_key(survival_id, "survival")) is not None
        presenter.view.logic_draft_changed.emit(camera_id, {"values": {"frames_per_cycle": 4}})
        assert editors[survival_id]["source_frames"] == ("0", "1", "2", "3"), "the draft says"
        editing = {"name": "alive", "terms": (*alive["terms"], {"op": "+"})}
        presenter.view.logic_draft_changed.emit(derive_id, {"values": {"variables": (same, editing)}})
        binding = presenter.logic[derive_id]
        assert binding.descriptor.describe_output_frames(
            binding.draft.values, "alive", ("frame", ("0", "1", "2", "3"))) is None
        assert editors[survival_id]["source_frames"] == ("0", "1", "2"), "what Derive published"

        # Frame Survival's readout-event axis is "pair": a Derive over it is
        # told that name with the pairs' labels, and what it passes on
        # follows each variable's pick or statistic on it.
        ends = {"name": "ends", "terms": ({"of": "survival", "at": "pair", "labels": ("first", "last")},)}
        over_pairs = presenter.add_logic(
            "derive", source_signal=stable_signal_key(survival_id, "survival"), values={"variables": (ends,)})
        projection = presenter.logic_editor_projection(over_pairs)
        assert (projection["source_frame_axis"], projection["source_frames"]) == ("pair", ("0→1", "0→2", "1→2"))
        downstream = presenter.add_logic("frame_survival", source_signal=stable_signal_key(over_pairs, "ends"))
        assert editors[downstream]["source_frames"] == ("0→1", "1→2")
        rate = {"name": "ends", "terms": ({"of": "survival"},), "statistic": "mean", "over": ("pair",)}
        presenter.view.logic_draft_changed.emit(over_pairs, {"values": {"variables": (rate,)}})
        assert editors[downstream]["source_frames"] == ("0",), "the pair axis reduced away"
    finally:
        for widget in (derive_editor, form):
            widget.close()
            widget.deleteLater()
        app.processEvents()


def test_a_producers_new_output_reaches_every_open_editors_sources(presenter) -> None:
    """A derive publishes the variables it names.  Naming another changes
    what every other node may choose as its source, so every other open
    editor is re-projected -- the way adding or removing a node already
    re-projects them.  A draft edit that leaves the outputs alone reaches
    only the editors of the nodes reading this one."""

    camera_id = presenter.add_logic("camera_measurement")
    producer_id = presenter.add_logic("derive")
    presenter.view.logic_draft_changed.emit(
        producer_id, {"source_signal": stable_signal_key(camera_id, "frames")}
    )
    consumer_id = presenter.add_logic("derive")
    assert consumer_id != producer_id
    total = stable_signal_key(producer_id, "total")
    assert total not in presenter.view.logic_editors[consumer_id]["source_options"]
    presenter.view.logic_draft_changed.emit(
        producer_id,
        {"values": {"variables": ({"name": "total", "terms": ({"of": "frames"},)},)}},
    )
    assert total in presenter.view.logic_editors[consumer_id]["source_options"]


def test_a_calibrations_api_fields_come_from_its_pulse_unless_chosen(presenter, session) -> None:
    """Empty API fields show the pulse's first three API parameters in
    period order; the draft itself stays empty, so the default follows the
    pulse; a field the operator chose stands.

    A calibration just added has no pulse and no API fields yet, so its
    draft does not project -- and its camera exposure still shows, because
    the node derives it from the two windows alone, and follows them as
    they are typed.  It was blank until every required field was filled."""

    import shutil

    node_id = presenter.add_logic("calibration")
    editor = presenter.view.logic_editors[node_id]
    assert editor["can_start"] is False
    assert editor["form_values"]["camera_exposure_seconds"] == 0.02
    presenter.view.logic_draft_changed.emit(
        node_id, {"values": {"reference_exposure_seconds": 0.03}}
    )
    assert presenter.view.logic_editors[node_id]["form_values"]["camera_exposure_seconds"] == 0.03

    fixture = Path(__file__).resolve().parents[2] / "zlc_atom" / "tests" / "pulses" / "imaging_template.json"
    shutil.copy(fixture, Path(session.workspace.root) / "pulses" / "calib.json")
    presenter.view.logic_draft_changed.emit(node_id, {"values": {"pulse_template": "calib.json"}})
    projection = presenter.logic_editor_projection(node_id)
    assert projection["form_values"]["reference_before_field"] == "duration:long_before"
    assert projection["form_values"]["readout_field"] == "duration:short"
    assert projection["form_values"]["reference_after_field"] == "duration:long_after"
    assert presenter.logic[node_id].draft.values["readout_field"] == "", "a default is not written into the draft"
    # The camera exposure is the node's, derived from the windows: shown,
    # disabled with the reason on it, following the reference window, and
    # never written into the draft.
    assert projection["form_values"]["camera_exposure_seconds"] == 0.03
    exposure = next(f for f in projection["form_spec"].fields if f.key == "camera_exposure_seconds")
    assert exposure.unavailable and "longest" in exposure.unavailable_reason
    presenter.view.logic_draft_changed.emit(node_id, {"values": {"reference_exposure_seconds": 0.05}})
    assert presenter.logic_editor_projection(node_id)["form_values"]["camera_exposure_seconds"] == 0.05
    assert presenter.logic[node_id].draft.values.get("camera_exposure_seconds") is None
    presenter.view.logic_draft_changed.emit(
        node_id, {"values": {"readout_field": "duration:long_after"}}
    )
    projection = presenter.logic_editor_projection(node_id)
    assert projection["form_values"]["readout_field"] == "duration:long_after"
    assert projection["form_values"]["reference_before_field"] == "duration:long_before"


def test_make_host_hands_the_descriptor_contract_to_its_node_host() -> None:
    """The host takes what the descriptor declares, never a node attribute.

    A derive reads the members its variables name: the declaration says
    there is an input, the instance says which siblings of it to fetch, and
    what it publishes is named by its draft, not by its kind.
    """

    from zlc_atom.authoring import AuthoringSchema
    from zlc_atom.nodes import DatasetInputSpec, LogicNodeDescriptor, NodeKind
    from zlc_runtime import DatasetOutputDeclaration

    def descriptor(api_name, kind, input_spec, **declared):
        return LogicNodeDescriptor(
            api_name,
            kind,
            AuthoringSchema(),
            description="Declare authored or static outputs from the selected source.",
            input_specs=(input_spec,),
            build=lambda **_values: object(),
            **declared,
        )

    expression = descriptor(
        "expression_processor",
        NodeKind.PROCESSOR,
        DatasetInputSpec("a", None, "exact"),
        declare_outputs=lambda values, _devices: (
            DatasetOutputDeclaration(str(values["name"]), "derive.value"),
        ),
    )
    output = DatasetOutputDeclaration("judged", "occupancy.judged")
    processor = descriptor(
        "explicit_processor",
        NodeKind.PROCESSOR,
        DatasetInputSpec("frames", "camera.frames", "exact"),
        outputs=(output,),
    )
    scan = descriptor(
        "explicit_scan",
        NodeKind.MEASUREMENT,
        DatasetInputSpec("source", None, "exact"),
        outputs=(DatasetOutputDeclaration("scan", "scan.result"),),
    )
    plane = SignalDataPlane()
    hosts: list = []

    def hosted(node_descriptor, node, instance_id, source_signal, values=None):
        host = make_host(
            node_descriptor,
            node,
            signal_plane=plane,
            instance_id=instance_id,
            source_signal=source_signal,
            values={} if values is None else values,
            devices={},
        )
        hosts.append(host)
        return host

    try:
        reading = hosted(
            expression,
            SimpleNamespace(dataset_input_siblings=("occupied", "frame_judged")),
            "derive-1",
            "@logic/occupancy/counts",
            {"name": "bright"},
        )
        assert reading._input_name == "a"
        assert reading._input_siblings == ("occupied", "frame_judged")
        assert [item.name for item in reading.dataset_output_declarations] == ["bright"]
        silent = hosted(
            expression, object(), "derive-2", "@logic/occupancy/counts", {"name": "value"}
        )
        assert silent._input_siblings == ()

        explicit = hosted(processor, object(), "processor-7", "@logic/camera-2/frames")
        assert explicit.instance_id == "processor-7"
        assert explicit.dataset_output_declarations == (output,)
        assert explicit._source_signal == "@logic/camera-2/frames"
        assert explicit._input_delivery == "exact"

        scan_host = hosted(scan, object(), "scan-3", "@logic/camera-2/frames")
        assert scan_host._mode == "worker"
        assert scan_host._source_signal == "@logic/camera-2/frames"
        assert scan_host._input_delivery == "exact"
    finally:
        try:
            for host in reversed(hosts):
                host.shutdown()
        finally:
            plane.close()


def test_missing_explicit_artifact_path_fails_start_and_keeps_the_draft(
    presenter, session, tmp_path
) -> None:
    camera, _snapshot = _one_shot(session)
    missing = tmp_path / "missing-calibration.json"
    node_id = presenter.add_logic(
        "occupancy",
        source_signal=camera.signal_key("frames"),
        artifact_inputs={"calibrations": [{"path": str(missing), "frames": ["rest"]}]},
    )

    assert presenter.start_logic(node_id) is False
    assert presenter.logic[node_id].host is None
    assert presenter.logic[node_id].draft.artifact_inputs == {
        "calibrations": ({"path": str(missing), "frames": ("rest",)},)
    }
    assert "missing-calibration.json" in presenter.logic[node_id].draft_error


def test_calibration_pulse_is_a_workspace_file_picker(
    presenter,
) -> None:
    from pulse_fixture import pulse_document

    template = presenter.session.workspace.pulses / "imaging_template.json"
    template.write_bytes(pulse_document("imaging_template.json"))
    invalid = presenter.session.workspace.pulses / "not-calibration.json"
    invalid.write_text(
        "{}", encoding="utf-8"
    )

    node_id = presenter.add_logic("calibration")
    projection = presenter.logic_editor_projection(node_id)
    pulse = next(
        field
        for field in projection["form_spec"].fields
        if field.key == "pulse_template"
    )
    assert pulse.kind == "path"
    assert Path(pulse.base_dir) == presenter.session.workspace.pulses
    assert pulse.file_filter == "Calibration pulse template (*.json)"
    # No pulse is named for the operator: pulses are workspace files, and
    # the one this node runs is the operator's to pick.
    assert projection["form_values"]["pulse_template"] == ""
    assert projection["form_values"]["repeats"] == 200
    assert "timeout_seconds" not in projection["form_values"]
    assert projection["can_start"] is False
    # The row's label and the workspace's folder name, not the field key
    # and an absolute path the row elides mid-word.
    assert "Calibration pulse template must be a file in pulses/" in projection["issues"]

    presenter.update_logic_draft(node_id, values={"pulse_template": template.name})
    projection = presenter.logic_editor_projection(node_id)
    # A file in the pulse folder is shown by its name there -- the part
    # that tells pulses apart -- whether the draft wrote it so or whole;
    # the field's tooltip names the whole path (its base_dir is that folder).
    assert projection["form_values"]["pulse_template"] == template.name
    assert projection["can_start"] is True
    presenter.update_logic_draft(node_id, values={"pulse_template": str(template.resolve())})
    projection = presenter.logic_editor_projection(node_id)
    assert projection["form_values"]["pulse_template"] == template.name
    assert presenter.logic[node_id].draft.values["pulse_template"] == str(template.resolve()), (
        "showing it shorter does not rewrite the draft")
    assert projection["can_start"] is True
    # A file anywhere else is shown whole, and refused by the folder's name.
    outside = presenter.session.workspace.root / "imaging_template.json"
    outside.write_bytes(pulse_document("imaging_template.json"))
    presenter.update_logic_draft(node_id, values={"pulse_template": str(outside)})
    projection = presenter.logic_editor_projection(node_id)
    assert projection["form_values"]["pulse_template"] == str(outside.resolve())
    assert "Calibration pulse template must be a file in pulses/" in projection["issues"]

    presenter.update_logic_draft(
        node_id,
        values={"pulse_template": str(invalid)},
    )
    projection = presenter.logic_editor_projection(node_id)
    assert projection["can_start"] is False
    assert any("not-calibration.json" in issue for issue in projection["issues"])

    # Calibrating saved frames reads no pulse and holds no device: the
    # broken pulse it still names stays in the draft and refuses nothing,
    # no camera or sequencer is offered or claimed, and New frames brings
    # the refusal back with the pulse.
    presenter.update_logic_draft(node_id, values={"frame_source": "saved"})
    projection = presenter.logic_editor_projection(node_id)
    assert projection["can_start"] is True, projection["issues"]
    assert projection["device_options"] == {}
    assert presenter.logic[node_id].draft.values["pulse_template"] == str(invalid)
    presenter.update_logic_draft(node_id, values={"frame_source": "new"})
    projection = presenter.logic_editor_projection(node_id)
    assert any("not-calibration.json" in issue for issue in projection["issues"])


def test_slm_feedback_form_shows_a_numeric_exposure_blank_for_the_current_camera_setting(presenter) -> None:
    node_id = presenter.add_logic("slm_feedback")
    projection = presenter.logic_editor_projection(node_id)
    exposure = next(
        field
        for field in projection["form_spec"].fields
        if field.key == "exposure_seconds"
    )
    pulse = next(
        field
        for field in projection["form_spec"].fields
        if field.key == "pulse_template"
    )
    assert exposure.kind == "float"
    assert exposure.default is None
    assert projection["form_values"]["exposure_seconds"] is None
    assert pulse.kind == "path"
    assert projection["form_values"]["pulse_template"] == ""
    assert projection["can_start"] is False
    presenter.update_logic_draft(node_id, values={"feedback_mode": "ramsey_frequency"})
    external = presenter.logic_editor_projection(node_id)
    assert set(external["device_options"]) == {"slm"}
    assert external["source_required"] and external["acquisition_enabled"]
    assert not any("pulse_template" in issue or "camera.adapter" in issue or "sequencer.streamer" in issue for issue in external["issues"])
    assert external["artifact_form_spec"].keys == projection["artifact_form_spec"].keys
    presenter.update_logic_draft(node_id, values={"feedback_mode": "qcmos_loading_rate"})
    camera = presenter.logic_editor_projection(node_id)
    assert set(camera["device_options"]) == {"camera", "sequencer", "slm"}
    assert not camera["source_required"] and not camera["acquisition_enabled"]



@pytest.mark.parametrize("retire", ("remove", "board", "window"))
def test_artifact_contract_resolves_once_and_passes_exact_typed_value(
    presenter,
    tmp_path,
    retire,
) -> None:
    import json

    from threading import Event, current_thread, main_thread

    from zlc_atom.authoring import AuthoringField, AuthoringSchema
    from zlc_atom.nodes import (
        ArtifactCodec,
        ArtifactInputSpec,
        LogicNodeDescriptor,
        NodeKind,
        ResolvedArtifact,
        WorkspaceResourceSpec,
    )

    builds: list[object] = []
    decodes: list[Path] = []
    runs, closed = [], []
    closing, release = Event(), Event()
    release.set()

    def decode(path: Path) -> object:
        decodes.append(path)
        return json.loads(path.read_text(encoding="utf-8"))

    def build(*, artifact_path, pulse_resource):
        builds.append(artifact_path)
        node = SimpleNamespace(artifact=artifact_path, pulse=pulse_resource,
                               prepared=None, close_failure=False)

        def execute(context):
            if node.prepared is None:
                node.prepared = object()
            runs.append((node.artifact.value, node.pulse.value, node.prepared,
                         context.run_directory))
            context.report_progress("inputs accepted")

        def close():
            assert current_thread() is not main_thread()
            if node.prepared is not None:
                closed.append(node)
                closing.set()
                assert release.wait(2)
                node.prepared = None
                if node.close_failure:
                    raise RuntimeError("discard resource close failed")

        node.execute, node.close = execute, close
        return node

    descriptor = LogicNodeDescriptor(
        "artifact_consumer",
        NodeKind.TASK,
        AuthoringSchema((AuthoringField("pulse_template", "resource", "Pulse", "", required=True),)),
        description="Prepare a saved artifact and Pulse resource for a finite run.",
        input_specs=(
            ArtifactInputSpec(
                "artifact_path",
                "Probe artifact",
                ArtifactCodec(
                    "probe",
                    "Probe artifacts (*.json)",
                    (".json",),
                    decode,
                ),
            ),
        ),
        workspace_resources=(WorkspaceResourceSpec(
            "pulse_template", "probe.pulse", "pulses", (".json",),
            lambda path: json.loads(path.read_text(encoding="utf-8")),
            argument_name="pulse_resource",
        ),),
        node_previews=(),
        build=build,
    )
    presenter.catalog = LogicCatalog((descriptor,))
    selected = tmp_path / "selected.json"
    selected.write_text('{"format":"probe"}', encoding="utf-8")
    pulse = presenter.session.workspace.root / "pulses" / "restart-probe.json"
    pulse.write_text('{"period":1}', encoding="utf-8")
    node_id = presenter.add_logic(
        "artifact_consumer",
        values={"pulse_template": str(pulse)},
        artifact_inputs={"artifact_path": str(selected)},
    )

    projection = presenter.logic_editor_projection(node_id)
    assert projection["can_start"] is True
    artifact_field = projection["artifact_form_spec"].fields[0]
    assert artifact_field.file_filter == "Probe artifacts (*.json)"
    decodes.clear()
    assert presenter.start_logic(node_id) is True
    assert decodes == [selected.resolve()]
    resolved = builds[0]
    assert isinstance(resolved, ResolvedArtifact)
    assert resolved is presenter.logic[node_id].finalization.artifacts[
        "artifact_path"
    ]
    assert resolved.path == selected.resolve()
    assert resolved.value == {"format": "probe"}
    deadline = time.monotonic() + 2
    while presenter.logic[node_id].host.running and time.monotonic() < deadline:
        presenter.poll_logic()
        time.sleep(.001)
    binding = presenter.logic[node_id]
    original_host, original_node = binding.host, binding.node
    assert original_host.observation.phase == "done"
    original_prepared = original_node.prepared
    decodes.clear()
    selected.write_text('{"format":"probe","revision":2}', encoding="utf-8")
    pulse.write_text('{"period":2}', encoding="utf-8")
    assert presenter.start_logic(node_id)
    deadline = time.monotonic() + 2
    while (binding.pending is not None or binding.host.running) and time.monotonic() < deadline:
        presenter.poll_logic()
        time.sleep(.001)
    assert binding.host is not original_host and binding.node is not original_node
    assert decodes == [selected.resolve()]
    assert runs[1][:2] == ({"format": "probe", "revision": 2}, {"period": 2})
    assert runs[1][2] is not original_prepared
    assert runs[1][3] != runs[0][3]
    assert closed == [original_node]

    # Every replacement waits for its old owner to release off the UI
    # thread, rather than requiring another Start.
    second_host, second_node = binding.host, binding.node
    closing.clear()
    selected.write_text('{"format":"probe","optical":1}', encoding="utf-8")
    release.clear()
    assert presenter.start_logic(node_id)
    assert closing.wait(1)
    assert binding.pending is not None and binding.host is second_host
    release.set()
    deadline = time.monotonic() + 2
    while (binding.pending is not None or binding.host.running) and time.monotonic() < deadline:
        presenter.poll_logic()
        time.sleep(.001)
    assert binding.host is not second_host
    assert binding.host.observation.phase == "done"
    assert runs[2][2] is not original_prepared
    assert closed == [original_node, second_node]
    if retire == "remove":
        discarded = presenter._build_logic_candidate(
            binding, presenter._finalize_logic_binding(binding, force=True))
        discarded.node.prepared = object()
        discarded.node.close_failure = True
        closing.clear()
        release.clear()
        presenter._discard_candidate(binding, discarded)
        assert closing.wait(1)
        assert not discarded.host.closed
        release.set()
        deadline = time.monotonic() + 2
        while not discarded.host.closed and time.monotonic() < deadline:
            presenter.beat()
            time.sleep(.001)
        assert discarded.host.closed
        assert discarded.node.prepared is None
        presenter.beat()
        errors = [text for severity, text in presenter.view.status
                  if severity == "error" and "discard resource close failed" in text]
        assert len(errors) == 1
    final_node = binding.node
    closing.clear()
    release.clear()
    close_requests = []
    if retire == "remove":
        assert presenter.remove_logic(node_id) is False
    elif retire == "board":
        assert presenter.clear_board()
    else:
        presenter._request_close = lambda: close_requests.append(True)
        assert presenter.close() is False
        presenter.beat()
    assert closing.wait(1)
    assert node_id in presenter.logic
    release.set()
    deadline = time.monotonic() + 2
    while node_id in presenter.logic and time.monotonic() < deadline:
        presenter.beat() if retire == "window" else presenter.poll_logic()
        time.sleep(.001)
    assert node_id not in presenter.logic
    assert closed == ([original_node, second_node, discarded.node, final_node]
                      if retire == "remove" else [original_node, second_node, final_node])
    if retire == "window":
        assert presenter._closed
        assert close_requests == [True], "resource release needed a second Close click"


def test_reading_in_photoelectrons_is_offered_only_when_the_camera_can(
    presenter,
) -> None:
    """Unavailable is an effective Off, without erasing the operator's draft.

    The default returns when a capable camera is selected again; an explicit
    Off remains Off.  Workbench needs no camera-type branch to do either.
    """

    from zlc_atom.devices.camera.photoelectrons import PHOTOELECTRONS

    node_id = presenter.add_logic("camera_measurement")
    projection = presenter.logic_editor_projection(node_id)
    field = next(
        item for item in projection["form_spec"].fields if item.key == PHOTOELECTRONS
    )
    assert field.kind == "bool" and not field.unavailable
    assert projection["form_values"][PHOTOELECTRONS] is True
    assert presenter.logic_editor_projection(node_id)["can_start"] is True

    presenter.update_logic_draft(
        node_id, device_keys={"camera": "mot_camera"}
    )
    projection = presenter.logic_editor_projection(node_id)
    field = next(
        item for item in projection["form_spec"].fields if item.key == PHOTOELECTRONS
    )
    assert field.unavailable
    assert "states no photoelectron conversion" in field.unavailable_reason
    assert projection["form_values"][PHOTOELECTRONS] is False
    assert projection["can_start"] is True
    assert presenter.logic[node_id].draft.values[PHOTOELECTRONS] is True

    # Whether or not the rest of the draft projects -- a calibration with its
    # required pulse still empty does not -- what the camera cannot do is a
    # fact about the camera.
    calibration = presenter.add_logic(
        "calibration", device_keys={"camera": "mot_camera"}, open_editor=False
    )
    projection = presenter.logic_editor_projection(calibration)
    assert projection["can_start"] is False
    field = next(
        item for item in projection["form_spec"].fields if item.key == PHOTOELECTRONS
    )
    assert field.unavailable
    assert projection["form_values"][PHOTOELECTRONS] is False

    presenter.update_logic_draft(node_id, device_keys={"camera": "camera"})
    projection = presenter.logic_editor_projection(node_id)
    assert projection["form_values"][PHOTOELECTRONS] is True
    assert projection["can_start"] is True

    presenter.update_logic_draft(node_id, values={PHOTOELECTRONS: False})
    presenter.update_logic_draft(
        node_id, device_keys={"camera": "mot_camera"}
    )
    projection = presenter.logic_editor_projection(node_id)
    assert projection["form_values"][PHOTOELECTRONS] is False
    assert projection["can_start"] is True
    presenter.update_logic_draft(node_id, device_keys={"camera": "camera"})
    projection = presenter.logic_editor_projection(node_id)
    assert projection["form_values"][PHOTOELECTRONS] is False
    assert projection["can_start"] is True


def test_an_armed_silent_source_admits_a_scan_draft(bench) -> None:
    """A declared source may not yet have data or a reserved generation.

    A configured panel fit is such a producer before its first real frame;
    admission must not label its parameter incompatible or start its camera.
    Undeclared names and incompatible contracts remain rejected.
    """

    catalog = LogicCatalog()
    descriptor = catalog.get("seamless_scan")
    signal = "@logic/camera/frames"

    def finalized(*, armed: bool, offered=True):
        plane = SimpleNamespace(
            latest_publication=lambda _name: None,
            is_generation_live=lambda _name: armed,
            describe_signals=lambda: (),
        )
        return finalize_logic_draft(
            descriptor,
            LogicDraft(source_signal=signal),
            installation=bench.installation,
            signal_plane=plane,
            workspace=bench.workspace,
            source_options=(signal,) if offered else (),
        )

    # An issue names the source by its row on the page, never by the
    # producer, whose name may change while the issue stands.
    source_issues = [
        text for text in finalized(armed=True).issues if "not declared" in text
    ]
    assert source_issues == []
    waiting_issues = [
        text for text in finalized(armed=False).issues if "not declared" in text
    ]
    assert waiting_issues == []
    refused = [text for text in finalized(armed=False, offered=False).issues
               if "not declared" in text]
    assert refused and not [text for text in refused if "camera" in text], refused

    with _console_over(bench) as presenter:
        panel = presenter.add_blank_panel("curve")
        panel.state = replace(panel.state, fit={"model": "gaussian_offset"})
        parameter = f"@logic/{panel.panel_id}/amplitude"
        assert bench.signal_plane.latest_publication(parameter) is None
        assert not bench.signal_plane.is_generation_live(parameter)
        assert parameter in presenter._source_options(descriptor, "scan")
        from zlc_atom.nodes import DatasetInputSpec

        camera_only = replace(descriptor, input_specs=(DatasetInputSpec("signal", "camera.frames", "exact"),))
        assert parameter not in presenter._source_options(camera_only, "scan")
        panel.state = replace(panel.state, published_outputs={"amplitude": False})
        assert parameter not in presenter._source_options(descriptor, "scan")
        camera_id = presenter.add_logic("camera_measurement")
        scan_id = presenter.add_logic("seamless_scan")
        projection = presenter.logic_editor_projection(scan_id)
        field = next(field for field in projection["form_spec"].fields
                     if field.key == "acquisition_logic")
        assert field.kind == "choice"
        assert {choice.value for choice in field.choices} == {"", camera_id}
        assert "settle_seconds" not in projection["form_values"]
        presenter.update_logic_draft(scan_id, values={"acquisition_logic": scan_id})
        assert any("not an available Measurement" in text for text in
                   presenter.logic_editor_projection(scan_id)["issues"])
        presenter.update_logic_draft(scan_id, values={
            "acquisition_logic": camera_id,
        })
        assert not any("not an available Measurement" in text for text in
                       presenter.logic_editor_projection(scan_id)["issues"])
        saved = presenter.layout()
        assert presenter.apply_layout(saved)
        restored = presenter.logic_editor_projection(scan_id)
        assert restored["form_values"]["acquisition_logic"] == camera_id
        assert "settle_seconds" not in restored["form_values"]


def test_a_renamed_node_is_called_by_its_name_and_addressed_by_its_id(presenter, session) -> None:
    """The operator renames a node; everything a person reads follows, on the
    same turn, and nothing that addresses it moves -- not while it runs.

    One name, kept on the node's binding: its row, its Edit page, every
    picker that offers it, the panels showing its signals, the status lines
    and MCP all read it from there.  Its id stays every signal key, lease,
    reference and run record's address.
    """

    from zlc_workbench.device_use import DeviceClaim, DeviceUseBusy

    camera_id, frames, before_rename = _started_camera(presenter, session)
    assert camera_id == "camera_measurement"
    _settle_panel_hosts(presenter, lambda: any(
        panel.state.signal == frames and panel.accepted_surface is not None
        for panel in presenter.panels.values()
    ))
    panel = next(panel for panel in presenter.panels.values() if panel.state.signal == frames)
    occupancy_id = presenter.add_logic("occupancy", source_signal=frames)
    scan_id = presenter.add_logic("seamless_scan", values={"acquisition_logic": camera_id})
    presenter.edit_logic(camera_id)
    assert presenter.edit_panel(panel.panel_id)
    binding = presenter.logic[camera_id]
    host, generation, revision = binding.host, binding.host.generation, binding.draft_revision
    record, published = panel.state, host.published_signals()
    shown_from = len(presenter.view.status)

    presenter.view.logic_name_committed.emit(camera_id, "  Camera A ")

    # Identity: the same run, the same keys, the same panel record.
    assert binding.name == "Camera A" and binding.host is host and host.running
    assert (host.instance_id, host.generation) == (camera_id, generation)
    assert host.published_signals() == published == (frames,)
    assert panel.state == record and binding.draft_revision == revision
    # What a person reads: the row, every Edit page, the pickers, the panel.
    row = presenter.view._rows[camera_id]
    assert row.title == "Camera A"
    assert row.publishes[0][2].endswith("· Camera A/frames")
    camera_page = presenter.view.logic_editors[camera_id]
    assert (camera_page["name"], camera_page["name_problem"]) == ("Camera A", "")
    occupancy_page = presenter.view.logic_editors[occupancy_id]
    assert occupancy_page["source_signal"] == frames
    assert occupancy_page["source_groups"][frames] == "Camera A"
    # ONE spelling of the signal: the closed Source picker reads what the
    # row's status reads, and so do the panel's pickers.
    assert occupancy_page["source_texts"][frames] == "Camera A/frames"
    assert presenter.view._cards[panel.panel_id].texts[frames] == "Camera A/frames"
    scan_field = next(field for field in presenter.view.logic_editors[scan_id]["form_spec"].fields
                      if field.key == "acquisition_logic")
    assert ("Camera A", camera_id) in {(choice.label, choice.value) for choice in scan_field.choices}
    surface = presenter.view.panel_parameter_surfaces[panel.panel_id]
    assert surface["caption"] == f"{panel.panel_id}@Camera A/frames"
    assert surface["panel_name"] == "Camera A/frames"
    headings = {heading for heading, _leaves in presenter.view._cards[panel.panel_id].choices}
    assert "Camera A" in headings and camera_id not in headings
    panel_page = presenter.view.panel_editors[panel.panel_id]
    assert (panel_page["producer_node_id"], panel_page["producer_name"]) == (camera_id, "Camera A")
    assert panel_page["frozen_signal"] == frames
    assert panel_page["frozen_signal_text"] == "Camera A/frames"
    assert panel_page["signal_texts"][frames] == "Camera A/frames"
    state = presenter.control_state()
    assert {(row["node_id"], row["name"]) for row in state["logic"]} >= {(camera_id, "Camera A")}
    assert next(item["name"] for item in state["panels"] if item["panel_id"] == panel.panel_id) == "Camera A/frames"
    assert presenter.view.status[shown_from] == ("task", f"renamed {camera_id} to Camera A")
    # Nowhere a person reads -- after the line that states the rename --
    # still calls it by its id.
    pages = [presenter.view.logic_editors[node] for node in (camera_id, occupancy_id, scan_id)]
    said = [
        *(text for _severity, text in presenter.view.status[shown_from + 1:]),
        *(text for row in presenter.view.logic_rows for text in (row.title, *row.state)),
        *(text for row in presenter.view.logic_rows for item in row.publishes for text in item),
        *(str(page[key]) for page in pages for key in ("name", "status", "issues")),
        *(text for page in pages for text in (
            *page["source_groups"].values(), *page["source_labels"].values(), *page["source_texts"].values())),
        *presenter.view._cards[panel.panel_id].texts.values(), *panel_page["signal_texts"].values(),
        *(choice.label for choice in scan_field.choices),
        surface["caption"], surface["panel_name"], *headings, panel_page["producer_name"],
    ]
    assert not [text for text in said if camera_id in text], said

    # A device refusal names the owner as the operator calls it.
    with pytest.raises(DeviceUseBusy, match="device is in use by Camera A$"):
        session.device_use.acquire_command(
            object(), "probe", (DeviceClaim("camera", "camera", session.camera),)
        )
    # History: a run is stamped with the name it was built under.
    assert before_rename.run_record["node"] == camera_id
    assert before_rename.run_record["node_name"] == camera_id
    assert presenter.start_logic(camera_id)
    deadline = time.monotonic() + 10.0
    latest = None
    while time.monotonic() < deadline:
        session.fire(shots=1)
        presenter.beat()
        latest = session.signal_plane.latest_publication(frames)
        if latest is not None and latest.run_record.get("node_name") == "Camera A":
            break
        time.sleep(0.005)
    assert latest.run_record["node"] == camera_id
    assert latest.run_record["node_name"] == "Camera A"

    # A second camera on the same device waits for the first: by its name.
    second = presenter.add_logic(
        "camera_measurement", values={"exposure_seconds": 0.002, "repeat": 0},
        device_keys={"camera": "camera"}, open_editor=False,
    )
    assert second == "camera_measurement2"
    holder = presenter.logic[camera_id].host
    assert presenter.start_logic(second)
    assert presenter.logic[second].pending is not None
    assert presenter.view._rows[second].state[1] == "waiting for Camera A"
    assert presenter.view.status[-1] == ("task", f"{second} queued while Camera A stops")
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline and holder.running:
        presenter.beat()
        time.sleep(0.005)
    assert not holder.running
    assert presenter.stop_logic(second)
    # An empty name is the internal ID, as the empty Name box's hint says.
    assert presenter.rename_logic(second, "Camera C")
    assert presenter.rename_logic(second, "")
    assert presenter.logic[second].name == second
    assert presenter.view.status[-1] == ("task", f"renamed Camera C to {second}")

    # Refusals from the Edit page: one sentence each, under its Name line
    # and on the strip; the name is unchanged.  Names a person reads alike
    # -- another case, a no-break or doubled space, full-width letters -- are
    # one name.  The current name needs no report and clears the sentence.
    presenter.edit_logic(second)
    for typed, problem in (
        ("a/b", "A name is one line of text without “/”."),
        ("a\u200bb", "A name is one line of text without “/”."),
        ("Panel-2", "“Panel-2” is how a panel is named; choose another name."),
        ("CAMERA A", "Another node is already called “Camera A”."),
        ("Camera\u00a0A", "Another node is already called “Camera A”."),
        ("Camera  A", "Another node is already called “Camera A”."),
        ("\uff23\uff41\uff4d\uff45\uff52\uff41 \uff21", "Another node is already called “Camera A”."),
        ("Camera_Measurement", "“camera_measurement” is another node's internal ID; choose another name."),
        ("Signals", "“Signals” already names a signal producer; choose another name."),
    ):
        presenter.view.logic_name_committed.emit(second, typed)
        assert presenter.logic[second].name == second, typed
        assert presenter.view.logic_editors[second]["name_problem"] == problem
        assert presenter.view.status[-1] == ("warning", f"{second}: {problem}")
    reports = len(presenter.view.status)
    presenter.view.logic_name_committed.emit(second, second)
    assert len(presenter.view.status) == reports
    assert presenter.view.logic_editors[second]["name_problem"] == ""
    refused = presenter.control_action("logic.rename", {"node_id": second, "name": "camera a"})
    assert (refused["accepted"], refused["status"]) == (False, "rejected")
    assert refused["error"] == f"{second}: Another node is already called “Camera A”."
    renamed = presenter.control_action("logic.rename", {"node_id": occupancy_id, "name": "Loading"})
    assert (renamed["accepted"], renamed["status"]) == (True, "renamed")
    assert renamed["state"]["logic"][0]["name"] == "Loading"
    same = presenter.control_action("logic.rename", {"node_id": occupancy_id, "name": "Loading"})
    assert (same["accepted"], same["status"]) == (True, "unchanged"), same

    # Minting skips every name and id; an added node may be named at once.
    assert presenter.rename_logic(second, "occupancy2")
    assert presenter.add_logic("occupancy", open_editor=False) == "occupancy3"
    named = presenter.add_logic("occupancy", name="Counting", open_editor=False)
    assert presenter.logic[named].name == "Counting"
    assert presenter.add_logic("occupancy", name="panel-1", open_editor=False) == ""
    assert presenter.add_logic("occupancy", node_id="a/b", open_editor=False) == ""

    # A node that is gone has no name left: what named it reads its id.
    assert presenter.remove_logic(camera_id)
    surface = presenter.view.panel_parameter_surfaces[panel.panel_id]
    assert surface["caption"] == f"{panel.panel_id}@{camera_id}/frames"


def test_a_board_keeps_its_names_and_refuses_one_no_operator_could_type(
    bench_presenter, tmp_path
) -> None:
    """A saved board writes each node's name beside its id; a file without
    names reads every node by its id, and a name the console would refuse
    makes the whole file one sentence."""

    from zlc_workbench.console_layout import (
        LayoutDocument, LayoutError, read_experiment_recipe, resolve_layout,
    )

    presenter = bench_presenter
    camera_id = presenter.add_logic("camera_measurement", open_editor=False)
    occupancy_id = presenter.add_logic(
        "occupancy", source_signal=stable_signal_key(camera_id, "frames"), open_editor=False,
    )
    assert presenter.rename_logic(camera_id, "Camera A")
    path = tmp_path / "board.json"
    assert presenter.save_layout(str(path))
    tree = json.loads(path.read_text(encoding="utf-8"))
    entry = tree["logic"][0]
    assert list(entry)[:3] == ["node_id", "name", "api_name"]
    assert (entry["node_id"], entry["name"]) == (camera_id, "Camera A")

    assert presenter.rename_logic(camera_id, camera_id)
    assert presenter.load_layout(str(path))
    assert presenter.logic[camera_id].name == "Camera A"
    assert presenter.view._rows[camera_id].title == "Camera A"
    assert presenter.logic_editor_projection(occupancy_id)["source_groups"][
        stable_signal_key(camera_id, "frames")
    ] == "Camera A"

    del entry["name"]
    assert LayoutDocument.from_tree(tree).logic[0].name == camera_id
    tree["logic"][1]["name"] = "Loading"
    for name, problem in (
        ("LOADING", "logic entry 1: Another node is already called “Loading”."),
        ("Occupancy", "logic entry 1: “occupancy” is another node's internal ID; choose another name."),
        ("panel-3", "logic entry 1: “panel-3” is how a panel is named; choose another name."),
        ("a/b", "logic entry 1: A name is one line of text without “/”."),
    ):
        entry["name"] = name
        with pytest.raises(LayoutError) as refused:
            LayoutDocument.from_tree(tree)
        assert str(refused.value) == problem
    assert read_experiment_recipe(Path(presenter.session.workspace.root), "trap_lifetime")
    # A name a notebook's producer heads on this session would merge with it
    # in every picker: that row is called by its internal ID, and the load
    # says so.
    entry["name"] = "Calibration"
    loaded = resolve_layout(
        LayoutDocument.from_tree(tree),
        catalog=presenter.catalog,
        installation=presenter.session.installation,
        panel_ids=tuple(f"panel-{index + 1}" for index in range(len(tree["panels"]))),
        external_outputs=(("@logic/calibration/frames", "camera.frames"),),
    )
    assert loaded.logic[0].name == camera_id
    assert (
        f"logic {camera_id} is called by its internal ID: “Calibration” already "
        "names a signal producer; choose another name."
    ) in loaded.notes
