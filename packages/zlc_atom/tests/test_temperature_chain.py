import zou_lab_control

"""The temperature Task, end to end on the virtual bench.

The physics under test IS the product's purpose.  A real calibration measures
the sites and their thresholds; the temperature Task then takes the camera and
the sequencer for itself, plays the release template with the board advancing
``t_off`` from its own scan table, judges every fired cycle with the same
readout the occupancy node runs, pairs the two probe windows, and reports how
fast the atoms stop coming back.

The virtual world loses atoms while the trap is off, so the direct observable
must fall as ``t_off`` grows.  The test does not compare that curve with a
temperature or escape model: this node owns only the measured survival data.

What this file guards, beyond "it runs":

* the Task OWNS THE CHAIN: an operator supplies two devices and a calibration,
  and nothing else -- no camera node to start first, no signal to hand over,
  no exposure to keep in step by hand;
* the PAIRING is per site and per shot: survival is 1, 0, or "not a fact",
  never a brightness ratio, and its mean projection is the pooled fraction of
  the loaded sites that came back;
* the SITE axis survives into the output, so one dead trap can still be seen;
* the release times reach the dataset as its point axis, with their unit;
* the Task's sealed results are retained: they stay on the plane after the run is
  terminal and the host is shut down, which is what keeps a panel alive;
* the saved artifact's curve is the same curve the dataset published, in
  SECONDS, with no temperature model, fit, or extra crossing conclusion.
"""

import json
from pathlib import Path
import time

import numpy as np
from zlc_data import SITE
from zlc_data.figure_archive import FIGURE_SCHEMA, read_archive
from zlc_pulse import compile_sequence, resolve_api_parameters
from zlc_pulse.schedule import trigger_windows
from zlc_runtime import NodeHost, SignalDataPlane

from zlc_atom.install import create_installation
from zlc_atom.nodes import (
    ResolvedWorkspaceResource,
    discover_logic_nodes,
)
from zlc_atom.nodes.scan import PULSE_PARAM_FAMILY, SCAN_PULSE_CONTRACT, ScanAxis, ScanPlan
from tests.pulse_fixture import IMAGING_PULSE_RESOURCE, pulse_sequence


#: The release times played, in the template's own unit (ms), and how many
#: whole sweeps of them.  Microseconds, because that is where a recapture
#: curve for micro-kelvin atoms in a micron trap actually lives; eight sweeps
#: over thirty-five sites is enough loaded pairs per point that the measured
#: fraction is the physics and not the counting noise.
T_OFF_MS = (0.004, 0.010, 0.016, 0.024)
REPEATS = 8


def _host(descriptor: object, node: object, plane: SignalDataPlane) -> NodeHost:
    return NodeHost(
        node,
        plane,
        instance_id=node.instance_id,
        kind=descriptor.kind.value,
        dataset_output_declarations=descriptor.outputs,
        required_artifacts={
            output.name: output.contract_id
            for output in descriptor.artifact_outputs
        },
        task_name=descriptor.api_name,
    )


def _wait_terminal(host: object, *, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and not host.observation.terminal:
        host.poll()
        time.sleep(0.005)
    observed = host.observation
    assert observed.error is None, observed.error
    assert observed.terminal, "the host never finished"


def test_temperature_template_spaces_twenty_millisecond_exposures() -> None:
    installation = create_installation("virtual")
    try:
        board = installation.device("sequencer").describe()
        authored = pulse_sequence("temperature_template.json")
        for release_ms in T_OFF_MS:
            sequence = resolve_api_parameters(authored, {"t_off": release_ms})
            program = compile_sequence(sequence, board.geometry, board.clock_hz)
            starts = tuple(
                start / program.clock_hz
                for start, _end in trigger_windows(program, "emCCD")
            )
            assert len(starts) == 2
            assert starts[1] - starts[0] >= 0.02
    finally:
        installation.close()


def test_the_temperature_task_publishes_release_recapture_survival(
    tmp_path: Path,
) -> None:
    installation = create_installation("virtual")
    plane = SignalDataPlane()
    descriptors = {value.api_name: value for value in discover_logic_nodes()}
    assert [
        (preview.output.name, preview.plot_kind)
        for preview in descriptors["temperature"].node_previews
    ] == [("survival", "curve")]
    assert dict(descriptors["temperature"].node_previews[0].semantic) == {
        "fate:point:temperature.t_off": "x",
        "fate:cell_data:calibration.site": "reduce",
        "reduction": "mean",
    }
    sequencer = installation.device("sequencer")
    camera = installation.device("camera")
    host = None
    try:
        # --- 1. A real calibration: the site map and thresholds the Task reads,
        #        and the exposure it will drive the camera at.
        calibration_node = descriptors["calibration"].instantiate(
            camera=camera,
            camera_key="camera",
            sequencer=sequencer,
            sequencer_key="sequencer",
            pulse_template=IMAGING_PULSE_RESOURCE.path.name,
            pulse_resource=IMAGING_PULSE_RESOURCE,
            signal_plane=plane,
            repeats=30,
        )
        calibration_host = _host(
            descriptors["calibration"], calibration_node, plane
        )
        calibration_host.start(
            run_root=tmp_path,
            input_summary={"repeats": 30},
        )
        _wait_terminal(calibration_host, timeout=120.0)
        calibration_host.shutdown()
        artifact_path = calibration_node.result.artifact_path
        calibration = descriptors["temperature"].input_specs[0].codec.resolve(
            artifact_path
        )

        # --- 2. The Task, and nothing else.  Two devices, one calibration, one
        #        release plan: no camera node was started, no signal was chosen,
        #        and no exposure was typed anywhere -- the Task takes the one
        #        the calibration's thresholds were measured at.
        sequence = pulse_sequence("temperature_template.json")
        plan = ScanPlan((ScanAxis(PULSE_PARAM_FAMILY + "t_off", T_OFF_MS),))
        task = descriptors["temperature"].instantiate(
            sequencer=sequencer,
            sequencer_key="sequencer",
            camera=camera,
            camera_key="camera",
            signal_plane=plane,
            calibration=calibration,
            pulse_template="temperature_template.json",
            pulse_resource=ResolvedWorkspaceResource(
                Path("temperature_template.json"),
                SCAN_PULSE_CONTRACT,
                sequence,
            ),
            # Authored the way the editor authors it: the plan field is the
            # JSON text that editor writes back into the draft.
            plan=json.dumps(plan.to_tree()),
            repeats=REPEATS,
        )
        assert (
            task._camera.request.exposure_seconds
            == calibration.value.frame_contract.exposure_seconds
        ), "the Task judges frames at the exposure its thresholds were measured at"

        host = _host(descriptors["temperature"], task, plane)
        host.start(
            run_root=tmp_path,
            input_summary={"plan": plan.to_tree(), "repeats": REPEATS},
        )
        deadline = time.monotonic() + 420.0
        while time.monotonic() < deadline and not host.observation.terminal:
            plane.freeze()
            host.poll()
            time.sleep(0.005)
        observed = host.observation
        assert observed.error is None, observed.error
        assert observed.terminal

        survival_signal = host.signal_key("survival")
        survival_value = plane.current_dataset(survival_signal)
        survival_publication = plane.latest_publication(survival_signal)
        assert survival_publication is not None
        survival_event = survival_publication.value(survival_signal)
        assert survival_event is not None
        assert survival_event.run_record["named_devices"] == {
            "sequencer": "sequencer",
            "camera": "camera",
        }
        assert "camera" in survival_event.event_record["device_snapshots"]

        # --- The axes: repeats x release times x sites, and the site axis is
        #     kept so one dead trap is still visible.
        schema = survival_value.block.schema
        from zlc_plot.semantics import composed_spec
        from zlc_workbench.panel_catalog import task_console_fitting_spec

        preview = descriptors["temperature"].node_previews[0]
        composed_spec(
            schema,
            task_console_fitting_spec(schema, preview.plot_kind, ""),
            preview.semantic,
        )
        assert schema.repeat_domain.size == REPEATS
        assert tuple(axis.name for axis in schema.repeat_domain.axes[-2:]) == (
            "scan repeat",
            "run repeat",
        )
        assert tuple(axis.size for axis in schema.repeat_domain.axes[-2:]) == (
            REPEATS,
            1,
        )
        assert schema.point_domain.size == len(T_OFF_MS)
        axis = schema.point_domain.axes[0]
        assert axis.name == "t_off"
        assert axis.unit == "ms"
        assert axis.coordinates == T_OFF_MS
        site_axes = tuple(
            axis for axis in schema.cell_domain.axes if axis.role == SITE
        )
        assert len(site_axes) == 1, (
            "survival must keep the site axis, so a per-site answer stays "
            f"readable; the cell axes are {schema.cell_domain.axes}"
        )
        assert site_axes[0].size == calibration.value.n_sites

        # --- The pairing: one loaded site either came back or it did not, and
        #     a site that held nothing answers nothing.
        survival = np.asarray(survival_value.block.values, dtype=float)
        assert survival.shape == (REPEATS, len(T_OFF_MS), calibration.value.n_sites)
        np.testing.assert_array_equal(
            survival_value.expanded_validity(),
            np.isfinite(survival),
        )
        judged = survival[np.isfinite(survival)]
        assert judged.size, "no site was loaded anywhere in the run"
        assert set(np.unique(judged).tolist()) <= {0.0, 1.0}, (
            "survival is a per-site recapture, not a ratio of brightness: "
            f"{np.unique(judged)[:8].tolist()}"
        )
        assert np.isnan(survival).any(), (
            "an empty trap answers nothing about recapture and must stay NaN"
        )

        # --- The curve is the mean projection of this one Dataset: validity
        #     is the loaded-pair denominator and no second rate history exists.
        loaded = np.count_nonzero(np.isfinite(survival), axis=(0, 2))
        recaptured = np.nansum(survival, axis=(0, 2))
        rate = recaptured / loaded

        # --- The measured observable: survival falls as trap-off grows.  No
        #     escape/temperature model is evaluated or fitted here.
        pooled = np.nanmean(survival, axis=(0, 2))
        assert np.all(np.isfinite(pooled)), (
            f"a release time recaptured nothing: {pooled.tolist()}"
        )
        assert np.all(np.diff(pooled) < 0), (
            f"survival must fall with t_off: {pooled.round(3).tolist()}"
        )

        # --- The artifact: the same curve, in SECONDS, and nothing inferred
        #     beyond the binary recapture observations and their pooled rate.
        result = host.final_result
        assert set(result) == {"artifact_path"}
        saved = Path(result["artifact_path"])
        assert saved.is_file() and saved.parent == host.run_directory / "final"
        payload = json.loads(saved.read_text(encoding="utf-8"))
        assert set(payload) == {"format", "t_off", "run_record"}
        assert payload["t_off"] == {"unit": "ms", "values": list(T_OFF_MS)}
        assert set(payload["run_record"]["device_snapshots"]) == {
            "camera", "sequencer"
        }
        curve = payload["run_record"]["curve"]
        np.testing.assert_allclose(curve["survival_rate"], rate, rtol=0, atol=1e-12)
        assert sum(curve["loaded_pairs"]) == judged.size
        np.testing.assert_allclose(
            curve["t_off_seconds"],
            np.asarray(T_OFF_MS) * 1e-3,
            rtol=1e-12,
        )
        assert "release_time_1e_seconds" not in curve
        assert "release_time_model" not in curve
        assert "temperature_kelvin" not in curve, (
            "nothing on this bench declares the trap's reach, so a kelvin "
            "number here would be the operator's own input squared"
        )
        summary = json.loads(
            (host.run_directory / "summary.json").read_text(encoding="utf-8")
        )
        assert summary["curve"] == curve
        assert summary["points"] == len(T_OFF_MS)
        assert (host.run_directory / "summary.txt").is_file()

        figure_path = host.run_directory / "figures" / "survival.npz"
        preview_path = host.run_directory / "figures" / "survival.png"
        assert figure_path.is_file() and preview_path.is_file()
        info, arrays = read_archive(figure_path)
        assert info["schema"] == FIGURE_SCHEMA
        assert set(
            info["sections"]["source"]["run_record"]["device_snapshots"]
        ) == {"camera", "sequencer"}
        from zlc_plot import PlotKind, read_figure_plot

        figure_data, recipe = read_figure_plot(info, arrays, "data")
        assert recipe["spec"].kind is PlotKind.CURVE
        np.testing.assert_array_equal(
            figure_data.block.values,
            survival_value.block.values,
        )
        registered = {artifact.name: artifact for artifact in host.artifacts}
        assert registered["artifact_path"].role == "final"
        assert registered["temperature_summary"].role == "summary"
        assert registered["temperature_summary_text"].role == "summary"
        assert registered["survival_figure"].contract_id == FIGURE_SCHEMA
        assert registered["survival_preview"].role == "preview"
        assert registered["survival_preview"].contract_id == ""

        # --- A Task's results outlive its run: the panel that watched them is
        #     still holding data after the host is gone.
        host.shutdown()
        host = None
        assert plane.current_dataset(survival_signal) is survival_value, (
            "a Task's results must outlive the run that produced them"
        )
    finally:
        if host is not None:
            host.shutdown()
        plane.close()
        installation.close()
