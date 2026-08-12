import zou_lab_control_v2

"""The temperature Task, end to end on the virtual bench.

The physics under test IS the product's purpose.  A real calibration measures
the sites and their thresholds; the temperature Task then takes the camera and
the sequencer for itself, plays the release template with the board advancing
``t_off`` from its own scan table, judges every fired cycle with the same
readout the occupancy node runs, pairs the two probe windows, and reports how
fast the atoms stop coming back.

The virtual world loses atoms the way a trap does -- the fast ones walk out
while the light is off -- so the curve the Task measures must be the curve the
world would predict.  That prediction is read from the world itself, never
written down here.

What this file guards, beyond "it runs":

* the Task OWNS THE CHAIN: an operator supplies two devices and a calibration,
  and nothing else -- no camera node to start first, no signal to hand over,
  no exposure to keep in step by hand;
* the PAIRING is per site and per shot: survival is 1, 0, or "not a fact",
  never a brightness ratio, and the published rate IS the pooled fraction of
  the loaded sites that came back;
* the SITE axis survives into the output, so one dead trap can still be seen;
* the release times reach the dataset as its point axis, with their unit;
* the Task's results are FINAL: they are still on the plane after the run is
  terminal and the host is shut down, which is what keeps a panel alive;
* the saved artifact's curve is the same curve the dataset published, in
  SECONDS, and the release time it reports is read off that curve.
"""

import json
from pathlib import Path
import time

import numpy as np
from zlc_data import SITE
from zlc_pulse import sequence_from_tree
from zlc_runtime import NodeHost, SignalDataPlane

from zlc_atom.install import create_installation
from zlc_atom.nodes import (
    ResolvedWorkspaceResource,
    discover_logic_nodes,
    temperature_pulse_template_bytes,
)
from zlc_atom.nodes.scan import PULSE_PARAM_FAMILY, SCAN_PULSE_CONTRACT, ScanAxis, ScanPlan
from tests.pulse_fixture import IMAGING_PULSE_RESOURCE


#: The release times played, in the template's own unit (ms), and how many
#: whole sweeps of them.  Microseconds, because that is where a recapture
#: curve for micro-kelvin atoms in a micron trap actually lives; eight sweeps
#: over thirty-five sites is enough loaded pairs per point that the measured
#: fraction is the physics and not the counting noise.
T_OFF_MS = (0.004, 0.010, 0.016, 0.024)
REPEATS = 8


def _wait_terminal(host: object, *, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and not host.observation.terminal:
        host.poll()
        time.sleep(0.005)
    observed = host.observation
    assert observed.error is None, observed.error
    assert observed.terminal, "the host never finished"


def test_the_temperature_task_recovers_the_worlds_recapture_curve(
    tmp_path: Path,
) -> None:
    installation = create_installation("virtual")
    plane = SignalDataPlane()
    descriptors = {value.api_name: value for value in discover_logic_nodes()}
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
            pulse_resource=IMAGING_PULSE_RESOURCE,
            artifact_directory=tmp_path,
            repeats=30,
        )
        calibration_host = NodeHost(calibration_node, plane)
        calibration_host.start()
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
        sequence = sequence_from_tree(
            json.loads(temperature_pulse_template_bytes().decode("utf-8"))
        )
        plan = ScanPlan((ScanAxis(PULSE_PARAM_FAMILY + "t_off", T_OFF_MS),))
        task = descriptors["temperature"].instantiate(
            sequencer=sequencer,
            sequencer_key="sequencer",
            camera=camera,
            camera_key="camera",
            signal_plane=plane,
            calibration=calibration,
            pulse_resource=ResolvedWorkspaceResource(
                Path("temperature_template.json"),
                SCAN_PULSE_CONTRACT,
                sequence,
            ),
            artifact_directory=tmp_path,
            # Authored the way the editor authors it: the plan field is the
            # JSON text that editor writes back into the draft.
            plan=json.dumps(plan.to_tree()),
            repeats=REPEATS,
        )
        assert (
            task._camera.request.exposure_seconds
            == calibration.value.frame_contract.exposure_seconds
        ), "the Task judges frames at the exposure its thresholds were measured at"

        host = NodeHost(task, plane)
        host.start()
        deadline = time.monotonic() + 240.0
        while time.monotonic() < deadline and not host.observation.terminal:
            plane.freeze()
            host.poll()
        observed = host.observation
        assert observed.error is None, observed.error
        assert observed.terminal

        rate_signal = host.signal_key("survival_rate")
        survival_value = plane.freeze().value(host.signal_key("survival"))
        rate_value = plane.freeze().value(rate_signal)
        assert survival_value is not None and rate_value is not None, (
            "the Task published no survival"
        )

        # --- The axes: repeats x release times x sites, and the site axis is
        #     kept so one dead trap is still visible.
        schema = survival_value.snapshot.block.schema
        assert schema.repeat_axis.size == REPEATS
        assert schema.point_table.row_count == len(T_OFF_MS)
        column = schema.point_table.columns[0]
        assert column.name == "t_off"
        assert column.unit == "ms"
        assert tuple(column.values) == T_OFF_MS
        site_axes = tuple(
            axis for axis in schema.cell_schema.data_axes if axis.role == SITE
        )
        assert len(site_axes) == 1, (
            "survival must keep the site axis, so a per-site answer stays "
            f"readable; the cell axes are {schema.cell_schema.data_axes}"
        )
        assert site_axes[0].size == calibration.value.n_sites

        # --- The pairing: one loaded site either came back or it did not, and
        #     a site that held nothing answers nothing.
        survival = np.asarray(survival_value.block.values, dtype=float)
        assert survival.shape == (REPEATS, len(T_OFF_MS), calibration.value.n_sites)
        judged = survival[np.isfinite(survival)]
        assert judged.size, "no site was loaded anywhere in the run"
        assert set(np.unique(judged).tolist()) <= {0.0, 1.0}, (
            "survival is a per-site recapture, not a ratio of brightness: "
            f"{np.unique(judged)[:8].tolist()}"
        )
        assert np.isnan(survival).any(), (
            "an empty trap answers nothing about recapture and must stay NaN"
        )

        # --- The rate IS the pooled fraction of the loaded sites, per shot.
        rate = np.asarray(rate_value.block.values, dtype=float).reshape(
            REPEATS, len(T_OFF_MS)
        )
        with np.errstate(invalid="ignore"):
            per_shot = np.nanmean(survival, axis=2)
        np.testing.assert_allclose(rate, per_shot, rtol=0, atol=1e-12)

        # --- The physics: the measured curve is the curve THIS WORLD would
        #     predict for these release times.  The prediction comes from the
        #     world's own model -- an atom leaves because it is fast enough --
        #     so this asserts the whole chain, not a formula copied here.
        pooled = np.nanmean(survival, axis=(0, 2))
        assert np.all(np.isfinite(pooled)), (
            f"a release time recaptured nothing: {pooled.tolist()}"
        )
        assert np.all(np.diff(pooled) < 0), (
            f"survival must fall with t_off: {pooled.round(3).tolist()}"
        )
        planted = np.asarray(
            [
                installation.world.release_survival(value * 1e-3)
                for value in T_OFF_MS
            ],
            dtype=float,
        )
        assert np.all(np.abs(pooled - planted) <= 0.12), (
            f"measured {pooled.round(3).tolist()} against the world's own "
            f"{planted.round(3).tolist()}"
        )

        # --- The artifact: the same curve, in SECONDS, and a release time read
        #     off it rather than derived from a number nobody measured.
        result = host.final_result
        saved = Path(result["artifact_path"])
        assert saved.is_file() and saved.parent == tmp_path
        payload = json.loads(saved.read_text(encoding="utf-8"))
        assert payload["t_off"] == {"unit": "ms", "values": list(T_OFF_MS)}
        curve = payload["run_record"]["curve"]
        np.testing.assert_allclose(curve["survival_rate"], pooled, rtol=0, atol=1e-12)
        assert sum(curve["loaded_pairs"]) == judged.size
        np.testing.assert_allclose(
            curve["t_off_seconds"],
            np.asarray(T_OFF_MS) * 1e-3,
            rtol=1e-12,
        )
        crossing = curve["release_time_1e_seconds"]
        assert crossing == result["release_time_1e_seconds"]
        assert crossing is not None, (
            "this sweep falls past 1/e, so the release time must be readable: "
            f"{curve['survival_rate']}"
        )
        assert min(T_OFF_MS) * 1e-3 <= crossing <= max(T_OFF_MS) * 1e-3
        assert "temperature_kelvin" not in curve, (
            "nothing on this bench declares the trap's reach, so a kelvin "
            "number here would be the operator's own input squared"
        )

        # --- A Task's results outlive its run: the panel that watched them is
        #     still holding data after the host is gone.
        assert rate_value.coverage is None and not rate_value.transient
        host.shutdown()
        host = None
        assert plane.freeze().value(rate_signal) is not None, (
            "a Task's results must outlive the run that produced them"
        )
    finally:
        if host is not None:
            host.shutdown()
        plane.close()
        installation.close()
