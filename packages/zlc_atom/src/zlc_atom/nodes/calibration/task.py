"""Finite calibration orchestration over pulse, camera, and readout analysis."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from zlc_data import SITE
from zlc_runtime import DatasetOutputDeclaration, FinalDatasetOutput
from zlc_runtime import SignalPublication

from zlc_atom.data import snapshot_from_array
from zlc_atom.devices.camera.contract import CameraAdapter, CameraFrameRecord
from zlc_atom.nodes._framework.pulse_source import arm_sequencer, ResolvedPulse, resolve_pulse
from zlc_atom.nodes.camera_measurement import CameraMeasurementNode, MeasurementResult
from .calibration import CalibrationResult, FrameContract, calibrate
from zlc_atom.nodes._framework.descriptor import NodeKind, runtime_kind
from zlc_atom.nodes._framework.generation import ProducerRuns
from zlc_atom.nodes._framework.provenance import ProvenanceRecorder


_CALIBRATION_DECLARATION = DatasetOutputDeclaration("calibration", "calibration.readout.v1")
_REPORT_DECLARATION = DatasetOutputDeclaration("report", "calibration.report.v1")


@dataclass(frozen=True)
class CalibrationRunResult:
    """The complete product of one automated calibration task."""

    analysis: CalibrationResult
    capture: MeasurementResult
    reference: MeasurementResult
    short: MeasurementResult
    publication: SignalPublication | None = None
    pulse_name: str = "calibration"

    @property
    def calibration(self):
        return self.analysis.calibration

    @property
    def report(self) -> Mapping[str, Any]:
        return self.analysis.report


@dataclass
class CalibrationTask:
    camera: CameraAdapter
    sequencer: object
    signal_plane: object
    grid_shape: tuple[int, int] = (2, 3)
    method: str = "box"
    roi_radius: int = 1
    reducer: str = "mean"
    repeats: int = 30
    pulse_name: str = "calibration"
    pulse_override: object | None = None
    pulse_search_paths: tuple[str | Path, ...] = (Path.cwd() / "pulses",)
    expected_centers_xy: object | None = None
    timeout: float | None = None
    result: CalibrationRunResult | None = None
    #: One revision line for this producer's publications.
    runs: ProducerRuns = field(default_factory=ProducerRuns)
    #: What apparatus this ran on, for the archive to record.  A task that
    #: commands a camera and a sequencer and then cannot say which ones leaves
    #: a saved figure describing an apparatus that produced none of its data.
    provenance: ProvenanceRecorder = field(default_factory=ProvenanceRecorder)

    #: Derived from the domain layer, never declared twice.
    kind: str = runtime_kind(NodeKind.TASK)

    def __post_init__(self) -> None:
        if not isinstance(self.camera, CameraAdapter):
            raise TypeError("camera must implement CameraAdapter")
        if not callable(getattr(self.sequencer, "load", None)):
            raise TypeError("sequencer must expose load")
        if not callable(getattr(self.sequencer, "fire", None)):
            raise TypeError("sequencer must expose fire")
        if self.signal_plane is None:
            raise TypeError("signal_plane must be supplied by the runtime owner")
        grid = tuple(int(value) for value in self.grid_shape)
        if len(grid) != 2 or any(value <= 0 for value in grid):
            raise ValueError("grid_shape must contain two positive dimensions")
        object.__setattr__(self, "grid_shape", grid)
        repeats = int(self.repeats)
        if repeats <= 0:
            raise ValueError("repeats must be positive")
        object.__setattr__(self, "repeats", repeats)
        name = str(self.pulse_name).strip()
        if not name:
            raise ValueError("pulse_name must be non-empty")
        object.__setattr__(self, "pulse_name", name)
        paths = tuple(Path(value).expanduser().resolve() for value in self.pulse_search_paths)
        if not paths:
            raise ValueError("pulse_search_paths must not be empty")
        object.__setattr__(self, "pulse_search_paths", paths)
        if self.timeout is not None and float(self.timeout) <= 0:
            raise ValueError("timeout must be positive")

    @property
    def dataset_output_declarations(self) -> tuple[DatasetOutputDeclaration, ...]:
        return (_CALIBRATION_DECLARATION, _REPORT_DECLARATION)

    @property
    def instance_id(self) -> str:
        return "calibration"

    def signal_key(self, output_name: str) -> str:
        if output_name not in {"calibration", "report"}:
            raise KeyError(f"unknown calibration output {output_name!r}")
        return f"@logic/calibration/{output_name}"

    def _resolve_pulse(self) -> ResolvedPulse:
        return resolve_pulse(
            self.pulse_name,
            search_paths=self.pulse_search_paths,
            override=self.pulse_override,
        )

    @staticmethod
    def _frame_exposures(pulse: ResolvedPulse) -> tuple[float, float, float]:
        windows = int(pulse.metadata.get("camera_windows", 0))
        if windows != 3:
            raise ValueError(f"calibration pulse {pulse.name!r} must declare exactly three camera windows")
        if pulse.metadata.get("repeat_forever", False):
            raise ValueError(f"pulse {pulse.name!r} is repeat_forever and cannot finish calibration")
        try:
            exposures = tuple(float(value) for value in pulse.metadata["frame_exposures"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("calibration pulse must declare frame_exposures=(long, short, long)") from exc
        if len(exposures) != 3 or any(not np.isfinite(value) or value <= 0 for value in exposures):
            raise ValueError("calibration pulse frame_exposures must contain three positive finite values")
        if not np.isclose(exposures[0], exposures[2], rtol=0.0, atol=1e-12):
            raise ValueError("calibration pulse outer reference exposures must be equal")
        if tuple(pulse.metadata.get("reference_frame_indices", ())) != (0, 2):
            raise ValueError("calibration pulse reference_frame_indices must be (0, 2)")
        if int(pulse.metadata.get("short_frame_index", -1)) != 1:
            raise ValueError("calibration pulse short_frame_index must be 1")
        semantics = tuple(pulse.metadata.get("frame_semantics", ()))
        if semantics != (
            "reference_long_before",
            "short_readout",
            "reference_long_after",
        ):
            raise ValueError("calibration pulse frame_semantics must be long-short-long")
        return exposures[0], exposures[1], exposures[2]

    def _arm_sequencer(self, pulse: ResolvedPulse) -> None:
        """Load the program the way every caller loads one."""

        arm_sequencer(self.sequencer, pulse.program, pulse.metadata)

    def _capture(
        self,
        measurement: CameraMeasurementNode,
        pulse: ResolvedPulse,
        repeats: int,
    ) -> MeasurementResult:
        # The pulse describes its own camera bracket; taking the frame count from
        # anywhere else lets a three-window bracket be collected as one frame per
        # shot, silently discarding two thirds of every measurement.
        frames_per_cycle = int(pulse.metadata["camera_windows"])
        capture = measurement.prepare(repeat=repeats, frames_per_cycle=frames_per_cycle)
        try:
            self._arm_sequencer(pulse)
            for _ in range(repeats):
                self.sequencer.fire()
                # Wait for the shot to finish before starting the next one.  The
                # real board detects its commands on a rising edge and refuses a
                # second fire while the first is still running, so firing in a
                # bare loop takes exactly one shot on hardware while a virtual
                # sequencer that models no firing state accepts them all.
                report = self.sequencer.wait_done(self.timeout)
                # The report is the point of waiting.  A calibration built on a
                # shot that errored or underran is a calibration of nothing,
                # and it used to be indistinguishable from a good one.
                if report is None:
                    raise TimeoutError(
                        "a calibration shot was fired and never reported done"
                        + (
                            ""
                            if self.timeout is None
                            else f" within {float(self.timeout):g}s"
                        )
                    )
                fault = getattr(report, "fault", "")
                if fault:
                    raise RuntimeError(f"calibration shot: {fault}")
            return capture.collect()
        except BaseException:
            try:
                capture.close()
            finally:
                safe = getattr(self.sequencer, "safe", None)
                if callable(safe):
                    safe()
            raise

    @staticmethod
    def _split_capture(capture: MeasurementResult) -> tuple[MeasurementResult, MeasurementResult]:
        reference_cycles: list[tuple[CameraFrameRecord, CameraFrameRecord]] = []
        short_cycles: list[tuple[CameraFrameRecord]] = []
        for cycle in capture.cycles:
            if len(cycle) != 3:
                raise RuntimeError("calibration capture cycle must contain long-short-long frames")
            reference_cycles.append((cycle[0], cycle[2]))
            short_cycles.append((cycle[1],))
        reference = MeasurementResult(tuple(reference_cycles), capture.publication, capture.terminal)
        short = MeasurementResult(tuple(short_cycles), capture.publication, capture.terminal)
        return reference, short

    def _publish(
        self,
        analysis: CalibrationResult,
        *,
        publish_final: object | None = None,
    ) -> SignalPublication | None:
        plane_publish = getattr(self.signal_plane, "publish_final", None)
        latest = getattr(self.signal_plane, "latest_publication", None)
        if not callable(plane_publish) or not callable(latest):
            raise TypeError("signal_plane must implement the final publication contract")
        if publish_final is None:
            # The notebook path owns its own run.  Hosted, the NodeHost has
            # already begun the generation and beginning a second one here would
            # make the publication belong to a run nobody is watching.
            self.runs.begin(self.signal_plane, self)
        thresholds = np.asarray(analysis.calibration.thresholds, dtype="<f8")[None, :]
        site_fidelity = np.asarray(analysis.report["site_fidelity"], dtype="<f8")[None, :]
        outputs = {
            "calibration": FinalDatasetOutput(
                _CALIBRATION_DECLARATION,
                snapshot_from_array(
                    thresholds,
                    producer=self.instance_id,
                    signal="calibration",
                    roles=(SITE,),
                    generation=self.runs.generation,
                    revision=self.runs.next_revision(),
                ),
            ),
            "report": FinalDatasetOutput(
                _REPORT_DECLARATION,
                snapshot_from_array(
                    site_fidelity,
                    producer=self.instance_id,
                    signal="report",
                    roles=(SITE,),
                    generation=self.runs.generation,
                    revision=self.runs.next_revision(),
                ),
            ),
        }
        if publish_final is None:
            plane_publish(self, outputs)
        else:
            publish_final(outputs)
        publication = latest(self.signal_key("calibration"))
        if not isinstance(publication, SignalPublication):
            raise RuntimeError("signal plane did not expose the calibration publication")
        return publication

    def run(self, *, publish_final: object | None = None) -> CalibrationRunResult:
        try:
            pulse = self._resolve_pulse()
            frame_exposures = self._frame_exposures(pulse)
            measurement = CameraMeasurementNode(
                camera=self.camera,
                signal_plane=self.signal_plane,
                producer="calibration_camera",
                timeout=self.timeout,
            )
            capture = self._capture(measurement, pulse, self.repeats)
            reference, short = self._split_capture(capture)
            working_point = self.camera.capture_working_point()
            contract = FrameContract(
                working_point.frame_shape_yx,
                exposure_seconds=frame_exposures[1],
            )
            analysis = calibrate(
                reference.cycles,
                short.frames,
                frame_contract=contract,
                grid_shape=self.grid_shape,
                method=self.method,
                roi_radius=self.roi_radius,
                reducer=self.reducer,
                expected_centers_xy=self.expected_centers_xy,
            )
            publication = self._publish(analysis, publish_final=publish_final)
            self.result = CalibrationRunResult(
                analysis,
                capture,
                reference,
                short,
                publication,
                self.pulse_name,
            )
            # After the result exists, so the record carries the calibration's
            # own fingerprint alongside the devices it was measured on.
            self.provenance.reset()
            self.provenance.capture(self)
            return self.result
        except BaseException:
            safe = getattr(self.sequencer, "safe", None)
            if callable(safe):
                safe()
            raise

    def execute(self, context: object) -> dict[str, object]:
        """Hosted entry point: the same calibration, published through the host.

        A NodeHost has already begun the generation, so this adopts it instead
        of starting a second one, and its publications go through the host's
        context so the host can observe them.  Everything else is the identical
        code the notebook path runs, because a second implementation is how a
        virtual bench and a real one start to disagree.

        Without this the descriptor declared a task the runtime could not
        drive: the console offered Add Logic -> calibration, and starting it
        failed instantly with "finite node must provide execute(ctx)".
        """

        self.runs.adopt(context.generation)
        result = self.run(publish_final=context.publish_final)
        return {
            "sites": len(result.calibration.thresholds),
            "signal": self.signal_key("calibration"),
        }

    @property
    def calibration(self):
        if self.result is None:
            raise RuntimeError("calibration task has not run")
        return self.result.calibration

    @property
    def report(self) -> Mapping[str, Any]:
        if self.result is None:
            raise RuntimeError("calibration task has not run")
        return self.result.report


__all__ = ["CalibrationRunResult", "CalibrationTask"]
