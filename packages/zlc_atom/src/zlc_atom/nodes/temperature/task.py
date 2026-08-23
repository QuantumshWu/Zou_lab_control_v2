"""Release-recapture: one scan of the release time, read as survival.

WHAT THE EXPERIMENT IS.  The traps are loaded, the atoms are photographed, the
trap is switched OFF for ``t_off`` and back ON, and they are photographed
again.  An atom that was fast enough to leave the trap's reach while the light
was off is gone from the second picture.  Sweeping ``t_off`` and watching how
fast the atoms stop coming back is how a bench measures how hot they are.

THIS TASK OWNS THE WHOLE CHAIN.  It holds the sequencer and the camera, plays
the release template with the board advancing ``t_off`` from its own scan
table, reads the cycles that fired program triggers, judges every frame with
the calibrated readout, pairs the two probe windows, and publishes survival.
An operator picks two devices and a calibration and presses Start; being asked
to first run a camera node and then hand its signal over was asking them to
wire up a fact this Task already knows.

THE PAIRING IS THIS TASK'S OWN SEMANTICS, AND THAT IS WHY IT LIVES HERE.  The
two probe windows of ONE fired cycle are a pair: the first says which sites
were loaded, the second says which of those survived.  ``occupancy`` judges
one frame at a time and knows nothing about what the frame beside it means --
a survival special case there would be a temperature experiment hiding inside
a general classifier.  So the judging IS ``OccupancyProcessor``, asked cycle
by cycle; only the pairing is written here.

WHAT IT DOES NOT REPORT.  A temperature, fitted lifetime, or crossing derived
from the curve.  Turning survival into any of those needs a physical model
this node does not own.  It publishes only the binary recapture observations
and their pooled fraction against the authored trap-off time.

WHAT IT PUBLISHES.  Survival keeps the SITE axis -- per site, per release
time, per repeat -- because a trap that never recaptures is a fact about that
trap, and averaging it away is how you never find out.  The operator's curve
is the mean projection of this same truth, not a second accumulated Dataset.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
from zlc_data import SCAN_POINT, SITE, AxisId, OwnedSnapshot, PointColumn, PointTable
from zlc_data.figure_archive import FIGURE_SCHEMA
from zlc_durable import atomic_write_text, durable_makedirs, write_readable_json
from zlc_pulse import TIME_UNIT_TO_NS, PulseSequence
from zlc_runtime import (
    DatasetCoverage,
    DatasetOutputDeclaration,
    LiveDatasetOutput,
)

from zlc_atom.data import snapshot_from_array
from zlc_atom.nodes.calibration import ReadoutModelKind, TrapCalibration
from zlc_atom.nodes.calibration.calibration import reads_photoelectrons
from zlc_atom.nodes.camera_measurement.measurement import (
    CameraCycleSource,
    CameraMeasurementNode,
    CameraMeasurementRequest,
)
from zlc_atom.nodes.occupancy import OccupancyProcessor
from zlc_atom.nodes.scan import (
    PULSE_PARAM_FAMILY,
    SCAN_OUTPUT,
    ScanPlan,
    SeamlessScanMeasurement,
    bind_plan,
    scan_ports_for,
)


#: The knob a release-recapture template must offer: the duration of the
#: period the trap is off for.  A template without it is refused by name when
#: the plan is bound, against everything that template does offer.
T_OFF_PARAMETER = "t_off"

#: One release, two pictures.  The pairing is the whole measurement, so a
#: template that opens any other number of probe windows per cycle is refused
#: when the camera is armed against it, rather than paired by guesswork.
PROBE_FRAMES = 2

#: Survival per (repeat, release time, site).  Its validity is the loaded-pair
#: denominator, so a mean projection is exactly the pooled recapture curve.
SURVIVAL_OUTPUT = DatasetOutputDeclaration("survival", "temperature.survival")

#: The saved result: the survival table against the authored trap-off time.
TEMPERATURE_ARTIFACT_CONTRACT = "temperature.release-recapture"

#: How long the pulse stays stopped before the table plays.  Every cycle loads
#: its own atoms, so this is only what the bench needs to reach the state the
#: first cycle starts from -- not a knob of the measurement.
SETTLE_SECONDS = 0.05


def _seconds(values: object, unit: str) -> np.ndarray:
    """Plan values in the port's own time unit, as seconds.

    The pulse package owns what a time unit means; nothing here re-spells it.
    """

    try:
        nanoseconds = float(TIME_UNIT_TO_NS[str(unit)])
    except KeyError:
        raise ValueError(
            f"the release axis is measured in {unit!r}, which is not a time "
            f"unit; it must be one of {tuple(TIME_UNIT_TO_NS)}"
        ) from None
    return np.asarray(values, dtype=float) * nanoseconds * 1e-9


class TemperatureTask:
    """Play the release plan, judge every cycle, publish the survival curve."""

    instance_id = "temperature"

    def __init__(
        self,
        *,
        sequencer: object,
        sequencer_key: str,
        camera: object,
        camera_key: str,
        signal_plane: object,
        sequence: PulseSequence,
        calibration: TrapCalibration,
        calibration_path: str | Path,
        plan: ScanPlan,
        repeats: int,
        exposure_seconds: float | None = None,
        model_kind: ReadoutModelKind | None,
    ) -> None:
        if not isinstance(calibration, TrapCalibration):
            raise TypeError("calibration must be TrapCalibration")
        if not isinstance(sequence, PulseSequence):
            raise TypeError("sequence must be PulseSequence")
        if not isinstance(plan, ScanPlan):
            raise TypeError("plan must be ScanPlan")
        release_port = PULSE_PARAM_FAMILY + T_OFF_PARAMETER
        if len(plan.axes) != 1 or plan.axes[0].port != release_port:
            played = tuple(axis.port for axis in plan.axes)
            raise ValueError(
                "release-recapture sweeps the release time and nothing else; "
                f"this plan sweeps {played}"
            )
        # Binding is where the plan meets the pulse: a template that declares
        # no release, or a release time outside what the board can play, is
        # refused here, by name, before anything is armed.
        ports = bind_plan(plan, scan_ports_for(sequence))
        self._devices = {"camera": str(camera_key), "sequencer": str(sequencer_key)}
        self._calibration = calibration
        self._calibration_path = Path(calibration_path).expanduser().resolve()
        self._model = calibration.select_model(model_kind)
        self._port = ports[0]
        self._t_off = plan.axes[0].values
        self._repeats = int(repeats)
        if self._repeats < 1:
            raise ValueError("repeats must be at least 1")
        # The calibration's exposure is the DEFAULT, because reproducing the
        # condition the thresholds were measured at is what an operator
        # usually wants and typing it again is how it goes wrong.  It is not a
        # rule: a calibration records what it did, and whether another
        # exposure is comparable is physics the operator judges.  A run that
        # differs says so in its record rather than being refused.
        contract = calibration.frame_contract
        if exposure_seconds is None and contract.exposure_seconds is None:
            raise ValueError(
                "this calibration does not record the exposure its thresholds "
                "were measured at, so a release-recapture run must be given one"
            )
        self._exposure_seconds = float(
            contract.exposure_seconds if exposure_seconds is None else exposure_seconds
        )
        self._calibrated_exposure_seconds = (
            None if contract.exposure_seconds is None else float(contract.exposure_seconds)
        )
        self._camera = CameraMeasurementNode(
            camera=camera,  # type: ignore[arg-type]
            request=CameraMeasurementRequest(
                camera_key=camera_key,
                exposure_seconds=self._exposure_seconds,
                roi_xywh=contract.roi_xywh,
                repeat=self._repeats * len(self._t_off),
                frames_per_cycle=PROBE_FRAMES,
                # The calibration's own numbers, not a choice of this task's:
                # its thresholds only apply to frames in the unit they were
                # fitted on, and this node has no reason to differ.
                photoelectrons=reads_photoelectrons(self._calibration),
            ),
            signal_plane=signal_plane,
            producer=self.instance_id,
        )
        # One readout contract for the whole bench: the same classifier the
        # occupancy node runs, asked once per cycle.  What temperature adds is
        # what the two frames of a cycle MEAN to each other.
        self._occupancy = OccupancyProcessor(
            calibration,
            calibration_path=self._calibration_path,
            producer=self.instance_id,
            model_kind=model_kind,
        )
        self._scan = SeamlessScanMeasurement(
            sequencer=sequencer,
            source=CameraCycleSource(self._camera),
            sequence=sequence,
            plan=plan,
            ports=ports,
            repeats=self._repeats,
            shots_per_point=1,
            settle_seconds=SETTLE_SECONDS,
            producer=self.instance_id,
        )
        self._written = 0

    @property
    def dataset_output_declarations(self) -> tuple[DatasetOutputDeclaration, ...]:
        # The frames are published too: they are the evidence the survival
        # numbers were read from, and while the run is going they are the live
        # view of it.
        return (SCAN_OUTPUT, SURVIVAL_OUTPUT)

    def _judge(self, value: object, *, row: int, visit: int) -> dict[str, object]:
        """One landed cycle: which sites held an atom, and which held it still.

        Asked while the cycle is still a cycle.  Once the scan has written it
        into the dataset, its two frames are two rows of a point table beside
        every other release time, and the pair is no longer addressable.

        The answer goes out with that cycle, so the curve an operator started
        this Task to watch grows a point at a time instead of appearing when
        it is already too late to change anything about the run.
        """

        outputs = self._occupancy.evaluate(value)
        occupied = np.asarray(
            outputs["occupied"].snapshot.block.values, dtype=bool
        )[0]
        valid = np.asarray(
            outputs["occupied"].snapshot.expanded_validity(),
            dtype=bool,
        )[0]
        eligible = valid[0] & valid[1] & occupied[0]
        survival = np.where(eligible, occupied[1].astype("<f8"), np.nan)
        self._written += 1
        event = snapshot_from_array(
            survival[None, None, :],
            producer=self.instance_id,
            signal=SURVIVAL_OUTPUT.name,
            roles=(SCAN_POINT, SITE),
            axis_specs={SITE: self._calibration.site_map.site_axis},
            point_columns={SCAN_POINT: self._event_point_column()},
            generation="temperature-event",
            revision=self._written,
            validity=eligible[None, None, :],
        )
        schema = event.block.schema
        canonical = replace(
            schema,
            repeat_axis=replace(
                schema.repeat_axis,
                size=self._repeats,
            ),
            point_table=PointTable(
                len(self._t_off),
                (self._point_column(),),
            ),
        )
        return {
            SURVIVAL_OUTPUT.name: LiveDatasetOutput(
                SURVIVAL_OUTPUT,
                event,
                DatasetCoverage(
                    written_cells=self._written,
                    total_cells=self._repeats * len(self._t_off),
                ),
                canonical_schema=canonical,
                cell_origin=(visit, row),
            ),
        }

    def _curve(self, snapshot: OwnedSnapshot) -> dict[str, object]:
        """The pooled recapture fraction against the release time.

        Pooled, not averaged over per-shot fractions: every loaded site is one
        Bernoulli trial, and a shot that loaded three atoms says less than one
        that loaded thirty.  The Dataset validity is exactly that denominator,
        so this artifact calculation and the panel's mean projection read the
        same committed truth.
        """

        values = np.asarray(snapshot.block.values, dtype=float)
        valid = np.asarray(snapshot.expanded_validity(), dtype=bool)
        loaded = np.sum(valid, axis=(0, 2), dtype=float)
        recaptured = np.sum(np.where(valid, values, 0.0), axis=(0, 2))
        fraction = np.divide(
            recaptured,
            loaded,
            out=np.full(loaded.shape, np.nan, dtype="<f8"),
            where=loaded > 0,
        )
        seconds = _seconds(self._t_off, self._port.unit)
        return {
            "t_off_seconds": [float(value) for value in seconds],
            "loaded_pairs": [int(value) for value in loaded],
            "recaptured_pairs": [int(value) for value in recaptured],
            "survival_rate": [
                float(value) if np.isfinite(value) else None for value in fraction
            ],
            "atom": "Rb-87",
        }

    @staticmethod
    def _event_point_column() -> PointColumn:
        return PointColumn(
            AxisId("temperature.t_off"),
            T_OFF_PARAMETER,
            SCAN_POINT,
            PointColumn.NUMERIC,
            (0.0,),
        )

    def _point_column(self) -> PointColumn:
        """The release times, as the point axis every output shares."""

        return PointColumn(
            AxisId("temperature.t_off"),
            T_OFF_PARAMETER,
            SCAN_POINT,
            PointColumn.NUMERIC,
            tuple(float(value) for value in self._t_off),
            unit=self._port.unit or None,
        )

    def _run_record(self, curve: dict[str, object]) -> dict[str, object]:
        return {
            "node": self.instance_id,
            "parameters": {
                "calibration_path": str(self._calibration_path),
                "model_kind": self._model.kind.value,
                "probe_frames": PROBE_FRAMES,
                # Both numbers, always: an archive that records only what was
                # used cannot answer "were these thresholds measured under
                # this condition?", and that question is the whole reason the
                # exposure may now differ.
                "exposure_seconds": self._exposure_seconds,
                "calibrated_exposure_seconds": self._calibrated_exposure_seconds,
                "exposure_matches_calibration": (
                    self._calibrated_exposure_seconds is not None
                    and abs(self._exposure_seconds - self._calibrated_exposure_seconds)
                    <= 1e-12
                ),
            },
            "named_devices": dict(self._devices),
            "scan": self._scan.run_record(),
            "camera": self._camera.run_record,
            "curve": dict(curve),
        }

    def execute(self, context: object) -> dict[str, object]:
        self._written = 0
        self._scan.acquire(context, on_point=self._judge)
        context.report_progress("Reading survival")
        survival = context.current_dataset(SURVIVAL_OUTPUT.name)
        curve = self._curve(survival)
        record = self._run_record(curve)
        context.report_progress("Saving survival")
        if context.cancel_requested():
            raise RuntimeError("the temperature task was cancelled")
        context.seal_terminal()
        artifact = {
            "format": TEMPERATURE_ARTIFACT_CONTRACT,
            "t_off": {
                "unit": self._port.unit,
                "values": [float(value) for value in self._t_off],
            },
            "run_record": record,
        }
        final_root = context.run_directory / "final"
        durable_makedirs(final_root)
        artifact_path = write_readable_json(
            final_root / "temperature.json",
            artifact,
        )
        context.register_artifact(
            "artifact_path",
            artifact_path,
            role="final",
            contract_id=TEMPERATURE_ARTIFACT_CONTRACT,
        )
        summary = {
            "format": "zlc.temperature.summary",
            "exposure_seconds": self._exposure_seconds,
            "points": len(self._t_off),
            "repeats": self._repeats,
            "curve": curve,
        }
        summary_path = write_readable_json(
            context.run_directory / "summary.json", summary
        )
        summary_text = [
            "Release-recapture survival",
            f"Exposure seconds: {self._exposure_seconds}",
            f"Repeats: {self._repeats}",
        ]
        summary_text.extend(
            f"{time_value} s: {rate} ({loaded} loaded pairs)"
            for time_value, rate, loaded in zip(
                curve["t_off_seconds"],
                curve["survival_rate"],
                curve["loaded_pairs"],
                strict=True,
            )
        )
        summary_text_path = atomic_write_text(
            context.run_directory / "summary.txt",
            "\n".join(summary_text) + "\n",
        )
        context.register_artifact(
            "temperature_summary", summary_path, role="summary"
        )
        context.register_artifact(
            "temperature_summary_text", summary_text_path, role="summary"
        )

        from zlc_plot import AxisRef, CurvePlot, PlotLabels, save_figure_artifact

        figure_base = context.run_directory / "figures" / "survival"
        try:
            preview_path, figure_path = save_figure_artifact(
                figure_base,
                plot_input=survival,
                spec=CurvePlot(
                    AxisRef.point("temperature.t_off"),
                    labels=PlotLabels(
                        title="Release-recapture survival",
                        x="Trap-off time",
                        y="Survival",
                    ),
                ),
                parameters={},
                size="4x4",
                source={"task": self.instance_id, "signal": SURVIVAL_OUTPUT.name},
            )
        except BaseException:
            figure_path = figure_base.with_suffix(".npz")
            if figure_path.is_file():
                context.register_artifact(
                    "survival_figure",
                    figure_path,
                    role="figure",
                    contract_id=FIGURE_SCHEMA,
                )
            raise
        context.register_artifact(
            "survival_figure",
            figure_path,
            role="figure",
            contract_id=FIGURE_SCHEMA,
        )
        context.register_artifact(
            "survival_preview",
            preview_path,
            role="preview",
        )
        return {"artifact_path": artifact_path}


__all__ = [
    "PROBE_FRAMES",
    "SURVIVAL_OUTPUT",
    "TEMPERATURE_ARTIFACT_CONTRACT",
    "T_OFF_PARAMETER",
    "TemperatureTask",
]
