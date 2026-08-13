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

WHAT IT DOES NOT REPORT.  A temperature.  Turning this curve into kelvin needs
the trap's own reach and depth -- how far an atom may drift and still be
recaptured -- and nothing on this bench declares them.  A number derived from
an operator typing a radius into this form would be that radius squared, not a
measurement, so what is published is the recapture curve and the release time
at which it has fallen to 1/e: read off the measured points, no model assumed.

WHAT IT PUBLISHES.  Survival keeps the SITE axis -- per site, per release
time, per repeat -- because a trap that never recaptures is a fact about that
trap, and averaging it away is how you never find out.  The rate beside it is
the per-point recapture fraction an operator watches the curve in.  All of it
is FINAL: this Task's results outlive its run.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from zlc_data import SCAN_POINT, SITE, AxisId, PointColumn
from zlc_durable import unique_path, write_readable_json
from zlc_pulse import TIME_UNIT_TO_NS, PulseSequence
from zlc_runtime import (
    DatasetCoverage,
    DatasetOutputDeclaration,
    FinalDatasetOutput,
    LiveDatasetOutput,
)

from zlc_atom.data import snapshot_from_array
from zlc_atom.nodes.calibration import ReadoutModelKind, TrapCalibration
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

#: Survival per (repeat, release time, site), and the recapture fraction per
#: (repeat, release time).  Both keep the repeat axis, so the panel averages
#: over repeats the same way it does for every other measurement.
SURVIVAL_OUTPUT = DatasetOutputDeclaration("survival", "temperature.survival.v1")
SURVIVAL_RATE_OUTPUT = DatasetOutputDeclaration(
    "survival_rate", "temperature.survival-rate.v1"
)

#: The saved result: the survival table and the release time it decays over.
TEMPERATURE_ARTIFACT_CONTRACT = "temperature.release-recapture.v1"

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


def _one_over_e_crossing(seconds: np.ndarray, fraction: np.ndarray) -> float | None:
    """The release time where the recapture fraction has fallen to 1/e.

    Read off the measured curve by straight interpolation between the two
    points that bracket the crossing.  No decay model: ballistic escape is
    not an exponential, and fitting one would report a lifetime this
    experiment does not have.  ``None`` when the sweep never gets there --
    a run that came back with a curve is still a run.
    """

    order = np.argsort(seconds)
    x = np.asarray(seconds, dtype=float)[order]
    y = np.asarray(fraction, dtype=float)[order]
    keep = np.isfinite(y)
    x, y = x[keep], y[keep]
    if x.size < 2 or not y[0] > 0.0:
        return None
    target = y[0] / math.e
    below = np.flatnonzero(y <= target)
    if not below.size:
        return None
    index = int(below[0])
    if index == 0:
        return float(x[0])
    x0, x1 = float(x[index - 1]), float(x[index])
    y0, y1 = float(y[index - 1]), float(y[index])
    if y0 == y1:
        return x1
    return x0 + (y0 - target) * (x1 - x0) / (y0 - y1)


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
        model_kind: ReadoutModelKind | None,
        camera_trigger_port: str,
        artifact_directory: str | Path,
    ) -> None:
        if not isinstance(calibration, TrapCalibration):
            raise TypeError("calibration must be TrapCalibration")
        if not isinstance(sequence, PulseSequence):
            raise TypeError("sequence must be PulseSequence")
        if not isinstance(plan, ScanPlan):
            raise TypeError("plan must be ScanPlan")
        directory = Path(artifact_directory).expanduser().resolve()
        if not directory.is_dir():
            raise ValueError("artifact_directory must be an existing directory")
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
        self._artifact_directory = directory
        self._port = ports[0]
        self._t_off = plan.axes[0].values
        self._repeats = int(repeats)
        port = str(camera_trigger_port).strip()
        if not port:
            raise ValueError("camera_trigger_port must be non-empty")
        self._camera_trigger_port = port
        if self._repeats < 1:
            raise ValueError("repeats must be at least 1")
        # The readout exposure is the one the CALIBRATION measured its
        # thresholds at: judging frames taken at any other exposure against
        # them is judging by numbers that mean something else.  So the camera
        # this Task drives is configured from the artifact, and there is no
        # exposure on its form to disagree with.
        contract = calibration.frame_contract
        if contract.exposure_seconds is None:
            raise ValueError(
                "this calibration does not record the exposure its thresholds "
                "were measured at, so a release-recapture run cannot reproduce it"
            )
        self._camera = CameraMeasurementNode(
            camera=camera,  # type: ignore[arg-type]
            request=CameraMeasurementRequest(
                camera_key=camera_key,
                exposure_seconds=float(contract.exposure_seconds),
                roi_xywh=contract.roi_xywh,
                repeat=self._repeats * len(self._t_off),
                frames_per_cycle=PROBE_FRAMES,
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
            source=CameraCycleSource(
                self._camera, trigger_channel=self._camera_trigger_port
            ),
            sequence=sequence,
            plan=plan,
            ports=ports,
            repeats=self._repeats,
            shots_per_point=1,
            settle_seconds=SETTLE_SECONDS,
            producer=self.instance_id,
        )
        self._before = np.zeros(
            (self._repeats, len(self._t_off), calibration.n_sites), dtype=bool
        )
        self._after = np.zeros_like(self._before)
        #: Which (repeat, release time) cells have been judged, so a growing
        #: publication can say how much of itself is measured.
        self._seen = np.zeros(self._before.shape[:2], dtype=bool)
        self._generation = ""

    @property
    def dataset_output_declarations(self) -> tuple[DatasetOutputDeclaration, ...]:
        # The frames are published too: they are the evidence the survival
        # numbers were read from, and while the run is going they are the live
        # view of it.
        return (SCAN_OUTPUT, SURVIVAL_OUTPUT, SURVIVAL_RATE_OUTPUT)

    def _judge(self, value: object, *, row: int, visit: int) -> dict[str, object]:
        """One landed cycle: which sites held an atom, and which held it still.

        Asked while the cycle is still a cycle.  Once the scan has written it
        into the dataset, its two frames are two rows of a point table beside
        every other release time, and the pair is no longer addressable.

        The answer goes out with that cycle, so the curve an operator started
        this Task to watch grows a point at a time instead of appearing when
        it is already too late to change anything about the run.
        """

        revision = visit * len(self._t_off) + row + 1
        result = self._occupancy.process(
            value.snapshot,
            generation=self._generation,
            revision=revision,
        )
        occupied = np.asarray(result.occupied, dtype=bool)[0]
        self._before[visit, row] = occupied[0]
        self._after[visit, row] = occupied[1]
        self._seen[visit, row] = True
        return {
            SURVIVAL_OUTPUT.name: LiveDatasetOutput(
                SURVIVAL_OUTPUT,
                self._survival_snapshot(self._survival(), revision),
                DatasetCoverage(
                    written_cells=int(self._seen.sum()),
                    total_cells=int(self._seen.size),
                ),
            ),
            SURVIVAL_RATE_OUTPUT.name: LiveDatasetOutput(
                SURVIVAL_RATE_OUTPUT,
                self._rate_snapshot(self._pooled()[2], revision),
                DatasetCoverage(
                    written_cells=int(np.count_nonzero(self._seen.any(axis=0))),
                    total_cells=int(len(self._t_off)),
                ),
            ),
        }

    def _survival(self) -> np.ndarray:
        """Survival per site, per release time, per repeat.

        A site that held no atom before the release answers nothing about
        recapture, so its survival is NaN -- not zero, which would be a
        measured loss that never happened.
        """

        return np.where(self._before, self._after.astype("<f8"), np.nan)

    def _pooled(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Loaded pairs, recaptured pairs, and their fraction per release time.

        One computation, whether it is going on screen half-finished or into
        the artifact at the end.
        """

        loaded = np.sum(self._before, axis=(0, 2), dtype=float)
        recaptured = np.sum(self._before & self._after, axis=(0, 2), dtype=float)
        fraction = np.divide(
            recaptured,
            loaded,
            out=np.full(loaded.shape, np.nan, dtype="<f8"),
            where=loaded > 0,
        )
        return loaded, recaptured, fraction

    def _survival_snapshot(self, survival: np.ndarray, revision: int):
        return snapshot_from_array(
            survival,
            producer=self.instance_id,
            signal=SURVIVAL_OUTPUT.name,
            roles=(SCAN_POINT, SITE),
            axis_specs={SITE: self._calibration.site_map.site_axis},
            point_columns={SCAN_POINT: self._point_column()},
            generation=self._generation,
            revision=int(revision),
        )

    def _rate_snapshot(self, fraction: np.ndarray, revision: int):
        # One repeat, because pooling IS across the repeats: there is nothing
        # left for a panel to average, so what it draws cannot disagree with
        # what was saved.
        return snapshot_from_array(
            np.asarray(fraction, dtype="<f8")[None, :],
            producer=self.instance_id,
            signal=SURVIVAL_RATE_OUTPUT.name,
            roles=(SCAN_POINT,),
            point_columns={SCAN_POINT: self._point_column()},
            generation=self._generation,
            revision=int(revision),
        )

    def _curve(self) -> dict[str, object]:
        """The pooled recapture fraction against the release time.

        Pooled, not averaged over per-shot fractions: every loaded site is one
        Bernoulli trial, and a shot that loaded three atoms says less than one
        that loaded thirty.  This IS the published rate -- the signal reads
        the list this record carries -- because two ways to average the same
        run produce two different curves under one name, and the operator was
        watching one while the artifact kept the other.
        """

        loaded, recaptured, fraction = self._pooled()
        seconds = _seconds(self._t_off, self._port.unit)
        crossing = _one_over_e_crossing(seconds, fraction)
        return {
            "t_off_seconds": [float(value) for value in seconds],
            "loaded_pairs": [int(value) for value in loaded],
            "recaptured_pairs": [int(value) for value in recaptured],
            "survival_rate": [float(value) for value in fraction],
            "release_time_1e_seconds": crossing,
            "release_time_model": (
                "read off the measured curve: the release time where the "
                "recapture fraction has fallen to 1/e of its shortest-release "
                "value, interpolated between the two points that bracket it"
            ),
            "atom": "Rb-87",
        }

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
            },
            "named_devices": dict(self._devices),
            "scan": self._scan.run_record(),
            "camera": self._camera.run_record,
            "curve": dict(curve),
        }

    def execute(self, context: object) -> dict[str, object]:
        generation = str(getattr(context.generation, "value", context.generation))
        self._generation = generation
        frames = self._scan.acquire(context, on_point=self._judge)
        context.report_progress("Reading survival")
        survival = self._survival()
        curve = self._curve()
        record = self._run_record(curve)
        # The last publication of a run that has been publishing all along,
        # so its revision continues that count rather than restarting it.
        revision = self._repeats * len(self._t_off) + 1
        outputs = {
            SCAN_OUTPUT.name: FinalDatasetOutput(SCAN_OUTPUT, frames, record),
            SURVIVAL_OUTPUT.name: FinalDatasetOutput(
                SURVIVAL_OUTPUT,
                self._survival_snapshot(survival, revision),
                record,
            ),
            SURVIVAL_RATE_OUTPUT.name: FinalDatasetOutput(
                SURVIVAL_RATE_OUTPUT,
                self._rate_snapshot(self._pooled()[2], revision),
                record,
            ),
        }
        artifact_path = unique_path(self._artifact_directory, "temperature", ".json")
        context.report_progress("Saving temperature")
        write_readable_json(
            artifact_path,
            {
                "format": TEMPERATURE_ARTIFACT_CONTRACT,
                "t_off": {
                    "unit": self._port.unit,
                    "values": [float(value) for value in self._t_off],
                },
                "run_record": record,
            },
        )
        context.publish_final(outputs)
        return {
            "artifact_path": artifact_path,
            "release_time_1e_seconds": curve["release_time_1e_seconds"],
        }


__all__ = [
    "PROBE_FRAMES",
    "SURVIVAL_OUTPUT",
    "SURVIVAL_RATE_OUTPUT",
    "TEMPERATURE_ARTIFACT_CONTRACT",
    "T_OFF_PARAMETER",
    "TemperatureTask",
]
