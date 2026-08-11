"""Release-recapture: one scan of the release time, read as survival.

WHAT THE EXPERIMENT IS.  The traps are loaded, the atoms are photographed, the
trap is switched OFF for ``t_off`` and back ON, and they are photographed
again.  An atom that drifted further than the trap can recapture is gone from
the second picture.  Sweeping ``t_off`` and watching how fast the atoms stop
coming back is how a bench measures how hot they are.

THE PAIRING IS THIS TASK'S OWN SEMANTICS, AND THAT IS WHY IT LIVES HERE.  The
two probe windows of ONE fired cycle are a pair: the first says which sites
were loaded, the second says which of those survived.  ``occupancy`` judges
one frame at a time and knows nothing about what the frame beside it means --
a survival special case there would be a temperature experiment hiding inside
a general classifier.  What temperature borrows is the readout CONTRACT
(``TrapCalibration.detect``), not a second opinion about frames.

THE SWEEP IS A SCAN, SO IT IS THE SCAN ENGINE.  ``t_off`` is a period duration
the board can advance from its own scan table, and the site camera is
triggered by the same fired program, so the frames arrive in played order with
no host in the loop: exactly the board-advanced (seamless) engine, run whole
from ``nodes/scan``.  This Task adds no loop of its own; it says what the
frames MEAN once they have landed.

WHAT IT PUBLISHES.  Survival keeps the SITE axis -- per site, per release
time, per repeat -- because a trap that never recaptures is a fact about that
trap, and averaging it away is how you never find out.  The rate beside it is
the per-point recapture fraction an operator watches the curve in.  All of it
is FINAL: this Task's results outlive its run.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from zlc_data import SCAN_POINT, SITE, AxisId, PointColumn
from zlc_durable import unique_path, write_readable_json
from zlc_pulse import TIME_UNIT_TO_NS, PulseSequence
from zlc_runtime import DatasetOutputDeclaration, FinalDatasetOutput

from zlc_atom.data import snapshot_from_array
from zlc_atom.nodes.calibration import ReadoutModelKind, TrapCalibration
from zlc_atom.nodes.scan import (
    PULSE_PARAM_FAMILY,
    SCAN_OUTPUT,
    ScanAxis,
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
#: source that publishes any other number of frames per cycle is refused
#: rather than paired by guesswork.
PROBE_FRAMES = 2

#: Survival per (repeat, release time, site), and the recapture fraction per
#: (repeat, release time).  Both keep the repeat axis, so the panel averages
#: over repeats the same way it does for every other measurement.
SURVIVAL_OUTPUT = DatasetOutputDeclaration("survival", "temperature.survival.v1")
SURVIVAL_RATE_OUTPUT = DatasetOutputDeclaration(
    "survival_rate", "temperature.survival-rate.v1"
)

#: The saved result: the survival table, the decay it was fitted to, and the
#: temperature that decay implies.
TEMPERATURE_ARTIFACT_CONTRACT = "temperature.release-recapture.v1"

#: 87-Rb, the atom this bench traps: 86.909 180 5 u.
RB87_MASS_KG = 1.443160648e-25
BOLTZMANN_J_PER_K = 1.380649e-23

#: How long the pulse stays stopped before the table plays.  Every cycle loads
#: its own atoms, so this is only what the bench needs to reach the state the
#: first cycle starts from -- not a knob of the measurement.
SETTLE_SECONDS = 0.05


def _seconds(values: Sequence[float], unit: str) -> np.ndarray:
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
    """Scan the release time, pair the probes, fit the decay, save the result."""

    instance_id = "temperature"

    def __init__(
        self,
        *,
        sequencer: object,
        signal_plane: object,
        signal_name: str,
        source_generation: object,
        sequence: PulseSequence,
        calibration: TrapCalibration,
        calibration_path: str | Path,
        t_off_values: Sequence[float],
        repeats: int,
        model_kind: ReadoutModelKind | None,
        capture_radius_m: float,
        artifact_directory: str | Path,
    ) -> None:
        if not isinstance(calibration, TrapCalibration):
            raise TypeError("calibration must be TrapCalibration")
        if not isinstance(sequence, PulseSequence):
            raise TypeError("sequence must be PulseSequence")
        directory = Path(artifact_directory).expanduser().resolve()
        if not directory.is_dir():
            raise ValueError("artifact_directory must be an existing directory")
        radius = float(capture_radius_m)
        if not np.isfinite(radius) or radius <= 0:
            raise ValueError("capture_radius_m must be positive and finite")
        plan = ScanPlan(
            (ScanAxis(PULSE_PARAM_FAMILY + T_OFF_PARAMETER, tuple(t_off_values)),)
        )
        # Binding is where the plan meets the pulse: a template that declares
        # no release, or a release time outside what the board can play, is
        # refused here, by name, before anything is armed.
        ports = bind_plan(plan, scan_ports_for(sequence))
        self._calibration = calibration
        self._calibration_path = Path(calibration_path).expanduser().resolve()
        self._model = calibration.select_model(model_kind)
        self._artifact_directory = directory
        self._capture_radius_m = radius
        self._port = ports[0]
        self._t_off = plan.axes[0].values
        self._scan = SeamlessScanMeasurement(
            sequencer=sequencer,
            signal_plane=signal_plane,
            signal_name=signal_name,
            source_generation=source_generation,
            sequence=sequence,
            plan=plan,
            ports=ports,
            repeats=int(repeats),
            shots_per_point=1,
            settle_seconds=SETTLE_SECONDS,
            producer=self.instance_id,
        )

    @property
    def dataset_output_declarations(self) -> tuple[DatasetOutputDeclaration, ...]:
        # The frames are published too: they are the evidence the survival
        # numbers were read from, and while the run is going they are the live
        # view of it.
        return (SCAN_OUTPUT, SURVIVAL_OUTPUT, SURVIVAL_RATE_OUTPUT)

    def _cycles(self, frames: object) -> np.ndarray:
        """The scan dataset as (repeat, release time, probe, y, x).

        The scan writes one publication per played point, and a publication is
        one camera CYCLE whose frames are its point rows, so the dataset's
        points are (release time) x (probe window) in that order.  Anything
        else is a source that cannot be paired.
        """

        values = np.asarray(frames.block.values)
        repeats, points = values.shape[0], values.shape[1]
        releases = len(self._t_off)
        per_point, remainder = divmod(points, releases)
        if remainder or per_point != PROBE_FRAMES:
            raise ValueError(
                "release-recapture pairs the two probe windows of one cycle, "
                f"and this source published {points / releases:g} frames per "
                "release time"
            )
        return values.reshape(repeats, releases, PROBE_FRAMES, *values.shape[2:])

    def _occupancy(self, cycles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Which sites held an atom before the release, and which after.

        One readout contract, asked once per frame: the calibration that
        measured these sites also owns what counts as occupied.
        """

        occupied = np.asarray(
            [
                [
                    [
                        self._calibration.detect(
                            image, model_kind=self._model.kind
                        ).occupied
                        for image in pair
                    ]
                    for pair in repeat
                ]
                for repeat in cycles
            ],
            dtype=bool,
        )
        return occupied[:, :, 0, :], occupied[:, :, 1, :]

    def _survival(
        self,
        before: np.ndarray,
        after: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Survival per site, and the recapture fraction per (repeat, point).

        A site that held no atom before the release answers nothing about
        recapture, so its survival is NaN -- not zero, which would be a
        measured loss that never happened.
        """

        survival = np.where(before, after.astype("<f8"), np.nan)
        loaded = np.sum(before, axis=-1)
        rate = np.divide(
            np.sum(before & after, axis=-1, dtype=float),
            loaded,
            out=np.full(loaded.shape, np.nan, dtype="<f8"),
            where=loaded > 0,
        )
        return survival, rate

    def _fit(self, before: np.ndarray, after: np.ndarray) -> dict[str, object]:
        """The decay of the POOLED recapture fraction, and what it implies.

        Pooled, not averaged over per-shot fractions: every loaded site is one
        Bernoulli trial, and a shot that loaded three atoms says less than one
        that loaded thirty.

        The lifetime is the 1/e time of that decay.  Turning it into a
        temperature needs one apparatus fact this Task cannot measure -- how
        far an atom may drift and still be recaptured -- so the operator
        declares it, and what is reported is the ballistic escape estimate it
        implies, named as such in the artifact.
        """

        loaded = np.sum(before, axis=(0, 2), dtype=float)
        recaptured = np.sum(before & after, axis=(0, 2), dtype=float)
        empty = tuple(
            float(value)
            for value, count in zip(self._t_off, loaded, strict=True)
            if count <= 0
        )
        if empty:
            raise ValueError(
                f"no trap was loaded at release times {empty}, so there is "
                "nothing to recapture there"
            )
        fraction = recaptured / loaded
        dark = tuple(
            float(value)
            for value, kept in zip(self._t_off, fraction, strict=True)
            if kept <= 0
        )
        if dark:
            raise ValueError(
                f"every atom was lost at release times {dark}, so the decay "
                "has no logarithm there; measure shorter releases"
            )
        seconds = _seconds(self._t_off, self._port.unit)
        slope, intercept = (
            float(value) for value in np.polyfit(seconds, np.log(fraction), 1)
        )
        if slope >= 0.0:
            raise ValueError(
                "the recapture fraction does not fall with the release time "
                f"({tuple(round(float(value), 3) for value in fraction)}), so "
                "this run has no release-recapture lifetime"
            )
        lifetime = -1.0 / slope
        speed = self._capture_radius_m / lifetime
        return {
            "loaded_pairs": [int(value) for value in loaded],
            "recaptured_pairs": [int(value) for value in recaptured],
            "survival_rate": [float(value) for value in fraction],
            "lifetime_seconds": lifetime,
            "amplitude": float(np.exp(intercept)),
            "decay_model": "survival = amplitude * exp(-t_off / lifetime)",
            "temperature_kelvin": (
                RB87_MASS_KG * speed * speed / (3.0 * BOLTZMANN_J_PER_K)
            ),
            "temperature_model": (
                "ballistic escape: an atom leaves the recapture radius in one "
                "lifetime, so v_rms = radius / lifetime and T = m v_rms^2 / (3 kB)"
            ),
            "capture_radius_m": self._capture_radius_m,
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

    def _run_record(self, fit: dict[str, object]) -> dict[str, object]:
        return {
            "node": self.instance_id,
            "parameters": {
                "calibration_path": str(self._calibration_path),
                "model_kind": self._model.kind.value,
                "capture_radius_m": self._capture_radius_m,
                "probe_frames": PROBE_FRAMES,
            },
            "scan": self._scan.run_record(),
            "fit": dict(fit),
        }

    def execute(self, context: object) -> dict[str, object]:
        frames = self._scan.acquire(context)
        context.report_progress("Reading survival")
        cycles = self._cycles(frames)
        before, after = self._occupancy(cycles)
        survival, rate = self._survival(before, after)
        fit = self._fit(before, after)
        record = self._run_record(fit)
        generation = str(
            getattr(context.generation, "value", context.generation)
        )
        # This Task publishes its results once, at the end; the revision only
        # has to be its own and positive, and the sweep count is that number.
        revision = self._scan.repeats
        point_column = self._point_column()
        outputs = {
            SCAN_OUTPUT.name: FinalDatasetOutput(SCAN_OUTPUT, frames, record),
            SURVIVAL_OUTPUT.name: FinalDatasetOutput(
                SURVIVAL_OUTPUT,
                snapshot_from_array(
                    survival,
                    producer=self.instance_id,
                    signal=SURVIVAL_OUTPUT.name,
                    roles=(SCAN_POINT, SITE),
                    axis_specs={SITE: self._calibration.site_map.site_axis},
                    point_columns={SCAN_POINT: point_column},
                    generation=generation,
                    revision=revision,
                ),
                record,
            ),
            SURVIVAL_RATE_OUTPUT.name: FinalDatasetOutput(
                SURVIVAL_RATE_OUTPUT,
                snapshot_from_array(
                    rate,
                    producer=self.instance_id,
                    signal=SURVIVAL_RATE_OUTPUT.name,
                    roles=(SCAN_POINT,),
                    point_columns={SCAN_POINT: point_column},
                    generation=generation,
                    revision=revision,
                ),
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
            "lifetime_seconds": fit["lifetime_seconds"],
            "temperature_kelvin": fit["temperature_kelvin"],
        }


__all__ = [
    "PROBE_FRAMES",
    "SURVIVAL_OUTPUT",
    "SURVIVAL_RATE_OUTPUT",
    "TEMPERATURE_ARTIFACT_CONTRACT",
    "T_OFF_PARAMETER",
    "TemperatureTask",
]
