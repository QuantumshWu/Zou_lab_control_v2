# zlc_atom

`zlc_atom` has a headless foundation for atom-device contracts, installation,
logic-node discovery, and shared science. Concrete device and node plugins may
own their product-specific Qt or `zlc_plot` surfaces without pulling those
dependencies back into the foundation. It contains the camera/sequencer/SLM
contracts and the experiment's measurement, processor, and task leaves.
Virtual implementations live only under
`devices/simulation/`: `VirtualCamera` satisfies the same runtime-checkable
`CameraAdapter` contract as DCAM/Pylon, and `VirtualSequencer` is a
`SequencerDevice` over the same pulse-device surface as hardware. `VirtualSLM`
implements the same narrow `SlmAdapter` phase contract that a future real
Hamamatsu plugin will implement once its actual SDK is available.

The `slm.virtual` leaf opens its concrete SLM Editor lazily from the loaded
device card. The Editor has one continuous non-negative target, solves its
latest edit in the background, and leaves the commanded phase unchanged while
loading or saving targets and phases. Only **Send to SLM** takes a short
exclusive claim and applies a phase; closing the Editor preserves that phase.

The calibration mathematics under `nodes/calibration/` is headless and has no
Qt dependency. Calibration consumes a project-owned
`zlc.pulse.v1` JSON document through the public `zlc_pulse` codec/resolution
API; there is no Python pulse-module format or second resolver. After one
calibration run produces one result, the Calibration plugin saves its JSON and
passes that same result directly to the public `zlc_plot` API for six report
images. The Atom foundation does not depend on plotting; this plugin-local
report belongs to the Calibration task itself.

Node results also expose `zlc-data` role-axis `OwnedSnapshot` artifacts, so
repeat, site, and readout-event meaning is carried by `DatasetSchema` rather
than inferred from array shape.

## Leaf pattern

Add a measurement or processor by creating its package plus a
`logic_node.py` that exports `LOGIC_NODE`. Discovery is derived from those
files, so the composition framework and UI do not need edits. A device type is
similarly declared in a discovered `device_types.py` module with a factory and
authoring schema.

The current device set is deliberately closed to camera, sequencer, and SLM
capabilities. Virtual, DCAM, and Pylon camera adapters implement the same
`CameraAdapter` contract; Workbench resolves any compatible named instance such as
`camera` or `mot_camera` instead of hard-coding an instance name.

## Leaf tutorial

A new device leaf stays discoverable and self-contained:

```python
from zlc_atom.authoring import AuthoringSchema
from zlc_atom.install import DeviceTypeDescriptor, InstalledLeaf

def factory(context, key, values):
    device = build_device(context, values)
    return InstalledLeaf(key, "example.device", device, {})

DEVICE_TYPE = DeviceTypeDescriptor(
    "example.device", "example", AuthoringSchema(()), (), factory=factory
)
```

Place that descriptor in a `device_types.py` below `src/zlc_atom/devices/` and
the rglob discovery test will collect it without editing the graph. Logic
leaves follow the same pattern with a `logic_node.py` exporting `LOGIC_NODE`.
Factories must return declared capability instances; startup failures are
reported per leaf in `Installation.failures`, while independent leaves remain
usable and close in reverse order.

## Execution layers

The runtime has one direction of responsibility:

| Layer | Owns | Does not own |
| --- | --- | --- |
| Device | camera/sequencer protocols, buffers, trigger routing, virtual imaging world | calibration policy or analysis |
| Measurement | arm/read/finish observation and publication of camera frames | pulse selection or `sequencer.load/fire` |
| Task | pulse resolution, sequencer load/fire, repeated capture, progress/current preview, calibration, JSON artifact, and its six report-image saves | reusable readout mathematics or renderer internals |
| Processor | consume a frames signal plus calibration and derive counts/occupied/rate with lineage | excitation or camera control |

For a manually controlled experiment, the notebook calls `resolve_pulse`,
`sequencer.load`, and `sequencer.fire` around a pure camera measurement. For
an automated experiment, `CalibrationTask.run()` owns that whole sequence and
returns a saved calibration artifact. Calibration discovers its site count and
centers from acquired images; it accepts no authored grid rows, columns, or
site count. While hosted it publishes only the current `capture_preview` for
Monitor. When the loop finishes it computes one result, writes one plain JSON,
and passes the same in-memory result to `zlc_plot` to save site-map, fidelity,
three classifier grids, and a PSF-kernel grid. Workbench neither renders nor
opens those report files, and no calibration object/report blob is put on the
signal plane.
`OccupancyProcessor` consumes an explicit frames signal plus the typed saved
calibration and selects `default`, `box`, `psf`, or `uniform_psf` readout.
It owns only each frame's per-site counts, occupied boolean, validity, and
pooled occupancy rate. The concrete Temperature Task reuses those facts for
its authored before/trap-off/after cycles: only a valid, initially occupied
pair is a survival trial. It publishes the binary per-site `survival` dataset
and pooled `survival_rate` against trap-off time, whose declared preview is an
ordinary curve. It does not fit a temperature or lifetime and does not derive
a 1/e crossing.

The `slm_feedback` Task accepts one complete 5 x 7 sparse target aligned to the
35 calibrated atom sites. It extracts exact grouped qCMOS images with the
frozen site/PSF model, subtracts the dark response, normalizes by each site's
dark-to-bright response, and averages occupancy-weighted fluorescence across
many shots. This is a loading/survival-mediated feedback observable, not a
claim that one atom's PSF brightness is trap intensity. The Task directly
updates `w_i *= (GM(F) / F_i) ** 0.45`; it performs no Zernike, modal, hidden
aberration, or continuous-wavefront fit, and it makes no claim about pixels
between the qCMOS-observed sites. Success reapplies and saves the independently
validated phase; Stop or failure restores the incoming phase.
The shipped defaults are functional acquisition defaults, not evidence that a
1% finite-shot acceptance is quick: direct qCMOS shot-noise screens place that
gate at tens of millions of shots. A pre/post ratio is deliberately not used;
it cancels loading information and adds a second frame's noise. The Task must
therefore fail honestly when its authored shot budget is insufficient rather
than lower the criterion or read virtual hidden truth.

The supported product path discovers seven logic descriptors: `calibration`,
`camera_measurement`, `occupancy`, `seamless_scan`, `slm_feedback`,
`stepped_scan`, and `temperature`. They are hosted through the real runtime
plane: virtual Calibration writes a plain workspace JSON, Camera Measurement
publishes finite or `Repeat = 0` infinite frames, and Occupancy consumes the
frames key plus JSON path. Camera Measurement retains its per-row Auto preview:
the cycle's `frames` signal uses `facet_grid`, with one or many readout-event
rows as authored; switching Auto preview off only prevents the panel from
opening automatically.
Virtual and physical cameras differ only below the adapter boundary.

## Executable integration path

Run from the monorepo after importing the root bootstrap before any `zlc_*`
package, so the checkout under test cannot be confused with an older editable
installation:

```powershell
python -c "import zou_lab_control_v2; import zlc_atom; print(zlc_atom.__file__)"
pytest -q
```

The formal virtual guard uses the real `zlc_runtime.SignalDataPlane`, catalog,
descriptor and `NodeHost` paths. Finite processors drain ordered `FollowTap`
events or consume a retained final snapshot once; infinite sources expose only
latest data and no loss telemetry. `zlc_pulse` remains a separate protocol
package, and the virtual sequencer has no alternate analysis path or
`if virtual` branch.

## Current package boundary

Real camera adapters remain under `devices/camera/`; all virtual camera,
sequencer, shared-world geometry, and virtual descriptors remain under
`devices/simulation/`. Installation descriptors expose only operator-owned
settings. Logic descriptors own their authoring schema, typed artifact/resource
inputs, dataset outputs, artifact outputs, and task preview declaration;
Workbench and Qt consume those contracts rather than rebuilding them.
