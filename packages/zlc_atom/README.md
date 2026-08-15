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
and the real Hamamatsu LCOS-SLM X15213 leaf implement the same narrow
`SlmAdapter` canonical-radians contract.

Both `slm.virtual` and `slm.hamamatsu_x15213` open the concrete SLM Editor
lazily from the loaded device card. The Editor has one continuous non-negative
target and solves only its latest edit into a Mask/base phase in the background.
The default **Mask** page retains two `490 x 357` logical target/Final plots.
Mask ROI, common Wavefront controls (Steering X/Y, Z4-Z6 and Reset), and
Advanced Z2/Z3/Z7-Z11 controls are separate scrollable pages; the Advanced tab
also reports how many hidden coefficients are active. The Mask can be limited
to an authored ROI; full-raster X/Y carrier and unit-pupil Noll coefficients
are then added to make the displayed Final canonical phase. Target JSON and
Mask/Final NPZ load/save never write hardware.
An imported raw phase-mask image must be two-dimensional 8-bit grayscale and is
mapped as `gray * 2*pi/256`; it may match the complete device or an enabled ROI
and is explicitly not the vendor correction/LUT. Loading Final resets the
layers for an exact roundtrip. Only **Send to SLM** takes a short exclusive
claim and applies Final; closing the Editor preserves the currently commanded
phase.

The X15213 leaf supports the series' `1272 x 1024` active LCOS raster through
either official USB frame memory or a `1280 x 1024 @ approximately 60 Hz` DVI
display. Generic **Scan hardware** reports a USB candidate only when the local
official `hpkSLMdaLV.dll` and `hpkSLMda.dll` can open a controller and read its
head serial. It reports attached displays with the required raster/timing as
DVI candidates, but display enumeration cannot prove which one is the SLM, so
the operator must confirm it. Scan does not send a phase. USB performs byte
readback; DVI requires an exact unscaled full-raster presenter. Radians-to-gray
uses the bench-measured `two_pi_gray`; wavelength is record-only and no LUT is
inferred. The vendor correction BMP stays on the experiment machine and is
loaded by path as either `1272 x 1024` active or `1280 x 1024` full-raster
8-bit grayscale, with authored sign/offset, active-X, flips, and settle. These
paths are ready for experiment-machine bring-up, while development-machine
mock/readback tests are not optical acceptance.

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
35 calibrated atom sites. It extracts the same per-site BOX feature as
Calibration and averages the raw finite values only for shots that the shared
`OccupancyProcessor` marks occupied; empty shots never enter the denominator,
and the loop does not divide by each site's dark/bright response. This is the
direct qCMOS observable requested by the experiment, not a claim that one
atom's PSF brightness is hidden trap truth. The Task directly
updates `w_i *= (GM(F) / F_i) ** 0.45`; it performs no Zernike, modal, hidden
aberration, or continuous-wavefront fit, and it makes no claim about pixels
between the qCMOS-observed sites. Success reapplies and saves the independently
validated phase; Stop or failure restores the incoming phase.
The shipped defaults are functional acquisition defaults, not evidence of 1%
finite-shot acceptance. Earlier shot-count screens depended on a rejected
depth-dependent loading model and are not current evidence. The raw occupied
BOX observable must pass fresh exact-qCMOS, multi-seed validation; until then
the Task fails honestly when its authored budget is insufficient rather than
lowering the criterion or reading virtual hidden truth.

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

In the shared virtual world, every cooling rising edge while `trap` is high is
a fresh shot: each currently active trap receives an independent draw at the
same authored base loading probability, while an inactive/missing trap has
probability zero.
Local coherent trap depth does not exponentially or otherwise raise loading;
it affects occupied-atom qCMOS brightness and release survival. Changing the
SLM topology therefore removes atoms from vanished traps, and restoring a trap
does not resurrect its old atom without a later cooling load.

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
