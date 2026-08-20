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
target and solves only its latest edit into a Pattern/base phase in the
background. It has only **Pattern** and **Wavefront** pages. Pattern retains two
independent `2x2 = 490 x 357` logical target and pre-correction science-phase
plots. A shared **Size** selector also controls the independent Wavefront
preview; scrollable canvases avoid overlap or clipping at larger presets.

The main page exposes only Input pupil, Zernike, and vendor correction. Input
pupil defaults to a centered Gaussian whose `1/e^2` intensity diameter is 70%
of the SLM height; its center and X/Y diameters are editable, and Off means
uniform full-raster solver illumination. Wavefront puts full-raster Steering
X/Y and Noll Z4-Z11 under the same Zernike switch. Pattern authoring offers
exact-spacing Grid, geometrically staggered Checkerboard, Gaussian, Flat Top,
and English/Chinese Text with both minimum site spacing and atom budget.
Target JSON and science-phase NPZ load/save never write hardware. Loading a
science phase resets the Zernike layer for an exact roundtrip. Only **Send to
SLM** takes a short exclusive claim and applies the science phase; closing the
Editor preserves the currently commanded phase.

The X15213 leaf supports the series' `1272 x 1024` active LCOS raster through
either official USB frame memory or a `1280 x 1024 @ approximately 60 Hz` DVI
display. Generic **Scan hardware** reports a USB candidate only when the local
official `hpkSLMdaLV.dll` and `hpkSLMda.dll` can open a controller and read its
head serial. It reports attached displays with the required raster/timing as
DVI candidates, but display enumeration cannot prove which one is the SLM, so
the operator must confirm it. Scan does not send a phase. USB performs byte
readback; DVI requires an exact unscaled full-raster presenter and places the
active image in the left 1272 columns, leaving the right eight columns zero.

Each supported head has a small profile under `devices/slm/profiles/` with its
serial, readout wavelength, and full 256-point phase calibration curve.
Authored wavelength builds the nonlinear phase-code-to-drive LUT from that
curve; it is not record-only and `two_pi_gray` is a computed readout (for
LSH0804382 at 852 nm it is 225), not an editable constant. A local native
`L 1272 x 1024` correction BMP is added to the phase code modulo 256 before the
LUT. A filename containing its source wavelength is unwrapped and converted to
the authored wavelength without resizing or interpolation. These paths are
ready for experiment-machine bring-up, while development-machine mock/readback
tests prove frame bytes rather than controller state or optical acceptance.

The calibration mathematics under `nodes/calibration/` is headless and has no
Qt dependency. Calibration consumes a project-owned
`zlc.pulse.v1` JSON document through the public `zlc_pulse` codec/resolution
API; there is no Python pulse-module format or second resolver. After one
calibration run produces one result, the Calibration plugin saves its JSON and
passes that same result directly to the public `zlc_plot` API for six report
images. The Atom foundation does not depend on plotting; this plugin-local
report belongs to the Calibration task itself. Per-site report thresholds are
passed as coordinate-addressed site targets, so facet insertion/reordering
cannot move a threshold by index. Camera sample archives stream through the
single public `zlc_data.write_figure_archive()` encoder into an atomic file;
they do not materialize whole-archive bytes or import Workbench PanelState.

Logic Nodes commit `zlc-data` role-axis event chunks, so repeat, site, and
readout-event meaning is carried by `DatasetSchema` rather than inferred from
array shape. Runtime owns their accumulated current/partial/final
`OwnedSnapshot`; a plugin does not keep a second live history.

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
| Processor | consume a frames signal plus calibration and publish counts, occupied validity, and the exact frame judged with lineage | excitation or camera control |

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
signal plane. Each classifier grid binds every finite threshold to the
canonical `calibration.site` coordinate that measured it rather than to the
current facet index.
`OccupancyProcessor` consumes an explicit frames signal plus the typed saved
calibration and selects `default`, `box`, `psf`, or `uniform_psf` readout.
It publishes three same-publication Dataset siblings: per-site photoelectron
`counts`; per-site boolean `occupied`, whose component validity is the sole
truth for whether each site was readable; and `frame_judged`, the exact source
camera-frame snapshot that was classified. `frame_judged` preserves the source
bytes, axes, and validity rather than creating a second image truth. There is
no separate validity signal and no occupancy-rate output. The generic
`zlc_plot` overlay declaration and same-run geometry document let any
compatible presenter join `occupied` to `frame_judged` without knowing the
Occupancy plugin. The concrete Temperature Task reuses the occupied values and
their expanded validity for its authored before/trap-off/after cycles: only a
valid, initially occupied pair is a survival trial. It publishes the binary
per-site `survival` dataset only. Its declared preview and artifact both pool
that same dataset and its validity into survival rate against trap-off time;
there is no second rate history. It does not fit a temperature or lifetime and
does not derive a 1/e crossing.

The `slm_feedback` Task accepts one sparse target point per calibrated site.
Every coarse or validation measurement is a canonical Camera Measurement
generation under the stable companion producer `<task>/camera`: it commits all
`repeat=N` three-frame cycles, seals them, selects readout-event `frame=1`, and
uses every shot in the repeat statistics. The displayed camera preview applies
that same frame selection and repeat mean. The estimator subtracts each BOX
model's dark mean but does not threshold shots through Occupancy; loading and
occupied-atom brightness both remain part of the measured all-shot
fluorescence observable.

The controller updates `w_i *= (GM(F) / F_i) ** 0.25`; it performs no Zernike,
modal, hidden-aberration, or continuous-wavefront fit and makes no claim about
pixels between measured sites. The Task's own typed previews are the latest
candidate phase and the complete uniformity-history curve. Success reapplies
and saves the independently validated phase. Stop accepts the best valid
measured phase available, republishes that phase beside the current history,
applies it, and returns its durable artifact; a genuine failure restores the
incoming phase. The default 100 coarse and 100 validation shots are acquisition
defaults, not proof that a finite run must reach 1%.

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
descriptor and `NodeHost` paths. Exact processors replay and then follow each
committed event chunk; latest display processors explicitly coalesce. A
processor started on a sealed generation consumes its canonical
`current_dataset()` once. `zlc_pulse` remains a separate protocol package, and
the virtual sequencer has no alternate analysis path or `if virtual` branch.

## Current package boundary

Real camera adapters remain under `devices/camera/`; all virtual camera,
sequencer, shared-world geometry, and virtual descriptors remain under
`devices/simulation/`. Installation descriptors expose only operator-owned
settings. Logic descriptors own their authoring schema, typed artifact/resource
inputs, dataset outputs, artifact outputs, and task preview declaration;
Workbench and Qt consume those contracts rather than rebuilding them.
