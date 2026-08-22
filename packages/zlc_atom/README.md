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

One apparatus-level `simulation` mapping owns the virtual world's
`image_shape_yx`, `grid_shape_yx`, `seed`, and optional workspace-local
`world_profile`. The profile path is resolved under the workspace before any
device factory runs. `camera.virtual` consequently authors only camera facts
(currently exposure) and consumes the world's image geometry; the independent
`camera.virtual_mot` keeps its own frame geometry and does not declare a site
grid. The installation grammar is strict: files with the former camera-owned
world fields are refused with the required root migration instead of silently
creating a second owner.

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
A strict Target JSON stores intensity plus objective only for Editor authoring
import/export. Run consumers take no separate Target: a strict Science Context
v2 NPZ is their sole Target truth and freezes it together with Pattern/base
phase, numeric pupil,
operator wavefront, typed reusable system-correction reference and command
receipt. Loading a v2 Context atomically adopts all of those authored facts but
never writes hardware. A v1 Context loads only as an explicit authoring
migration with `target=None`; the Editor preserves its current Target draft,
while registered Feedback refuses the legacy Context until it is resaved as v2.
Only **Send to SLM** takes an exclusive claim
and applies the composed science phase. If a Task changes the command or
correction mapping, the Editor shows the divergence and refuses the old draft
until explicit Adopt or Context reload; closing preserves the currently
commanded phase.

The X15213 server supports the series' `1272 x 1024` active LCOS raster only
through the official USB frame-memory SDK. It loads the primary vendor library
from an optional explicit directory, `HAMAMATSU_SLM_SDK`, `PATH`, a discovered
vendor installation, or the normal Windows loader; it does not reject a valid
installation by preflighting a second DLL in the same folder. A new real
adapter starts with unknown command truth. A command becomes known only after
write, display-slot selection, exact active-frame readback, and the profile's
settle wait all complete.

The computer attached to the USB controller runs `bin/slm_server.bat` and is
the only process that loads the SDK. Like `sequencer.hardware`, the single real
device type `slm.hamamatsu_x15213` stores only its server host and port (default
`18862`) in the apparatus; `127.0.0.1` is the same-machine form. The proxy reads
identity, shape, current
phase and receipt once when the installation opens, then serves all Editor
display/state reads from that local immutable cache.  Target edits never cross
the network. A healthy **Send to SLM** or Task phase command sends one bounded
raw command: an 8-byte length header, strict-JSON metadata (at most 1 MiB), and
canonical `float32` phase bytes (at most 16 MiB). The apply carries expected
command and mapping revisions, so a stale window cannot overwrite a command
issued by another client. A malformed/partial/oversize reply or uncertain
transport outcome marks the local command truth unknown; the next command
first describes current physical truth and then applies. Correction and
profile paths remain server-local and are fixed by the server command line
rather than being sent from a remote UI. As required by the experiment remote
policy, this service deliberately has no authentication or TLS and belongs
only on the trusted laboratory LAN; it must not be exposed to the public
Internet. With the default `0.0.0.0` listen bind, startup prints both the
same-machine endpoint and every discovered LAN IPv4 address in the exact
host/port form entered in the device configuration.

Each supported head has a strict profile under `devices/slm/profiles/` with
model, serial, working and phase-curve wavelengths, curve provenance, and
settle provenance. Authored wavelength builds the nonlinear
phase-code-to-drive LUT from that curve; `two_pi_gray` remains computed rather
than authored. A native `L 1272 x 1024` correction BMP is added modulo 256
before the LUT. Loading or toggling correction advances a mapping revision, and
each command receipt freezes the USB/profile/wavelength/orientation/correction
facts it used. A map labelled for another wavelength is rejected because the
repository has no measured two-dimensional unwrap authority. The bundled
LSH0804382 provenance is explicitly incomplete; development mocks prove the
software byte path, not the vendor ABI, controller, or optical acceptance.

USB experiment-machine acceptance remains an unexecuted runbook:

1. Record the full head model/serial, controller and SDK/DLL versions, then
   bind every ctypes function from that installed official header.
2. Confirm the selected profile's curve source, measurement wavelength and
   uncertainty instead of treating the bundled values as a calibration claim.
3. Verify vendor-correction encoding, serial, wavelength, sign and native pixel
   orientation; exercise correction Off/On under the same device claim.
4. Send asymmetric corner/gray patterns and confirm X/Y orientation plus exact
   active-frame USB readback at the controller.
5. Measure optical response for representative increasing and decreasing gray
   transitions, then replace the pending settle value with the accepted worst
   case and its source.
6. Record the resulting command receipt and separately validate any declared
   pupil-phase-map or target-response-map system correction; controller bytes
   and optical correction remain different evidence.

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
returns a saved calibration artifact. Without registration artifacts,
Calibration discovers its site count and centers from acquired images and
accepts no authored grid rows, columns, or site count. Optionally, one strict
Science Context v2 registers its frozen Target as the full stable roster; there
is no second Target input. Sites not captured in the calibration remain
explicit predicted boxes instead of disappearing. One topology validator runs
both when that SiteMap is created and when it is loaded, rechecking geometry,
rounded BOX overlap/bounds and runner-up uniqueness. The mapping trusts the
apparatus convention of positive Target x/y to camera x/y with no reflection,
shear or axis swap. An incompletely observed x/y/180°-symmetric support, or
another equally plausible alternate mapping, is refused until the Target has
an asymmetric spatial fiducial. Calibration capture does not claim or command
the SLM, so the experiment
workflow must keep the selected Context command unchanged throughout capture;
Feedback checks it again before starting. While hosted Calibration publishes
only the current `capture_preview` for Monitor. When the loop finishes it
computes one result, writes one plain JSON, and passes that same result to
`zlc_plot` for the six reports. Workbench neither renders nor opens those files,
and no calibration object/report blob is put on the signal plane. Each
classifier grid binds every finite threshold to the canonical
`calibration.site` coordinate that measured it rather than to the current facet
index. The BOX readout model also persists its dark sample count and sample
variance; legacy `n=0`/`NaN` artifacts remain readable but cannot authorize
registered Feedback.

Camera Measurement and camera-backed Calibration request photoelectrons by
default. A camera with a complete configured offset/scale publishes converted
`float32` values; a camera with no conversion makes the switch unavailable and
falls back effectively to its native raw counts. The immutable Dataset and run
record carry that effective truth: raw values retain the camera's `count` unit,
photoelectron values are dimensionless, and live preview, saved samples,
replay and downstream processors all use the same choice.

`OccupancyProcessor` consumes an explicit frames signal plus the typed saved
calibration and selects `default`, `box`, `psf`, or `uniform_psf` readout.
It publishes three same-publication Dataset siblings: per-site numeric
`counts` in the source frame's effective unit; per-site boolean `occupied`,
whose component validity is the sole truth for whether each site was readable;
and `frame_judged`, the exact source camera-frame snapshot that was classified.
`frame_judged` preserves the source bytes, axes, and validity rather than
creating a second image truth. There is no separate validity signal and no
occupancy-rate output. The generic
`zlc_plot` overlay declaration and same-run geometry document let any
compatible presenter join `occupied` to `frame_judged` without knowing the
Occupancy plugin. The concrete Temperature Task reuses the occupied values and
their expanded validity for its authored before/trap-off/after cycles: only a
valid, initially occupied pair is a survival trial. It publishes the binary
per-site `survival` dataset only. Its declared preview and artifact both pool
that same dataset and its validity into survival rate against trap-off time;
there is no second rate history. It does not fit a temperature or lifetime and
does not derive a 1/e crossing.

The `slm_feedback` Task takes only a registered Calibration and one strict
Science Context v2. That Context's frozen spots Target is the unique Target
truth, and the Calibration must be registered to the same Context. The full
roster, including predicted zero-capture sites, remains the stable site index.
At execution start the current SLM phase, complete command receipt
and mapping revision must still exactly equal the frozen incoming Context or
the Task fails before changing hardware. Every coarse or validation measurement
is a canonical Camera Measurement generation under the stable companion
producer `<task>/camera`: it commits all `repeat=N` three-frame cycles, seals
them, selects readout-event `frame=1`, and uses every shot in the repeat
statistics. The displayed preview applies that same selection and repeat mean.
The estimator integrates raw BOX samples and subtracts the persisted dark mean;
it neither thresholds through Occupancy nor divides by Calibration bright mean.
Its total SEM combines shot SEM with `dark sample variance / n` exactly once, so
loading and occupied-atom brightness remain part of the measured all-shot
fluorescence observable. Every measurement must replay Calibration's effective
raw/photoelectron mode and exact raw dtype/count-unit provenance; electron mode
also requires the same offset and scale. Saturation is derived from the raw
integer maximum and transformed into the effective unit, never inferred from a
converted floating-point dtype.

Each candidate may accumulate at most three coarse batches. A site whose
simultaneous lower fluorescence bound still crosses zero is censored, not
fabricated as a valid value. Before any valid candidate exists, at most three
bootstrap updates raise only censored target sites by a clipped log step of
`0.2` while renormalizing total target power; after a valid best exists, a
censored candidate rolls back to it. The controller otherwise updates only the
frozen Context's Pattern using uncertainty, clipped steps, trust and rollback;
it performs no Zernike, modal, hidden-aberration or continuous-wavefront fit.
If the run ends or is stopped without a valid result, incoming candidate 0 is
saved with no invented measurement. With a valid best, Stop accepts and
reapplies it; a Stop already requested before the initial solve makes zero
solver calls. A genuine failure restores the known incoming command. Each
candidate Context freezes its actual evolving Target and can be selected
directly for the next run because neither descriptor asks for a Target file.

The held-out validation applies its 95% family correction across both sites and
the maximum number of looks. It uses bounded batches without dropping a tail
(for example, 101 shots become 99 + 2), and remains bounded by the authored
shot limit and 60 seconds. The terminal Science Context reports `accepted` or
`inconclusive` with the measured estimate and simultaneous interval. In the
persistent virtual weak-site vertical, Calibration observed 34 of 35 sites;
three censored candidates led to a valid fourth candidate, after which honest
validation was inconclusive and retained that valid best. This is software/
virtual evidence, not hardware or optical acceptance.

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
