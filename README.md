# Zou Lab Control

Neutral-atom experiment control shipped as one Python distribution with eight
internal dependency layers. The distribution bootstrap is `zou_lab_control`;
the internal packages remain the eight `zlc_*` dependency boundaries.

## Install one product

The root `pyproject.toml` is the only distribution manifest and
`constraints.txt` is the only dependency constraint surface. The eight
`packages/zlc_*/src` trees are internal layers, not standalone wheels.

```powershell
python -m pip install -c constraints.txt -e ".[notebook]"
zlc check
```

On a fresh Windows machine, run `bin\install_requirements.bat` once to install
the constrained dependencies and root product. Afterwards the launchers can be
double-clicked directly. Every checkout command, including FPGA build/program
and resource estimation, resolves Python and activates that checkout's
`zou_lab_control` bootstrap through one shared owner. Pulling source changes
therefore does not require reinstalling the package merely to expose a renamed
or new module. A release wheel uses the same constraint surface:

```powershell
python -m pip install -c constraints.txt "zou_lab_control-2.0.0-py3-none-any.whl[dev]"
zlc check
```

The wheel contains the product bootstrap, all eight layers, Calibration/Scan
templates, the SLM profile, Plot font, and tracked FPGA RTL/XDC/Tcl assets.

## Product commands and launchers

The installed command is manifest-driven:

```text
zlc task_console      experiment Device Manager -> Task Console flow
zlc device_manager    Device Manager
zlc pulse_editor      Pulse Editor
zlc figure_viewer     saved Figure archive viewer
zlc pulse_server      separated FPGA-machine owner
zlc slm_server        separated DVI-default / explicit-USB SLM owner
zlc fpga              FPGA geometry/resource tool
zlc check             installed-product provenance check
zlc evidence          formal evidence lanes
```

The Windows files in `bin\` are thin shortcuts. Their shared Python resolver
anchors imports to the current checkout, then they select one manifest command
and forward the original argument vector once. `install_requirements.bat` is
the sole installed-only mode, so its final check cannot be masked by source.

```text
bin\experiment.bat             Device Manager Init -> Task Console
bin\pulse_editor.bat           Pulse Editor
bin\figure_viewer.bat          Figure Viewer
bin\estimate_resources.bat     current checkout geometry/resource estimate
bin\build_and_program.bat      build if needed, then program volatile FPGA;
                               build-only/flash remain explicit modes
```

The machine a board or SLM head is plugged into serves it from the bench
itself: install `sequencer.local` / `slm.hamamatsu_x15213_local` in the Device
Manager and Init starts the server in-process; once published with Remote,
the device card's Log button tails that device's own narration. The headless `pulse_server` / `slm_server` commands remain
for a machine without a bench window.

An experiment window uses the experiment folder it is started in (the nearest
folder at or above the working directory holding `pulses\` or
`apparatus.json`); failing that `ZLC_WORKSPACE`, and failing that the
checkout's own untracked `workspace\`. An installed product has no checkout
folder and refuses to guess: pass `--workspace` or set `ZLC_WORKSPACE`. Set
`ZLC_PY_CMD` only when the intended Python is outside normal discovery.

A Config file under `config_values\` is how the BOARD is calibrated: named
values such as channel delays and DAC biases. It is loaded onto the sequencer,
not into a pulse, so every pulse on the bench plays through the same set. The
sequencer's Init names the file (left empty, none is loaded; no
`current.json` is read implicitly), and the Pulse Editor's Config tab loads,
edits and saves it; only a saved file ever reaches a Fire. A field bound to a
name the loaded file does not provide plays its authored default and shows as
not overridden; a value that is used but malformed or in the wrong unit is
reported, not guessed.

A pulse's own API parameters are a different thing: they belong to the pulse
file, and a node that loads a pulse offers a per-run table for them, so one
run can hold a different number without editing the pulse.

## One runtime path

Virtual and physical devices use the same descriptor/catalog/NodeHost/session
path:

```text
Calibration Task -> run folder + calibration JSON + typed report Figures
Camera Measurement -> canonical frames Dataset
Occupancy Processor(frames + Calibration) -> counts/occupied/judged frame
SLM Editor -> strict Target/Science Context -> explicit Send
SLM Feedback(Calibration + Science Context + pulse + exposure) -> report + final Context
Panel -> matching data/fit/overlay -> data-backed Figure + PNG preview
```

### GPU SLM rearrangement

This implementation is still under optical and performance validation.
Its current `converged` flag checks intended-site endpoint intensities and
authored dark sites, not moving-well shape, background light or finite-refresh
behavior. Do not use that flag as experimental approval to transport atoms.
The solver uses phase projection and amplitude updates. A depth-two Anderson
step mixes the current frame's recent log-amplitude residuals in both coarse
updates and actual encoded-field corrections. Each frame keeps its own history;
changing correction damping resets it. Native-grid checks remain authoritative.
Opposite horizontal Fourier coordinates share cosine/sine products. Each
physical pixel is still reconstructed and receives its own pupil amplitude;
the center and unpaired negative edge are counted exactly once. This reduces
matrix multiplication work without assuming a symmetric beam or cropping the SLM.
The retired per-frame Newton/Jacobian solver and unvalidated neural predictor
are not retained as alternate implementations.

The optional numerical API is in `zlc_atom.devices.slm.solver`. Install its
CUDA dependencies with `python -m pip install -e ".[slm-gpu]"`. Ordinary
discovery, the static SLM Editor, and Feedback do not import CUDA or change
backend implicitly.

`prepare_rearrangement(...)` prepares fixed source/target geometry, endpoint
holograms and GPU resources **before** imaging. Coordinates are integer Y,X
indices in the native centered Fourier image, not camera pixels or micrometres.
The input `pupil_phase` is a measured incident aberration to compensate; it is
not an intentional carrier or the device's vendor response correction.
Preparation also reserves two output buffers in each needed pinned-memory size
class, bounded by the declared geometry and matching range. This is empty storage,
not precomputed occupancy-dependent answers. The 1024×1272 benchmark reserves
496 MiB for a 32-bin maximum range, or 2032 MiB for 80 bins; preparation time and
RAM are reported separately from online latency. The pool belongs to this working
point and unused buffers are released on close, without flushing global caches.

Apply `prepared["initial_phase_codes"]`, acquire a boolean occupancy vector in
the exact source-site order, then call:

```python
sequence = compute_rearrangement(
    prepared, occupied, surplus_policy="discard",  # explicitly extinguish unused sites
)
# All sequence["phase_codes"] frames are now independent, read-only host arrays.
# At a separately characterized, safe device cadence:
for codes in sequence["phase_codes"]:
    slm.apply_phase_codes(codes)
# When this fixed optical working point is no longer needed:
prepared["close"]()
```

The physical SLM proxy accepts this encoding through its existing serialized
command lane; orientation, correction, wavelength mapping, acknowledgements,
and optical settling stay with the server. Calculation never sends frames on
its own. There is no claim that a liquid-crystal display or the atoms move in
the computation time. Frame fractions are spatial progress, not an invented
hardware clock.

Preparation requires explicit `shape_yx`, `pupil_amplitude`, `matching_radii`
and `minimum_separation`, plus the source and target arrays. The solver chooses
the first feasible authored maximum-distance bound, minimizes squared distance
within it, and checks the piecewise-linear paths including between frames.
Motion emits three substeps per integer planner segment; returned positions
describe those actual fractional Fourier coordinates without rounding.
Insufficient atoms, infeasible assignments, collisions and unconverged phase
sequences are rejected. The default intended-site intensity max/min gate is 1.01;
explicit diagnostic `require_converged=False` returns the actual metrics, not
a successful quality verdict. `iterations=None` uses a prepared fixed initial
schedule followed by bounded correction of only failed encoded frames. An
explicit integer requests that many projection/weight updates per frame and
disables additional adaptive updates; zero still encodes and measures the result.

Optional `endpoint_data` supplies already prepared source/target phase codes
and synthesis coefficients at the same geometry, pupil and requested intensities.
These are fixed optical working points, not cached occupancy-dependent answers
or a neural model. Actual fields are measured from the encoded patterns, while
intensity gates remain relative to the requested site weights. A prepared target
can be reused only at its exact endpoint and after satisfying that same gate.
Closing the prepared GPU workspace does not invalidate already returned host maps.

Online timing covers occupancy-dependent planning, clearance, every generated
full-frame mask, quantization and host transfer. CUDA/JIT/endpoint preparation
is separate. Network-only inference, one iteration, and first-frame latency
are not complete-rearrangement measurements. The returned metrics include
intensity error, trap-phase changes and sampled site power; numerical
pixel-transition models are not experimental atom-survival evidence.

Task Console, Device Control, Pulse Editor and SLM Editor share one
`ExperimentSession`, named devices, signal plane and sequencer. Loaded-device
cards expose Control and Close. Adding, removing, renaming or reconfiguring a
device does not recreate the whole session: unchanged canonical device leaves,
Task Console and Panels are retained, while a key-scoped maintenance barrier
stops only conflicting Logic/commands. A role rename preserves stable
`instance_id`; partial close/build failures keep every still-open leaf reachable,
project the effective live config, and remain retryable.

Runtime owns complete live/final Dataset truth. Plot surfaces keep one
active+latest solve and atomically show matching data/fit revisions. The three
save actions stay separate: Save Layout writes stopped wiring, Save Screenshot
writes a GUI image, and Panel Save Fig writes only that panel's frozen data/image
plus actual run/device provenance.

Every hosted Task allocates its unique run folder only when execution actually
starts. A run writes two records about itself, each created once and never
replaced: `start.json` at Start (identity, normalized inputs, `started_at`) and
`run.json` when the run is over (terminal status, stop reason, last progress,
registered artifacts, failure). Nothing is written in between, so a folder
holding `start.json` and no `run.json` is a run that did not finish. Tasks save
only curated domain outputs; Runtime does not dump all live data or
intermediate shots. Calibration and SLM Feedback use this same lifecycle.

Calibration's threshold method is the operator's choice and defaults to
`gaussian`: every site fits an unlabelled two-component Gaussian mixture to all
of its finite short-shot signals, keeps the fitted population weights, and takes
as threshold the analytic crossing of the two weighted component curves between
their means -- the point that minimises the fitted populations' total
misclassification. Reference labels never enter the Gaussian parameters,
weights or threshold; they serve only the Empirical threshold and the reported
actual fidelity. A site whose fit, populations or crossing is invalid uses the
empirical threshold that maximises actual accuracy over its labelled samples,
and selecting `empirical` explicitly uses that for every site. Histogram
threshold lines always show the final deployed classifier, and the drawn
Gaussian curves reuse the saved parameters and weights rather than a second
fit. The report saves the overall actual fidelity at the final threshold on all
valid labelled data (with dark/bright conditional values) and, for Gaussian
sites, the theoretical fidelity integrated from the fitted population weights.

Calibration can optionally pause once after site detection for operator review.
TaskConsole shows the detected SiteMap over the reference average; the operator
may exclude unwanted diffraction/ghost sites by point, list, or rectangle and
then continue. Capture and detection are not repeated, all downstream models
are fit once from the retained SiteMap, and the run saves the reviewed map as a
data-backed Figure plus PNG alongside the ordinary report.

### Camera and qCMOS

Camera Measurement owns its exposure, ROI, frames-per-cycle and repeat; Pulse
timing does not infer or validate camera exposure. Real and virtual adapters
publish only frames actually acquired, with source ordinals contiguous from zero
for each arm.

qCMOS applies only changed ROI/exposure settings. An unchanged Start does not
rewrite the complete sensor working point, and Camera Measurement freezes the
authoritative readback returned by configuration instead of issuing another full
property query. Auto Panels connect to the canonical publication/preview signal;
they must not wait for a redundant reconfiguration or a fixed five-second poll.
Photoelectron output is used only when the device provides complete conversion;
otherwise the effective run falls back to native counts while the authored draft
remains visible.

### SLM and fluorescence feedback

The real device type is `slm.hamamatsu_x15213`; apparatus parameters are server
host and port. The server is the sole DVI/USB output owner. DVI exact-raster is
the default and does not load the vendor DLL; USB is selected explicitly. The
trusted-lab-LAN proxy has no TLS/authentication and must not be exposed publicly.

Target stores intensity/objective. Science Context stores the frozen Target,
the pre-command 16-bit circular Pattern, semantic pupil/operator parameters,
correction reference and command receipt. Numeric pupil, operator wavefront and
composite phase are rebuilt by the same SLM core formulas rather than duplicated
as full rasters. Loading a Context adopts that frozen phase without solving and
never writes hardware;
only explicit Send or the Feedback task's own confirmed apply establishes a
known command. Both formats are strict current-only and carry no numeric format
version; unsupported files are refused.

Calibration remains an SLM-independent camera/readout artifact and supplies only
registered site BOX geometry to Feedback. Feedback uses one canonical single-
frame Camera Measurement batch per candidate (100 authored shots by default). A
site is observable only when the constrained two-Gaussian fit of its whole batch
is numerically valid, meets the component and separation conditions and wins by
full-data ΔBIC > 10; an ordinary fit that does not is a single (not loaded), and
a numeric or acquisition failure is invalid and holds its share. A dark site
moves toward the loading edge by bracketed bisection, or one resolution step
along its probed direction, funded in share space by the loaded sites whose own
loop step does not point the other way, each giving at most one resolution step
per candidate with total power conserved; a loaded site
on the loading ramp (bright fraction below half the array median) holds; only
formal double updates use `feedback_gain`. Every next acquisition requires a
confirmed different phase. The task stops when three consecutive formal
candidates have a split-half variance indistinguishable from zero, or at the
authored update limit (12 by default), and keeps the completely measured
candidate with the smallest split-half variance plus its standard error; it has
no built-in uniformity-ratio stop and no hidden retry or validation batch.

Feedback stores a stable site table and curated per-candidate BOX samples,
fit/classification, weights, actions, metrics, phase-change facts and command
receipts. It does not save raw camera frames; every completed candidate has one
compact standalone Science Context whose Pattern is exactly the phase frozen
before that candidate's shots. Its summary and six important plots cover uniformity, site signals,
weights, selected-site histograms, initial/selected camera means and
initial/selected phases. Each plot is a `zlc.figure` NPZ plus same-stem PNG;
normal completion or Stop produces one final Science Context.

The virtual plant uses one commanded-phase Fourier trap roster, one shared
non-symmetric imaging PSF and a fixed apparatus aberration independent of Target
or grid. With 20 µK cooling, traps below 500 µK do not load; the 520 µK nominal
depth deliberately places ordinary optical nonuniformity near that edge.

### Pulse and FPGA

Pulse execution has three explicit hardware layers: Scan repeats walk the
complete table, Run repeats play the complete Pulse at one row, and named,
nestable PulseBrackets loop only their authored timeline subsets. Camera
frames-per-cycle and Dataset repeat remain independent acquisition facts.
The Pulse server alone owns UART/JTAG hardware; normal disconnect drives SAFE,
UART auto-selection requires the word-63 fingerprint, and explicit UART failure
does not silently choose another port.

The FPGA host validates part/device identity, target ABI, clock, geometry,
counts and delay-FIFO capacity before Load. SAFE independently gates TTL/DAC;
DONE waits through final FIFO/latch completion. Vivado scratch stays under the
FPGA build root. `run_server` never programs hardware; the default
`build_and_program` path builds/reuses a valid project and then programs volatile
FPGA state, while flash is always explicit. Timing/build evidence and the
remaining board acceptance boundary are recorded in `IMPLEMENTATION_PLAN.md`
and `packages/zlc_pulse/fpga/README.md`.

## Persistence and notebook

Figure uses stable `zlc.figure`; Calibration uses
`zlc.calibration.readout`; Pulse uses `zlc.pulse`; Target uses `zlc.slm.target`;
Science Context uses `zlc.slm.science-context`. Readers accept only the current
complete grammar. Existing workspace files are not converted and may be outside
the current grammar; formats have no alias or numeric version.

A Figure NPZ is the primary artifact and contains typed Dataset data, exact Plot
recipe, overlay, viewport and causal lineage. PNG is only its preview.
FigureViewer publishes the archive's typed Datasets as sealed Runtime signals
and reopens the default recipe through the same Panel/SelectionBridge/Plot host
path as TaskConsole; it never guesses plot semantics from array shape. Additional
fixed-kind panels select those archive signals or Viewer-created derived signals
through ordinary Setting controls.

The only supported tutorial is
`packages/zlc_workbench/notebooks/usage.ipynb`. It uses the installed product, a
temporary workspace, virtual devices, canonical Camera Measurement publication
and `NotebookView`; it contains no hardware cell, source-path bootstrap, saved
output or execution count.

## Evidence lanes

Automated release lanes run from a fresh wheel outside the checkout:

```powershell
zlc evidence software --repo C:\path\to\Zou_lab_control
zlc evidence gui_offscreen --repo C:\path\to\Zou_lab_control
zlc evidence virtual_vertical --repo C:\path\to\Zou_lab_control
zlc evidence notebook_offline --repo C:\path\to\Zou_lab_control
```

`real_screen` and `hardware` are manual-only; the CLI reports them as
`NOT EXECUTED` and never touches a device. Software/virtual evidence must not be
presented as real monitor, camera, optical SLM/Feedback, FPGA program/flash or
external DAC/TTL acceptance. The authoritative manual runbooks are the package
READMEs; receipts record product/module paths, device identities, requested and
actual working points, raw evidence, timestamp, operator observations and
pass/fail.

## Layer boundaries

| Layer | Owns | Must not own |
|---|---|---|
| `zlc_data` | immutable scientific schema, validity, selection and codecs | Runtime, Qt, devices, paths |
| `zlc_durable` | atomic publication and workspace paths | scientific meaning |
| `zlc_runtime` | lifecycle, canonical accumulation, fronts and scheduling | plugin physics, plotting, Qt |
| `zlc_plot` | projection, rendering, fit, overlay and selector | Task/device ownership |
| `zlc_ui` | Qt views and plain view models | Runtime/Plot/device/domain truth |
| `zlc_pulse` | pulse model, compiler, wire and transport | measurement policy |
| `zlc_atom` | device plugins and atom-science nodes | Workbench composition |
| `zlc_workbench` | session/composition/device claims/layout | plugin science or second pipelines |

`ARCHITECTURE_DESIGN.md` records current product invariants.
`IMPLEMENTATION_PLAN.md` records the active implementation checkpoint, pending
current-tree verification, retained hardware build evidence and explicit
experiment-machine acceptance boundary.
