# Zou Lab Control

Neutral-atom experiment control shipped as one Python distribution with eight
internal dependency layers.

## Install one product

The root `pyproject.toml` is the only distribution manifest and
`constraints.txt` is the only dependency constraint surface. The eight
`packages/zlc_*/src` trees are internal layers, not standalone wheels.

```powershell
python -m pip install -c constraints.txt -e ".[notebook]"
zlc check
```

On Windows, `bin\install_requirements.bat` performs the constrained root install
and provenance check. A release wheel uses the same constraint surface:

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

The Windows files in `bin\` are thin shortcuts. They resolve Python, select one
manifest command, and forward the original argument vector once; they do not
modify `PYTHONPATH` or rebuild arguments.

```text
bin\experiment.bat             Device Manager Init -> Task Console
bin\pulse_editor.bat           Pulse Editor
bin\figure_viewer.bat          Figure Viewer
bin\run_server.bat             pulse server only; never builds/programs FPGA
bin\slm_server.bat             sole SLM server owner; DVI default
bin\estimate_resources.bat     installed geometry/resource estimate
bin\build_and_program.bat      build if needed, then program volatile FPGA;
                               build-only/flash remain explicit modes
```

Run experiment windows from the folder owning `pulses\`, `data\`, and
`apparatus.json`, or pass `--workspace`. Set `ZLC_PY_CMD` only when the intended
Python is outside normal discovery.

## One runtime path

Virtual and physical devices use the same descriptor/catalog/NodeHost/session
path:

```text
Calibration Task -> calibration JSON + report images
Camera Measurement -> canonical frames Dataset
Occupancy Processor(frames + Calibration) -> counts/occupied/judged frame
SLM Editor -> strict Target/Science Context -> explicit Send
SLM Feedback(Calibration + Science Context + pulse + exposure) -> completed/stalled Context
Panel -> matching data/fit/overlay -> archive-first Save Fig
```

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

Target v2 stores intensity/objective. Science Context v2 stores the frozen
Target, numeric pupil, Pattern/base phase, operator wavefront, correction
reference and command receipt. Loading/saving artifacts never writes hardware;
only explicit Send or the Feedback task's own confirmed apply establishes a
known command.

Calibration remains an SLM-independent camera/readout artifact and supplies only
registered site BOX geometry to Feedback. Feedback uses one canonical single-
frame Camera Measurement batch per candidate (100 authored shots by default).
Finite single/double Gaussian fits are valid; dark-only sites receive the
authored absolute boost, loaded sites change relatively under feedback gain and
maximum-step bounds, invalid sites hold, and history protects the learned loading
floor. Every next acquisition requires a confirmed different phase. The task
runs all authored updates (12 by default), then keeps the best measured
candidate; it has no built-in uniformity magic-number stop and no hidden retry or
validation batch.

The virtual plant uses one commanded-phase Fourier trap roster, one shared
non-symmetric imaging PSF and a fixed apparatus aberration independent of Target
or grid. With 20 µK cooling, traps below 500 µK do not load; the 520 µK nominal
depth deliberately places ordinary optical nonuniformity near that edge.

### Pulse and FPGA

Pulse cycle count, Camera frames-per-cycle and Dataset repeat are independent.
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

Figure writer emits v2; its one reader migrates the exact supported v1 grammar
and rejects unknown formats. Calibration writer emits
`zlc.calibration.readout/v1`; its owner migrates only the two known unversioned
roots without inventing missing statistics.

The only supported tutorial is
`packages/zlc_workbench/notebooks/usage.ipynb`. It uses the installed product, a
temporary workspace, virtual devices, canonical Camera Measurement publication
and `NotebookView`; it contains no hardware cell, source-path bootstrap, saved
output or execution count.

## Evidence lanes

Automated release lanes run from a fresh wheel outside the checkout:

```powershell
zlc evidence software --repo C:\path\to\Zou_lab_control_v2
zlc evidence gui_offscreen --repo C:\path\to\Zou_lab_control_v2
zlc evidence virtual_vertical --repo C:\path\to\Zou_lab_control_v2
zlc evidence notebook_offline --repo C:\path\to\Zou_lab_control_v2
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

`ARCHITECTURE_DESIGN.md` records final invariants. `IMPLEMENTATION_PLAN.md`
records the current M7 checkpoint, evidence already obtained, pending final
lanes, and explicit experiment-machine acceptance boundary.
