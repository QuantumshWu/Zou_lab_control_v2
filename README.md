# Zou lab control

Neutral-atom experiment control: one repository, one distribution, eight layers.

The current product path uses the same descriptor/catalog/NodeHost/TaskConsole
route for virtual and physical adapters:

```text
Calibration Task -> one result -> calibration JSON + six report images
Camera Measurement -> frames signal
Occupancy Processor(frames + calibration path) -> occupancy data
SLM Editor Pattern -> base phase + optional Zernike -> science phase -> explicit Send
SLM Feedback(calibration + target + pulse) -> grouped qCMOS fluorescence -> accepted phase NPZ
Image/other Plot Panel -> Panel Edit Save Fig
```

TaskConsole and every device Control share the same `Experiment` session,
named devices, virtual world, and sequencer; Pulse and SLM Editors open on
demand from their loaded device cards and do not create a second session or IPC
service. SLM feedback uses the raw occupied-shot site BOX brightness at the 35
calibrated qCMOS sites for direct intensity-ratio correction; empty shots do
not enter the mean and no per-site dark/bright normalization is applied. It
neither treats a single atom's PSF brightness as trap intensity nor fits a
hidden wavefront, and it does not claim to observe unmeasured dense-target pixels.
Its 1% finite-shot gate is not claimed complete until the current raw-BOX,
fresh-loading implementation passes exact qCMOS multi-seed validation.
Insufficient authored budgets fail without retaining a phase; no hidden
virtual truth or relaxed threshold is used to manufacture acceptance. Earlier
shot-count estimates based on depth-dependent loading are obsolete.

The real SLM device type is `slm.hamamatsu_x15213`. **Scan hardware** uses the
same descriptor route as every other device: it can find an official USB SDK
controller and head serial, or offer attached `1280 x 1024 @ approximately 60
Hz` displays as DVI candidates. A DVI candidate is not an identity proof; the
operator confirms which display is physically connected. The Editor has only
**Pattern** and **Wavefront** pages. Pattern keeps the independent target and
pre-correction science-phase plots at the established `2x2 = 490 x 357`
logical size; a shared **Size** selector also controls the independent
Wavefront preview, and scrollable canvases keep larger presets from overlapping
or being clipped. The main page exposes only Input pupil, Zernike, and vendor
correction. The default pupil is a centered Gaussian with a `1/e^2` intensity
diameter equal to 70% of the SLM height; Off uses uniform full-raster solver
illumination. Pattern offers exact Grid, geometrically staggered Checkerboard,
Gaussian, Flat Top, and English/Chinese Text with minimum spacing and an atom
budget. Steering X/Y and Noll Z4-Z11 share the one Zernike switch. Loading or
saving target/science phase does not write the SLM; only **Send to SLM** applies
the science phase.

The vendor correction BMP stays on the experiment computer, can be loaded and
A/B enabled in the Editor, and is composed by the adapter only on the next
explicit Send. A serial-specific profile under `devices/slm/profiles/` supplies
the full calibrated phase curve. Wavelength builds the nonlinear LUT from that
curve, so the displayed 2-pi gray is computed rather than manually authored
(LSH0804382 at 852 nm gives 225). Correction is added modulo 256 before this
LUT, stays at native `1272 x 1024`, and a wavelength-labelled map is converted
without resizing. DVI uses the left 1272 columns and leaves the right eight
zero. Development-machine readback tests are not optical proof; the experiment
machine can directly Scan/configure/Send for final bring-up.

Virtual loading follows the same shot logic the experiment expects: every
cooling rise while the trap output is high independently redraws each active
trap at one base loading probability. Coherent local depth controls active
topology, occupied qCMOS brightness and release survival, but never
exponentially boosts loading; a removed trap loses its atom and cannot
resurrect it merely by reappearing.
Calibration publishes only its current capture preview while
running; after the loop it computes one result, writes its JSON, and passes that
same result to `zlc_plot` for six PNG report images. Workbench does not display
or re-analyse the report. Camera Measurement owns per-run exposure/ROI and uses
`Repeat = 0` for infinite acquisition.

The three TaskConsole save actions are intentionally separate: header **Save
Layout** writes stopped pipeline/layout wiring without data, header **Save
Screenshot** writes one ordinary image of the GUI, and Panel Edit **Save Fig**
writes only that panel's displayed frozen image/data plus its run-time call-chain
metadata and actual device snapshots.

```
bin\install_requirements.bat   once per machine: numpy, matplotlib, PyQt5, ...
bin\experiment.bat             Device Manager Init -> TaskConsole; device Control windows on demand
bin\pulse_editor.bat           the pulse window on its own
bin\figure_viewer.bat          open a saved figure archive; no experiment session needed
bin\update.bat                 git pull, re-check dependencies, prove it still imports
bin\run_server.bat             the pulse server, on the machine wired to the board
bin\build_and_program.bat      synthesise the bitstream and load it onto the FPGA
bin\estimate_resources.bat     what the current board geometry costs on the part
```

Everything a human clicks is in `bin\`, the three FPGA scripts included. They
operate on `packages\zlc_pulse\fpga`, where the RTL, the board config and the
build tree live: being clickable and being owned are different questions, and
only the first one is about where a file sits.

Run them from your experiment folder — the one holding `pulses\`, `data\` and
`apparatus.json`. They are deliberately not passed a workspace: a
double-clicked launcher starts in `bin\`, which is nobody's experiment.

They find Python themselves, and print which one they found: PATH first,
then conda's base and the usual Anaconda / Miniconda / python.org install
folders. Anaconda in particular recommends staying off PATH, and a machine
equipped that way is still a machine with Python on it. If yours lives
somewhere else, say so once and every launcher obeys:

```
setx ZLC_PY_CMD "C:\Users\you\anaconda3\python.exe"
```

## Nothing is installed, and that is deliberate

This checkout is not `pip install`ed.  The code is reached by PATH: every
launcher puts this folder on `PYTHONPATH` and runs
`python -m zou_lab_control_v2 <window>`, and importing that package is what
puts all eight layers ahead of anything else.  A notebook does the same with
one line:

```python
import zou_lab_control_v2   # first, before any zlc_* import
import zlc_workbench
```

The reason is a failure this project has already paid for.  Install the same
names twice -- once from here, once from somewhere else -- and `import
zlc_data` answers with whichever pip wired up, so a change made here appears
to do nothing, repeatedly, with nothing on screen to say why.  So
`zou_lab_control_v2` refuses out loud if a `zlc_*` module was imported before
it: arriving second is exactly the case that used to pass silently.

On the experiment machine the update is therefore just a pull.
`bin\update.bat` does it, re-checks the dependency list, and proves the code
still imports -- before you find out during a run.

## The eight layers, and what each is allowed to know

Each lives under `packages/` with its own `src/`, `tests/`, `docs/` and
`notebooks/`. The order is the dependency order — a layer may know the ones
above it and must not know the ones below.

| layer | owns | may not |
|---|---|---|
| `zlc_data` | what a dataset IS: axes, validity, snapshots, manifests | everything else |
| `zlc_durable` | writing a file so a crash cannot half-write it | domain, plots, Qt |
| `zlc_runtime` | generations, publication, board-coherent ticks | what any signal MEANS |
| `zlc_plot` | drawing, fitting, panel layout | scheduling, devices |
| `zlc_ui` | windows and controls | data, plots, domain — **and no Qt escapes it** |
| `zlc_pulse` | the sequence model, the compiler, the wire | anything above |
| `zlc_atom` | physics: headless foundation plus concrete node/device plugins and their own report or control surface | Qt/plot dependencies in foundation/common/install/framework; Workbench-owned plugin science |
| `zlc_workbench` | composition and wiring, and nothing else | inventing domain or UI |

Two rules are mechanically enforced rather than remembered, because both were
broken before they were checked:

* **Every window is one call and one handle.** `zlc_ui` hands out
  `open_pulse_editor()`, `open_task_console()`, `open_figure_viewer()`,
  `open_device_manager()`; each answers with a handle carrying signals and
  `set_*` methods. Nothing outside `zlc_ui` holds a QWidget, and
  `packages/zlc_workbench/tests/test_gui_seam.py` fails if anything tries.
  Whatever the outside can hold, the outside will assemble — and then what is
  on screen has two owners.
* **Every package is reached through its front door.** No layer imports
  another's submodule; if a name is needed, it goes on that layer's facade with
  its contract and its tutorial updated. The count went 45 → 0.

## Running the tests

```
python -m pytest
```

All eight suites, from the top: a change that crosses two layers is checked by
both in one run, which is the point of merging them.

## Where the code lives

This tree is the product. The eight standalone repositories beside it
(`Github/zlc_data`, `Github/zlc_ui`, …) are where it was built and are kept for
their history; they are not a second place to edit. If `import zlc_data`
resolves to one of them, this checkout is not the one running — which
`python -m zou_lab_control_v2 check` reports explicitly.
