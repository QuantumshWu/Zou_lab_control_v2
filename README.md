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
zlc warm_numba        compile or verify every numba kernel's disk cache
zlc warm_slm_gpu      compile the SLM rearrangement GPU code; choose its analysis product
zlc capture           capture one window on the real screen for acceptance
```

The Windows files in `bin\` are thin shortcuts. Their shared Python resolver
anchors imports to the current checkout, then they select one manifest command
(the two `migrate_*` launchers run their checkout tool instead) and forward the
original argument vector once. `install_requirements.bat` is
the installed-only owner (also used by the GPU installer), so its final check
cannot be masked by source.

```text
bin\experiment.bat             Device Manager Init -> Task Console
bin\pulse_editor.bat           Pulse Editor
bin\figure_viewer.bat          Figure Viewer
bin\estimate_resources.bat     current checkout geometry/resource estimate
bin\build_and_program.bat      build if needed, then program volatile FPGA;
                               build-only/flash remain explicit modes
bin\warm_numba_cache.bat       zlc warm_numba from this checkout
bin\warm_slm_gpu.bat           zlc warm_slm_gpu from this checkout (per GPU and driver)
bin\install_slm_gpu.bat        install/check pinned CUDA dependencies; log progress
bin\migrate_pulses.bat         one-shot offline migration of saved pulses and
                               Config files (originals kept beside them)
bin\migrate_units.bat          one-shot rewrite of saved pulses to one unit
                               spelling (originals kept beside them)
```

### Optional AI / MCP access

MCP attaches to this same experiment window and Session; it never opens a
second hardware connection. Install the optional transport with the **same
Python used by the experiment**, then start it explicitly:

```powershell
python -m pip install -c constraints.txt -e ".[mcp]"
$env:ZLC_MCP_TOKEN = "replace-with-a-long-random-secret"
.\bin\experiment.bat --mcp-port 18864
```

Connect a Streamable HTTP MCP client to `http://127.0.0.1:18864/mcp`, with
`Authorization: Bearer <the same secret>`. The service binds only loopback and
rejects browser-origin requests. Do not put the secret in a tracked file.
Use a client tool/HTTP timeout of at least 40 seconds when using `lab_wait`
(its bounded wait is at most 30 seconds); do not turn transport timeouts into
automatic write retries. Keep the client connection/session for repeated use.
With no flag there is no listener or MCP dependency loaded. The command above
allows AI to operate the experiment by default; add **`--mcp-read-only`** only
when observation without control is wanted. This is
trusted experiment access, not a Python sandbox: existing plugin expressions
retain their ordinary capabilities. Hardware limits, ownership and normal
manual confirmations still apply. The MCP client does not get an independent
device owner or permission to bypass them.

- `lab_state`: one compact overview of current nodes, signals, panels, errors
  and readiness. Use `node_id`/`panel_id` for focused follow-ups,
  `after_sequence` for new messages, and `details=true` for authored values.
- `lab_catalog`: short node index by default; `api_name` returns one plugin's
  schema, `node_id`/`panel_id` returns the current editor projection. Units,
  choices and validation come from the same descriptors as the GUI.
- `lab_files` lists bounded pages of workspace files. A Logic resource field
  states its own `relative_to` directory: for example, a Pulse resource takes
  `imaging.json` relative to `pulses/`, while an editor file action takes the
  workspace-relative `pulses/imaging.json`. Absolute paths inside the workspace
  are also accepted. Scan Plan and API override formats are documented in the
  existing Seamless Scan fields, not in a second MCP-specific grammar.
- `lab_signal`: bounded numeric pages with exact generation/revision,
  validity, uncertainty and logical-axis coordinates. Default reads the latest
  publication chunk; `dataset=true` reads the current dataset/retained window.
  Page offsets refer to physical storage. Live pages can advance between reads;
  use saved Figure data when a whole fixed dataset is needed.
- `lab_screenshot`: current Console/Devices window, or an opened `device_id`
  Control, including its active modal. PNG is returned directly. It captures
  the actual current UI without re-rendering the experiment or other apps.
- `lab_wait`: wait for the specified node generation to become ready/terminal;
  the GUI and Stop remain usable. A timeout does not stop or retry the run.
- `lab_action`: the normal Logic/Panel operations, plus `view.monitor` and
  `view.logic` for direct navigation. For example, `logic.add` takes
  `{"api_name":"camera_measurement","values":{"repeat":1,"frames_per_cycle":3}}`.
  Read the current draft revision before Start; use the observed generation
  for Stop. A figure save selects `.png`, `.pdf` or `.svg` and also writes its
  companion science `.npz`. Check subsequent reports for asynchronous errors.
  `layout.save`/`layout.load` take a workspace JSON `path`; loading restores
  stopped drafts, never starts an experiment. Replacing active Logic requires
  explicit `stop_running=true`; `layout_pending` reports pending replacement.
- `lab_initialize`, `lab_device_action` and `lab_pulse`: explicitly authorized
  Init, ordinary Control edits, and bound Pulse Editor Open/Fire/Stop. Prepare
  the Pulse before arming a triggered camera; waiting indefinitely for an
  external trigger is not guaranteed by an acquisition's timeout. Fire keeps
  the Pulse's authored repeats and Config semantics.
- `lab_device_state` / `lab_pulse_state` read already-open Controls without
  issuing hardware reads, including in read-only mode.
- `lab_editor_catalog` / `lab_editor_state` / `lab_editor_action` expose the
  existing specialized Pulse and SLM editors after opening their Control.
  The catalog supplies valid action names, field IDs, units and arguments.
  Pulse actions edit periods, TTL/DAC, delays, repeats, brackets and bindings;
  named Config edit/load/save and explicit Pulse save share the GUI paths.
  `validate` compiles asynchronously without Load or Fire. Detailed state
  distinguishes authored defaults, loaded Config, unsaved Config, compiler
  preview and device-confirmed digest; never infer physical timing from draft
  period lengths alone. Dirty Pulse/Config replacement requires explicit
  `discard_changes=true`.
  Config-covered values and file-backed Subpulse contents follow the same
  read-only rules as the GUI; switch to Default or edit their source instead.
  `subpulse.*` actions select, group, bind sources, edit, insert and save using
  the existing Subpulse owner. File paths are explicit; an empty source path
  clears the reference. State distinguishes embedded defaults from the selected
  file draft and its save target/dirty state. Use the catalog for arguments,
  including `bracket_count` bindings; editor revision also covers Subpulse
  selection and draft changes. State inspection does not reread source files:
  use `config.refresh` or `validate` when updated saved sources must be checked.
  SLM actions edit Target/pupil/operator, solve, Send/Adopt, and load/save Target
  or Science Context. Inspection does not solve or Send, and command receipts
  do not claim optical settling. `expected_revision` rejects stale editor
  edits; SLM revision includes both target and operator/phase changes. Large
  phase arrays stay in their normal artifacts rather than routine JSON replies.
- `lab_artifact` reads workspace reports/previews; `lab_health` distinguishes
  the protocol listener from GUI responsiveness and does not claim hardware
  readiness.

Accepted/queued is **not** started, ready, successful, saved, or physically safe.
Inspect the exact run's terminal error and its data validity. A timed-out queued
GUI request is cancelled before execution; an already-started request with no
reply has an unknown outcome and must not be automatically replayed. Disconnect
does not stop, restart or take over an experiment. Human physical actions stay
human. High-frequency acquisition, Pulse and SLM playback stay entirely local;
AI/MCP is between experiment decisions, not between frames or shots.

For a hardware-free first trial, add `--template virtual --workspace <an empty
existing directory>` to the launcher, then use `lab_initialize`. No real device
is opened by that template. Test screenshots and artifacts belong in that
workspace, not in Git.

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

The optional numerical API lives in `zlc_atom.devices.slm.solver`. On Windows,
double-click `bin\install_slm_gpu.bat` to install the manifest's pinned
`cupy-cuda12x[ctk]` dependencies through the existing product installer. It shows
progress, preserves the original error and exit code, writes
`%TEMP%\zlc-slm-gpu-install.log`, and checks the selected interpreter, GPU,
a real dot product and FFT. It does not install or update the NVIDIA driver.
For an explicitly selected Python, the equivalent constrained install is:

```powershell
python -m pip install -c constraints.txt -e ".[slm-gpu]"
```

Ordinary plugin discovery, the static SLM Editor and Feedback do not import CUDA
or silently change backend. An import failure identifies the interpreter and
original cause; a CUDA device/driver failure remains a separate error.

`prepare_rearrangement(...)` prepares fixed source/target coordinate rosters,
the source optical state and bounded GPU/pinned-host resources before imaging.
Coordinates are Y,X pixels in the native centered Fourier image, not camera
pixels or micrometres. The same full pupil and optical operator are used for
synthesis and checks; device orientation, vendor correction and wavelength
mapping remain with the physical SLM owner. A supplied source phase preserves
the actual starting command. Preparation time and memory are separate from
online latency.

The caller supplies available integer indices in the original source roster.
The planner matches `min(available sources, target sites)` and minimizes the
longest actual path, detours included: surplus sources are discarded, while
shortage fills a subset and reports the unfilled targets. Camera classification
and target selection belong to the concrete Task, not the solver.

```python
plan = plan_rearrangement(prepared, available_source_indices)
sequence = compute_rearrangement(
    prepared, plan, motion_frames=16, support_tolerance=1.01,
    require_converged=False,  # report maps outside a gate instead of raising after the sequence
)
# Assigned paths produce N read-only maps; empty occupancy produces none.
# sequence["gate_misses"] marks the maps outside a gate.
if len(sequence["phase_codes"]):
    slm.prepare_phase_sequence(sequence["phase_codes"], 1 / 60)
    receipt = slm.play_phase_sequence()
prepared["close"]()
```

The Task uses the same solver with `frame_ready(index, readonly_frame)`:
each map is submitted as soon as the GPU has finished it and its codes are on
the host. A map outside a gate is still submitted and reported; only a
computation failure (Stop or a non-finite field) stops submission. A private
two-frame uploader queue and a two-frame device queue overlap later computation,
mapping/upload and local paced playback. The complete caller-owned read-only
movie remains available for archival; published frames are never overwritten.
The streaming device API is `prepare_phase_sequence(None, interval,
frame_count=N)` followed by ordered `submit_phase_frame(index, frame)` and the
existing playback call. The Task starts playback on the first map and skips
preparation/playback for empty/no-change results.

`motion_frames=N` is the total displayed map count: it includes the endpoint
and excludes the already displayed source. Unselected source light fades within
these same N maps; there is no extra removal prefix. Empty occupancy is a
zero-match result with no new phase playback. Paths may cross in space at
different times, but the actual continuous segments between displayed positions
must satisfy the minimum-spacing constraint; unsafe shortcuts are rejected,
not repaired by silently adding maps.
If straight paths conflict, bounded equal-cost pair swaps, movement ordering
and local waypoints are checked. The report distinguishes the assignment lower
bound from the actual routed distance; this is not a globally optimal
collision-constrained path planner. No valid schedule found does not mean
physical impossibility.

Every map is checked against three gates, all reported in `gate_misses`:
kept-trap uniformity (weighted max/min, Motion and Final tolerances below),
surplus-atom sites on the removal map and the next one (at most 0.2% of that
atom's initial trap) and stray light (at most 20% of the current weakest kept
trap on intermediate maps, 5% on the last map). Kept traps need not stay as
deep as initially: each trap's power relative to its initial trap
(`brightness_minimum_to_initial` and its maximum and mean) is reported, not
gated. The gates judge the displayed maps; the liquid-crystal transition between
two maps is not gated (its measured range is given with the Task below). The
algorithm and what the stray-light number means are described there too. Phase
steps and movement per map are reported alongside. No numerical gate proves
atom release, moving-well shape, liquid-crystal response or survival.

Optimization removes work before making the remaining operations faster:
continuous clearance uses conservative pair-distance and path-box bounds, with
the same exact segment formula for candidates; preparation builds only the
bands its geometry can use. There is no GPU-name, trap-count or chosen-grid
shortcut. Held output buffers remain independently owned across another
computation and workspace close.

Photo-to-map times on an RTX 5070 Laptop GPU (full 1024 × 1272 maps, Fixed
frame count; from the occupancy list to the first and to the last map on the
host, planning, sampling and computation included; this solver and commit
`de73d721` alternate in one process):

| Case | Maps | Photo → first map | vs `de73d721` | Photo → last map | vs `de73d721` |
| --- | ---: | ---: | ---: | ---: | ---: |
| Recorded load (run 14, 30 atoms), 35 sites → 16 | 15 | 5.1 ms | 0.60 | 17.5 ms | 0.52 |
| Recorded load (run 8, 25 atoms), 35 sites → 16 | 15 | 5.9 ms | 0.65 | 15.6 ms | 0.45 |
| 400-site lattice → 100, 55% loaded | 28 | 15.2 ms | 0.42 | 36.8 ms | 0.42 |
| 400-site lattice → 100, 86% loaded | 15 | 20.0 ms | 0.43 | 39.0 ms | 0.59 |
| 400-site lattice → 100, 86% loaded, two maps | 2 | 20.9 ms | 0.48 | 25.3 ms | 0.64 |
| Irregular 120 sites → 40, 60% loaded | 46 | 6.2 ms | 0.45 | 33.8 ms | 0.37 |
| 400-site lattice → 100, 20% loaded (partial fill) | 87 | 8.9 ms | 0.19 | 69.7 ms | 0.35 |
| Run 14 onto a central 3 × 3 target | 15 | 4.4 ms | 0.74 | 19.5 ms | 0.90 |
| 13 atoms on the run 14 sites → 16 (partial fill) | 16 | 5.0 ms | 0.78 | 20.2 ms | 0.60 |
| Run 14, central 3 × 5 target already full | 16 | 5.1 ms | 0.86 | 19.8 ms | 0.97 |
| Run 14, filled target, one removal map | 1 | 4.5 ms | 1.01 | 4.5 ms | 0.97 |

Ratios are medians of paired rounds (16 for the first seven rows, 12 for the
others) with the GPU about 25–45% busy with other work. The last map's stray
light is measured on its delivered codes after it has been handed over (about
4 ms on this GPU), so `compute_rearrangement` returns that much after the last
map. With **Maximum camera step** the movement is sampled before computation,
which adds 1–3 ms to the first map; the already full target and the single map
then take about as long as with `de73d721`, which does not darken their surplus
sites.

Replays of the 14 recorded lab loads (on 16-, 9- and 4-site targets and on 1 × 2
to 2 × 2 targets), 400-site lattices loaded 20–95%, irregular and partially
filled arrays and weighted End Targets (10,591 maps, each scored by independent
propagation of its delivered codes) put 0.2% of the maps outside the
uniformity, surplus-site or stray-light gate. No intermediate map exceeds the
20% stray-light gate; how often it stays within 5% of the weakest kept trap
depends on the plan:

| Plans | Intermediate maps within 5% stray light |
| --- | ---: |
| Recorded loads onto 16-, 9- and 4-site targets | 95%, 100%, 98% |
| 400-site lattice loaded 55–65% | 98–99.5% |
| 400-site lattice loaded 75–80% | 95% |
| 400-site lattice loaded 86% | 86% |
| 400-site lattice loaded 90% / 95% | 66% / 33% (up to 12%) |
| Weighted End Targets | about 90% |
| Already full targets that only lose surplus atoms | 99.5% |
| Two or three kept traps (1 × 2, 2 × 1, 1 × 3 targets) | 23% |

With only two or three kept traps the phase-only hologram puts 5–11% (up to
14%) into the neighbouring lattice sites on most maps.

Already full 2 × 2 to 5 × 5 targets that only lose surplus atoms (8,190 maps,
1–16 maps per plan) have no map outside a gate, and with two or more maps no
liquid-crystal transition after the first one takes a kept trap below its
initial depth; a single map, which **Maximum camera step** chooses for them,
starts from the prepared destination, and the transition from the photograph can
take a trap down to 3–60% of its initial depth. Such plans with weighted End
Targets miss about 6% of their maps (intermediate uniformity up to 1.18; with
weights drawn from 0.25–1 the last map's weakest trap is up to 13% shallower
than requested), and with two to four kept traps about 7% of their last maps
reach 5.3–5.6% stray light. Partially filled last maps (449 plans) stay below
4.7% stray light, below 3.0% with uniform weights. Removal maps that are also
the last map on 400-site and irregular arrays loaded 60–95% (332 maps, one or
two maps per plan) stay within their gates except one at 5.1% stray light. Small
targets leave a few maps far outside their gates, all reported: 0.8% of the 1 ×
3 transport maps reach uniformity 1.14–1.70, 4% of the crowded next maps leave a
surplus-atom site at 1–8.6 times its gate, and a 2 × 2 removal map with one or
two surplus atoms reaches 1.17 and four times its gate; in all 0.45% of the
small-target maps. On about 50,000 replayed maps the reported `gate_misses`
match the independent scores except that one removal map at 5.1%, reported at
3.6% because its 64 watched points were full; reported last-map stray light is
0.94–1.00 of the whole plane's on a filled last map and typically 0.96–1.00
otherwise, as low as 0.55 when a peak lies between native samples.

**Frame count** selects fixed total maps or **Maximum camera step**. Fixed mode
uses **Movement frames** (default 16); camera-step mode uses the maximum 2D
Euclidean movement in **original sensor pixels**, not Fourier bins or binned
image indices. It uses the existing Calibration/ROI/binning conversion, retains
each path waypoint, and computes the actual count after occupancy is known.
The inactive input retains its value. Both policies include one initial
source-removal map when needed before moving; it is not an additional frame. Automatic
duration is therefore pending until matching. An infeasible Pulse budget is
reported, not repaired by skipping frames or enlarging the requested step.

At the fixed defaults, 16 maps / 60 Hz is about 267 ms nominal. Computation,
upload and playback overlap; startup stalls extend the measured window. Authored
time slots do not add a profile tail wait. **Motion intensity tolerance (%)**
(default 10%) applies to all but the last map; **Final intensity tolerance (%)**
(default 1%) applies to the last map. These are the weighted maximum/minimum
ratio minus one. The report keeps these computed gates and, after the photographs, checks the
whole focal plane of every played map with fourfold zero padding: its strongest
peak farther than 2.5 Fourier pixels from every current kept trap, over the
weakest current kept trap, is judged against the same 20%/5% as
`focal_plane_accepted`, kept apart from the computed `quality_accepted`. Common trap-depth changes due to redistribution are allowed;
uniformity is not an atom-survival guarantee.

**Rearrangement Measurement** (`rearrangement_measurement`) integrates the same
solver/playback into ordinary camera acquisition. Select Camera, SLM, Calibration,
initial SLM Science Context and optionally an End Target. Each cycle's first
frame starts internal classification/rearrangement; its last frame restores the
initial Context. Use at least two frames per cycle; the last frame need not be a
verification image. Only the normal `frames` signal is published, after each full
cycle. Repeat 0 monitors continuously; finite Repeat works like Camera Measurement.
The node does not load or fire a Pulse. It can be selected as an Acquisition logic;
the operator's Pulse must leave room for movement before the last frame and for
restoring the initial SLM state before the next cycle. Source restoration uses
the device's ordinary single-image apply, including its configured settling time.
A too-early cycle boundary cancels the move, restores source and reports the error,
instead of applying stale occupancy in the next cycle.

Repeated Start reuses compatible GPU preparation, not old occupancy or results.
Files and Pulse inputs are reread each time; changed optical inputs rebuild the
working point. The solver retains one exclusively borrowed working point until
process exit, an optical-input change, or a CUDA failure; Stop and removing a
node do not discard healthy preparation. Task and Recording start fresh runs and
can reuse the same working point. The source phase is still re-established and the camera
armed for every experiment. A minimum longest move is not a minimum playback time.
Passing geometric clearance does not validate atomic transport. Pupil center
and illumination must match the real beam, and phase
continuity must be interpreted relative to the physical optical axis. A configured
60 Hz cadence is not a measured liquid-crystal optical response.

Actual Config-filled periods and nested loops determine the conservative
verification deadline. Camera reception continues independently during GPU/SLM
work. Late playback or an evidently early verification photo is rejected;
a host receive timestamp is not a physical exposure timestamp. The physical
SLM owner plays submitted frames locally as they arrive, confirms every logical
step, and applies remaining final settle only once. Repeated holds reuse a prior
confirmation rather than inventing new display ACKs. Update both client and server
for sequence protocol v2. DVI acknowledgements prove software rendering,
not vblank or liquid-crystal settling.
Queue admission/upload acknowledgement is not display acknowledgement. Missing
frames or slow acknowledgements extend actual cadence; frames are not skipped.
If a later map's computation fails (Stop or a non-finite field), playback is
cancelled and joined: earlier maps may already have played. The report keeps the
actual confirmed prefix and partial outcome, not an assertion that nothing was
shown.

The three automatic previews are before photo + occupancy, after photo +
occupancy, and source/final phase. In fixed-frame mode the **trajectory** signal stores full ordered
X,Y positions by step and source identity, including the initial position;
it is not automatically plotted as Y against frame. Camera-step mode does not
publish guessed live trajectory/quality axes before their size is known; both
remain available as the same saved typed Figure/NPZ artifacts with actual N. Saved
`figures/trajectory_2d.npz` + `.png` overlay commanded paths on the actual before
photograph, in its camera coordinates. They retain the same complete SiteMap,
status, labels, image display and site-spacing-based sizes as the before/after
photos. Existing Target-to-camera registration and the common image-coordinate
conversion project the generated Fourier paths; no second path is planned.
A thinner path ends in one arrow drawn short of the destination centre, without
an extra endpoint dot; the numerical endpoint is unchanged. Site labels keep
their ordinary positions and font. Same-status site/path/head geometry is unioned before applying alpha once,
so joins and overlaps do not darken. There are no per-path colors or endpoint
boxes. A shared frame range and compact exceptional-order tags keep text sparse.
The Figure keeps the true before-image Dataset and typed ordered camera XY
vertices, including holds and backtracking, so FigureViewer can redraw it.
Unselected sites keep their original static markers, without movement arrows.
Original Fourier coordinates remain in the trajectory signal and numeric
archive; `motion_camera_xy` records the registered camera projection.
These are commanded trap paths, not measured atom tracks, and geometric
registration alone does not independently verify optical handedness.

After verification or failure, the run saves `summary.json`/`summary.txt`,
`data/rearrangement.npz`, both available photographs and
counts/occupied/validity/thresholds, source/end Target and frozen input facts,
assignment and paths, actual Pulse/device receipts, and exact phase codes when
**Save phase sequence** is enabled (default on). Important Figures are typed
NPZ + PNG pairs. No extra final Science Context is written. Report rendering
and movie archival stay out of the photo-to-playback critical path, and so does
the live display: the before photograph is published to the window only once
playback has finished.

**Shots** (default 1) fires the Pulse that many times in one Start, each shot
photographing, rearranging and verifying on its own from a fresh source phase.
With more than one, every shot keeps its atom-by-atom data in
`data/shot_NNN.npz`; the first completed shot (and any unfinished shot before
it) also keeps the phase sequence, the optical diagnostics, the figures and the
whole-SLM arrays; the other shots keep no whole-SLM array. The source and end
Target images are saved as their lit pixels, and a completed shot's final phase
only as the last frame of its saved sequence. A failed shot (a missed deadline included) is recorded and the
next one fires; a board that cannot be made safe, or Stop, ends the batch.
`summary.json`/`summary.txt` then sum the moving and stationary destinations
judged and occupied over the shots whose verification was accepted, with
1-sigma Wilson intervals, and give each timing's median and range.

Preparation, readout, matching, complete calculation/host transfer, upload,
playback/final settle and saving are separate timing windows; nested or
overlapping windows must not be added. First host frame handed to playback
(`first_frame_ready_for_playback`), first display acknowledgement, callback time
(including backpressure) and total completion are reported separately.
Stop/failure saves partial evidence
and the last confirmed phase; unknown device outcomes remain explicit.
Target filling is not per-atom identity-tracked survival.

The report separates moving and stationary assigned targets, their judged and
occupied endpoints, and deliberately discarded atoms. Unknown or unaccepted
verification is not counted as transport success. Requested frame rate is
separate from observed logical-step/new-presentation cadence, photo receive
and camera timestamp intervals, and the complete online window; these summaries
derive from existing receipts rather than a new timing mechanism.

Numerical/RPC/Task and public Figure tests cover played prefixes, cancellation,
early upload EOF, retained output ownership, and exact static/dynamic overlays.
The final combined native virtual Task flow passed with actual GPU computation,
16 ordered logical confirmations, two valid photographs, exact final phase and
matching photo/path identifiers. Fresh FigureViewer reopened that saved result.
All test windows and render children were closed. Sparse and 100/400-site 4×4 path figures were checked
on screen and on save/reopen, with all source IDs retained and no measured text
overlap/clipping. Compact 2×2 views remain pixel-limited for exceptional timing tags;
use the existing size/zoom controls. Full Agg overlay drawing still costs about
32/181 ms for 100/400 paths in the measured report-view cohort; it is outside
the photo-to-playback path, not a new fast scientific renderer. Experimental
SLM optical response and atom-survival acceptance remain pending.

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

Camera Measurement owns its frames-per-cycle and repeat. Exposure and ROI left
blank keep whatever the camera is set to now (not the whole sensor); a stated
value is set on the camera and stays there until another run states another.
Pulse timing does not infer or validate camera exposure. Real and virtual adapters
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
FPGA build root. The pulse server never programs hardware; the default
`build_and_program` path builds/reuses a valid project and then programs volatile
FPGA state, while flash is always explicit. Timing/build evidence and the
remaining board acceptance boundary are recorded in `IMPLEMENTATION_PLAN.md`
and `packages/zlc_pulse/fpga/README.md`.

## Persistence and notebook

Figure uses stable `zlc.figure`; Calibration uses
`zlc.calibration.readout`; Pulse uses `zlc.pulse`; Target uses `zlc.slm.target`;
Science Context uses `zlc.slm.science-context`. Readers accept only the current
complete grammar and never convert what they read: a workspace file outside it
is refused, and the `bin\migrate_*` launchers are the explicit one-shot
converters. Formats have no alias or numeric version.

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
