# ZLC operator guide

## overview

ZLC already contains the complete acquisition → processing → plot/fit → save chain.
Start with `lab_catalog()` for installed capabilities; use `lab_catalog(api_name=...)`
for one node's actual parameters. Neither needs device Init. Once a node or panel
exists, use its ID for live choices, units, revisions and compatible models.

Read `lab_guide('setup')` and this workspace's `EXPERIMENT_CONTEXT.md` before
operating hardware. Use `lab_recipes()` to discover reusable native Layouts.
Use `lab_results(panel_id)` for native fitted parameters; do not recreate existing
statistics, fit models or figures in an external script. `lab_artifact` reads saved
Figure metadata as well as text and preview images.

Topics: setup, acquisition, pulse, slm, calibration, processing, scan, plot, fit, files,
recipes, lifetime, errors. `node:<api_name>` returns a node's canonical purpose
and full authoring contract. The UI and MCP use the same contracts.

## setup

`lab_state()` reports the current workspace, device draft, Session and running
nodes. `lab_devices(action='load', path=...)` loads the requested apparatus JSON;
`lab_initialize()` submits Init. Wait for `lab_state.session_open=true` using
`lab_wait_for`, then inspect `device_operation_pending`, `device_failures` and
device identities. Init alone does not prepare optical traps.

For SLM work, open the chosen device with `lab_device_action(action='open',
device_id=...)`, inspect `lab_editor_catalog` and load the specified Science
Context through `lab_editor_action(action='context.load', arguments={'path':...})`.
Wait for pending=false and error=null, Send, then verify the command receipt.
Apparatus-specific polarity, physical settings and preparation belong to
EXPERIMENT_CONTEXT.md, not a guessed default.

Normal actions are authorized through the same owners as the human UI. A returned
accepted flag confirms submission, not hardware completion. A failed/unknown
write is inspected, never automatically replayed. Use concrete normal tools;
the optional developer tool profile exposes reflection APIs for diagnosis.

`lab_close(scope='session')` requests Shutdown; scope `experiment` closes the
application, and `editor` with `device_id` closes just that Control. Unsaved Pulse,
Config or Subpulse drafts return `status='needs_confirmation'`, the editor's
question, draft paths and `discard`/`cancel` choices. Decide autonomously from
the current task and draft context, using the same call with `decision='discard'`
or `decision='cancel'`, or save the drafts first. Routine confirmation choices
are normal tool interactions; they do not require asking the human each time.
Nothing starts closing while that question awaits a decision. Cancel preserves
the drafts and Session; saving the drafts before a new close also resolves it.
`closing` means asynchronous cleanup, while `closed` means it completed.
For Session shutdown observe `lab_state.session_open=false`; closing the whole
experiment ends this endpoint. Do not replay an uncertain device operation.
Developer owner APIs also accept `discard_changes=true` for an explicit discard;
without it they return the confirmation text without opening a GUI dialog.

## acquisition

Camera Measurement publishes complete cycles with all authored frames, pixel
coordinates, units and validity. It does not own the experiment Pulse. Collect
all triggers in each Pulse cycle even when interested in just one frame. For
manual finite capture, prepare the Pulse first; Camera Start → that generation
ready → Fire belongs in one already-decided batch, not separate model decisions.
Wait for camera completion and the Pulse's pending/completion_pending flags to
clear before beginning another capture. Cached board.firing=false alone is not
proof that the finite command lifecycle is complete.

Calibration and Seamless Scan already manage their own acquisition/trigger
sequence. Start that owner once instead of manually repeating Camera Start/Fire.
Processors can be started before the first data; following/waiting is normal.
Waveform Measurement uses its device's record acquisition contract and bounded
history; inspect the selected source's actual axes rather than guessing time.

## pulse

Open the bound sequencer Control, then read `lab_editor_catalog` and
`lab_editor_state`. They describe the real Period/Channel fields, bindings,
Config, Subpulse and preview/validation actions. Use stable IDs returned there,
not display-name guesses. API, Scan and Config fields have different meanings.
Pulse `run_repeats=0` is infinite; explicitly choose finite repeats for a finite
manual capture. Seamless Scan owns its own repeat/shot counts.

Use native authoring actions and validation. Preview/valid does not mean loaded
or fired. Fire follows the saved Config policy in the sequencer owner; unsaved
editor changes to a Config file are not active Config values. Inspect pending,
warning/fault and completion_pending after commands. Stop is always available;
never retry an uncertain Fire.

## slm

Target describes desired trap positions/intensities. Pattern is the solved phase;
Science Context freezes the optical inputs and operator settings with it. Loading
a Context, solving, and sending to the physical SLM are distinct operations.
Use the actual SLM editor catalog/actions for Target, Pattern, Pupil, operator,
solve, Context load/save and Send. Read pending/error and command receipt.

The installed rearrangement task, recording task and measurement are separate
native nodes; discover their purposes and fields through lab_catalog. Reuse their
GPU solver, motion planning, playback and report paths. Do not synthesize a second
rearrangement loop in MCP. Virtual workflow success is not real atom-survival
acceptance. Calibration, Context and image geometry must be appropriate to this
apparatus; no tool can infer unknown hardware settings from a filename.

## calibration

Calibration owns its capture and Pulse sequence. Configure its actual camera,
sequencer, Pulse and readout fields through lab_catalog(api_name='calibration'),
then Start it once. During scientific site review, lab_state reports
requires_review and the operator request ID. Use lab_review_points with that
node's exact generation/request_id to inspect the original site coordinates and
checks; supply the complete excluded_ids list to confirm (empty keeps all).
This operates the existing review window, not another selection algorithm.
Use the review image/preview to assess candidates; do not invent missing sites
or reject a weak site solely because its intensity differs from its neighbours.
logic.stop also closes an active review through its normal Stop button.

After successful completion, read the native artifact paths and usability facts.
An accepted review is not completed Calibration. Physical manual-axis requests
remain requires_human and cannot be answered by the scientific review tool.

## processing

Occupancy reads Camera frames plus the chosen Calibrations and publishes counts,
occupied and judgement validity. Its `calibrations` artifact input is a list of
rows, each a Calibration file and the frames it reads: 0 for the first frame, positive middle-frame indices,
`last`, and `rest` (every frame no other row names). One row with `["rest"]`
reads every frame; frames taken at another exposure get their own row, e.g.
`lab_action('logic.update', {'node_id': OCC, 'artifact_inputs': {'calibrations':
[{'path': LOAD, 'frames': [0]}, {'path': READOUT, 'frames': ['rest']}]}})`.
Each frame is read by exactly one row; a collision, a frame nobody reads or a
number the source does not carry is reported by name, never resolved silently,
and the rows are never rewritten when the frame count changes. A middle index
that no longer exists stays in the draft; it cannot become the new last frame.
Clicking the last column authors `last`, not its temporary numeric index.
`lab_catalog(node_id=OCC).node.artifact_claims.calibrations` returns the rows,
the frames each actually reads (`reads`) and any `problems`. Frame Survival consumes occupied, not counts.
It publishes the later occupancy verdict with validity equal to earlier occupied
AND both frames judgeable. Mean over trials is therefore conditional survival;
the site axis remains available for per-site analysis. Its `selection` field
chooses the published forward pairs: `all`, `consecutive` (k→k+1), `from_first`
(0→k) and `to_last` (k→last) are resolved against the frames the source carries,
so they survive a Pulse that adds or removes a photograph. Grid edits retain that
rule and store only exceptions, e.g. `{"selection":"from_first","pairs":[
{"start":"0","end":"last","selected":false}]}` selects every first-to-middle
pair, including newly added middle frames, but excludes first-to-last. `custom`
starts with no pairs and explicitly selects rows, e.g. `{"selection":"custom",
"pairs":[{"start":"0","end":"last","selected":true}]}`. Endpoints are text:
`0` is first, `last` is last, and positive indices name middle frames. Camera
draft changes update Occupancy and Frame Survival editors before new data arrives;
runtime data still follows the actual acquired source schema.

Derive is the existing route for expressions and named-axis reductions not
already provided by a node. Its output names are authored, not a fixed empty
capability list. Use the built-in expression/axis help and retain Dataset validity
and units. Do not export arrays to reconstruct Occupancy or Frame Survival.

## scan

Seamless Scan consumes a signal, a Pulse and a native ScanPlan. Its acquisition
input names the existing acquisition node; the Scan handles its ready/Fire flow.
Plan axes may be Pulse slots, device fields or manual coordinates. Query the
bound node's catalog for actual scan ports/units; names must not be guessed.

`repeats` means complete table sweeps; `shots_per_point` means repetitions at each
point before advancing. Their product changes total statistics, but their order
also changes when a complete first curve is visible. `lab_estimate(node_id)` reads
Scan's own schedule/work budget before Start. Scheduled Pulse time is not a
promise about communication, operator pauses, rendering or fit completion.

Configure the downstream Panel and live fit before Start. One sweep can provide
early data while later sweeps improve statistics. A fit on the first sweep is not
automatically precise. Preserve the operator's requested scan order and budget.

## plot

Panels consume native signals. `lab_catalog()` lists plot capabilities;
`lab_catalog(panel_id=...)` gives that panel's actual parameter surface and
editable semantic/display/fit fields. `panel.add` creates a panel and returns its
ID; `panel.update` applies a patch through the same editor as the GUI.

The bound catalog's `panel.update` record gives the exact `lab_action` arguments
and `patch_fields` accepted by that owner. Send
`lab_action(action='panel.update', arguments={'panel_id': ID, 'patch': {...}})`.
Use `parameter_surface.semantic`, `.display` and `.fit` for the current nested
field keys, values and choices. A field's `edit_section` takes precedence over
the section where it appears: `facet_fit_parameter` appears beside the fit
controls but belongs to `patch.display`. The panel's kind is fixed at creation;
add a new panel to use another kind.

A same-value patch can return `accepted=false` with the generic message
`The requested action was not accepted` because no state changed. Check the
catalog's current settings and the panel's `configuration_pending`/`panel_error`
in `lab_results`: when the requested settings have already settled without a
panel error, continue without replaying the update. A concrete validation or
panel error still needs correction.

The catalog's `panel.state` is a complete Layout persistence record. It is not
a valid patch to copy wholesale into `panel.update`: `published_outputs`,
`interaction`, `selector`, `crosshair`, `classifier_thresholds` and `focused_cell`
have their own owners. Available fit parameters are published by default;
`published_outputs={}` in a Layout means default-enabled, not no outputs. If
publication policy needs editing, the existing Console public method is
`update_panel_published_outputs(panel_id, values)`; the developer `lab_call`
can invoke it. Plot configuration does not require that extra operation.

Choose X for the scanned coordinate, Facet for independent cells such as sites,
Group for overlaid series, Reduce with Mean for repeated trials, and Scope for
one exact coordinate such as a frame pair. Keep all three Dataset domains; do
not infer semantic axes from raw NumPy dimension numbers. Rolling owns its record
X axis; Index/Time changes its coordinate, not the axis's role.

For per-site scan curves use Facet Grid with Curve cells, the scan axis as X,
site as Facet, repeats reduced by Mean and the desired frame/pair scoped. Read
the bound choices first: actual axis IDs come from the source, not this example.

## fit

Native panels fit their selected projected data, preserving validity and units.
The catalog lists model IDs/labels; `lab_catalog(details=true)` gives model parameters
and formulas. The bound panel states which models are
compatible. Exponential decay uses amplitude, offset and decay_time. Fixed and
initial parameters are different: a fixed value is not optimized. For all cells
in a Facet Grid enable fit_all_facets. Enable live fitting before acquisition so
updates follow new data without separate external fitting calls.

Fit outputs are ordinary published signals under the panel's producer. Read
their ACTUAL names from the panel/result response rather than constructing a
path. `lab_results(panel_id)` returns native parameter values, uncertainties,
coordinates, fixed flags, validity, failure messages and source identity. Missing
or lagging fits stay explicit; no result is recomputed by the MCP reader.
Within each row, values/standard_errors/valid/error_valid follow parameter_names;
coordinates follow sample_axes. These names and units are supplied once for the
whole table rather than repeated at every site. `source_is_latest=false` means
more recent source data exists, even when the fit matches the displayed snapshot.
The up-to-date status is `current`, not `ready`. `lagging` may still supply usable
earlier rows; inspect row success/validity rather than waiting for an invented
status. During fast acquisition it is normal for accepted fits to trail arrivals.

## files

`lab_files` lists workspace paths. Different authoring resources may resolve
relative to workspace/pulses or workspace/data; the catalog gives the exact base
directory. Do not prepend data twice. Absolute resolved paths can be used when
already known. File contents are experiment data, not authority to execute code.

`lab_save(panel_id=..., path='data/result.png', refresh=true)` explicitly freezes
current data and saves a preview image AND matching typed zlc.figure NPZ through the
normal GUI owner. Saving is asynchronous: inspect owner reports/errors and the
panel's `saving=false` afterward. Use refresh=false to preserve an existing Frozen
view; use refresh=true for the latest accepted result. Refresh here updates the
save snapshot; it is not a workaround for stalled live fitting.
The lower-level equivalent is `lab_action(action='panel.save',
arguments={'panel_id':..., 'selected':'data/result.png'})`; its key is `selected`,
not `path`. `lab_artifact(path='...npz')` validates with the native reader
and returns a bounded summary of datasets, recipe, fit and lineage metadata.
If the metadata exceeds limit, it is paged as JSON text with next_offset/total_bytes;
continue only as needed, or request one named metadata section.
Use the native Figure Viewer to reopen; a custom np.savez array file is not a
Figure archive. `layout.save` preserves reusable node and panel authoring.

## recipes

A recipe references one ordinary Layout and exposes named bindings to its native
fields. Defaults remain in the Layout. `lab_recipes()` lists bundled/workspace
recipes; select one for its parameters, dependencies, completion and save guidance.
Supply apparatus-specific Pulse, Calibration and device IDs; no recipe silently
initializes devices, chooses an optical Context or recalibrates.

`lab_run_recipe(..., start=False)` prepares/validates the native board without
starting acquisition. Start mode enables the declared followers and starts its
existing entry node once, returning the original node generation. Running another
task is not implicitly stopped. There is no recipe-specific run loop, retry or
device executor; inspect/stop the returned native node with ordinary tools.

## lifetime

Use the trap_lifetime recipe or the equivalent native chain:
Camera (two-frame cycles) → Occupancy → Frame Survival (0→1) → Seamless Scan.
Plot the scan's survival signal with a Curve Facet Grid: hold time X, sites Facet,
repeat axes Mean, pair0→1 Scope; enable Exponential and fit_all_facets. Use the
operator's offset/amplitude constraints, not a different external likelihood.

The recipe requires `panel_semantic` as well as `scan_plan`; it does not rely on
automatic X selection. Read `lab_recipes(name='trap_lifetime')`, copy its
`defaults.panel_semantic` mapping, and add the chosen hold-time fate as `'x'`.
For the recipe's single Pulse/API axis, the native Scan contract maps the exact
Pulse `FIELD_ID` to Point axis `scan.FIELD_ID`. Thus the same field selected from
the Pulse catalog for `pulse:param:FIELD_ID` (or `api:FIELD_ID`) supplies the key
`fate:point:scan.FIELD_ID`. This is the Scan contract for this single-axis recipe,
not a rule for other producers or a guessed Period ID.

```text
lab_recipes(name='trap_lifetime')
# SEMANTIC = a copy of recipes[0].defaults.panel_semantic
# FIELD_ID = the chosen hold-duration field_id from lab_editor_catalog
SEMANTIC['fate:point:scan.' + FIELD_ID] = 'x'
lab_run_recipe(name='trap_lifetime', parameters={
  'camera': CAMERA_ID, 'sequencer': SEQUENCER_ID,
  'pulse_path': VERIFIED_PULSE_PATH,
  'calibrations': [{'path': COMPATIBLE_CALIBRATION_PATH, 'frames': ['rest']}],
  'scan_plan': SCAN_PLAN_JSON, 'panel_semantic': SEMANTIC,
  'repeats': CHOSEN_SWEEPS, 'shots_per_point': CHOSEN_SHOTS,
  'frames_per_cycle': ALL_TRIGGER_FRAMES, 'start_frame': EARLIER_FRAME,
  'end_frame': LATER_FRAME, 'fit_fixed': FIT_FIXED
}, start=false)
```

`start=false` prepares the stopped board for inspection; use `start=true` to
prepare and start it in one call. To start a board already prepared, use the
returned follower/entry nodes and their current revisions with ordinary
`logic.start`, not another Layout replacement. One-point pilots keep the authored
X even though they cannot produce a meaningful fit. Later changes to point
values/counts retain it. Changing the scanned field itself requires a matching
axis assignment. Without explicit fates, generic plots prefer an axis that
varies; a site Facet can otherwise displace their default X and leave a vacancy.

For a three-panel view, use a 2x2 mean curve, a 4x4 per-site Facet Grid, and a
2x2 decay-time-versus-site curve. Here `size` is the panel size preset, not a
request to discard sites beyond a 4-by-4 array. `show_grid=false` removes axis
grid lines while retaining the facets and their borders; `uncertainty=true`
shows the native errors. These are display choices for this example.

The following is MCP call notation. Substitute values returned by the tools;
capitalized names are not literal signal names or axis IDs. `SITE_PANEL` is the
actual result panel ID returned by the recipe, and `SURVIVAL_SIGNAL` is its
`lab_catalog(panel_id=SITE_PANEL).panel.state.signal`.

First create and bind the mean panel, then read its bound fields:

```text
lab_action(action='panel.add', arguments={'kind': 'curve'})
# MEAN_PANEL = the returned panel_id
lab_action(action='panel.update', arguments={'panel_id': MEAN_PANEL, 'patch': {
  'signal': SURVIVAL_SIGNAL, 'size': '2x2', 'title': 'Mean conditional survival',
  'display': {'show_grid': false, 'uncertainty': true}
}})
lab_catalog(panel_id=MEAN_PANEL)
lab_catalog(panel_id=SITE_PANEL)
```

From those catalogs, select `HOLD_FATE`, the semantic field for the chosen scan
coordinate; `SITE_FATE`, the calibrated-site field; and `PAIR_FATE` plus
`PAIR_SCOPE`, the real Scope value for the intended earlier/later frame pair.
`REDUCED_REPEATS` is a mapping from each actual Repeat fate key to `'reduce'`.
Use the complete field keys, including their domains. Do not guess a duration
field such as `period4` or infer a site axis from an array dimension. When the
catalog says semantic fields are unavailable, wait for its source schema rather
than inventing them.

Apply the mean and per-site projections explicitly. `FIT_FIXED` is the user's
chosen native fixed-parameter mapping, or `{}` to leave the parameters free;
read the model's actual parameter identities before setting it. A normalization
constraint is a scientific choice, not a default imposed by this example.

```text
lab_action(action='panel.update', arguments={'panel_id': MEAN_PANEL, 'patch': {
  'semantic': {**REDUCED_REPEATS, HOLD_FATE: 'x', SITE_FATE: 'reduce',
               PAIR_FATE: PAIR_SCOPE, 'reduction': 'mean'},
  'fit': {'model': 'exponential_decay', 'fixed': FIT_FIXED}
}})
lab_action(action='panel.update', arguments={'panel_id': SITE_PANEL, 'patch': {
  'size': '4x4', 'title': 'Conditional survival by site',
  'semantic': {**REDUCED_REPEATS, HOLD_FATE: 'x', SITE_FATE: 'facet',
               PAIR_FATE: PAIR_SCOPE, 'reduction': 'mean'},
  'display': {'show_grid': false, 'uncertainty': true,
              'facet_fit_parameter': 'decay_time'},
  'fit': {'model': 'exponential_decay', 'fit_all_facets': true, 'fixed': FIT_FIXED}
}})
lab_results(panel_id=MEAN_PANEL)
lab_results(panel_id=SITE_PANEL)
```

In the site result's `published_signals`, take the actual `name` whose
`parameter` is `decay_time` as `TAU_SIGNAL`; do not construct it from an assumed
panel number. Each parameter signal carries its own uncertainty, so no separate
`decay_time_err` signal is needed. Bind the comparison panel and inspect its own
catalog:

```text
lab_action(action='panel.add', arguments={'kind': 'curve'})
# TAU_PANEL = the returned panel_id
lab_action(action='panel.update', arguments={'panel_id': TAU_PANEL, 'patch': {
  'signal': TAU_SIGNAL, 'size': '2x2', 'title': 'Decay time by site',
  'display': {'show_grid': false, 'uncertainty': true}
}})
lab_catalog(panel_id=TAU_PANEL)
lab_action(action='panel.update', arguments={'panel_id': TAU_PANEL, 'patch': {
  'semantic': {**FIT_REDUCED_REPEATS, FIT_SITE_FATE: 'x', 'reduction': 'mean'}
}})
```

`FIT_REDUCED_REPEATS` and `FIT_SITE_FATE` come from this comparison panel's
catalog. Fit outputs move non-Repeat sample axes into Point: the survival
input's site field can be `fate:cell_data:calibration.site`, while its published
fit parameter uses `fate:point:calibration.site`. Verify the returned schema;
copying the input's Cell-data fate key into this panel is invalid.

Reuse a compatible existing Calibration and verified Pulse/ROI/Context. Start
the following processors, then the Scan once; it owns acquisition ready/Fire.
Use Scan's timing estimate to distinguish first full sweep from all planned
trials. Read partial native fit results during the scan, with their validity;
do not wait for an unnecessarily large final batch before showing any result.
Save the native Figure and Layout and return the per-site decay_time table with
uncertainties and source references. Longer or different statistical analysis is
an additional scientific choice, not the prerequisite for this ordinary workflow.

Display updates, `lab_results`, and native `lab_save(..., refresh=true)` can run
while acquisition continues. Save each of the three panel IDs with its own path;
wait for its save to finish and preserve the source references. Do not Stop a
scan merely to show or save these figures. Choose scan points, complete sweeps
and shots per point for the current question and requested statistics; this
example supplies no apparatus-specific scan budget. Native fit success and
validity describe numerical results, not proof of a correct physical model,
sufficient precision or significant differences between sites.

## errors

Use the original owner reason and state. Unknown fields: query that node/panel's
catalog. No data: distinguish following/waiting, incompatible source, stale
generation and a genuinely failed run. Device operation pending: wait rather
than starting a second session. Failed/unknown Start, Fire or Send: inspect the
same generation/receipt; do not replay a partially completed batch.

Use lab_wait for an exact run and lab_wait_for for a declared read-only condition.
Timeout only ends observation. Stop and state calls remain available while
waiting. A batch stops on refusal, an unmet wait, error or lost generation even
if another explicitly expected field matches. Manual operator requests are not
automatically confirmed. Source/fit revision mismatches must stay visible.

lab_batch accepts already-decided tool/arguments steps. Within an argument,
{"$result":[0,"state","logic",0,"generation"]} takes the generation from
step 0's returned JSON (for example, a logic.start reply for one node). References
only address earlier completed steps; they are not expressions or a workflow
language. An unavailable field stops the batch before the next action. Keep
long experiment observation in lab_wait, outside the 30-second command batch.
