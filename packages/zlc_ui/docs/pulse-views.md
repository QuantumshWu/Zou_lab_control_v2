# `zlc_ui.pulse` view contract

`zlc_ui.pulse` contains only Qt views and frozen, plain view models. A
presenter owns the pulse document, projects it into these records, connects
the intent signals below, and feeds the narrow `set_*` methods back into the
views. No class in this package imports `zlc_pulse`, `zlc_plot`, or a pulse
domain object.

Outside the package the editor is one flat `PulseEditorHandle`
(`zlc_ui.pulse.handle`, opened through `open_pulse_editor`): signals to hear
and methods to call, with no QWidget reachable, so no host can push a value
straight into one panel and keep a second copy of the picture.  The pages
below are what the handle drives; the handle's names are listed at the end.

## View models

The public records are `ConnectionChoiceVM`, `ConnectionVM`, `FieldVM`,
`PortRowVM`, `PeriodVM`, `BracketVM`,
`DelayRowVM`, `ScheduleVM`, `BindingRecord`, `ScanPageRecord`, `ConfigPageRecord`,
`TargetPortRecord`, and `TargetWidthRule`. They are frozen dataclasses and contain only strings,
numbers, booleans, and tuples. `FieldVM.text` always shows the editable default;
its independent `scan` and `source` fields describe the binding. `effective_text`
and `source_text` describe a saved external override without replacing that default.

`ScheduleVM` carries `(document_generation, revision)`. `PulseScheduleView`
rejects an older pair and accepts an identical pair idempotently; a different
record at the same pair raises because one revision must have one projection.
The value-level setters below rewrite the held `ScheduleVM` under an unchanged
pair, which is safe because the presenter advances its revision on every
accepted state change: a full push after a value edit always carries a newer
pair. Its `brackets` (`tuple[BracketVM, ...]`, outermost first) are the named,
possibly nested timeline-internal loops, while `run_repeats` is the
independent complete-Pulse count shown on Edit (`0 = infinite`).

The inline binding button opens a Fluent popup with a Scan checkbox and
Default/API/Config source choice. It emits `binding_committed`, and the presenter
returns the accepted `FieldVM`. The popup does not edit Config names or files.
S/A/C badges contain no numeric identity; Scan defaults remain editable.

For a DAC channel, `PortRowVM.kind == "dac"` plus a
`PeriodVM.analog` record renders the mode choices supplied by `ScheduleVM`
and the numeric `FluentScanLineEdit`.  A presenter can bind that value through
the same signal.  `Hold` is the authoring projection of no `AnalogStep`;
choosing `Edge` or `Ramp` with a value creates the domain step.

## Schedule page

```python
view = PulseScheduleView()
view.set_schedule(schedule_vm) -> bool
view.set_period(period_vm) -> None
view.set_delay_row(delay_row_vm) -> None
view.set_port_label(key, label) -> None
view.set_visible_ports(tuple[str, ...]) -> None
view.set_summary(total_text, total_tooltip, period_count,
                 visible_text, summary_text, scan_summary_text) -> None
view.set_connection(connection_vm) -> None
view.set_control_state(running, synchronized, file_dirty,
                       *, can_run, can_stop) -> None
view.set_capabilities(can_sync, can_hold, can_step) -> None
```

The value-level setters (`set_period`, `set_delay_row`, `set_port_label`,
`set_visible_ports`) update the accepted `ScheduleVM` the view holds as
well as the controls, so the next rebuild from that model shows the same
thing the controls do.  A port whose `kind` changes under the same key is
rebuilt as a new row.  `set_visible_ports` re-flags the rows only; the
`visible_text` count is worded by the presenter and arrives with
`set_summary`.

`PeriodCard`, `ChannelNamesPanel`, `ChannelPanel`, `BracketPost`, and
`PulseDragContainer` are reusable subviews. `ScheduleVM.item_order` derives
one visual order of `(kind, id)` items from periods and bracket anchors;
both post and period drags emit `reorder_items_requested(item_order)`.
Add targets the next visual item, not the next period, so both sides of a
post are distinct gaps. Neither drag mutates local state; the presenter
commits period order and bracket anchors atomically in the new `ScheduleVM`.
Empty brackets remain editable; running or saving requires nonempty content.
`bracket_committed(bracket_id, start_period_id, end_period_id, count)` moves or
recounts one bracket; `bracket_add_requested(start_period_id, end_period_id,
count)` asks for a new one (the presenter names it) and
`bracket_remove_requested(bracket_id)` deletes one. The schedule page
also emits `document_name_committed`, `port_label_committed`,
`period_name_committed`, `duration_committed`, `digital_committed`,
`analog_committed`, `delay_committed`, `binding_committed`,
`insert_period_requested`, `insert_spacer_requested`,
`reorder_items_requested`, `remove_period_requested`,
`run_repeats_committed`, `visible_ports_committed`, `fill_port_requested`,
`clear_port_requested`, `run_requested`, `stop_requested`, `sync_requested`,
`save_requested`, `load_requested`, `config_requested`,
`connection_requested`, and `feedback_requested`.

## Scan, Config, target, and preview pages

`PulseScanView` accepts one `set_page(ScanPageRecord)` projection -- the
record carries the slot text, the field `BindingRecord`s, the table text, the
scan code and whether it is dirty, repeats, busy, progress text and progress
polling -- plus `set_repeats_range(minimum, default)`, `set_repeats`,
`set_progress_text`, `set_workspace_busy`, `set_run_dirty`, and
`set_progress_polling`.  Text the operator is typing is never overwritten
by a projection while they are typing it.

`PulseConfigView.set_page(ConfigPageRecord)` projects the editing file, dirty
values table, active saved file and this Pulse's field-to-name references.
New/Load/Refresh/Save/Save as/Unload emit intents; Qt never reads or writes files.
Editing Name/Value/Unit emits raw row text so incomplete input stays editable.
Bindings can reuse one name across multiple fields. Saved values are read-only
facts from the presenter, never inferred from the unsaved table. Load Array is
on Scan; Edit has only a compact jump-to-Config status button.

`PulseTargetView` accepts `set_ports(records, editable, status_text,
reserved=())`, `set_width_rules(digital, dac)`, and `set_feedback(text)`.
`reserved` names ports the target holds with no row on the page (a clock no
DAC latches with); Add never mints one. The `apply_requested` payload is
`tuple[TargetPortRecord, ...]`; manifest construction and domain validation
stay in the presenter.

`PulsePreviewView` accepts `set_size_names(tuple[str, ...])`,
`set_preview_size(size)`, `set_status`, `show_placeholder`, and
`mount_content(widget, logical_size=..., wheel_target=...)`. The latter is a
QWidget mount point, not a renderer. It emits `include_off_toggled`,
`selectors_toggled`, `size_committed`, and `save_requested`.  Whether the
shown size is pinned by the operator or chosen by the content is the
presenter's fact; the view keeps no copy of it.

## Editor shell

`PulseEditorView` composes Edit, Preview, Scan, Config, and Target tabs and exposes
`set_title`, `set_summary`, `set_status_color`, `ask_open_path`,
`ask_save_path`, `confirm`, `show_warning`, `finish_close`, and the
`close_requested`/`clear_all_requested` signals. It does not know a
controller. A product opens it through the single `open_pulse_editor` handle;
the view does not provide a parallel raw-window launcher.

## `PulseEditorHandle`

The port is flat: one set of names for the whole window, prefixed by the page
where two pages would otherwise collide (the schedule's and the scan page's
run, the schedule's and the preview's save).

Signals: the window's `close_requested`, `closed`, `device_label_changed`,
`page_changed`; the document's `document_name_committed`,
`clear_all_requested`, `save_requested`, `load_requested`; the Config page's
`config_new_requested`, `config_load_requested`, `config_refresh_requested`,
`config_save_requested`, `config_save_as_requested`,
`config_unload_requested`, `config_entries_edited`,
`config_binding_committed`; the schedule's signals listed above under the same
names, with Run as `fire_requested` (its `config_requested` only turns the
window to the Config page and never leaves it); the scan page's
`scan_array_load_requested`, `scan_source_edited`, `scan_repeats_committed`,
`scan_hold_requested`, `scan_step_requested`, `scan_program_load_requested`,
`scan_template_requested`, `scan_run_requested`, `scan_array_save_requested`,
`scan_progress_refresh_requested`; the preview's
`preview_include_off_toggled`, `preview_size_committed`,
`preview_selectors_toggled`, `preview_save_requested`; and the target's
`target_apply_requested`.

Methods: `close`, `set_close_guard`, `finish_close`, `is_visible`, `restore`,
`window_size`, `window_title`, `set_device_label`, `set_title`, `set_summary`,
`set_status_color`, `set_capabilities`, `show_done`, `show_warning`,
`current_page`, `ask_open_path`, `ask_save_path`, `confirm_config_discard`,
`confirm_pulse_discard`;
`set_config_page`; `set_schedule`, `set_period`, `set_delay_row`,
`set_port_label`, `set_schedule_summary`, `set_visible_ports`,
`set_control_state`, `set_connection`; `set_scan_page`,
`set_scan_progress_text`; `preview_include_off_rows`, `preview_size`, `set_preview_size`,
`set_preview_size_names`, `set_preview_status`, `show_preview_placeholder`,
`show_preview(host)`; `set_target_ports`, `set_target_width_rules`,
`set_target_feedback`.  The preview arrives as its plot host, never as a
widget: this package may not import the package that draws.
