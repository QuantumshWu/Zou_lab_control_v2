# `zlc_ui.pulse` view contract

`zlc_ui.pulse` contains only Qt views and frozen, plain view models. A
presenter owns the pulse document, projects it into these records, connects
the intent signals below, and feeds the narrow `set_*` methods back into the
views. No class in this package imports `zlc_pulse`, `zlc_plot`, or a pulse
domain object.

## View models

The public records are `ConnectionChoiceVM`, `ConnectionVM`, `FieldVM`,
`PortRowVM`, `PeriodVM`, `RepeatVM`,
`DelayRowVM`, `ScheduleVM`, `ScanPageRecord`, `TargetPortRecord`, and
`TargetWidthRule`. They are frozen dataclasses and contain only strings,
numbers, booleans, and tuples. `FieldVM.text` is already display-ready: a
presenter replaces a binding with `sN`/`aN` before constructing the record.

`ScheduleVM` carries `(document_generation, revision)`. `PulseScheduleView`
rejects an older pair and accepts an identical pair idempotently; a different
record at the same pair raises because one revision must have one projection.

The inline dot is an intent control, not a local model.  A presenter handles
`binding_cycle_requested`, asks the public `zlc_pulse.cycle_binding_kind()`
domain API for the next state, updates its `FieldVM`, and sends the new record
back to the view.  The UI package owns no copy of that transition.

For a DAC channel, `PortRowVM.kind == "dac"` plus a
`PeriodVM.analog` record renders the mode choices supplied by `ScheduleVM`
and the numeric `FluentScanLineEdit`.  A presenter can bind that value through
the same signal.  `Hold` is the authoring projection of no `AnalogStep`;
choosing `Edge` or `Ramp` with a value creates the domain step.

## Schedule page

```python
view = PulseScheduleView()
view.set_schedule(schedule_vm) -> bool
view.accept_local_commit(generation, revision) -> None
view.set_period(period_vm) -> None
view.set_delay_row(delay_row_vm) -> None
view.set_port_label(key, label) -> None
view.set_visible_ports(tuple[str, ...]) -> None
view.set_summary(total_text, total_tooltip, period_count,
                 visible_text, summary_text, scan_summary_text) -> None
view.set_scan_source(use_loaded, path) -> None
view.set_scan_busy(busy) -> None
view.set_connection(connection_vm) -> None
view.set_control_state(running, synchronized, file_dirty) -> None
view.set_capabilities(can_sync, can_hold, can_step) -> None
```

`PeriodCard`, `ChannelNamesPanel`, `ChannelPanel`, `RepeatBracket`, and
`PulseDragContainer` are reusable subviews. A drag emits
`move_period_requested(period_id, before_period_id)` and does not mutate the
local order; the presenter commits the new `ScheduleVM`. The schedule page
also emits `document_name_committed`, `port_label_committed`,
`period_name_committed`, `duration_committed`, `digital_committed`,
`analog_committed`, `delay_committed`, `binding_cycle_requested`,
`insert_period_requested`, `move_period_requested`,
`remove_period_requested`, `repeat_committed`, `visible_ports_committed`,
`clear_port_requested`, `clear_all_requested`, the run/save/load/connection
signals, and `feedback_requested`.

## Scan, target, and preview pages

`PulseScanView` accepts `set_page(ScanPageRecord)`, `set_repeats_range(minimum,
default)`, `set_repeats`, `set_scan_code`, `replace_scan_draft`,
`acknowledge_scan_draft`, `set_scan_table_text`, `set_slots_text`,
`set_progress_text`, `set_workspace_busy`, `set_run_dirty`, and
`set_progress_polling`. The three draft methods are intentionally separate:
the presenter can acknowledge a revision without overwriting text currently
being typed.

`PulseTargetView` accepts `set_ports(records, editable, status_text)`,
`set_width_rules(digital, dac)`, and `set_feedback(text)`. Its
`apply_requested` payload is `tuple[TargetPortRecord, ...]`; manifest
construction and domain validation stay in the presenter.

`PulsePreviewView` accepts `set_size_names(tuple[str, ...])`,
`set_preview_size(size, pinned=...)`, `set_status`, `show_placeholder`, and
`mount_content(widget, logical_size=..., wheel_target=...)`. The latter is a
QWidget mount point, not a renderer. It emits `include_off_toggled`,
`selectors_toggled`, `size_committed`, and `save_requested`.

## Editor shell

`PulseEditorView` composes Edit, Preview, Scan, and Target tabs and exposes
`set_title`, `set_summary`, `set_status_color`, `ask_open_path`,
`ask_save_path`, `confirm`, `show_warning`, `finish_close`, and the
`close_requested`/`clear_all_requested` signals. It does not know a
controller. `launch_pulse_editor_window` is the optional Fluent frameless
wrapper for a human demo or host application.
