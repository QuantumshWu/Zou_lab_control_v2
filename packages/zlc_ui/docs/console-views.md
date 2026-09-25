# Console view contracts

A task console crosses into the composition root through one object,
`TaskConsoleHandle` (`zlc_ui.console.handle`).  The window owns its cards,
logic rows and editors; the outside names panels and logic rows by id and
never builds or receives a widget.  A drawn panel arrives as its plot host
(`show_panel(panel_id, host)`), never as a widget: this package may not import
the package that draws.  The views after the handle are the window's
internals -- pure Qt views that own no records, data, run lifecycle,
rendering, scheduling or persistence.

## `TaskConsoleHandle`

What the operator did, going out:

```python
# the window
closed = pyqtSignal()
# the board
add_panel_requested = pyqtSignal(str)          # plot kind
add_logic_requested = pyqtSignal(str)          # logic kind
pause_toggled = pyqtSignal(bool)
selectors_toggled = pyqtSignal(bool)
save_layout_requested = pyqtSignal()
load_layout_requested = pyqtSignal()
clear_board_requested = pyqtSignal()
save_screenshot_requested = pyqtSignal()
stop_task_requested = pyqtSignal()
panel_order_committed = pyqtSignal(tuple)      # panel_id order
# one named panel (the first argument is its panel_id)
panel_remove_requested = pyqtSignal(str)
panel_edit_requested = pyqtSignal(str)
panel_plot_error = pyqtSignal(str, str)        # a refusal its plot surface reported
panel_state_changed = pyqtSignal(str, object)  # one Setting-form patch
panel_snapshot_refresh_requested = pyqtSignal(str)
panel_save_figure_requested = pyqtSignal(str, str)
panel_editor_closed = pyqtSignal(str)
panel_publisher_edit_requested = pyqtSignal(str)
panel_publisher_draft_changed = pyqtSignal(str, object)
# one named logic row (the first argument is its node_id)
logic_start_requested = pyqtSignal(str)
logic_auto_preview_changed = pyqtSignal(str, bool)
logic_stop_requested = pyqtSignal(str)
logic_edit_requested = pyqtSignal(str)
logic_remove_requested = pyqtSignal(str)
logic_draft_changed = pyqtSignal(str, object)
logic_refresh_requested = pyqtSignal(str)      # re-read that node's files
```

What the window shows, coming in:

```python
# the window
close(); close_later(); set_close_guard(guard)
is_visible() -> bool
plots_visible() -> bool    # Monitor board or a panel's Edit on screen; False when minimized
window_size() -> tuple[int, int]; window_title() -> str; device_pixel_ratio() -> float
# board vocabulary and header
set_panel_kinds(kinds, current=""); set_logic_kinds(kinds)
set_panel_intervals(intervals, default_interval); set_panel_sizes(sizes, default_size)
set_grid_cell_kinds(kinds)
set_paused(paused); set_selectors(enabled); set_summary(text)
show_status(text, severity)            # idle|warning|task|error
set_task_takeover(active)
# questions asked of the operator
choose_signal(rows) -> str | None
ask_save_path(caption, suggested, filter) -> str; ask_open_path(caption, start, filter) -> str
save_screenshot(path) -> str; show_warning(title, text)
review_points(surface, points, *, title, message="", confirm_label="Continue",
              initial_excluded=()) -> tuple[str, ...] | None
manual_axis_setting(*, title, message) -> bool   # Continue / Stop for a manual scan axis
confirm_board_replaced(title, message, *, confirm_text) -> bool
# panels, by id
add_panel(panel_id, title); remove_panel(panel_id); panel_ids() -> tuple[str, ...]
set_panel_order(order)
show_panel(panel_id, host)             # host or None; the window picks the widget
present_panel_front(panel_id, front) -> bool
set_panel_signal_choices(panel_id, groups, *, current="", overlay_groups=(), overlay_current="")
set_panel_projection(panel_id, state, parameter_surface)
set_panel_status(panel_id, text, *, error)
set_panel_selectors_enabled(panel_id, enabled)
# a panel's Edit tab
open_panel_editor(panel_id, projection); update_panel_editor(panel_id, projection) -> bool
show_panel_editor(panel_id, host); focus_panel_editor(panel_id) -> bool
close_panel_editor(panel_id) -> bool
set_panel_snapshot_status(panel_id, status); set_panel_producer_projection(panel_id, projection)
# panel-owned signals (ROI/fit outputs), shown beside the logic rows
set_panel_publishers(publishers)       # ((panel_id, ((name, shape, description), ...)), ...)
open_panel_publisher_editor(panel_id, projection); update_panel_publisher_editor(...) -> bool
has_panel_publisher_editor(panel_id); focus_panel_publisher_editor(panel_id)
close_panel_publisher_editor(panel_id)
# logic rows, by id
add_logic_row(node_id, kind, offers_preview=True); remove_logic_row(node_id)
set_logic_state(node_id, state, status_text="")          # idle|running|error
set_logic_commands(node_id, *, can_start, can_stop)
set_logic_auto_preview(node_id, enabled)
set_logic_publishes(node_id, rows)     # ((name, shape_text, description), ...)
open_logic_editor(node_id, projection); update_logic_editor(node_id, projection) -> bool
has_logic_editor(node_id); focus_logic_editor(node_id); close_logic_editor(node_id)
```

`groups` for signal choices is `(producer_label, ((display_label, key), ...))`;
the key is an opaque string the window does not interpret.  A panel's widget
is chosen by the composition root's `plot_surface` policy (one widget per
host), so a board can present a same-shot group atomically through
`present_panel_front`.

## `PanelCardView` (internal)

```python
state_changed = pyqtSignal(object)  # one Setting-form patch: {key: value, ...}
remove_requested = pyqtSignal()
edit_requested = pyqtSignal()
drag_started = pyqtSignal(tuple)  # (x: int, y: int), once per drag gesture
dropped = pyqtSignal(tuple)  # (x: int, y: int), card-local drop point
geometry_changed = pyqtSignal()
plot_error = pyqtSignal(str)  # relayed from the mounted surface's errorOccurred

set_surface(widget: QWidget | None) -> None
set_signal_choices(groups, *, current="", overlay_groups=(), overlay_current="") -> None
set_size_choices(sizes: tuple[str, ...], default_size: str) -> None
set_interval_choices(intervals: tuple[int, ...], default_interval: int) -> None
set_cell_kind_choices(kinds: tuple[str, ...]) -> None
set_panel_projection(state, surface) -> None
set_status(text: str, *, error: bool) -> None
set_selectors_enabled(enabled: bool) -> None
```

`new_panel_card(...)` builds one; TaskConsole and FigureViewer both use it.
Title, signal, size and update interval are edited in the card's Setting
form only, and every edit there leaves as one `state_changed` patch; the
card keeps no other control for them.

The title strip names the panel by the surface's `caption`, exactly as the
presenter composed it: a signal's name is read by the runtime that owns its
grammar, never by the card.  The caption is display only -- the state's
`title` stays the panel's editable name -- and a card no presenter has
captioned is called by that name.

`set_selectors_enabled(False)` suspends the mounted plot's whole pointer
transport -- area, zoom, pan, hover and double-click focus -- and the
ordinary wheel stays with the surrounding page; On restores all of it.

## `ConsoleBoardView` (internal)

```python
order_committed = pyqtSignal(tuple)  # tuple[str, ...], panel_id order

set_cards(cards: tuple[PanelCardView, ...]) -> None
```

Construct the board with an injected `BoardMetrics(gap)` policy; each
card's rectangle is the card's own.  After `set_cards`, the board owns the
only live geometry calculation: it packs the cards, repacks them when its
width changes, moves the dragged card freely without a placeholder/ghost,
and commits the new `panel_id` order on release.  One `panel_id` is one
card: a different object arriving under an id already on the board retires
the one it replaces, and a card retired in the middle of a drag is ignored
when it drops.  The presenter persists that order and each card's size; it
never sends pixel rectangles back to the view.  `order_committed` is the
only board-level reorder payload.

## `LogicRowView` (internal)

```python
start_requested = pyqtSignal()
stop_requested = pyqtSignal()
edit_requested = pyqtSignal()
remove_requested = pyqtSignal()
auto_preview_changed = pyqtSignal(bool)

set_state(state: str, status_text: str = "") -> None  # idle|running|error
set_commands(*, can_start: bool, can_stop: bool) -> None
set_preview_offered(offered: bool) -> None
set_auto_preview(enabled: bool) -> None
set_task_takeover(active: bool) -> None
set_publishes(rows: tuple[tuple[str, str, str], ...]) -> None
```

Each publish row is `(name, shape_text, description)`.

## `StatusStrip` (internal)

```python
show_status(text: str, severity: str) -> None  # idle|warning|task|error
```

The newest message is the one shown, coloured by its severity; an empty
message returns the strip to its last idle text.

## `TaskConsoleView` (internal)

```python
add_panel_requested = pyqtSignal(str)
add_logic_requested = pyqtSignal(str)
pause_toggled = pyqtSignal(bool)
selectors_toggled = pyqtSignal(bool)
save_layout_requested = pyqtSignal()
load_layout_requested = pyqtSignal()
clear_board_requested = pyqtSignal()
save_screenshot_requested = pyqtSignal()
stop_task_requested = pyqtSignal()
panel_order_committed = pyqtSignal(tuple)
editor_close_requested = pyqtSignal(object)

set_panel_kinds(kinds, current="") -> None
set_logic_kinds(kinds) -> None
set_cards(cards: tuple[PanelCardView, ...]) -> None
set_logic_rows(rows: tuple[LogicRowView, ...]) -> None
set_paused(paused: bool) -> None
set_selectors(enabled: bool) -> None
set_task_takeover(active: bool) -> None
show_status(text: str, severity: str) -> None
set_summary(text: str) -> None
```

The header names the window (`task`) as a label; it is not an input.  The
handle forwards the board signals and drives the rest; nothing outside the
window calls these directly.

The shell is assembled from the reusable views.  There is no mode flag for
alternative composition; a host can mount the shell wherever it needs it.

## `DeviceManagerView`

```python
device_add_requested = pyqtSignal(str)
device_remove_requested = pyqtSignal(str)
role_committed = pyqtSignal(str, str)
type_picked = pyqtSignal(str, str)
parameter_committed = pyqtSignal(str, str)

set_device_choices(choices: tuple[tuple[str, str, str], ...]) -> None
set_devices(devices: tuple[tuple[str, str, str, str], ...]) -> None
set_form_spec(
    instance_id: str,
    spec: FormSpec,
    values: tuple[tuple[str, object], ...],
) -> None
read_values(instance_id: str) -> tuple[tuple[str, object], ...]
show_status(text: str, severity: str) -> None
```

Choice rows are `(display_label, opaque_key, domain)`.  Device rows are
`(instance_id, role, type_key, domain)`, grouped on the surface by domain;
the host owns their catalog, identity, and persistence.  A projected record
is never reported back as a pick: `type_picked` fires only for the
operator's own choice.  Form fields use the shared `FormSpec` contract, and
the view only reports which instance or field the operator edited.

## `DeviceControlView`

Each entry of the projection's `fields` carries, beside `current` and
`desired`, `device_limits`: the instrument's own `(low, high)` range for the
field in the field's unit, or `None`.  The view shows it read-only in its
own column beside the editable window, in the spelling the row is read in;
it is a value, never a control.
