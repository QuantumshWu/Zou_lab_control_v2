# Console view contracts

These classes are pure Qt views.  They do not own records, data, run
lifecycle, rendering, scheduling, or persistence.  A host supplies fake or
real `QWidget` surfaces and plain values, then connects operator-intent
signals to its presenter.

## `PanelCardView`

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
set_panel_state(state) -> None
set_panel_projection(state, surface) -> None
set_status(text: str, *, error: bool) -> None
set_selectors_enabled(enabled: bool) -> None
```

Title, signal, size and update interval are edited in the card's Setting
form only, and every edit there leaves as one `state_changed` patch; the
card keeps no other control for them.

`set_selectors_enabled(False)` suspends the mounted plot's whole pointer
transport -- area, zoom, pan, hover and double-click focus -- and the
ordinary wheel stays with the surrounding page; On restores all of it.

`groups` is `(producer_label, ((display_label, key), ...))`.  The key is an
opaque string; the view does not interpret it.

## `ConsoleBoardView`

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
the one it replaces.  The presenter persists that order and each card's
size; it never sends pixel rectangles back to the view.  `order_committed`
is the only board-level reorder payload.

## `LogicRowView`

```python
start_requested = pyqtSignal()
stop_requested = pyqtSignal()
edit_requested = pyqtSignal()
remove_requested = pyqtSignal()

set_state(state: str, status_text: str = "") -> None  # idle|running|error
set_publishes(rows: tuple[tuple[str, str, str], ...]) -> None
```

Each publish row is `(name, shape_text, description)`.

## `StatusStrip`

```python
show_status(text: str, severity: str) -> None  # idle|warning|task|error
```

The newest message is the one shown, coloured by its severity; an empty
message returns the strip to its last idle text.

## `TaskConsoleView`

```python
add_panel_requested = pyqtSignal(str)
add_logic_requested = pyqtSignal(str)
pause_toggled = pyqtSignal(bool)
selectors_toggled = pyqtSignal(bool)
save_layout_requested = pyqtSignal()
load_layout_requested = pyqtSignal()
save_screenshot_requested = pyqtSignal()
stop_task_requested = pyqtSignal()
panel_order_committed = pyqtSignal(tuple)
editor_close_requested = pyqtSignal(object)

set_cards(cards: tuple[PanelCardView, ...]) -> None
set_logic_rows(rows: tuple[LogicRowView, ...]) -> None
show_status(text: str, severity: str) -> None
set_summary(text: str) -> None
```

The header names the window (`task`) as a label; it is not an input.

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
