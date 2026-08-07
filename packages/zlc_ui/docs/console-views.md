# Console view contracts

These classes are pure Qt views.  They do not own records, data, run
lifecycle, rendering, scheduling, or persistence.  A host supplies fake or
real `QWidget` surfaces and plain values, then connects operator-intent
signals to its presenter.

## `PanelCardView`

```python
signal_picked = pyqtSignal(str)
size_picked = pyqtSignal(str)
update_ms_picked = pyqtSignal(int)
title_committed = pyqtSignal(str)
remove_requested = pyqtSignal()
edit_requested = pyqtSignal()
dropped = pyqtSignal(tuple)  # (x: int, y: int), card-local drop point

set_surface(widget: QWidget | None) -> None
set_signal_choices(groups: tuple[tuple[str, tuple[tuple[str, str], ...]], ...]) -> None
set_status(text: str, *, error: bool) -> None
set_selectors_enabled(enabled: bool) -> None
```

`groups` is `(producer_label, ((display_label, key), ...))`.  The key is an
opaque string; the view does not interpret it.

## `ConsoleBoardView`

```python
order_committed = pyqtSignal(tuple)  # tuple[str, ...], panel_id order

set_cards(cards: tuple[PanelCardView, ...]) -> None
grab_board() -> QPixmap
```

Construct the board with an injected `BoardMetrics(gap, card_size)` policy.
After `set_cards`, the board owns the only live geometry calculation: it
packs the cards, repacks them when its width changes, moves the dragged card
freely without a placeholder/ghost, and commits the new `panel_id` order on
release.  The presenter
persists that order and each card's size; it never sends pixel rectangles
back to the view.  `order_committed` is the only board-level reorder payload,
and `grab_board()` captures the current board for a host-owned image action.

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

Visible status priority is `error > task > warning > idle`.

## `TaskConsoleView`

```python
add_panel_requested = pyqtSignal()
add_logic_requested = pyqtSignal()
pause_toggled = pyqtSignal(bool)
selectors_toggled = pyqtSignal(bool)
save_requested = pyqtSignal()
load_requested = pyqtSignal()
save_image_requested = pyqtSignal()

set_cards(cards: tuple[PanelCardView, ...]) -> None
set_logic_rows(rows: tuple[LogicRowView, ...]) -> None
show_status(text: str, severity: str) -> None
set_summary(text: str) -> None
```

The shell is assembled from the reusable views.  There is no mode flag for
alternative composition; a host can mount the shell wherever it needs it.

## `DeviceManagerView`

```python
device_add_requested = pyqtSignal(str)
device_remove_requested = pyqtSignal(str)
role_committed = pyqtSignal(str, str)
type_picked = pyqtSignal(str, str)
parameter_committed = pyqtSignal(str, str)

set_device_choices(choices: tuple[tuple[str, str], ...]) -> None
set_devices(devices: tuple[tuple[str, str, str], ...]) -> None
set_form_spec(
    instance_id: str,
    spec: FormSpec,
    values: tuple[tuple[str, object], ...],
) -> None
read_values(instance_id: str) -> tuple[tuple[str, object], ...]
show_status(text: str, severity: str) -> None
```

Choice rows are `(display_label, opaque_key)`.  Device rows are
`(instance_id, role, type_key)`; the host owns their catalog, identity, and
persistence.  Form fields use the shared `FormSpec` contract, and the view
only reports which instance or field the operator edited.
