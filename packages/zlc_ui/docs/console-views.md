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
logic_revert_requested = pyqtSignal(str)       # put the running configuration back in the draft
logic_edit_requested = pyqtSignal(str)
logic_remove_requested = pyqtSignal(str)
logic_draft_changed = pyqtSignal(str, object)
logic_name_committed = pyqtSignal(str, str)    # node_id, the finished Name text
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
set_task_takeover(owner)               # the running Task's node_id; "" when none
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
set_panel_publishers(publishers)       # ((panel_id, panel name, ((name, shape, description), ...)), ...)
open_panel_publisher_editor(panel_id, projection); update_panel_publisher_editor(...) -> bool
has_panel_publisher_editor(panel_id); focus_panel_publisher_editor(panel_id)
close_panel_publisher_editor(panel_id)
# logic rows, by id
add_logic_row(node_id, kind, offers_preview=True); remove_logic_row(node_id)
set_logic_name(node_id, name, api_name="")               # what the node is called, and its type
set_logic_state(node_id, state, status_text="")          # idle|running|error
set_logic_commands(node_id, *, can_start, can_stop, pending=False)
set_logic_auto_preview(node_id, enabled)
set_logic_publishes(node_id, rows)     # ((name, shape_text, description), ...)
open_logic_editor(node_id, projection); update_logic_editor(node_id, projection) -> bool
has_logic_editor(node_id); focus_logic_editor(node_id); close_logic_editor(node_id)
```

`groups` for signal choices is `(producer_label, ((display_label, key), ...))`;
the key is an opaque string the window does not interpret, and every heading
and label is display text the presenter composed (a Logic node's producer
heading is its Name).  An editor's tab follows what it edits: `Edit · <name>`
and `Edit panel · <panel name>` are retitled on every update.  A panel's widget
is chosen by the composition root's `plot_surface` policy (one widget per
host), so a board can present a same-shot group atomically through
`present_panel_front`.

## `PanelCardView` (internal)

```python
state_changed = pyqtSignal(object)  # one Setting-form patch: {key: value, ...}
remove_requested = pyqtSignal()
edit_requested = pyqtSignal()
drag_started = pyqtSignal(tuple)  # global pointer (x, y), once per drag gesture
drag_moved = pyqtSignal(tuple)  # global pointer (x, y), after every move the card follows
dropped = pyqtSignal(tuple)  # global pointer (x, y) at the release that ends a drag
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
keep_commands_within(right: int) -> None
```

`new_panel_card(...)` builds one; TaskConsole and FigureViewer both use it.
Title, signal, size and update interval are edited in the card's Setting
form only, and every edit there leaves as one `state_changed` patch; the
card keeps no other control for them.

The title strip names the panel by the surface's `caption`, exactly as the
presenter composed it: a signal's name is read by the runtime that owns its
grammar, never by the card.  Beside it the presenter projects `panel_name`,
what the panel is called -- what the Setting form's "Panel name" and the Edit
page show, a renamed node's `<name>/<output>` while the title is automatic.
Both are display only -- the state's `title` stays the panel's record -- and
a card no presenter has captioned is called by its title.

A card is laid out the same with or without a picture.  Its strip always
reads two lines -- the caption (and data sizes) above, the axis names below
-- and a card with no picture says why on that second line instead: `Choose
a signal in Setting`, or `No data yet` once a signal is picked.  Its body is
the blank frame its size preset plans, and `set_surface(None)` returns a
card to that frame.  `handle_rect()` is what carries the card: a drawn
card's strip (`strip_rect()`) -- its body is its plot's -- and the whole of
an empty card, whose blank body shows the open hand; where the board puts
a carried card does not depend on where it is held.  Setting is as tall as every other button and the header `×` is
square at that height: the quiet one-glyph remover (red only under the
pointer).  Removing a panel from the header `×` takes two clicks
(`FluentButton(confirm_twice=...)`): the first arms the white-on-red
confirm, a second within the system double-click interval removes, and left
alone it returns to rest.  The Remove in its Setting frame removes the same
panel and takes two clicks the same way.  Setting
is held down (the active face) while its frame is open.  On a card
wider than what its viewport shows, the two ride the viewport's right edge
(`keep_commands_within(right)`), as far in from it as from the card's own
end, so a panel is edited or removed without scrolling sideways; scrolled
to its end, they are back in the corner.

The Setting frame has the Edit page's sections, built by the same
`panel_section`: Panel, then Semantic, Display and Fit as the plot owner
declares them, each under its title a section gap apart, every control on
one label column (`share_label_column`); a section with nothing declared
is not shown.  Its height is its form's, bounded by its page, not by the
card: it opens under the card's strip where the page has the room and
over the card where it has more -- never over the strip -- its right edge
under Setting's.  Its close `×` is the quiet glyph (`role="quiet"`), a
control high and square, ending where the fields end.

`set_selectors_enabled(False)` suspends the mounted plot's whole pointer
transport -- area, zoom, pan, hover and double-click focus -- and the
ordinary wheel stays with the surrounding page; On restores all of it.

## `ConsoleBoardView` (internal)

```python
order_committed = pyqtSignal(tuple)  # tuple[str, ...], panel_id order

set_cards(cards: tuple[PanelCardView, ...]) -> None
```

The board's gap -- between cards and from its edges -- is one card pad
(`card_pad()`) unless a `BoardMetrics(gap)` policy is injected, so its first
card starts where a white page's content starts; each card's rectangle is
the card's own.  After `set_cards`, the board owns the only live geometry
calculation, and one rule set places every card, whatever their sizes:
every card has a place, a card let go onto another changes places with it,
a card let go on empty space goes there, cards that overlap make way, and
everything falls north-west.

- **Places.**  A drop records where it left every card, for the board's
  column count (how many of the narrowest card fit side by side); nothing
  else writes them.  A card no drop has placed yet -- one just added --
  takes the first free slot after them, top row left to right, in board
  order, at the width the board has now, so Add tiles the width, including
  for cards added before the window was first laid out.  A column count no
  drop was made at shows the nearest wider count's arrangement packed
  narrower (with none wider, the nearest narrower one): narrowing the window
  packs without destroying the arrangement, widening restores it, and a drop
  made while narrow does not overwrite the wide arrangement.
- **Gravity.**  A card rises until a card above it stops it and slides left
  until a card beside it stops it, until neither move is possible.  A card
  stopped by a card above it slides left no further than that card's left
  edge -- with several above it, the leftmost one's: a card put below
  another stays below it, and does not slide along its bottom and rise into
  a hole beside it.  Rising and sliding stop at the first card in the way,
  so falling passes no card, and no card searches the board for a hole.
- **Carrying.**  The slot is chosen in the board as it stood when the card
  was picked up, which stands still while the card is carried; picking a
  card up does not let the others fall into its hole.  The slot is the
  place under the pointer, on the line of the carried card's strip,
  nearest where the card is drawn: a 2x4 held near Setting and an empty
  4x4 held by its body land where the same card held at the left of its
  strip would.  Rows come only from the cards in the slot's column, so a
  short card's edge never cuts a tall card beside it.
- **Dropping** (`drop`) **swaps, then applies gravity.**  The card let go
  onto -- the one standing over the slot's top-left corner -- takes the
  place the carried card stood in, and the carried card takes the slot;
  let go on empty space, the carried card takes the slot.  Where cards of
  different sizes then overlap, they settle in priority: the two that
  swapped first, the one further up and left (in reading order) keeping its
  place, then every other card in the reading order of where it stands.  A
  card that overlaps one already settled moves right just clear of it if
  its row has the room -- on the board, passing no card on the way, and
  never across a card it makes way for -- otherwise down its own column to
  the first clear place, the cards below it in that column moving down
  with it so a column keeps its order (a card its column carried down makes
  way further down only).  Then
  every card falls north-west until nothing moves.  Put back where it was
  picked up, the card leaves every card where it stood.

While a card is carried it follows the pointer, the other cards already
stand where the drop will leave them -- the preview and the drop are one
computation -- and the slot is drawn under it as an accent-tint card with a
dashed edge.  Near the viewport's top or bottom edge the board scrolls under
the pointer; its scroll area keeps the vertical bar's gutter
(`FluentScrollArea.reserve_vertical_gutter()`), so the row a carried card
reserves never narrows the board under the drag.  On release the new
`panel_id` order -- the cards' reading order -- is committed.
`text_left()` is where a card's strip text
starts, from the board's left edge, for a note on an empty board to start
at.  On every pack, scroll and viewport resize the board tells each card
where the viewport's right edge cuts it, for its Setting and `×`.  One
`panel_id` is one card: a different object arriving under an id already on
the board retires the one it replaces, and a card retired in the
middle of a drag is ignored when it drops.  The presenter persists that
order and each card's size; it never sends pixel rectangles back to the
view.  `order_committed` is the only board-level reorder payload.

## `LogicRowView` (internal)

```python
start_requested = pyqtSignal()
stop_requested = pyqtSignal()
edit_requested = pyqtSignal()
remove_requested = pyqtSignal()
auto_preview_changed = pyqtSignal(bool)

set_name(text: str, details: tuple[str, ...] = ()) -> None
set_state(state: str, status_text: str = "") -> None  # idle|draft|running|error
set_commands(*, can_start: bool, can_stop: bool, pending: bool = False) -> None
set_preview_offered(offered: bool) -> None
set_auto_preview(enabled: bool) -> None
set_task_takeover(active: bool, *, owns_task: bool = False) -> None
set_publishes(rows: tuple[tuple[str, str, str], ...]) -> None
```

Each publish row is `(name, shape_text, description)`.  The row shows the
node's Name, elided in the middle in its fixed column so names that differ at
their end still read apart; its tooltip is the whole name, then the node's
type and `internal ID <node_id>` (`details`).  The name arrives only by
`set_logic_name`, and the row is not where it is renamed -- its Edit page is.
A panel publisher's row shows the panel's name, its panel id in the tooltip.

Start and Stop are two fixed slots on the row and on the node's Edit page,
both always shown.  Start reads `Starting…` while `pending`, `Restart` while
`can_stop` (running, following a source, or queued) and `Start` otherwise,
at the width of its widest word, so nothing beside it moves; Stop follows
`can_stop`, not the row's state.  While a Task owns the console every
command is disabled except the owning Task's own Stop (`owns_task`), which
raises the same `stop_requested` as the status strip's "Stop task", and Edit,
which opens a page that is read-only but for its Name.

A state is drawn as its status word (`STATE_WORDS`): a draft as `idle`, a
run as `running`, a failure as `error`, the dot and the status line from the
one `STATUS_CUES` table, on the row and on the node's Edit page alike.
Removing a node takes two clicks, from the row's `×` and from the Remove on
its Edit page alike: the first arms the white-on-red confirm, and
`remove_requested` comes with the second.  Disabled (a Task taking the
console over) or hidden, an armed remover returns to rest.

A node's Edit page is three bands and only the middle one scrolls: the
header (the Name, then `(api_name · kind)` in grey -- just `(kind)` while
the name is the type's own -- and the status line) above, the rows, and the
action bar (Remove | Revert, Plot, Start, Stop) as the page's foot, on the
rows' left and right edges.  A long name gives up its middle in the header;
it never widens the window.  A newly opened page starts at its first row;
after that its bar is the operator's, across tab switches.  The first row is
the Name, in the page's one label column, and the first stop of Tab; its
box's tooltip and empty-box hint are the node's internal ID.  It sends
(`name_committed`) what the operator TYPED, once, on Return or when the focus
leaves the box -- a box nobody typed in sends nothing, so it never sends back
a name an agent has since replaced; an emptied box is sent as "", which the
presenter takes as the internal ID.  Escape takes the typing back.  A refused
name is one sentence under the line, in the control column, while the
refused text is in the box; leaving the box puts the node's name back and
sends it, which withdraws the refusal.  It stays editable while a Task holds
the console.  The tab reads `Edit · <name>` and is retitled when the node is
renamed.  A rename is not an edit: it is not part of the draft, so it never
marks a running node `· edited` and Revert leaves it alone.

A running node's page stays editable, and the edits are the next run's.  The
projection's `edited` says the draft differs from the configuration the
current run (running, following or queued) was started with.  While it holds,
Revert -- a standard command in its own fixed slot left of the Plot group,
always shown -- is enabled and raises `revert_requested`
(`logic_revert_requested` on the handle), which makes that whole configuration
the draft again -- replacing it, so a device or artifact input only the edit
named goes too -- without restarting; Restart wears the dirty star and the
primary role (`set_dirty`, the On Pulse rule) with the tooltip "Restart to run
the edited configuration".  A Restart of an unedited run is a standard
command; Start, with nothing running, is the page's primary.  The presenter
also ends the row's and the header's status with `· edited`, and a Restart
it refused while the run goes on adds `; Restart refused: <why>` there until
the next edit.  Only Restart or Revert ends an edit: a following processor
whose source restarts, or a Measurement a Task restarts to acquire from,
rebuilds the configuration it was started with, so the cue stays, and while it
follows its status names that configuration's source even when the edit has
issues -- those are the next run's.  A queued Restart's configuration is the
one the node runs only until the Restart is discarded (Stop, another node
taking its device); then it is the old run's again.

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
set_console_state(word: str) -> None  # idle|running|error
set_selectors(enabled: bool) -> None
set_task_takeover(active: bool) -> None
show_status(text: str, severity: str) -> None
set_summary(text: str) -> None
set_editor_tab_title(editor, title: str) -> None
```

Pause is a toggle command: `set_paused(True)` turns its label to Resume and
holds it down (the active face) until the presenter says the console runs
again; the click only asks.  Its slot fits both words, so the flip moves
nothing beside it.  The header dot is the console's state, which
the handle keeps as the worst of its logic nodes': `error` while one has
failed, `running` while one runs, `idle` otherwise.

The header names the window (`task`) as a label; it is not an input.  Its
one chooser lists `Plot: <kind>` and `<Layer>: <api_name>` entries -- a node
TYPE by its api_name, its layer in words (`zlc_ui.status.name_words`); a
node itself is called by its Name.  The command beside it names what it adds
for the chosen family, `Add panel` or `Add node`, in one slot wide enough
for either.  The handle forwards the board signals and drives the rest;
nothing outside the window calls these directly.

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
