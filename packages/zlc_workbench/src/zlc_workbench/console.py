"""The task console: a window over the same session a notebook drives.

A presenter connects mute views to a live session and does nothing else.  It
holds no physics, no rendering, no signal mechanism -- when it looks like it
wants one, that thing belongs to whichever package owns the subject, and the
presenter's job is to ask it.

The presenter is Qt-free by construction: it receives already-built views and
talks to them through their declared setters and signals.  That is what lets it
be tested headlessly, and it is why the same code path serves the notebook: the
session below it does not know a window exists.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
import time
from typing import Any

from zlc_durable import unique_path

from .board import LiveBoard
from .logic import (
    LogicBinding, LogicCatalog, build_arguments, dataset_inputs, make_host,
)
from .presentation import PlotPanelPort
from .selection import attach_selection_bridge
from .topology import project_signals


__all__ = ["ConsolePresenter", "PanelBinding"]


def _file_stem(text: str) -> str:
    """One panel's name, as something a filesystem will take.

    A signal key is "@logic/cm/frames" and a title is whatever the operator
    typed, so neither can go straight into a path.  Only characters that are
    safe everywhere survive; a name that reduces to nothing falls back rather
    than producing a file called ".png".
    """

    kept = "".join(
        character if (character.isalnum() or character in "-_.") else "-"
        for character in str(text).strip()
    ).strip("-.")
    while "--" in kept:
        kept = kept.replace("--", "-")
    return kept[:60] or "panel"


@dataclass
class PanelBinding:
    """One panel: which signal it shows, and what is drawing it."""

    panel_id: str
    signal: str
    host: Any
    port: PlotPanelPort
    title: str = ""
    #: Which kind of plot this panel IS.  Chosen when it was added and kept
    #: through a retarget: an operator who asked for a curve did not ask for
    #: it to become an image the moment they point it somewhere else.
    kind: str = ""
    #: How big the card is.  Recorded HERE because the card is on the other
    #: side of the wall now: a board that cannot say how big its own panels
    #: are cannot be written down and put back.
    size: str = ""
    #: Live derivation from selections drawn on this panel, if it has one.
    bridge: Any = None
    selections: Any = None
    #: The last failure already shown, so one refusal is reported once.
    reported_error: Any = None


class ConsolePresenter:
    """Wires a console view to a running session."""

    def __init__(
        self,
        session: object,
        view: object,
        *,
        make_host: Callable[[object, str, str], Any],
        panel_kinds: Callable[[], Sequence[tuple[str, str]]] | None = None,
        spec_for: Callable[[object, str], Any] | None = None,
        choose_signal: Callable[[Sequence[tuple]], str | None] | None = None,
        open_saved: Callable[[str], object] | None = None,
        edit_panel: Callable[[Any, str], object] | None = None,
        release_bootstrap: Callable[[], object] | None = None,
        choose_logic: Callable[[Sequence[tuple]], str | None] | None = None,
        edit_logic: Callable[[Any, Mapping[str, Any]], Mapping[str, Any] | None] | None = None,
        intervals: Sequence[int] = (100, 200, 400, 800),
        default_interval_ms: int = 400,
    ) -> None:
        self.session = session
        self.view = view
        self._make_host = make_host
        # What kinds of panel exist, and whether one dataset admits one.  Both
        # belong to the plotting package; this only asks.
        self._panel_kinds = panel_kinds
        self._spec_probe = spec_for
        # Asking is the window's job and answering is not, so the question
        # arrives as a callable: the presenter stays Qt-free and a notebook can
        # answer it with a name it already knows.  The window's own answer is
        # its handle's choose_signal; a notebook passes its own.
        self._choose_signal = choose_signal
        # Reading a saved run is a different window over a different subject,
        # so the console asks for it rather than growing one.
        self._open_saved = open_saved
        # Opening a panel's own plot controls is Qt work the presenter asks for
        # rather than does.
        self._edit_panel = edit_panel
        # The opening monitor holds the camera armed so the first panel is not
        # empty.  A camera IS held by one owner at a time, so the first logic
        # node that wants it cannot have it until this lets go.
        self._release_bootstrap = release_bootstrap
        # A logic node is what publishes a signal; a panel only shows one.  The
        # question "which node type" and the settings form are Qt and arrive as
        # callables, so the presenter stays headless.  The ROW is not one of
        # them any more: the window makes its own rows.
        self._choose_logic = choose_logic
        self._edit_logic = edit_logic
        self.logic: dict[str, LogicBinding] = {}
        self.catalog = LogicCatalog()
        self.panels: dict[str, PanelBinding] = {}
        #: Monotonic, so a panel id is never handed out twice in one session.
        self._panel_serial = 0
        # What every card's picker was last told, so it is only rebuilt when
        # the offer really changed.
        self._offered_groups: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = ()
        self._paused = False
        self._deriving = True
        #: How often a new panel redraws.  The board's default, kept so a panel
        #: and the card that reports it cannot state different numbers.
        self._default_interval_ms = int(default_interval_ms)

        kinds = tuple(self._panel_kinds() if self._panel_kinds is not None else ())
        self._default_panel_kind = kinds[0][0] if kinds else ""
        setter = getattr(self.view, "set_panel_kinds", None)
        if setter is not None:
            setter(kinds, self._default_panel_kind)

        self.board = LiveBoard(
            session.signal_plane,
            lambda: tuple(binding.port for binding in self.panels.values()),
            intervals=intervals,
            default_interval_ms=default_interval_ms,
        )
        self._connect()

    # ------------------------------------------------------------------ wiring

    def _connect(self) -> None:
        """Every outbound signal the view offers gets an answer here.

        A view signal with no listener is a control that looks live and does
        nothing, which is worse than a control that is visibly absent.
        """

        self.view.pause_toggled.connect(self.set_paused)
        self.view.save_requested.connect(self.save)
        self.view.save_image_requested.connect(self.save_images)
        self.view.add_panel_requested.connect(self.add_selected_panel)
        self.view.selectors_toggled.connect(self.set_deriving)
        self.view.load_requested.connect(self.open_saved)
        self.view.add_logic_requested.connect(self.add_chosen_logic)
        self.view.save_board_requested.connect(self.save_board)
        self.view.load_board_requested.connect(self.load_board)
        self.view.panel_order_committed.connect(self.reorder_panels)
        # Every control on a card is a decision about ONE named panel, wired
        # once here rather than re-strung by whoever built the widget.
        self.view.panel_remove_requested.connect(self.remove_panel)
        self.view.panel_signal_picked.connect(self.retarget_panel)
        self.view.panel_size_picked.connect(self.resize_panel)
        self.view.panel_update_ms_picked.connect(self.set_panel_interval)
        self.view.panel_title_committed.connect(self.rename_panel)
        self.view.panel_edit_requested.connect(self.edit_panel)
        self.view.logic_start_requested.connect(self.start_logic)
        self.view.logic_stop_requested.connect(self.stop_logic)
        self.view.logic_edit_requested.connect(self.edit_logic)
        self.view.logic_remove_requested.connect(self.remove_logic)
        self.set_paused(False)
        self.set_deriving(True)

    # ------------------------------------------------------------------ panels

    def add_panel(
        self, signal: str, initial: object, *, title: str = "", kind: str = ""
    ) -> PanelBinding:
        """Show a signal, as ``kind`` when one is asked for.

        With no kind the plotting package decides from the data, which is what
        a notebook wants; the window always names one, because the operator
        picked it beside the button.
        """

        # Never reused.  Minted from the panel COUNT, an id came back the moment
        # anything was removed -- and the second panel to hold it overwrote the
        # first in place: its plotting host and selection bridge were never
        # closed, and two live bridges published derived signals under one name.
        self._panel_serial += 1
        panel_id = f"panel-{self._panel_serial}"
        host = self._make_host(initial, signal, str(kind))
        port = PlotPanelPort(
            panel_id,
            signal,
            host,
            display_interval_ms=self._default_interval_ms,
            shown=initial,
        )
        binding = PanelBinding(panel_id, signal, host, port, title or signal, str(kind))
        self.panels[panel_id] = binding

        # A box drawn on this panel derives a new signal.  The bridge belongs to
        # the runtime and the meaning to the domain; the presenter only connects
        # the two.  Probed, not caught: a host that cannot report selections has
        # no derivation, but a host that can and then fails must say so loudly
        # rather than leave a panel that silently stops answering gestures.
        self._apply_deriving(binding)

        # The id AND the title: a card is asked which panel it is by the board
        # and by the drop-order path, and asked what to caption itself by the
        # operator.  One argument for both meant every card in the real window
        # was captioned "Panel" and knew its own id as its title.
        self.view.add_panel(panel_id, binding.title)
        self.view.show_panel(panel_id, host)
        self.view.set_panel_selectors_enabled(panel_id, self._deriving)
        self._offer_panel(panel_id)
        self._summarise()
        return binding

    def reorder_panels(self, order: Sequence[str]) -> bool:
        """Take the order the operator dragged the cards into.

        Where the cards are IS the panel order: it decides what a saved figure
        contains and in what sequence, and a board that rearranges itself back
        on the next redraw is a board that ignores the operator.
        """

        wanted = [str(panel_id) for panel_id in order if str(panel_id) in self.panels]
        if len(wanted) != len(self.panels):
            wanted += [panel_id for panel_id in self.panels if panel_id not in wanted]
        if wanted == list(self.panels):
            return False
        self.panels = {panel_id: self.panels[panel_id] for panel_id in wanted}
        self.view.set_panel_order(wanted)
        return True

    def _offer_panel(self, panel_id: str) -> None:
        """Tell one card what it may show and how often it is redrawing.

        Its six intents are wired once, at the port, rather than per card:
        every control on a card is a decision about THAT panel, and each used
        to be re-connected by whoever built the widget -- six wires per panel
        across the wall, and a builder that forgot one left a control that
        looked configurable and was not.
        """

        self.view.set_panel_signal_choices(
            panel_id, self.signal_groups(), current=self.panels[panel_id].signal
        )
        # What this panel's redraw interval actually is.  The card used to open
        # its box on a literal of its own.
        self.view.set_panel_update_ms(
            panel_id, self.panels[panel_id].port.display_interval_ms
        )

    def signal_groups(self) -> tuple[tuple[str, tuple[tuple[str, str], ...]], ...]:
        """Every signal a card may be pointed at, gathered under its producer.

        A flat list of forty keys is a list nobody reads; grouped by who
        publishes them, the same forty are four short lists.  The grouping is
        the producer the plane already records, not a naming convention parsed
        out of the key.
        """

        groups: dict[str, list[tuple[str, str]]] = {}
        for name, label, _state, producer, _derived in self.offered_signals(
            include_shown=True
        ):
            groups.setdefault(producer or "signals", []).append((label, name))
        return tuple(
            (producer, tuple(leaves)) for producer, leaves in groups.items()
        )

    def _refresh_signal_choices(self) -> None:
        """Offer newly published signals on cards that are already open.

        Pushed only when the offer actually changes: a combo rebuilt on every
        beat is a combo that closes itself while an operator is reading it.
        """

        groups = self.signal_groups()
        if groups == self._offered_groups:
            return
        self._offered_groups = groups
        for panel_id in self.view.panel_ids():
            binding = self.panels.get(panel_id)
            self.view.set_panel_signal_choices(
                panel_id, groups, current=binding.signal if binding is not None else ""
            )

    def retarget_panel(self, panel_id: str, signal: str) -> bool:
        """Point one panel at a different signal, keeping its place on the board.

        Rebuilt rather than mutated: a plotting host is built around the shape
        of what it draws, and a frame of pixels is not a place to discover that
        the new signal is an image where the old one was a curve.
        """

        binding = self.panels.get(panel_id)
        if binding is None or not signal or signal == binding.signal:
            return False
        value = self.session.signal_plane.freeze().value(str(signal))
        if value is None:
            self._report(f"{signal} has not published yet", severity="warning")
            return False
        title = binding.title if binding.title != binding.signal else str(signal)
        # The panel keeps the kind it was added as -- unless this data cannot
        # be drawn that way, in which case the data decides rather than the
        # card going blank.
        kind = binding.kind
        if kind and self._spec_for(value.snapshot, kind) is None:
            kind = ""
            self._report(
                f"{signal} cannot be drawn as a {binding.kind.replace('_', ' ')};"
                " showing what it fits",
                severity="warning",
            )
        self._release_panel(binding)
        host = self._make_host(value.snapshot, str(signal), kind)
        replaced = PanelBinding(
            panel_id,
            str(signal),
            host,
            PlotPanelPort(
                panel_id,
                str(signal),
                host,
                display_interval_ms=binding.port.display_interval_ms,
                shown=value.snapshot,
            ),
            title,
            kind,
        )
        self.panels[panel_id] = replaced
        self.view.show_panel(panel_id, host)
        self._apply_deriving(replaced)
        self._report(f"{panel_id} now shows {signal}", severity="task")
        self._summarise()
        return True

    def resize_panel(self, panel_id: str, size: str) -> bool:
        """One panel's size preset, applied to the plot as well as the card.

        The card and the figure inside it have to agree: a card resized around
        a figure that stayed 2x2 is a big card with a small picture in it.
        """

        binding = self.panels.get(panel_id)
        if binding is None:
            return False
        set_size = getattr(binding.host, "set_size", None)
        if callable(set_size):
            try:
                result = set_size(str(size))
                if hasattr(result, "result"):
                    result.result()
            except Exception as error:
                self._report(f"{panel_id}: {error}", severity="error")
                return False
        self.view.set_panel_size(panel_id, str(size))
        self.panels[panel_id] = replace(binding, size=str(size))
        return True

    def set_panel_interval(self, panel_id: str, interval_ms: int) -> bool:
        """How often one panel redraws.

        Per panel, not per board: a camera worth watching at 10 Hz sits beside
        a fit result that changes once a run, and one interval for both either
        wastes the machine or hides the camera.
        """

        binding = self.panels.get(panel_id)
        if binding is None:
            return False
        binding.port.set_display_interval(int(interval_ms))
        return True

    def rename_panel(self, panel_id: str, title: str) -> bool:
        binding = self.panels.get(panel_id)
        if binding is None:
            return False
        binding.title = str(title).strip() or binding.signal
        return True

    def edit_panel(self, panel_id: str) -> bool:
        """Open the plot's own semantic controls for one panel.

        What a plot can be told to show belongs to zlc_plot, which already
        offers the panel of controls; this only puts it in front of the
        operator for the panel they clicked.
        """

        binding = self.panels.get(panel_id)
        if binding is None:
            return False
        if self._edit_panel is None:
            self._report("this console cannot open panel settings", severity="warning")
            return False
        try:
            self._edit_panel(binding.host, binding.title)
        except Exception as error:
            self._report(f"{binding.title}: {error}", severity="error")
            return False
        return True

    def _release_panel(self, binding: PanelBinding) -> None:
        """Let go of one panel's derivation and its plotting host."""

        if binding.selections is not None:
            binding.selections.close()
        if binding.bridge is not None:
            binding.bridge.close()
        binding.bridge = binding.selections = None
        binding.host.close()

    def add_selected_panel(self, kind: str = "") -> PanelBinding | None:
        """Add a panel of the kind chosen beside the button.

        That is what the control says it does.  It used to ignore the kind
        entirely and open a modal signal chooser, so the combo beside Add Panel
        described a choice the button did not make -- and a board where every
        signal was already shown opened a blank list.

        The signal is the first published one this kind can actually draw,
        preferring one not already on the board.  Which signal a panel shows is
        a per-panel decision the card's own picker already owns, so asking for
        it up front asked twice.
        """

        wanted = str(kind or self._default_panel_kind)
        frozen = self.session.signal_plane.freeze()
        offered = [name for name, *_rest in self.offered_signals()]
        shown = [name for name, *_rest in self.offered_signals(include_shown=True)]
        if not shown:
            self._report(
                "nothing has published yet, so there is nothing to show",
                severity="warning",
            )
            return None
        for signal in offered + [name for name in shown if name not in offered]:
            value = frozen.value(signal)
            if value is None:
                continue
            if self._spec_for(value.snapshot, wanted) is None:
                continue
            binding = self.add_panel(signal, value.snapshot, kind=wanted)
            self._report(f"showing {binding.title}", severity="task")
            return binding
        self._report(
            f"nothing published can be drawn as a {wanted.replace('_', ' ')}",
            severity="warning",
        )
        return None

    def open_saved(self) -> object | None:
        """Open a saved figure, in whatever window the host provides for it."""

        if self._open_saved is None:
            self._report("this console cannot open saved figures", severity="warning")
            return None
        return self._open_saved(str(self.session.day_folder()))

    def offered_signals(
        self, *, include_shown: bool = False
    ) -> tuple[tuple[str, str, str, str, str], ...]:
        """What exists now, as the rows a chooser renders.

        The Add Panel chooser hides what is already on screen; a card's own
        picker cannot, because the signal it is currently showing has to be one
        of the choices it offers.
        """

        rows = project_signals(
            self.session.signal_plane,
            shown={binding.signal for binding in self.panels.values()},
        )
        return tuple(
            (row.name, row.label, row.state, row.producer, row.derived_from)
            for row in rows
            if include_shown or not row.shown
        )

    def remove_panel(self, panel_id: str) -> None:
        binding = self.panels.pop(panel_id, None)
        if binding is not None:
            self._release_panel(binding)
        self.view.remove_panel(panel_id)
        self._summarise()

    # ------------------------------------------------------------------ running

    def set_paused(self, paused: bool) -> None:
        """The presenter owns the answer; the window is told what it now is."""

        self._paused = bool(paused)
        self.view.set_paused(self._paused)
        self._summarise()

    def set_deriving(self, deriving: bool) -> None:
        """Whether a box drawn on a panel derives a signal from it.

        Off is not cosmetic: a bridge that is closed publishes nothing and
        holds no processor, so an operator who is only looking at data stops
        paying for derivations they did not ask for.
        """

        self._deriving = bool(deriving)
        self.view.set_selectors(self._deriving)
        for panel_id, binding in self.panels.items():
            self._apply_deriving(binding)
            # And the card, which was left showing its selector control live
            # while the bridge behind it was closed.
            self.view.set_panel_selectors_enabled(panel_id, self._deriving)
        self._report(
            "selections derive signals" if self._deriving else "selections are display only",
            severity="task",
        )

    def beat(self) -> None:
        """One display beat.  A paused console still draws nothing new."""

        if self._paused:
            return
        self.board.tick()
        self.board.commit()
        self._report_panel_errors()
        self.poll_logic()
        self._refresh_signal_choices()

    # -------------------------------------------------------------- the board

    #: What a written-down board says it is.  Versioned because a board outlives
    #: the session that drew it -- it is meant to be reopened tomorrow.
    LAYOUT_FORMAT = "zlc.console-board/v1"

    def layout(self) -> dict[str, Any]:
        """The board as a portable document: what is on it, in what order.

        A board is the arrangement an operator built -- which signals, drawn how
        big, redrawing how often, and which nodes are producing them.  It took
        an afternoon to arrange and did not survive closing the window, because
        nothing here could say what it was.  Saving DATA is a different act and
        already had a button; this is the other one.

        Only what an operator chose is written.  Panel ids, hosts and ports are
        this session's bookkeeping and are rebuilt on the way back in.
        """

        return {
            "format": self.LAYOUT_FORMAT,
            "panels": [
                {
                    "signal": binding.signal,
                    "title": binding.title,
                    "kind": binding.kind,
                    "size": binding.size,
                    "interval_ms": int(binding.port.display_interval_ms),
                }
                for binding in self.panels.values()
            ],
            "logic": [
                {
                    "api_name": str(binding.descriptor.api_name),
                    "values": dict(binding.values),
                    "source_signal": str(binding.source_signal),
                }
                for binding in self.logic.values()
            ],
        }

    def apply_layout(self, document: Mapping[str, Any]) -> bool:
        """Put a written-down board back, on whatever is publishing now.

        The nodes go up first: a panel names a signal, and a signal exists only
        because something is producing it.  A panel whose signal nobody
        publishes today is reported and skipped rather than refused wholesale --
        a board saved with four panels and reopened against three live signals
        is still three quarters of an afternoon's work.
        """

        if str(document.get("format", "")) != self.LAYOUT_FORMAT:
            self._report("that file is not a saved board", severity="error")
            return False
        for panel_id in tuple(self.panels):
            self.remove_panel(panel_id)
        for node_id in tuple(self.logic):
            self.remove_logic(node_id)
        for entry in document.get("logic", ()):
            self.add_logic(
                str(entry.get("api_name", "")),
                values=dict(entry.get("values", {})),
                source_signal=str(entry.get("source_signal", "")),
            )
        front = self.session.signal_plane.freeze()
        missing: list[str] = []
        for entry in document.get("panels", ()):
            signal = str(entry.get("signal", ""))
            value = front.value(signal)
            if value is None:
                missing.append(signal)
                continue
            binding = self.add_panel(
                signal,
                value.snapshot,
                title=str(entry.get("title", "")),
                kind=str(entry.get("kind", "")),
            )
            if entry.get("size"):
                self.resize_panel(binding.panel_id, str(entry["size"]))
            if entry.get("interval_ms"):
                self.set_panel_interval(binding.panel_id, int(entry["interval_ms"]))
        if missing:
            self._report(
                f"nothing is publishing {', '.join(sorted(set(missing)))}; "
                "the rest of the board is back",
                severity="warning",
            )
        self._summarise()
        return True

    def save_board(self) -> str:
        """Write the board down, wherever the operator says."""

        import json

        path = self.view.ask_save_path(
            "Save board", str(self.session.day_folder()), "Boards (*.json)"
        )
        if not path:
            return ""
        try:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(self.layout(), handle, indent=1, ensure_ascii=False)
        except Exception as error:
            self._report(f"cannot save the board: {error}", severity="error")
            return ""
        self._report(f"board saved to {path}", severity="task")
        return str(path)

    def load_board(self) -> bool:
        """Put a written-down board back."""

        import json

        path = self.view.ask_open_path(
            "Load board", str(self.session.day_folder()), "Boards (*.json)"
        )
        if not path:
            return False
        try:
            with open(path, encoding="utf-8") as handle:
                document = json.load(handle)
        except Exception as error:
            self._report(f"cannot read that board: {error}", severity="error")
            return False
        return self.apply_layout(document)

    def save(self) -> object | None:
        """Write one archive, and say what happened either way."""

        try:
            return self._save()
        except Exception as error:
            self._report(f"cannot save: {error}", severity="error")
            return None

    def _save(self) -> object | None:
        """Save every panel's current data, with the record that explains it.

        The snapshots go in whole rather than as their values: an archive that
        keeps only the numbers cannot be reopened as the figure it was, and the
        axes are exactly what nobody can reconstruct afterwards.
        """

        # ONE front for the whole archive.  Freezing per panel let a processor
        # deliver between the first panel and the last, so a saved figure could
        # hold half a board from one shot beside half from the next -- the exact
        # incoherence the board-coherent tick exists to prevent, reintroduced at
        # the moment the board is written down.
        front = self.session.signal_plane.freeze()
        arrays: dict[str, Any] = {}
        for binding in self.panels.values():
            value = front.value(binding.signal)
            if value is None:
                continue
            arrays[binding.panel_id] = value.snapshot
        if not arrays:
            self._report("no panel has data to save", severity="warning")
            return None
        written = self.session.save_figure(
            "console",
            arrays=arrays,
            nodes=self._producing_nodes(),
            panel={
                binding.panel_id: {"signal": binding.signal, "title": binding.title}
                for binding in self.panels.values()
            },
        )
        # Where it went, because pressing Save and pressing nothing looked the
        # same: no message on success, none on failure either.
        self._report(f"saved {len(arrays)} panel(s) to {Path(str(written)).name}", severity="task")
        return written

    def _producing_nodes(self) -> tuple[Any, ...]:
        """Everything that produced what is on screen, for the archive to record.

        The session's own nodes AND the ones started in this window.  Only the
        session's were recorded, so a figure saved after running a calibration
        or an occupancy processor here carried provenance for the opening
        monitor and nothing else -- an archive that describes an apparatus
        which produced none of its data.
        """

        nodes = list(getattr(self.session, "nodes", ()))
        for binding in self.logic.values():
            if binding.node is not None and not any(
                item is binding.node for item in nodes
            ):
                nodes.append(binding.node)
        return tuple(nodes)

    def save_images(self) -> tuple[str, ...]:
        """Write every panel as a picture, and say what happened either way."""

        try:
            return self._save_images()
        except Exception as error:
            self._report(f"cannot save images: {error}", severity="error")
            return ()

    def _save_images(self) -> tuple[str, ...]:
        """Write each panel exactly as it looks, beside the day's data.

        The plotting host renders its own file, so what lands on disk is the
        panel rather than a second drawing of the same numbers -- and it lands
        in the day folder the run's data went to, because a picture separated
        from its dataset is the thing nobody can interpret later.
        """

        folder = self.session.day_folder()
        written: list[str] = []
        for binding in self.panels.values():
            # Named for what it SHOWS.  Files went out as panel-1.png, panel-2
            # ... which in a day folder of thirty pictures is a set nobody can
            # tell apart -- and the panel has always carried a title.
            path = unique_path(folder, _file_stem(binding.title or binding.signal), ".png")
            result = binding.host.save(path)
            if hasattr(result, "result"):
                result.result()
            written.append(str(path))
        if written:
            self._report(f"saved {len(written)} image(s) to {folder}", severity="task")
        else:
            self._report("no panels to save", severity="warning")
        return tuple(written)

    def _apply_deriving(self, binding: PanelBinding) -> None:
        """Attach or release one panel's derivation.

        Off means the bridge is gone, not merely quiet: a closed bridge retires
        its processors, so an operator who is only looking at data stops paying
        to re-cut a region on every publication.  Turning it back on builds a
        new one, because closing is final by design -- a bridge that could be
        reopened would have to decide what its old generation now means.
        """

        if self._deriving:
            if binding.bridge is not None or not hasattr(binding.host, "subscribe_selection"):
                return
            binding.bridge, binding.selections = attach_selection_bridge(
                self.session.signal_plane,
                binding.host,
                binding.signal,
                bridge_id=binding.panel_id,
            )
            return
        if binding.selections is not None:
            binding.selections.close()
        if binding.bridge is not None:
            binding.bridge.close()
        binding.bridge = binding.selections = None

    def _report_panel_errors(self) -> None:
        """Say what a gesture could not do, once, where an operator looks.

        These arrive from inside a plot callback whose exceptions are swallowed
        by design.  Unreported, the panel simply stops answering boxes.
        """

        for panel_id, binding in self.panels.items():
            error = getattr(binding.selections, "last_error", None) or getattr(
                binding.port, "last_error", None
            )
            if error is None or error is binding.reported_error:
                continue
            binding.reported_error = error
            self._report(f"{binding.title}: {error}", severity="error")
            # And on the card itself, which has a status line nothing wrote to.
            # A board-wide line says which panel; the panel says it is the one.
            if panel_id in self.view.panel_ids():
                self.view.set_panel_status(panel_id, str(error), error=True)

    def _report(self, text: str, *, severity: str) -> None:
        show = getattr(self.view, "show_status", None)
        if show is not None:
            show(text, severity)


    # ------------------------------------------------------------------- logic

    def logic_offer(self) -> tuple[tuple[str, str, str, str], ...]:
        """Every node type, and what stops each one being added HERE.

        (api_name, kind, what it publishes, why it cannot be built yet).

        The catalog lists what exists; only this bench knows what it can supply,
        so only this can say whether a type is addable.  The chooser used to
        write "available" beside every row -- a claim made by the one place with
        no way to check it -- so picking a node that needs a calibration this
        bench has not produced looked like a broken node instead of an order to
        do things in.

        The reason comes from actually attempting the build, through the same
        function ``add_logic`` uses.  A second copy of "what does this need"
        would be a second answer, and the two would disagree exactly when it
        mattered.
        """

        offer = []
        artifacts = self._logic_artifacts()
        for api_name, kind, publishes in self.catalog.rows():
            descriptor = self.catalog.get(api_name)
            blocked = ""
            try:
                build_arguments(
                    descriptor,
                    installation=self.session.installation,
                    signal_plane=self.session.signal_plane,
                    values={},
                    artifacts=artifacts,
                    extras=self._logic_extras(),
                )
            except Exception as error:
                blocked = str(error)
            offer.append((api_name, kind, publishes, blocked))
        return tuple(offer)

    def add_chosen_logic(self) -> str:
        """Ask which node type, then add one.  Asking is the window's job."""

        if self._choose_logic is None:
            self._report("this console cannot ask which node to add", severity="warning")
            return ""
        chosen = self._choose_logic(self.logic_offer())
        return self.add_logic(str(chosen)) if chosen else ""

    def add_logic(
        self,
        api_name: str,
        *,
        values: Mapping[str, Any] | None = None,
        source_signal: str = "",
    ) -> str:
        """Host one node of a type, ready to start.

        Built but not started: a node that ran the moment it was added would
        fire the sequence before the operator had seen its settings.
        """

        descriptor = self.catalog.get(api_name)
        if descriptor is None:
            self._report(f"no logic node named {api_name!r}", severity="warning")
            return ""
        wants = dataset_inputs(descriptor)
        if wants and not source_signal:
            # A processor is built around the signal it reads, and the runtime
            # refuses to host a reactive node that was never told which one.
            # Nothing asked, so Add Logic -> occupancy failed with "reactive
            # node requires exactly one input signal key" -- a sentence about
            # the runtime, in answer to a question nobody put to the operator.
            if self._choose_signal is None:
                self._report(
                    f"{api_name} reads a signal and this console cannot ask which",
                    severity="warning",
                )
                return ""
            self._report(
                f"which signal should {api_name} read?", severity="task"
            )
            source_signal = str(
                self._choose_signal(self.offered_signals(include_shown=True)) or ""
            )
            if not source_signal:
                return ""

        node_id = self._free_logic_id(descriptor.api_name)
        if descriptor.device_requirements and self._release_bootstrap is not None:
            # A node that needs a device cannot share it with the bootstrap
            # monitor, which armed the camera before any node existed.  Adding
            # one used to fail with "already armed", which reads as a broken
            # node rather than an owner that had not let go.
            release, self._release_bootstrap = self._release_bootstrap, None
            try:
                release()
            except Exception as error:
                self._report(f"cannot release the opening monitor: {error}", severity="error")
                return ""
        try:
            arguments = build_arguments(
                descriptor,
                installation=self.session.installation,
                signal_plane=self.session.signal_plane,
                values=dict(values or {}),
                source_signal=source_signal,
                artifacts=self._logic_artifacts(),
                extras=self._logic_extras(),
            )
            node = descriptor.instantiate(**arguments)
            host = make_host(
                descriptor,
                node,
                signal_plane=self.session.signal_plane,
                instance_id=node_id,
                request_owner_wake=self.board.wake.request_owner_wake,
            )
        except Exception as error:
            self._report(f"cannot add {api_name}: {error}", severity="error")
            return ""
        kind = str(getattr(descriptor.kind, "value", descriptor.kind))
        self.view.add_logic_row(node_id, kind)
        binding = LogicBinding(
            node_id,
            descriptor,
            host,
            node=node,
            values=descriptor.authoring_schema.freeze(dict(values or {})),
            source_signal=str(source_signal),
        )
        self.logic[node_id] = binding
        self._show_logic(binding)
        self._summarise()
        self._report(f"added {node_id}", severity="task")
        return node_id

    def start_logic(self, node_id: str) -> bool:
        binding = self.logic.get(str(node_id))
        if binding is None:
            return False
        try:
            binding.host.start()
        except Exception as error:
            self._report(f"{node_id}: {error}", severity="error")
            self._show_logic(binding)
            return False
        self._show_logic(binding)
        self._summarise()
        self._report(f"{node_id} started", severity="task")
        return True

    def stop_logic(self, node_id: str) -> bool:
        binding = self.logic.get(str(node_id))
        if binding is None:
            return False
        binding.host.cancel("the operator pressed Stop")
        self._show_logic(binding)
        self._summarise()
        return True

    def edit_logic(self, node_id: str) -> bool:
        """Change a node's settings.  Rebuilt, because a node IS its settings.

        A running node is not edited underneath itself: what it is publishing
        was produced by what it was built with, and swapping that mid-run would
        make the record of the run a lie.
        """

        binding = self.logic.get(str(node_id))
        if binding is None:
            return False
        if self._edit_logic is None:
            self._report("this console cannot edit node settings", severity="warning")
            return False
        if binding.host.running:
            self._report(f"stop {node_id} before changing it", severity="warning")
            return False
        edited = self._edit_logic(binding.descriptor, dict(binding.values))
        if edited is None:
            return False
        descriptor, source = binding.descriptor, binding.source_signal
        if not self.remove_logic(node_id):
            return False
        return bool(
            self.add_logic(descriptor.api_name, values=edited, source_signal=source)
        )

    def remove_logic(self, node_id: str) -> bool:
        """Take a node away, once it has actually stopped.

        A running node cannot simply be dropped: it is still holding a camera
        and still publishing, and a row taken off screen while that is true is
        a node nobody can reach to stop.  So Remove asks it to stop and the row
        stays, saying so, until it has -- which the beat notices.  Nothing here
        waits, because what would be waiting is the window.
        """

        binding = self.logic.get(str(node_id))
        if binding is None:
            return False
        binding.removing = True
        if binding.host.running:
            binding.host.cancel("the operator removed this node")
            binding.host.poll()
        if binding.host.running:
            self._show_logic(binding)
            self._report(f"{node_id} is stopping", severity="task")
            return False
        return self._retire_logic(binding)

    def _retire_logic(self, binding: LogicBinding) -> bool:
        """Let go of a node that has stopped."""

        self.logic.pop(binding.node_id, None)
        try:
            binding.host.shutdown()
        except Exception as error:
            self._report(f"{binding.node_id}: {error}", severity="error")
        self.view.remove_logic_row(binding.node_id)
        self._summarise()
        self._report(f"removed {binding.node_id}", severity="task")
        return True

    def poll_logic(self) -> None:
        """One look at every hosted node, and the rows that changed."""

        for binding in tuple(self.logic.values()):
            try:
                binding.host.poll()
            except Exception as error:
                self._report(f"{binding.node_id}: {error}", severity="error")
            if binding.removing and not binding.host.running:
                self._retire_logic(binding)
                continue
            self._show_logic(binding)

    def _show_logic(self, binding: LogicBinding) -> None:
        """What one node is doing, pushed only when it changed.

        A row rewritten every beat is a row an operator cannot read a status
        off, because the text they were halfway through replaced itself.
        """

        observed = binding.host.observation
        if observed.error:
            state, status = "error", observed.error
        elif observed.running:
            state, status = "running", observed.phase
        else:
            state, status = "idle", observed.phase
        published = tuple(
            (name, binding.descriptor.api_name, "live" if observed.running else "held")
            for name in binding.host.published_signals()
        )
        shown = (state, status, published)
        if shown == binding.shown:
            return
        binding.shown = shown
        self.view.set_logic_state(binding.node_id, state, status)
        self.view.set_logic_publishes(binding.node_id, published)

    def _spec_for(self, snapshot: object, kind: str) -> Any:
        """Whether this data can be drawn as ``kind``, as the plotting package sees it.

        A probe, not a guess: the same call that builds the spec answers it, so
        offering a kind and building it cannot disagree.
        """

        if self._spec_probe is None or not kind:
            return object()
        try:
            return self._spec_probe(snapshot, kind)
        except Exception:
            return None

    def _logic_extras(self) -> dict[str, Any]:
        """Facts this bench can supply beyond its devices and the signal plane."""

        extras: dict[str, Any] = {}
        workspace = getattr(self.session, "workspace", None)
        if workspace is not None:
            extras["pulse_search_paths"] = (workspace.pulses,)
        return extras

    def _logic_artifacts(self) -> dict[str, Any]:
        """What the nodes already running here have produced, by contract.

        Some nodes are built ON another node's result: occupancy needs the
        TrapCalibration a calibration task worked out, and declares that as an
        artifact input.  ``build_arguments`` has always taken artifacts and
        nobody ever passed any, so the declaration was answered by nothing and
        occupancy could not be added at all -- before OR after a calibration
        ran, which reads as a broken node rather than an order to do things in.

        A node's descriptor names its outputs and their contracts, and the node
        object carries the finished one under that name.  A task that has not
        run yet raises when asked, which is it saying "not yet", so it is
        simply not on offer.
        """

        produced: dict[str, Any] = {}
        for binding in self.logic.values():
            if binding.node is None:
                continue
            for output in getattr(binding.descriptor, "outputs", ()):
                try:
                    value = getattr(binding.node, output.name)
                except Exception:
                    continue
                if value is not None:
                    produced.setdefault(output.contract_id, value)
        return produced

    def _free_logic_id(self, api_name: str) -> str:
        if api_name not in self.logic:
            return api_name
        index = 2
        while f"{api_name}{index}" in self.logic:
            index += 1
        return f"{api_name}{index}"

    def _summarise(self) -> None:
        state = "paused" if self._paused else "running"
        running = sum(1 for item in self.logic.values() if item.host.running)
        nodes = f", {running}/{len(self.logic)} node(s) running" if self.logic else ""
        self.view.set_summary(f"{len(self.panels)} panel(s), {state}{nodes}")

    def close(self, *, node_stop_seconds: float = 10.0) -> None:
        # Nodes first: one still running publishes into a plane the panels are
        # being taken off, and a worker left alive keeps the process up with no
        # window to show for it.  Here, unlike Remove, waiting is right -- the
        # window is going away and there is nothing left to keep responsive --
        # but it is still bounded, so a wedged node cannot hold the process.
        deadline = time.monotonic() + float(node_stop_seconds)
        for binding in tuple(self.logic.values()):
            binding.removing = True
            if binding.host.running:
                binding.host.cancel("the console is closing")
        while self.logic and time.monotonic() < deadline:
            self.poll_logic()
            if any(item.host.running for item in self.logic.values()):
                time.sleep(0.01)
        for binding in tuple(self.logic.values()):
            self._retire_logic(binding)
        for panel_id in list(self.panels):
            self.remove_panel(panel_id)
        self.board.close()
