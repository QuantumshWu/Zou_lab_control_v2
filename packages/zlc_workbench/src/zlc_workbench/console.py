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

from zlc_plot import parameter_controls
from zlc_plot.primitives import ImageFrame

from .board import LiveBoard
from .console_layout import (
    LAYOUT_FORMAT as CONSOLE_LAYOUT_FORMAT,
    LayoutDocument,
    LayoutError,
    LogicLayoutEntry,
    ResolvedLayout,
    resolve_layout,
)
from .device_use import DeviceClaim, DeviceUseBusy
from .logic import (
    LogicBinding,
    LogicCandidate,
    LogicCatalog,
    LogicDraft,
    artifact_input_specs,
    build_arguments,
    dataset_inputs,
    device_key_options,
    make_host,
    stable_signal_key,
)
from .image_overlay import ImageOverlayResolver, ResolvedImagePresentation
from .panel_save import (
    capture_run_chain,
    save_panel_figure as _save_panel_figure,
)
from .panel_state import PanelFrozenData, PanelState, restore_semantic_choice
from .presentation import PlotPanelPort
from .selection import attach_selection_bridge, subscribe_committed_selection
from .topology import project_signals


__all__ = ["ConsolePresenter", "PanelBinding", "PanelState"]


_UNCHANGED = object()


_QUICK_DISPLAY_FIELDS: Mapping[str, frozenset[str]] = {
    "curve": frozenset(
        {"x_label", "y_label", "show_grid", "relim_mode", "y_min", "y_max"}
    ),
    "image": frozenset(
        {
            "colormap",
            "relim_mode",
            "color_min",
            "color_max",
            "show_colorbar",
        }
    ),
    "histogram": frozenset({"bin_count", "density", "log_y"}),
    "rolling": frozenset({"window", "y_min", "y_max", "show_grid"}),
    "facet_grid": frozenset({"facet_display_unit", "show_grid"}),
}


@dataclass
class PanelBinding:
    """One runtime binding around the panel's single authored state."""

    panel_id: str
    state: PanelState
    host: Any = None
    port: PlotPanelPort | None = None
    #: Edit deliberately keeps one frozen data revision until Refresh.  It is
    #: not panel configuration and therefore does not live in ``PanelState``.
    frozen_data: PanelFrozenData | None = None
    frozen_stale: bool = False
    #: Panel Edit owns a second, frozen plotting surface.  It is deliberately
    #: not the live monitor host and therefore has no PlotPanelPort.
    editor_host: Any = None
    editor_selections: Any = None
    editor_open: bool = False
    #: Live derivation from selections drawn on this panel, if it has one.
    bridge: Any = None
    selections: Any = None
    #: The last failure already shown, so one refusal is reported once.
    reported_error: Any = None
    #: Exact publication used to construct a not-yet-board-anchored host.
    display_publication: Any = None
    #: Image annotation is resolved from exact publications at this composition
    #: boundary.  Its revision is independent of the dataset snapshot revision.
    overlay_resolver: ImageOverlayResolver = field(default_factory=ImageOverlayResolver)
    overlay_revision: int = -1
    overlay_publication: Any = None
    image_presentation: ResolvedImagePresentation | None = None
    #: UI-neutral parameter descriptions projected from this host's public
    #: zlc_plot control plane.  This is editor metadata, not a second authored
    #: state; accepted values still live only in ``state``.
    parameter_surface: Mapping[str, object] = field(default_factory=dict)

    @property
    def signal(self) -> str:
        return self.state.signal

    @property
    def title(self) -> str:
        return self.state.title

    @property
    def kind(self) -> str:
        return self.state.kind

    @property
    def size(self) -> str:
        return self.state.size


@dataclass(frozen=True)
class _LayoutCandidate:
    logic: tuple[LogicBinding, ...]
    panels: tuple[PanelBinding, ...]
    panel_serial: int
    missing_signals: tuple[str, ...] = ()
    incompatible_panels: tuple[tuple[str, str], ...] = ()


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
        open_saved: Callable[[str], object] | None = None,
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
        # Reading a saved run is a different window over a different subject,
        # so the console asks for it rather than growing one.
        self._open_saved = open_saved
        self.logic: dict[str, LogicBinding] = {}
        self.catalog = LogicCatalog()
        self._artifact_completion_order = 0
        self.panels: dict[str, PanelBinding] = {}
        #: Monotonic, so a panel id is never handed out twice in one session.
        self._panel_serial = 0
        # What every card's picker was last told, so it is only rebuilt when
        # the offer really changed.
        self._offered_groups: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = ()
        self._paused = False
        self._deriving = True
        self._shown_console_summary: str | None = None
        #: How often a new panel redraws.  The board's default, kept so a panel
        #: and the card that reports it cannot state different numbers.
        self._default_interval_ms = int(default_interval_ms)

        kinds = tuple(self._panel_kinds() if self._panel_kinds is not None else ())
        self._panel_kind_labels = {
            str(key): str(label or key) for key, label in kinds
        }
        self._default_panel_kind = kinds[0][0] if kinds else ""
        setter = getattr(self.view, "set_panel_kinds", None)
        if setter is not None:
            setter(kinds, self._default_panel_kind)
        logic_setter = getattr(self.view, "set_logic_kinds", None)
        if logic_setter is not None:
            logic_setter(self.logic_offer())

        self.board = LiveBoard(
            session.signal_plane,
            lambda: tuple(
                binding.port
                for binding in self.panels.values()
                if binding.port is not None
            ),
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
        self.view.save_layout_requested.connect(self.save_layout)
        self.view.load_layout_requested.connect(self.load_layout)
        self.view.save_screenshot_requested.connect(self.save_screenshot)
        self.view.add_panel_requested.connect(self.add_selected_panel)
        self.view.selectors_toggled.connect(self.set_deriving)
        self.view.add_logic_requested.connect(self.add_logic)
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
        draft_changed = getattr(self.view, "logic_draft_changed", None)
        if draft_changed is not None:
            draft_changed.connect(self._logic_draft_changed)
        panel_changed = getattr(self.view, "panel_state_changed", None)
        if panel_changed is not None:
            panel_changed.connect(self.update_panel_state)
        refresh_requested = getattr(
            self.view, "panel_snapshot_refresh_requested", None
        )
        if refresh_requested is not None:
            refresh_requested.connect(self.refresh_panel_snapshot)
        producer_apply = getattr(self.view, "panel_producer_apply_requested", None)
        if producer_apply is not None:
            producer_apply.connect(self.apply_panel_producer)
        save_figure = getattr(self.view, "panel_save_figure_requested", None)
        if save_figure is not None:
            save_figure.connect(self.save_panel_figure)
        editor_closed = getattr(self.view, "panel_editor_closed", None)
        if editor_closed is not None:
            editor_closed.connect(self._panel_editor_closed)
        self.set_paused(False)
        self.set_deriving(True)

    # ------------------------------------------------------------------ panels

    def add_blank_panel(
        self,
        kind: str,
        *,
        signal: str = "",
        title: str = "",
        size: str = "2x2",
        interval_ms: int | None = None,
        semantic: Mapping[str, Any] | None = None,
        display: Mapping[str, Any] | None = None,
        fit: Mapping[str, Any] | None = None,
        site_overlay: str = "off",
    ) -> PanelBinding | None:
        """Author one fixed-kind panel before any signal has published.

        A panel is configuration first and a plot host second.  The empty card
        is intentional: its Setting/Edit projections own the later signal
        choice, while the plot surface is mounted only when that signal has an
        actual compatible publication.
        """

        wanted = str(kind or self._default_panel_kind)
        if wanted not in self._panel_kind_labels:
            self._report(
                f"{wanted.replace('_', ' ') or 'that plot kind'} is not available "
                "on TaskConsole",
                severity="warning",
            )
            return None

        self._panel_serial += 1
        panel_id = f"panel-{self._panel_serial}"
        base_title = self._panel_kind_labels[wanted]
        used_titles = {binding.state.title for binding in self.panels.values()}
        generated_title = base_title
        suffix = 2
        while generated_title in used_titles:
            generated_title = f"{base_title} {suffix}"
            suffix += 1
        state = PanelState(
            signal=str(signal).strip(),
            kind=wanted,
            size=str(size or "2x2"),
            interval_ms=(
                self._default_interval_ms
                if interval_ms is None
                else int(interval_ms)
            ),
            title=str(title).strip() or generated_title,
            semantic=dict(semantic or {}),
            display=dict(display or {}),
            fit=dict(fit or {}),
            site_overlay=str(site_overlay),
        )
        binding = PanelBinding(panel_id, state)
        self.panels[panel_id] = binding
        self.view.add_panel(panel_id, state.title)
        self.view.show_panel(panel_id, None)
        self.view.set_panel_selectors_enabled(panel_id, self._deriving)
        self._publish_panel_state(binding)
        self._refresh_console_projection()

        # Layout restore may already name a live signal.  Re-enter the exact
        # same state-replacement path Setting uses; no second mounting path.
        if state.signal:
            self.update_panel_state(panel_id, {"signal": state.signal})
        return binding

    def add_panel(
        self,
        signal: str,
        initial: object,
        *,
        title: str = "",
        kind: str = "",
        size: str = "",
        interval_ms: int | None = None,
        semantic: Mapping[str, Any] | None = None,
        display: Mapping[str, Any] | None = None,
        fit: Mapping[str, Any] | None = None,
        site_overlay: str = "off",
        initial_publication: object | None = None,
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
        state = PanelState(
            signal=str(signal),
            kind=str(kind),
            size=str(size),
            interval_ms=(
                self._default_interval_ms
                if interval_ms is None
                else int(interval_ms)
            ),
            title=str(title).strip() or str(signal),
            semantic=dict(semantic or {}),
            display=dict(display or {}),
            fit=dict(fit or {}),
            site_overlay=str(site_overlay),
        )
        front = self.session.signal_plane.freeze()
        current = front.value(state.signal)
        publication = initial_publication
        if (
            publication is None
            and current is not None
            and current.snapshot.ref == initial.ref
        ):
            publication = front.publication(state.signal)
        exact_value = self._publication_value(publication, state.signal)

        # Build the binding first so plot projection callbacks share its one
        # overlay resolver/revision stream from the initial frame onward.
        binding = PanelBinding(panel_id, state, None, None)
        plot_input = initial
        if exact_value is not None and publication is not None:
            plot_input = self._project_panel_input(
                binding, exact_value, publication, state=state
            )

        # Hosts are created from the OwnedSnapshot contract; an Image overlay
        # is then applied through its public overlay seam.  This keeps host
        # factories schema-oriented while the panel still presents one atomic
        # ImageFrame transaction.
        host = self._make_host(initial, state.signal, state.kind)
        try:
            self._configure_panel_host(host, state)
            self._apply_plot_input_overlay(host, plot_input)
            if not state.size:
                state = replace(state, size=self._panel_host_size(host))
                binding.state = state
            binding.parameter_surface = self._describe_panel_parameters(host, state)
            state = self._state_with_described_semantics(
                state, binding.parameter_surface
            )
            binding.state = state
        except Exception:
            host.close()
            raise
        port = PlotPanelPort(
            panel_id,
            state.signal,
            host,
            display_interval_ms=state.interval_ms,
            shown=plot_input,
            project_input=lambda value, pub: self._project_panel_input(
                binding, value, pub
            ),
            replace_host=lambda projected, value, pub: self._replace_panel_host(
                binding, projected, value, pub
            ),
            on_presented=lambda pub, projected: self._panel_presented(
                binding, pub, projected
            ),
        )
        binding.host = host
        binding.port = port
        binding.display_publication = publication
        binding.frozen_data = self._panel_frozen_data(
            binding,
            snapshot=initial,
            publication=publication,
            plot_input=plot_input,
        )
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
        self.view.add_panel(panel_id, binding.state.title)
        self.view.show_panel(panel_id, host)
        self.view.set_panel_selectors_enabled(panel_id, self._deriving)
        self._publish_panel_state(binding)
        self._refresh_console_projection()
        return binding

    @staticmethod
    def _publication_value(publication: object | None, signal: str) -> object | None:
        value = getattr(publication, "value", None)
        return value(str(signal)) if callable(value) else None

    def _project_panel_input(
        self,
        binding: PanelBinding,
        value: object,
        publication: object,
        *,
        state: PanelState | None = None,
    ) -> object:
        """Compose the exact plot input for one immutable publication."""

        selected = binding.state if state is None else state
        snapshot = getattr(value, "snapshot", None)
        if snapshot is None:
            raise TypeError("a panel signal value must carry an OwnedSnapshot")
        if selected.kind != "image":
            return snapshot
        previous = binding.image_presentation
        if (
            binding.overlay_publication is publication
            and previous is not None
            and previous.requested_mode == selected.site_overlay
            and previous.frame.snapshot is snapshot
        ):
            return previous.frame
        binding.overlay_revision += 1
        presentation = binding.overlay_resolver.resolve(
            value,
            publication,
            mode=selected.site_overlay,
            overlay_revision=binding.overlay_revision,
        )
        binding.overlay_publication = publication
        binding.image_presentation = presentation
        return presentation.frame

    def _apply_plot_input_overlay(self, host: object, plot_input: object) -> None:
        if not isinstance(plot_input, ImageFrame):
            return
        update = getattr(host, "update_image_overlay", None)
        if not callable(update):
            raise TypeError("an Image panel host must accept a typed point overlay")
        self._await_panel_operation(update(plot_input.overlay))

    @staticmethod
    def _overlay_annotation(
        binding: PanelBinding,
        publication: object | None,
        plot_input: object,
    ) -> dict[str, Any]:
        presentation = binding.image_presentation
        if (
            presentation is None
            or binding.overlay_publication is not publication
            or presentation.frame is not plot_input
        ):
            return {}
        annotation: dict[str, Any] = {
            "requested_mode": presentation.requested_mode,
            "resolved_mode": presentation.resolved_mode,
        }
        if presentation.calibration_path is not None:
            annotation["calibration_path"] = presentation.calibration_path
        if presentation.note is not None:
            annotation["note"] = presentation.note
        return annotation

    def _panel_frozen_data(
        self,
        binding: PanelBinding,
        *,
        snapshot: object,
        publication: object | None,
        plot_input: object,
    ) -> PanelFrozenData:
        return PanelFrozenData(
            binding.state.signal,
            publication,
            snapshot,
            plot_input,
            capture_run_chain(self.session.signal_plane, publication),
            self._overlay_annotation(binding, publication, plot_input),
        )

    def _panel_presented(
        self,
        binding: PanelBinding,
        publication: object,
        _plot_input: object,
    ) -> None:
        """Track the exact live event separately from Panel Edit's frozen one."""

        binding.display_publication = publication
        stale = (
            binding.frozen_data is not None
            and binding.frozen_data.publication is not publication
        )
        if stale != binding.frozen_stale:
            binding.frozen_stale = stale
            self.refresh_panel_editor(binding.panel_id)

    def _replace_panel_host(
        self,
        binding: PanelBinding,
        plot_input: object,
        value: object,
        publication: object,
    ) -> object:
        """Replace a plot host at a signal-generation boundary."""

        host = self._make_host(value.snapshot, binding.state.signal, binding.state.kind)
        try:
            self._configure_panel_host(host, binding.state)
            self._apply_plot_input_overlay(host, plot_input)
            binding.parameter_surface = self._describe_panel_parameters(
                host, binding.state
            )
        except Exception:
            host.close()
            raise

        old_host = binding.host
        if binding.selections is not None:
            binding.selections.close()
        if binding.bridge is not None:
            binding.bridge.close()
        binding.bridge = binding.selections = None
        binding.host = host
        binding.display_publication = publication
        self.view.show_panel(binding.panel_id, host)
        self._apply_deriving(binding)
        if old_host is not None:
            old_host.close()
        return host

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
            panel_id, self.signal_groups(), current=self.panels[panel_id].state.signal
        )
        # What this panel's redraw interval actually is.  The card used to open
        # its box on a literal of its own.
        self.view.set_panel_update_ms(
            panel_id, self.panels[panel_id].state.interval_ms
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
                panel_id,
                groups,
                current=binding.state.signal if binding is not None else "",
            )
        front = self.session.signal_plane.freeze()
        for panel_id, binding in tuple(self.panels.items()):
            if (
                binding.host is None
                and binding.state.signal
                and front.value(binding.state.signal) is not None
            ):
                self.update_panel_state(panel_id, {"signal": binding.state.signal})

    def retarget_panel(self, panel_id: str, signal: str) -> bool:
        """Point one fixed-kind panel at a different compatible signal."""

        return self.update_panel_state(panel_id, {"signal": str(signal)})

    def resize_panel(self, panel_id: str, size: str) -> bool:
        """One panel's size preset, applied to the plot as well as the card.

        The card and the figure inside it have to agree: a card resized around
        a figure that stayed 2x2 is a big card with a small picture in it.
        """

        return self.update_panel_state(panel_id, {"size": str(size)})

    def set_panel_interval(self, panel_id: str, interval_ms: int) -> bool:
        """How often one panel redraws.

        Per panel, not per board: a camera worth watching at 10 Hz sits beside
        a fit result that changes once a run, and one interval for both either
        wastes the machine or hides the camera.
        """

        return self.update_panel_state(
            panel_id, {"interval_ms": int(interval_ms)}
        )

    def rename_panel(self, panel_id: str, title: str) -> bool:
        return self.update_panel_state(panel_id, {"title": str(title)})

    def edit_panel(self, panel_id: str) -> bool:
        """Open or focus the panel's non-modal Edit projection."""

        binding = self.panels.get(panel_id)
        if binding is None:
            return False
        projection = self.panel_editor_projection(panel_id)
        opened = getattr(self.view, "open_panel_editor", None)
        focused = getattr(self.view, "focus_panel_editor", None)
        if callable(opened):
            opened(panel_id, projection)
            binding.editor_open = True
            if binding.editor_host is None and binding.frozen_data is not None:
                try:
                    self._replace_panel_editor_host(binding)
                except Exception as error:
                    self._report(
                        f"cannot open {binding.state.title} plot editor: {error}",
                        severity="error",
                    )
                    return False
        if callable(focused):
            focused(panel_id)
        if not callable(opened) and not callable(focused):
            self._report("this console cannot open panel settings", severity="warning")
            return False
        return True

    def update_panel_state(
        self,
        panel_id: str,
        patch: Mapping[str, Any],
    ) -> bool:
        """Replace the one state read by the card, editor and plot binding."""

        binding = self.panels.get(str(panel_id))
        if binding is None:
            return False
        changes = dict(patch)
        allowed = {
            "signal",
            "kind",
            "size",
            "interval_ms",
            "title",
            "semantic",
            "display",
            "fit",
            "site_overlay",
        }
        unknown = tuple(name for name in changes if name not in allowed)
        if unknown:
            self._report(
                f"{panel_id}: unknown panel state field {unknown[0]!r}",
                severity="error",
            )
            return False
        current = binding.state
        if "kind" in changes and str(changes["kind"]) != current.kind:
            self._report(
                f"{panel_id}: plot kind is fixed; add another panel to use "
                f"{str(changes['kind']).replace('_', ' ')}",
                severity="warning",
            )
            return False

        signal = str(changes.get("signal", current.signal)).strip()
        title = str(changes.get("title", current.title)).strip()
        if signal != current.signal and "title" not in changes and current.title == current.signal:
            title = signal
        title = (
            title
            or signal
            or self._panel_kind_labels.get(current.kind, current.kind.replace("_", " "))
        )
        merged: dict[str, Any] = {
            "signal": signal,
            "size": str(changes.get("size", current.size)),
            "interval_ms": int(changes.get("interval_ms", current.interval_ms)),
            "title": title,
            "site_overlay": str(changes.get("site_overlay", current.site_overlay)),
        }
        for name in ("semantic", "display", "fit"):
            values = dict(getattr(current, name))
            if name in changes:
                values.update(dict(changes[name]))
            merged[name] = values
        try:
            candidate = replace(current, **merged)
        except Exception as error:
            self._report(f"{panel_id}: {error}", severity="error")
            return False
        needs_mount = bool(candidate.signal) and (
            candidate.signal != current.signal
            or binding.host is None
            or binding.port is None
        )
        if candidate == current and not needs_mount:
            return False

        host_patch = dict(changes)
        if candidate.title != current.title:
            host_patch["title"] = candidate.title

        if not candidate.signal:
            if binding.host is not None or binding.port is not None:
                self._release_panel_editor(binding)
                self._release_panel(binding)
                self.view.show_panel(panel_id, None)
            binding.state = candidate
            binding.parameter_surface = {}
            binding.frozen_data = None
            binding.frozen_stale = False
            binding.display_publication = None
            self._publish_panel_state(binding)
            self._refresh_console_projection()
            return True

        if needs_mount:
            front = self.session.signal_plane.freeze()
            value = front.value(candidate.signal)
            if value is None:
                if candidate.signal != current.signal and binding.host is not None:
                    self._report(
                        f"{candidate.signal} has not published yet",
                        severity="warning",
                    )
                    return False
                binding.state = candidate
                binding.parameter_surface = {}
                self._publish_panel_state(binding)
                self._refresh_console_projection()
                self._report(
                    f"{candidate.signal} has not published yet; {panel_id} remains ready",
                    severity="warning",
                )
                return True
            if candidate.kind and self._spec_for(value.snapshot, candidate.kind) is None:
                if candidate.signal != current.signal and binding.host is not None:
                    self._report(
                        f"{candidate.signal} cannot be drawn as a "
                        f"{candidate.kind.replace('_', ' ')}",
                        severity="warning",
                    )
                    return False
                binding.state = candidate
                binding.parameter_surface = {}
                self._publish_panel_state(binding)
                self._refresh_console_projection()
                self._report(
                    f"{candidate.signal} cannot be drawn as a "
                    f"{candidate.kind.replace('_', ' ')}; the panel remains ready",
                    severity="warning",
                )
                return True
            publication = front.publication(candidate.signal)
            try:
                plot_input = self._project_panel_input(
                    binding,
                    value,
                    publication,
                    state=candidate,
                )
                host = self._make_host(
                    value.snapshot, candidate.signal, candidate.kind
                )
                self._configure_panel_host(host, candidate)
                self._apply_plot_input_overlay(host, plot_input)
                parameter_surface = self._describe_panel_parameters(host, candidate)
                candidate = self._state_with_described_semantics(
                    candidate, parameter_surface
                )
                if binding.editor_host is not None:
                    self._apply_panel_host_patch(
                        binding.editor_host, current, candidate, host_patch
                    )
            except Exception as error:
                if "host" in locals():
                    host.close()
                self._report(f"{panel_id}: {error}", severity="error")
                return False
            self._release_panel(binding)
            binding.host = host
            binding.port = PlotPanelPort(
                panel_id,
                candidate.signal,
                host,
                display_interval_ms=candidate.interval_ms,
                shown=plot_input,
                project_input=lambda current_value, pub: self._project_panel_input(
                    binding, current_value, pub
                ),
                replace_host=lambda projected, current_value, pub: self._replace_panel_host(
                    binding, projected, current_value, pub
                ),
                on_presented=lambda pub, projected: self._panel_presented(
                    binding, pub, projected
                ),
            )
            binding.display_publication = publication
            binding.state = candidate
            binding.parameter_surface = parameter_surface
            if binding.frozen_data is None:
                binding.frozen_data = self._panel_frozen_data(
                    binding,
                    snapshot=value.snapshot,
                    publication=publication,
                    plot_input=plot_input,
                )
                binding.frozen_stale = False
            else:
                binding.frozen_stale = True
            binding.reported_error = None
            self.view.show_panel(panel_id, host)
            self._apply_deriving(binding)
            if (
                binding.editor_open
                and binding.editor_host is None
                and not binding.frozen_stale
            ):
                try:
                    self._replace_panel_editor_host(binding)
                except Exception as error:
                    self._report(
                        f"cannot mount {binding.state.title} plot editor: {error}",
                        severity="error",
                    )
            if "site_overlay" in changes:
                self._refresh_panel_overlay_state(binding, candidate)
            self._report(
                f"{panel_id} now shows {candidate.signal}", severity="task"
            )
        else:
            if binding.host is None or binding.port is None:
                binding.state = candidate
                self._publish_panel_state(binding)
                self._refresh_console_projection()
                return True
            try:
                self._apply_panel_host_patch(binding.host, current, candidate, host_patch)
                if binding.editor_host is not None:
                    self._apply_panel_host_patch(
                        binding.editor_host, current, candidate, host_patch
                    )
                if "site_overlay" in changes:
                    self._refresh_panel_overlay_state(binding, candidate)
                if candidate.interval_ms != current.interval_ms:
                    binding.port.set_display_interval(candidate.interval_ms)
            except Exception as error:
                self._report(f"{panel_id}: {error}", severity="error")
                return False
            binding.state = candidate
            binding.parameter_surface = self._describe_panel_parameters(
                binding.host, candidate
            )

        self._publish_panel_state(binding)
        self._refresh_console_projection()
        return True

    @staticmethod
    def _await_panel_operation(operation: object) -> object:
        return operation.result() if hasattr(operation, "result") else operation

    def _panel_host_size(self, host: object) -> str:
        describe = getattr(host, "describe_display", None)
        if not callable(describe):
            return ""
        operation = self._await_panel_operation(describe())
        description = getattr(operation, "value", operation)
        return str(getattr(description, "size", ""))

    @staticmethod
    def _plot_operation_value(operation: object) -> object:
        """Unwrap one public raster operation without knowing its host type."""

        resolved = ConsolePresenter._await_panel_operation(operation)
        return getattr(resolved, "value", resolved)

    @staticmethod
    def _control_document(control: object) -> dict[str, object]:
        """Project zlc_plot's frontend-neutral control into plain typed data."""

        semantic = bool(getattr(control, "semantic", False))
        choices: list[tuple[str, object]] = []
        for choice in tuple(getattr(control, "choices", ())):
            if semantic:
                value, label = choice
            else:
                value, label = choice, str(choice).replace("_", " ").title()
            choices.append((str(label), value))
        kind = getattr(getattr(control, "kind", ""), "value", None)
        return {
            "key": str(getattr(control, "name")),
            "label": str(getattr(control, "label")),
            "kind": str(kind or getattr(control, "kind", "text")),
            "value": getattr(control, "value", None),
            "allow_none": bool(getattr(control, "allow_none", False)),
            "choices": tuple(choices),
            "minimum": getattr(control, "minimum", None),
            "maximum": getattr(control, "maximum", None),
            "step": getattr(control, "step", None),
        }

    def _describe_panel_parameters(
        self,
        host: object,
        state: PanelState,
    ) -> Mapping[str, object]:
        """Read the complete parameter surface from zlc_plot's public host API."""

        describe_display = getattr(host, "describe_display", None)
        describe_semantics = getattr(host, "describe_semantics", None)
        if not callable(describe_display) or not callable(describe_semantics):
            return {}
        display_description = self._plot_operation_value(describe_display())
        semantic_description = self._plot_operation_value(describe_semantics())
        display_controls = parameter_controls(
            display_description.parameter_schema,
            display_description.display_state.values,
            choice_overrides=display_description.parameter_choices,
        )
        semantic_entries = tuple(
            {
                "key": str(field.name),
                "label": str(field.label),
                "kind": "choice",
                "value": field.value,
                "allow_none": not bool(field.required),
                "choices": tuple(
                    (str(label), value) for value, label in tuple(field.choices)
                ),
                "minimum": None,
                "maximum": None,
                "step": None,
            }
            for field in tuple(semantic_description.fields)
            if str(field.name) != "kind"
        )
        quick = _QUICK_DISPLAY_FIELDS.get(state.kind, frozenset())
        display_entries: list[dict[str, object]] = []
        site_overlay: dict[str, object] | None = None
        for control in display_controls:
            entry = self._control_document(control)
            name = str(entry["key"])
            if name == "site_overlay":
                site_overlay = entry
                site_overlay["value"] = state.site_overlay
                continue
            # Panel title has a dedicated shared field, so it must not be
            # offered again as an independent display override.
            if name == "title":
                continue
            entry["quick"] = name in quick
            display_entries.append(entry)

        models_member = getattr(host, "fit_models", ())
        models_operation = models_member() if callable(models_member) else models_member
        models = tuple(self._plot_operation_value(models_operation) or ())
        current_model = state.fit.get("model")
        model_choices = [
            (str(getattr(model, "display_name")), str(getattr(model, "model_id")))
            for model in models
        ]
        if current_model is not None and not any(
            current_model == value for _label, value in model_choices
        ):
            model_choices.insert(0, (str(current_model), current_model))
        fit_entries = (
            {
                "key": "model",
                "label": "Fit model",
                "kind": "choice",
                "value": current_model,
                "allow_none": True,
                "choices": tuple(model_choices),
                "minimum": None,
                "maximum": None,
                "step": None,
            },
        ) if models or current_model is not None else ()
        return {
            "semantic": semantic_entries,
            "display": tuple(display_entries),
            "fit": fit_entries,
            "site_overlay": site_overlay,
        }

    @staticmethod
    def _state_with_described_semantics(
        state: PanelState,
        surface: Mapping[str, object],
    ) -> PanelState:
        """Keep layout-loaded semantic choices typed in the one PanelState."""

        if not state.semantic:
            return state
        described = {
            str(entry["key"]): entry.get("value")
            for entry in tuple(surface.get("semantic", ()))
        }
        resolved = {
            name: described.get(name, value)
            for name, value in state.semantic.items()
        }
        return replace(state, semantic=resolved)

    def _configure_panel_host(self, host: object, state: PanelState) -> None:
        """Apply authored overrides through zlc_plot's public control plane."""

        for name, value in state.semantic.items():
            apply_semantic = getattr(host, "apply_semantic", None)
            if callable(apply_semantic):
                self._await_panel_operation(
                    apply_semantic(
                        name,
                        self._restore_panel_semantic(host, name, value),
                    )
                )
        display = dict(state.display)
        display["title"] = state.title
        if state.kind == "image":
            display["site_overlay"] = state.site_overlay
        if display:
            set_parameters = getattr(host, "set_parameters", None)
            if callable(set_parameters):
                self._await_panel_operation(set_parameters(display))
        if state.size:
            set_size = getattr(host, "set_size", None)
            if callable(set_size):
                self._await_panel_operation(set_size(state.size))
        self._apply_panel_fit(host, {}, state.fit)

    def _restore_panel_semantic(
        self,
        host: object,
        name: str,
        saved: object,
    ) -> object:
        """Resolve a JSON layout value through this host's typed choices."""

        describe = getattr(host, "describe_semantics", None)
        if not callable(describe):
            return saved
        description = self._plot_operation_value(describe())
        return restore_semantic_choice(description, name, saved)

    @staticmethod
    def _compatible_fit_ids(host: object) -> frozenset[str]:
        models_member = getattr(host, "fit_models", ())
        operation = models_member() if callable(models_member) else models_member
        models = tuple(ConsolePresenter._plot_operation_value(operation) or ())
        return frozenset(str(getattr(model, "model_id")) for model in models)

    def _apply_panel_fit(
        self,
        host: object,
        current: Mapping[str, Any],
        candidate: Mapping[str, Any],
    ) -> None:
        """Submit a compatible fit asynchronously through the public host API."""

        before = current.get("model")
        selected = candidate.get("model")
        if selected == before:
            return
        if selected is None:
            clear_fit = getattr(host, "clear_fit", None)
            if callable(clear_fit):
                clear_fit()
            return
        if str(selected) not in self._compatible_fit_ids(host):
            return
        fit = getattr(host, "fit", None)
        if callable(fit):
            # Fit completion intentionally stays asynchronous.  zlc_plot owns
            # revision/generation acceptance and paints only a current result.
            fit(str(selected), live=True)

    def _apply_panel_host_patch(
        self,
        host: object,
        current: PanelState,
        candidate: PanelState,
        patch: Mapping[str, Any],
    ) -> None:
        if "semantic" in patch:
            apply_semantic = getattr(host, "apply_semantic", None)
            if callable(apply_semantic):
                for name, value in dict(patch["semantic"]).items():
                    self._await_panel_operation(
                        apply_semantic(
                            name,
                            self._restore_panel_semantic(host, name, value),
                        )
                    )
        display_patch = dict(patch.get("display", {}))
        if "title" in patch:
            display_patch["title"] = candidate.title
        if "site_overlay" in patch and candidate.kind == "image":
            display_patch["site_overlay"] = candidate.site_overlay
        if display_patch:
            set_parameters = getattr(host, "set_parameters", None)
            if callable(set_parameters):
                self._await_panel_operation(set_parameters(display_patch))
        if candidate.size != current.size:
            set_size = getattr(host, "set_size", None)
            if callable(set_size):
                self._await_panel_operation(set_size(candidate.size))
        if "fit" in patch:
            self._apply_panel_fit(host, current.fit, candidate.fit)

    def _refresh_panel_overlay_state(
        self,
        binding: PanelBinding,
        state: PanelState,
    ) -> None:
        """Recompose live and frozen Image annotation for one state change."""

        if state.kind != "image":
            return
        if binding.host is None or binding.port is None:
            return
        frozen = binding.frozen_data
        if frozen is not None and frozen.publication is not None:
            frozen_value = self._publication_value(
                frozen.publication, frozen.signal
            )
            if frozen_value is not None:
                frozen_input = self._project_panel_input(
                    binding,
                    frozen_value,
                    frozen.publication,
                    state=state,
                )
                binding.frozen_data = replace(
                    frozen,
                    plot_input=frozen_input,
                    overlay=self._overlay_annotation(
                        binding, frozen.publication, frozen_input
                    ),
                )
                if binding.editor_host is not None:
                    self._apply_plot_input_overlay(
                        binding.editor_host, frozen_input
                    )
                    self._refresh_panel_editor_selection(binding)

        publication = (
            binding.port.presented_publication() or binding.display_publication
        )
        value = self._publication_value(publication, state.signal)
        if value is None or publication is None:
            return
        plot_input = self._project_panel_input(
            binding,
            value,
            publication,
            state=state,
        )
        self._apply_plot_input_overlay(binding.host, plot_input)

    def _direct_producer_node_id(self, signal: str) -> str | None:
        for binding in self.logic.values():
            if any(
                stable_signal_key(binding.node_id, output.name) == str(signal)
                for output in binding.descriptor.outputs
            ):
                return binding.node_id
        return None

    def panel_editor_projection(self, panel_id: str) -> dict[str, Any] | None:
        """Plain, widget-free state consumed by the non-modal Panel Edit tab."""

        binding = self.panels.get(str(panel_id))
        if binding is None:
            return None
        frozen = binding.frozen_data
        producer_node_id = self._direct_producer_node_id(binding.state.signal)
        return {
            "panel_id": binding.panel_id,
            "state": binding.state.document(),
            "parameter_surface": binding.parameter_surface,
            "signal_options": self.signal_groups(),
            "kind_read_only": True,
            "frozen_signal": None if frozen is None else frozen.signal,
            "frozen_publication": None if frozen is None else frozen.publication,
            "frozen_snapshot": None if frozen is None else frozen.snapshot,
            "stale": bool(binding.frozen_stale),
            "producer_node_id": producer_node_id,
            "producer_logic": (
                None
                if producer_node_id is None
                else self.logic_editor_projection(producer_node_id)
            ),
        }

    def refresh_panel_editor(self, panel_id: str) -> bool:
        projection = self.panel_editor_projection(panel_id)
        if projection is None:
            return False
        update = getattr(self.view, "update_panel_editor", None)
        if callable(update):
            update(str(panel_id), projection)
        return True

    def _replace_panel_editor_host(self, binding: PanelBinding) -> object:
        """Mount a new independent host for Edit's exact frozen plot input."""

        frozen = binding.frozen_data
        if frozen is None:
            raise RuntimeError(f"{binding.panel_id} has no frozen plot input")
        plot_input = (
            frozen.snapshot if frozen.plot_input is None else frozen.plot_input
        )
        # Host factories choose a spec from the underlying dataset schema.
        # ImageFrame's independently revisioned overlay is applied below as
        # part of this same frozen presentation transaction.
        initial = getattr(plot_input, "snapshot", plot_input)
        host = self._make_host(initial, frozen.signal, binding.state.kind)
        selections = None
        try:
            self._configure_panel_host(host, binding.state)
            self._apply_plot_input_overlay(host, plot_input)
            selections = subscribe_committed_selection(
                host,
                lambda selection, expected=frozen, expected_host=host: (
                    self._route_panel_editor_selection(
                        binding.panel_id,
                        expected_host,
                        expected,
                        selection,
                    )
                ),
            )
        except Exception:
            if selections is not None:
                selections.close()
            host.close()
            raise

        mount = getattr(self.view, "show_panel_editor", None)
        if not callable(mount):
            selections.close()
            host.close()
            raise RuntimeError("this console cannot mount a Panel Edit plot surface")

        old_host = binding.editor_host
        old_selections = binding.editor_selections
        binding.editor_host = host
        binding.editor_selections = selections
        try:
            mount(binding.panel_id, host)
        except Exception:
            binding.editor_host = old_host
            binding.editor_selections = old_selections
            selections.close()
            host.close()
            raise
        if old_selections is not None:
            old_selections.close()
        if old_host is not None:
            old_host.close()
        return host

    def _refresh_panel_editor_selection(self, binding: PanelBinding) -> None:
        """Rebind one unchanged editor host to a replaced frozen record."""

        host = binding.editor_host
        frozen = binding.frozen_data
        if host is None or frozen is None:
            return
        selections = subscribe_committed_selection(
            host,
            lambda selection, expected=frozen, expected_host=host: (
                self._route_panel_editor_selection(
                    binding.panel_id,
                    expected_host,
                    expected,
                    selection,
                )
            ),
        )
        previous = binding.editor_selections
        binding.editor_selections = selections
        if previous is not None:
            previous.close()

    def _release_panel_editor(self, binding: PanelBinding) -> None:
        """Detach and close Edit's subscription and frozen plotting host."""

        host = binding.editor_host
        selections = binding.editor_selections
        binding.editor_host = None
        binding.editor_selections = None
        mount = getattr(self.view, "show_panel_editor", None)
        if callable(mount) and host is not None:
            mount(binding.panel_id, None)
        try:
            if selections is not None:
                selections.close()
        finally:
            if host is not None:
                host.close()

    def _panel_editor_closed(self, panel_id: str) -> None:
        binding = self.panels.get(str(panel_id))
        if binding is not None:
            binding.editor_open = False
            self._release_panel_editor(binding)

    def close_panel_editor(self, panel_id: str) -> bool:
        binding = self.panels.get(str(panel_id))
        if binding is not None:
            binding.editor_open = False
            self._release_panel_editor(binding)
        close = getattr(self.view, "close_panel_editor", None)
        if callable(close):
            close(str(panel_id))
            return True
        return False

    def refresh_panel_snapshot(self, panel_id: str) -> bool:
        binding = self.panels.get(str(panel_id))
        if binding is None:
            return False
        front = self.session.signal_plane.freeze()
        value = front.value(binding.state.signal)
        if value is None:
            self._report(
                f"{binding.state.signal} has not published yet",
                severity="warning",
            )
            return False
        publication = front.publication(binding.state.signal)
        plot_input = self._project_panel_input(binding, value, publication)
        frozen = self._panel_frozen_data(
            binding,
            snapshot=value.snapshot,
            publication=publication,
            plot_input=plot_input,
        )
        previous = binding.frozen_data
        previous_stale = binding.frozen_stale
        binding.frozen_data = frozen
        binding.frozen_stale = False
        if binding.editor_host is not None:
            try:
                self._replace_panel_editor_host(binding)
            except Exception as error:
                binding.frozen_data = previous
                binding.frozen_stale = previous_stale
                self._report(
                    f"cannot refresh {binding.state.title} plot editor: {error}",
                    severity="error",
                )
                return False
        self.refresh_panel_editor(panel_id)
        return True

    def save_panel_figure(self, panel_id: str) -> object | None:
        """Save only the exact frozen data currently shown in Panel Edit."""

        binding = self.panels.get(str(panel_id))
        if binding is None:
            return None
        frozen = binding.frozen_data
        if frozen is None:
            self._report(
                f"{panel_id} has no frozen data to save",
                severity="warning",
            )
            return None
        selected = self.view.ask_save_path(
            "Save panel figure",
            str(self.session.day_folder()),
            "Panel figure (*.png *.npz)",
        )
        if not selected:
            return None
        try:
            written = _save_panel_figure(
                selected,
                state=binding.state,
                frozen=frozen,
                make_host=self._make_host,
                configure_host=self._configure_panel_host,
            )
        except Exception as error:
            self._report(
                f"cannot save {binding.state.title}: {error}",
                severity="error",
            )
            return None
        self._report(
            f"panel saved to {written.image.name} and {written.archive.name}",
            severity="task",
        )
        return written

    def apply_panel_producer(self, panel_id: str) -> bool:
        binding = self.panels.get(str(panel_id))
        if binding is None:
            return False
        producer_node_id = self._direct_producer_node_id(binding.state.signal)
        if producer_node_id is None:
            self._report(
                f"{panel_id} has no editable direct producer",
                severity="warning",
            )
            return False
        return self.start_logic(producer_node_id)

    def _publish_panel_state(self, binding: PanelBinding) -> None:
        """Push one accepted replacement to every view of the same state."""

        set_state = getattr(self.view, "set_panel_state", None)
        if callable(set_state):
            set_state(binding.panel_id, binding.state)
        set_parameter_surface = getattr(
            self.view, "set_panel_parameter_surface", None
        )
        if callable(set_parameter_surface):
            set_parameter_surface(binding.panel_id, binding.parameter_surface)
        self._offer_panel(binding.panel_id)
        if binding.state.size:
            self.view.set_panel_size(binding.panel_id, binding.state.size)
        set_title = getattr(self.view, "set_panel_title", None)
        if callable(set_title):
            set_title(binding.panel_id, binding.state.title)
        self.refresh_panel_editor(binding.panel_id)

    def _release_panel(self, binding: PanelBinding) -> None:
        """Let go of one panel's derivation and its plotting host."""

        if binding.selections is not None:
            binding.selections.close()
        if binding.bridge is not None:
            binding.bridge.close()
        binding.bridge = binding.selections = None
        host = binding.host
        binding.host = None
        binding.port = None
        if host is not None:
            host.close()

    def add_selected_panel(self, kind: str = "") -> PanelBinding | None:
        """Author the chosen fixed kind; signal wiring is a later panel edit."""

        binding = self.add_blank_panel(str(kind or self._default_panel_kind))
        if binding is not None:
            self._report(
                f"added {binding.title}; choose a signal in Setting",
                severity="task",
            )
        return binding

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
            self._release_panel_editor(binding)
            self._release_panel(binding)
        self.close_panel_editor(panel_id)
        self.view.remove_panel(panel_id)
        self._refresh_console_projection()

    # ------------------------------------------------------------------ running

    def set_paused(self, paused: bool) -> None:
        """The presenter owns the answer; the window is told what it now is."""

        self._paused = bool(paused)
        self.view.set_paused(self._paused)
        self._refresh_console_projection()

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

    LAYOUT_FORMAT = CONSOLE_LAYOUT_FORMAT

    def _layout_document(self) -> LayoutDocument:
        """Freeze the current stopped authoring state into its one codec model."""

        return LayoutDocument(
            tuple(binding.state for binding in self.panels.values()),
            tuple(
                LogicLayoutEntry(
                    binding.node_id,
                    str(binding.descriptor.api_name),
                    binding.draft.values,
                    binding.draft.source_signal,
                    binding.draft.device_keys,
                    binding.draft.artifact_inputs,
                )
                for binding in self.logic.values()
            ),
        )

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

        return self._layout_document().to_tree()

    def _external_signal_contracts(self) -> tuple[tuple[str, str], ...]:
        rows: list[tuple[str, str]] = []
        for node in tuple(getattr(self.session, "nodes", ()) or ()):
            signal_key = getattr(node, "signal_key", None)
            if not callable(signal_key):
                continue
            for declaration in tuple(
                getattr(node, "dataset_output_declarations", ()) or ()
            ):
                rows.append(
                    (
                        str(signal_key(declaration.name)),
                        str(declaration.contract_id),
                    )
                )
        return tuple(rows)

    def _prepare_layout_panels(self, resolved: ResolvedLayout) -> _LayoutCandidate:
        """Build every drawable panel off-board before retiring current state."""

        front = self.session.signal_plane.freeze()
        serial = self._panel_serial
        panels: list[PanelBinding] = []
        missing: list[str] = []
        incompatible: list[tuple[str, str]] = []
        used_titles: set[str] = set()
        try:
            for saved in resolved.panels:
                serial += 1
                panel_id = f"panel-{serial}"
                base_title = self._panel_kind_labels[saved.kind]
                generated_title = base_title
                suffix = 2
                while generated_title in used_titles:
                    generated_title = f"{base_title} {suffix}"
                    suffix += 1
                state = replace(
                    saved,
                    size=saved.size or "2x2",
                    title=saved.title.strip() or generated_title,
                )
                used_titles.add(state.title)
                binding = PanelBinding(panel_id, state)
                panels.append(binding)
                if not state.signal:
                    continue
                value = front.value(state.signal)
                if value is None:
                    missing.append(state.signal)
                    continue
                if self._spec_for(value.snapshot, state.kind) is None:
                    incompatible.append((state.signal, state.kind))
                    continue
                publication = front.publication(state.signal)
                plot_input = self._project_panel_input(
                    binding,
                    value,
                    publication,
                    state=state,
                )
                host = self._make_host(value.snapshot, state.signal, state.kind)
                try:
                    self._configure_panel_host(host, state)
                    self._apply_plot_input_overlay(host, plot_input)
                    surface = self._describe_panel_parameters(host, state)
                    state = self._state_with_described_semantics(state, surface)
                except Exception:
                    host.close()
                    raise
                binding.state = state
                binding.host = host
                binding.parameter_surface = surface
                binding.port = PlotPanelPort(
                    panel_id,
                    state.signal,
                    host,
                    display_interval_ms=state.interval_ms,
                    shown=plot_input,
                    project_input=lambda current, pub, item=binding: (
                        self._project_panel_input(item, current, pub)
                    ),
                    replace_host=lambda projected, current, pub, item=binding: (
                        self._replace_panel_host(item, projected, current, pub)
                    ),
                    on_presented=lambda pub, projected, item=binding: (
                        self._panel_presented(item, pub, projected)
                    ),
                )
                binding.display_publication = publication
                binding.frozen_data = self._panel_frozen_data(
                    binding,
                    snapshot=value.snapshot,
                    publication=publication,
                    plot_input=plot_input,
                )
        except Exception as error:
            for binding in panels:
                if binding.host is not None:
                    binding.host.close()
                    binding.host = None
                    binding.port = None
            raise LayoutError(f"cannot prepare the layout panels: {error}") from error
        return _LayoutCandidate(
            resolved.logic,
            tuple(panels),
            serial,
            tuple(missing),
            tuple(incompatible),
        )

    def _build_layout_candidate(self, document: LayoutDocument) -> _LayoutCandidate:
        resolved = resolve_layout(
            document,
            catalog=self.catalog,
            installation=self.session.installation,
            panel_kinds=tuple(self._panel_kind_labels),
            external_outputs=self._external_signal_contracts(),
        )
        return self._prepare_layout_panels(resolved)

    def _commit_layout_candidate(self, candidate: _LayoutCandidate) -> None:
        """Replace the board once, after its complete candidate already exists."""

        for panel_id, binding in tuple(self.panels.items()):
            self.close_panel_editor(panel_id)
            self._release_panel(binding)
            self.view.remove_panel(panel_id)
        for node_id, binding in tuple(self.logic.items()):
            if binding.host is not None:
                binding.host.shutdown()
            close_editor = getattr(self.view, "close_logic_editor", None)
            if callable(close_editor):
                close_editor(node_id)
            self.view.remove_logic_row(node_id)

        self.panels.clear()
        self.logic.clear()
        self._panel_serial = candidate.panel_serial
        self._offered_groups = ()
        self._shown_console_summary = None

        for binding in candidate.logic:
            self.logic[binding.node_id] = binding
            kind = str(
                getattr(binding.descriptor.kind, "value", binding.descriptor.kind)
            )
            self.view.add_logic_row(binding.node_id, kind)
        for binding in candidate.panels:
            self.panels[binding.panel_id] = binding
            self.view.add_panel(binding.panel_id, binding.state.title)
            self.view.show_panel(binding.panel_id, binding.host)
            self.view.set_panel_selectors_enabled(binding.panel_id, self._deriving)
            self._publish_panel_state(binding)
            self._apply_deriving(binding)
        self._refresh_console_projection()
        self._refresh_signal_choices()

    def apply_layout(
        self,
        document: Mapping[str, Any] | LayoutDocument,
    ) -> bool:
        """Put a written-down board back, on whatever is publishing now.

        The nodes go up first: a panel names a signal, and a signal exists only
        because something is producing it.  A panel whose signal nobody
        publishes today is reported and skipped rather than refused wholesale --
        a board saved with four panels and reopened against three live signals
        is still three quarters of an afternoon's work.
        """

        if any(
            binding.pending is not None
            or (binding.host is not None and binding.host.running)
            for binding in self.logic.values()
        ):
            self._report(
                "stop running logic before loading a board",
                severity="warning",
            )
            return False
        try:
            parsed = (
                document
                if isinstance(document, LayoutDocument)
                else LayoutDocument.from_tree(document)
            )
            candidate = self._build_layout_candidate(parsed)
        except Exception as error:
            self._report(f"cannot load the layout: {error}", severity="error")
            return False
        self._commit_layout_candidate(candidate)
        if candidate.missing_signals:
            self._report(
                "nothing is publishing "
                f"{', '.join(sorted(set(candidate.missing_signals)))}; "
                "those panels remain available for rewiring",
                severity="warning",
            )
        if candidate.incompatible_panels:
            descriptions = ", ".join(
                f"{signal} as {kind}"
                for signal, kind in candidate.incompatible_panels
            )
            self._report(
                f"cannot draw {descriptions}; those panels remain available for rewiring",
                severity="warning",
            )
        return True

    def save_layout(self) -> str:
        """Write only the stopped, reusable pipeline/layout document."""

        path = self.view.ask_save_path(
            "Save TaskConsole layout", str(self.session.day_folder()), "Layouts (*.json)"
        )
        if not path:
            return ""
        try:
            self._layout_document().write(path)
        except Exception as error:
            self._report(f"cannot save the layout: {error}", severity="error")
            return ""
        self._report(f"layout saved to {path}", severity="task")
        return str(path)

    def load_layout(self) -> bool:
        """Restore a layout as stopped drafts without building devices."""

        path = self.view.ask_open_path(
            "Load TaskConsole layout", str(self.session.day_folder()), "Layouts (*.json)"
        )
        if not path:
            return False
        try:
            document = LayoutDocument.read(path)
        except Exception as error:
            self._report(f"cannot read that layout: {error}", severity="error")
            return False
        return self.apply_layout(document)

    def save_screenshot(self) -> str:
        """Save one ordinary image of the whole current TaskConsole GUI."""

        path = self.view.ask_save_path(
            "Save TaskConsole screenshot",
            str(self.session.day_folder()),
            "PNG images (*.png)",
        )
        if not path:
            return ""
        target = Path(path)
        if not target.suffix:
            target = target.with_suffix(".png")
        try:
            written = self.view.save_screenshot(str(target))
        except Exception as error:
            self._report(f"cannot save screenshot: {error}", severity="error")
            return ""
        self._report(f"screenshot saved to {written}", severity="task")
        return str(written)

    def _apply_deriving(self, binding: PanelBinding) -> None:
        """Attach or release one panel's derivation.

        Off means the bridge is gone, not merely quiet: a closed bridge retires
        its processors, so an operator who is only looking at data stops paying
        to re-cut a region on every publication.  Turning it back on builds a
        new one, because closing is final by design -- a bridge that could be
        reopened would have to decide what its old generation now means.
        """

        if self._deriving:
            if (
                binding.host is None
                or binding.bridge is not None
                or not hasattr(binding.host, "subscribe_selection")
            ):
                return
            binding.bridge, binding.selections = attach_selection_bridge(
                self.session.signal_plane,
                binding.host,
                binding.signal,
                bridge_id=binding.panel_id,
                on_committed=lambda selection: self._route_panel_selection(
                    binding.panel_id, selection
                ),
            )
            return
        if binding.selections is not None:
            binding.selections.close()
        if binding.bridge is not None:
            binding.bridge.close()
        binding.bridge = binding.selections = None

    def _route_panel_selection(self, panel_id: str, selection: object) -> None:
        """Apply one committed semantic selection to its direct producer draft.

        The descriptor owns whether a selection means anything and how its
        coordinates map to authored fields.  This method only follows the
        panel's exact signal publication back to the row and supplies public
        run-time device readback as data-only context.
        """

        binding = self.panels.get(str(panel_id))
        if binding is None:
            return
        if binding.port is None:
            return
        publication = binding.port.presented_publication()
        if publication is None:
            # A newly-created host already displays ``shown`` before the first
            # board beat anchors its generation.  ``display_publication`` was
            # frozen beside that exact immutable snapshot; never substitute a
            # newer latest value for what the operator selected on screen.
            publication = binding.display_publication
        self._route_exact_panel_selection(
            panel_id,
            binding.state.signal,
            publication,
            selection,
        )

    def _route_panel_editor_selection(
        self,
        panel_id: str,
        host: object,
        frozen: PanelFrozenData,
        selection: object,
    ) -> None:
        """Route only a commit from the still-current, non-stale frozen view."""

        binding = self.panels.get(str(panel_id))
        if (
            binding is None
            or binding.editor_host is not host
            or binding.frozen_data is not frozen
            or binding.frozen_stale
        ):
            return
        self._route_exact_panel_selection(
            panel_id,
            frozen.signal,
            frozen.publication,
            selection,
            expected_snapshot=frozen.snapshot,
        )

    def _route_exact_panel_selection(
        self,
        panel_id: str,
        signal: str,
        publication: object | None,
        selection: object,
        *,
        expected_snapshot: object | None = None,
    ) -> None:
        """Map one selection using only the publication behind its surface."""

        if publication is None:
            raise RuntimeError(
                f"{panel_id} selection has no exact displayed publication"
            )
        value = self._publication_value(publication, signal)
        if value is None:
            raise RuntimeError(
                f"{panel_id} selection publication does not contain {signal}"
            )
        if (
            expected_snapshot is not None
            and getattr(value, "snapshot", None) is not expected_snapshot
        ):
            raise RuntimeError(
                f"{panel_id} selection publication is not its frozen snapshot"
            )
        producer_node_id = self._direct_producer_node_id(signal)
        if producer_node_id is None:
            return
        producer = self.logic.get(producer_node_id)
        if producer is None:
            return

        record = getattr(publication, "run_record", {})
        snapshots = (
            record.get("device_snapshots", {})
            if isinstance(record, Mapping)
            else {}
        )
        context: dict[str, Any] = {"device_snapshots": snapshots}
        if isinstance(snapshots, Mapping) and len(snapshots) == 1:
            actual = next(iter(snapshots.values()))
            if isinstance(actual, Mapping):
                context.update(actual)
        patch = producer.descriptor.selection_patch(
            selection,
            draft=dict(producer.draft.values),
            context=context,
        )
        if patch is not None:
            self.update_logic_draft(producer_node_id, values=patch)

    def _report_panel_errors(self) -> None:
        """Say what a gesture could not do, once, where an operator looks.

        These arrive from inside a plot callback whose exceptions are swallowed
        by design.  Unreported, the panel simply stops answering boxes.
        """

        for panel_id, binding in self.panels.items():
            error = (
                getattr(binding.editor_selections, "last_error", None)
                or getattr(binding.selections, "last_error", None)
                or getattr(binding.port, "last_error", None)
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
        """Every addable row type without resolving or building a run."""

        return tuple(
            (api_name, kind, publishes, "")
            for api_name, kind, publishes in self.catalog.rows()
        )

    def add_logic(
        self,
        api_name: str,
        *,
        node_id: str = "",
        values: Mapping[str, Any] | None = None,
        source_signal: str = "",
        device_keys: Mapping[str, str] | None = None,
        artifact_inputs: Mapping[str, str] | None = None,
        open_editor: bool = True,
    ) -> str:
        """Create one stopped row draft; Start is the first build boundary."""

        descriptor = self.catalog.get(api_name)
        if descriptor is None:
            self._report(f"no logic node named {api_name!r}", severity="warning")
            return ""
        selected_id = str(node_id).strip() or self._free_logic_id(descriptor.api_name)
        if selected_id in self.logic:
            self._report(f"logic row {selected_id!r} already exists", severity="error")
            return ""
        drafted_values = {
            field.name: field.default for field in descriptor.authoring_schema.fields
        }
        drafted_values.update(dict(values or {}))
        options = device_key_options(
            descriptor,
            installation=self.session.installation,
        )
        selected_devices = {
            requirement.argument_name: str(
                dict(device_keys or {}).get(
                    requirement.argument_name,
                    options[requirement.argument_name][0]
                    if options[requirement.argument_name]
                    else "",
                )
            )
            for requirement in descriptor.device_requirements
        }
        selected_artifacts = self._default_artifact_inputs(descriptor)
        supplied_artifacts = dict(artifact_inputs or {})
        unknown_artifacts = set(supplied_artifacts) - set(selected_artifacts)
        if unknown_artifacts:
            self._report(
                f"{descriptor.api_name} has no artifact inputs "
                f"{sorted(unknown_artifacts)!r}",
                severity="error",
            )
            return ""
        selected_artifacts.update(
            {str(name): str(path) for name, path in supplied_artifacts.items()}
        )
        kind = str(getattr(descriptor.kind, "value", descriptor.kind))
        self.view.add_logic_row(selected_id, kind)
        binding = LogicBinding(
            selected_id,
            descriptor,
            LogicDraft(
                values=drafted_values,
                source_signal=str(source_signal),
                device_keys=selected_devices,
                artifact_inputs=selected_artifacts,
            ),
        )
        self.logic[selected_id] = binding
        for other_id in self.logic:
            if other_id != selected_id:
                self.refresh_logic_editor(other_id)
        self._refresh_console_projection()
        if open_editor:
            self._open_logic_editor(binding)
        self._report(f"added {selected_id}", severity="task")
        return selected_id

    def logic_editor_projection(self, node_id: str) -> dict[str, Any] | None:
        """Plain state consumed by Logic Edit and future producer projections."""

        binding = self.logic.get(str(node_id))
        if binding is None:
            return None
        from .authoring_form import (
            display_value,
            project_artifact_inputs,
            project_schema,
        )

        options = device_key_options(
            binding.descriptor,
            installation=self.session.installation,
        )
        artifact_specs = artifact_input_specs(binding.descriptor)
        workspace = getattr(self.session, "workspace", None)
        artifact_base_dir = str(getattr(workspace, "data", ""))
        return {
            "node_id": binding.node_id,
            "api_name": str(binding.descriptor.api_name),
            "kind": str(
                getattr(binding.descriptor.kind, "value", binding.descriptor.kind)
            ),
            "form_spec": project_schema(binding.descriptor.authoring_schema),
            "form_values": {
                name: display_value(value)
                for name, value in binding.draft.values.items()
            },
            "artifact_form_spec": project_artifact_inputs(
                artifact_specs,
                base_dir=artifact_base_dir,
            ),
            "artifact_values": dict(binding.draft.artifact_inputs),
            "artifact_results": self._artifact_results(binding),
            "source_required": bool(dataset_inputs(binding.descriptor)),
            "source_signal": binding.draft.source_signal,
            "source_options": self._source_options(binding.descriptor),
            "device_keys": dict(binding.draft.device_keys),
            "device_options": options,
            "running": bool(binding.host is not None and binding.host.running),
            "pending": binding.pending is not None,
            "error": binding.draft_error,
        }

    def _open_logic_editor(self, binding: LogicBinding) -> bool:
        projection = self.logic_editor_projection(binding.node_id)
        opened = getattr(self.view, "open_logic_editor", None)
        focused = getattr(self.view, "focus_logic_editor", None)
        if callable(opened):
            opened(binding.node_id, projection)
        if callable(focused):
            focused(binding.node_id)
        return callable(opened) or callable(focused)

    def refresh_logic_editor(self, node_id: str) -> bool:
        projection = self.logic_editor_projection(node_id)
        if projection is None:
            return False
        update = getattr(self.view, "update_logic_editor", None)
        if callable(update):
            update(str(node_id), projection)
        for panel_id, panel in self.panels.items():
            if self._direct_producer_node_id(panel.state.signal) == str(node_id):
                self.refresh_panel_editor(panel_id)
        return True

    def update_logic_draft(
        self,
        node_id: str,
        *,
        values: Mapping[str, Any] | None = None,
        source_signal: object = _UNCHANGED,
        device_keys: Mapping[str, str] | None = None,
        artifact_inputs: Mapping[str, str] | None = None,
    ) -> bool:
        """Patch the row draft without mutating its current run."""

        binding = self.logic.get(str(node_id))
        if binding is None:
            return False
        if values is not None:
            binding.draft.values.update(dict(values))
        if source_signal is not _UNCHANGED:
            binding.draft.source_signal = str(source_signal)
        if device_keys is not None:
            binding.draft.device_keys.update(
                {str(name): str(key) for name, key in device_keys.items()}
            )
        if artifact_inputs is not None:
            binding.draft.artifact_inputs.update(
                {str(name): str(path) for name, path in artifact_inputs.items()}
            )
        binding.draft_error = ""
        self._refresh_console_projection()
        self.refresh_logic_editor(binding.node_id)
        return True

    def _logic_draft_changed(self, node_id: str, patch: Mapping[str, Any]) -> None:
        source = patch["source_signal"] if "source_signal" in patch else _UNCHANGED
        self.update_logic_draft(
            node_id,
            values=patch.get("values"),
            source_signal=source,
            device_keys=patch.get("device_keys"),
            artifact_inputs=patch.get("artifact_inputs"),
        )

    def start_logic(self, node_id: str) -> bool:
        binding = self.logic.get(str(node_id))
        if binding is None:
            return False
        try:
            candidate = self._build_logic_candidate(binding)
        except Exception as error:
            binding.draft_error = str(error)
            self._report(f"{node_id}: {error}", severity="error")
            self._refresh_console_projection()
            self.refresh_logic_editor(binding.node_id)
            return False

        binding.draft_error = ""
        self._discard_pending(binding)
        try:
            candidate.reservation = self.session.device_use.prepare_logic(
                binding.owner_token,
                binding.node_id,
                candidate.claims,
                stop=candidate.host.cancel,
                superseded=lambda: self._discard_candidate(binding, candidate),
            )
        except DeviceUseBusy as error:
            self._discard_candidate(binding, candidate)
            binding.draft_error = str(error)
            self._report(f"{node_id}: {error}", severity="error")
            self._refresh_console_projection()
            self.refresh_logic_editor(binding.node_id)
            return False
        except Exception as error:
            self._discard_candidate(binding, candidate)
            binding.draft_error = str(error)
            self._report(f"{node_id}: {error}", severity="error")
            self._refresh_console_projection()
            self.refresh_logic_editor(binding.node_id)
            return False

        blockers = set(candidate.reservation.waiting_for)
        if blockers:
            binding.pending = candidate
            self._refresh_console_projection()
            self.refresh_logic_editor(binding.node_id)
            self._report(
                f"{node_id} queued while {', '.join(sorted(blockers))} stops",
                severity="task",
            )
            return True
        return self._activate_candidate(binding, candidate)

    def stop_logic(self, node_id: str) -> bool:
        binding = self.logic.get(str(node_id))
        if binding is None:
            return False
        self._discard_pending(binding)
        if binding.host is not None:
            binding.host.cancel("the operator pressed Stop")
        self._refresh_console_projection()
        self.refresh_logic_editor(binding.node_id)
        return True

    def edit_logic(self, node_id: str) -> bool:
        binding = self.logic.get(str(node_id))
        if binding is None:
            return False
        return self._open_logic_editor(binding)

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
        self._discard_pending(binding)
        if binding.host is not None and binding.host.running:
            binding.host.cancel("the operator removed this node")
            binding.host.poll()
        if binding.host is not None and binding.host.running:
            self._refresh_console_projection()
            self._report(f"{node_id} is stopping", severity="task")
            return False
        return self._retire_logic(binding)

    def _retire_logic(self, binding: LogicBinding) -> bool:
        """Let go of a node that has stopped."""

        self._discard_pending(binding)
        if binding.host is not None:
            if binding.host.running:
                self._report(
                    f"{binding.node_id} is still stopping",
                    severity="warning",
                )
                return False
            try:
                binding.host.shutdown()
            except Exception as error:
                self._report(f"{binding.node_id}: {error}", severity="error")
                return False
        if binding.lease is not None:
            binding.lease.release()
            binding.lease = None
        self.logic.pop(binding.node_id, None)
        close_editor = getattr(self.view, "close_logic_editor", None)
        if callable(close_editor):
            close_editor(binding.node_id)
        self.view.remove_logic_row(binding.node_id)
        for other_id in self.logic:
            self.refresh_logic_editor(other_id)
        for panel_id in self.panels:
            self.refresh_panel_editor(panel_id)
        self._refresh_console_projection()
        self._report(f"removed {binding.node_id}", severity="task")
        return True

    def poll_logic(self) -> None:
        """One look at every hosted node, and the rows that changed."""

        for binding in tuple(self.logic.values()):
            if binding.host is not None:
                try:
                    binding.host.poll()
                except Exception as error:
                    self._report(f"{binding.node_id}: {error}", severity="error")
                self._capture_artifact_results(binding)
                if not binding.host.running and binding.lease is not None:
                    binding.lease.release()
                    binding.lease = None
            if binding.removing and (
                binding.host is None or not binding.host.running
            ):
                if self._retire_logic(binding):
                    continue

        for binding in tuple(self.logic.values()):
            candidate = binding.pending
            if candidate is None:
                continue
            if not candidate.waiting_for:
                binding.pending = None
                self._activate_candidate(binding, candidate)
        self._refresh_console_projection()

    def _show_logic(self, binding: LogicBinding) -> None:
        """What one node is doing, pushed only when it changed.

        A row rewritten every beat is a row an operator cannot read a status
        off, because the text they were halfway through replaced itself.
        """

        host = binding.host
        if host is None:
            state = "error" if binding.draft_error else "idle"
            status = binding.draft_error or "not started"
        else:
            observed = host.observation
            if observed.error:
                state, status = "error", observed.error
            elif observed.running:
                state, status = "running", observed.phase
            elif binding.draft_error:
                state, status = "error", binding.draft_error
            else:
                state, status = "idle", observed.phase
        if binding.pending is not None:
            waiting = ", ".join(sorted(binding.pending.waiting_for))
            status = f"waiting for {waiting}" if waiting else "restart queued"
        names = (
            host.published_signals()
            if host is not None
            else tuple(
                stable_signal_key(binding.node_id, output.name)
                for output in binding.descriptor.outputs
            )
        )
        published = tuple(
            (
                name,
                binding.descriptor.api_name,
                "live" if host is not None and host.running else "held",
            )
            for name in names
        )
        artifacts = self._artifact_results(binding)
        shown = (state, status, published, artifacts)
        if shown == binding.shown:
            return
        binding.shown = shown
        self.view.set_logic_state(binding.node_id, state, status)
        self.view.set_logic_publishes(binding.node_id, published)
        self.refresh_logic_editor(binding.node_id)

    def _source_options(self, descriptor: Any) -> tuple[str, ...]:
        """Stable keys whose declared Dataset contract matches this input."""

        contracts = {
            str(spec.contract_id) for spec in dataset_inputs(descriptor)
        }
        if not contracts:
            return ()
        compatible: set[str] = set()
        for binding in self.logic.values():
            for output in binding.descriptor.outputs:
                if str(output.contract_id) in contracts:
                    compatible.add(stable_signal_key(binding.node_id, output.name))
        for node in tuple(getattr(self.session, "nodes", ()) or ()):
            signal_key = getattr(node, "signal_key", None)
            if not callable(signal_key):
                continue
            for declaration in tuple(
                getattr(node, "dataset_output_declarations", ()) or ()
            ):
                if str(getattr(declaration, "contract_id", "")) in contracts:
                    compatible.add(str(signal_key(declaration.name)))
        return tuple(sorted(compatible))

    def _validate_dataset_source(self, binding: LogicBinding) -> None:
        wants = dataset_inputs(binding.descriptor)
        if not wants:
            return
        source = binding.draft.source_signal.strip()
        if not source:
            raise ValueError("source_signal must be selected")
        if source not in self._source_options(binding.descriptor):
            contracts = ", ".join(spec.contract_id for spec in wants)
            raise ValueError(
                f"{source!r} is not declared as a compatible {contracts} Dataset"
            )
        if self.session.signal_plane.latest_publication(source) is None:
            raise LookupError(f"{source!r} has not published a Dataset yet")

    def _build_logic_candidate(self, binding: LogicBinding) -> LogicCandidate:
        """Freeze and build one complete candidate without touching old runs."""

        self._validate_dataset_source(binding)
        arguments = build_arguments(
            binding.descriptor,
            installation=self.session.installation,
            signal_plane=self.session.signal_plane,
            values=dict(binding.draft.values),
            source_signal=binding.draft.source_signal,
            artifact_inputs=binding.draft.artifact_inputs,
            extras=self._logic_extras(),
            device_keys=binding.draft.device_keys,
        )
        node = binding.descriptor.instantiate(**arguments)
        host = make_host(
            binding.descriptor,
            node,
            signal_plane=self.session.signal_plane,
            instance_id=binding.node_id,
            request_owner_wake=self.board.wake.request_owner_wake,
        )
        claims = tuple(
            DeviceClaim(
                requirement.argument_name,
                binding.draft.device_keys.get(requirement.argument_name, ""),
                arguments[requirement.argument_name],
                requirement.access,
            )
            for requirement in binding.descriptor.device_requirements
        )
        return LogicCandidate(node, host, claims)

    def _discard_pending(self, binding: LogicBinding) -> None:
        candidate, binding.pending = binding.pending, None
        if candidate is None:
            return
        self._discard_candidate(binding, candidate)

    def _discard_candidate(
        self,
        binding: LogicBinding,
        candidate: LogicCandidate,
    ) -> None:
        if binding.pending is candidate:
            binding.pending = None
        if candidate.reservation is not None:
            candidate.reservation.abort()
            candidate.reservation = None
        try:
            candidate.host.shutdown()
        except Exception as error:
            self._report(f"{binding.node_id}: {error}", severity="error")

    def _activate_candidate(
        self,
        binding: LogicBinding,
        candidate: LogicCandidate,
    ) -> bool:
        """Replace a stopped generation and start the already-built candidate."""

        old_host = binding.host
        if old_host is not None:
            if old_host.running:
                binding.pending = candidate
                self._refresh_console_projection()
                return True
            try:
                old_host.shutdown()
            except Exception as error:
                self._discard_candidate(binding, candidate)
                binding.draft_error = str(error)
                self._report(f"{binding.node_id}: {error}", severity="error")
                self._refresh_console_projection()
                return False
        reservation = candidate.reservation
        if reservation is None:
            self._discard_candidate(binding, candidate)
            binding.draft_error = "logic candidate lost its device reservation"
            self._refresh_console_projection()
            return False
        try:
            lease = reservation.commit()
        except DeviceUseBusy:
            binding.pending = candidate
            self._refresh_console_projection()
            return True
        except Exception as error:
            self._discard_candidate(binding, candidate)
            binding.draft_error = str(error)
            self._report(f"{binding.node_id}: {error}", severity="error")
            self._refresh_console_projection()
            return False
        candidate.reservation = None
        binding.node = candidate.node
        binding.host = candidate.host
        binding.lease = lease
        binding.pending = None
        binding.artifact_results = ()
        binding.artifact_result_host = None
        binding.artifact_completion_order = 0
        try:
            candidate.host.start()
        except Exception as error:
            lease.release()
            binding.lease = None
            binding.draft_error = str(error)
            self._report(f"{binding.node_id}: {error}", severity="error")
            self._refresh_console_projection()
            return False
        binding.draft_error = ""
        self._refresh_console_projection()
        self._report(f"{binding.node_id} started", severity="task")
        return True

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
        extras["artifact_directory"] = self.session.day_folder()
        return extras

    def _artifact_results(
        self,
        binding: LogicBinding,
    ) -> tuple[Mapping[str, str], ...]:
        """Already-observed saved paths from one successful current host."""

        if binding.artifact_result_host is binding.host:
            return binding.artifact_results
        return ()

    def _capture_artifact_results(self, binding: LogicBinding) -> None:
        """Freeze artifact paths when polling first observes terminal success."""

        host = binding.host
        if (
            host is None
            or host.running
            or not host.terminal
            or host.observation.error is not None
            or not host.final_result_resolved
        ):
            return
        if binding.artifact_result_host is host:
            return
        result = host.final_result
        rows: list[Mapping[str, str]] = []
        for output in getattr(binding.descriptor, "artifact_outputs", ()):
            value = (
                result.get(output.name)
                if isinstance(result, Mapping)
                else getattr(result, output.name, None)
            )
            if value is None:
                continue
            path = str(Path(value).expanduser().resolve())
            rows.append(
                {
                    "name": str(output.name),
                    "contract_id": str(output.contract_id),
                    "path": path,
                }
            )
        binding.artifact_result_host = host
        binding.artifact_results = tuple(rows)
        if rows:
            self._artifact_completion_order += 1
            binding.artifact_completion_order = self._artifact_completion_order

    def _default_artifact_inputs(self, descriptor: object) -> dict[str, str]:
        """Freeze the latest observed matching artifact into a new row draft."""

        available: dict[str, tuple[int, str]] = {}
        for binding in self.logic.values():
            for row in self._artifact_results(binding):
                contract = str(row["contract_id"])
                candidate = (binding.artifact_completion_order, str(row["path"]))
                if candidate[0] >= available.get(contract, (-1, ""))[0]:
                    available[contract] = candidate
        return {
            spec.name: available.get(str(spec.contract_id), (-1, ""))[1]
            for spec in artifact_input_specs(descriptor)
        }

    def _free_logic_id(self, api_name: str) -> str:
        if api_name not in self.logic:
            return api_name
        index = 2
        while f"{api_name}{index}" in self.logic:
            index += 1
        return f"{api_name}{index}"

    def _refresh_console_projection(self) -> None:
        """Project logic rows and header chrome from one current state read.

        Runtime polling, command transitions, and structural edits all end
        here.  Rows and the running count therefore cannot observe different
        moments of the same host lifecycle.
        """

        for binding in tuple(self.logic.values()):
            self._show_logic(binding)
        state = "paused" if self._paused else "running"
        running = sum(
            1
            for item in self.logic.values()
            if item.host is not None and item.host.running
        )
        nodes = f", {running}/{len(self.logic)} node(s) running" if self.logic else ""
        summary = f"{len(self.panels)} panel(s), {state}{nodes}"
        if summary == self._shown_console_summary:
            return
        self._shown_console_summary = summary
        self.view.set_summary(summary)

    def close(self, *, node_stop_seconds: float = 10.0) -> None:
        # Nodes first: one still running publishes into a plane the panels are
        # being taken off, and a worker left alive keeps the process up with no
        # window to show for it.  Here, unlike Remove, waiting is right -- the
        # window is going away and there is nothing left to keep responsive --
        # but it is still bounded, so a wedged node cannot hold the process.
        deadline = time.monotonic() + float(node_stop_seconds)
        for binding in tuple(self.logic.values()):
            binding.removing = True
            self._discard_pending(binding)
            if binding.host is not None and binding.host.running:
                binding.host.cancel("the console is closing")
        while self.logic and time.monotonic() < deadline:
            self.poll_logic()
            if any(
                item.host is not None and item.host.running
                for item in self.logic.values()
            ):
                time.sleep(0.01)
        running = tuple(
            binding.node_id
            for binding in self.logic.values()
            if binding.host is not None and binding.host.running
        )
        if running:
            names = ", ".join(running)
            raise TimeoutError(f"logic nodes did not stop before close: {names}")
        for binding in tuple(self.logic.values()):
            if not self._retire_logic(binding):
                raise RuntimeError(
                    f"logic node {binding.node_id!r} could not release its host"
                )
        for panel_id in list(self.panels):
            self.remove_panel(panel_id)
        self.board.close()
